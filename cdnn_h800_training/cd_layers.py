#!/usr/bin/env python3
"""
=============================================================================
cd_layers.py — Communication Dynamics Neural Network Layers (PyTorch)
=============================================================================

PyTorch reimplementation of Pan (2026) CD-NN layers, optimized for
multi-GPU H800 training with DeepSeek-V3-style cost-reduction techniques.

  1. CDLinear           — Block-circulant linear layer (FFT-diagonalized)
  2. ShannonDropout     — alpha_CD = 0.0118 principled noise injection
  3. RMSNorm            — DeepSeek-V3 style normalization (fp32 reduction)
  4. CDAttention        — Latent-KV attention with CD-compressed projections
  5. CDMoELayer         — Mixture-of-Experts with CDLinear experts
  6. CDTransformerBlock — Full transformer block (attention + CD-MoE)
  7. fisher_reg_loss    — Closed-form Fisher regularizer (Parseval identity)

Reference:
  L. Pan (2026), "Communication Dynamics Neural Networks:
  FFT-Diagonalized Layers for Improved Hessian Conditioning
  at Reduced Parameter Count", arXiv (Paper III)

Authors: L. Pan (Ainnocence Inc.)
License: MIT

-----------------------------------------------------------------------------
NOTE ON FP8 / DeepSeek kernels
-----------------------------------------------------------------------------
The actual compute path runs in BF16 under autocast (correct and stable on
H800). True end-to-end FP8 on H800 should be obtained from DeepSeek's
production-tested kernels rather than ad-hoc `_scaled_mm` calls:
  - DeepGEMM  : FP8 dense + grouped/masked MoE GEMM (github.com/deepseek-ai/DeepGEMM)
  - FlashMLA  : Hopper MLA decode kernel        (github.com/deepseek-ai/FlashMLA)
  - DeepEP    : expert-parallel all-to-all       (github.com/deepseek-ai/DeepEP)
`use_fp8` is kept as a forward-looking flag and a small reference
`fp8_matmul` helper is provided, but it is NOT on the default hot path.
=============================================================================
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# CD Physical Constants (from Pan & Tanik, Paper I)
# ---------------------------------------------------------------------------
ALPHA_CD = 0.0118        # Shannon noise constant (Na D-doublet calibration)
LAMBDA_CEIL = 4.0        # Sadovskii lattice-instability ceiling


# =============================================================================
# FP8 reference helper (OPTIONAL — not on the default compute path)
# =============================================================================
def fp8_cast(tensor: torch.Tensor, dtype=None) -> Tuple[torch.Tensor, torch.Tensor]:
    """Cast a tensor to FP8 with a single global amax scale.

    Always returns ``(fp8_tensor, scale)`` so callers can unpack uniformly.
    Returns BF16 (scale=1) when FP8 is unavailable.
    """
    if dtype is None:
        dtype = getattr(torch, "float8_e4m3fn", None)
    if dtype is None:  # no FP8 support in this build
        return tensor.bfloat16(), torch.ones((), device=tensor.device, dtype=torch.float32)

    amax = tensor.abs().amax().clamp(min=1e-12)
    fp8_max = 448.0 if dtype == torch.float8_e4m3fn else 57344.0  # E4M3 / E5M2
    scale = (fp8_max / amax).to(torch.float32)
    return (tensor.float() * scale).to(dtype), scale


def fp8_matmul(a: torch.Tensor, b: torch.Tensor, use_fp8: bool = True) -> torch.Tensor:
    """Reference FP8 GEMM via ``torch._scaled_mm`` (2-D inputs only).

    Falls back to BF16 matmul. For production FP8 use DeepGEMM instead.
    """
    fp8 = getattr(torch, "float8_e4m3fn", None)
    if use_fp8 and fp8 is not None and a.is_cuda and a.dim() == 2 and b.dim() == 2:
        try:
            a_fp8, a_scale = fp8_cast(a, fp8)
            b_fp8, b_scale = fp8_cast(b, fp8)
            return torch._scaled_mm(
                a_fp8, b_fp8.t().contiguous().t(),
                scale_a=(1.0 / a_scale).reshape(1, 1),
                scale_b=(1.0 / b_scale).reshape(1, 1),
                out_dtype=torch.bfloat16,
            )
        except (RuntimeError, AttributeError):
            pass
    return torch.matmul(a.bfloat16(), b.bfloat16())


# =============================================================================
# 1. CDLinear: BLOCK-CIRCULANT LAYER (FFT-DIAGONALIZED)
# =============================================================================
class CDLinear(nn.Module):
    """Block-circulant linear layer with FFT-diagonalized Hessian.

    Maps R^{n_in} -> R^{n_out} via a block-circulant weight matrix with
    block size B. Only the first column of each circulant block is learned,
    giving a B x parameter reduction vs a dense layer.

    Forward (per output/input block pair, summed over input blocks):
        y_block = IFFT( FFT(c) * FFT(x_block) )          [circular convolution]

    Theory (Pan 2026):
      - Hessian eigenvalues of each block = |FFT(c)|^2  (Theorem 1)
      - Under pre-whitening, condition number k -> 1     (Theorem 2)
    """

    def __init__(self, n_in: int, n_out: int, block: int = 5,
                 use_fp8: bool = True, bias: bool = True, impl: str = "matmul"):
        super().__init__()
        self.block = block
        self.use_fp8 = use_fp8
        self.impl = impl                 # "matmul", "fft", or "dense" (baseline)

        # Pad dimensions to multiples of block
        self.n_in_raw = n_in
        self.n_out_raw = n_out
        self.n_in = ((n_in + block - 1) // block) * block
        self.n_out = ((n_out + block - 1) // block) * block
        self.K_in = self.n_in // block
        self.K_out = self.n_out // block

        std = math.sqrt(2.0 / self.n_in)
        if impl == "dense":
            # Full dense weight: the non-circulant baseline (no compression).
            # Same module interface so the rest of the model is untouched; the
            # Fisher/conditioning regularizers skip dense layers (c is None).
            self.weight = nn.Parameter(torch.randn(self.n_out, self.n_in) * std)
            self.c = None
        else:
            # First-column coefficients of each circulant block: (K_out, K_in, B)
            # This is the ONLY learned weight — B x fewer than dense W.
            self.weight = None
            self.c = nn.Parameter(torch.randn(self.K_out, self.K_in, block) * std)

        self.bias = nn.Parameter(torch.zeros(self.n_out)) if bias else None

        # Circulant gather index: C[i, j] = c[(i - j) mod B]. Used by the matmul
        # path to build the effective weight from c on the fly (no extra params).
        if impl != "dense":
            idx = (torch.arange(block).unsqueeze(1)
                   - torch.arange(block).unsqueeze(0)) % block
            self.register_buffer("_circ_idx", idx, persistent=False)

    def _dense_weight(self, dtype):
        """Build the effective dense weight (n_out, n_in) from the circulant
        generators c. Differentiable; transient (recomputed each forward, and
        under gradient checkpointing recomputed in backward), so it costs no
        stored activation that scales with batch×seq."""
        Wb = self.c[:, :, self._circ_idx]               # (K_out, K_in, B, B)
        W = Wb.permute(0, 2, 1, 3).reshape(self.n_out, self.n_in)
        return W.to(dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (..., n_in_raw) -> (..., n_out_raw)"""
        orig_shape = x.shape[:-1]
        x = x.reshape(-1, x.shape[-1])
        n = x.shape[0]
        if x.shape[-1] < self.n_in:
            x = F.pad(x, (0, self.n_in - x.shape[-1]))

        if self.impl == "dense":
            y = F.linear(x, self.weight, self.bias)
        elif self.impl == "matmul":
            # Tensor-core dense GEMM with a weight built from c. Mathematically
            # identical to the circular convolution but uses BF16 matmul and
            # keeps no token-scaled FFT intermediates -> high MFU, low memory.
            W = self._dense_weight(x.dtype)
            y = F.linear(x, W, self.bias)
        else:
            y = self._forward_fft(x)
        y = y[..., :self.n_out_raw]
        return y.reshape(*orig_shape, self.n_out_raw)

    def _forward_fft(self, x: torch.Tensor) -> torch.Tensor:
        """Reference FFT-domain circular convolution (fp32; slower on GPU at
        small block size, kept for verification / large-block use)."""
        n = x.shape[0]
        x_blk = x.reshape(n, self.K_in, self.block)
        x_fft = torch.fft.fft(x_blk.float(), dim=-1)
        c_fft = torch.fft.fft(self.c.float(), dim=-1)
        y_fft = torch.einsum('oik,nik->nok', c_fft, x_fft)
        y_blk = torch.fft.ifft(y_fft, dim=-1).real
        y = y_blk.reshape(n, self.n_out).to(x.dtype)
        if self.bias is not None:
            y = y + self.bias
        return y

    # --- diagnostics (no grad) -------------------------------------------
    @torch.no_grad()
    def hessian_spectrum(self) -> torch.Tensor:
        """FFT-diagonalized Hessian eigenvalues |FFT(c)|^2 (Theorem 1)."""
        if self.c.numel() == 0:          # FSDP flat-param placeholder; nothing to do
            return self.c.new_zeros(0)
        c_fft = torch.fft.fft(self.c.float(), dim=-1)
        return c_fft.abs().pow(2).flatten()

    @torch.no_grad()
    def condition_number(self) -> float:
        """Hessian condition number k = max(eig) / min(eig) (Theorem 2)."""
        spec = self.hessian_spectrum()
        spec = spec[spec > 1e-12]
        if spec.numel() == 0:
            return float('nan')
        return float(spec.max() / spec.min())

    # --- differentiable regularizer term ---------------------------------
    def fft_energy(self) -> torch.Tensor:
        """Differentiable sum of Hessian eigenvalues, sum_k |FFT(c)|^2.

        By Parseval/Plancherel, sum_k |FFT(c)|^2 = B * ||c||^2 exactly, so we
        return ``block * c.pow(2).sum()``. This is:
          * fully differentiable (the FFT-then-abs path is detached),
          * autocast / FSDP safe (no FFT, separable across shards),
          * numerically benign (well-conditioned weight-decay-like penalty).
        """
        if self.c is None:
            return torch.zeros((), device=self.weight.device)
        return self.block * self.c.float().pow(2).sum()

    def spectral_flatness_penalty(self, eps: float = 1e-8) -> torch.Tensor:
        """Conditioning penalty matching CDNN paper Eq. (9): the variance of the
        (half-)log Hessian spectrum, averaged over circulant blocks.

            L = mean_blocks  Var_k( 1/2 * log s_k ),   s_k = |FFT(c)_k|^2

        Var_k(1/2 log s_k) = Var_k( log|FFT(c)_k| ). This is >= 0, equals 0 iff
        the spectrum is flat (the kappa = 1 optimum of Theorem 2), is fully
        differentiable through to the circulant coefficients c, scale-invariant
        (a global scaling of c shifts every log by a constant, leaving the
        variance unchanged), and bounded for any finite-magnitude spectrum --
        unlike the trace-of-inverse form tr(I^-1) = sum 1/sigma^2, which diverges
        as sigma -> 0. Penalizes only the spectral spread that drives kappa from 1.
        """
        if self.c is None:                       # dense baseline: no spectrum
            return torch.zeros((), device=self.weight.device)
        cf = torch.fft.fft(self.c.float(), dim=-1)
        half_log_s = (cf.abs() + eps).log()          # 1/2 log s_k = log|FFT(c)_k|
        return half_log_s.var(dim=-1, unbiased=False).mean()

    def spectral_flatness_blocks(self, eps: float = 1e-8) -> torch.Tensor:
        """Per-circulant-block spectral-spread penalty, flattened to (K_out*K_in,).
        Used by ``fisher_reg_loss`` for worst-block (p-norm / max) aggregation so
        a few catastrophic blocks are not diluted by the well-conditioned majority
        (the mean-aggregation limitation seen at language-model scale)."""
        if self.c is None:
            return self.weight.new_zeros(0)
        cf = torch.fft.fft(self.c.float(), dim=-1)
        half_log_s = (cf.abs() + eps).log()
        return half_log_s.var(dim=-1, unbiased=False).reshape(-1)

    @property
    def compression_ratio(self) -> float:
        if self.c is None:
            return 1.0
        dense = self.n_in_raw * self.n_out_raw
        cd = self.c.numel() + (self.bias.numel() if self.bias is not None else 0)
        return dense / cd

    def extra_repr(self) -> str:
        nparam = self.weight.numel() if self.c is None else self.c.numel()
        return (f"n_in={self.n_in_raw}, n_out={self.n_out_raw}, "
                f"block={self.block}, impl={self.impl}, params={nparam}, "
                f"compression={self.compression_ratio:.1f}x")


# =============================================================================
# 2. ShannonDropout
# =============================================================================
class ShannonDropout(nn.Module):
    """Dropout at fixed rate alpha_CD = 0.0118 (Paper I)."""

    def __init__(self, p: float = ALPHA_CD):
        super().__init__()
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p <= 0:
            return x
        return F.dropout(x, p=self.p, training=True)


# =============================================================================
# 3. RMSNorm (DeepSeek-V3 style, fp32 reduction)
# =============================================================================
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.to(dtype)) * self.weight


# =============================================================================
# 4. CDAttention — latent-KV attention with CD-compressed projections
# =============================================================================
class CDAttention(nn.Module):
    """Attention with a low-rank KV latent (MLA-style) and CD-compressed
    projection matrices.

    Uses ``F.scaled_dot_product_attention`` so it dispatches to FlashAttention /
    memory-efficient kernels on H800 and applies the causal mask without
    materializing an (T x T) tensor.

    This is a *simplified* MLA: the KV is compressed to ``kv_lora_rank`` and
    decompressed per head; RoPE is applied to the full Q/K head dims. For the
    production DeepSeek MLA decode path (decoupled RoPE + paged KV cache) use
    FlashMLA at inference time.
    """

    def __init__(self, dim: int, n_heads: int = 16,
                 kv_lora_rank: int = 512,
                 block: int = 5,
                 max_seq_len: int = 4096,
                 use_fp8: bool = True, impl: str = "matmul"):
        super().__init__()
        assert dim % n_heads == 0, "dim must be divisible by n_heads"
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        assert self.head_dim % 2 == 0, "head_dim must be even for RoPE"
        self.kv_lora_rank = kv_lora_rank

        self.wq = CDLinear(dim, n_heads * self.head_dim, block=block, use_fp8=use_fp8, impl=impl)
        self.wkv_down = CDLinear(dim, kv_lora_rank, block=block, use_fp8=use_fp8, impl=impl)
        self.wk_up = CDLinear(kv_lora_rank, n_heads * self.head_dim, block=block, use_fp8=use_fp8, impl=impl)
        self.wv_up = CDLinear(kv_lora_rank, n_heads * self.head_dim, block=block, use_fp8=use_fp8, impl=impl)
        self.wo = CDLinear(n_heads * self.head_dim, dim, block=block, use_fp8=use_fp8, impl=impl)
        self.kv_norm = RMSNorm(kv_lora_rank)

        self.register_buffer(
            'freqs_cis', self._precompute_freqs(self.head_dim, max_seq_len),
            persistent=False
        )

    @staticmethod
    def _precompute_freqs(dim: int, max_len: int, theta: float = 10000.0):
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_len).float()
        freqs = torch.outer(t, freqs)
        return torch.polar(torch.ones_like(freqs), freqs)        # (max_len, dim/2)

    def _apply_rope(self, x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
        # x: (B, n_heads, T, head_dim)
        xc = torch.view_as_complex(
            x.float().reshape(*x.shape[:-1], -1, 2).contiguous()
        )                                                        # (B, nh, T, hd/2)
        freqs = freqs[None, None, :x.shape[-2], :]               # (1,1,T,hd/2)
        x_rot = torch.view_as_real(xc * freqs).flatten(-2)
        return x_rot.type_as(x)

    def forward(self, x: torch.Tensor,
                mask: Optional[torch.Tensor] = None,
                is_causal: bool = True) -> torch.Tensor:
        B, T, D = x.shape

        q = self.wq(x).reshape(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        kv_latent = self.kv_norm(self.wkv_down(x))               # (B, T, rank)
        k = self.wk_up(kv_latent).reshape(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.wv_up(kv_latent).reshape(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        freqs = self.freqs_cis[:T].to(x.device)
        q = self._apply_rope(q, freqs)
        k = self._apply_rope(k, freqs)

        # FlashAttention / mem-efficient kernel. If an explicit additive mask
        # is supplied, use it; otherwise rely on the fast is_causal path.
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=mask,
            is_causal=(mask is None and is_causal),
        )
        out = out.transpose(1, 2).reshape(B, T, D)
        return self.wo(out)


# =============================================================================
# 5. CDMoELayer — Mixture-of-Experts with CDLinear experts
# =============================================================================
class CDMoELayer(nn.Module):
    """MoE FFN with CDLinear experts and DeepSeek-V3 auxiliary-loss-free
    load balancing (a learnable per-expert bias adjusted from observed load).

    Optimizations vs a naive implementation:
      * one GEMM group per expert (E iterations) instead of n_active x E,
      * gather/scatter via index_select / index_add_ (no overlapping in-place
        autograd writes, no per-(k,e) boolean .any() host syncs),
      * gate weights computed from the *raw* logits; the balancing bias only
        affects which experts are selected (DeepSeek-V3 design).

    For multi-GPU expert parallelism + FP8 dispatch, route this through
    DeepEP + DeepGEMM grouped GEMM in production.
    """

    def __init__(self, dim: int, n_experts: int = 64,
                 n_active: int = 6, ffn_mult: float = 2.667,
                 block: int = 5, use_fp8: bool = True,
                 bias_update_speed: float = 1e-3, impl: str = "matmul"):
        super().__init__()
        self.dim = dim
        self.n_experts = n_experts
        self.n_active = min(n_active, n_experts)
        # Speed used by the EXTERNAL updater (see update_router_biases); the
        # forward pass itself never mutates router state so it stays
        # deterministic under activation checkpointing.
        self.bias_update_speed = bias_update_speed

        hidden = int(dim * ffn_mult)
        self.hidden_dim = ((hidden + block - 1) // block) * block

        self.gate = nn.Linear(dim, n_experts, bias=False)

        self.gate_proj = nn.ModuleList(
            [CDLinear(dim, self.hidden_dim, block=block, use_fp8=use_fp8, impl=impl) for _ in range(n_experts)])
        self.up_proj = nn.ModuleList(
            [CDLinear(dim, self.hidden_dim, block=block, use_fp8=use_fp8, impl=impl) for _ in range(n_experts)])
        self.down_proj = nn.ModuleList(
            [CDLinear(self.hidden_dim, dim, block=block, use_fp8=use_fp8, impl=impl) for _ in range(n_experts)])

        # Shared expert (always active, DeepSeek-V3 style)
        self.shared_gate = CDLinear(dim, self.hidden_dim, block=block, use_fp8=use_fp8, impl=impl)
        self.shared_up = CDLinear(dim, self.hidden_dim, block=block, use_fp8=use_fp8, impl=impl)
        self.shared_down = CDLinear(self.hidden_dim, dim, block=block, use_fp8=use_fp8, impl=impl)

        # Auxiliary-loss-free load-balancing bias (selection only; not learned
        # by backprop). Updated BETWEEN steps by update_router_biases(), never
        # inside forward, so forward/recompute stay identical.
        self.register_buffer('expert_bias', torch.zeros(n_experts))
        # Most recent per-expert load (idempotent under checkpoint recompute).
        self.register_buffer('last_load', torch.zeros(n_experts))

    def _expert_ffn(self, idx: int, tokens: torch.Tensor) -> torch.Tensor:
        return self.down_proj[idx](F.silu(self.gate_proj[idx](tokens)) * self.up_proj[idx](tokens))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        x_flat = x.reshape(-1, self.dim)                         # (N, dim)
        N = x_flat.shape[0]

        logits = self.gate(x_flat.float())                       # (N, E)
        routing = logits + self.expert_bias                      # bias = selection only
        topk_val, topk_idx = routing.topk(self.n_active, dim=-1)  # (N, k)

        # Combine weights come from the RAW logits (bias excluded).
        sel_logits = torch.gather(logits, 1, topk_idx)           # (N, k)
        weights = sel_logits.softmax(dim=-1).to(x.dtype)         # (N, k)

        # Shared expert (always on)
        out = self.shared_down(
            F.silu(self.shared_gate(x_flat)) * self.shared_up(x_flat)
        )

        # Grouped dispatch. Sort the (token, slot) assignments by expert ONCE so
        # each expert consumes a contiguous slice -- this replaces E full boolean
        # scans (each O(N*k)) and E tensor allocations with a single argsort and
        # cheap slicing, and a single host sync for the per-expert counts. The
        # math is identical to the per-expert loop (verified to <1e-5).
        flat_expert = topk_idx.reshape(-1)                       # (N*k,)
        flat_weight = weights.reshape(-1, 1)                     # (N*k, 1)
        token_ids = torch.arange(N, device=x.device).repeat_interleave(self.n_active)

        order = torch.argsort(flat_expert)
        fe, ftok, fw = flat_expert[order], token_ids[order], flat_weight[order]
        counts = torch.bincount(fe, minlength=self.n_experts).tolist()  # one sync

        start = 0
        for e in range(self.n_experts):
            cnt = counts[e]
            if cnt == 0:
                continue
            tok = ftok[start:start + cnt]
            ye = self._expert_ffn(e, x_flat[tok]) * fw[start:start + cnt].to(x.dtype)
            out = out.index_add(0, tok, ye.to(out.dtype))
            start += cnt

        # Stash load for the external balancer. copy_ writes the SAME value on
        # checkpoint recompute (routing is deterministic here), so it is safe.
        if self.training:
            with torch.no_grad():
                load = torch.bincount(flat_expert, minlength=self.n_experts).float()
                self.last_load.copy_(load)

        return out.reshape(orig_shape)


# =============================================================================
# 6. CDTransformerBlock
# =============================================================================
class CDTransformerBlock(nn.Module):
    """Pre-norm transformer block: CD-MLA attention + CD-MoE FFN."""

    def __init__(self, dim: int, n_heads: int = 16,
                 kv_lora_rank: int = 512,
                 n_experts: int = 64, n_active: int = 6,
                 block: int = 5, max_seq_len: int = 4096,
                 use_fp8: bool = True, impl: str = "matmul"):
        super().__init__()
        self.attn_norm = RMSNorm(dim)
        self.attn = CDAttention(
            dim=dim, n_heads=n_heads, kv_lora_rank=kv_lora_rank,
            block=block, max_seq_len=max_seq_len, use_fp8=use_fp8, impl=impl,
        )
        self.ffn_norm = RMSNorm(dim)
        self.ffn = CDMoELayer(
            dim=dim, n_experts=n_experts, n_active=n_active,
            block=block, use_fp8=use_fp8, impl=impl,
        )
        self.dropout = ShannonDropout(ALPHA_CD)

    def forward(self, x: torch.Tensor,
                mask: Optional[torch.Tensor] = None,
                is_causal: bool = True) -> torch.Tensor:
        h = x + self.dropout(self.attn(self.attn_norm(x), mask=mask, is_causal=is_causal))
        out = h + self.dropout(self.ffn(self.ffn_norm(h)))
        return out


# =============================================================================
# Auxiliary-loss-free load balancing — called once per OPTIMIZER step,
# OUTSIDE the (possibly checkpointed) forward pass.
# =============================================================================
@torch.no_grad()
def update_router_biases(model: nn.Module) -> None:
    """Nudge each CDMoELayer's expert_bias against overloaded experts
    (DeepSeek-V3 auxiliary-loss-free balancing). Safe to call after
    optimizer.step(); does not touch autograd or the checkpointed graph."""
    for m in model.modules():
        if isinstance(m, CDMoELayer) and m.bias_update_speed > 0:
            load = m.last_load
            if load.sum() == 0:
                continue
            err = load - load.mean()
            m.expert_bias.add_(-m.bias_update_speed * err.sign())


# =============================================================================
# 7. Fisher Information Regularizer (closed-form, differentiable, FSDP-safe)
# =============================================================================
def fisher_reg_loss(model: nn.Module, lambda_F: float = 1e-4,
                    mode: str = "energy", agg: str = "mean",
                    p: float = 4.0) -> torch.Tensor:
    """Regularizer over the CD layers.

    mode="energy" (default): L = lambda * sum_layers sum_k |FFT(c)|^2, via the
        Parseval identity sum_k |FFT(c)|^2 = B*||c||^2. Controls the SUM of
        Hessian eigenvalues (scale) — effectively L2; keep lambda tiny (~1e-8).

    mode="conditioning": flattens each circulant block's spectrum (paper Eq. 9)
        and drives kappa -> 1 (Theorem 2). Block penalties are combined by ``agg``:
          - "mean"  : average over blocks (gentle; can dilute the worst blocks).
          - "pnorm" : (mean(L_b^p))^(1/p) — smooth emphasis on the worst blocks
                      (p ~ 4-8); use when a few catastrophic blocks keep
                      mean/worst kappa high while best-block kappa already falls.
          - "max"   : the single worst block (most aggressive on the tail).
        Natural lambda ~1e-2..1e-1.

    Dense (baseline) layers contribute nothing (their c is None).
    """
    mods = [m for m in model.modules()
            if isinstance(m, CDLinear) and m.c is not None]
    if not mods:
        return torch.zeros((), device=next(model.parameters()).device)
    if mode == "conditioning":
        if agg == "mean":
            return lambda_F * torch.stack([m.spectral_flatness_penalty()
                                           for m in mods]).mean()
        blocks = torch.cat([m.spectral_flatness_blocks() for m in mods])
        if agg == "max":
            return lambda_F * blocks.max()
        return lambda_F * blocks.pow(p).mean().pow(1.0 / p)   # generalized p-mean
    return lambda_F * torch.stack([m.fft_energy() for m in mods]).sum()


# =============================================================================
# Self-test
# =============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("CD-NN PyTorch Layers: Smoke Test")
    print("=" * 60)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    print("\n[1] CDLinear (128 -> 256, block=5)...")
    layer = CDLinear(128, 256, block=5).to(device)
    x = torch.randn(4, 128, device=device)
    y = layer(x)
    print(f"    {tuple(x.shape)} -> {tuple(y.shape)} | "
          f"params={layer.c.numel()} | compression={layer.compression_ratio:.1f}x | "
          f"kappa={layer.condition_number():.2f}")

    print("\n[2] CDAttention (dim=512, heads=8, kv_rank=128)...")
    attn = CDAttention(512, n_heads=8, kv_lora_rank=128, block=5).to(device)
    x = torch.randn(2, 32, 512, device=device)
    print(f"    {tuple(x.shape)} -> {tuple(attn(x).shape)}")

    print("\n[3] CDMoELayer (dim=512, 8 experts, top-2)...")
    moe = CDMoELayer(512, n_experts=8, n_active=2, block=5).to(device)
    x = torch.randn(2, 32, 512, device=device)
    print(f"    {tuple(x.shape)} -> {tuple(moe(x).shape)}")

    print("\n[4] CDTransformerBlock...")
    blk = CDTransformerBlock(dim=512, n_heads=8, kv_lora_rank=128,
                             n_experts=8, n_active=2, block=5).to(device)
    x = torch.randn(2, 32, 512, device=device)
    y = blk(x)
    y.sum().backward()
    print(f"    {tuple(x.shape)} -> {tuple(y.shape)} | backward OK | "
          f"params={sum(p.numel() for p in blk.parameters()):,}")

    print("\nAll smoke tests passed.")
