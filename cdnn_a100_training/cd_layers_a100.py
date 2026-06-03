#!/usr/bin/env python3
"""
=============================================================================
cd_layers_a100.py — Communication Dynamics Neural Network Layers
                     (PyTorch + CUDA, A100 Ampere Optimized)
=============================================================================

PyTorch reimplementation of Pan (2026) CD-NN layers, optimized for
multi-GPU A100 training with DeepSeek-V3–style cost-reduction techniques,
adapted from H800/Hopper to A100/Ampere architecture.

Key A100 adaptations (vs H800):
  - TF32 Tensor Cores (19.5 TFLOPS) instead of FP8
  - BF16 mixed precision via torch.cuda.amp (no FP8 path)
  - NVLink 3.0 (600 GB/s) topology-aware communication
  - 80GB HBM2e bandwidth-tuned batch sizes
  - torch.backends optimizations for Ampere SM80

Layers:
  1. CDLinear          — Block-circulant linear layer (FFT-diagonalized)
  2. CDMoELayer        — Mixture-of-Experts with CDLinear experts
  3. ShannonDropout    — alpha_CD = 0.0118 principled noise injection
  4. CDAttention       — Multi-head Latent Attention with CD-compressed KV
  5. CDTransformerBlock — Full transformer block combining MLA + CD-MoE

Reference:
  L. Pan (2026), "Communication Dynamics Neural Networks:
  FFT-Diagonalized Layers for Improved Hessian Conditioning
  at Reduced Parameter Count", arXiv (Paper III)

Authors: L. Pan (Ainnocence Inc.)
License: MIT
=============================================================================
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

# ---------------------------------------------------------------------------
# CD Physical Constants (from Pan & Tanik, Paper I)
# ---------------------------------------------------------------------------
ALPHA_CD = 0.0118        # Shannon noise constant (Na D-doublet calibration)
LAMBDA_CEIL = 4.0        # Sadovskii lattice-instability ceiling


# =============================================================================
# A100 Precision Utilities — TF32 + BF16 (Ampere architecture)
# =============================================================================
def setup_a100_precision():
    """Configure A100 Tensor Core precision settings.

    A100 (SM80 / Ampere) supports:
      - TF32 (TensorFloat-32): 19-bit effective, 19.5 TFLOPS (vs FP32 9.7)
      - BF16: 312 TFLOPS via Tensor Cores
      - FP16: 312 TFLOPS via Tensor Cores
      - INT8: 624 TOPS (inference only)

    Unlike H800/Hopper, A100 does NOT support native FP8 compute.
    We use TF32 for single-precision ops and BF16 for mixed-precision.
    """
    # Enable TF32 for matmuls — 2× speedup over FP32 on A100
    torch.backends.cuda.matmul.allow_tf32 = True
    # Enable TF32 for cuDNN convolutions
    torch.backends.cudnn.allow_tf32 = True
    # Enable cuDNN benchmark mode for optimal kernel selection
    torch.backends.cudnn.benchmark = True


def bf16_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """BF16-accelerated matrix multiply for A100 Tensor Cores.

    A100 achieves 312 TFLOPS in BF16, compared to 19.5 in TF32.
    This is the primary fast-path for CDLinear on Ampere.
    """
    return torch.matmul(a.bfloat16(), b.bfloat16()).float()


# =============================================================================
# 1. CDLinear: BLOCK-CIRCULANT LAYER (FFT-DIAGONALIZED) — A100 version
# =============================================================================
class CDLinear(nn.Module):
    """Block-circulant linear layer with FFT-diagonalized Hessian.

    Maps R^{n_in} -> R^{n_out} via a block-circulant weight matrix with
    block size B = 2L+1 (the polygon multiplicity from CD theory).

    Key theoretical properties (Theorems 1-2 from Pan 2026):
      - Hessian eigenvalues = |FFT(input_block)|^2, readable from a single FFT
      - Under pre-whitening: κ(Hessian) = 1 exactly
      - Empirical κ ≤ 1 + O(√(B/N)) on N samples
      - Parameter count: n_in × n_out / B  (B× compression vs dense)

    A100 notes:
      - FFT operations run on TF32 Tensor Cores when in float32
      - No FP8 path (Ampere limitation); BF16 used in AMP context
      - cuFFT auto-selects optimal algorithm for A100's SM80 architecture

    Parameters
    ----------
    n_in, n_out : int
        Input / output dimensions. Internally padded to multiples of `block`.
    block : int
        CD polygon multiplicity B = 2L+1. Choose from {1, 3, 5, 7, ...}.
    """

    def __init__(self, n_in: int, n_out: int, block: int = 5,
                 bias: bool = True):
        super().__init__()
        self.block = block

        # Pad dimensions to multiples of block
        self.n_in_raw = n_in
        self.n_out_raw = n_out
        self.n_in = ((n_in + block - 1) // block) * block
        self.n_out = ((n_out + block - 1) // block) * block
        self.K_in = self.n_in // block
        self.K_out = self.n_out // block

        # First-row coefficients of each circulant block: (K_out, K_in, B)
        # This is the ONLY learned parameter — B× fewer than dense W
        std = math.sqrt(2.0 / self.n_in)
        self.c = nn.Parameter(torch.randn(self.K_out, self.K_in, block) * std)

        if bias:
            self.bias = nn.Parameter(torch.zeros(self.n_out))
        else:
            self.bias = None

        # Pre-compute FFT twiddle factors (fixed, not learned)
        self.register_buffer(
            'dft_matrix',
            torch.fft.fft(torch.eye(block), dim=-1),
            persistent=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, *, n_in_raw) -> (batch, *, n_out_raw)

        On A100, the FFT runs via cuFFT which auto-selects the optimal
        algorithm for the SM80 architecture. TF32 is used for float32
        operations when torch.backends.cuda.matmul.allow_tf32 is True.
        """
        orig_shape = x.shape[:-1]
        x = x.reshape(-1, x.shape[-1])
        batch = x.shape[0]

        # Pad input if needed
        if x.shape[-1] < self.n_in:
            x = F.pad(x, (0, self.n_in - x.shape[-1]))

        # Reshape to (batch, K_in, B) for block-circulant products
        x_blk = x.reshape(batch, self.K_in, self.block)

        # === FFT-domain multiplication (core CD operation) ===
        # Instead of spatial-domain circulant matmul O(K_out * K_in * B^2),
        # we do pointwise multiply in FFT domain: O(K_out * K_in * B log B)
        x_fft = torch.fft.fft(x_blk, dim=-1)           # (batch, K_in, B)
        c_fft = torch.fft.fft(self.c, dim=-1)           # (K_out, K_in, B)

        # Batched pointwise multiply and sum over K_in
        # y_fft[n, o, k] = sum_{i} c_fft[o, i, k] * x_fft[n, i, k]
        #   n = batch, o = K_out, i = K_in, k = block (FFT bin)
        y_fft = torch.einsum('oik,nik->nok', c_fft, x_fft)  # (batch, K_out, B)
        y_blk = torch.fft.ifft(y_fft, dim=-1).real       # (batch, K_out, B)

        y = y_blk.reshape(batch, self.n_out)

        if self.bias is not None:
            y = y + self.bias

        # Slice off padding
        y = y[..., :self.n_out_raw]
        y = y.reshape(*orig_shape, self.n_out_raw)
        return y

    def hessian_spectrum(self) -> torch.Tensor:
        """Return FFT-diagonalized Hessian eigenvalues (Theorem 1).

        These are |FFT(c)|^2 for each circulant block — readable
        without any matrix decomposition.
        """
        with torch.no_grad():
            c_fft = torch.fft.fft(self.c, dim=-1)
            return torch.abs(c_fft).pow(2).flatten()

    def condition_number(self) -> float:
        """Compute Hessian condition number κ = max(eig) / min(eig).

        Theorem 2 (Pan 2026): For pre-whitened inputs, κ = 1 exactly.
        Empirically, κ ≤ 1 + O(√(B/N)).
        """
        spec = self.hessian_spectrum()
        spec = spec[spec > 1e-12]
        if spec.numel() == 0:
            return float('nan')
        return float(spec.max() / spec.min())

    @property
    def compression_ratio(self) -> float:
        """Parameter compression vs equivalent dense layer."""
        dense_params = self.n_in_raw * self.n_out_raw
        cd_params = self.c.numel() + (self.bias.numel() if self.bias is not None else 0)
        return dense_params / cd_params

    def extra_repr(self) -> str:
        return (f"n_in={self.n_in_raw}, n_out={self.n_out_raw}, "
                f"block={self.block}, params={self.c.numel()}, "
                f"compression={self.compression_ratio:.1f}×")


# =============================================================================
# 2. ShannonDropout — CD-derived principled noise injection
# =============================================================================
class ShannonDropout(nn.Module):
    """Dropout at rate α_CD = 0.0118, derived from Shannon channel theory.

    Unlike standard dropout (tuned empirically in [0.1, 0.5]), CD prescribes
    a transferable fixed rate calibrated from the Na D-doublet spectral
    splitting in Paper I. This acts as a mild stochastic regularizer.
    """

    def __init__(self, p: float = ALPHA_CD):
        super().__init__()
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p <= 0:
            return x
        return F.dropout(x, p=self.p, training=self.training)


# =============================================================================
# 3. RMSNorm — DeepSeek-V3 style normalization
# =============================================================================
class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (used in DeepSeek-V3)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm * self.weight


# =============================================================================
# 4. CDAttention — Multi-head Latent Attention with CD-compressed KV
# =============================================================================
class CDAttention(nn.Module):
    """Multi-head Latent Attention (MLA) with CD-compressed projections.

    Combines DeepSeek-V3's MLA (low-rank KV compression) with CDNN's
    block-circulant projections for further parameter reduction.

    MLA compresses KV to a low-rank latent space, reducing KV cache from
    O(n_heads × d_head) to O(d_compress). The CD block-circulant structure
    adds an additional B× compression on all projection matrices.

    A100 notes:
      - SDPA (Scaled Dot-Product Attention) used when available
        (PyTorch 2.0+, flash attention kernel selection)
      - BF16 SDPA achieves ~312 TFLOPS on A100 Tensor Cores

    Parameters
    ----------
    dim : int
        Model dimension.
    n_heads : int
        Number of attention heads.
    kv_lora_rank : int
        Latent dimension for KV compression (MLA).
    qk_rope_dim : int
        Dimension for rotary positional encoding.
    block : int
        CD polygon multiplicity for circulant projections.
    """

    def __init__(self, dim: int, n_heads: int = 16,
                 kv_lora_rank: int = 512,
                 qk_rope_dim: int = 64,
                 block: int = 5,
                 max_seq_len: int = 4096):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.kv_lora_rank = kv_lora_rank
        self.qk_rope_dim = qk_rope_dim
        self.scale = self.head_dim ** -0.5

        # Query projection (CD-compressed)
        self.wq = CDLinear(dim, n_heads * self.head_dim, block=block)

        # KV down-projection to latent space (MLA core idea)
        self.wkv_down = CDLinear(dim, kv_lora_rank + qk_rope_dim, block=block)

        # KV up-projection from latent to per-head KV
        self.wk_up = CDLinear(kv_lora_rank, n_heads * self.head_dim, block=block)
        self.wv_up = CDLinear(kv_lora_rank, n_heads * self.head_dim, block=block)

        # Output projection
        self.wo = CDLinear(n_heads * self.head_dim, dim, block=block)

        # RMSNorm for KV latent (DeepSeek-V3 uses this)
        self.kv_norm = RMSNorm(kv_lora_rank)

        # RoPE frequencies
        self.register_buffer(
            'freqs_cis',
            self._precompute_freqs(self.head_dim, max_seq_len),
            persistent=False
        )

    @staticmethod
    def _precompute_freqs(dim: int, max_len: int, theta: float = 10000.0):
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_len)
        freqs = torch.outer(t, freqs)
        return torch.polar(torch.ones_like(freqs), freqs)

    def _apply_rope(self, x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
        """Apply rotary positional encoding.

        x: (B, n_heads, T, head_dim)
        freqs: (T, head_dim/2) complex
        """
        # reshape last dim into (head_dim/2, 2) and view as complex.
        # .contiguous() is required: view_as_complex needs a stride
        # divisible by 2 on all but the last dimension.
        x_complex = torch.view_as_complex(
            x.float().reshape(*x.shape[:-1], -1, 2).contiguous()
        )
        # freqs: (T, head_dim/2) -> broadcast over batch and heads
        T = x.shape[-2]
        f = freqs[:T, :x_complex.shape[-1]]            # (T, head_dim/2)
        f = f.reshape(1, 1, T, -1)                      # (1, 1, T, head_dim/2)
        x_rot = torch.view_as_real(x_complex * f).flatten(-2)
        return x_rot.type_as(x)

    def forward(self, x: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x: (batch, seq_len, dim)
        mask: (batch, 1, seq_len, seq_len) optional causal mask
        """
        B, T, D = x.shape

        # Query
        q = self.wq(x).reshape(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # MLA: compress KV to latent space, then decompress
        kv_compressed = self.wkv_down(x)

        # Split into KV latent + RoPE component
        kv_latent = kv_compressed[..., :self.kv_lora_rank]
        kv_rope = kv_compressed[..., self.kv_lora_rank:]

        # Normalize latent (DeepSeek-V3 practice)
        kv_latent = self.kv_norm(kv_latent)

        # Decompress to per-head K and V
        k = self.wk_up(kv_latent).reshape(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.wv_up(kv_latent).reshape(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE to Q and K
        freqs = self.freqs_cis[:T].to(x.device)
        q = self._apply_rope(q, freqs)
        k = self._apply_rope(k, freqs)

        # Scaled dot-product attention.
        # We use is_causal=True so SDPA builds the causal mask internally
        # (faster, and avoids dtype/shape pitfalls with an explicit boolean
        # mask under BF16 autocast + FSDP). Shannon dropout is applied at the
        # transformer-block level, so we do NOT add dropout on attention
        # weights here.
        if hasattr(F, 'scaled_dot_product_attention'):
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=True,
            )
        else:
            # Manual causal attention fallback (older torch without SDPA)
            attn = (q @ k.transpose(-2, -1)) * self.scale
            causal = torch.tril(torch.ones(T, T, device=q.device, dtype=torch.bool))
            attn = attn.masked_fill(~causal, float('-inf'))
            attn = F.softmax(attn, dim=-1)
            out = attn @ v

        out = out.transpose(1, 2).reshape(B, T, D)
        return self.wo(out)


# =============================================================================
# 5. CDMoELayer — Mixture-of-Experts with CDLinear experts
# =============================================================================
class CDMoELayer(nn.Module):
    """Mixture-of-Experts FFN using CDLinear experts.

    Combines DeepSeek-V3's MoE design with CDNN's block-circulant
    parameter efficiency.

    Cost reduction stack (A100):
      1. MoE: Only top-K experts active per token → K/N_experts compute
      2. CDLinear: Each expert uses B× fewer params than dense
      3. BF16/TF32: 2× throughput on A100 Tensor Cores (vs FP32)
      Combined: Theoretical ~(K/N) × (1/B) × 2 cost reduction

    Parameters
    ----------
    dim : int
        Model dimension.
    n_experts : int
        Total number of experts.
    n_active : int
        Number of experts activated per token (top-K routing).
    ffn_mult : float
        FFN hidden dimension multiplier.
    block : int
        CD polygon multiplicity for expert weights.
    """

    def __init__(self, dim: int, n_experts: int = 64,
                 n_active: int = 6, ffn_mult: float = 2.667,
                 block: int = 5):
        super().__init__()
        self.dim = dim
        self.n_experts = n_experts
        self.n_active = n_active
        self.hidden_dim = int(dim * ffn_mult)
        # Round hidden_dim to multiple of block
        self.hidden_dim = ((self.hidden_dim + block - 1) // block) * block

        # Router: dense projection to expert scores
        self.gate = nn.Linear(dim, n_experts, bias=False)

        # Expert networks: gate + up + down with CDLinear
        self.gate_proj = nn.ModuleList([
            CDLinear(dim, self.hidden_dim, block=block)
            for _ in range(n_experts)
        ])
        self.up_proj = nn.ModuleList([
            CDLinear(dim, self.hidden_dim, block=block)
            for _ in range(n_experts)
        ])
        self.down_proj = nn.ModuleList([
            CDLinear(self.hidden_dim, dim, block=block)
            for _ in range(n_experts)
        ])

        # Shared expert (always active, DeepSeek-V3 style)
        self.shared_gate = CDLinear(dim, self.hidden_dim, block=block)
        self.shared_up = CDLinear(dim, self.hidden_dim, block=block)
        self.shared_down = CDLinear(self.hidden_dim, dim, block=block)

        # Auxiliary-loss-free load balancing (DeepSeek-V3 innovation)
        self.expert_bias = nn.Parameter(torch.zeros(n_experts))

    def _route(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Top-K routing with auxiliary-loss-free balancing."""
        logits = self.gate(x)
        routing_logits = logits + self.expert_bias
        topk_vals, topk_ids = torch.topk(routing_logits, self.n_active, dim=-1)
        topk_weights = F.softmax(topk_vals, dim=-1)
        return topk_weights, topk_ids, logits

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len, dim) -> (batch, seq_len, dim)"""
        orig_shape = x.shape
        x_flat = x.reshape(-1, self.dim)
        N = x_flat.shape[0]

        # Route tokens to experts
        topk_weights, topk_ids, _ = self._route(x_flat)

        # Shared expert contribution (always active)
        shared_out = self.shared_down(
            F.silu(self.shared_gate(x_flat)) * self.shared_up(x_flat)
        )

        # Expert computation
        expert_out = torch.zeros_like(x_flat)
        for k in range(self.n_active):
            expert_idx = topk_ids[:, k]
            weight = topk_weights[:, k]

            for e in range(self.n_experts):
                token_mask = (expert_idx == e)
                if not token_mask.any():
                    continue
                tokens = x_flat[token_mask]
                # SwiGLU activation (DeepSeek-V3 style)
                gate_out = F.silu(self.gate_proj[e](tokens))
                up_out = self.up_proj[e](tokens)
                e_out = self.down_proj[e](gate_out * up_out)
                expert_out[token_mask] += weight[token_mask].unsqueeze(-1) * e_out

        output = shared_out + expert_out
        return output.reshape(orig_shape)


# =============================================================================
# 6. CDTransformerBlock — Full transformer block
# =============================================================================
class CDTransformerBlock(nn.Module):
    """Transformer block combining CD-MLA attention + CD-MoE FFN.

    Building block for the full CD-Transformer, combining:
      - CDLinear projections in attention (B× compression)
      - MLA for KV cache compression
      - MoE with CDLinear experts
      - Shannon dropout for principled regularization
      - RMSNorm (DeepSeek-V3 style)
    """

    def __init__(self, dim: int, n_heads: int = 16,
                 kv_lora_rank: int = 512,
                 n_experts: int = 64, n_active: int = 6,
                 block: int = 5, max_seq_len: int = 4096):
        super().__init__()
        self.attn_norm = RMSNorm(dim)
        self.attn = CDAttention(
            dim=dim, n_heads=n_heads,
            kv_lora_rank=kv_lora_rank,
            block=block, max_seq_len=max_seq_len
        )
        self.ffn_norm = RMSNorm(dim)
        self.ffn = CDMoELayer(
            dim=dim, n_experts=n_experts,
            n_active=n_active, block=block,
        )
        self.dropout = ShannonDropout(ALPHA_CD)

    def forward(self, x: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Pre-norm attention with residual
        h = x + self.dropout(self.attn(self.attn_norm(x), mask=mask))
        # Pre-norm MoE FFN with residual
        out = h + self.dropout(self.ffn(self.ffn_norm(h)))
        return out


# =============================================================================
# 7. Fisher Information Regularizer (closed-form for circulant)
# =============================================================================
def fisher_reg_loss(model: nn.Module, lambda_F: float = 1e-4) -> torch.Tensor:
    """Compute Fisher-information regularization for all CDLinear layers.

    For a circulant block C with first row c, the FFT-diagonal singular
    values are σ_k = |FFT(c)[k]|. A perfectly conditioned layer (Theorem 2,
    Pan 2026) has a *flat* spectrum: all σ_k equal, i.e. κ = 1.

    We penalize the variance of the log-spectrum, which is:
      - zero exactly when the spectrum is flat (κ = 1, the optimum),
      - differentiable w.r.t. the circulant coefficients c (gradient flows),
      - numerically bounded (unlike Σ 1/σ², which explodes for tiny σ_k).

    This is the well-conditioned-landscape objective from Theorem 2,
    expressed as a stable, trainable penalty.
    """
    reg = torch.tensor(0.0, device=next(model.parameters()).device)
    n = 0
    for module in model.modules():
        if isinstance(module, CDLinear):
            # Recompute the spectrum WITH gradient tracking so the penalty
            # actually shapes the circulant coefficients during training.
            c_fft = torch.fft.fft(module.c, dim=-1)
            sigma2 = (c_fft.real.pow(2) + c_fft.imag.pow(2)).clamp(min=1e-8)
            log_spec = 0.5 * torch.log(sigma2)        # log σ_k
            # Variance of log-spectrum: 0 iff perfectly flat (κ = 1)
            reg = reg + log_spec.var(unbiased=False)
            n += 1
    if n > 0:
        reg = reg / n
    return lambda_F * reg


# =============================================================================
# Self-test
# =============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("CD-NN PyTorch Layers: A100 Smoke Test")
    print("=" * 60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    if device == 'cuda':
        gpu_name = torch.cuda.get_device_name(0)
        print(f"GPU: {gpu_name}")
        setup_a100_precision()
        print(f"TF32 enabled: {torch.backends.cuda.matmul.allow_tf32}")
        print(f"cuDNN benchmark: {torch.backends.cudnn.benchmark}")

    # Test CDLinear
    print("\n[1] CDLinear (128 -> 256, block=5)...")
    layer = CDLinear(128, 256, block=5).to(device)
    x = torch.randn(4, 128, device=device)
    y = layer(x)
    print(f"    Input: {x.shape} -> Output: {y.shape}")
    print(f"    Params: {layer.c.numel()}, Compression: {layer.compression_ratio:.1f}×")
    print(f"    Hessian κ: {layer.condition_number():.2f}")

    # Test CDAttention
    print("\n[2] CDAttention (dim=512, heads=8, kv_rank=128)...")
    attn = CDAttention(512, n_heads=8, kv_lora_rank=128, block=5).to(device)
    x = torch.randn(2, 32, 512, device=device)
    y = attn(x)
    print(f"    Input: {x.shape} -> Output: {y.shape}")

    # Test CDMoELayer
    print("\n[3] CDMoELayer (dim=512, 8 experts, top-2)...")
    moe = CDMoELayer(512, n_experts=8, n_active=2, block=5).to(device)
    x = torch.randn(2, 32, 512, device=device)
    y = moe(x)
    print(f"    Input: {x.shape} -> Output: {y.shape}")

    # Test CDTransformerBlock
    print("\n[4] CDTransformerBlock...")
    block = CDTransformerBlock(
        dim=512, n_heads=8, kv_lora_rank=128,
        n_experts=8, n_active=2, block=5
    ).to(device)
    x = torch.randn(2, 32, 512, device=device)
    y = block(x)
    print(f"    Input: {x.shape} -> Output: {y.shape}")

    total_params = sum(p.numel() for p in block.parameters())
    print(f"    Total block params: {total_params:,}")

    print("\nAll smoke tests passed.")
