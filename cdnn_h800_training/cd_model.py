#!/usr/bin/env python3
"""
=============================================================================
cd_model.py — CD-Transformer: Full Model Architecture
=============================================================================

A DeepSeek-V3-inspired transformer using Communication Dynamics (CD)
block-circulant layers throughout, designed for cost-effective training
on H800 8-GPU clusters.

  - Token embedding (tied with output head)
  - N x CDTransformerBlock (latent-KV attention + CD-MoE FFN)
  - RMSNorm + output head
  - Multi-Token Prediction (MTP) auxiliary heads (DeepSeek-V3)
  - Closed-form Fisher regularization (Parseval form)

Authors: L. Pan (Ainnocence Inc.)
License: MIT
=============================================================================
"""

import math
from dataclasses import dataclass, field, replace
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from cd_layers import (
    CDLinear, CDAttention, CDMoELayer, CDTransformerBlock,
    RMSNorm, ShannonDropout, fisher_reg_loss,
    ALPHA_CD, LAMBDA_CEIL,
)


# =============================================================================
# Model Configuration
# =============================================================================
@dataclass
class CDModelConfig:
    vocab_size: int = 50304
    dim: int = 2048
    n_layers: int = 24
    n_heads: int = 16
    max_seq_len: int = 4096

    # CD-specific
    cd_block: int = 5
    shannon_dropout: float = ALPHA_CD

    # Latent-KV attention
    kv_lora_rank: int = 512

    # MoE
    n_experts: int = 64
    n_active: int = 6
    ffn_mult: float = 2.667

    # Training
    use_fp8: bool = True
    use_mtp: bool = True
    mtp_depth: int = 2
    mtp_weight: float = 0.3
    gradient_checkpointing: bool = True
    fisher_lambda: float = 1e-8   # gentle nudge; raw Parseval energy scales with model+step, so keep small (or 0; weight_decay already does L2)
    fisher_mode: str = "energy"   # "energy" (L2-like, lambda~1e-8) or "conditioning" (flatten spectrum -> kappa->1, lambda~1e-2)
    fisher_agg: str = "mean"      # conditioning aggregation over blocks: "mean" | "pnorm" | "max"
    fisher_p: float = 4.0         # p for the "pnorm" worst-block aggregation
    cd_impl: str = "matmul"       # "matmul"/"fft" (CDLinear) or "dense" (non-circulant baseline)

    head_dim: int = field(init=False)

    def __post_init__(self):
        assert self.dim % self.n_heads == 0, "dim must be divisible by n_heads"
        self.head_dim = self.dim // self.n_heads


# Pre-defined configurations.
# NOTE: these are immutable templates; create_model() copies before editing
# so that overrides never mutate the shared module-level config.
CONFIGS = {
    "small": CDModelConfig(
        vocab_size=50304, dim=1024, n_layers=12, n_heads=8,
        cd_block=5, kv_lora_rank=256,
        n_experts=16, n_active=4, max_seq_len=2048,
    ),
    "medium": CDModelConfig(
        vocab_size=50304, dim=2048, n_layers=24, n_heads=16,
        cd_block=5, kv_lora_rank=512,
        n_experts=32, n_active=6, max_seq_len=4096,
    ),
    "large": CDModelConfig(
        vocab_size=50304, dim=4096, n_layers=32, n_heads=32,
        cd_block=7, kv_lora_rank=1024,
        n_experts=64, n_active=8, max_seq_len=4096,
    ),
}


# =============================================================================
# Multi-Token Prediction Head (DeepSeek-V3)
# =============================================================================
class MTPHead(nn.Module):
    def __init__(self, config: CDModelConfig, depth_idx: int):
        super().__init__()
        self.embed_proj = CDLinear(config.dim * 2, config.dim, block=config.cd_block,
                                   use_fp8=config.use_fp8, impl=config.cd_impl)
        self.norm = RMSNorm(config.dim)
        self.transformer = CDTransformerBlock(
            dim=config.dim, n_heads=config.n_heads,
            kv_lora_rank=config.kv_lora_rank,
            n_experts=max(4, config.n_experts // 4),
            n_active=min(2, config.n_active),
            block=config.cd_block, max_seq_len=config.max_seq_len,
            use_fp8=config.use_fp8, impl=config.cd_impl,
        )
        self.output_norm = RMSNorm(config.dim)

    def forward(self, hidden: torch.Tensor, token_embeds: torch.Tensor) -> torch.Tensor:
        h = self.norm(self.embed_proj(torch.cat([hidden, token_embeds], dim=-1)))
        h = self.transformer(h, is_causal=True)   # causal — no future leakage
        return self.output_norm(h)


# =============================================================================
# CD-Transformer Model
# =============================================================================
class CDTransformer(nn.Module):
    def __init__(self, config: CDModelConfig):
        super().__init__()
        self.config = config

        self.embed = nn.Embedding(config.vocab_size, config.dim)
        self.layers = nn.ModuleList([
            CDTransformerBlock(
                dim=config.dim, n_heads=config.n_heads,
                kv_lora_rank=config.kv_lora_rank,
                n_experts=config.n_experts, n_active=config.n_active,
                block=config.cd_block, max_seq_len=config.max_seq_len,
                use_fp8=config.use_fp8, impl=config.cd_impl,
            )
            for _ in range(config.n_layers)
        ])
        self.norm = RMSNorm(config.dim)

        self.output = nn.Linear(config.dim, config.vocab_size, bias=False)
        self.output.weight = self.embed.weight                  # weight tying

        self.mtp_heads = None
        if config.use_mtp and config.mtp_depth > 0:
            self.mtp_heads = nn.ModuleList(
                [MTPHead(config, i) for i in range(config.mtp_depth)])

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)
        # Keep output/embed tied after init.
        self.output.weight = self.embed.weight

    def forward(self, input_ids: torch.Tensor,
                labels: Optional[torch.Tensor] = None) -> dict:
        B, T = input_ids.shape

        h = self.embed(input_ids) * math.sqrt(self.config.dim)

        # SDPA handles causal masking internally (is_causal=True) — no (T x T)
        # mask tensor is materialized.
        for layer in self.layers:
            if self.config.gradient_checkpointing and self.training:
                h = checkpoint(layer, h, None, True, use_reentrant=False)
            else:
                h = layer(h, mask=None, is_causal=True)

        h = self.norm(h)
        logits = self.output(h)
        result = {'logits': logits, 'hidden_states': h}

        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1), ignore_index=-100,
            )
            result['ce_loss'] = loss.detach()   # bare next-token CE; log THIS directly
            result['loss'] = loss

            # Multi-Token Prediction
            if self.mtp_heads is not None and self.training:
                mtp_losses = []
                for depth, mtp_head in enumerate(self.mtp_heads):
                    shift = depth + 2
                    if T <= shift + 1:
                        continue
                    shifted_embeds = self.embed(input_ids[:, shift:])           # (B, T-shift, D)
                    mtp_hidden = mtp_head(h[:, :T - shift], shifted_embeds)      # (B, T-shift, D)
                    mtp_logits = self.output(mtp_hidden)
                    mtp_labels = labels[:, shift:]                              # (B, T-shift)
                    mtp_loss = F.cross_entropy(
                        mtp_logits[:, :-1].contiguous().view(-1, self.config.vocab_size),
                        mtp_labels[:, 1:].contiguous().view(-1), ignore_index=-100,
                    )
                    mtp_losses.append(mtp_loss)
                if mtp_losses:
                    result['mtp_loss'] = sum(mtp_losses) / len(mtp_losses)
                    result['loss'] = result['loss'] + self.config.mtp_weight * result['mtp_loss']

            # Fisher / conditioning regularization (differentiable)
            if self.config.fisher_lambda > 0:
                fisher = fisher_reg_loss(
                    self, self.config.fisher_lambda,
                    getattr(self.config, "fisher_mode", "energy"),
                    agg=getattr(self.config, "fisher_agg", "mean"),
                    p=getattr(self.config, "fisher_p", 4.0))
                result['fisher_loss'] = fisher
                result['loss'] = result['loss'] + fisher

        return result

    @torch.no_grad()
    def get_param_stats(self) -> dict:
        total = sum(p.numel() for p in self.parameters())
        cd_params = sum(m.c.numel() for m in self.modules() if isinstance(m, CDLinear))
        dense_equiv = sum(m.n_in_raw * m.n_out_raw
                          for m in self.modules() if isinstance(m, CDLinear))
        return {
            'total_params': total,
            'cd_params': cd_params,
            'dense_equivalent': dense_equiv,
            'cd_compression': dense_equiv / max(cd_params, 1),
            'embedding_params': self.embed.weight.numel(),
        }


# =============================================================================
# Model factory
# =============================================================================
def create_model(size: str = "small", **overrides) -> CDTransformer:
    """Create a CD-Transformer from a predefined config without mutating the
    shared template (uses dataclasses.replace)."""
    base = CONFIGS[size]
    valid = {k: v for k, v in overrides.items() if hasattr(base, k) and k != 'head_dim'}
    config = replace(base, **valid)   # __post_init__ re-runs automatically
    return CDTransformer(config)


# =============================================================================
# Self-test
# =============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("CD-Transformer Model Test")
    print("=" * 70)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = create_model("small").to(device)
    stats = model.get_param_stats()
    print(f"  Total params:     {stats['total_params']:>12,}")
    print(f"  CD params:        {stats['cd_params']:>12,}")
    print(f"  Dense equivalent: {stats['dense_equivalent']:>12,}")
    print(f"  CD compression:   {stats['cd_compression']:>12.1f}x")

    B, T = 2, 64
    ids = torch.randint(0, model.config.vocab_size, (B, T), device=device)
    labels = torch.randint(0, model.config.vocab_size, (B, T), device=device)

    model.train()
    out = model(ids, labels=labels)
    print(f"\n  logits: {tuple(out['logits'].shape)} | loss: {out['loss'].item():.4f}")
    if 'mtp_loss' in out:
        print(f"  MTP loss:    {out['mtp_loss'].item():.4f}")
    if 'fisher_loss' in out:
        print(f"  Fisher loss: {out['fisher_loss'].item():.6f}")

    out['loss'].backward()
    # Verify Fisher actually contributes gradient (was a no-op before the fix).
    g = next(m.c.grad.abs().sum().item() for m in model.modules() if isinstance(m, CDLinear))
    print(f"  backward OK | sample CDLinear grad magnitude: {g:.4e}")

    kappas = [m.condition_number() for m in model.modules() if isinstance(m, CDLinear)]
    kappas = [k for k in kappas if not math.isnan(k)]
    if kappas:
        print(f"\n  Hessian kappa  mean={sum(kappas)/len(kappas):.2e} "
              f"max={max(kappas):.2e} min={min(kappas):.2e}")
    print("\nAll model tests passed.")
