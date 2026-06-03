#!/usr/bin/env python3
"""
=============================================================================
cd_model_a100.py — CD-Transformer: Full Model Architecture (A100 Optimized)
=============================================================================

A DeepSeek-V3-inspired transformer using Communication Dynamics (CD)
block-circulant layers throughout, adapted for 8× A100 GPU training.

A100 Ampere adaptations:
  - No FP8 path (Ampere SM80 does not support native FP8)
  - BF16 mixed precision via torch.cuda.amp
  - TF32 for float32 operations (automatic via torch.backends)
  - Flash Attention v2 via PyTorch 2.0+ SDPA
  - Memory bandwidth tuning for 80GB HBM2e (2 TB/s)

Architecture summary:
  - Token embedding + learned positional encoding
  - N × CDTransformerBlock (MLA attention + MoE FFN)
  - RMSNorm + output head with weight tying
  - Multi-Token Prediction (MTP) auxiliary heads (DeepSeek-V3)

Model configurations (A100-tuned batch sizes):
  - CD-Transformer-Small   (~300M total params)
  - CD-Transformer-Medium  (~2B total params)
  - CD-Transformer-Large   (~15B total params)

Authors: L. Pan (Ainnocence Inc.)
License: MIT
=============================================================================
"""

import math
from dataclasses import dataclass, field
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from cd_layers_a100 import (
    CDLinear, CDAttention, CDMoELayer, CDTransformerBlock,
    RMSNorm, ShannonDropout, fisher_reg_loss, setup_a100_precision,
    ALPHA_CD, LAMBDA_CEIL
)


# =============================================================================
# Model Configuration
# =============================================================================
@dataclass
class CDModelConfig:
    """Configuration for CD-Transformer model (A100 variant).

    A100-specific tuning notes:
      - BF16 is the primary mixed-precision dtype (312 TFLOPS)
      - TF32 used for float32 ops (19.5 TFLOPS, 2× over FP32)
      - No FP8 path; throughput gain comes from BF16 + flash attention
      - HBM2e at 2 TB/s (vs H800 HBM3 at 3.35 TB/s), so slightly
        lower memory-bandwidth-bound throughput for attention
    """
    # Model dimensions
    vocab_size: int = 32000
    dim: int = 2048
    n_layers: int = 24
    n_heads: int = 16
    max_seq_len: int = 4096

    # CD-specific
    cd_block: int = 5           # Polygon multiplicity B = 2L+1
    shannon_dropout: float = ALPHA_CD  # 0.0118

    # MLA (Multi-head Latent Attention)
    kv_lora_rank: int = 512     # KV compression dimension
    qk_rope_dim: int = 64      # RoPE dimension

    # MoE (Mixture of Experts)
    n_experts: int = 64         # Total experts per layer
    n_active: int = 6           # Active experts per token
    ffn_mult: float = 2.667     # FFN hidden dim multiplier

    # Training — A100 adapted
    use_bf16: bool = True       # BF16 mixed precision (A100 primary)
    use_tf32: bool = True       # TF32 for float32 matmuls
    use_mtp: bool = True        # Multi-Token Prediction
    mtp_depth: int = 2          # Number of MTP heads
    gradient_checkpointing: bool = True
    fisher_lambda: float = 1e-5 # Fisher regularization strength

    # Derived
    head_dim: int = field(init=False)

    def __post_init__(self):
        self.head_dim = self.dim // self.n_heads
        assert self.dim % self.n_heads == 0, "dim must be divisible by n_heads"


# Pre-defined configurations — tuned for 8× A100 80GB
CONFIGS = {
    "small": CDModelConfig(
        vocab_size=32000, dim=1024, n_layers=12, n_heads=8,
        cd_block=5, kv_lora_rank=256,
        n_experts=16, n_active=4, max_seq_len=2048,
    ),
    "medium": CDModelConfig(
        vocab_size=32000, dim=2048, n_layers=24, n_heads=16,
        cd_block=5, kv_lora_rank=512,
        n_experts=32, n_active=6, max_seq_len=4096,
    ),
    "large": CDModelConfig(
        vocab_size=32000, dim=4096, n_layers=32, n_heads=32,
        cd_block=7, kv_lora_rank=1024,
        n_experts=64, n_active=8, max_seq_len=4096,
    ),
}


# =============================================================================
# Multi-Token Prediction Head (DeepSeek-V3)
# =============================================================================
class MTPHead(nn.Module):
    """Multi-Token Prediction head.

    DeepSeek-V3 trains with an MTP objective that predicts multiple
    future tokens at each position, improving training signal density
    and enabling speculative decoding at inference time.
    """

    def __init__(self, config: CDModelConfig, depth_idx: int):
        super().__init__()
        self.embed_proj = CDLinear(
            config.dim * 2, config.dim,
            block=config.cd_block
        )
        self.norm = RMSNorm(config.dim)
        # Shared transformer layer for each MTP depth
        self.transformer = CDTransformerBlock(
            dim=config.dim,
            n_heads=config.n_heads,
            kv_lora_rank=config.kv_lora_rank,
            n_experts=max(4, config.n_experts // 4),  # Fewer experts for MTP
            n_active=min(2, config.n_active),
            block=config.cd_block,
            max_seq_len=config.max_seq_len,
        )
        self.output_norm = RMSNorm(config.dim)

    def forward(self, hidden: torch.Tensor,
                token_embeds: torch.Tensor) -> torch.Tensor:
        """
        hidden: (B, T, D) — previous depth's hidden states
        token_embeds: (B, T, D) — shifted token embeddings for next position
        """
        combined = torch.cat([hidden, token_embeds], dim=-1)
        h = self.embed_proj(combined)
        h = self.norm(h)
        h = self.transformer(h)
        return self.output_norm(h)


# =============================================================================
# CD-Transformer Model (A100)
# =============================================================================
class CDTransformer(nn.Module):
    """CD-Transformer: Communication Dynamics Transformer with DeepSeek-V3
    cost-reduction techniques, adapted for A100 Ampere GPUs.

    Architecture:
      1. Token embedding (dense, shared with output head)
      2. N × CDTransformerBlock
         - CDAttention (MLA with CD-compressed projections)
         - CDMoELayer (MoE with CDLinear experts)
         - ShannonDropout (α_CD = 0.0118)
         - RMSNorm (pre-norm)
      3. RMSNorm + output head (tied weights)
      4. Optional MTP auxiliary heads

    Cost reduction breakdown (vs dense transformer, A100):
      ┌──────────────────────────────────────────────────────┐
      │  Technique           │ Reduction Factor              │
      ├──────────────────────────────────────────────────────┤
      │  CDLinear (B=5)      │ 5× parameter compression      │
      │  MoE (64/6)         │ ~10× compute (6/64 active)    │
      │  MLA (KV compress)  │ ~4× KV cache reduction        │
      │  BF16/TF32 (A100)   │ ~2× throughput vs FP32        │
      │  Flash Attention     │ ~2-4× attention speedup       │
      │  Shannon dropout    │ Better generalization          │
      │  Fisher regularizer │ Better conditioning → faster   │
      └──────────────────────────────────────────────────────┘
    """

    def __init__(self, config: CDModelConfig):
        super().__init__()
        self.config = config

        # Configure A100 precision
        if config.use_tf32:
            setup_a100_precision()

        # Token embedding
        self.embed = nn.Embedding(config.vocab_size, config.dim)

        # Transformer layers
        self.layers = nn.ModuleList([
            CDTransformerBlock(
                dim=config.dim,
                n_heads=config.n_heads,
                kv_lora_rank=config.kv_lora_rank,
                n_experts=config.n_experts,
                n_active=config.n_active,
                block=config.cd_block,
                max_seq_len=config.max_seq_len,
            )
            for _ in range(config.n_layers)
        ])

        self.norm = RMSNorm(config.dim)

        # Output head (weight-tied with embedding)
        self.output = nn.Linear(config.dim, config.vocab_size, bias=False)
        self.output.weight = self.embed.weight  # Weight tying

        # Multi-Token Prediction heads
        self.mtp_heads = None
        if config.use_mtp:
            self.mtp_heads = nn.ModuleList([
                MTPHead(config, i) for i in range(config.mtp_depth)
            ])

        # Causal mask buffer
        mask = torch.triu(
            torch.ones(config.max_seq_len, config.max_seq_len), diagonal=1
        ).bool()
        self.register_buffer('causal_mask', ~mask, persistent=False)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights following DeepSeek-V3 practices."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, std=0.02)

    def forward(self, input_ids: torch.Tensor,
                labels: Optional[torch.Tensor] = None
                ) -> dict:
        """
        input_ids: (batch, seq_len) — token indices
        labels: (batch, seq_len) — target token indices for loss

        Returns:
            dict with 'logits', 'loss', 'mtp_losses', 'hidden_states'
        """
        B, T = input_ids.shape
        device = input_ids.device

        # Token embeddings
        h = self.embed(input_ids) * math.sqrt(self.config.dim)

        # Causal mask
        mask = self.causal_mask[:T, :T].unsqueeze(0).unsqueeze(0)

        # Transformer layers with optional gradient checkpointing
        for layer in self.layers:
            if self.config.gradient_checkpointing and self.training:
                h = checkpoint(layer, h, mask, use_reentrant=False)
            else:
                h = layer(h, mask=mask)

        h = self.norm(h)
        logits = self.output(h)

        result = {'logits': logits, 'hidden_states': h}

        if labels is not None:
            # Main language modeling loss
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100
            )
            result['loss'] = loss

            # Multi-Token Prediction losses
            if self.mtp_heads is not None and self.training:
                mtp_losses = []
                hidden = h
                for depth, mtp_head in enumerate(self.mtp_heads):
                    shift = depth + 2
                    if T > shift:
                        shifted_embeds = self.embed(input_ids[:, shift:])
                        shifted_embeds = F.pad(
                            shifted_embeds, (0, 0, 0, shift), value=0
                        )
                        mtp_hidden = mtp_head(hidden[:, :-shift], shifted_embeds[:, :-shift])
                        mtp_logits = self.output(mtp_hidden)

                        mtp_labels = labels[:, shift:shift + mtp_logits.shape[1]]
                        if mtp_labels.shape[1] > 0:
                            mtp_loss = F.cross_entropy(
                                mtp_logits[:, :-1].contiguous().view(-1, self.config.vocab_size),
                                mtp_labels[:, 1:].contiguous().view(-1),
                                ignore_index=-100
                            )
                            mtp_losses.append(mtp_loss)

                if mtp_losses:
                    result['mtp_loss'] = sum(mtp_losses) / len(mtp_losses)
                    result['loss'] = loss + 0.3 * result['mtp_loss']

            # Fisher regularization (CD-specific)
            if self.config.fisher_lambda > 0:
                fisher = fisher_reg_loss(self, self.config.fisher_lambda)
                result['fisher_loss'] = fisher
                result['loss'] = result['loss'] + fisher

        return result

    def get_param_stats(self) -> dict:
        """Report parameter statistics and compression ratios."""
        total = sum(p.numel() for p in self.parameters())
        cd_params = sum(
            m.c.numel() for m in self.modules() if isinstance(m, CDLinear)
        )
        dense_equiv = sum(
            m.n_in_raw * m.n_out_raw
            for m in self.modules() if isinstance(m, CDLinear)
        )
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
    """Create a CD-Transformer model from a predefined configuration.

    Args:
        size: One of "small", "medium", "large"
        **overrides: Override any config field
    """
    config = CONFIGS[size]
    for k, v in overrides.items():
        if hasattr(config, k):
            setattr(config, k, v)
    config.__post_init__()
    return CDTransformer(config)


# =============================================================================
# Self-test
# =============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("CD-Transformer Model Test (A100)")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    for size in ["small"]:
        print(f"\n--- {size.upper()} configuration ---")
        model = create_model(size).to(device)
        stats = model.get_param_stats()

        print(f"  Total params:      {stats['total_params']:>12,}")
        print(f"  CD params:         {stats['cd_params']:>12,}")
        print(f"  Dense equivalent:  {stats['dense_equivalent']:>12,}")
        print(f"  CD compression:    {stats['cd_compression']:>12.1f}×")
        print(f"  Embedding params:  {stats['embedding_params']:>12,}")

        # Forward pass test
        B, T = 2, 64
        ids = torch.randint(0, model.config.vocab_size, (B, T), device=device)
        labels = torch.randint(0, model.config.vocab_size, (B, T), device=device)

        model.train()
        out = model(ids, labels=labels)
        print(f"\n  Forward pass:")
        print(f"    Logits shape: {out['logits'].shape}")
        print(f"    Loss: {out['loss'].item():.4f}")
        if 'mtp_loss' in out:
            print(f"    MTP loss: {out['mtp_loss'].item():.4f}")
        if 'fisher_loss' in out:
            print(f"    Fisher loss: {out['fisher_loss'].item():.6f}")

        out['loss'].backward()
        print(f"    Backward pass: OK")

    print("\n" + "=" * 70)
    print("All model tests passed.")
