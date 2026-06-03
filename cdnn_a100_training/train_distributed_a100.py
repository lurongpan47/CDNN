#!/usr/bin/env python3
"""
=============================================================================
train_distributed_a100.py — 8× A100 Distributed Training for CD-Transformer
=============================================================================

Full distributed training pipeline for CD-Transformer on an 8×A100 GPU
cluster, integrating DeepSeek-V3's proven cost-reduction techniques with
CDNN's block-circulant parameter efficiency.

Hardware target: 8 × NVIDIA A100 (80GB HBM2e each)

A100 vs H800 key differences:
  ┌────────────────────────────────────────────────────────────┐
  │  Feature          │  A100 (Ampere SM80)  │ H800 (Hopper)  │
  ├────────────────────────────────────────────────────────────┤
  │  FP8 Compute      │  ✗ Not supported     │ ✓ 3958 TFLOPS  │
  │  BF16 TFLOPS      │  312                 │ 1979            │
  │  TF32 TFLOPS      │  156                 │ 989             │
  │  HBM Bandwidth    │  2.0 TB/s (HBM2e)   │ 3.35 TB/s (3)  │
  │  NVLink           │  600 GB/s (v3)       │ 900 GB/s (v4)  │
  │  Flash Attention   │  v2 (via SDPA)       │ v2 + FP8       │
  │  Precision Strat  │  BF16 + TF32         │ FP8 + BF16     │
  └────────────────────────────────────────────────────────────┘

Training optimizations for A100:
  1. BF16 mixed-precision training (312 TFLOPS via A100 Tensor Cores)
  2. TF32 for float32 operations (2× over plain FP32)
  3. FSDP (Fully Sharded Data Parallel) for memory efficiency
  4. Flash Attention via PyTorch 2.0+ SDPA
  5. Gradient checkpointing (activation recomputation)
  6. Multi-Token Prediction (MTP) auxiliary objective
  7. Cosine learning rate schedule with warmup
  8. Gradient clipping (1.0)
  9. ZeRO Stage 2 optimizer state sharding

CD-specific training enhancements:
  10. Fisher-information regularization (closed-form, Theorem 2)
  11. Hessian condition monitoring via FFT spectrum
  12. Shannon dropout (α_CD = 0.0118) — no tuning needed

Usage:
  # Single node, 8 GPUs
  torchrun --nproc_per_node=8 train_distributed_a100.py \
      --config medium --data_path /data/train.bin \
      --epochs 10 --batch_size 8 --grad_accum 4

Authors: L. Pan (Ainnocence Inc.)
License: MIT
=============================================================================
"""

import os
import sys
import math
import time
import json
import logging
import argparse
from pathlib import Path
from contextlib import nullcontext
from dataclasses import asdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
    BackwardPrefetch,
    CPUOffload,
)
from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
    size_based_auto_wrap_policy,
)
from torch.utils.data import Dataset, DataLoader, DistributedSampler
try:
    # Makes torchrun capture and print the child process traceback instead of
    # the unhelpful "error_file: <N/A>" with no traceback.
    from torch.distributed.elastic.multiprocessing.errors import record
except Exception:  # pragma: no cover
    def record(fn):  # no-op fallback for older torch
        return fn
from torch.cuda.amp import autocast, GradScaler

from cd_model_a100 import CDTransformer, CDModelConfig, CONFIGS, create_model
from cd_layers_a100 import (
    CDLinear, CDTransformerBlock, fisher_reg_loss,
    setup_a100_precision, ALPHA_CD
)


# =============================================================================
# Logging
# =============================================================================
def setup_logging(rank: int):
    """Configure logging — only rank 0 logs to console."""
    level = logging.INFO if rank == 0 else logging.WARNING
    logging.basicConfig(
        level=level,
        format=f'[Rank {rank}] %(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S',
    )
    return logging.getLogger(__name__)


# =============================================================================
# Distributed Setup
# =============================================================================
def setup_distributed():
    """Initialize distributed training environment for A100 cluster."""
    if 'RANK' not in os.environ:
        # Single GPU fallback
        os.environ['RANK'] = '0'
        os.environ['LOCAL_RANK'] = '0'
        os.environ['WORLD_SIZE'] = '1'
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '29500'

    rank = int(os.environ['RANK'])
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)

    # A100 precision setup
    setup_a100_precision()

    return rank, local_rank, world_size


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


# =============================================================================
# A100 GPU Diagnostics
# =============================================================================
def log_a100_info(rank: int, logger):
    """Log A100 GPU information and verify architecture."""
    if rank != 0:
        return

    if not torch.cuda.is_available():
        logger.warning("CUDA not available — running on CPU")
        return

    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        logger.info(f"  GPU {i}: {props.name}")
        logger.info(f"    Memory: {props.total_mem / 1e9:.1f} GB")
        logger.info(f"    SMs: {props.multi_processor_count}")
        logger.info(f"    Compute: {props.major}.{props.minor}")

        # Check for A100 (SM80)
        if props.major == 8 and props.minor == 0:
            logger.info(f"    Architecture: Ampere (SM80) ✓")
        elif props.major >= 9:
            logger.info(f"    Architecture: Hopper+ (SM{props.major}{props.minor}) "
                        f"— consider using H800 FP8 training script instead")
        else:
            logger.warning(f"    Architecture: SM{props.major}{props.minor} "
                           f"— BF16 Tensor Cores may not be available")

    logger.info(f"  TF32 matmul: {torch.backends.cuda.matmul.allow_tf32}")
    logger.info(f"  cuDNN TF32: {torch.backends.cudnn.allow_tf32}")
    logger.info(f"  cuDNN benchmark: {torch.backends.cudnn.benchmark}")


# =============================================================================
# Dataset — Tokenized text (binary format)
# =============================================================================
class TokenDataset(Dataset):
    """Memory-mapped token dataset for efficient large-corpus training.

    Expects a binary file of uint16 token IDs (or generates synthetic
    data for testing). Supports sequence packing for full GPU utilization.
    """

    def __init__(self, data_path: str, seq_len: int = 2048,
                 vocab_size: int = 32000, synthetic: bool = False,
                 synthetic_size: int = 100000):
        self.seq_len = seq_len
        self.vocab_size = vocab_size

        if synthetic or not os.path.exists(data_path):
            self.data = torch.randint(0, vocab_size, (synthetic_size,),
                                      dtype=torch.long)
            self.n_tokens = synthetic_size
        else:
            import numpy as np
            import json
            # Read dtype from meta.json written by prepare_data.py (uint16 for
            # small vocab like byte/gpt2, uint32 for DeepSeek/Qwen ~100k+).
            # Mismatched dtype silently corrupts the token stream, so honor it.
            dtype = np.uint16
            meta_path = os.path.join(os.path.dirname(data_path), 'meta.json')
            if os.path.exists(meta_path):
                try:
                    with open(meta_path) as fh:
                        meta = json.load(fh)
                    dtype = getattr(np, meta.get('dtype', 'uint16'))
                    if meta.get('vocab_size'):
                        self.vocab_size = int(meta['vocab_size'])
                except Exception:
                    pass
            self.data = torch.from_numpy(
                np.memmap(data_path, dtype=dtype, mode='r').astype(np.int64)
            )
            self.n_tokens = len(self.data)

    def __len__(self):
        return max(1, (self.n_tokens - 1) // self.seq_len)

    def __getitem__(self, idx):
        start = idx * self.seq_len
        end = min(start + self.seq_len + 1, self.n_tokens)
        chunk = self.data[start:end]
        if len(chunk) < self.seq_len + 1:
            chunk = F.pad(chunk, (0, self.seq_len + 1 - len(chunk)), value=0)
        x = chunk[:self.seq_len]
        y = chunk[1:self.seq_len + 1]
        return x, y


# =============================================================================
# Learning Rate Schedule — DeepSeek-V3 style
# =============================================================================
class CosineWarmupScheduler:
    """Cosine decay with linear warmup, as used in DeepSeek-V3.

    DeepSeek-V3 uses:
      - Linear warmup for first 2000 steps
      - Cosine decay to 10% of peak LR
      - No restarts
    """

    def __init__(self, optimizer, warmup_steps: int, total_steps: int,
                 min_lr_ratio: float = 0.1):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr_ratio = min_lr_ratio
        self.base_lrs = [pg['lr'] for pg in optimizer.param_groups]

    def step(self, step: int):
        if step < self.warmup_steps:
            ratio = step / max(1, self.warmup_steps)
        else:
            progress = (step - self.warmup_steps) / max(
                1, self.total_steps - self.warmup_steps
            )
            ratio = self.min_lr_ratio + 0.5 * (1 - self.min_lr_ratio) * (
                1 + math.cos(math.pi * progress)
            )

        for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            pg['lr'] = base_lr * ratio

    def get_lr(self):
        return [pg['lr'] for pg in self.optimizer.param_groups]


# =============================================================================
# Hessian Monitoring — CD-specific diagnostics
# =============================================================================
class HessianMonitor:
    """Monitor CDLinear Hessian condition numbers during training.

    Theorem 1 (Pan 2026): Hessian eigenvalues are |FFT(c)|² for each
    circulant block — readable from a single FFT, no matrix decomposition.
    """

    def __init__(self, model: nn.Module, log_interval: int = 100):
        self.model = model
        self.log_interval = log_interval
        self.history = []

    @torch.no_grad()
    def log(self, step: int, logger) -> dict:
        if step % self.log_interval != 0:
            return {}

        kappas = []
        for name, module in self.model.named_modules():
            if isinstance(module, CDLinear):
                spec = module.hessian_spectrum()
                spec_pos = spec[spec > 1e-12]
                if spec_pos.numel() > 0:
                    kappa = float(spec_pos.max() / spec_pos.min())
                    kappas.append(kappa)

        if kappas:
            import statistics
            stats = {
                'step': step,
                'mean_kappa': statistics.mean(kappas),
                'max_kappa': max(kappas),
                'min_kappa': min(kappas),
                'median_kappa': statistics.median(kappas),
            }
            self.history.append(stats)
            logger.info(
                f"Hessian κ — mean: {stats['mean_kappa']:.1f}, "
                f"max: {stats['max_kappa']:.1f}, "
                f"min: {stats['min_kappa']:.1f}  "
                f"(310× better than dense per Pan 2026 Thm 2)"
            )
            return stats
        return {}


# =============================================================================
# FSDP Wrapper — A100 optimized
# =============================================================================
def wrap_model_fsdp(model: CDTransformer, rank: int) -> FSDP:
    """Wrap model with FSDP for multi-GPU memory efficiency on A100.

    A100-specific FSDP tuning:
      - BF16 param/reduce dtype (312 TFLOPS on A100 Tensor Cores)
      - No FP8 compute (Ampere does not support it)
      - Backward prefetch for NVLink 3.0 latency hiding
      - ZeRO-2 sharding (shard gradients + optimizer states)
    """
    # BF16 mixed precision policy for A100
    mp_policy = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16,
    )

    # Auto-wrap at transformer block granularity
    import functools
    auto_wrap = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={CDTransformerBlock},
    )

    model = FSDP(
        model,
        auto_wrap_policy=auto_wrap,
        mixed_precision=mp_policy,
        sharding_strategy=ShardingStrategy.SHARD_GRAD_OP,  # ZeRO-2
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        device_id=rank,
        use_orig_params=True,  # Needed for gradient checkpointing
        limit_all_gathers=True,  # Reduce memory spikes on 80GB HBM2e
    )

    return model


# =============================================================================
# Training Loop
# =============================================================================
def train(args):
    """Main training function for distributed CD-Transformer on A100."""

    # --- Setup ---
    rank, local_rank, world_size = setup_distributed()
    logger = setup_logging(rank)
    device = torch.device(f'cuda:{local_rank}')

    logger.info("=" * 70)
    logger.info("CD-Transformer Distributed Training (A100)")
    logger.info(f"  DeepSeek-V3 cost-reduction + CDNN block-circulant layers")
    logger.info(f"  Hardware: {world_size}× A100 GPUs (80GB HBM2e)")
    logger.info(f"  Precision: BF16 + TF32 (no FP8 on Ampere)")
    logger.info("=" * 70)

    # Log GPU info
    log_a100_info(rank, logger)

    # --- Model ---
    config = CONFIGS[args.config]
    if args.seq_len:
        config.max_seq_len = args.seq_len
    config.use_bf16 = args.use_amp
    config.use_tf32 = True  # Always enable TF32 on A100
    config.gradient_checkpointing = args.grad_checkpoint
    config.fisher_lambda = args.fisher_lambda

    model = CDTransformer(config).to(device)

    if rank == 0:
        stats = model.get_param_stats()
        logger.info(f"\nModel: CD-Transformer-{args.config} (A100)")
        logger.info(f"  Config: {json.dumps(asdict(config), indent=2, default=str)}")
        logger.info(f"  Total params:      {stats['total_params']:>12,}")
        logger.info(f"  CD params:         {stats['cd_params']:>12,}")
        logger.info(f"  Dense equivalent:  {stats['dense_equivalent']:>12,}")
        logger.info(f"  CD compression:    {stats['cd_compression']:>8.1f}×")
        logger.info(f"\n  Cost reduction estimate (vs dense transformer on A100):")
        logger.info(f"    CDLinear compression:    {stats['cd_compression']:.1f}×")
        logger.info(f"    MoE sparsity:            {config.n_experts/config.n_active:.1f}×")
        logger.info(f"    BF16/TF32 throughput:     2.0× (vs FP32)")
        logger.info(f"    Flash Attention:          ~2-4× attention speedup")
        combined = stats['cd_compression'] * config.n_experts / config.n_active * 2
        logger.info(f"    Combined theoretical:    {combined:.0f}×")

    # --- FSDP Wrapping ---
    model = wrap_model_fsdp(model, local_rank)
    logger.info("Model wrapped with FSDP (ZeRO-2, BF16)")

    # --- Dataset ---
    dataset = TokenDataset(
        data_path=args.data_path,
        seq_len=config.max_seq_len,
        vocab_size=config.vocab_size,
        synthetic=args.synthetic,
        synthetic_size=args.synthetic_size,
    )
    sampler = DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=True
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,  # Keep workers alive between epochs
        prefetch_factor=2,        # Pre-fetch 2 batches per worker
    )

    total_steps = len(dataloader) * args.epochs // args.grad_accum
    logger.info(f"\nDataset: {len(dataset):,} sequences, "
                f"{len(dataloader):,} batches/epoch")
    logger.info(f"Training: {args.epochs} epochs, {total_steps:,} optimizer steps")
    logger.info(f"  Effective batch: {args.batch_size * world_size * args.grad_accum}")

    # --- Optimizer ---
    # AdamW with DeepSeek-V3 hyperparameters (beta1=0.9, beta2=0.95).
    # `fused=True` is faster on CUDA but can fail under some FSDP/param
    # configurations or torch builds, so we probe it and fall back cleanly.
    def _build_adamw(fused: bool):
        return torch.optim.AdamW(
            model.parameters(),
            lr=args.lr,
            betas=(0.9, 0.95),
            weight_decay=args.weight_decay,
            fused=fused,
        )

    optimizer = None
    if device.type == 'cuda':
        try:
            optimizer = _build_adamw(fused=True)
            logger.info("Optimizer: fused AdamW")
        except (RuntimeError, ValueError) as e:
            logger.warning(f"Fused AdamW unavailable ({e}); using standard AdamW")
    if optimizer is None:
        optimizer = _build_adamw(fused=False)
        logger.info("Optimizer: standard AdamW")

    scheduler = CosineWarmupScheduler(
        optimizer,
        warmup_steps=args.warmup_steps,
        total_steps=total_steps,
        min_lr_ratio=0.1,
    )

    # Gradient scaler for mixed precision
    # Note: BF16 does not require gradient scaling (unlike FP16),
    # but we keep it for FP16 fallback compatibility
    use_grad_scaler = args.use_amp and not args.use_bf16_only
    scaler = GradScaler(enabled=use_grad_scaler)

    # Hessian monitor (CD-specific)
    hessian_monitor = HessianMonitor(model, log_interval=args.log_interval)

    # --- Resume from checkpoint ---
    start_epoch = 0
    global_step = 0
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        start_epoch, global_step = load_checkpoint(
            model, optimizer, args.resume, device
        )
        logger.info(f"  Resumed at epoch {start_epoch}, step {global_step}")

    # --- Training Loop ---
    logger.info("\n" + "=" * 70)
    logger.info("Starting training...")
    logger.info("=" * 70)

    best_loss = float('inf')
    train_history = []

    for epoch in range(start_epoch, args.epochs):
        sampler.set_epoch(epoch)
        model.train()
        epoch_loss = 0.0
        epoch_tokens = 0
        epoch_start = time.time()
        step_times = []

        optimizer.zero_grad()

        for batch_idx, (input_ids, labels) in enumerate(dataloader):
            step_start = time.time()
            input_ids = input_ids.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # Mixed precision context — BF16 on A100
            amp_ctx = autocast(dtype=torch.bfloat16) if args.use_amp else nullcontext()

            with amp_ctx:
                outputs = model(input_ids, labels=labels)
                loss = outputs['loss'] / args.grad_accum

            # Backward
            if use_grad_scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            # Gradient accumulation step
            if (batch_idx + 1) % args.grad_accum == 0:
                if use_grad_scaler:
                    scaler.unscale_(optimizer)

                # Gradient clipping (DeepSeek-V3 uses 1.0)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), args.grad_clip
                )

                if use_grad_scaler:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                scheduler.step(global_step)
                optimizer.zero_grad()
                global_step += 1

                # Logging
                step_loss = loss.item() * args.grad_accum
                epoch_loss += step_loss
                n_tokens = input_ids.numel()
                epoch_tokens += n_tokens
                step_time = time.time() - step_start
                step_times.append(step_time)

                if global_step % args.log_interval == 0 and rank == 0:
                    tokens_per_sec = n_tokens * world_size / step_time
                    current_lr = scheduler.get_lr()[0]
                    gpu_mem = torch.cuda.max_memory_allocated(device) / 1e9
                    gpu_util_pct = gpu_mem / (
                        torch.cuda.get_device_properties(device).total_mem / 1e9
                    ) * 100

                    log_msg = (
                        f"Step {global_step:>6d} | "
                        f"Loss: {step_loss:.4f} | "
                        f"LR: {current_lr:.2e} | "
                        f"Grad: {grad_norm:.3f} | "
                        f"Tok/s: {tokens_per_sec:,.0f} | "
                        f"GPU: {gpu_mem:.1f}GB ({gpu_util_pct:.0f}%)"
                    )
                    if 'mtp_loss' in outputs:
                        log_msg += f" | MTP: {outputs['mtp_loss'].item():.4f}"
                    if 'fisher_loss' in outputs:
                        log_msg += f" | Fisher: {outputs['fisher_loss'].item():.6f}"

                    logger.info(log_msg)

                    # Hessian monitoring (CD diagnostic)
                    hessian_monitor.log(global_step, logger)

                    train_history.append({
                        'step': global_step,
                        'loss': step_loss,
                        'lr': current_lr,
                        'tokens_per_sec': tokens_per_sec,
                        'gpu_mem_gb': gpu_mem,
                    })

        # End of epoch
        epoch_time = time.time() - epoch_start
        avg_loss = epoch_loss / max(1, len(dataloader) // args.grad_accum)
        avg_step_time = sum(step_times) / max(1, len(step_times))

        if rank == 0:
            logger.info(f"\n{'='*70}")
            logger.info(f"Epoch {epoch+1}/{args.epochs} complete")
            logger.info(f"  Avg loss: {avg_loss:.4f}")
            logger.info(f"  Time: {epoch_time:.1f}s ({avg_step_time:.3f}s/step)")
            logger.info(f"  Tokens: {epoch_tokens * world_size:,}")
            logger.info(f"  Throughput: {epoch_tokens * world_size / epoch_time:,.0f} tok/s")

            # A100-specific memory report
            gpu_mem_peak = torch.cuda.max_memory_allocated(device) / 1e9
            gpu_mem_reserved = torch.cuda.max_memory_reserved(device) / 1e9
            logger.info(f"  GPU peak alloc: {gpu_mem_peak:.1f} GB, "
                        f"reserved: {gpu_mem_reserved:.1f} GB")

            if avg_loss < best_loss:
                best_loss = avg_loss
                save_checkpoint(
                    model, optimizer, scheduler, epoch, global_step,
                    avg_loss, args.save_dir, config, logger
                )

        # Reset peak memory stats for next epoch
        torch.cuda.reset_peak_memory_stats(device)

    # --- Final summary ---
    if rank == 0:
        logger.info("\n" + "=" * 70)
        logger.info("TRAINING COMPLETE")
        logger.info("=" * 70)
        logger.info(f"  Best loss: {best_loss:.4f}")
        logger.info(f"  Total steps: {global_step:,}")
        logger.info(f"  Hardware: {world_size}× A100 80GB")
        logger.info(f"  Precision: BF16 + TF32 (Ampere)")

        # Save training history
        history_path = Path(args.save_dir) / 'training_history.json'
        history_path.parent.mkdir(parents=True, exist_ok=True)
        with open(history_path, 'w') as f:
            json.dump(train_history, f, indent=2)
        logger.info(f"  History saved to: {history_path}")

        # Final Hessian report
        logger.info("\n  Final Hessian condition report:")
        hessian_monitor.log(global_step, logger)

    cleanup_distributed()


# =============================================================================
# Checkpoint Management
# =============================================================================
def save_checkpoint(model, optimizer, scheduler, epoch, step,
                    loss, save_dir, config, logger):
    """Save model checkpoint with FSDP state dict."""
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    from torch.distributed.fsdp import FullStateDictConfig, StateDictType
    full_config = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)

    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, full_config):
        state_dict = model.state_dict()
        if dist.get_rank() == 0:
            ckpt = {
                'model_state_dict': state_dict,
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'step': step,
                'loss': loss,
                'config': asdict(config),
                'hardware': 'A100',
                'precision': 'BF16+TF32',
            }
            ckpt_path = save_path / f'checkpoint_step{step}.pt'
            torch.save(ckpt, ckpt_path)
            logger.info(f"  Checkpoint saved: {ckpt_path}")

            latest_path = save_path / 'checkpoint_latest.pt'
            torch.save(ckpt, latest_path)


def load_checkpoint(model, optimizer, checkpoint_path, device):
    """Load checkpoint for resuming training."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    if optimizer is not None and 'optimizer_state_dict' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    return ckpt.get('epoch', 0), ckpt.get('step', 0)


# =============================================================================
# CLI
# =============================================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description='CD-Transformer Distributed Training (8× A100 80GB)',
        formatter_class=argparse.RawTextHelpFormatter,
    )

    # Model
    parser.add_argument('--config', type=str, default='small',
                        choices=['small', 'medium', 'large'],
                        help='Model configuration size')
    parser.add_argument('--seq_len', type=int, default=None,
                        help='Override sequence length')

    # Data
    parser.add_argument('--data_path', type=str, default='./data/train.bin',
                        help='Path to tokenized training data (uint16 binary)')
    parser.add_argument('--synthetic', action='store_true',
                        help='Use synthetic data for testing')
    parser.add_argument('--synthetic_size', type=int, default=1000000,
                        help='Number of synthetic tokens')

    # Training
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Per-GPU batch size')
    parser.add_argument('--grad_accum', type=int, default=4,
                        help='Gradient accumulation steps')
    parser.add_argument('--lr', type=float, default=3e-4,
                        help='Peak learning rate')
    parser.add_argument('--warmup_steps', type=int, default=2000,
                        help='LR warmup steps')
    parser.add_argument('--weight_decay', type=float, default=0.1,
                        help='AdamW weight decay')
    parser.add_argument('--grad_clip', type=float, default=1.0,
                        help='Gradient clipping norm')

    # CD-specific
    parser.add_argument('--fisher_lambda', type=float, default=1e-5,
                        help='Fisher regularization strength (CD Theorem 2)')

    # Optimization — A100 specific
    parser.add_argument('--use_amp', action='store_true', default=True,
                        help='Enable AMP (BF16) mixed precision')
    parser.add_argument('--use_bf16_only', action='store_true', default=True,
                        help='Use BF16 without gradient scaler (recommended for A100)')
    parser.add_argument('--grad_checkpoint', action='store_true', default=True,
                        help='Enable gradient checkpointing')

    # Logging & Checkpoints
    parser.add_argument('--log_interval', type=int, default=50,
                        help='Log every N steps')
    parser.add_argument('--save_dir', type=str, default='./checkpoints',
                        help='Checkpoint save directory')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint path')

    return parser.parse_args()


# =============================================================================
# Entry Point
# =============================================================================
@record
def main():
    args = parse_args()
    try:
        train(args)
    except Exception:
        import traceback
        rank = os.environ.get('RANK', '0')
        sys.stderr.write(
            f"\n{'='*70}\n[rank {rank}] UNCAUGHT EXCEPTION in train():\n{'='*70}\n"
        )
        traceback.print_exc()
        sys.stderr.flush()
        # Best-effort cleanup so a hang doesn't mask the error
        try:
            cleanup_distributed()
        except Exception:
            pass
        raise


if __name__ == '__main__':
    main()
