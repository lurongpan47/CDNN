#!/usr/bin/env python3
"""
=============================================================================
train_distributed.py — H800 8-GPU Distributed Training for CD-Transformer
=============================================================================

Distributed training for CD-Transformer on an 8xH800 node, integrating
DeepSeek-V3 cost-reduction techniques with CDNN block-circulant efficiency.

Fixed / hardened in this revision:
  * BF16 autocast WITHOUT GradScaler (GradScaler is for FP16 only).
  * Lazy uint16 memmap dataset (no longer loads the whole corpus as int64).
  * FSDP-correct checkpoint save/load (model + optimizer sharded state).
  * Working --resume.
  * Auxiliary-loss-free router-bias update each optimizer step.
  * Fused AdamW guarded for CUDA availability.

Usage:
  torchrun --standalone --nproc_per_node=8 train_distributed.py \
      --config medium --data_path ./data/train.bin --epochs 10

Authors: L. Pan (Ainnocence Inc.)
License: MIT
=============================================================================
"""

import os
import math
import time
import json
import logging
import argparse
from pathlib import Path
from contextlib import nullcontext
from dataclasses import asdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
    BackwardPrefetch,
    StateDictType,
    FullStateDictConfig,
    FullOptimStateDictConfig,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.utils.data import Dataset, DataLoader, DistributedSampler

from cd_model import CDTransformer, CONFIGS
from cd_layers import CDLinear, CDTransformerBlock, update_router_biases


# =============================================================================
# Logging
# =============================================================================
def setup_logging(rank: int):
    logging.basicConfig(
        level=logging.INFO if rank == 0 else logging.WARNING,
        format=f'[Rank {rank}] %(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S',
    )
    return logging.getLogger(__name__)


# =============================================================================
# Distributed setup
# =============================================================================
def setup_distributed():
    if 'RANK' not in os.environ:
        os.environ.setdefault('RANK', '0')
        os.environ.setdefault('LOCAL_RANK', '0')
        os.environ.setdefault('WORLD_SIZE', '1')
        os.environ.setdefault('MASTER_ADDR', 'localhost')
        os.environ.setdefault('MASTER_PORT', '29500')

    rank = int(os.environ['RANK'])
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
    return rank, local_rank, world_size


def cleanup_distributed():
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


# =============================================================================
# Dataset — lazy uint16 memmap (à la nanoGPT)
# =============================================================================
class TokenDataset(Dataset):
    """Memory-mapped uint16 token dataset.

    The previous version did ``np.memmap(...).astype(np.int64)`` which copied
    the ENTIRE corpus into RAM as 8-byte ints, defeating the memmap. Here the
    memmap stays on disk and only per-sample slices are materialized.
    """

    def __init__(self, data_path: str, seq_len: int = 2048,
                 vocab_size: int = 32000, synthetic: bool = False,
                 synthetic_size: int = 1_000_000):
        self.seq_len = seq_len
        self.vocab_size = vocab_size

        if synthetic or not os.path.exists(data_path):
            rng = np.random.default_rng(0)
            self.data = rng.integers(0, vocab_size, size=synthetic_size,
                                     dtype=np.uint16)
            self.n_tokens = synthetic_size
        else:
            self.data = np.memmap(data_path, dtype=np.uint16, mode='r')
            self.n_tokens = self.data.shape[0]

    def __len__(self):
        return max(1, (self.n_tokens - 1) // self.seq_len)

    def __getitem__(self, idx):
        start = idx * self.seq_len
        end = min(start + self.seq_len + 1, self.n_tokens)
        chunk = np.asarray(self.data[start:end], dtype=np.int64)   # small copy
        x = torch.from_numpy(chunk[:-1].copy())
        y = torch.from_numpy(chunk[1:].copy())
        if x.numel() < self.seq_len:
            x = F.pad(x, (0, self.seq_len - x.numel()))
            y = F.pad(y, (0, self.seq_len - y.numel()), value=-100)  # ignore pad
        return x, y


# =============================================================================
# LR schedule — cosine with warmup (DeepSeek-V3 style)
# =============================================================================
class CosineWarmupScheduler:
    def __init__(self, optimizer, warmup_steps, total_steps, min_lr_ratio=0.1):
        self.optimizer = optimizer
        self.warmup_steps = max(1, warmup_steps)
        self.total_steps = max(self.warmup_steps + 1, total_steps)
        self.min_lr_ratio = min_lr_ratio
        self.base_lrs = [pg['lr'] for pg in optimizer.param_groups]

    def step(self, step):
        if step < self.warmup_steps:
            ratio = step / self.warmup_steps
        else:
            progress = (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            progress = min(1.0, progress)
            ratio = self.min_lr_ratio + 0.5 * (1 - self.min_lr_ratio) * (
                1 + math.cos(math.pi * progress))
        for pg, base in zip(self.optimizer.param_groups, self.base_lrs):
            pg['lr'] = base * ratio

    def get_lr(self):
        return [pg['lr'] for pg in self.optimizer.param_groups]


# =============================================================================
# Hessian monitoring (CD diagnostic). Valid under ZeRO-2 (full params present).
# =============================================================================
class HessianMonitor:
    def __init__(self, model, log_interval=100, hessian_interval=None,
                 max_layers=24):
        self.model = model
        # kappa moves slowly and the full-model summon_full_params all-gather is
        # the most expensive thing on a logged step (it caused the throughput dip
        # in profiling). Sample it far less often than the loss log, and cap the
        # number of layers inspected so the cost stays bounded on large models.
        self.log_interval = log_interval
        hi = hessian_interval or max(log_interval, 500)
        # the monitor is only *called* on log_interval steps, so snap up to a
        # multiple of it (otherwise the two cadences rarely coincide)
        self.hessian_interval = max(log_interval, (hi // log_interval) * log_interval)
        self.max_layers = max_layers
        self.history = []

    @torch.no_grad()
    def log(self, step, logger):
        # All ranks evaluate this predicate identically (same step), so they
        # enter/skip the collective together — no deadlock.
        if step % self.hessian_interval != 0:
            return {}
        is_fsdp = isinstance(self.model, FSDP) or any(
            isinstance(m, FSDP) for m in self.model.modules())
        ctx = (FSDP.summon_full_params(self.model, writeback=False, recurse=True)
               if is_fsdp else nullcontext())
        kappas = []
        with ctx:
            cd = [m for _, m in self.model.named_modules()
                  if isinstance(m, CDLinear) and m.c is not None]
            # Inspect an evenly-spaced subset for a representative estimate.
            if len(cd) > self.max_layers:
                stride = len(cd) // self.max_layers
                cd = cd[::stride][:self.max_layers]
            for module in cd:
                if module.c.numel() == 0:        # still sharded/empty — skip
                    continue
                spec = module.hessian_spectrum()
                spec = spec[spec > 1e-12]
                if spec.numel() > 0:
                    kappas.append(float(spec.max() / spec.min()))
        if not kappas:
            return {}
        import statistics
        stats = {
            'step': step,
            'mean_kappa': statistics.mean(kappas),
            'max_kappa': max(kappas),
            'min_kappa': min(kappas),
            'median_kappa': statistics.median(kappas),
        }
        self.history.append(stats)
        rank0 = (not dist.is_initialized()) or dist.get_rank() == 0
        if rank0:
            logger.info(f"Hessian kappa — mean: {stats['mean_kappa']:.3g}, "
                        f"max: {stats['max_kappa']:.3g}, min: {stats['min_kappa']:.3g}")
        return stats


# =============================================================================
# FSDP wrapping
# =============================================================================
def wrap_model_fsdp(model, local_rank):
    mp_policy = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16,
    )
    import functools
    auto_wrap = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={CDTransformerBlock},
    )
    return FSDP(
        model,
        auto_wrap_policy=auto_wrap,
        mixed_precision=mp_policy,
        sharding_strategy=ShardingStrategy.SHARD_GRAD_OP,   # ZeRO-2
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        device_id=local_rank,
        use_orig_params=True,            # required for grad checkpointing
        limit_all_gathers=True,
    )


# =============================================================================
# Checkpointing (FSDP-correct)
# =============================================================================
def save_checkpoint(model, optimizer, scheduler, epoch, step, loss, save_dir, config, logger):
    save_path = Path(save_dir)
    cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    ocfg = FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, cfg, ocfg):
        model_sd = model.state_dict()
        optim_sd = FSDP.optim_state_dict(model, optimizer)   # gathers sharded opt state
    if dist.get_rank() == 0:
        save_path.mkdir(parents=True, exist_ok=True)
        ckpt = {
            'model_state_dict': model_sd,
            'optimizer_state_dict': optim_sd,
            'scheduler_base_lrs': scheduler.base_lrs,
            'epoch': epoch, 'step': step, 'loss': loss,
            'config': asdict(config),
        }
        torch.save(ckpt, save_path / f'checkpoint_step{step}.pt')
        torch.save(ckpt, save_path / 'checkpoint_latest.pt')
        logger.info(f"  Checkpoint saved: {save_path / f'checkpoint_step{step}.pt'}")


def load_checkpoint(model, optimizer, ckpt_path, logger):
    """Load an FSDP checkpoint. Broadcasts rank0 full state to all shards."""
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False) \
        if dist.get_rank() == 0 else None
    cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    ocfg = FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, cfg, ocfg):
        if dist.get_rank() == 0:
            model.load_state_dict(ckpt['model_state_dict'])
        if optimizer is not None and ckpt is not None and dist.get_rank() == 0:
            optim_sd = FSDP.optim_state_dict_to_load(
                model, optimizer, ckpt['optimizer_state_dict'])
            optimizer.load_state_dict(optim_sd)
    if dist.get_rank() == 0:
        logger.info(f"  Resumed from {ckpt_path}")
        return ckpt.get('epoch', 0), ckpt.get('step', 0)
    return 0, 0


# =============================================================================
# Training loop
# =============================================================================
def train(args):
    rank, local_rank, world_size = setup_distributed()
    logger = setup_logging(rank)
    device = torch.device(f'cuda:{local_rank}')

    if rank == 0:
        logger.info("=" * 70)
        logger.info("CD-Transformer Distributed Training")
        logger.info(f"  Hardware: {world_size}x H800 | ZeRO-2 FSDP | BF16 autocast")
        logger.info("=" * 70)

    # --- Config (copy so we never mutate the shared template) ---
    from dataclasses import replace
    config = replace(
        CONFIGS[args.config],
        use_fp8=args.use_fp8,
        gradient_checkpointing=args.grad_checkpoint,
        fisher_lambda=args.fisher_lambda,
        fisher_mode=args.fisher_mode,
        fisher_agg=args.fisher_agg,
        fisher_p=args.fisher_p,
        cd_impl=("dense" if args.dense else args.cd_impl),
        **({'max_seq_len': args.seq_len} if args.seq_len else {}),
    )

    model = CDTransformer(config).to(device)
    if rank == 0:
        s = model.get_param_stats()
        logger.info(f"\nModel: CD-Transformer-{args.config}")
        logger.info(f"  Total params:     {s['total_params']:>13,}")
        logger.info(f"  CD params:        {s['cd_params']:>13,}")
        logger.info(f"  Dense equivalent: {s['dense_equivalent']:>13,}")
        logger.info(f"  CD compression:   {s['cd_compression']:>11.1f}x")
        logger.info(f"  MoE sparsity:     {config.n_experts / config.n_active:>11.1f}x")

    model = wrap_model_fsdp(model, local_rank)
    if rank == 0:
        logger.info("Model wrapped with FSDP (ZeRO-2 SHARD_GRAD_OP)")

    # --- Data ---
    dataset = TokenDataset(
        data_path=args.data_path, seq_len=config.max_seq_len,
        vocab_size=config.vocab_size, synthetic=args.synthetic,
        synthetic_size=args.synthetic_size,
    )
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, sampler=sampler,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
        persistent_workers=args.num_workers > 0,
    )

    steps_per_epoch = max(1, len(dataloader) // args.grad_accum)
    total_steps = steps_per_epoch * args.epochs
    if rank == 0:
        logger.info(f"\nDataset: {len(dataset):,} sequences, {len(dataloader):,} batches/epoch")
        logger.info(f"Training: {args.epochs} epochs, ~{total_steps:,} optimizer steps")
        logger.info(f"  Effective batch: {args.batch_size * world_size * args.grad_accum}")

    # --- Optimizer / schedule ---
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
        fused=torch.cuda.is_available(),
    )
    # warmup: explicit --warmup_steps, or a fraction of total steps (more robust
    # across dataset sizes — fixes the prior run where warmup>total capped LR).
    warmup_steps = args.warmup_steps
    if args.warmup_frac and args.warmup_frac > 0:
        warmup_steps = max(1, int(args.warmup_frac * total_steps))
    warmup_steps = min(warmup_steps, max(1, total_steps - 1))
    if rank == 0:
        logger.info(f"  Warmup: {warmup_steps:,} steps "
                    f"({100*warmup_steps/max(1,total_steps):.1f}% of total)")
    scheduler = CosineWarmupScheduler(optimizer, warmup_steps, total_steps, 0.1)
    hessian_monitor = HessianMonitor(model, log_interval=args.log_interval,
                                    hessian_interval=args.hessian_interval)

    # --- Resume ---
    start_epoch, global_step = 0, 0
    if args.resume:
        start_epoch, global_step = load_checkpoint(model, optimizer, args.resume, logger)

    # BF16 needs no GradScaler (it shares FP32's exponent range).
    use_amp = args.use_amp
    best_loss = float('inf')
    train_history = []

    if rank == 0:
        logger.info("\n" + "=" * 70 + "\nStarting training...\n" + "=" * 70)

    for epoch in range(start_epoch, args.epochs):
        sampler.set_epoch(epoch)
        model.train()
        optimizer.zero_grad(set_to_none=True)

        epoch_loss, epoch_tokens, epoch_start = 0.0, 0, time.time()
        accum_loss, step_times = 0.0, []
        accum_ce, accum_mtp, accum_fisher = 0.0, 0.0, 0.0
        step_start = time.time()

        for batch_idx, (input_ids, labels) in enumerate(dataloader):
            input_ids = input_ids.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            amp_ctx = (torch.amp.autocast('cuda', dtype=torch.bfloat16)
                       if use_amp else nullcontext())
            with amp_ctx:
                outputs = model(input_ids, labels=labels)
                loss = outputs['loss'] / args.grad_accum

            loss.backward()
            accum_loss += loss.item()
            # accumulate the *components*, averaged over the grad_accum window, so
            # the log can report the bare cross-entropy DIRECTLY (no decoding from
            # the total loss, which was the source of the CE/perplexity ambiguity).
            accum_ce += outputs['ce_loss'].item() / args.grad_accum
            if 'mtp_loss' in outputs:
                accum_mtp += outputs['mtp_loss'].item() / args.grad_accum
            if 'fisher_loss' in outputs:
                accum_fisher += float(outputs['fisher_loss']) / args.grad_accum

            if (batch_idx + 1) % args.grad_accum == 0:
                grad_norm = model.clip_grad_norm_(args.grad_clip)  # FSDP-aware clip
                optimizer.step()
                scheduler.step(global_step)
                optimizer.zero_grad(set_to_none=True)
                update_router_biases(model)        # aux-loss-free balancing
                global_step += 1

                n_tokens = input_ids.numel()
                epoch_tokens += n_tokens
                epoch_loss += accum_loss
                step_time = time.time() - step_start
                step_times.append(step_time)

                if global_step % args.log_interval == 0:
                    if rank == 0:
                        tok_s = n_tokens * world_size * args.grad_accum / max(step_time, 1e-6)
                        gpu_mem = torch.cuda.max_memory_allocated(device) / 1e9
                        ppl = math.exp(min(accum_ce, 20))
                        msg = (f"Step {global_step:>6d} | Loss: {accum_loss:.4f} | "
                               f"CE: {accum_ce:.4f} | PPL: {ppl:.1f} | "
                               f"LR: {scheduler.get_lr()[0]:.2e} | "
                               f"GradNorm: {float(grad_norm):.3f} | "
                               f"Tok/s: {tok_s:,.0f} | GPU: {gpu_mem:.1f}GB")
                        if accum_mtp:
                            msg += f" | MTP: {accum_mtp:.4f}"
                        if accum_fisher:
                            msg += f" | Fisher: {accum_fisher:.4g}"
                        logger.info(msg)
                        train_history.append({
                            'step': global_step, 'loss': accum_loss, 'ce': accum_ce,
                            'perplexity': ppl, 'lr': scheduler.get_lr()[0],
                            'tokens_per_sec': tok_s, 'gpu_mem_gb': gpu_mem,
                        })
                    # ALL ranks must enter this: it runs an FSDP summon_full_params
                    # collective to materialize the flat-sharded CDLinear weights.
                    # (Only rank 0 emits the log line.) Calling it under `rank == 0`
                    # would deadlock the other ranks.
                    hessian_monitor.log(global_step, logger)

                accum_loss = 0.0
                accum_ce, accum_mtp, accum_fisher = 0.0, 0.0, 0.0
                step_start = time.time()

        avg_loss = epoch_loss / max(1, len(step_times))
        if rank == 0:
            dt = time.time() - epoch_start
            logger.info(f"\n{'='*70}\nEpoch {epoch+1}/{args.epochs} | avg loss {avg_loss:.4f} | "
                        f"{dt:.1f}s | {epoch_tokens*world_size/max(dt,1e-6):,.0f} tok/s")

        if avg_loss < best_loss:
            best_loss = avg_loss
        save_checkpoint(model, optimizer, scheduler, epoch, global_step,
                        avg_loss, args.save_dir, config, logger)

    if rank == 0:
        logger.info("\n" + "=" * 70 + f"\nTRAINING COMPLETE | best loss {best_loss:.4f} | "
                    f"{global_step:,} steps")
        hist_path = Path(args.save_dir) / 'training_history.json'
        hist_path.parent.mkdir(parents=True, exist_ok=True)
        with open(hist_path, 'w') as f:
            json.dump(train_history, f, indent=2)
        logger.info(f"  History: {hist_path}")

    cleanup_distributed()


# =============================================================================
# CLI
# =============================================================================
def parse_args():
    p = argparse.ArgumentParser(
        description='CD-Transformer Distributed Training (H800 8-GPU)',
        formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('--config', type=str, default='small',
                   choices=['small', 'medium', 'large'])
    p.add_argument('--seq_len', type=int, default=None)

    p.add_argument('--data_path', type=str, default='./data/train.bin')
    p.add_argument('--synthetic', action='store_true')
    p.add_argument('--synthetic_size', type=int, default=1_000_000)
    p.add_argument('--num_workers', type=int, default=4)

    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--grad_accum', type=int, default=4)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--warmup_steps', type=int, default=2000)
    p.add_argument('--warmup_frac', type=float, default=None,
                   help="set warmup as a fraction of total steps (overrides --warmup_steps)")
    p.add_argument('--dense', action='store_true',
                   help="train the non-circulant DENSE baseline (cd_impl=dense), "
                        "same architecture/schedule for a fair comparison")
    p.add_argument('--cd_impl', choices=['matmul', 'fft', 'dense'], default='matmul')
    p.add_argument('--fisher_agg', choices=['mean', 'pnorm', 'max'], default='mean',
                   help="conditioning aggregation over blocks (pnorm/max target worst blocks)")
    p.add_argument('--fisher_p', type=float, default=4.0,
                   help="p for --fisher_agg pnorm")
    p.add_argument('--weight_decay', type=float, default=0.1)
    p.add_argument('--grad_clip', type=float, default=1.0)

    p.add_argument('--fisher_lambda', type=float, default=1e-5)
    p.add_argument('--fisher_mode', choices=['energy', 'conditioning'],
                   default='energy',
                   help="'energy' = L2-like (lambda~1e-8); "
                        "'conditioning' = flatten spectrum, drives kappa->1 (lambda~1e-2)")

    # store_true flags default False; pass --use_fp8/--use_amp/--grad_checkpoint
    # to enable. Defaults below keep them ON unless --no_* is added later.
    p.add_argument('--use_fp8', action='store_true', default=True)
    p.add_argument('--use_amp', action='store_true', default=True)
    p.add_argument('--grad_checkpoint', action='store_true', default=True)

    p.add_argument('--log_interval', type=int, default=50)
    p.add_argument('--hessian_interval', type=int, default=500,
                   help='steps between (expensive) Hessian-kappa summon_full_params samples')
    p.add_argument('--save_dir', type=str, default='./checkpoints')
    p.add_argument('--resume', type=str, default=None)
    return p.parse_args()


if __name__ == '__main__':
    train(parse_args())
