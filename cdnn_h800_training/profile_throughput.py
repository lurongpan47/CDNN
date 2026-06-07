#!/usr/bin/env python3
"""
profile_throughput.py — measure CD-Transformer GPU utilization and find the
settings that maximize it. Run on ONE H800 (no torchrun needed).

It times forward+backward, reports peak memory, tokens/s, and estimated MFU,
sweeping:
  * CDLinear impl:   matmul (tensor-core, default)  vs  fft (old path)
  * gradient checkpointing: on vs off
  * (optionally) a batch-size sweep to find the largest that fits

Example:
  python profile_throughput.py --config small --seq_len 2048 --batch_size 16
  python profile_throughput.py --config small --seq_len 2048 --sweep_batch 8,16,24,32
  python profile_throughput.py --config small --compare_impl       # matmul vs fft

The headline number is MFU (model FLOPs utilization). 0.2% means the GPUs are
idle; the goal is to push it up by (a) the matmul impl, (b) turning OFF grad
checkpointing once memory allows, and (c) raising the batch size.
"""
from __future__ import annotations
import argparse
import time

import torch
import torch.nn as nn


def set_cd_impl(model, impl):
    from cd_layers import CDLinear
    n = 0
    for m in model.modules():
        if isinstance(m, CDLinear):
            m.impl = impl
            n += 1
    return n


def set_checkpoint(model, flag):
    if hasattr(model, "config") and hasattr(model.config, "gradient_checkpointing"):
        model.config.gradient_checkpointing = flag
    for m in model.modules():
        if hasattr(m, "gradient_checkpointing"):
            m.gradient_checkpointing = flag


def est_train_flops_per_token(model, seq_len):
    """Rough training FLOPs/token = 3 x forward. Forward linear FLOPs counted
    from the dense-equivalent of each CDLinear (active MoE-weighted), plus
    attention core. Mirrors benchmark_report's cost model."""
    from cd_layers import CDLinear
    import re
    cfg = model.config
    n_exp = getattr(cfg, "n_experts", 1) or 1
    n_act = getattr(cfg, "n_active", n_exp) or n_exp
    exp_rx = re.compile(r"(gate_proj|up_proj|down_proj|experts?)\.(\d+)")
    lin = 0.0
    for name, m in model.named_modules():
        if isinstance(m, CDLinear):
            factor = (n_act / n_exp) if exp_rx.search(name.lower()) else 1.0
            lin += factor * 2.0 * m.n_in * m.n_out          # dense-equivalent MACs
    dim = getattr(cfg, "dim", 0); n_layers = getattr(cfg, "n_layers", 0)
    attn = 4.0 * seq_len * dim * n_layers
    return 3.0 * (lin + attn)


def run(model, ids, steps, amp=True):
    dev = next(model.parameters()).device
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    torch.cuda.reset_peak_memory_stats(dev)
    # warmup
    for _ in range(2):
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            out = model(ids, labels=ids)
        out["loss"].backward(); opt.step()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            out = model(ids, labels=ids)
        out["loss"].backward(); opt.step()
    torch.cuda.synchronize()
    dt = (time.time() - t0) / steps
    peak = torch.cuda.max_memory_allocated(dev) / 1e9
    return dt, peak


def try_run(build, impl, ck, bs, args, dev, backoff=True):
    """Run one config; on OOM optionally halve the batch and retry. Returns
    (dt, peak, bs_used, model) or (None, None, None, None)."""
    while bs >= 1:
        m = build(); set_cd_impl(m, impl); set_checkpoint(m, ck)
        ids = torch.randint(0, m.config.vocab_size, (bs, args.seq_len), device=dev)
        try:
            dt, peak = run(m, ids, args.steps)
            return dt, peak, bs, m
        except RuntimeError as e:
            if "out of memory" in str(e).lower() and backoff and bs > 1:
                del m; torch.cuda.empty_cache()
                bs = bs // 2
                continue
            del m; torch.cuda.empty_cache()
            return None, None, None, None
    return None, None, None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="small")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--seq_len", type=int, default=2048)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--gpu_tflops", type=float, default=989.0)
    ap.add_argument("--compare_impl", action="store_true",
                    help="benchmark matmul vs fft")
    ap.add_argument("--sweep_batch", default=None,
                    help="comma list of batch sizes to try, e.g. 8,16,24,32")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("needs a CUDA GPU")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    dev = torch.device("cuda")
    from cd_model import create_model

    def build():
        return create_model(args.config).to(dev)

    def fmt(dt, peak, bs, model):
        tok = bs * args.seq_len / dt
        fpt = est_train_flops_per_token(model, args.seq_len)
        mfu = tok * fpt / (args.gpu_tflops * 1e12)
        return (f"{dt*1e3:8.0f} ms/step | {tok:9,.0f} tok/s | "
                f"peak {peak:5.1f} GB | MFU {mfu*100:5.1f}% (1 GPU)")

    print(f"Device: {torch.cuda.get_device_name(0)} | config={args.config} "
          f"seq={args.seq_len}")
    print("(single GPU, NO FSDP sharding — the real 8-GPU ZeRO-2 run has ~1/8 the "
          "optimizer-state memory per GPU, so it fits much larger batches)\n")

    if args.compare_impl:
        print("=== impl comparison (grad-checkpoint ON, fair memory) ===")
        for impl in ("fft", "matmul"):
            dt, peak, bs, m = try_run(build, impl, True, args.batch_size, args, dev)
            if dt is None:
                print(f"  impl={impl:6s}: OOM even at batch 1")
            else:
                tag = "" if bs == args.batch_size else f" [auto-reduced to bs={bs}]"
                print(f"  impl={impl:6s}: {fmt(dt, peak, bs, m)}{tag}")
                del m
            torch.cuda.empty_cache()
        print()

    if args.sweep_batch:
        print("=== batch sweep (impl=matmul) ===")
        for ck in (True, False):
            print(f"  grad_checkpoint={ck}:")
            for bs in [int(x) for x in args.sweep_batch.split(",")]:
                dt, peak, bsu, m = try_run(build, "matmul", ck, bs, args, dev, backoff=False)
                if dt is None:
                    print(f"    bs={bs:3d}: OOM"); torch.cuda.empty_cache(); break
                print(f"    bs={bs:3d}: {fmt(dt, peak, bs, m)}")
                del m; torch.cuda.empty_cache()
        print()

    if not args.compare_impl and not args.sweep_batch:
        print("=== single config (impl=matmul) ===")
        for ck in (True, False):
            dt, peak, bs, m = try_run(build, "matmul", ck, args.batch_size, args, dev)
            if dt is None:
                print(f"  grad_checkpoint={str(ck):5s}: OOM even at batch 1")
            else:
                tag = "" if bs == args.batch_size else f" [auto-reduced to bs={bs}]"
                print(f"  grad_checkpoint={str(ck):5s}: {fmt(dt, peak, bs, m)}{tag}")
                del m
            torch.cuda.empty_cache()

    print("\nNotes:\n"
          "  * MFU shown is per-GPU; multiply tok/s by #GPUs for the cluster rate.\n"
          "  * If matmul fits with grad_checkpoint OFF, that's usually fastest.\n"
          "  * Raise batch until peak memory approaches ~75 GB, then stop.\n"
          "  * To make matmul the trained default, it already is (CDLinear impl='matmul').")


if __name__ == "__main__":
    main()
