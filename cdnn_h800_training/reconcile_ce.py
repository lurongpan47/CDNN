#!/usr/bin/env python3
"""reconcile_ce.py — localize the train-log vs clean-eval cross-entropy gap.

The training log decoded a cross-entropy of ~6.0-6.2 while a clean eval of the
same checkpoint gave ~7.4-7.5. This harness computes every component on ONE
batch so the discrepancy can be attributed to a specific cause:

  1. decode check   : does (total_loss - mtp_weight*mtp - fisher) == bare CE?
                      (catches a wrong MTP-weight decode constant)
  2. train vs eval  : bare CE in model.train() vs model.eval() on the SAME batch
                      (isolates Shannon dropout + any training-only forward path)
  3. data identity  : are train.bin and the trained data actually the same tokens?
                      (--bin_b compares two bins by length + checksum + edge ids)

Run on the cluster:
  python reconcile_ce.py --checkpoint <ckpt.pt> --bin data/train.bin --seq_len 2048
  python reconcile_ce.py --bin data/train.bin --bin_b valdata/val.bin        # data check only
"""
import argparse, re, hashlib
import numpy as np
import torch
import torch.nn.functional as F


def _torch_load(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_model(ckpt_path, device):
    from cd_model import create_model
    ckpt = _torch_load(ckpt_path)
    cfg = ckpt.get("config", {})
    model = create_model("small", **cfg)
    sd = ckpt.get("model_state_dict", ckpt)
    clean = {re.sub(r"^(_fsdp_wrapped_module\.|module\.|_orig_mod\.)", "", k): v
             for k, v in sd.items()}
    model.load_state_dict(clean, strict=False)
    return model.to(device), cfg


def get_batch(bin_path, seq_len, bs, device, offset=0):
    toks = np.memmap(bin_path, dtype=np.uint16, mode="r")
    need = bs * seq_len + 1
    chunk = np.asarray(toks[offset:offset + need], dtype=np.int64)
    x = torch.from_numpy(chunk[:bs * seq_len].reshape(bs, seq_len)).to(device)
    y = torch.from_numpy(chunk[1:bs * seq_len + 1].reshape(bs, seq_len)).to(device)
    return x, y


def bare_ce(model, x, y):
    """Recompute the next-token CE directly from logits (no aux terms)."""
    out = model(x, labels=y)
    if "ce_loss" in out:
        return float(out["ce_loss"]), out
    logits = out["logits"]
    ce = F.cross_entropy(logits[:, :-1].reshape(-1, logits.size(-1)),
                         y[:, 1:].reshape(-1))
    return float(ce), out


def data_fingerprint(bin_path):
    toks = np.memmap(bin_path, dtype=np.uint16, mode="r")
    n = len(toks)
    head = np.asarray(toks[:10]).tolist()
    tail = np.asarray(toks[-10:]).tolist()
    h = hashlib.sha1(np.asarray(toks[:1_000_000]).tobytes()).hexdigest()[:12]
    return n, head, tail, h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint")
    ap.add_argument("--bin")
    ap.add_argument("--bin_b", help="second bin to compare for data identity")
    ap.add_argument("--seq_len", type=int, default=2048)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--mtp_weight", type=float, default=0.3)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    if args.bin:
        n, head, tail, h = data_fingerprint(args.bin)
        print(f"[data] {args.bin}: {n:,} tokens | head{head} tail{tail} | sha1(1M)={h}")
    if args.bin_b:
        n2, head2, tail2, h2 = data_fingerprint(args.bin_b)
        print(f"[data] {args.bin_b}: {n2:,} tokens | head{head2} tail{tail2} | sha1(1M)={h2}")
        print(f"[data] first-1M-token blocks identical: {h == h2}  "
              f"(if True, the two bins are NOT disjoint -> not a valid held-out split)")
        if not args.checkpoint:
            return

    if not args.checkpoint:
        print("no --checkpoint; data check only."); return

    dev = torch.device(args.device)
    model, cfg = load_model(args.checkpoint, dev)
    mtp_w = cfg.get("mtp_weight", args.mtp_weight)
    x, y = get_batch(args.bin, args.seq_len, args.batch_size, dev)

    # (2) train vs eval bare CE on the same batch
    model.eval()
    with torch.no_grad():
        ce_eval, out_eval = bare_ce(model, x, y)
    model.train()
    with torch.no_grad():
        ce_train, out_train = bare_ce(model, x, y)

    print("\n=== CE reconciliation (same batch) ===")
    print(f"  eval-mode  bare CE : {ce_eval:.4f}   (perplexity {np.exp(ce_eval):7.1f})")
    print(f"  train-mode bare CE : {ce_train:.4f}   (perplexity {np.exp(ce_train):7.1f})")
    print(f"  train - eval       : {ce_train - ce_eval:+.4f}  "
          f"<- Shannon dropout + training-only forward paths")

    # (1) decode check in train mode
    total = float(out_train["loss"])
    mtp = float(out_train.get("mtp_loss", 0.0))
    fis = float(out_train.get("fisher_loss", 0.0))
    decoded = total - mtp_w * mtp - fis
    print("\n=== decode check (train-mode total loss) ===")
    print(f"  total={total:.4f}  mtp={mtp:.4f} (w={mtp_w})  fisher={fis:.4g}")
    print(f"  decoded CE = total - w*mtp - fisher = {decoded:.4f}")
    print(f"  bare CE (train)                      = {ce_train:.4f}")
    print(f"  decode error                         = {decoded - ce_train:+.4f}  "
          f"(should be ~0; nonzero => wrong decode constant)")

    print("\n=== verdict ===")
    if abs(ce_train - ce_eval) > 0.3:
        print("  -> The gap is a TRAIN-vs-EVAL FORWARD difference (dropout/MoE/MTP path),")
        print("     not a decode error. The honest number is the eval-mode CE above.")
    elif abs(decoded - ce_train) > 0.1:
        print("  -> The gap is a DECODE error (MTP-weight constant). Use the logged")
        print("     'CE:' field from the patched trainer instead of decoding.")
    else:
        print("  -> train≈eval and decode≈bare: the logged/eval CE are consistent;")
        print("     any remaining gap is dataset-average vs single-minibatch sampling.")


if __name__ == "__main__":
    main()
