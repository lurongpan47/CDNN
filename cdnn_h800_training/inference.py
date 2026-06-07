#!/usr/bin/env python3
"""
inference.py — run, generate from, and test a trained CD-Transformer.

The model is a decoder-only causal LM. forward(input_ids) -> dict with:
    logits         (B, T, vocab)  next-token scores at every position
    hidden_states  (B, T, dim)
(loss / mtp_loss / fisher_loss appear only when labels are passed in training.)
Generation = take logits[:, -1], turn into probabilities, pick a token, append,
repeat. The MTP heads are inactive at eval, so only the main head is used.

Modes
-----
  # 0. architecture sanity — needs NO checkpoint and NO tokenizer
  python inference.py --selftest

  # 1. generate text from a prompt (needs GPT-2 tokenizer; see --tokenizer)
  python inference.py --checkpoint checkpoints/.../checkpoint_latest.pt \
                      --prompt "The discovery of" --max_new_tokens 80 --top_p 0.9

  # 2. interactive REPL
  python inference.py --checkpoint <ckpt> --interactive

  # 3. generate in raw token-id space (no tokenizer needed)
  python inference.py --checkpoint <ckpt> --ids "464,3666,318" --max_new_tokens 40

  # 4. measure quality: perplexity on a tokenized .bin
  python inference.py --checkpoint <ckpt> --eval_bin data/val.bin

Tokenizer note: HuggingFace may be blocked on your box. Either
  export HF_ENDPOINT=https://hf-mirror.com         # then --tokenizer gpt2
or download gpt2 tokenizer files once and pass  --tokenizer /path/to/gpt2dir.
Text modes degrade to id-mode automatically if no tokenizer is available.
"""
from __future__ import annotations
import argparse
import math
import re
import sys

import numpy as np
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# model loading
# ---------------------------------------------------------------------------
def _torch_load(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_model(ckpt_path, device):
    """Rebuild the exact architecture from the checkpoint's saved config and
    load weights, tolerating FSDP/compile key prefixes and tied embeddings."""
    from cd_model import create_model
    ckpt = _torch_load(ckpt_path)
    cfg = ckpt.get("config", {})
    model = create_model("small", **cfg)            # cfg overrides every field
    sd = ckpt.get("model_state_dict", ckpt)
    clean = {re.sub(r"^(_fsdp_wrapped_module\.|module\.|_orig_mod\.)", "", k): v
             for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(clean, strict=False)
    # tied weight may be absent in one of the two keys — re-tie to be safe
    if hasattr(model, "output") and hasattr(model, "embed"):
        model.output.weight = model.embed.weight
    model.eval().to(device)
    info = dict(step=ckpt.get("step"), loss=ckpt.get("loss"),
                vocab=getattr(model.config, "vocab_size", None),
                max_seq_len=getattr(model.config, "max_seq_len", None),
                n_params=sum(p.numel() for p in model.parameters()),
                missing=len(missing), unexpected=len(unexpected))
    return model, info


def build_selftest_model(device):
    from cd_model import create_model
    m = create_model("small", n_layers=2, dim=128, n_heads=4, n_experts=4,
                     n_active=2, max_seq_len=256, gradient_checkpointing=False,
                     use_mtp=False, fisher_lambda=0.0)
    return m.eval().to(device)


# ---------------------------------------------------------------------------
# tokenizer (optional)
# ---------------------------------------------------------------------------
def load_tokenizer(name_or_path):
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(name_or_path)
        if tok.eos_token_id is None:
            tok.eos_token = "<|endoftext|>"
        return tok
    except Exception as exc:
        print(f"[warn] tokenizer '{name_or_path}' unavailable ({exc}); "
              f"falling back to raw token-id mode.", file=sys.stderr)
        return None


# ---------------------------------------------------------------------------
# sampling
# ---------------------------------------------------------------------------
def _filter_logits(logits, top_k, top_p):
    """Apply top-k then nucleus (top-p) filtering to a (B, V) logit tensor."""
    if top_k and top_k > 0:
        k = min(top_k, logits.size(-1))
        kth = torch.topk(logits, k, dim=-1).values[:, -1, None]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    if top_p and 0 < top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
        cum = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        remove = cum > top_p
        remove[:, 1:] = remove[:, :-1].clone()   # keep the first token over p
        remove[:, 0] = False
        idx_remove = remove.scatter(-1, sorted_idx, remove)
        logits = logits.masked_fill(idx_remove, float("-inf"))
    return logits


@torch.no_grad()
def generate(model, input_ids, device, max_new_tokens=64, temperature=0.8,
             top_k=50, top_p=0.95, eos_id=None, greedy=False):
    """Autoregressive generation (no KV cache — recomputes the context each
    step; correct and simple, fine for testing; add a cache later for speed)."""
    model.eval()
    ids = input_ids.to(device)
    max_ctx = getattr(model.config, "max_seq_len", 4096)
    use_amp = device.type == "cuda"
    for _ in range(max_new_tokens):
        ctx = ids[:, -max_ctx:]
        amp = (torch.autocast("cuda", dtype=torch.bfloat16) if use_amp
               else torch.autocast("cpu", dtype=torch.bfloat16, enabled=False))
        with amp:
            logits = model(ctx)["logits"][:, -1, :].float()   # (B, V)
        if greedy or temperature <= 0:
            nxt = logits.argmax(-1, keepdim=True)
        else:
            logits = _filter_logits(logits / temperature, top_k, top_p)
            probs = F.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, num_samples=1)
        ids = torch.cat([ids, nxt], dim=1)
        if eos_id is not None and bool((nxt == eos_id).all()):
            break
    return ids


# ---------------------------------------------------------------------------
# perplexity evaluation
# ---------------------------------------------------------------------------
@torch.no_grad()
def eval_perplexity(model, bin_path, device, seq_len=1024, max_batches=200,
                    batch_size=8):
    toks = np.memmap(bin_path, dtype=np.uint16, mode="r")
    n_seq = (len(toks) - 1) // seq_len
    n_seq = min(n_seq, max_batches * batch_size)
    if n_seq < 1:
        return None
    use_amp = device.type == "cuda"
    total_loss, total_tok = 0.0, 0
    for b in range(0, n_seq, batch_size):
        rows = []
        for i in range(b, min(b + batch_size, n_seq)):
            s = i * seq_len
            rows.append(np.asarray(toks[s:s + seq_len + 1], dtype=np.int64))
        arr = np.stack(rows)
        ids = torch.from_numpy(arr[:, :-1]).to(device)
        lbl = torch.from_numpy(arr[:, 1:]).contiguous().to(device)
        amp = (torch.autocast("cuda", dtype=torch.bfloat16) if use_amp
               else torch.autocast("cpu", enabled=False))
        with amp:
            logits = model(ids)["logits"].float()
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                               lbl.reshape(-1), ignore_index=-100,
                               reduction="sum")
        total_loss += float(loss)
        total_tok += lbl.numel()
    nll = total_loss / max(total_tok, 1)
    return dict(loss=nll, perplexity=math.exp(min(nll, 20)), tokens=total_tok)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="CD-Transformer inference / test")
    ap.add_argument("--checkpoint", help="checkpoint .pt")
    ap.add_argument("--tokenizer", default="gpt2", help="HF name or local dir")
    ap.add_argument("--prompt", help="text prompt to continue")
    ap.add_argument("--ids", help="comma-separated token ids (no tokenizer needed)")
    ap.add_argument("--interactive", action="store_true")
    ap.add_argument("--eval_bin", help="tokenized .bin for perplexity")
    ap.add_argument("--selftest", action="store_true",
                    help="architecture sanity check (no checkpoint/tokenizer)")
    ap.add_argument("--max_new_tokens", type=int, default=64)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top_k", type=int, default=50)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--greedy", action="store_true")
    ap.add_argument("--seq_len", type=int, default=1024)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    device = torch.device(args.device)

    # ----- selftest: no checkpoint, no tokenizer -----------------------------
    if args.selftest:
        print(f"[selftest] building tiny model on {device} ...")
        model = build_selftest_model(device)
        ids = torch.randint(0, model.config.vocab_size, (2, 16), device=device)
        with torch.no_grad():
            out = model(ids)
        logits = out["logits"]
        print(f"  input_ids        : {tuple(ids.shape)}")
        print(f"  logits           : {tuple(logits.shape)}  "
              f"(= batch, seq_len, vocab={model.config.vocab_size})")
        print(f"  hidden_states    : {tuple(out['hidden_states'].shape)}")
        nxt = logits[:, -1, :].softmax(-1)
        print(f"  next-token probs : sum={float(nxt.sum(-1).mean()):.3f} (≈1.0), "
              f"argmax ids={logits[:, -1, :].argmax(-1).tolist()}")
        gen = generate(model, ids[:, :4], device, max_new_tokens=8, greedy=True)
        print(f"  greedy gen shape : {tuple(gen.shape)} (started from 4 tokens, +8)")
        print("[selftest] OK — forward, output shapes, and generation all work.")
        return

    if not args.checkpoint:
        sys.exit("need --checkpoint (or use --selftest)")

    print(f"[load] {args.checkpoint} on {device} ...")
    model, info = load_model(args.checkpoint, device)
    print(f"  params={info['n_params']:,} | vocab={info['vocab']} | "
          f"ctx={info['max_seq_len']} | trained step={info['step']} "
          f"| ckpt loss={info['loss']}")
    if info["missing"] or info["unexpected"]:
        print(f"  [note] state_dict load: {info['missing']} missing, "
              f"{info['unexpected']} unexpected keys (ok if only MTP/buffers).")

    # ----- perplexity eval ---------------------------------------------------
    if args.eval_bin:
        print(f"[eval] perplexity on {args.eval_bin} ...")
        res = eval_perplexity(model, args.eval_bin, device, seq_len=args.seq_len)
        if res is None:
            print("  eval set too small.")
        else:
            print(f"  loss={res['loss']:.4f}  perplexity={res['perplexity']:.2f}  "
                  f"over {res['tokens']:,} tokens")
        if not (args.prompt or args.ids or args.interactive):
            return

    tok = None
    if args.prompt or args.interactive:
        tok = load_tokenizer(args.tokenizer)

    def run_once(text=None, id_list=None):
        if id_list is not None:
            start = torch.tensor([id_list], dtype=torch.long)
        elif tok is not None:
            start = torch.tensor([tok.encode(text)], dtype=torch.long)
        else:
            print("  no tokenizer; pass --ids instead of --prompt.")
            return
        eos = tok.eos_token_id if tok is not None else None
        out_ids = generate(model, start, device,
                           max_new_tokens=args.max_new_tokens,
                           temperature=args.temperature, top_k=args.top_k,
                           top_p=args.top_p, eos_id=eos, greedy=args.greedy)[0].tolist()
        if tok is not None:
            print("  " + tok.decode(out_ids))
        else:
            print("  ids:", out_ids)

    if args.ids:
        id_list = [int(x) for x in args.ids.split(",") if x.strip() != ""]
        print("[gen] from token ids ...")
        run_once(id_list=id_list)
    elif args.prompt:
        print("[gen] continuation ...")
        run_once(text=args.prompt)

    if args.interactive:
        if tok is None:
            sys.exit("interactive mode needs a tokenizer (set HF_ENDPOINT or --tokenizer dir)")
        print("[interactive] type a prompt, Ctrl-C to quit.")
        try:
            while True:
                text = input("\n>>> ")
                if text.strip():
                    run_once(text=text)
        except (KeyboardInterrupt, EOFError):
            print("\nbye.")


if __name__ == "__main__":
    main()
