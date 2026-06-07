# CD-Transformer — Usage Guide

A decoder-only language model that combines **Communication-Dynamics block-circulant
layers (CDLinear)** with a **DeepSeek-V3-style backbone** (MLA-style attention,
mixture-of-experts, multi-token prediction), trained with FSDP on 8×H800.

This guide covers the three workflows: **data → training → inference**, plus the
**benchmark report** that compares the model against a same-architecture dense
(DeepSeek-V3-style) baseline. Architecture/theory details are in `ARCHITECTURE.md`.

```
cd_layers.py          CDLinear, attention, MoE, Fisher regularizer
cd_model.py           CDTransformer (config presets: small / medium / large)
train_distributed.py  FSDP training loop, checkpointing, kappa monitor
prepare_data.py       tokenize / pack corpora into train.bin / val.bin
launch_h800.sh        8xH800 torchrun launcher
inference.py          generation, perplexity eval, self-test
benchmark_report.py   parse logs + checkpoint -> HTML/MD/JSON report
```

---

## 0. Environment

```bash
conda activate myenv          # PyTorch 2.x with CUDA, on the H800 node
# China network: HuggingFace is blocked. Use the mirror for tokenizer/datasets:
export HF_ENDPOINT=https://hf-mirror.com
```

Key facts baked into the code:
- **Vocabulary is 50304** (GPT-2's 50257 padded to a multiple of 64). Your data
  must be tokenized with a tokenizer whose vocab <= 50304, or the embedding lookup
  hits a CUDA device-side assert.
- Token ids are stored as **uint16**, so any vocab > 65535 is rejected (guarded).
- Compute runs in **BF16**; `--use_fp8` is scaffolding only (see section 6).

---

## 1. Prepare data

`prepare_data.py` writes `train.bin` / `val.bin` (raw uint16 token ids).

```bash
# A) synthetic - instant, for smoke tests (ids are always in-range)
python prepare_data.py --source synthetic --output ./data \
    --vocab_size 32000 --max_tokens 2000000

# B) HuggingFace dataset through the mirror
export HF_ENDPOINT=https://hf-mirror.com
python prepare_data.py --source hf --dataset openwebtext --tokenizer gpt2 \
    --output ./data --max_tokens 1000000000

# C) your own text files (fully offline)
python prepare_data.py --source local --input_dir /path/to/corpus \
    --tokenizer gpt2 --output ./data
```

Notes
- `--source huggingface`, `--output_dir`, `--input_path` are accepted as aliases.
- The tokenizer (`gpt2`) also downloads from HuggingFace; the mirror covers it, or
  pre-download once and pass a local dir to `--tokenizer`.
- A real model needs a **real, large** corpus - hundreds of millions to billions of
  tokens. The 2M synthetic set is only for verifying the pipeline.

---

## 2. Train

### Quick launch (8xH800, single node)

```bash
# run detached so an SSH drop can't kill it
tmux new -s cdtrain
./launch_h800.sh small 2>&1 | tee train.log      # configs: small | medium | large
#   detach: Ctrl-b then d   |   reattach: tmux attach -t cdtrain
```

If your cluster uses Slurm, submit through `sbatch` instead of running `torchrun`
on a login node.

### What the launcher sets (per config, in `launch_h800.sh`)

| config | dim | layers | experts (active) | batch | grad_accum | seq | LR | warmup |
|---|---|---|---|---|---|---|---|---|
| small  | 1024 | 12 | 16 (4) | 16 | 2 | 2048 | 6e-4 | 1000 |
| medium | 2048 | 24 | 32 (6) | 8  | 4 | 4096 | 3e-4 | 2000 |
| large  | 4096 | 32 | 64 (8) | 4  | 8 | 4096 | 1.5e-4 | 4000 |

> **Set `warmup` below your total step count.** `total_steps ~=
> tokens_per_epoch / (batch*seq*grad_accum*world) * epochs`. If `warmup` exceeds
> that, the LR never ramps and the model barely learns. For short/smoke runs set
> `WARMUP=20` in the launcher.

### Manual `torchrun` (equivalent)

```bash
torchrun --standalone --nproc_per_node=8 train_distributed.py \
    --config small --data_path ./data \
    --batch_size 16 --grad_accum 2 --lr 6e-4 --warmup_steps 1000 \
    --epochs 20 --seq_len 2048 --save_dir ./checkpoints \
    --use_amp --grad_checkpoint --fisher_lambda 1e-8 --log_interval 50
```

### Resume

```bash
torchrun ... train_distributed.py ... --resume ./checkpoints/<run_dir>
```
Restores model + optimizer + scheduler + step (FSDP-correct sharded state).

### Reading the log

Each step logs the **total** loss, which is `CE + 0.3*MTP + Fisher`. Judge progress
by the **cross-entropy**, not the total - the report decodes it for you (section 4).
A fresh model logs total ~= 14 (CE ~= ln 50304 = 10.83); CE falling below ~10.8
means it is learning. `Hessian kappa` lines track layer conditioning (target: 1).

---

## 3. Inference

```bash
# architecture sanity - no checkpoint or tokenizer needed
python inference.py --selftest

# held-out quality (the number that matters): perplexity on val.bin
python inference.py --checkpoint checkpoints/<run>/checkpoint_latest.pt \
    --eval_bin data/val.bin

# generate text from a prompt (needs the GPT-2 tokenizer via the mirror)
export HF_ENDPOINT=https://hf-mirror.com
python inference.py --checkpoint <ckpt> --prompt "The discovery of" \
    --max_new_tokens 80 --top_p 0.9 --temperature 0.8
python inference.py --checkpoint <ckpt> --interactive

# generate in raw token-id space (no tokenizer)
python inference.py --checkpoint <ckpt> --ids "464,3666,318,257" --max_new_tokens 40 --greedy
```

The model emits `logits` of shape `(batch, seq_len, 50304)`; generation takes the
last position, applies temperature / top-k / top-p, samples, appends, repeats. The
script rebuilds the exact architecture from the checkpoint's saved config and strips
FSDP key prefixes automatically. Generation recomputes context each step (no KV
cache) - fine for testing; a cache can be added for fast serving.

---

## 4. Build the benchmark report

`benchmark_report.py` turns a training log (and optionally a checkpoint and a
validation set) into a self-contained report: **`report.html`** (charts embedded,
print-to-PDF if you want one), `report.md`, `metrics.json`, `metrics.csv`.

```bash
# full report: training + conditioning + cost-vs-dense + convergence + accuracy
python benchmark_report.py \
    --log train.log \
    --checkpoint checkpoints/<run>/checkpoint_latest.pt \
    --val_bin data/val.bin \
    --batch_size 16 --grad_accum 2 --world_size 8 --seq_len 2048 \
    --outdir report_out
```

What each section reports:

1. **Training** - total loss *and* decoded cross-entropy (CE), perplexity from CE,
   throughput, peak memory.
2. **Hessian conditioning** - kappa trend vs the Theorem-2 target of 1.
3. **Computational cost vs DeepSeek-V3-style dense baseline** - the baseline is the
   *same MLA+MoE+MTP architecture with dense projections* instead of block-circulant
   ones. Reports parameters and estimated FLOPs/token for both, and the
   **iso-parameter** view (how much more capacity CD packs per parameter). These are
   **analytical estimates from your config**, not DeepSeek-V3's published numbers.
4. **Convergence speed** - steps and tokens to reach CE thresholds, plus the
   end-of-run rate. Overlay a same-data dense baseline run with `--baseline_log` for
   a true head-to-head; without it, the CD curve is shown alone (no invented numbers).
5. **Inference accuracy** - held-out perplexity / bits-per-token from `--val_bin`.

Useful flags:
- `--baseline_log <log>` overlay a dense-baseline run's CE curve on convergence.
- `--tokens_per_step N` (or `--batch_size/--grad_accum/--world_size` with `--seq_len`)
  to get the tokens axis.
- `--mtp_weight 0.3` to decode CE (auto-read from the checkpoint config if present).

### How to get a real "vs DeepSeek-V3" comparison

The cost numbers (section 3) are computable from the architecture alone - no baseline
run needed. For an honest **convergence and accuracy** comparison, train the dense
baseline on the **same data and token budget**, then:

```bash
python benchmark_report.py --log cd.log --baseline_log dense.log \
    --checkpoint cd_ckpt.pt --val_bin data/val.bin --outdir report_out
```

Comparing to DeepSeek-V3's *published* scores would be apples-to-oranges (different
data, scale, tokenizer); the same-data dense baseline is the fair reference.

---

## 5. End-to-end example

```bash
export HF_ENDPOINT=https://hf-mirror.com
python prepare_data.py --source synthetic --output ./data --max_tokens 2000000
tmux new -s cdtrain
./launch_h800.sh small 2>&1 | tee train.log         # (Ctrl-b d to detach)
RUN=checkpoints/$(ls -t checkpoints | head -1)
python inference.py --checkpoint $RUN/checkpoint_latest.pt --eval_bin data/val.bin
python benchmark_report.py --log train.log --checkpoint $RUN/checkpoint_latest.pt \
    --val_bin data/val.bin --batch_size 16 --grad_accum 2 --world_size 8 \
    --seq_len 2048 --outdir report_out
```

---

## 6. Notes & gotchas

- **Vocab = 50304.** Match your tokenizer; gpt2 (50257) fits.
- **Fisher regularizer has two modes** (`--fisher_mode`):
  - `energy` (default): L2-like, controls weight scale only; keep `--fisher_lambda` ~`1e-8`.
  - `conditioning`: flattens each circulant block's spectrum to drive kappa -> 1
    (Theorem 2). **Use this to fix the kappa~1e10 / exploding-GradNorm problem.**
    Enable via the launcher: `FISHER_MODE=conditioning ./launch_h800.sh small`
    (auto-sets `--fisher_lambda 0.02`; raise toward `0.05` if kappa falls slowly).
- **Judge quality by CE, not total loss.** Total loss carries MTP + Fisher.
- **`--use_fp8` is a no-op.** Compute is BF16. For real FP8 on Hopper, integrate
  DeepSeek's MIT kernels - DeepGEMM (FP8 grouped/masked MoE GEMM), FlashMLA (MLA
  decode), DeepEP (expert-parallel all-to-all). The current CD win is
  parameters/memory/communication, not raw FLOPs at small block size.
- **Throughput:** if tokens/s is far below expectation, suspect tiny batch,
  gradient-checkpointing recompute, and small-block FFT being memory-bound.
- **Run detached** (`tmux`/`nohup`) so a dropped SSH session can't SIGHUP the job;
  resume with `--resume` from the last checkpoint.
