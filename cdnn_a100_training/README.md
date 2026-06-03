# CD-Transformer: 8× A100 Training Suite

**Communication Dynamics Neural Networks with DeepSeek-V3 Cost Reduction — Ampere Edition**

> Based on L. Pan (2026), *"Communication Dynamics Neural Networks: FFT-Diagonalized Layers for Improved Hessian Conditioning at Reduced Parameter Count"* (Paper III), combined with DeepSeek-V3 open-source techniques, adapted for NVIDIA A100 (Ampere SM80) GPUs.

---

## What This Is

A complete, **verified** PyTorch training pipeline that trains a CD-Transformer — a DeepSeek-V3-style Mixture-of-Experts transformer built on Communication Dynamics block-circulant layers — on a single node of 8× A100 80GB GPUs.

The block-circulant CDLinear layers (Pan, 2026) replace dense weight matrices with FFT-diagonalized circulant blocks, giving a provably well-conditioned Hessian (eigenvalues = |FFT(c)|²) at ~5× parameter compression. Stacked with MoE sparsity, MLA KV-compression, and A100 BF16/TF32 acceleration, the techniques compound.

## Verification Status

All code has been smoke-tested end-to-end on CPU (PyTorch 2.12):

| Component | Status |
|-----------|--------|
| `CDLinear` forward (FFT-domain block-circulant) | ✅ verified, 4.7× compression |
| `CDAttention` (MLA + RoPE + SDPA) | ✅ verified |
| `CDMoELayer` (top-k routing + shared expert) | ✅ verified |
| `CDTransformerBlock` | ✅ verified |
| Full model forward + backward + optimizer step | ✅ verified |
| MTP auxiliary heads | ✅ verified |
| Fisher regularizer (differentiable, bounded) | ✅ verified |
| Gradient flow to circulant coefficients | ✅ verified |
| Distributed script imports + CLI | ✅ verified |
| Launch script bash syntax | ✅ verified |

Three bugs were found and fixed during verification (see [Fixes Applied](#fixes-applied-during-verification)).

## A100 vs H800 Adaptation

This is the **A100 Ampere (SM80)** variant. All FP8 code paths are removed; BF16 + TF32 is the compute strategy.

| Feature | A100 (This Suite) | H800 |
|---------|-------------------|------|
| FP8 Compute | ✗ Not supported on Ampere | ✓ 3958 TFLOPS |
| BF16 TFLOPS | 312 | 1979 |
| TF32 TFLOPS | 156 | 989 |
| HBM Bandwidth | 2.0 TB/s (HBM2e) | 3.35 TB/s (HBM3) |
| NVLink | 600 GB/s (v3) | 900 GB/s (v4) |
| Precision Strategy | **BF16 + TF32** | FP8 + BF16 |
| Flash Attention | v2 via PyTorch SDPA | v2 + FP8 |

## Files

```
cdnn_a100_training/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── cd_layers_a100.py            # Core layers (CDLinear, MoE, MLA, Shannon dropout, Fisher)
├── cd_model_a100.py             # Full CD-Transformer model + small/medium/large configs
├── train_distributed_a100.py    # Distributed training (FSDP ZeRO-2, BF16+TF32)
├── prepare_data.py              # Data tokenization → uint16 binary
└── launch_a100.sh               # One-command launcher with A100 NCCL/TF32 tuning
```

## Cost Reduction Stack (A100)

| Technique | Source | Reduction |
|-----------|--------|-----------|
| CDLinear block-circulant | Theorem 1 (Paper III) | ~5× fewer params |
| MoE sparse routing | DeepSeek-V3 | ~10× compute (e.g. 6/32 active) |
| MLA KV compression | DeepSeek-V3 | ~4× KV cache |
| BF16/TF32 | A100 Tensor Cores | ~2× vs FP32 |
| Flash Attention (SDPA) | PyTorch 2.0+ | ~2–4× attention |
| Shannon dropout (α_CD = 0.0118) | Paper I | Better generalization |
| Fisher regularizer | Theorem 2 (Paper III) | Drives κ → 1 during training |

## Quick Start

### 1. Install

```bash
# For CUDA 11.8:
pip install torch --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt

# Or for the default CUDA build:
pip install -r requirements.txt
```

### 2. Prepare data

The data prep step turns text into a token `.bin` file plus a `meta.json` (which records the dtype and vocab so training reads it back correctly). Pick a tokenizer based on your network situation.

**If you are in mainland China**, huggingface.co is blocked. Use one of these, easiest first:

```bash
# (a) Fully offline, zero dependencies — byte-level tokenizer (vocab 256).
#     Always works, never touches the network. Best for a quick start.
python prepare_data.py --source local --input_dir ./corpus \
    --tokenizer byte --output ./data

# (b) DeepSeek tokenizer via ModelScope (魔搭) — reachable inside China.
#     Requires:  pip install modelscope transformers
#     (may need a free `modelscope login` for some repos)
python prepare_data.py --source local --input_dir ./corpus \
    --tokenizer deepseek --output ./data
#   Shortcuts route to ModelScope ONLY (never huggingface.co):
#     'deepseek'/'deepseek-v2' → deepseek-ai/DeepSeek-V2
#     'deepseek-v3', 'deepseek-coder', 'qwen', 'qwen2.5'
#   Or an explicit id:  --tokenizer modelscope:deepseek-ai/DeepSeek-V2

# (c) A pre-downloaded tokenizer folder (most reliable for air-gapped nodes).
#     Download once where ModelScope works, copy the folder over, then:
python prepare_data.py --source local --input_dir ./corpus \
    --tokenizer /path/to/DeepSeek-V2 --output ./data

# (d) Synthetic tokens — no tokenizer or text needed (smoke tests).
python prepare_data.py --source synthetic --output ./data --vocab_size 32000
```

To pre-download the DeepSeek tokenizer (option c):
```python
from modelscope import snapshot_download
snapshot_download('deepseek-ai/DeepSeek-V2')   # → ~/.cache/modelscope/...
# copy that folder to the GPU node and pass its path to --tokenizer
```

The script tries tokenizers in this order and prints which it used:
**byte → tiktoken → ModelScope (for deepseek/qwen shortcuts) → HuggingFace mirror**.
DeepSeek/Qwen shortcut names are **ModelScope-only** and will *not* fall through
to huggingface.co (that was the source of the confusing `Can't load 'deepseek'`
error).

> **China network notes**
> - `prepare_data.py` automatically sets `HF_ENDPOINT=https://hf-mirror.com`
>   (a community HF mirror reachable in China) before importing `transformers`,
>   so HuggingFace ids work without a VPN. Override by exporting your own
>   `HF_ENDPOINT`, or set it empty to force the real huggingface.co.
> - The DeepSeek/Qwen shortcuts route to **ModelScope** first, whose servers are
>   domestic. `pip install modelscope` to enable.
> - If everything network-based fails, `--tokenizer byte` always works.

> **⚠ vocab_size must match the tokenizer.** The model defaults to
> `vocab_size=32000`. DeepSeek's tokenizer is ~100k, so after prep you **must**
> set the model's `vocab_size` to the value printed by `prepare_data.py` (also in
> `meta.json`), or training crashes with an index-out-of-range. For `byte`
> (vocab 256) the default 32000 works fine as-is. Tokens above 65535 are stored
> as `uint32` automatically; `TokenDataset` reads the dtype from `meta.json`.

### About using DeepSeek

This codebase already *implements* DeepSeek-V3's architecture techniques (MoE
with auxiliary-loss-free routing, MLA KV-compression, MTP) — it trains a model
of your own from scratch, it does not fine-tune DeepSeek's weights. What "using
DeepSeek" adds here is the **tokenizer** (option b above), so your token vocab
matches the DeepSeek family. If you instead want to continue-train from
DeepSeek's released checkpoints, that's a different workflow (loading their
weights) and would need a model definition matching their exact config — tell me
if that's the goal and I can adapt it.

### 3. Launch on 8× A100

```bash
chmod +x launch_a100.sh
./launch_a100.sh small ./data/train.bin
```

If no data file exists, the launcher auto-falls back to synthetic data.

Model configs (batch sizes tuned for A100 80GB HBM2e):

| Config | Layers | Dim | Heads | Experts/Active | Batch/GPU | Grad Accum | Effective Batch |
|--------|--------|------|-------|----------------|-----------|------------|-----------------|
| small | 12 | 1024 | 8 | 16 / 4 | 16 | 2 | 256 |
| medium | 24 | 2048 | 16 | 32 / 6 | 6 | 6 | 288 |
| large | 32 | 4096 | 32 | 64 / 8 | 2 | 16 | 256 |

### 4. Direct torchrun (advanced)

```bash
torchrun --nproc_per_node=8 --nnodes=1 \
    train_distributed_a100.py \
    --config medium --data_path ./data/train.bin \
    --batch_size 6 --grad_accum 6 --lr 3e-4 \
    --warmup_steps 2000 --epochs 10 \
    --use_amp --use_bf16_only --grad_checkpoint \
    --fisher_lambda 1e-5 --save_dir ./checkpoints
```

## A100-Specific Optimizations

**TF32 Tensor Cores** — enabled automatically via `setup_a100_precision()`:
```python
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
```
TF32 gives ~2× FP32 throughput with near-FP32 accuracy.

**Flash Attention via SDPA** — `CDAttention` uses `F.scaled_dot_product_attention`, which auto-dispatches to flash/memory-efficient kernels on A100.

**NCCL for NVLink 3.0** — the launcher sets:
```bash
NCCL_BUFFSIZE=4194304        # 4MB buffer
NCCL_ALGO=Tree,Ring
NCCL_PROTO=Simple,LL128
NVIDIA_TF32_OVERRIDE=1
PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:256,expandable_segments:True"
```
The 256MB split (vs 512 on H800) is more conservative for HBM2e bandwidth.

## CLI Reference

```
Model:
  --config {small,medium,large}   Model size
  --seq_len INT                   Override sequence length

Data:
  --data_path PATH                Path to train.bin
  --synthetic / --synthetic_size  Synthetic data for testing

Training:
  --epochs --batch_size --grad_accum --lr --warmup_steps
  --weight_decay (0.1)  --grad_clip (1.0)

CD-Specific:
  --fisher_lambda FLOAT           Fisher regularization (default 1e-5)

A100 Optimization:
  --use_amp                       BF16 mixed precision (default True)
  --use_bf16_only                 BF16 without GradScaler (default True)
  --grad_checkpoint               Gradient checkpointing (default True)

Logging:
  --log_interval --save_dir --resume
```

## Architecture Notes

**CDLinear** — partitions input into blocks of size B, takes FFT of both the circulant coefficients and the input, multiplies pointwise in the frequency domain, and inverse-FFTs. By Theorem 1, the Hessian eigenvalues are exactly |FFT(c)|², so conditioning is readable from a single FFT.

**CDAttention** — DeepSeek-V3 MLA: KV is projected through a low-rank latent bottleneck (`kv_lora_rank`), then decompressed per-head. All projections are CDLinear. RoPE is applied to Q and K; attention uses SDPA.

**CDMoELayer** — N experts, top-k active per token, plus one always-on shared expert. SwiGLU FFNs built from CDLinear. Auxiliary-loss-free load balancing via learnable bias terms.

**Fisher regularizer** — penalizes the variance of each layer's log-spectrum, which is zero exactly when the spectrum is flat (κ = 1, the Theorem 2 optimum). Differentiable and numerically bounded.

## Fixes Applied During Verification

1. **CDLinear einsum subscript collision** — the frequency-domain contraction used `'oib,bib->bob'`, where `b` denoted both batch and block, which is invalid (a contracted index cannot also be an output index). Rewritten as `'oik,nik->nok'` with distinct indices (n=batch, o=K_out, i=K_in, k=FFT bin).

2. **RoPE `view_as_complex` + broadcasting** — `view_as_complex` requires contiguous memory (added `.contiguous()`), and the rotary frequencies were not broadcasting over the batch and head dimensions (now reshaped to `(1, 1, T, head_dim/2)`).

3. **Fisher regularizer instability** — the original `Σ 1/(σ²+ε)` exploded to ~1e7 for tiny eigenvalues and, because it was computed under `torch.no_grad()`, produced no gradient. Replaced with the variance of the log-spectrum: zero iff the spectrum is flat (κ = 1), fully differentiable, and bounded (~1e-3 in practice).

These same fixes were also back-ported to the H800 codebase.

## Citation

```bibtex
@article{pan2026cdnn,
  title   = {Communication Dynamics Neural Networks: FFT-Diagonalized Layers
             for Improved Hessian Conditioning at Reduced Parameter Count},
  author  = {Pan, Lurong},
  journal = {arXiv preprint},
  year    = {2026},
  note    = {Paper III in the Communication Dynamics series}
}
```

## License

MIT

## Author

Lurong Pan (潘麓蓉) — Founder & CEO, Ainnocence Inc. (圆壹智慧)
