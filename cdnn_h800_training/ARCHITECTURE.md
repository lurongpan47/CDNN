# CD-Transformer: Communication Dynamics Neural Networks with DeepSeek-V3 Cost Reduction

**FFT-Diagonalized Block-Circulant Layers for Efficient Large-Scale Training on H800 Clusters**

> Based on L. Pan (2026), *"Communication Dynamics Neural Networks: FFT-Diagonalized Layers for Improved Hessian Conditioning at Reduced Parameter Count"* (Paper III), combined with DeepSeek-V3 open-source techniques for frontier-model efficiency.

---

## Overview

CD-Transformer fuses two complementary ideas:

1. **CDNN Theory (Pan, 2026)** — Block-circulant linear layers where the Hessian eigenvalues equal |FFT(c)|², enabling provably well-conditioned optimization (κ → 1 under pre-whitening). A single circulant block of size *B* compresses parameters by ~*B*× while preserving expressivity.

2. **DeepSeek-V3 Cost Reduction** — Mixture-of-Experts (MoE) with auxiliary-loss-free routing, Multi-head Latent Attention (MLA) for KV-cache compression, FP8 mixed-precision training, and Multi-Token Prediction (MTP) auxiliary objectives.

Together, these techniques compound multiplicatively:

| Technique | Source | Reduction Factor |
|-----------|--------|-----------------|
| CDLinear block-circulant | Theorem 1 (Paper III) | ~5× fewer parameters per layer |
| MoE sparse routing | DeepSeek-V3 | ~10× compute (e.g., 6/32 active experts) |
| MLA KV compression | DeepSeek-V3 | ~4× KV cache memory |
| FP8 compute | H800 native | ~2× throughput |
| Shannon dropout (α_CD) | Paper I (Pan & Tanik) | Better generalization per FLOP |
| Fisher regularization | Theorem 2 (Paper III) | Closed-form; prevents over-sharpening |

## Hardware Target

- **Single node, 8× NVIDIA H800** (80 GB HBM3 each)
- 64 GB shared system memory
- NVLink interconnect (auto-configured via NCCL environment)

## Repository Structure

```
cdnn_deepseek_training/
├── README.md                 # This file
├── cd_layers.py              # Core PyTorch layers (CDLinear, MoE, MLA, Shannon dropout)
├── cd_model.py               # Full CD-Transformer model + pre-defined configs
├── train_distributed.py      # Distributed training with FSDP, Hessian monitoring
├── prepare_data.py           # Data tokenization and binary packing
└── launch_h800.sh            # One-command launcher with NCCL optimizations
```

## Quick Start

### 1. Environment Setup

```bash
# PyTorch 2.1+ with CUDA 12.x (required for FP8 support)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Optional: for data preparation from HuggingFace
pip install transformers datasets tiktoken
```

### 2. Prepare Data

**Option A — HuggingFace dataset:**
```bash
python prepare_data.py \
    --source huggingface \
    --dataset openwebtext \
    --tokenizer gpt2 \
    --output_dir ./data \
    --max_tokens 100000000
```

**Option B — Local text files:**
```bash
python prepare_data.py \
    --source local \
    --input_path /path/to/corpus.txt \
    --tokenizer gpt2 \
    --output_dir ./data
```

**Option C — Synthetic data (for testing):**
```bash
python prepare_data.py \
    --source synthetic \
    --output_dir ./data \
    --max_tokens 10000000
```

This produces `data/train.bin` and `data/val.bin` as memory-mapped uint16 token arrays.

### 3. Launch Training

```bash
chmod +x launch_h800.sh
./launch_h800.sh small ./data/train.bin
```

The launcher accepts `small`, `medium`, or `large` as the first argument:

| Config | Layers | Dim | Heads | Experts | Active | Seq Len | Approx. Params |
|--------|--------|------|-------|---------|--------|---------|---------------|
| small | 12 | 1024 | 8 | 16 | 4 | 2048 | ~300M total |
| medium | 24 | 2048 | 16 | 32 | 6 | 4096 | ~2B total |
| large | 32 | 4096 | 32 | 64 | 8 | 4096 | ~15B total |

If no data file is found at the specified path, the launcher automatically falls back to synthetic data generation.

### 4. Direct `torchrun` Launch (Advanced)

```bash
torchrun --nproc_per_node=8 --nnodes=1 \
    train_distributed.py \
    --config medium \
    --data_path ./data/train.bin \
    --batch_size 4 \
    --grad_accum 8 \
    --lr 3e-4 \
    --warmup_steps 2000 \
    --epochs 10 \
    --use_fp8 \
    --use_amp \
    --grad_checkpoint \
    --fisher_lambda 1e-5 \
    --save_dir ./checkpoints
```

## Architecture Details

### CDLinear — Block-Circulant FFT Layer

The core building block replaces standard `nn.Linear` with a block-circulant matrix parameterized by its first row per block. The forward pass:

1. Partition input into blocks of size *B*
2. Compute FFT of circulant parameters: `Λ = FFT(c)`
3. Compute FFT of input blocks: `X̂ = FFT(x)`
4. Element-wise multiply: `Ŷ = Λ ⊙ X̂`
5. Inverse FFT: `y = IFFT(Ŷ)`

By **Theorem 1** (Paper III), the Hessian eigenvalues of this layer are exactly `|Λ_k|²`, making the loss landscape analyzable in closed form. The condition number κ = max|Λ|²/min|Λ|² can be driven to 1 via pre-whitening (Theorem 2).

```python
from cd_layers import CDLinear

layer = CDLinear(in_features=1024, out_features=1024, block_size=4, use_fp8=True)
kappa = layer.condition_number()   # Hessian condition number (Theorem 1)
spectrum = layer.hessian_spectrum() # Full eigenvalue spectrum
```

### CDAttention — Multi-head Latent Attention

Combines DeepSeek-V3's MLA with CDNN compression:
- Low-rank KV projection through a shared latent bottleneck (`kv_lora_rank`)
- Decoupled RoPE on a separate head for positional encoding
- CDLinear used for the latent compression projections

### CDMoELayer — Mixture of CD-Experts

- *N* total experts, *k* active per token (e.g., 32 total, 6 active)
- Each expert is a SwiGLU FFN built with CDLinear layers
- One shared expert always active (DeepSeek-V3 style)
- Auxiliary-loss-free load balancing via learnable bias terms
- Token-choice top-k routing with softmax gating

### Shannon Dropout

Fixed dropout rate α_CD = 0.0118, derived from information-theoretic analysis of communication channel capacity (Paper I, Pan & Tanik). Applied between transformer sub-layers.

### Fisher Regularization

Closed-form regularizer computed from CDLinear FFT coefficients:

```
L_Fisher = λ · Σ_layers Σ_k |Λ_k|²
```

Prevents Hessian eigenvalue explosion without requiring second-order gradient computation.

### Multi-Token Prediction

DeepSeek-V3–style MTP heads predict the next *D* tokens simultaneously from the final hidden state. Lighter sub-networks (fewer experts, smaller dimensions) generate auxiliary losses that improve representation learning.

## Training Infrastructure

### FSDP (Fully Sharded Data Parallelism)

- ZeRO Stage 2: shards optimizer states and gradients across 8 GPUs
- Auto-wrapping at `CDTransformerBlock` granularity
- BF16 mixed precision for compute and communication (see FP8 note below)

### Hessian Monitoring

The `HessianMonitor` tracks CDLinear condition numbers during training, logging:
- Per-layer κ (condition number)
- Aggregate mean and max κ across all CD layers
- Spectral statistics (min/max eigenvalues)

This lets you verify empirically that the Theorem 1 predictions hold during optimization.

### FP8 Compute — Current Status & DeepSeek Integration Path

**Important:** in this codebase the actual matmul compute runs in **BF16**, not FP8.
The `--use_fp8` flag and the `fp8_cast`/`fp8_matmul` helpers are scaffolding — the
quantize/dequantize helper is not on the hot compute path, so enabling the flag does
not currently change numerics or throughput. This is intentional: a correct, hardware-
accelerated FP8 path on H800 (Hopper) should use DeepSeek's open-source kernels rather
than a hand-rolled cast. To get real FP8 speedups, wire in:

- **DeepGEMM** (`github.com/deepseek-ai/DeepGEMM`) — FP8 dense and grouped/masked MoE
  GEMMs with JIT compilation; ~1350+ TFLOPS on Hopper. Use
  `m_grouped_gemm_fp8_fp8_bf16_nt_contiguous` for the MoE expert path.
- **FlashMLA** (`github.com/deepseek-ai/FlashMLA`) — Hopper MLA decode kernel (BF16,
  paged KV, block size 64) for the attention path during inference/decoding.
- **DeepEP** (`github.com/deepseek-ai/DeepEP`) — expert-parallel all-to-all dispatch/
  combine (FP8 dispatch over NVLink) if you scale MoE beyond a single node.

Until those are integrated, the safe and correct default is BF16 AMP (`--use_amp`),
which is what the training loop actually uses.

### Checkpointing

- FSDP full-state-dict checkpoints saved at configurable intervals
- Includes model, optimizer, scheduler, and epoch/step state
- Resume with `--resume ./checkpoints/path/`

## CLI Reference

All arguments for `train_distributed.py`:

```
Model:
  --config {small,medium,large}   Pre-defined model size (default: small)
  --seq_len INT                   Override sequence length

Data:
  --data_path PATH                Path to train.bin (default: ./data/train.bin)
  --synthetic                     Force synthetic data generation
  --synthetic_size INT            Number of synthetic tokens (default: 1M)

Training:
  --epochs INT                    Number of epochs (default: 10)
  --batch_size INT                Per-GPU micro-batch size (default: 8)
  --grad_accum INT                Gradient accumulation steps (default: 4)
  --lr FLOAT                      Peak learning rate (default: 3e-4)
  --warmup_steps INT              LR warmup steps (default: 2000)
  --weight_decay FLOAT            AdamW weight decay (default: 0.1)
  --grad_clip FLOAT               Gradient norm clipping (default: 1.0)

CDNN-Specific:
  --fisher_lambda FLOAT           Fisher regularization coefficient (default: 1e-5)

Efficiency:
  --use_fp8                       Enable FP8 compute (default: True)
  --use_amp                       Enable BF16 AMP (default: True)
  --grad_checkpoint               Enable gradient checkpointing (default: True)

Logging & Saving:
  --log_interval INT              Steps between log entries (default: 50)
  --save_dir PATH                 Checkpoint output directory
  --resume PATH                   Resume from checkpoint directory
```

## Theoretical Background

The CD-Transformer builds on three papers in the Communication Dynamics series:

- **Paper I** (Pan & Tanik) — Fourier-channel energy eigenvalue framework; derives α_CD = 0.0118 from Na D-doublet calibration and the Sadovskii lattice-instability ceiling λ_ceil = 4.0.

- **Paper II** (Pan & Tanik) — Communication channel capacity bounds and their neural network analogues.

- **Paper III** (Pan, 2026) — This paper. CDLinear layer definition, Theorem 1 (Hessian = |FFT(c)|²), Theorem 2 (κ → 1 under pre-whitening), Fisher regularizer derivation, and MNIST empirical validation (reported in the paper as 97.50% accuracy at 3.8× compression with a 310× κ improvement; these are the paper's MNIST figures and have not been re-validated at transformer scale in this repo — use `HessianMonitor` to check κ on your own runs).

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
