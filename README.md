# CDNN — Communication Dynamics Neural Networks

**FFT-diagonalized block-circulant layers with provably well-conditioned Hessians, plus a reference PyTorch implementation that integrates them with DeepSeek-V3–style Mixture-of-Experts on 8×A100 / 8×H800 GPU clusters.**

> Pan, L. (2026). *Communication Dynamics Neural Networks: FFT-Diagonalized Layers for Improved Hessian Conditioning at Reduced Parameter Count.* arXiv:2605.08171. [→ paper PDF](paper/CDNN_paper_v3.pdf)

---

## What this is

A CDLinear layer replaces a dense weight matrix with a block-circulant matrix of block size B (where B = 2ℓ+1 follows from the polygon multiplicity of Communication Dynamics theory). Three useful properties drop out of the construction:

1. **Closed-form Hessian** — the Hessian of MSE loss is diagonalized by the DFT with eigenvalues |FFT(input)|², readable from a single FFT per mini-batch with no matrix decomposition (Theorem 1).
2. **Conditioning bound** — under input pre-whitening, the population Hessian condition number is exactly κ = 1; the empirical κ is bounded by 1 + O(√(B/N)) (Theorem 2).
3. **Parameter compression** — B× fewer weight parameters than a dense layer of the same input/output dimensions.

On the 8×8 MNIST benchmark, a 3-layer CD-MLP at B=4 reaches 97.50% ± 0.23% test accuracy with **3.8× fewer parameters** and a **310× smaller Hessian condition number** than a parameter-matched dense baseline, across three seeds.

This repository contains the original NumPy reference implementation, the LaTeX source of the paper, and a production PyTorch implementation that integrates CDLinear with DeepSeek-V3's Mixture-of-Experts, Multi-head Latent Attention, and Multi-Token Prediction techniques for training at scale on contemporary GPU clusters.

## Repository layout

```
CDNN/
├── paper/                         # LaTeX source and compiled PDF
│   ├── main.tex                   # REVTeX 4-2, builds with pdflatex
│   ├── cd_mnist_curves.png        # Fig. 1
│   ├── cd_mnist_spectrum.png      # Fig. 2
│   └── CDNN_paper_v3.pdf          # Compiled 18-page PDF
│
├── cdnn/                          # Original NumPy CDLinear + MNIST experiment
│   ├── cd_nn.py                   # Pure-NumPy CDLinear library, hand-derived backward
│   ├── cd_mnist_experiment.py     # MNIST benchmark (reproduces Table I)
│   ├── run_aggregate.py           # 3-seed aggregation script
│   └── cd_mnist_aggregate.json    # Cached results
│
├── cdnn_a100_training/            # Reference PyTorch implementation (verified)
│   ├── README.md                  # A100-specific usage and design notes
│   ├── cd_layers_a100.py          # CDLinear, CDAttention (MLA), CDMoELayer, RMSNorm
│   ├── cd_model_a100.py           # CD-Transformer + small/medium/large configs
│   ├── train_distributed_a100.py  # FSDP ZeRO-2, BF16+TF32, MTP, Hessian monitoring
│   ├── prepare_data.py            # Multi-backend tokenizer (byte/tiktoken/ModelScope/HF)
│   ├── launch_a100.sh             # One-command launcher with NVLink 3.0 NCCL tuning
│   └── requirements.txt
│
└── cdnn_h800_training/            # Variant for H800/Hopper with FP8
    ├── cd_layers.py               # Same logic + FP8 compute path
    ├── cd_model.py
    ├── train_distributed.py
    ├── prepare_data.py
    └── launch_h800.sh
```

## Quick start

### Reproduce the MNIST result from the paper

```bash
pip install numpy scikit-learn matplotlib
cd cdnn/
python cd_mnist_experiment.py        # single seed
python run_aggregate.py              # all three seeds, regenerates the table
```

### Train a CD-Transformer on 8×A100 80GB

```bash
cd cdnn_a100_training/
pip install -r requirements.txt
chmod +x launch_a100.sh

# Smoke test (synthetic data, no tokenizer required):
./launch_a100.sh small

# Real training: prepare tokens first, then launch.
python prepare_data.py --source local --input_dir ./corpus --tokenizer byte --output ./data
./launch_a100.sh medium ./data/train.bin
```

See `cdnn_a100_training/README.md` for full details — including how to use a DeepSeek or Qwen tokenizer via ModelScope inside mainland China, and how to recover the actual error if `torchrun` exits silently.

### Train on 8×H800 (Hopper, FP8)

```bash
cd cdnn_h800_training/
./launch_h800.sh medium ./data/train.bin
```

H800 uses FP8 compute (≈3958 TFLOPS) plus BF16 communication for an additional ~2× throughput over the A100 BF16+TF32 path.

## What was verified, and what was not

End-to-end verification of the PyTorch implementation was performed on CPU using a deliberately small configuration (vocab 256, d_model 64, 2 layers, 4 experts). The probe exercises every code path: CDLinear forward/backward, CDAttention (SDPA flash attention), MoE routing, MTP auxiliary heads, Fisher regularizer, gradient checkpointing, and an AdamW optimizer step. Three concrete bugs were caught and corrected during this verification — they are documented in Section VIII of the paper, but for completeness:

1. **CDLinear FFT einsum** — the original subscript `'oib,bib->bob'` reused `b` for both batch and block, which is invalid. Corrected to `'oik,nik->nok'` with distinct indices.
2. **RoPE under BF16** — `view_as_complex` rejected non-contiguous tensors, and rotary frequencies were not broadcasting over batch/heads. Fixed with `.contiguous()` and explicit reshape.
3. **Fisher regularizer** — the original Σ 1/(σ²+ε) form blew up for tiny eigenvalues and was computed inside `torch.no_grad()`, so it produced no gradient. Replaced with the variance of the log-spectrum, which is exactly zero at the κ=1 optimum, differentiable, and bounded.

**Not yet verified on real hardware in this release:** the 8×A100 distributed run itself, BF16/TF32 Tensor Core kernels, FSDP sharding, and ModelScope DeepSeek tokenizer download. The script logic was syntactically and semantically validated, but a smoke run on one A100 is recommended before kicking off a long job.

## Reproducing the paper

The MNIST experiment (Sec. VI of the paper) is one command:
```bash
cd cdnn/ && python run_aggregate.py
```

This produces both `cd_mnist_aggregate.json` and the two figures in `paper/`.

To rebuild the paper:
```bash
cd paper/ && pdflatex main.tex && pdflatex main.tex
```
Requires REVTeX 4-2 (Debian: `texlive-publishers`). Compiles in 18 pages with no warnings.

## Citation

```bibtex
@article{pan2026cdnn,
  title   = {Communication Dynamics Neural Networks: FFT-Diagonalized
             Layers for Improved Hessian Conditioning at Reduced Parameter Count},
  author  = {Pan, Lurong},
  journal = {arXiv preprint arXiv:2605.08171},
  year    = {2026},
  note    = {Paper III in the Communication Dynamics series}
}
```

## License

MIT — see [LICENSE](LICENSE).

## Author

Lurong Pan (潘麓蓉) — Founder and CEO, [Ainnocence Inc.](https://ainnocence.com)
lurong.pan@ainnocence.com
