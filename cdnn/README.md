# Pure-NumPy reference implementation

This is the original implementation that backs the MNIST experiment in Section VI of the paper. It is pure NumPy with hand-derived backward passes, verified against finite-difference gradient checks to < 1e-4 relative error. Use this code to reproduce Table I and Figures 1–2 of the paper, and as a small, dependency-light reference for the CDLinear math.

For training at scale (Mixture-of-Experts, multi-GPU, BF16/FP8), see:
- [`../cdnn_a100_training/`](../cdnn_a100_training/) — PyTorch + FSDP, 8×A100, BF16+TF32
- [`../cdnn_h800_training/`](../cdnn_h800_training/) — PyTorch + FSDP, 8×H800, FP8

## Files

| File | Purpose |
|------|---------|
| `cd_nn.py` | CDLinear forward + analytic backward (FFT-domain), Hessian spectrum, Shannon dropout, Fisher regularizer |
| `cd_mnist_experiment.py` | 3-layer CD-MLP vs dense MLP on `sklearn.datasets.load_digits` |
| `run_aggregate.py` | Run all three seeds and aggregate (regenerates Table I numbers) |
| `cd_mnist_aggregate.json` | Cached results across seeds {0, 1, 2} |

## Reproduce the paper

```bash
pip install numpy scikit-learn matplotlib

# Single run:
python cd_mnist_experiment.py

# All three seeds, full Table I:
python run_aggregate.py
```

Expected output (from the paper, 3 seeds, mean ± std):

| Model | Params | Test acc. | Hessian κ |
|---|---|---|---|
| Dense MLP | 8,970 | 98.15% ± 0.47% | 5.9 × 10⁶ |
| CD-MLP B=4 | 2,380 | 97.50% ± 0.23% | 1.9 × 10⁴ |
| CD-MLP B=8 | 1,296 | 96.39% ± 1.13% | 5.1 × 10² |

## License

MIT.
