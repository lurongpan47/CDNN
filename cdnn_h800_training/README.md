# CD-Transformer: 8×H800 Training Suite

**Communication Dynamics Neural Networks with DeepSeek-V3 Cost Reduction — Hopper Edition (FP8)**

This is the H800 (Hopper SM90) variant of the CD-Transformer training pipeline. It uses native FP8 compute (≈3958 TFLOPS on Hopper Tensor Cores) for the linear projections, BF16 for non-FP8-friendly ops, and is tuned for NVLink 4.0 topology (900 GB/s).

For an A100-compatible build (no FP8), see [../cdnn_a100_training/](../cdnn_a100_training/).

## Files

| File | Purpose |
|------|---------|
| `cd_layers.py` | CDLinear, CDAttention (MLA), CDMoELayer, RMSNorm, Fisher reg — with FP8 compute path |
| `cd_model.py` | Full CD-Transformer model + small / medium / large configs |
| `train_distributed.py` | FSDP ZeRO-2, BF16 + FP8, MTP, Hessian monitoring |
| `prepare_data.py` | Multi-backend tokenizer (byte / tiktoken / ModelScope / HuggingFace) |
| `launch_h800.sh` | One-command launcher with NVLink 4.0 NCCL tuning |

## Quick start

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
chmod +x launch_h800.sh

# Synthetic smoke test:
./launch_h800.sh small

# Real training:
python prepare_data.py --source local --input_dir ./corpus --tokenizer byte --output ./data
./launch_h800.sh medium ./data/train.bin
```

## H800 vs A100

| Feature | H800 (this) | A100 |
|---------|-------------|------|
| FP8 compute | ✓ 3958 TFLOPS | ✗ not supported |
| BF16 TFLOPS | 1979 | 312 |
| HBM | HBM3, 3.35 TB/s | HBM2e, 2.0 TB/s |
| NVLink | v4, 900 GB/s | v3, 600 GB/s |
| Precision strategy | FP8 + BF16 | BF16 + TF32 |

The model architecture is identical between the two variants; only the precision stack and NCCL/memory tuning differ. The same three correctness fixes documented in Section VIII of the paper (CDLinear einsum, RoPE under BF16, Fisher regularizer numerical stability) have been back-ported here.

## License

MIT.
