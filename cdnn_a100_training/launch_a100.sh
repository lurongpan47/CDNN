#!/bin/bash
# =============================================================================
# launch_a100.sh — Launch CD-Transformer training on 8× A100 GPU cluster
# =============================================================================
#
# CD-Transformer training with DeepSeek-V3 cost-reduction techniques,
# adapted for NVIDIA A100 Ampere architecture.
#
# A100 vs H800 key optimizations:
#   - BF16 + TF32 precision (no FP8 on Ampere)
#   - NVLink 3.0 topology (600 GB/s, vs H800 NVLink 4.0 at 900 GB/s)
#   - 80GB HBM2e (2 TB/s bandwidth, vs H800 HBM3 at 3.35 TB/s)
#   - Flash Attention v2 via PyTorch SDPA
#   - NCCL tuned for A100 NVLink/PCIe topology
#
# Hardware: Single node, 8× NVIDIA A100 (80GB HBM2e)
#
# Usage:
#   chmod +x launch_a100.sh
#   ./launch_a100.sh [small|medium|large] [data_path]
#
# Authors: L. Pan (Ainnocence Inc.)
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL_SIZE="${1:-small}"
DATA_PATH="${2:-./data/train.bin}"
SAVE_DIR="./checkpoints/a100_${MODEL_SIZE}_$(date +%Y%m%d_%H%M%S)"

# Hardware
NUM_GPUS=8
NUM_NODES=1

# Training hyperparameters — A100 tuned
# A100 has ~60% the BF16 throughput of H800, so we adjust batch sizes
# to maintain similar GPU memory utilization (80GB HBM2e)
case "$MODEL_SIZE" in
    small)
        BATCH_SIZE=16       # Same as H800 (small model fits easily)
        GRAD_ACCUM=2
        LR=6e-4
        WARMUP=1000
        EPOCHS=20
        SEQ_LEN=2048
        ;;
    medium)
        BATCH_SIZE=6        # Reduced from H800's 8 (lower BW → safer margin)
        GRAD_ACCUM=6        # Increased to maintain effective batch size ~288
        LR=3e-4
        WARMUP=2000
        EPOCHS=10
        SEQ_LEN=4096
        ;;
    large)
        BATCH_SIZE=2        # Reduced from H800's 4 (HBM2e has less BW)
        GRAD_ACCUM=16       # Increased to maintain effective batch size ~256
        LR=1.5e-4
        WARMUP=4000
        EPOCHS=5
        SEQ_LEN=4096
        ;;
    *)
        echo "Unknown model size: $MODEL_SIZE (choose: small, medium, large)"
        exit 1
        ;;
esac

EFFECTIVE_BATCH=$((BATCH_SIZE * NUM_GPUS * GRAD_ACCUM))

echo "============================================================"
echo "CD-Transformer Training Launch (A100)"
echo "============================================================"
echo "Model:           CD-Transformer-${MODEL_SIZE}"
echo "GPUs:            ${NUM_GPUS}× A100 80GB"
echo "Precision:       BF16 + TF32 (Ampere)"
echo "Batch/GPU:       ${BATCH_SIZE}"
echo "Grad accum:      ${GRAD_ACCUM}"
echo "Effective batch: ${EFFECTIVE_BATCH}"
echo "Learning rate:   ${LR}"
echo "Warmup steps:    ${WARMUP}"
echo "Seq length:      ${SEQ_LEN}"
echo "Epochs:          ${EPOCHS}"
echo "Data:            ${DATA_PATH}"
echo "Save dir:        ${SAVE_DIR}"
echo "============================================================"

# ---------------------------------------------------------------------------
# GPU Detection and Validation
# ---------------------------------------------------------------------------
if command -v nvidia-smi &> /dev/null; then
    echo ""
    echo "GPU Info:"
    nvidia-smi --query-gpu=index,name,memory.total,driver_version \
               --format=csv,noheader 2>/dev/null || true
    echo ""

    # Check if A100 is detected
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo "unknown")
    if [[ "$GPU_NAME" == *"A100"* ]]; then
        echo "✓ A100 GPU detected: $GPU_NAME"
    elif [[ "$GPU_NAME" == *"H100"* ]] || [[ "$GPU_NAME" == *"H800"* ]]; then
        echo "⚠ Hopper GPU detected ($GPU_NAME) — consider using launch_h800.sh"
        echo "  for FP8 acceleration. Proceeding with BF16 training..."
    else
        echo "⚠ GPU: $GPU_NAME (not A100). BF16/TF32 may still work."
    fi
fi

# ---------------------------------------------------------------------------
# Environment setup for A100 (Ampere SM80)
# ---------------------------------------------------------------------------

# --- NCCL optimizations for A100 NVLink 3.0 ---
# A100 DGX uses NVLink 3.0 (600 GB/s total bidirectional per GPU)
# Different from H800 NVLink 4.0 (900 GB/s)
export NCCL_IB_DISABLE=0               # Enable InfiniBand (if present)
export NCCL_NET_GDR_LEVEL=2            # GPU Direct RDMA level
export NCCL_P2P_LEVEL=NVL              # NVLink peer-to-peer
export NCCL_SHM_DISABLE=0              # Enable shared memory transport
export NCCL_SOCKET_IFNAME=eth0         # Primary network interface
export NCCL_BUFFSIZE=4194304           # 4MB NCCL buffer (tuned for A100 NVLink)

# A100 NVLink 3.0 specific: tree algorithm is better for small messages
export NCCL_ALGO=Tree,Ring
export NCCL_PROTO=Simple,LL128

# --- CUDA settings for Ampere ---
export CUDA_DEVICE_MAX_CONNECTIONS=1   # Overlap compute and communication

# TF32 — the key A100 acceleration (2× FP32 throughput)
# TF32 uses 19-bit mantissa in a 32-bit format, giving nearly FP32 accuracy
# at 2× the throughput (156 TFLOPS TF32 vs 19.5 TFLOPS FP32)
export NVIDIA_TF32_OVERRIDE=1          # Force TF32 for all ops

# Memory configuration for A100 80GB HBM2e
# HBM2e has 2 TB/s bandwidth (vs H800 HBM3 at 3.35 TB/s)
# Slightly more conservative memory allocation
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:256,expandable_segments:True"

# Avoid NCCL stream recording overhead
export TORCH_NCCL_AVOID_RECORD_STREAMS=1

# CPU threading
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

# PyTorch compile settings (for PyTorch 2.0+)
export TORCH_COMPILE_DEBUG=0
export TORCHDYNAMO_VERBOSE=0

# ---------------------------------------------------------------------------
# Data check
# ---------------------------------------------------------------------------
USE_SYNTHETIC=""
if [ ! -f "$DATA_PATH" ]; then
    echo ""
    echo "WARNING: Data file not found at ${DATA_PATH}"
    echo "         Using synthetic data for testing."
    echo "         To use real data, run prepare_data.py first."
    echo ""
    USE_SYNTHETIC="--synthetic --synthetic_size 2000000"
fi

# ---------------------------------------------------------------------------
# Launch training
# ---------------------------------------------------------------------------
mkdir -p "$SAVE_DIR"

# Save launch config
cat > "${SAVE_DIR}/launch_config.json" << EOF
{
    "model_size": "${MODEL_SIZE}",
    "hardware": "A100_80GB",
    "precision": "BF16_TF32",
    "num_gpus": ${NUM_GPUS},
    "batch_size": ${BATCH_SIZE},
    "grad_accum": ${GRAD_ACCUM},
    "effective_batch": ${EFFECTIVE_BATCH},
    "lr": ${LR},
    "warmup_steps": ${WARMUP},
    "seq_len": ${SEQ_LEN},
    "epochs": ${EPOCHS},
    "data_path": "${DATA_PATH}",
    "fp8": false,
    "bf16": true,
    "tf32": true,
    "launched_at": "$(date -Iseconds)"
}
EOF

echo ""
echo "Launching torchrun with ${NUM_GPUS} processes (A100 BF16+TF32)..."
echo ""

torchrun \
    --standalone \
    --nproc_per_node=${NUM_GPUS} \
    train_distributed_a100.py \
    --config "$MODEL_SIZE" \
    --data_path "$DATA_PATH" \
    --batch_size "$BATCH_SIZE" \
    --grad_accum "$GRAD_ACCUM" \
    --lr "$LR" \
    --warmup_steps "$WARMUP" \
    --epochs "$EPOCHS" \
    --seq_len "$SEQ_LEN" \
    --save_dir "$SAVE_DIR" \
    --use_amp \
    --use_bf16_only \
    --grad_checkpoint \
    --fisher_lambda 1e-5 \
    --log_interval 50 \
    $USE_SYNTHETIC \
    2>&1 | tee "${SAVE_DIR}/training.log"

echo ""
echo "============================================================"
echo "Training complete. Outputs saved to: ${SAVE_DIR}"
echo "============================================================"
