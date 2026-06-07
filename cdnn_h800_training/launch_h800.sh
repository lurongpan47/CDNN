#!/bin/bash
# =============================================================================
# launch_h800.sh — Launch CD-Transformer training on 8× H800 GPU cluster
# =============================================================================
#
# CD-Transformer training with DeepSeek-V3 cost-reduction techniques:
#   - CDNN block-circulant layers (5× parameter compression)
#   - MoE sparse experts (10× compute reduction)
#   - FP8 mixed precision (2× H800 throughput)
#   - FSDP ZeRO-2 (memory-efficient sharding)
#   - Multi-Token Prediction auxiliary objective
#   - Fisher regularization (CD Theorem 2)
#
# Hardware: Single node, 8× NVIDIA H800 (80GB HBM3), 64GB shared memory
#
# Usage:
#   chmod +x launch_h800.sh
#   ./launch_h800.sh [small|medium|large] [data_path]
#
# Authors: L. Pan (Ainnocence Inc.)
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL_SIZE="${1:-small}"
DATA_PATH="${2:-./data/train.bin}"
SAVE_DIR="./checkpoints/${MODEL_SIZE}_$(date +%Y%m%d_%H%M%S)"

# Hardware
NUM_GPUS=8
NUM_NODES=1

# Training hyperparameters (DeepSeek-V3 aligned)
case "$MODEL_SIZE" in
    small)
        BATCH_SIZE=${BATCH_SIZE:-16}
        GRAD_ACCUM=${GRAD_ACCUM:-2}
        LR=${LR:-6e-4}
        WARMUP=${WARMUP:-1000}
        EPOCHS=${EPOCHS:-20}
        SEQ_LEN=${SEQ_LEN:-2048}
        ;;
    medium)
        BATCH_SIZE=8
        GRAD_ACCUM=4
        LR=3e-4
        WARMUP=2000
        EPOCHS=10
        SEQ_LEN=4096
        ;;
    large)
        BATCH_SIZE=4
        GRAD_ACCUM=8
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

# --- Fisher regularizer ----------------------------------------------------
# mode "energy"      : L2-like (controls weight scale only); lambda ~1e-8
# mode "conditioning": flattens the CD spectrum, drives kappa -> 1 (Theorem 2);
#                      USE THIS to fix the kappa~1e10 / exploding-GradNorm issue.
#                      lambda is on a different scale here (~1e-2 .. 1e-1).
FISHER_MODE=${FISHER_MODE:-energy}
if [ "$FISHER_MODE" = "conditioning" ]; then
    FISHER_LAMBDA=${FISHER_LAMBDA:-0.02}
else
    FISHER_LAMBDA=${FISHER_LAMBDA:-1e-8}
fi
# conditioning aggregation over blocks: mean | pnorm | max (pnorm/max target the
# worst blocks so a few catastrophic ones aren't diluted by the well-conditioned majority)
FISHER_AGG=${FISHER_AGG:-mean}
# warmup as a fraction of total steps (robust across dataset sizes); empty -> use WARMUP
WARMUP_FRAC=${WARMUP_FRAC:-}
# DENSE=1 trains the non-circulant baseline (same architecture & schedule)
DENSE=${DENSE:-0}

EFFECTIVE_BATCH=$((BATCH_SIZE * NUM_GPUS * GRAD_ACCUM))

echo "============================================================"
echo "CD-Transformer Training Launch"
echo "============================================================"
echo "Model:           CD-Transformer-${MODEL_SIZE}"
echo "GPUs:            ${NUM_GPUS}× H800"
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
# Environment setup for H800
# ---------------------------------------------------------------------------

# NCCL optimizations for H800 NVLink (single node, intra-node NVLink only)
#
# This is a SINGLE-NODE job (NUM_NODES=1): all 8 GPUs talk over NVLink/PCIe,
# so InfiniBand is not used. We intentionally do NOT force NCCL_IB_DISABLE=0
# or pin NCCL_SOCKET_IFNAME=eth0 — on many China-hosted H800 boxes the RDMA
# NIC is absent/misconfigured and the host interface is not named "eth0",
# which makes NCCL hang at init. Let NCCL auto-detect instead.
export NCCL_P2P_LEVEL=NVL           # prefer NVLink for peer-to-peer
export NCCL_P2P_DISABLE=0
export NCCL_SHM_DISABLE=0
# Disable IB for this single-node run (no inter-node traffic). Comment out
# the next line for multi-node runs and set NCCL_SOCKET_IFNAME to your NIC.
export NCCL_IB_DISABLE=1
# If you DO have a working IB fabric and want GDR, instead set:
#   export NCCL_IB_DISABLE=0
#   export NCCL_NET_GDR_LEVEL=2
#   export NCCL_SOCKET_IFNAME=<your_iface>   # e.g. bond0 / ens / eth0
# Uncomment for verbose init logs when debugging a hang:
# export NCCL_DEBUG=INFO

# CUDA settings
export CUDA_DEVICE_MAX_CONNECTIONS=1  # Overlap compute and communication
export TORCH_NCCL_AVOID_RECORD_STREAMS=1  # Reduce memory fragmentation

# PyTorch settings
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512,expandable_segments:True"
export OMP_NUM_THREADS=8

# FP8 support (H800/H100 Hopper architecture)
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1

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
    "num_gpus": ${NUM_GPUS},
    "batch_size": ${BATCH_SIZE},
    "grad_accum": ${GRAD_ACCUM},
    "effective_batch": ${EFFECTIVE_BATCH},
    "lr": ${LR},
    "warmup_steps": ${WARMUP},
    "seq_len": ${SEQ_LEN},
    "epochs": ${EPOCHS},
    "data_path": "${DATA_PATH}",
    "launched_at": "$(date -Iseconds)"
}
EOF

echo ""
echo "Launching torchrun with ${NUM_GPUS} processes..."
echo ""

torchrun \
    --standalone \
    --nproc_per_node=${NUM_GPUS} \
    train_distributed.py \
    --config "$MODEL_SIZE" \
    --data_path "$DATA_PATH" \
    --batch_size "$BATCH_SIZE" \
    --grad_accum "$GRAD_ACCUM" \
    --lr "$LR" \
    --warmup_steps "$WARMUP" \
    --epochs "$EPOCHS" \
    --seq_len "$SEQ_LEN" \
    --save_dir "$SAVE_DIR" \
    --use_fp8 \
    --use_amp \
    --grad_checkpoint \
    --fisher_lambda $FISHER_LAMBDA \
    --fisher_mode $FISHER_MODE \
    --fisher_agg $FISHER_AGG \
    ${WARMUP_FRAC:+--warmup_frac $WARMUP_FRAC} \
    $([ "$DENSE" = "1" ] && echo --dense) \
    --log_interval 50 \
    $USE_SYNTHETIC \
    2>&1 | tee "${SAVE_DIR}/training.log"

echo ""
echo "============================================================"
echo "Training complete. Outputs saved to: ${SAVE_DIR}"
echo "============================================================"
