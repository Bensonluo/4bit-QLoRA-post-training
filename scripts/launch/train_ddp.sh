#!/usr/bin/env bash
# Pure DDP launch — the recommended multi-GPU strategy for QLoRA.
#
# DDP gives each rank a full copy of the (4-bit) model and replicates gradients.
# It's the most stable combo with bitsandbytes NF4 quantization — no parameter
# partitioning means zero conflict with QLoRA.
#
# Usage:
#   ./scripts/launch/train_ddp.sh [NUM_GPUS] [MODEL] [OUTPUT_DIR]
#
# Examples:
#   ./scripts/launch/train_ddp.sh                    # auto-detect GPU count
#   ./scripts/launch/train_ddp.sh 4                  # 4 GPUs, default model
#   ./scripts/launch/train_ddp.sh 4 Qwen/Qwen3-1.7B  # 4 GPUs + bigger model
#   ./scripts/launch/train_ddp.sh 2 Qwen/Qwen3-0.6B ./outputs/ddp_0.6b
set -euo pipefail

cd "$(dirname "$0")/../.."   # project root

NUM_GPUS=${1:-$(python -c 'import torch; print(torch.cuda.device_count())')}
MODEL=${2:-"Qwen/Qwen3-0.6B"}
OUTPUT_DIR=${3:-"./outputs/ddp_$(basename "$MODEL")"}

echo "============================================"
echo "  DDP Training (QLoRA-safe)"
echo "  GPUs: ${NUM_GPUS}  |  Model: ${MODEL}"
echo "============================================"

torchrun \
    --nproc_per_node="${NUM_GPUS}" \
    --nnodes=1 \
    scripts/train_sft_distributed.py \
        --model-name "${MODEL}" \
        --distributed-preset ddp \
        --output-dir "${OUTPUT_DIR}" \
        --batch-size 1 \
        --gradient-accumulation-steps 8 \
        --epochs 3
