#!/usr/bin/env bash
# DeepSpeed ZeRO Stage 1 launch — shards optimizer states across GPUs.
#
# Use when the model + gradients fit per-GPU but the optimizer state
# (Adam = ~2x params in fp32) causes OOM. Safe with QLoRA.
#
# Usage:
#   ./scripts/launch/train_deepspeed_z1.sh [NUM_GPUS] [MODEL]
set -euo pipefail

cd "$(dirname "$0")/../.."

NUM_GPUS=${1:-$(python -c 'import torch; print(torch.cuda.device_count())')}
MODEL=${2:-"Qwen/Qwen3-1.7B"}
OUTPUT_DIR=${3:-"./outputs/ds_z1_$(basename "$MODEL")"}

echo "============================================"
echo "  DeepSpeed ZeRO-1 Training"
echo "  GPUs: ${NUM_GPUS}  |  Model: ${MODEL}"
echo "============================================"

torchrun \
    --nproc_per_node="${NUM_GPUS}" \
    --nnodes=1 \
    scripts/train_sft_distributed.py \
        --model-name "${MODEL}" \
        --distributed-preset zero_stage_1 \
        --output-dir "${OUTPUT_DIR}" \
        --batch-size 1 \
        --gradient-accumulation-steps 8
