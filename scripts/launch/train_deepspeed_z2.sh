#!/usr/bin/env bash
# DeepSpeed ZeRO Stage 2 launch — shards optimizer states + gradients.
#
# THE RECOMMENDED strategy for QLoRA multi-GPU training:
#   - Big memory win (Adam state + grad tensors partitioned)
#   - Parameters stay un-partitioned on each rank, so bitsandbytes NF4 quant
#     keeps working without conflict.
#
# Usage:
#   ./scripts/launch/train_deepspeed_z2.sh [NUM_GPUS] [MODEL]
set -euo pipefail

cd "$(dirname "$0")/../.."

NUM_GPUS=${1:-$(python -c 'import torch; print(torch.cuda.device_count())')}
MODEL=${2:-"Qwen/Qwen3-1.7B"}
OUTPUT_DIR=${3:-"./outputs/ds_z2_$(basename "$MODEL")"}

echo "============================================"
echo "  DeepSpeed ZeRO-2 Training  (QLoRA recommended)"
echo "  GPUs: ${NUM_GPUS}  |  Model: ${MODEL}"
echo "============================================"

torchrun \
    --nproc_per_node="${NUM_GPUS}" \
    --nnodes=1 \
    scripts/train_sft_distributed.py \
        --model-name "${MODEL}" \
        --distributed-preset zero_stage_2 \
        --output-dir "${OUTPUT_DIR}" \
        --batch-size 1 \
        --gradient-accumulation-steps 8
