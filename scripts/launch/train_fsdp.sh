#!/usr/bin/env bash
# FSDP full_shard launch — the PyTorch-native DEFAULT for multi-GPU LLM training.
#
# FSDP (Fully Sharded Data Parallel) is what Meta uses to train Llama and what
# HuggingFace recommends first. It shards params + grads + optimizer state across
# GPUs (≈ DeepSpeed ZeRO-3) with zero third-party dependencies.
#
# Run in bf16 full precision (--quantization-bits 0) — that's the standard 2026
# practice. FSDP + bnb 4-bit quantization can be unstable.
#
# Usage:
#   ./scripts/launch/train_fsdp.sh [NUM_GPUS] [MODEL] [OUTPUT_DIR]
#
# Examples:
#   ./scripts/launch/train_fsdp.sh 4                       # 4 GPUs, default model
#   ./scripts/launch/train_fsdp.sh 8 Qwen/Qwen3-14B        # scale up
set -euo pipefail

cd "$(dirname "$0")/../.."   # project root

NUM_GPUS=${1:-$(python -c 'import torch; print(torch.cuda.device_count())')}
MODEL=${2:-"Qwen/Qwen3-1.7B"}
OUTPUT_DIR=${3:-"./outputs/fsdp_$(basename "$MODEL")"}

echo "============================================"
echo "  FSDP Training (PyTorch-native, full_shard)"
echo "  GPUs: ${NUM_GPUS}  |  Model: ${MODEL}"
echo "  Precision: bf16 (full, no quantization)"
echo "============================================"

torchrun \
    --nproc_per_node="${NUM_GPUS}" \
    --nnodes=1 \
    scripts/train_sft_distributed.py \
        --model-name "${MODEL}" \
        --distributed-preset fsdp_full \
        --quantization-bits 0 \
        --output-dir "${OUTPUT_DIR}" \
        --batch-size 2 \
        --gradient-accumulation-steps 4 \
        --epochs 3
