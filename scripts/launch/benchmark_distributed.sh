#!/usr/bin/env bash
# Distributed benchmark: single-GPU vs N-GPU DDP vs DeepSpeed ZeRO stages.
#
# Runs the SAME model/dataset under different strategies and collects throughput
# + peak memory, so you can see the actual scaling. Output is human-readable;
# paste the numbers into benchmark/*.md (sanitized — no hostnames/cluster info).
#
# Usage:
#   ./scripts/launch/benchmark_distributed.sh [MODEL] [NUM_GPUS]
#
# Requirements:
#   - Multiple CUDA GPUs (else only the single-GPU row runs)
#   - deepspeed installed for ZeRO stages (pip install deepspeed)
set -euo pipefail

cd "$(dirname "$0")/../.."

MODEL=${1:-"Qwen/Qwen3-1.7B"}
NUM_GPUS=${2:-$(python -c 'import torch; print(torch.cuda.device_count())')}
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BASE_OUT="./outputs/benchmark_${TIMESTAMP}"

echo "============================================"
echo "  Distributed Benchmark"
echo "  Model: ${MODEL}  |  GPUs: ${NUM_GPUS}"
echo "  Output: ${BASE_OUT}"
echo "============================================"

# Small dataset for fast benchmark — we care about throughput, not final loss.
COMMON_ARGS="--model-name ${MODEL} --max-samples 200 --epochs 1 --batch-size 1 --gradient-accumulation-steps 4"

run_one() {
    local label="$1"; shift
    echo ""
    echo ">>> [${label}] starting..."
    "$@" 2>&1 | tee "${BASE_OUT}/${label}.log" || echo "    (FAILED — see log)"
    echo "<<< [${label}] done"
}

mkdir -p "${BASE_OUT}"

# 1. Single-GPU baseline (no torchrun).
run_one "single_gpu" \
    python scripts/train_sft_distributed.py ${COMMON_ARGS} \
        --output-dir "${BASE_OUT}/single_gpu" --distributed-preset ddp

if [ "${NUM_GPUS}" -gt 1 ]; then
    # 2. Pure DDP.
    run_one "ddp_${NUM_GPUS}gpu" \
        torchrun --nproc_per_node="${NUM_GPUS}" scripts/train_sft_distributed.py ${COMMON_ARGS} \
            --output-dir "${BASE_OUT}/ddp" --distributed-preset ddp

    # 3. FSDP full_shard (PyTorch-native default, bf16 full precision).
    run_one "fsdp_full_${NUM_GPUS}gpu" \
        torchrun --nproc_per_node="${NUM_GPUS}" scripts/train_sft_distributed.py ${COMMON_ARGS} \
            --output-dir "${BASE_OUT}/fsdp" --distributed-preset fsdp_full --quantization-bits 0

    # 4. DeepSpeed ZeRO-2 (for comparison with FSDP).
    if python -c "import deepspeed" 2>/dev/null; then
        run_one "ds_zero2_${NUM_GPUS}gpu" \
            torchrun --nproc_per_node="${NUM_GPUS}" scripts/train_sft_distributed.py ${COMMON_ARGS} \
                --output-dir "${BASE_OUT}/ds_z2" --distributed-preset zero_stage_2

        # 5. DeepSpeed ZeRO-3 + offload (extreme scale).
        run_one "ds_zero3_offload_${NUM_GPUS}gpu" \
            torchrun --nproc_per_node="${NUM_GPUS}" scripts/train_sft_distributed.py ${COMMON_ARGS} \
                --output-dir "${BASE_OUT}/ds_z3" --distributed-preset zero_stage_3_offload --quantization-bits 0
    else
        echo ""
        echo "⚠ deepspeed not installed — skipping ZeRO benchmarks. Install with: pip install deepspeed"
    fi
else
    echo ""
    echo "⚠ Only ${NUM_GPUS} GPU detected — multi-GPU benchmarks skipped."
fi

echo ""
echo "============================================"
echo "  Benchmark complete. Logs in ${BASE_OUT}/"
echo "  Extract samples/sec + peak_mem from the .log files"
echo "  and paste (sanitized) into benchmark/README.md"
echo "============================================"
