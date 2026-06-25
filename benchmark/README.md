# Distributed Training Benchmarks

> Methodology + result tables for single-GPU vs multi-GPU scaling. **Fill in the numbers** after running `./scripts/launch/benchmark_distributed.sh`.

---

## 📐 Methodology

### Setup
- **Model**: Qwen3-1.7B (4-bit QLoRA, LoRA r=16)
- **Dataset**: Alpaca (200 samples, single epoch)
- **Hardware**: _<fill in: e.g. 4× NVIDIA RTX 4090 24GB>_
- **Software**: PyTorch `<version>`, transformers `<version>`, deepspeed `<version>`
- **Measurement**: `train_samples_per_second` (HF Trainer's `train_runtime` summary), peak VRAM via `torch.cuda.max_memory_allocated`

> ⚠️ **Anonymization note**: If you're benchmarking on employer hardware, do NOT include hostnames, internal cluster names, or proprietary config. Report only GPU model + count. Numbers alone are fine to publish.

### What we measure
- **Throughput**: samples/sec — higher is better
- **Speedup**: multi-GPU throughput ÷ single-GPU throughput
- **Efficiency**: speedup ÷ GPU count (1.0 = perfect linear scaling)
- **Peak memory**: max VRAM/GPU during training

---

## 📊 Results: Single-GPU vs DDP

> Run: `./scripts/launch/benchmark_distributed.sh Qwen/Qwen3-1.7B 4`

| Config | GPUs | samples/sec | Speedup | Efficiency | Peak VRAM/GPU |
|--------|------|-------------|---------|------------|---------------|
| Single-GPU | 1 | _<fill>_ | 1.0× | 100% | _<fill>_ GB |
| DDP | 2 | _<fill>_ | _<fill>× | _<fill>% | _<fill>_ GB |
| DDP | 4 | _<fill>_ | _<fill>× | _<fill>% | _<fill>_ GB |

**Expected behavior**: DDP throughput scales near-linearly (80-95% efficiency) until communication overhead dominates. Peak VRAM per GPU stays roughly constant (DDP doesn't shard).

---

## 📊 Results: DeepSpeed ZeRO Stages

> Run with `--distributed-preset zero_stage_1/2` on 4 GPUs.

| Strategy | shards what? | samples/sec | Peak VRAM/GPU | vs DDP memory |
|----------|-------------|-------------|---------------|----------------|
| DDP | nothing | _<fill>_ | _<fill>_ GB | baseline |
| ZeRO-1 | optimizer state | _<fill>_ | _<fill>_ GB | _<fill>% |
| ZeRO-2 | optimizer + grads | _<fill>_ | _<fill>_ GB | _<fill>% |
| ZeRO-3 + offload | everything (+ CPU) | _<fill>_ | _<fill>_ GB | _<fill>% |

**Expected behavior**: ZeRO-2 typically saves 30-50% VRAM vs DDP for QLoRA (the Adam optimizer state is the dominant memory consumer). Throughput drops slightly (~5-10%) due to extra communication. ZeRO-3 trades more memory for more overhead, and may be unstable with 4-bit quantization.

---

## 💰 Cost Analysis (Cloud GPU)

For a representative 50k-sample SFT run on Qwen3-7B (4-bit QLoRA):

| Provider | GPU | Count | $/GPU/hr | Est. time | Est. cost |
|----------|-----|-------|----------|-----------|-----------|
| _<provider>_ | _<gpu>_ | 1 | _$_ | _<h>_ | _$_ |
| _<provider>_ | _<gpu>_ | 4 | _$_ | _<h>_ | _$_ |

> AutoDL (China) often offers 4× consumer cards at ¥15-25/GPU/hr. Lambda Labs / Vast.ai for US-based A100s.

---

## 📝 How to Update This File

1. Run the benchmark: `./scripts/launch/benchmark_distributed.sh Qwen/Qwen3-1.7B 4`
2. Open `outputs/benchmark_<timestamp>/` — one `.log` per strategy
3. Extract `train_samples_per_second` and peak memory from each log
4. Paste into the tables above (replace `_<fill>_`)
5. Commit with message: `docs(benchmark): add Qwen3-1.7B scaling results`

Keep results **anonymized** — GPU model and count only, no host/cluster identifiers.
