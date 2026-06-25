# Distributed LLM Training Guide — FSDP-First, Multi-GPU Scaling

> The industry-standard open-source stack for 2026: **FSDP (PyTorch-native) in bf16 full precision**, with DeepSpeed as the scale-beyond option. Resource constraints are not assumed — we use what Meta/HuggingFace actually recommend.

---

## 📌 Table of Contents

- [The 2026 Standard Stack](#-the-2026-standard-stack)
- [Prerequisites](#-prerequisites)
- [Strategy Selection](#-strategy-selection)
- [Quick Start: 3 Commands](#-quick-start-3-commands)
- [How It Works](#-how-it-works)
- [Why FSDP over DeepSpeed](#-why-fsdp-over-deepspeed)
- [When You Still Need DeepSpeed](#-when-you-still-need-deepspeed)
- [Step-by-Step Walkthrough](#-step-by-step-walkthrough)
- [Benchmarking](#-benchmarking)
- [Troubleshooting](#-troubleshooting)
- [Going Further: Multi-node](#-going-further-multi-node)

---

## 🎯 The 2026 Standard Stack

| Layer | Choice | Why |
|-------|--------|-----|
| **Strategy** | **FSDP** (PyTorch native) | No third-party deps; what Meta uses for Llama; HF's first recommendation |
| **Precision** | **bf16 full precision** | No quantization — full model quality, and FSDP shards it fine |
| **Launcher** | **torchrun** | Ships with PyTorch; sets up process groups |
| **Scale-beyond** | **DeepSpeed ZeRO-3 + offload** | When FSDP OOMs (huge models or CPU/NVMe offload needed) |
| **Fallback** | **DDP** | Plain replication — the baseline and QLoRA companion |

> **Why not QLoRA + ZeRO-2 anymore?** That combo was a *resource-constrained* workaround. When GPUs aren't scarce, bf16 + FSDP gives better model quality, simpler code, and full parameter training — which is what production teams actually ship. QLoRA still lives in this repo for genuinely memory-tight scenarios, but it's no longer the distributed default.

---

## 🔧 Prerequisites

### Hardware
- **2+ NVIDIA GPUs** (A100/H100/RTX 4090/etc.) on one node — FSDP is designed for this
- bf16 support (all Ampere+ GPUs: A100, H100, RTX 30/40 series)

### Software
```bash
# Base (already in this project)
pip install torch transformers peft trl

# FSDP: nothing extra — it ships with PyTorch ≥ 1.12
python -c "import torch.distributed; print('FSDP ready')"

# DeepSpeed (optional, for ZeRO-3 + offload benchmarking)
pip install deepspeed
```

Verify multi-GPU visibility:
```bash
python -c "import torch; print(f'{torch.cuda.device_count()} GPUs visible')"
nvidia-smi -L
```

---

## 🧭 Strategy Selection

```
How big is your model (per node)?
│
├── fits on 1 GPU (≤7B bf16 on 80GB)
│   └──► DDP   (simplest, near-linear scaling)
│
├── needs sharding across GPUs (7B–70B)
│   │
│   ├── standard case  ──► FSDP full_shard  ⭐ THE DEFAULT
│   │
│   └── FSDP OOMs / need CPU or NVMe offload
│                         ──► DeepSpeed ZeRO-3 + offload
│
└── >70B or multi-node   ──► FSDP (multi-node) or ZeRO-3 + NVMe offload
```

### TL;DR recommendations

| Your situation | Use this |
|----------------|----------|
| 2-4× A100, Qwen3-7B | **FSDP full_shard** |
| 8× A100, Qwen3-14B | **FSDP full_shard** |
| 4× A100, Qwen3-30B | **FSDP full_shard** (may need ZeRO-3 if OOM) |
| 8× A100, Qwen3-72B | **DeepSpeed ZeRO-3 + CPU offload** |
| DPO (policy + ref model, 2x VRAM) | **FSDP full_shard** (each model sharded) |
| Single 24GB GPU, must use QLoRA | Single-GPU QLoRA (existing `train_sft.py`) |

---

## 🚀 Quick Start: 3 Commands

### 1. Single-GPU baseline (development)
```bash
python scripts/train_sft.py --model-name Qwen/Qwen3-0.6B
```

### 2. FSDP — the standard multi-GPU path
```bash
# 4 GPUs, bf16 full precision (no quantization)
./scripts/launch/train_fsdp.sh 4 Qwen/Qwen3-1.7B
```

### 3. DeepSpeed ZeRO-3 + offload (extreme scale)
```bash
torchrun --nproc_per_node=8 scripts/train_sft_distributed.py \
    --model-name Qwen/Qwen3-72B \
    --distributed-preset zero_stage_3_offload \
    --quantization-bits 0
```

The training loop code is **identical** across all three — only the launcher and strategy flag differ.

---

## 🔬 How It Works

This project's distributed capability is **configuration-driven**. The training loop is the standard HuggingFace `Trainer.train()`, which natively supports FSDP and DeepSpeed. We only inject the strategy into `TrainingArguments`:

```
┌─────────────────────────────────────────────────────────┐
│  torchrun --nproc_per_node=4 scripts/train_sft_distributed.py
│       │       (injects: LOCAL_RANK, RANK, WORLD_SIZE)
│       ▼                                                  │
│  ┌─────────────────────────────────────────────┐        │
│  │ train_sft_distributed.py                    │        │
│  │  - reads --distributed-preset               │        │
│  │  - resolve_distributed_config() → fsdp="..."│        │
│  │  - rank 0 prints banner; others silent      │        │
│  └────────────────────┬────────────────────────┘        │
│                       ▼                                  │
│  ┌─────────────────────────────────────────────┐        │
│  │ SFTTrainer.setup_trainer()                  │        │
│  │  - injects fsdp= + fsdp_config= (or         │        │
│  │    deepspeed=) into TrainingArguments       │        │
│  │  - HF Trainer handles process group +       │        │
│  │    FSDP wrap + all-reduce internally        │        │
│  └────────────────────┬────────────────────────┘        │
│                       ▼                                  │
│  ┌─────────────────────────────────────────────┐        │
│  │ load_model()                                │        │
│  │  - detects distributed mode                 │        │
│  │  - device_map = {"": local_rank}  (NOT auto)│        │
│  │  - FSDP re-shards the model itself after    │        │
│  └─────────────────────────────────────────────┘        │
└─────────────────────────────────────────────────────────┘
```

### The one critical fix: `device_map`

In single-GPU mode, `device_map="auto"` lets HuggingFace place the model on the GPU. **Under FSDP/DDP this breaks** — `auto` would pipeline the model across GPUs (model-parallel), but FSDP needs to load a full copy per rank and then shard it itself. So in distributed mode we pin to the current rank's GPU:

```python
# src/models/loader.py
if dist_info.is_distributed:
    model_kwargs["device_map"] = {"": dist_info.local_rank}  # full copy on my GPU
else:
    model_kwargs["device_map"] = config.device_map  # single-GPU: "auto"
```

This is the single most common source of "why does FSDP crash" bugs, and it's handled automatically here.

---

## 🥇 Why FSDP over DeepSpeed

| Aspect | FSDP | DeepSpeed |
|--------|------|-----------|
| **Dependencies** | Zero (PyTorch built-in) | Separate library + compiled kernels |
| **HF integration** | First-class (`fsdp=` arg) | First-class (`deepspeed=` arg) |
| **torch.compile** | ✅ Compatible | ⚠️ Often conflicts |
| **Used by** | Meta (Llama), HF defaults | Microsoft, many research labs |
| **CPU/NVMe offload** | ✅ (newer PyTorch) | ✅ Mature |
| **Debugging** | Simpler (one library) | More moving parts |

**Bottom line**: in 2026, FSDP is the default unless you specifically need DeepSpeed's mature offload or are following a DeepSpeed-specific recipe. They're algorithmically equivalent at the ZeRO-3 / full_shard level.

---

## 🐘 When You Still Need DeepSpeed

Reach for DeepSpeed ZeRO-3 + offload when:

1. **FSDP OOMs on huge models** — ZeRO-3's NVMe offload can spill to fast SSD, useful for 70B+ on limited GPUs
2. **You need a specific DeepSpeed feature** — e.g., MoE training, sparse attention, Big Science recipes
3. **Following a DeepSpeed-specific paper/recipe** — easier to match configs exactly

```bash
torchrun --nproc_per_node=8 scripts/train_sft_distributed.py \
    --model-name Qwen/Qwen3-72B \
    --distributed-preset zero_stage_3_offload \
    --quantization-bits 0   # bf16 full precision
```

---

## 🪜 Step-by-Step Walkthrough

### Step 1: Verify your GPUs

```bash
python -c "import torch; print(f'{torch.cuda.device_count()} GPUs'); [print(f'  GPU {i}: {torch.cuda.get_device_name(i)}') for i in range(torch.cuda.device_count())]"
```

### Step 2: (Optional) Install DeepSpeed

```bash
pip install deepspeed
ds_report   # verify kernels compiled
```

### Step 3: Launch FSDP

```bash
# Auto-detects GPU count, runs in bf16 full precision
./scripts/launch/train_fsdp.sh 4 Qwen/Qwen3-1.7B

# Or invoke torchrun directly with full control
torchrun --nproc_per_node=4 scripts/train_sft_distributed.py \
    --model-name Qwen/Qwen3-1.7B \
    --distributed-preset fsdp_full \
    --quantization-bits 0 \
    --output-dir ./outputs/my_run
```

### Step 4: What you'll see

```
============================================
  FSDP Training (PyTorch-native, full_shard)
  GPUs: 4  |  Model: Qwen/Qwen3-1.7B
  Precision: bf16 (full, no quantization)
============================================
Loading model: Qwen/Qwen3-1.7B
  Platform: 4× NVIDIA A100 (80.0GB VRAM each)
  Distributed mode: rank 0/4 (local_rank=0)
  Device pinned to GPU 0 (DDP/DeepSpeed expects one full copy per rank)
✓ FSDP enabled: full_shard
Distributed training engaged: world_size=4, strategy=full_shard
{'train_runtime': 1234.5, 'train_samples_per_second': 45.6, ...}
```

Only **rank 0** prints the banner — other ranks stay silent so logs are readable.

---

## 📊 Benchmarking

Run the built-in benchmark to compare strategies:

```bash
./scripts/launch/benchmark_distributed.sh Qwen/Qwen3-1.7B 4
```

This runs the same model/dataset under: single-GPU → DDP → FSDP → ZeRO-2 → ZeRO-3, logging throughput and memory for each. Results land in `outputs/benchmark_<timestamp>/`.

See [`benchmark/README.md`](../benchmark/README.md) for the methodology and how to fill in the results table.

---

## 🐛 Troubleshooting

### "CUDA out of memory" under FSDP/DDP

**Cause**: `device_map="auto"` leaked through (model got pipelined instead of sharded).

**Fix**: Confirm you're launching via `torchrun`, not plain `python`. The loader only pins `device_map` when `WORLD_SIZE > 1`. The launch scripts handle this automatically.

### FSDP: "RuntimeError: You are trying to save a model that has been wrapped with FSDP"

**Cause**: FSDP shards parameters across GPUs; saving needs a gather step.

**Fix**: HF Trainer's `save_model()` handles this when `fsdp="full_shard"` is set. Don't call `model.save_pretrained()` directly — use `trainer.save_model()`, which our SFT/DPO trainers already do.

### NaN losses with FSDP + quantization

**Cause**: FSDP full_shard shards 4-bit bnb params, which can corrupt the NF4 layout.

**Fix**: Use bf16 full precision (`--quantization-bits 0`). This is the standard practice anyway. The launcher warns if you combine full_shard with quantization.

### DeepSpeed: "Cannot find communicator" / missing kernels

**Cause**: DeepSpeed compiled without your CUDA version.

**Fix**:
```bash
DS_BUILD_OPS=1 pip install deepspeed --global-option="build_ext" --global-option="-j8"
```

### "Default process group has not been initialized"

**Cause**: Code called `dist.*` before HF Trainer set up the process group.

**Fix**: Don't manually call `torch.distributed.init_process_group`. HF Trainer does this. Our `setup_distributed()` only pins the device; it doesn't init the group.

### Throughput *worse* on multiple GPUs

**Cause**: Communication overhead exceeds compute savings (small models, PCIe without NVLink, tiny batch).

**Fix**: Increase per-device batch size or reduce GPU count. Models ≤1B rarely benefit from >4 GPUs.

---

## 🚀 Going Further: Multi-node

This guide covers single-node multi-GPU. For multi-node (e.g., 2 servers × 8 GPUs):

```bash
# Node 0 (master)
torchrun --nnodes=2 --nproc_per_node=8 --node_rank=0 \
    --master_addr=<node0_ip> --master_port=29500 \
    scripts/train_sft_distributed.py --distributed-preset fsdp_full --quantization-bits 0

# Node 1
torchrun --nnodes=2 --nproc_per_node=8 --node_rank=1 \
    --master_addr=<node0_ip> --master_port=29500 \
    scripts/train_sft_distributed.py --distributed-preset fsdp_full --quantization-bits 0
```

Requires low-latency interconnect (InfiniBand strongly recommended).

### Custom configs

Pass a custom FSDP mode or DeepSpeed JSON:
```bash
# FSDP sharded_grad_scaled (≈ ZeRO-2 — shards grads + optimizer, not params)
torchrun --nproc_per_node=4 scripts/train_sft_distributed.py \
    --fsdp-mode sharded_grad_scaled --quantization-bits 0

# Custom DeepSpeed JSON
torchrun --nproc_per_node=4 scripts/train_sft_distributed.py \
    --deepspeed-config /path/to/my_zero.json
```

---

## 📚 Further Reading

- [PyTorch FSDP tutorial](https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html) — official
- [HF Docs: FSDP integration](https://huggingface.co/docs/transformers/fsdp) — `fsdp=` and `fsdp_config=`
- [FSDP paper](https://arxiv.org/abs/2304.11277) — how full sharding works
- [DeepSpeed config reference](https://www.deepspeed.ai/docs/config-json/) — for ZeRO stages
- [ZeRO paper](https://arxiv.org/abs/1910.02054) — the algorithm FSDP full_shard implements
