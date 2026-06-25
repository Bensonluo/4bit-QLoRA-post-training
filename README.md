<div align="center">

# 4-bit QLoRA Post-Training Framework

**Fine-tune LLMs from 0.6B to 14B on consumer hardware. Cross-platform (NVIDIA GPU / Apple Silicon / CPU), with a full ML platform dashboard for config → train → monitor → evaluate.**

[![Live Dashboard](https://img.shields.io/badge/LIVE-DASHBOARD-brightgreen?style=for-the-badge&logo=vercel)](https://benluo.art/qlora-dashboard/)
[![GitHub stars](https://img.shields.io/github/stars/Bensonluo/4bit-QLoRA-post-training?style=for-the-badge)](https://github.com/Bensonluo/4bit-QLoRA-post-training/stargazers)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Transformers-yellow)](https://huggingface.co/)
[![Qwen3](https://img.shields.io/badge/Qwen3-0.6B--14B-6D4AAE)](https://github.com/QwenLM/Qwen)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B)](https://streamlit.io/)

<!-- 🎬 录制说明:用 kap/licecap 录 30 秒 dashboard 操作流程,存到 docs/assets/dashboard.gif -->
<img src="docs/assets/dashboard.gif" alt="QLoRA Dashboard Demo" width="80%">

*🎬 Replace this with a 30s GIF of the dashboard — see [Recording Guide](#-demo-recording-guide) below*

</div>

---

## 📌 Table of Contents

- [Why This Project](#-why-this-project)
- [Key Highlights](#-key-highlights)
- [Supported Models & Hardware](#-supported-models--hardware)
- [Quick Start](#-quick-start)
- [Distributed Training (FSDP / DeepSpeed)](#-distributed-training-fsdp--deepspeed)
- [Model Registry (Lifecycle & Lineage)](#-model-registry-lifecycle--lineage)
- [Dashboard Tour](#-dashboard-tour)
- [Domain Adaptation](#-domain-adaptation)
- [Project Structure](#-project-structure)
- [中文说明](#-中文说明)

---

## 💡 Why This Project

Most QLoRA tutorials assume an A100 and stop at `trainer.train()`. Reality for most practitioners:

- ❌ You have an **RTX 4060 (8GB)** or a **MacBook Pro M2**, not a datacenter GPU
- ❌ You need to actually **compare models**, not just train one and guess if it's better
- ❌ Apple Silicon users are stuck — most QLoRA guides are CUDA-only
- ❌ "Fine-tuning" feels like a black box with no UI to visualize what's happening

This project solves all of them:

> 🔥 **Train Qwen3-4B in 8GB VRAM** with 4-bit QLoRA — or train **Qwen3-14B on Apple Silicon** in bf16 — with a Streamlit dashboard for the entire lifecycle and MLflow for experiment tracking.

It's a **complete MLOps reference** for consumer-hardware LLM post-training: SFT, DPO, domain adaptation, evaluation, and side-by-side model comparison.

---

## ✨ Key Highlights

<div align="center">

| 🚀 Training | 📊 Tracking | 🎯 Evaluation |
|:---:|:---:|:---:|
| SFT + DPO + Domain Adapt | MLflow + **Model Registry** ⭐ | Difficulty-stratified |
| Cross-platform auto-detect | Live loss curves | Multi-model comparison |
| **FSDP + DeepSpeed** ⭐ | Run diff viewer + lineage | Confidence calibration |

| 🍎 Apple Silicon | 🖥️ NVIDIA | 📋 Reporting |
|:---:|:---:|:---:|
| bf16 via MPS | 4-bit QLoRA | Markdown exec summary |
| Up to 14B on 64GB | 84% VRAM savings | Cost estimation |
| Zero-config detect | Multi-GPU scale-out | Deploy recommendations |

| 📈 Stats | | |
|:---:|:---:|:---:|
| **84%** VRAM savings (NVIDIA) | **0.6B–14B** model range | **3** post-training techniques |
| **4** dashboard pages | **5+** model families | **FSDP + DeepSpeed + DDP** distributed |

</div>

### 🧠 What makes it different

1. **True cross-platform** — one codebase, auto-detects CUDA / MPS / CPU, no config flags
2. **Full lifecycle dashboard** — not just training, but experiment management + evaluation + comparison
3. **Domain adaptation system** — pluggable domains with a built-in medical entity showcase (Chinese drug/hospital name normalization)
4. **Honest evaluation** — difficulty-stratified metrics (easy/medium/hard) instead of one aggregate number
5. **Executive summaries** — auto-generated Markdown reports with cost estimation and deployment recommendations

---

## 🖥️ Supported Models & Hardware

### Model Compatibility

| Model | NVIDIA VRAM (4-bit) | Apple Silicon 64GB (bf16) |
|-------|---------------------|---------------------------|
| Qwen3 0.6B | ~1.2 GB | ~1 GB |
| Qwen3 1.7B | ~2.0 GB | ~2 GB |
| Qwen3 4B | ~3.5 GB | ~4 GB |
| Qwen3 8B | ~6.0 GB | ~8 GB |
| Qwen3 14B | ⚠️ Needs 16GB+ | ~14 GB |
| Llama 3.2 1B | ~1.8 GB | ~2 GB |
| Llama 3.2 3B | ~4.5 GB | ~6 GB |
| Qwen 0.5B / 1.5B | ~1.5 / ~2.3 GB | ~1 / ~3 GB |

### Hardware Requirements (any one)

- 🖥️ **NVIDIA GPU** — 8GB+ VRAM (RTX 4060 / 3060 / 4070 sufficient for ≤4B models)
- 🍎 **Apple Silicon** — 16GB+ unified memory (M1/M2/M3/M4 Pro/Max/Ultra)
- 💻 **CPU** — for testing/validation only (not recommended for real training)

---

## 🚀 Quick Start

### Option 1: Dashboard (recommended)

```bash
git clone https://github.com/Bensonluo/4bit-QLoRA-post-training.git
cd 4bit-QLoRA-post-training

python -m venv venv && source venv/bin/activate
pip install -e ".[ui]"        # Installs MLflow + Streamlit + Plotly

# Launch MLflow + Streamlit dashboard
python scripts/launch_dashboard.py
```

Open http://localhost:8501 → pick a preset → click **Start Training**.

### Option 2: Try the Live Dashboard

Don't want to install? **[Try the dashboard online →](https://benluo.art/qlora-dashboard/)**

### Option 3: CLI training

```bash
# Quick validation run (5–10 min)
python scripts/train_quick_test.py

# Medical entity domain training
python scripts/train_medical_entity.py --poc    # 8GB NVIDIA GPU
python scripts/train_medical_entity.py --mac    # Apple Silicon
python scripts/train_medical_entity.py          # 24GB GPU (full)

# Custom dataset SFT
python scripts/train_sft.py --train-file data/custom/my_data.jsonl

# DPO preference optimization
python scripts/train_dpo.py --quick-test
```

> 💡 **In China?** Set HF mirror to avoid download timeouts:
> ```bash
> export HF_ENDPOINT=https://hf-mirror.com
> ```

---

## 🌐 Distributed Training (FSDP / DeepSpeed)

Scale from a single GPU to multi-GPU with **zero training-loop changes**, using the **industry-standard 2026 stack**: FSDP (PyTorch-native) in bf16 full precision, with DeepSpeed for scale-beyond.

### One-liner launches

```bash
# FSDP — the PyTorch-native DEFAULT for multi-GPU LLM training (what Meta uses for Llama)
./scripts/launch/train_fsdp.sh 4 Qwen/Qwen3-1.7B

# DDP — plain replication, the simple baseline
./scripts/launch/train_ddp.sh 4 Qwen/Qwen3-1.7B

# DeepSpeed ZeRO-3 + offload — extreme scale (70B+, CPU/NVMe offload)
torchrun --nproc_per_node=8 scripts/train_sft_distributed.py \
    --model-name Qwen/Qwen3-72B --distributed-preset zero_stage_3_offload --quantization-bits 0
```

### Strategy at a glance

| Strategy | Shards | When to use |
|----------|--------|-------------|
| **FSDP full_shard** ⭐ | params + grads + optimizer | **DEFAULT** — standard multi-GPU, PyTorch-native |
| FSDP sharded_grad_scaled | grads + optimizer only | Lighter sharding (≈ ZeRO-2) |
| DDP | Nothing (full copy/GPU) | Simple baseline; model fits on one GPU |
| DeepSpeed ZeRO-2 | optimizer + gradients | Same idea as FSDP, via DeepSpeed |
| DeepSpeed ZeRO-3 + offload | everything (+ CPU/NVMe) | Extreme scale beyond FSDP |

> **2026 practice**: run FSDP in **bf16 full precision** (`--quantization-bits 0`). Full-parameter sharding (FSDP full_shard / ZeRO-3) is the standard; QLoRA is reserved for genuinely memory-tight single-GPU scenarios. The launcher warns if you combine full sharding with 4-bit quantization.

### Benchmark it

```bash
./scripts/launch/benchmark_distributed.sh Qwen/Qwen3-1.7B 4
```

Runs single-GPU → DDP → FSDP → ZeRO-2 → ZeRO-3 and logs throughput + memory. Paste (sanitized) results into [`benchmark/README.md`](benchmark/README.md).

📖 **Full guide**: [`docs/distributed_training_guide.md`](docs/distributed_training_guide.md) — FSDP vs DeepSpeed, strategy selection, QLoRA caveats, multi-node setup, troubleshooting.

---

## 🗂️ Model Registry (Lifecycle & Lineage)

Close the loop after fine-tuning: **merge → register → stage → trace**. Every registered model version links back to the exact training run that produced it (params + metrics), so you always know which model is in Production and why.

### Automatic registration

Flip two config flags and the trainer does the rest:

```python
LoggingConfig(
    use_mlflow=True,                # tracking must be on
    register_model=True,            # 🆕 auto-register after training
    registry_model_name="Qwen3-1.7B-QLoRA",
    merge_before_register=True,     # merge LoRA into base before logging
    registry_stage="Staging",
)
```

After `save_model()`, the trainer automatically: (1) merges the adapter into the base, (2) logs the merged model to MLflow, (3) registers it as a new version, (4) stages it. Registration failures never fail the training run.

### Manual registration (no retraining)

```bash
# Merge a previously-trained adapter
python scripts/merge_adapter.py \
    --adapter-dir outputs/sft/run-xxx \
    --output-dir outputs/merged/run-xxx

# Register it
python scripts/registry_cli.py register \
    --model-dir outputs/merged/run-xxx \
    --name Qwen3-1.7B-QLoRA
```

### Manage the lifecycle

```bash
# List all versions + stages
python scripts/registry_cli.py list

# Promote to Production
python scripts/registry_cli.py transition \
    --model-name Qwen3-1.7B-QLoRA --version 3 --stage Production

# Trace a version back to its training (params + metrics)
python scripts/registry_cli.py info \
    --model-name Qwen3-1.7B-QLoRA --version 3
```

📖 **Full guide**: [`docs/model_registry_guide.md`](docs/model_registry_guide.md) — lifecycle, lineage, troubleshooting. Includes 3 tracking bug fixes (DPO callback mount, double-write removal, lineage link).

---

## 📊 Dashboard Tour

Four pages covering the full ML lifecycle:

| Page | What you do there |
|------|-------------------|
| 🧪 **Training Lab** | Pick preset (⚡ Quick / 🔥 Standard / 🚀 Full) → configure hyperparams → launch → watch live loss curves |
| 📈 **Experiments** | Browse all MLflow runs, filter by status/model, compare params, view metric diffs |
| 🎯 **Evaluation** | Domain-specific charts: accuracy by difficulty, entity type breakdown, calibration curves |
| ⚖️ **Model Comparison** | Side-by-side metric deltas, auto-generated executive summary, cost estimation |

```bash
pip install -e ".[ui]"
streamlit run ui/app.py
```

---

## 🏥 Domain Adaptation

Domains are self-contained modules under `domains/`:

```
domains/medical_entity/
├── prepare_data.py    # Dataset preparation
├── evaluate.py        # Domain-specific evaluation
├── data/              # Train/val/test splits
└── eval/              # Custom eval logic + reports
```

### Built-in: Chinese Medical Entity Matching

Fine-tune Qwen3 to normalize drug names and hospital names — with **difficulty-stratified evaluation** (easy/medium/hard) and entity-type breakdown (drug, hospital, etc.).

### Adding your own domain

1. Add a dataset class in `src/data/`
2. Add config in `config/domains/`
3. Create `domains/your_domain/` with `prepare_data.py`, `evaluate.py`, `data/`
4. Register a chart adapter in `ui/components/domain_adapters.py`

---

## 📁 Project Structure

```
4bit-QLoRA-post-training/
├── config/                 # Configs + presets + model registry
│   ├── base.py             # Model / training / LoRA / logging configs
│   ├── sft.py  dpo.py      # Technique-specific presets
│   ├── models.yaml         # VRAM table + LoRA target modules
│   └── domains/            # Domain training presets
├── src/
│   ├── models/             # Loading, quantization, merging
│   ├── data/               # Alpaca / Finance / Medical / DPO loaders
│   ├── training/           # SFT + Domain + DPO trainers + callbacks
│   ├── evaluation/         # Metrics, generation, comparison
│   ├── tracking/           # MLflow integration + runner
│   └── utils/              # Platform detection, logging, memory
├── ui/                     # Streamlit dashboard
│   ├── app.py              # Entry point
│   ├── components/         # Reusable charts, filters, adapters
│   └── pages/              # 4 dashboard pages
├── domains/                # Self-contained domain modules
├── scripts/                # CLI entry points (train/eval/merge)
├── notebooks/              # Educational Jupyter notebooks
├── docs/                   # Theory + tutorials
└── tests/                  # Unit + integration tests
```

---

## 🛠️ Configuration Examples

### Training Presets

```python
# config/sft.py
QUICK_TEST = TrainingConfig(
    model_name="Qwen/Qwen3-0.6B",
    lora_r=16, lora_alpha=32,
    max_samples=100, num_epochs=1,
)

STANDARD = TrainingConfig(
    model_name="Qwen/Qwen3-4B",
    lora_r=32, lora_alpha=64,
    max_samples=5000, num_epochs=3,
)
```

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `HF_ENDPOINT` | Hugging Face mirror (China) | `https://huggingface.co` |
| `CUDA_VISIBLE_DEVICES` | GPU selection | `0` |
| `MLFLOW_TRACKING_URI` | MLflow server | `file:./outputs/mlruns` |

---

## 🐛 Troubleshooting

<details>
<summary><b>Common issues</b></summary>

**Out of memory on 8GB VRAM**
```bash
python scripts/train_medical_entity.py --poc  # Uses Qwen3-4B with reduced seq length
```

**Model downloads stuck (China)**
```bash
export HF_ENDPOINT=https://hf-mirror.com
```

**Apple Silicon slower than expected**
- Check Activity Monitor → GPU History
- Ensure MPS is available: `python -c "import torch; print(torch.backends.mps.is_available())"`
- Use bf16 (default); fp32 will be ~3× slower

**MLflow UI not loading**
```bash
pip install mlflow
python scripts/launch_dashboard.py  # Starts both MLflow + Streamlit
```

</details>

---

## 🗺️ Roadmap

- [x] Cross-platform training (NVIDIA / Apple Silicon / CPU)
- [x] SFT + DPO + Domain Adaptation
- [x] Streamlit dashboard (4 pages)
- [x] MLflow experiment tracking
- [x] Medical entity domain showcase
- [x] Difficulty-stratified evaluation
- [ ] GRPO (Group Relative Policy Optimization)
- [ ] vLLM deployment integration
- [ ] More domains: legal, finance, code

---

## 🤝 Contributing

Portfolio project, but PRs welcome — especially:
- 🎯 New domain adapters (legal, finance, code, etc.)
- 🍎 Apple Silicon performance optimizations
- 📊 New evaluation metrics or visualizations
- 🐛 Bug fixes with a failing test

---

## 📜 License

[MIT](LICENSE) — free for personal and commercial use.

If this project helped you fine-tune on budget hardware, please ⭐ star the repo.

---

## 📬 Contact

- 💼 **Portfolio**: [benluo.art](https://benluo.art)
- 🐙 **GitHub**: [@Bensonluo](https://github.com/Bensonluo)
- 💬 **Issues**: [GitHub Issues](https://github.com/Bensonluo/4bit-QLoRA-post-training/issues)

---

## 🇨🇳 中文说明

**LLM 后训练框架** — 在消费级硬件上微调 0.6B–14B 大模型。

### 核心亮点

- **跨平台训练**:自动检测 NVIDIA GPU(4-bit QLoRA)/ Apple Silicon(bf16 MPS)/ CPU
- **三种后训练技术**:SFT(监督微调)、DPO(直接偏好优化)、领域适配
- **Streamlit 全生命周期面板**:配置 → 训练 → 监控 → 评估 → 对比,4 个页面
- **MLflow 实验追踪**:自动记录指标、参数对比、运行历史
- **领域适配系统**:内置医疗实体匹配示范(中文药品名/医院名归一化)
- **难度分层评测**:简单/中等/困难三档,带置信度校准
- **执行摘要自动生成**:Markdown 报告 + 成本估算 + 部署建议
- **Qwen3 全系列支持**:0.6B / 1.7B / 4B / 8B / 14B,LoRA r=16–64

### 快速开始

```bash
git clone https://github.com/Bensonluo/4bit-QLoRA-post-training.git
cd 4bit-QLoRA-post-training
pip install -e ".[ui]"
python scripts/launch_dashboard.py
# 打开 http://localhost:8501
```

### 显存参考

| 模型 | NVIDIA 4-bit | Apple Silicon 64GB |
|------|-------------|--------------------|
| Qwen3-4B | ~3.5 GB | ~4 GB |
| Qwen3-8B | ~6.0 GB | ~8 GB |
| Qwen3-14B | 需 16GB+ | ~14 GB |

> 💡 国内用户加镜像:`export HF_ENDPOINT=https://hf-mirror.com`

---

<details>
<summary>🎬 Demo Recording Guide (for maintainers)</summary>

### How to record the hero GIF

1. **Tool**: [Kap](https://getkap.co/) (Mac) or [licecap](https://www.cockos.com/licecap/) (cross-platform)
2. **Content** (~30s):
   - 0-5s: Open dashboard, show system status (auto-detected platform)
   - 5-15s: Pick a preset in Training Lab, click Start Training
   - 15-20s: Show live loss curve updating
   - 20-30s: Switch to Model Comparison, show side-by-side deltas
3. **Save to**: `docs/assets/dashboard.gif` (keep under 5MB)
4. **Update**: Replace the placeholder `<img>` in the hero section

</details>

<!--
RECORDING_TODO:
1. Record dashboard.gif → docs/assets/dashboard.gif
2. Replace placeholder img tag in hero section
3. Verify Live Dashboard URL (benluo.art/qlora-dashboard/) returns 200
-->
