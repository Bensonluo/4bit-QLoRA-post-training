# LLM Post-Training Portfolio Project

> A complete QLoRA framework for fine-tuning LLMs on consumer hardware. Supports NVIDIA GPU, Apple Silicon, and CPU with automatic platform detection. Includes a full ML platform UI for configuration, training, evaluation, and model comparison.

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/pytorch-2.1+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🎯 Overview

This project demonstrates end-to-end LLM post-training using **QLoRA (Quantized Low-Rank Adaptation)** optimized for consumer hardware. It supports **NVIDIA GPU** (4-bit quantization), **Apple Silicon** (bf16 via MPS), and **CPU** with automatic platform detection.

### What's Included

- **🖥️ ML Platform Dashboard** — Streamlit UI covering the full lifecycle: configure → train → monitor → evaluate → compare
- **📊 MLflow Tracking** — Automatic experiment logging, metric history, and run comparison
- **🏥 Domain Adaptation System** — Extensible domain modules with built-in medical entity showcase
- **🤖 Multi-Model Support** — Qwen2.5, Qwen3, Llama 3.2, Phi-3, Gemma 2

### Key Features

- ✅ **Cross-Platform**: Auto-detects NVIDIA GPU (CUDA), Apple Silicon (MPS), or CPU — no config needed
- ✅ **Memory Efficient**: Train 1.5B models on 8GB VRAM (NVIDIA) or run up to ~14B on Apple Silicon 64GB
- ✅ **4-bit QLoRA**: 84% memory reduction with minimal quality loss (NVIDIA only)
- ✅ **Apple Silicon Native**: bf16 training via Metal Performance Shaders
- ✅ **MLflow Tracking**: Automatic experiment logging with metric history and run comparison
- ✅ **Streamlit Dashboard**: Visual UI for training configuration, live monitoring, evaluation, and model comparison
- ✅ **Domain Adaptation**: Extensible domain system with built-in medical entity evaluation
- ✅ **DPO Support**: Direct Preference Optimization for alignment
- ✅ **Qwen3 Ready**: Full support for Qwen3 series (0.6B to 14B)
- ✅ **Production Ready**: Comprehensive tests, logging, monitoring

## 🚀 Quick Start

### Prerequisites

**Hardware (one of):**
- NVIDIA GPU with 8GB+ VRAM (RTX 4060, etc.)
- Apple Silicon Mac with 16GB+ unified memory (M1/M2/M3/M4 Pro/Max/Ultra)
- CPU (for testing/validation only, not recommended for actual training)

**Software:**
- Python 3.10+
- CUDA 11.8+ (NVIDIA only)

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/4bit-QLoRA-post-training.git
cd 4bit-QLoRA-post-training

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install base dependencies
pip install -r requirements.txt

# Install with UI support (MLflow + Streamlit + Plotly)
pip install -e ".[ui]"
```

### Start the Dashboard

```bash
# Launch MLflow + Streamlit dashboard
python scripts/launch_dashboard.py

# Or launch Streamlit only
streamlit run ui/app.py
```

Open http://localhost:8501 to access the dashboard.

### Quick CLI Test

```bash
# Quick training test (5-10 minutes)
python scripts/train_quick_test.py

# Medical entity domain training
python scripts/train_medical_entity.py --poc    # 8GB GPU
python scripts/train_medical_entity.py --mac    # Apple Silicon
```

## 📁 Project Structure

```
4bit-QLoRA-post-training/
├── config/                    # Configuration system
│   ├── base.py              # Model, training, LoRA, logging configs
│   ├── sft.py               # SFT-specific configs + presets
│   ├── dpo.py               # DPO-specific configs + presets
│   ├── models.yaml          # Model registry (VRAM, LoRA modules)
│   └── domains/             # Domain-specific training presets
│       └── medical_entity.py  # Qwen3 medical entity configs
├── src/                       # Source code
│   ├── models/              # Model loading, quantization, merging
│   ├── data/                # Dataset loaders (Alpaca, Finance, Medical, DPO)
│   ├── training/            # Trainers (SFT, Domain, DPO) + callbacks
│   ├── evaluation/          # Metrics, generation, comparison
│   ├── tracking/            # MLflow integration + training runner
│   └── utils/               # Platform detection, logging, memory, remote
├── ui/                        # Streamlit Dashboard
│   ├── app.py               # Entry point (system status, quick actions)
│   ├── config.py            # UI shared config
│   ├── components/          # Reusable charts, filters, domain adapters
│   └── pages/               # 4 dashboard pages
│       ├── 00_Training_Lab.py      # Configure → Launch → Monitor
│       ├── 01_Experiments.py       # Run history + comparison
│       ├── 02_Evaluation.py        # Domain eval visualization
│       └── 03_Model_Comparison.py  # Side-by-side deltas
├── domains/                   # Self-contained domain modules
│   └── medical_entity/      # Data, evaluation, reports
├── scripts/                   # CLI scripts
│   ├── train_quick_test.py  # Quick validation run
│   ├── train_sft.py         # Generic SFT training
│   ├── train_dpo.py         # DPO preference training
│   ├── train_medical_entity.py  # Medical domain training
│   ├── launch_dashboard.py  # Start MLflow + Streamlit
│   ├── merge_lora.py        # Merge adapters into base model
│   └── evaluate.py          # Model evaluation
├── outputs/                   # Training outputs + MLflow tracking (gitignored)
├── notebooks/                 # Educational Jupyter notebooks
├── docs/                      # Theory + tutorials
└── tests/                     # Unit and integration tests
```

## 🖥️ Dashboard

The Streamlit dashboard provides a visual interface for the entire training lifecycle.

### Pages

| Page | Purpose |
|---|---|
| **Training Lab** | Configure hyperparameters, select presets (Quick/Standard/Full), launch training, monitor live loss curves |
| **Experiments** | Browse all MLflow runs, filter by status/model/experiment, compare metrics, view parameter diffs |
| **Evaluation** | Domain-specific evaluation charts (accuracy by difficulty, entity type breakdown, calibration curves) |
| **Model Comparison** | Side-by-side metric comparison with deltas, executive summary, cost estimation |

### Dashboard Quick Start

```bash
pip install -e ".[ui]"
streamlit run ui/app.py
```

- Select a preset (⚡ Quick Test / 🔥 Standard / 🚀 Full Run) in the Training Lab
- Click **Start Training** — the run appears in the Activity tab with live loss curves
- Switch to **Experiments** to see all historical runs
- Run a domain evaluation and view results in **Evaluation**

## 🎓 Learning Outcomes

### Technical Skills
- **QLoRA**: 4-bit quantization + Low-Rank Adaptation
- **Cross-Platform ML**: Platform abstraction layer (CUDA/MPS/CPU)
- **MLOps**: MLflow experiment tracking, Streamlit dashboards
- **Domain Adaptation**: Extensible domain system with evaluation pipelines
- **DPO**: Direct Preference Optimization for alignment

### Production Practices
- Clean, modular code architecture
- Comprehensive testing (unit + integration)
- Documentation-driven development
- Error handling and edge cases

## 📊 Performance

### Hardware Optimization

| Model | VRAM (4-bit, NVIDIA) | Apple Silicon 64GB (bf16) |
|-------|----------------------|---------------------------|
| Qwen 0.5B | ~1.5 GB | ~1 GB |
| Qwen 1.5B | ~2.3 GB | ~3 GB |
| Qwen3 0.6B | ~1.2 GB | ~1 GB |
| Qwen3 1.7B | ~2.0 GB | ~2 GB |
| Qwen3 4B | ~3.5 GB | ~4 GB |
| Llama 3.2 1B | ~1.8 GB | ~2 GB |
| Llama 3.2 3B | ~4.5 GB | ~6 GB |
| Qwen3 8B | ~6.0 GB | ~8 GB |
| Qwen3 14B | OOM (needs 16GB+) | ~14 GB |

### Training Results

**Quick Test (Qwen 1.5B, 100 samples, 1 epoch, MPS)**:
- Training time: ~82 seconds
- Status: Successful

**Medical Entity POC (Qwen3-4B, 4-bit, RTX 4060 8GB)**:
- Training time: ~15-30 min/epoch
- VRAM usage: ~5-6 GB
- Status: Successful

## 🔧 Configuration Examples

### Medical Entity Domain (Qwen3)

```bash
# POC on 8GB GPU
python scripts/train_medical_entity.py --poc

# Full training on 24GB GPU
python scripts/train_medical_entity.py

# Apple Silicon (14B bf16, needs 64GB)
python scripts/train_medical_entity.py --mac
```

### Custom Dataset SFT

```bash
# Add your data
cat > data/custom/my_data.jsonl << 'EOF'
{"instruction": "Your question", "input": "", "output": "Your answer"}
EOF

# Train
python scripts/train_sft.py --train-file data/custom/my_data.jsonl
```

### DPO Preference Training

```bash
# Quick test (100 samples, 1 epoch)
python scripts/train_dpo.py --quick-test

# Finance-specific DPO with auto-filtering
python scripts/train_dpo.py --auto-filter --max-samples 5000
```

## 🏥 Domain Adaptation System

Domains are self-contained modules under `domains/`. Each domain includes:

- `prepare_data.py` — Dataset preparation
- `evaluate.py` — Domain-specific evaluation
- `data/` — Train/validation/test splits
- `eval/` — Custom evaluation logic and reports

### Built-in Domain: Medical Entity

Chinese medical entity matching with difficulty filtering (easy/medium/hard) and entity type breakdown (drug, hospital, etc.).

### Adding a New Domain

1. Create dataset class in `src/data/`
2. Create config in `config/domains/`
3. Create domain directory under `domains/` with `prepare_data.py`, `evaluate.py`, `data/`
4. Register a chart adapter in `ui/components/domain_adapters.py`

## 🐛 Troubleshooting

### Common Issues

**Out of memory on 8GB VRAM**
```bash
# Reduce sequence length or model size
python scripts/train_medical_entity.py --poc  # Uses Qwen3-4B
```

**Model downloads stuck**
```bash
# Use Hugging Face mirror (China)
export HF_ENDPOINT=https://hf-mirror.com
```

**MLflow not connecting**
```bash
# Ensure MLflow is installed
pip install mlflow
```

## 🤝 Contributing

This is a portfolio project, but suggestions welcome!

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📝 License

MIT License - see [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- [Hugging Face](https://huggingface.co/) — Transformers, PEFT, TRL libraries
- [Qwen Team](https://github.com/QwenLM/Qwen) — Qwen2.5 and Qwen3 models
- [Tim Dettmers](https://github.com/TimDettmers/bitsandbytes) — bitsandbytes
- [Microsoft Research](https://www.microsoft.com/en-us/research/) — QLoRA paper
- [Edward Hu](https://github.com/hiyouga/LoRA) — LoRA paper
- [MLflow](https://mlflow.org/) — Experiment tracking
- [Streamlit](https://streamlit.io/) — Dashboard framework

## 📧 Contact

- **GitHub**: [bensonluo](https://github.com/Bensonluo)
- **Email**: luopengllpp@yahoo.com
