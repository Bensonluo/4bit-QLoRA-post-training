# LLM Post-Training Portfolio Project

> A complete 4-bit QLoRA framework for fine-tuning LLMs on consumer GPUs, with finance domain specialization and Mac+Windows distributed training workflow.

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/pytorch-2.1+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🎯 Overview

This project demonstrates end-to-end LLM post-training using **4-bit QLoRA (Quantized Low-Rank Adaptation)** optimized for consumer GPUs. It implements multiple fine-tuning techniques (SFT, Domain Adaptation, DPO) and showcases production-ready ML engineering practices.

### Key Features

- ✅ **Memory Efficient**: Train 1.5B parameter models on 8GB VRAM (RTX 4060)
- ✅ **4-bit QLoRA**: 84% memory reduction with minimal quality loss
- ✅ **Finance Specialization**: Domain-adapted models for financial/investment tasks
- ✅ **DPO Support**: Direct Preference Optimization for alignment
- ✅ **Distributed Workflow**: Mac development + Windows GPU training
- ✅ **Production Ready**: Comprehensive tests, logging, monitoring
- ✅ **China Optimized**: Hugging Face mirror integration for fast downloads
- ✅ **Extensible Architecture**: Easy to add new techniques (continued pre-training, KTO, ORPO)

## 🚀 Quick Start

### Prerequisites

**Minimum Hardware:**
- GPU: NVIDIA RTX 4060 8GB VRAM (or similar)
- RAM: 16GB system memory
- Storage: 30GB free space

**Software:**
- Python 3.10+
- CUDA 11.8+ or 12.x
- SSH access (if using remote Windows machine)

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/4bit-QLoRA-post-training.git
cd 4bit-QLoRA-post-training

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Quick Test

```python
# Quick training test (10-15 minutes)
python scripts/train_quick_test.py
```

### Full Training

```python
# Full finance domain training (2-3 hours)
python scripts/train_finance_full.py
```

## 📁 Project Structure

```
4bit-QLoRA-post-training/
├── config/                    # Configuration files
│   ├── base.py              # Model, training, LoRA configs
│   ├── sft.py               # SFT-specific configs
│   ├── models.yaml          # Available models
│   └── windows.yaml         # Windows-specific config
├── src/                       # Source code
│   ├── models/              # Model loading, quantization
│   ├── data/                # Dataset loaders, preprocessing
│   ├── training/            # Trainers (SFT, Domain, DPO)
│   ├── evaluation/          # Metrics, generation, comparison
│   └── utils/               # Utilities (logging, memory, remote execution)
├── scripts/                  # CLI scripts
│   ├── train_quick_test.py  # Quick test (100 samples)
│   ├── train_finance_full.py # Full finance training (50K samples)
│   ├── train_sft.py         # Generic SFT training
│   ├── train_dpo.py         # DPO preference training
│   ├── train_remote.py      # Execute training on remote Windows
│   ├── merge_lora.py        # Merge LoRA adapters
│   ├── evaluate.py          # Model evaluation
│   └── download_data.py     # Dataset downloader
├── notebooks/                # Educational Jupyter notebooks
│   └── 00_setup.ipynb       # Environment setup and testing
├── docs/                     # Documentation
│   ├── theory/              # SFT, QLoRA, DPO theory
│   └── tutorials/           # Getting started, finance guide, Windows setup
├── data/custom/              # Custom datasets
│   └── examples/            # Example finance QA pairs
├── outputs/                  # Training outputs (gitignored)
│   ├── test-quick/          # Quick test results
│   └── finance-sft/         # Full training results
└── tests/                    # Unit and integration tests
```

## 🎓 Learning Outcomes

This project demonstrates mastery of:

### Technical Skills
- **QLoRA**: 4-bit quantization + Low-Rank Adaptation
- **SFT**: Supervised Fine-Tuning with instruction datasets
- **Domain Adaptation**: Specializing models for specific domains
- **DPO**: Direct Preference Optimization for alignment
- **MLOps**: Training pipelines, logging, checkpointing
- **Distributed Training**: Mac dev + Windows GPU workflow

### Theoretical Understanding
- Cross-entropy loss optimization
- LoRA mathematics (rank factorization)
- 4-bit NF4 quantization
- DPO preference optimization and alignment
- Gradient checkpointing and memory optimization

### Production Practices
- Clean, modular code architecture
- Comprehensive testing (unit + integration)
- CI/CD with GitHub Actions
- Documentation-driven development
- Error handling and edge cases

## 📊 Performance

### Hardware Optimization

| Model | VRAM (4-bit) | RTX 4060 8GB |
|-------|--------------|---------------|
| Qwen 0.5B | ~1.5 GB | ✅ Perfect |
| Qwen 1.5B | ~2.3 GB | ✅ Perfect |
| Llama 3.2 3B | ~4.5 GB | ✅ Good |
| Phi-3 3.8B | ~5.5 GB | ⚠️ Tight |

### Training Results

**Quick Test (Qwen 0.5B, 10 samples, 1 epoch)**:
- Training time: 14 seconds
- Train loss: 1.773 → 1.901 (eval)
- VRAM usage: ~2.2 GB
- Status: ✅ Successful

## 🛠️ Technologies Used

### Core Framework
- **transformers** (Hugging Face) - Model loading
- **peft** - LoRA implementation
- **bitsandbytes** - 4-bit quantization
- **trl** - SFT and DPO trainers
- **accelerate** - Multi-GPU/CPU offloading

### Training
- **torch** - Deep learning framework
- **datasets** - Dataset management
- **wandb/tensorboard** - Experiment tracking
- **rich** - Beautiful console output

### Development
- **pytest** - Testing framework
- **ruff** - Linting and formatting
- **mypy** - Type checking
- **pre-commit** - Git hooks
- **jupyter** - Interactive notebooks

## 📖 Documentation

- [Getting Started](docs/tutorials/getting_started.md) - Quick start guide
- [Windows Setup](docs/tutorials/windows_training_setup.md) - Windows machine setup
- [Finance Training](docs/tutorials/finance_training.md) - Domain specialization guide
- [SFT Theory](docs/theory/sft.md) - Supervised fine-tuning theory
- [QLoRA Theory](docs/theory/qlora.md) - Quantization and LoRA deep dive
- [DPO Theory](docs/theory/dpo.md) - Direct Preference Optimization guide
- [Training Log](TRAINING_LOG.md) - Experiment tracking

## 🎯 Use Cases

### 1. Instruction Tuning
Teach models to follow instructions using Alpaca-format datasets.

### 2. Domain Adaptation
Specialize models for specific domains (finance, medical, code, etc.).

### 3. Custom Datasets
Fine-tune on your own data with easy-to-use format.

## 🔧 Configuration Examples

### Finance Domain Training

```python
from config.sft import FINANCE_SFT_CONFIG

# Optimized for RTX 4060 8GB
model_config = FINANCE_SFT_CONFIG.model  # Qwen 1.5B, 4-bit
training_config = FINANCE_SFT_CONFIG.training  # BS=1, GA=8
lora_config = FINANCE_SFT_CONFIG.lora  # r=16, alpha=32
data_config = FINANCE_SFT_CONFIG.data  # 50K finance samples
```

### Custom Dataset

```bash
# Add your data
cat > data/custom/my_data.jsonl << EOF
{"instruction": "Your question", "input": "", "output": "Your answer"}
EOF

# Train
python scripts/train_sft.py --train-file data/custom/my_data.jsonl
```

### DPO Preference Training

```bash
# Quick test (100 samples, 1 epoch)
python scripts/train_dpo.py --quick-test

# Full DPO training with default config
python scripts/train_dpo.py

# Finance-specific DPO with auto-filtering
python scripts/train_dpo.py --auto-filter --max-samples 5000

# Custom preference dataset
python scripts/train_dpo.py --dataset-name path/to/preferences.jsonl --beta 0.2
```

## 📈 Results

### Domain Adaptation (Finance)

After training on finance-filtered data, the model can:
- Explain financial concepts (P/E ratio, diversification, etc.)
- Analyze investment strategies
- Discuss risk management
- Provide financial education

**Example Output:**
```
Q: What is dollar-cost averaging?
A: Dollar-cost averaging (DCA) is an investment strategy that involves
investing a fixed amount of money at regular intervals, regardless of
market conditions. This approach reduces the impact of volatility...
```

## 🐛 Troubleshooting

### Common Issues

**Problem**: Out of memory on 8GB VRAM
```bash
# Solution: Reduce sequence length
--max-length 512  # Instead of 1024
```

**Problem**: Training too slow
```bash
# Solution: Adjust batch size and gradient accumulation
--batch-size 2 --gradient-accumulation-steps 4
```

**Problem**: Model downloads stuck
```bash
# Solution: Use Hugging Face mirror (China)
export HF_ENDPOINT=https://hf-mirror.com
```

See [docs/tutorials/getting_started.md](docs/tutorials/getting_started.md) for more troubleshooting.

## 🤝 Contributing

This is a portfolio project, but suggestions welcome!

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📝 License

MIT License - see [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- [Hugging Face](https://huggingface.co/) - Transformers, PEFT, TRL libraries
- [Qwen Team](https://github.com/QwenLM/Qwen) - Qwen2.5 model
- [Tim Dettmers](https://github.com/TimDettmers/bitsandbytes) - bitsandbytes
- [Microsoft Research](https://www.microsoft.com/en-us/research/) - QLoRA paper
- [Edward Hu](https://github.com/hiyouga/LoRA) - LoRA paper

## 📧 Contact

- **GitHub**: [bensonluo](https://github.com/Bensonluo)
- **Email**: luopengllpp@yahoo.com
