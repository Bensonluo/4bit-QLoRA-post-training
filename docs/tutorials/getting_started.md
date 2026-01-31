# Getting Started Guide

Welcome to the 4-bit QLoRA Post-Training Framework! This guide will help you get started with fine-tuning LLMs on consumer hardware.

## Prerequisites

### Hardware

**Minimum Requirements:**
- GPU: NVIDIA RTX 4060 8GB VRAM
- RAM: 16GB system memory
- Storage: 30GB free space

**Recommended:**
- GPU: NVIDIA RTX 4070+ 12GB+ VRAM
- RAM: 32GB system memory
- Storage: 50GB+ SSD

### Software

- Python 3.10+
- Git
- CUDA 11.8+ (or 12.x)
- SSH access (if using remote Windows machine)

## Installation

### 1. Clone Repository

```bash
git clone https://github.com/yourusername/4bit-QLoRA-post-training.git
cd 4bit-QLoRA-post-training
```

### 2. Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate (Linux/Mac)
source venv/bin/activate

# Activate (Windows)
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
# Core dependencies
pip install -r requirements.txt

# Development dependencies (optional)
pip install -r requirements-dev.txt
```

### 4. Verify Installation

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
```

Expected output:
```
PyTorch: 2.1.0+
CUDA: True
```

## Quick Start

### Option 1: Jupyter Notebook (Recommended for Learning)

1. Start Jupyter:
```bash
jupyter notebook
```

2. Open `notebooks/00_setup.ipynb`

3. Run through the notebook to:
   - Test your GPU
   - Load a small model
   - Test generation
   - Verify everything works

### Option 2: Command Line (Production Training)

#### Train on Local Machine

```bash
# Use finance preset (recommended)
python scripts/train_sft.py --finance-mode

# Custom configuration
python scripts/train_sft.py \
    --model-name Qwen/Qwen2.5-1.5B-Instruct \
    --dataset yahma/alpaca-cleaned \
    --epochs 3 \
    --batch-size 1 \
    --gradient-accumulation-steps 8 \
    --output-dir ./outputs/finance-model
```

#### Train on Remote Windows Machine

```bash
# Ensure SSH is configured
ssh windows "echo 'Connected'"

# Train remotely
python scripts/train_remote.py --finance-mode

# Or with custom arguments
python scripts/train_remote.py \
    --script scripts/train_sft.py \
    -- --finance-mode \
    --epochs 3
```

## Your First Training Run

### Step 1: Test Environment

Run the setup notebook:
```bash
jupyter notebook notebooks/00_setup.ipynb
```

This will:
- ✅ Verify GPU availability
- ✅ Check VRAM
- ✅ Test model loading
- ✅ Test generation

### Step 2: Train with Sample Data

Quick test with small dataset:
```bash
python scripts/train_sft.py \
    --dataset yahma/alpaca-cleaned \
    --max-samples 1000 \
    --epochs 1 \
    --output-dir ./outputs/test-run
```

**Expected time:** ~10 minutes on RTX 4060

### Step 3: Full Finance Training

```bash
python scripts/train_sft.py \
    --finance-mode \
    --epochs 3 \
    --output-dir ./outputs/finance-sft
```

**Expected time:** ~2-3 hours on RTX 4060

### Step 4: Merge LoRA Adapters

After training, merge adapters into base model:
```bash
python scripts/merge_lora.py \
    --base-model Qwen/Qwen2.5-1.5B-Instruct \
    --adapter-path ./outputs/finance-sft \
    --output-dir ./outputs/finance-merged
```

### Step 5: Evaluate Model

```bash
python scripts/evaluate.py \
    --model-path ./outputs/finance-merged \
    --dataset yahma/alpaca-cleaned \
    --max-samples 100
```

## Configuration Guide

### Using Presets

The framework includes optimized presets:

```python
from config.sft import FINANCE_SFT_CONFIG

# Use preset directly
from src.training import run_sft_training
run_sft_training(
    model_config=FINANCE_SFT_CONFIG.model,
    training_config=FINANCE_SFT_CONFIG.training,
    lora_config=FINANCE_SFT_CONFIG.lora,
    data_config=FINANCE_SFT_CONFIG.data,
    logging_config=FINANCE_SFT_CONFIG.logging,
)
```

### Custom Configuration

Create `config/my_training.yaml`:
```yaml
model:
  name: "Qwen/Qwen2.5-1.5B-Instruct"
  quantization_bits: 4
  max_length: 1024

training:
  output_dir: "./outputs/my-model"
  num_epochs: 3
  batch_size: 1
  gradient_accumulation_steps: 8
  learning_rate: 2e-4
  gradient_checkpointing: true
  bf16: true

lora:
  r: 16
  lora_alpha: 32
  lora_dropout: 0.05
  target_modules: ["q_proj", "v_proj", "k_proj", "o_proj"]

data:
  dataset_name: "yahma/alpaca-cleaned"
  max_samples: 50000
  validation_split: 0.1

logging:
  use_wandb: false
  use_tensorboard: true
  log_memory: true
```

Then use it:
```bash
python scripts/train_sft.py --config config/my_training.yaml
```

## Using Your Own Data

### 1. Prepare Data

Create `data/custom/my_data.jsonl`:
```json
{"instruction": "What is P/E ratio?", "input": "", "output": "P/E ratio measures..."}
{"instruction": "Explain diversification", "input": "", "output": "Diversification is..."}
```

### 2. Train

```bash
python scripts/train_sft.py \
    --dataset data/custom/my_data.jsonl \
    --epochs 5 \
    --output-dir ./outputs/my-custom-model
```

See `data/custom/README.md` for more details.

## Monitoring Training

### TensorBoard

```bash
# Start TensorBoard
tensorboard --logdir ./outputs/logs

# Open in browser: http://localhost:6006
```

### Weights & Biases

```bash
python scripts/train_sft.py \
    --finance-mode \
    --use-wandb \
    --wandb-project my-finance-llm \
    --wandb-run-name experiment-1
```

## Common Issues

### Issue: Out of Memory

**Solution:**
```bash
# Reduce max length
--max-length 512

# Or reduce batch size
--batch-size 1

# Increase gradient accumulation
--gradient-accumulation-steps 16
```

### Issue: Slow Training

**Solution:**
```bash
# Disable gradient checkpointing (if VRAM allows)
# Edit config or add flags to reduce logging frequency

# Use Flash Attention 2 (enabled by default)
```

### Issue: SSH Connection Fails

**Solution:**
```bash
# Test SSH connection
ssh windows

# If fails, set up SSH keys:
ssh-keygen -t ed25519
ssh-copy-id windows
```

## Project Structure

```
4bit-QLoRA-post-training/
├── config/              # Configuration files
├── src/                 # Source code
│   ├── models/         # Model loading
│   ├── data/           # Dataset loaders
│   ├── training/       # Trainers
│   ├── evaluation/     # Evaluation metrics
│   └── utils/          # Utilities
├── scripts/            # CLI scripts
├── notebooks/          # Jupyter notebooks
├── data/               # Datasets
│   └── custom/         # Your custom data
├── docs/               # Documentation
└── outputs/            # Training outputs
```

## Next Steps

1. **Learn Theory**: Read `docs/theory/sft.md`
2. **Finance Domain**: See `docs/tutorials/finance_training.md`
3. **Advanced Techniques**: Learn DPO in `docs/theory/dpo.md`
4. **Experiment**: Try different models and datasets!

## Getting Help

- **GitHub Issues**: Report bugs and feature requests
- **Documentation**: Check `docs/` folder
- **Examples**: See `notebooks/` for complete examples

## Checklist

Before training:

- [ ] Dependencies installed
- [ ] GPU drivers working (`nvidia-smi`)
- [ ] Enough disk space (30GB+)
- [ ] Dataset prepared
- [ ] Configuration reviewed

After training:

- [ ] Check training loss curves
- [ ] Evaluate on validation set
- [ ] Generate test samples
- [ ] Merge LoRA adapters
- [ ] Save final model

## Resources

- [Hugging Face Transformers](https://huggingface.co/docs/transformers)
- [PEFT Documentation](https://huggingface.co/docs/peft)
- [QLoRA Paper](https://arxiv.org/abs/2305.14314)
- [Project README](../README.md)

Happy training! 🚀
