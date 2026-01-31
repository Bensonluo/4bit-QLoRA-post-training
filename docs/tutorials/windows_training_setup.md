# Windows Training Setup Guide

This guide explains how to set up and use a Windows machine for LLM training while keeping your main development on Mac.

## Architecture

```
Mac (Development)              Windows (Training)
├─ Code editing                ├─ GPU computation (RTX 4060)
├─ Data processing             ├─ Model training
├─ Experimentation              └─ Results storage
├─ Git tracking
└─ SSH orchestration
```

## Prerequisites

### Windows Machine
- **GPU**: NVIDIA RTX 4060 (8GB VRAM)
- **OS**: Windows 11 with WSL2
- **Python**: 3.10+ (installed in WSL2)
- **Drivers**: NVIDIA drivers installed

### Mac Machine
- **Git**: For version control
- **SSH**: For remote command execution
- **Python**: Optional (for local testing)

## Setup Instructions

### 1. Install NVIDIA Drivers (Windows)

**IMPORTANT**: Do this on Windows PowerShell, NOT in WSL.

```powershell
# Check if drivers installed
nvidia-smi

# If error, download from:
# https://www.nvidia.com/Download/index.aspx
# Select: RTX 4060, Windows 11
```

### 2. Install CUDA Toolkit (WSL2)

```bash
# SSH to Windows
ssh windows

# Verify in WSL2
wsl nvidia-smi
# Should show RTX 4060
```

### 3. Create Python Virtual Environment

```bash
# SSH to Windows
ssh windows

cd ~/4bit-QLoRA-post-training

# Create virtual environment
python3 -m venv venv

# Activate
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip
```

### 4. Install PyTorch with CUDA

```bash
# With venv activated
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Verify
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
# Should print: CUDA: True
```

### 5. Install Project Dependencies

```bash
# With venv activated
pip install -r requirements.txt
```

### 6. Configure Hugging Face Mirror (China)

**IMPORTANT**: Required for fast model downloads in China.

```bash
# Add to ~/.bashrc for automatic activation
echo 'export HF_ENDPOINT=https://hf-mirror.com' >> ~/.bashrc
source ~/.bashrc

# Or set manually each session
export HF_ENDPOINT=https://hf-mirror.com
```

### 7. Verify Setup

```bash
# Run verification
python << 'EOF'
import torch
print(f'✅ PyTorch {torch.__version__}')
print(f'✅ CUDA {torch.cuda.is_available()}')
print(f'✅ GPU {torch.cuda.get_device_name(0)}')

from transformers import AutoModelForCausalLM
from peft import LoraConfig
import bitsandbytes
from trl import SFTTrainer
print('✅ All dependencies OK')

from config.sft import FINANCE_SFT_CONFIG
print(f'✅ Config: {FINANCE_SFT_CONFIG.model.name}')
EOF
```

## Usage

### Option 1: Quick Test (Recommended First)

```bash
# SSH to Windows
ssh windows

cd ~/4bit-QLoRA-post-training
source venv/bin/activate

# Run quick test (10-15 minutes)
python scripts/train_quick_test.py
```

### Option 2: Full Training

```bash
# SSH to Windows
ssh windows

cd ~/4bit-QLoRA-post-training
source venv/bin/activate

# Run full finance training (2-3 hours)
python scripts/train_finance_full.py
```

### Option 3: Background Training with Logging

```bash
# SSH to Windows
ssh windows

cd ~/4bit-QLoRA-post-training
source venv/bin/activate

# Start in background
nohup python scripts/train_finance_full.py > ~/training.log 2>&1 &

# Monitor logs
tail -f ~/training.log
```

### Option 4: From Mac (Recommended)

```bash
# From Mac terminal
cd ~/4bit-QLoRA-post-training

# Execute training on Windows via SSH
python scripts/train_remote.py --finance-mode

# Or sync and run
rsync -avz . windows:~/4bit-QLoRA-post-training/
ssh windows "cd ~/4bit-QLoRA-post-training && source venv/bin/activate && python scripts/train_finance_full.py"
```

## Syncing Work

### From Mac to Windows (Before Training)

```bash
# Sync all code changes
rsync -avz --progress \
    --exclude 'venv/' \
    --exclude '__pycache__/' \
    --exclude '*.pyc' \
    --exclude '.git/' \
    --exclude 'outputs/' \
    . windows:~/4bit-QLoRA-post-training/
```

### From Windows to Mac (After Training)

```bash
# Sync trained models back
rsync -avz windows:~/4bit-QLoRA-post-training/outputs/ ./outputs/

# Sync logs
rsync -avz windows:~/4bit-QLoRA-post-training/*.log ./logs/
```

## Important Notes

### Flash Attention 2

**NOT REQUIRED**: Flash Attention 2 is not installed. Always use:
```python
use_flash_attention=False
```

### Hugging Face Mirror

**REQUIRED IN CHINA**: Always set HF_ENDPOINT:
```bash
export HF_ENDPOINT=https://hf-mirror.com
```

### Virtual Environment

**REQUIRED**: Always activate venv before training:
```bash
cd ~/4bit-QLoRA-post-training
source venv/bin/activate
```

## Troubleshooting

### Issue: nvidia-smi not found

**Solution**: Install NVIDIA drivers on Windows (NOT WSL)
1. Open PowerShell on Windows
2. Download from NVIDIA website
3. Install and restart Windows

### Issue: CUDA not available in PyTorch

**Solution**: Reinstall PyTorch
```bash
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Issue: Out of Memory

**Solution**: Reduce batch size or sequence length
```python
batch_size=1
gradient_accumulation_steps=16
max_length=512  # Reduce from 1024
```

### Issue: Flash Attention Error

**Solution**: Disable Flash Attention (already done in scripts)
```python
use_flash_attention=False
```

## Training Configuration

### Optimized for RTX 4060 8GB:

```python
Model: Qwen/Qwen2.5-1.5B-Instruct
├─ 4-bit quantization (NF4)
├─ VRAM usage: ~2.3 GB
└─ Remaining: ~5.7 GB

Training:
├─ Batch size: 1
├─ Gradient accumulation: 8
├─ Effective batch: 8
├─ Gradient checkpointing: True
└─ Mixed precision: BF16

LoRA:
├─ Rank: 16
├─ Alpha: 32
└─ Target: q_proj, v_proj (minimal)
```

## Monitoring

### Check GPU Usage

```bash
# In WSL2
watch -n 1 nvidia-smi
```

### Check Training Progress

```bash
# View logs
tail -f ~/training.log

# Or use TensorBoard
tensorboard --logdir ~/4bit-QLoRA-post-training/outputs/logs
```

## File Locations

### Windows Machine
- Project: `~/4bit-QLoRA-post-training/`
- Models: `~/.cache/huggingface/hub/`
- Outputs: `~/4bit-QLoRA-post-training/outputs/`
- Logs: `~/4bit-QLoRA-post-training/outputs/logs/`

### Mac Machine
- Project: `~/Documents/GitHub/4bit-QLoRA-post-training/`
- Outputs (synced): `./outputs/`

## Success Indicators

You'll know everything works when:

1. ✅ `nvidia-smi` shows RTX 4060
2. ✅ `torch.cuda.is_available()` returns `True`
3. ✅ Model downloads quickly (with HF mirror)
4. ✅ Quick test completes successfully
5. ✅ Training loss decreases
6. ✅ Checkpoints are saved

## Next Steps

After successful training:

1. **Merge LoRA adapters**:
   ```bash
   python scripts/merge_lora.py \
       --base-model Qwen/Qwen2.5-1.5B-Instruct \
       --adapter-path ./outputs/finance-sft \
       --output-dir ./outputs/finance-merged
   ```

2. **Evaluate model**:
   ```bash
   python scripts/evaluate.py \
       --model-path ./outputs/finance-merged
   ```

3. **Sync results back to Mac**:
   ```bash
   rsync -avz windows:~/4bit-QLoRA-post-training/outputs/ ./outputs/
   ```

4. **Commit to Git**:
   ```bash
   git add .
   git commit -m "Add finance training results"
   git push
   ```
