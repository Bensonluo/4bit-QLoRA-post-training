# Windows Configuration

This file contains Windows-specific configuration for remote training.

## Windows Machine Configuration

### Hardware
- **GPU**: NVIDIA GeForce RTX 4060 Laptop GPU
- **VRAM**: 8 GB
- **RAM**: System memory in Windows machine

### Software
- **OS**: Windows 11 with WSL2
- **Python**: 3.12.3
- **PyTorch**: 2.5.1+cu121
- **CUDA**: Available via WSL2

### SSH Access
- **Hostname**: windows
- **User**: dministrator
- **Project Path**: ~/4bit-QLoRA-post-training

## Important Configuration Notes

### 1. Hugging Face Mirror (Required in China)

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

**Why**: Hugging Face is blocked/throttled in China. The mirror provides fast downloads.

**How to set permanently**:
```bash
echo 'export HF_ENDPOINT=https://hf-mirror.com' >> ~/.bashrc
source ~/.bashrc
```

### 2. Flash Attention 2

**Status**: Not installed

**Impact**: Minimal - training works fine without it

**Configuration**:
```python
use_flash_attention=False
```

### 3. Virtual Environment

**Location**: ~/4bit-QLoRA-post-training/venv

**Activation**:
```bash
cd ~/4bit-QLoRA-post-training
source venv/bin/activate
```

**Auto-activation** (optional):
```bash
echo 'source ~/4bit-QLoRA-post-training/venv/bin/activate' >> ~/.bashrc
```

## Training Configuration

### Optimized Settings for RTX 4060 8GB

```yaml
Model: Qwen/Qwen2.5-1.5B-Instruct
  quantization_bits: 4
  use_flash_attention: false
  max_length: 1024

Training:
  batch_size: 1
  gradient_accumulation_steps: 8
  gradient_checkpointing: true
  bf16: true
  learning_rate: 2e-4
  num_epochs: 3

LoRA:
  r: 16
  lora_alpha: 32
  lora_dropout: 0.05
  target_modules: ["q_proj", "v_proj"]

Expected VRAM: ~2.3 GB
Remaining: ~5.7 GB
```

## Common Commands

### SSH Access
```bash
ssh windows
```

### Monitor Training
```bash
# Check process
ps aux | grep train_sft.py

# Check logs
tail -f ~/training_output.log

# Check GPU (if available)
nvidia-smi
```

### Sync Files
```bash
# From Mac to Windows
rsync -avz --progress \
    --exclude 'venv/' \
    --exclude '__pycache__/' \
    . windows:~/4bit-QLoRA-post-training/

# From Windows to Mac
rsync -avz windows:~/4bit-QLoRA-post-training/outputs/ ./outputs/
```

## Troubleshooting

### Model Download Issues

**Problem**: Downloads are very slow or timeout

**Solution**: Ensure HF_ENDPOINT is set
```bash
echo $HF_ENDPOINT
# Should show: https://hf-mirror.com
```

### GPU Not Detected

**Problem**: torch.cuda.is_available() returns False

**Solution**: Install NVIDIA drivers on Windows (PowerShell, not WSL)

### Out of Memory

**Problem**: Training crashes with OOM error

**Solution**: Reduce batch size or sequence length
```python
batch_size = 1  # Already minimal
max_length = 512  # Reduce from 1024
gradient_accumulation_steps = 16  # Increase from 8
```

## Performance Expectations

### Quick Test (Qwen 0.5B)
- Download: 30-40 seconds (first time)
- Training: 10-15 minutes
- VRAM: ~1.5 GB

### Full Training (Qwen 1.5B)
- Download: 1-2 minutes (first time)
- Training: 2-3 hours
- VRAM: ~2.3 GB

## Notes

- This configuration is specifically for the Windows training machine
- Mac is used for development, Windows for computation
- All scripts are version controlled in Git
- Training results are synced back to Mac for review
