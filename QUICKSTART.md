# Quick Start Guide - Windows Training Setup

## 🚀 One-Command Setup (From Mac)

```bash
# Run the automated setup
bash /tmp/setup_windows_training.sh
```

This will:
- ✅ Sync project to Windows
- ✅ Install PyTorch with CUDA
- ✅ Install all dependencies
- ✅ Verify installation

**Estimated time:** 15-20 minutes

---

## 📋 Manual Setup Steps

If automated setup fails, follow these steps:

### 1. Install NVIDIA Drivers (Windows Side)

Open PowerShell on Windows (NOT WSL):
```powershell
# Check if drivers installed
nvidia-smi

# If not found, download from:
# https://www.nvidia.com/Download/index.aspx
# Select: RTX 4060, Windows 11
```

### 2. Install PyTorch with CUDA (WSL2)

```bash
# SSH to Windows
ssh windows

# Install PyTorch
pip3 install --upgrade pip
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Verify
python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
# Should print: CUDA: True
```

### 3. Install Project Dependencies

```bash
cd ~/4bit-QLoRA-post-training
pip3 install -r requirements.txt
```

### 4. Verify Setup

```bash
# Test GPU
nvidia-smi

# Test PyTorch
python3 -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}')"

# Test imports
python3 -c "from transformers import AutoModelForCausalLM; print('✅ OK')"
```

---

## 🎯 Test Training

### Quick Test (5 minutes)

```bash
ssh windows
cd ~/4bit-QLoRA-post-training

# Test with tiny dataset
python3 scripts/train_sft.py \
    --max-samples 100 \
    --epochs 1 \
    --output-dir ./outputs/test
```

### Full Finance Training (2-3 hours)

```bash
# From Mac (recommended - monitor from Mac)
cd ~/Documents/GitHub/4bit-QLoRA-post-training
python scripts/train_remote.py --finance-mode

# Or directly on Windows
ssh windows
cd ~/4bit-QLoRA-post-training
python3 scripts/train_sft.py --finance-mode
```

---

## 📊 Monitor Training

### On Mac (if using train_remote.py)
```bash
# Progress shows automatically
# Logs saved to: ./outputs/logs/
```

### On Windows
```bash
# In another terminal
ssh windows
watch -n 1 nvidia-smi  # Monitor GPU

# View logs
tail -f ~/4bit-QLoRA-post-training/outputs/sft/training.log

# TensorBoard
tensorboard --logdir ~/4bit-QLoRA-post-training/outputs/logs
```

---

## ✅ Verification Checklist

Before training starts, verify:

- [ ] NVIDIA drivers installed (run `nvidia-smi` in WSL2)
- [ ] PyTorch with CUDA working (run Python test above)
- [ ] Project synced to Windows
- [ ] All dependencies installed
- [ ] GPU detected: RTX 4060 8GB

---

## 🐛 Troubleshooting

### Issue: nvidia-smi not found in WSL2

**Cause:** NVIDIA drivers not installed on Windows

**Fix:**
1. Open PowerShell on Windows (outside WSL)
2. Download drivers from https://www.nvidia.com/Download/index.aspx
3. Install and restart Windows
4. Verify: `nvidia-smi` in PowerShell

### Issue: CUDA not available in PyTorch

**Symptoms:** `torch.cuda.is_available()` returns `False`

**Fix:**
```bash
pip3 uninstall torch torchvision torchaudio
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Issue: bitsandbytes installation fails

**Fix:**
```bash
pip3 install bitsandbytes>=0.41.0
```

### Issue: Out of memory during training

**Fix:** Reduce batch size in config
```python
batch_size = 1
gradient_accumulation_steps = 16  # Increase this
max_length = 512  # Reduce this
```

---

## 🎓 After Setup Complete

1. **Learn the theory**
   ```bash
   open docs/theory/qlora.md
   open docs/theory/sft.md
   ```

2. **Run first training**
   ```bash
   python scripts/train_remote.py --finance-mode
   ```

3. **Evaluate results**
   ```bash
   python scripts/evaluate.py \
       --model-path ./outputs/finance-merged \
       --max-samples 50
   ```

---

## 📞 Need Help?

- Check `/tmp/windows_setup.md` for detailed setup guide
- Review `docs/tutorials/getting_started.md`
- Test each step individually if automated script fails

---

## 🎉 Success Indicators

You'll know setup worked when:

1. ✅ `nvidia-smi` shows RTX 4060
2. ✅ `torch.cuda.is_available()` returns `True`
3. ✅ All imports work without errors
4. ✅ Test training completes without OOM errors

Then you're ready to train! 🚀
