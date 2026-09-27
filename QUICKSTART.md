# Quick Start Guide

## 🧩 产品主路径：从业务目标到可用模型（Data Intake 工作台）

```bash
python -m venv venv && source venv/bin/activate
pip install -e ".[ui]"
python scripts/launch_dashboard.py        # 端口被占时加 --mlflow-port 5001
```

打开 http://localhost:8501 → 「🧩 分析我的目标与数据」：

1. **描述业务目标 + 上传一份 CSV 样例**（不用先整理成训练格式）。
2. **分析**：配置了 Agent 服务（BYOK，如 GLM Coding Plan）就点「联合分析目标与数据」；
   **没有任何密钥也能开始**——展开「没有 Agent 服务？用基础分析开始」，选择答案列与业务分组字段即可（产品内置的确定性判断，如实声明不判断业务含义）。
3. **核对真实转换预览**并确认业务含义 → 提供全量数据 → 验证并确认 → 生成分区。
4. **准备并启动训练**（预检有提示会先暂停等你核对）。参数不用怕填错——
   页面「高级：手工配置训练参数」有逐参数大白话解释和推荐起步值：
   小数据起步：1–2 轮 / batch 1 / 梯度累积 4 / LoRA rank 8 / 学习率见下表。

   | 数据量 | 学习率起步 | 依据 |
   |---|---|---|
   | ≥ 2,000 条 | 2e-4 | 通用可靠默认 |
   | < 2,000 条 | 5e-5 – 1e-4 | 小数据集用大学习率易过拟合/不稳，业界指南一致建议降档 |

   > 以上是外部指南的汇总启发（Unsloth 指南、Raschka 实践笔记等，2026），
   > 不是本产品的实测结论；以你自己的同题对照结果为准。

5. **同题对照**：基座 vs 微调在同一固定开发集比较；输出截断/复述指令等问题会在结果表中标出并给出核查方向。
6. **改进迭代**：从坏例出发提出改进轮次，一次授权自动执行到三模型对照（后台独立推进，关闭页面不影响），完成后由你决定采用/继续/停止/证据不足。

> 诚实边界：训练完成与对照完成不等于业务达标；最终验收用独立保留的测试集，通过标准由你在运行前确认。

CLI 等价命令见 `python scripts/data_intake.py --help`（intake/train/eval/iteration 全套）。

---

## 🖥️ Windows Training Setup（远程 GPU 场景）

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
