# Training Log

This document tracks all training runs and their results.

## Quick Test #1 - Proof of Concept

**Date**: 2025-01-30

**Configuration**:
- Model: Qwen/Qwen2.5-0.5B-Instruct
- Quantization: 4-bit NF4
- Dataset: Alpaca-cleaned (100 samples, filtered to 10 finance)
- Epochs: 1
- Batch size: 1
- Gradient accumulation: 4
- LoRA rank: 8
- Hardware: RTX 4060 8GB

**Results**:
- Status: ✅ Success
- Training time: 14 seconds
- Train loss: 1.773
- Eval loss: 1.901
- Trainable parameters: 540,672 (0.17%)
- Samples/sec: 0.708

**Key Learnings**:
1. ✅ HF mirror (https://hf-mirror.com) required for China
2. ✅ Flash Attention 2 not installed (use_flash_attention=False)
3. ✅ 4-bit QLoRA works perfectly on 8GB VRAM
4. ✅ Training pipeline end-to-end functional

**Output**: `./outputs/test-quick/`

---

## DPO Implementation

**Date**: 2025-01-30

**Implementation Details**:
- DPO (Direct Preference Optimization) framework added
- Eliminates need for separate reward model (unlike RLHF)
- Uses preference pairs (prompt, chosen, rejected)
- Frozen reference model for reward computation

**Files Created**:
- `config/dpo.py` - DPO configuration classes
  - DPOConfig: beta, max_length, loss_type
  - ReferenceModelConfig: frozen reference model
  - PreferenceDataConfig: preference dataset config
  - DPOTrainingConfig: complete training config
- `src/training/dpo_trainer.py` - DPO trainer implementation
- `docs/theory/dpo.md` - comprehensive DPO theory guide
- `scripts/train_dpo.py` - CLI training script

**Key Features**:
- Default beta: 0.1 (conservative)
- Support for multiple loss types (sigmoid, hinge, ipo, pairwise)
- Memory optimized for 8GB VRAM
- Finance-specific auto-filtering
- W&B and TensorBoard integration

**Usage**:
```bash
# Quick test
python scripts/train_dpo.py --quick-test

# Full training
python scripts/train_dpo.py

# Finance DPO
python scripts/train_dpo.py --auto-filter --beta 0.2
```

**Status**: ✅ Implementation complete, ready for testing

---

## Full Finance Training (Pending)

**Planned Configuration**:
- Model: Qwen/Qwen2.5-1.5B-Instruct
- Dataset: Alpaca-cleaned (50K samples, finance-filtered)
- Epochs: 3
- Batch size: 1
- Gradient accumulation: 8
- LoRA rank: 16
- Expected time: 2-3 hours

**Start Date**: TBD
**Status**: Ready to run

---

## Experiment Notes

### Environment Setup

**Windows Machine**:
- OS: Windows 11 with WSL2
- GPU: NVIDIA GeForce RTX 4060 Laptop GPU (8GB)
- Python: 3.12.3
- PyTorch: 2.5.1+cu121
- CUDA: Available via WSL2

**Critical Configuration**:
```bash
# Required for model downloads in China
export HF_ENDPOINT=https://hf-mirror.com

# Flash Attention 2 not installed
use_flash_attention=False

# Always use virtual environment
source ~/4bit-QLoRA-post-training/venv/bin/activate
```

### Issues and Solutions

| Issue | Solution |
|-------|----------|
| Hugging Face blocked/throttled in China | Use HF_ENDPOINT=https://hf-mirror.com |
| Flash Attention 2 not installed | Set use_flash_attention=False |
| eval_strategy deprecated | Changed to eval_strategy (fixed in code) |
| Externally managed Python environment | Create virtual environment (venv) |

### Performance Metrics

**Quick Test Performance**:
- Model loading: ~35 seconds (with HF mirror)
- Training: 14 seconds for 1 epoch
- Total: <1 minute

**VRAM Usage**:
- Base model (4-bit): ~0.8 GB
- LoRA adapters: ~0.4 GB
- Training overhead: ~1 GB
- Total: ~2.2 GB (well within 8GB limit)

---

## Next Training Runs

### Run #2: Medium Dataset Test
- Model: Qwen 0.5B
- Samples: 1,000
- Epochs: 2
- Purpose: Test with larger dataset

### Run #3: Full Finance Training
- Model: Qwen 1.5B
- Samples: 50,000
- Epochs: 3
- Purpose: Production finance model

### Run #4: Different LoRA Configs
- Experiment with different LoRA ranks (8, 16, 32)
- Compare performance vs memory usage

---

## Model Versions

Trained models will be saved in `./outputs/`:

1. `test-quick/` - Proof of concept
2. `finance-sft/` - Full finance model (pending)
3. `finance-merged/` - Merged base model + adapters (pending)

---

## How to Replicate

All training runs are replicable using the provided scripts:

```bash
# Quick test
python scripts/train_quick_test.py

# Full training
python scripts/train_finance_full.py
```

See `docs/tutorials/windows_training_setup.md` for complete setup instructions.
