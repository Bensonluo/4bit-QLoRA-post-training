# QLoRA: Quantized Low-Rank Adaptation

## Overview

QLoRA (Quantized Low-Rank Adaptation) enables efficient fine-tuning of large language models on consumer hardware by combining:
1. **4-bit quantization** of base model weights
2. **Low-rank adapters** (LoRA) for trainable parameters
3. **Memory optimizations** (gradient checkpointing, mixed precision)

This makes it possible to fine-tune 7B+ parameter models on GPUs with as little as 8GB VRAM.

## Quantization Basics

### What is Quantization?

Quantization reduces the precision of model weights:
- **FP32 (32-bit)**: $4.29 \times 10^9$ possible values
- **FP16 (16-bit)**: $65,536$ possible values
- **INT8 (8-bit)**: $256$ possible values
- **NF4 (4-bit)**: $16$ possible values

### Why Quantize?

| Precision | VRAM for 7B Model | Speed | Quality Loss |
|-----------|-------------------|-------|--------------|
| FP32 | ~28GB | Baseline | None |
| FP16 | ~14GB | 2x faster | Negligible |
| INT8 | ~7GB | 2-3x faster | Small |
| **NF4 (QLoRA)** | **~3-4GB** | **Similar to INT8** | **Minimal** |

## 4-bit NormalFloat (NF4)

### The Problem with Uniform Quantization

LLM weights follow a **normal distribution** (bell curve), not a uniform distribution.

Using uniform 4-bit quantization on normally distributed data loses precision in the high-density regions (around 0).

### NF4 Solution

NF4 is optimized for normally distributed data:
- Quantiles are carefully chosen
- More precision where weights are concentrated (near 0)
- Better preservation of model capabilities

### QLoRA Innovations

1. **NF4 Quantization**: Optimized for normal distributions
2. **Double Quantization**: Quantize the quantization constants too!
3. **Flash Attention**: Faster attention computation
4. **Paged Optimizers**: CPU offloading for optimizer states

## LoRA (Low-Rank Adaptation)

### The Core Idea

Instead of fine-tuning all parameters, train **small adapter matrices**:

$$W_{new} = W_{frozen} + \Delta W = W_{frozen} + BA$$

Where:
- $W_{frozen} \in \mathbb{R}^{d \times d}$: Frozen base weights
- $B \in \mathbb{R}^{d \times r}$: Learnable adapter (rank r)
- $A \in \mathbb{R}^{r \times d}$: Learnable adapter (rank r)
- $r \ll d$: Rank (typically 8, 16, 32)

### Parameter Count Comparison

For a $7$B parameter model with $d = 4096$:

| Method | Trainable Parameters | VRAM |
|--------|---------------------|------|
| Full Fine-tuning | ~7B | ~28GB (FP32) |
| **LoRA (r=16)** | **~40M** | **~4GB** |

**Reduction**: $99.4\%$ fewer parameters!

### LoRA Hyperparameters

#### 1. Rank ($r$)

**Effect**: Controls expressiveness of adapters

| Rank | Parameters | Best For | VRAM Impact |
|------|-----------|----------|-------------|
| 8 | Minimal | Simple tasks, small datasets | Lowest |
| 16 | Balanced | **Most cases (recommended)** | Low |
| 32 | More expressive | Complex tasks, large datasets | Medium |
| 64+ | Maximum | Specialized applications | Higher |

**Rule of thumb**: Start with r=16, increase if underfitting.

#### 2. Alpha ($\alpha$)

**Effect**: Scaling factor for LoRA update

$$\Delta W = \frac{\alpha}{r} BA$$

**Typical values**: $\alpha = 2r$

Examples:
- r=8 → $\alpha$=16
- r=16 → $\alpha$=32
- r=32 → $\alpha$=64

#### 3. Target Modules

Which layers to apply LoRA to:

```python
# Qwen/Llama models (recommended)
target_modules = [
    "q_proj",   # Query projection
    "v_proj",   # Value projection
    "k_proj",   # Key projection
    "o_proj",   # Output projection
    "gate_proj",  # FFN gate
    "up_proj",    # FFN up
    "down_proj",  # FFN down
]

# Minimal (saves VRAM)
target_modules = ["q_proj", "v_proj"]

# Aggressive (more parameters)
target_modules = all linear layers
```

## Memory Optimization Techniques

### 1. Gradient Checkpointing

**Trade compute for memory**

Instead of storing all intermediate activations:
- Recompute activations during backward pass
- Saves ~50% activation memory
- Slows training by ~20%

**Enable when**:
- ✅ VRAM limited (8GB or less)
- ✅ Large batch sizes (>1)
- ✅ Long sequences (>512 tokens)

### 2. Mixed Precision Training

**Use lower precision for activations**

```python
# RTX 30xx/40xx (recommended)
bf16=True  # Brain float 16

# Older GPUs
fp16=True   # Float 16
```

**Savings**: ~30-40% memory

**Warning**: Watch for NaN loss (reduce learning rate if occurs).

### 3. Gradient Accumulation

**Simulate larger batch sizes**

```python
batch_size = 1
gradient_accumulation_steps = 8

# Effective batch size = 1 × 8 = 8
```

**Why?**
- VRAM limits batch size to 1-2
- Accumulate gradients over 8 steps
- Update every 8 steps
- Same effective batch as BS=8

### 4. Optimizer Offloading

**Move optimizer states to CPU**

```python
# Not supported in bitsandbytes yet
# Coming soon!
```

## Complete QLoRA Training Loop

```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model

# 1. Quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

# 2. Load model (4-bit)
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-1.5B-Instruct",
    quantization_config=bnb_config,
    device_map="auto",
)

# 3. Prepare for k-bit training
from peft import prepare_model_for_kbit_training
model = prepare_model_for_kbit_training(model)

# 4. LoRA config
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

# 5. Apply LoRA
model = get_peft_model(model, lora_config)

# 6. Train!
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
print(f"Trainable: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")
```

## VRAM Usage Breakdown

For Qwen 1.5B with QLoRA:

| Component | VRAM |
|-----------|------|
| Base model (4-bit) | ~1.8 GB |
| LoRA adapters (r=16) | ~0.4 GB |
| Activations (BS=1, seq=512) | ~1.2 GB |
| Optimizer states | ~0.8 GB |
| Gradients | ~0.4 GB |
| **Total** | **~4.6 GB** |

✅ **Fits comfortably in 8GB VRAM!**

## Performance Tips

### For 8GB VRAM (RTX 4060, etc.)

```python
ModelConfig(
    quantization_bits=4,
    max_length=1024,
)

TrainingConfig(
    batch_size=1,
    gradient_accumulation_steps=8,
    gradient_checkpointing=True,
    bf16=True,
)

LoRAConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
)
```

### For 12GB VRAM (RTX 4070, etc.)

```python
ModelConfig(
    quantization_bits=4,
    max_length=2048,
)

TrainingConfig(
    batch_size=2,
    gradient_accumulation_steps=4,
    gradient_checkpointing=True,
    bf16=True,
)

LoRAConfig(
    r=32,  # Can afford larger adapters
    lora_alpha=64,
)
```

## Common Issues & Solutions

### 1. Out of Memory (OOM)

**Symptoms**: CUDA out of memory error

**Solutions** (in order):
1. Reduce `max_length` (1024 → 512)
2. Reduce `batch_size` (2 → 1)
3. Increase `gradient_accumulation_steps`
4. Enable `gradient_checkpointing=True`
5. Reduce `lora_r` (32 → 16 → 8)

### 2. Loss Becomes NaN

**Symptoms**: Training loss becomes NaN

**Solutions**:
1. Lower learning rate (2e-4 → 1e-4)
2. Enable gradient clipping (`max_grad_norm=1.0`)
3. Check data quality (remove NaNs, infinite values)
4. Try FP16 instead of BF16

### 3. Slow Training

**Symptoms**: Training takes too long

**Solutions**:
1. Enable Flash Attention 2
2. Increase `batch_size` (if VRAM allows)
3. Reduce `logging_steps`
4. Use `gradient_checkpointing=False` (if VRAM allows)

## Benchmarks

Training speed for Qwen 1.5B on RTX 4060 (8GB):

| Config | Tokens/sec | Time/epoch |
|--------|-----------|------------|
| BS=1, GA=8, GC=True | ~800 | ~10 min (50K tokens) |
| BS=1, GA=4, GC=True | ~900 | ~9 min |
| BS=2, GA=4, GC=False | ~1200 | ~7 min |

## Advanced: When NOT to Use QLoRA

### Consider full fine-tuning for:

1. **Research**: Studying fundamental properties
2. **Production-scale**: Have access to A100s/H100s
3. **Maximum Performance**: Need every bit of accuracy
4. **Architecture Changes**: Modifying model structure

### QLoRA limitations:

1. **Slightly lower quality**: ~1-2% worse than full fine-tuning
2. **Adapter merging**: Required before deployment
3. **Complexity**: More moving parts

## Resources

- [QLoRA Paper: Dettmers et al. (2023)](https://arxiv.org/abs/2305.14314)
- [LoRA Paper: Hu et al. (2021)](https://arxiv.org/abs/2106.09685)
- [bitsandbytes GitHub](https://github.com/TimDettmers/bitsandbytes)
- [PEFT Documentation](https://huggingface.co/docs/peft)

## Next Steps

- **SFT Guide**: Learn supervised fine-tuning in `sft.md`
- **DPO Guide**: Learn preference optimization in `dpo.md`
- **Finance Tutorial**: Domain-specific training in `../tutorials/finance_training.md`
