# Supervised Fine-Tuning (SFT)

## Overview

Supervised Fine-Tuning (SFT) is the process of training a pre-trained language model on a specific dataset with labeled examples. It's the foundation for creating task-specific or domain-specific language models.

## What is SFT?

SFT takes a general-purpose pre-trained model (like Qwen, Llama, or GPT) and further trains it on:
- **Instruction-response pairs**: Teach the model to follow instructions
- **Domain-specific data**: Adapt the model to specialized domains (finance, medical, legal)
- **Task-specific data**: Optimize for specific tasks (summarization, translation, etc.)

## Mathematical Foundation

### Cross-Entropy Loss

SFT minimizes the cross-entropy loss between the model's predictions and the target responses:

$$L_{CE} = -\sum_{i=1}^{V} y_i \log(\hat{y}_i)$$

Where:
- $y_i$ is the true token probability (1 for correct token, 0 otherwise)
- $\hat{y}_i$ is the predicted probability for token $i$
- $V$ is the vocabulary size

### Training Objective

For a dataset of $N$ examples, the total loss is:

$$L = \frac{1}{N} \sum_{j=1}^{N} L_{CE}^{(j)}$$

### Data Format

The standard Alpaca format:

```
### Instruction:
{instruction}

### Input:
{optional_input}

### Response:
{target_response}
```

## When to Use SFT

### ✅ Use SFT for:

1. **Instruction Following**: Teaching models to follow user instructions
2. **Domain Adaptation**: Specializing in finance, medical, legal, etc.
3. **Style Transfer**: Changing response style or tone
4. **Knowledge Injection**: Teaching new domain knowledge
5. **Task Optimization**: Improving performance on specific tasks

### ❌ Don't use SFT for:

1. **Unsafe Content**: Generating harmful, illegal, or unethical content
2. **Copyrighted Material**: Reproducing protected content
3. **Personal Data**: Training on private information without consent

## Best Practices

### 1. Data Quality > Quantity

```
❌ Bad: 100,000 low-quality examples
✅ Good: 5,000 high-quality, diverse examples
```

### 2. Balanced Dataset

Ensure your training data covers:
- Different question types
- Various difficulty levels
- Edge cases and exceptions
- Domain-specific terminology

### 3. Appropriate Temperature

During generation:
- **Low temperature (0.1-0.3)**: More focused, deterministic outputs
- **Medium temperature (0.5-0.7)**: Balanced creativity and focus
- **High temperature (0.8-1.0)**: More creative, diverse outputs

### 4. Hyperparameter Tuning

Key hyperparameters for SFT:

| Parameter | Typical Range | Effect |
|-----------|--------------|--------|
| Learning Rate | 1e-5 to 5e-4 | Too high: unstable; Too low: slow convergence |
| Batch Size | 1-4 (for 8GB VRAM) | Limited by VRAM |
| Epochs | 1-5 | More epochs = better fit, risk of overfitting |
| LoRA r | 8-32 | Higher = more parameters, more expressiveness |
| LoRA alpha | 2*r | Typically 2x the rank |

## QLoRA + SFT

### Why QLoRA?

Traditional fine-tuning requires massive VRAM:
- 7B model @ 16-bit: ~14GB VRAM
- 7B model @ 4-bit QLoRA: ~3GB VRAM

**QLoRA enables training on consumer GPUs!**

### QLoRA Components

1. **4-bit NormalFloat (NF4) Quantization**: Optimized for normally distributed weights
2. **Low-Rank Adapters (LoRA)**: Train small adapter matrices instead of full model
3. **Gradient Checkpointing**: Trade compute for memory
4. **Mixed Precision**: Use bfloat16/float16 for activations

### LoRA Mathematics

Instead of updating weight matrix $W \in \mathbb{R}^{d \times d}$, LoRA learns:

$$W' = W + \Delta W = W + BA$$

Where:
- $B \in \mathbb{R}^{d \times r}$ (rank r)
- $A \in \mathbb{R}^{r \times d}$ (rank r)
- $r \ll d$ (typically r=8, 16, 32)

**Parameters**: $2 \times d \times r$ vs $d \times d$ (huge savings!)

## Practical Example

### Training a Finance Model

```python
from config.sft import FINANCE_SFT_CONFIG
from src.training import run_sft_training

# Use the finance preset
run_sft_training(
    model_config=FINANCE_SFT_CONFIG.model,
    training_config=FINANCE_SFT_CONFIG.training,
    lora_config=FINANCE_SFT_CONFIG.lora,
    data_config=FINANCE_SFT_CONFIG.data,
    logging_config=FINANCE_SFT_CONFIG.logging,
)
```

This will:
1. Load Qwen 1.5B with 4-bit quantization (~2GB VRAM)
2. Load finance-filtered Alpaca dataset
3. Train with LoRA (r=16, ~500MB additional)
4. Save LoRA adapters for later merging

## Evaluation Metrics

### Quantitative Metrics

1. **Perplexity**: Lower is better
   - Measures how well model predicts test data
   - Good for comparing models

2. **Token Accuracy**: Higher is better
   - Percentage of correctly predicted tokens
   - Less informative for generation tasks

3. **Domain-Specific Metrics**: Task-dependent
   - Finance: Prediction accuracy, risk assessment
   - Medical: Diagnosis accuracy
   - Code: Execution success rate

### Qualitative Evaluation

1. **Human Evaluation**: Gold standard but expensive
2. **Side-by-Side Comparison**: Compare model outputs
3. **A/B Testing**: Real-world user preference
4. **Error Analysis**: Examine failure cases

## Common Pitfalls

### 1. Overfitting

**Symptoms**: Low training loss, high validation loss

**Solutions**:
- Reduce training epochs
- Increase regularization (dropout, weight decay)
- Use more diverse training data

### 2. Catastrophic Forgetting

**Symptoms**: Model loses general capabilities

**Solutions**:
- Include general data in training mix
- Use lower learning rate
- Regularize towards base model

### 3. Hallucination

**Symptoms**: Model generates incorrect information

**Solutions**:
- Ensure training data is accurate
- Add uncertainty prompts ("I'm not sure, but...")
- Post-processing filters

## Resources

- [Original LoRA Paper](https://arxiv.org/abs/2106.09685)
- [QLoRA Paper](https://arxiv.org/abs/2305.14314)
- [Alpaca Dataset](https://crfm.stanford.edu/2023/03/13/alpaca.html)
- [Hugging Face PEFT](https://huggingface.co/docs/peft)

## Next Steps

- **DPO**: Learn preference optimization in `dpo.md`
- **QLoRA Deep Dive**: See `qlora.md`
- **Finance Training**: See domain-specific guide in `../tutorials/finance_training.md`
