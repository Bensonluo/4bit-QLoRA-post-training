"""DPO (Direct Preference Optimization) theory and implementation guide."""

## What is DPO?

**Direct Preference Optimization (DPO)** is a technique for aligning language models with human preferences **without training a separate reward model**.

It was introduced in the paper [*Direct Preference Optimization: Your Language Model is Secretly a Reward Model*](https://arxiv.org/abs/2305.14290) by Rafailov et al. (2023).

### The Problem with RLHF

Traditional RLHF (Reinforcement Learning from Human Feedback) requires:
1. **Reward model** trained on human preference data
2. **Policy model** optimized using the reward model
3. **Complex training pipeline** with multiple stages

**Challenges**:
- Reward model can be inaccurate
- Training is unstable
- Requires significant compute
- Complex to implement

### DPO Solution

DPO **eliminates the reward model** and directly optimizes the policy using preference pairs.

## Mathematical Foundation

### RLHF Objective (for comparison)

PPO-based RLHF optimizes:

$$\begin{align}
\text{Objective} &= \mathbb{E}_{x,y_w,y_l \sim \mathcal{D}} \left[ \log \sigma \left(\frac{\pi_\theta(y_w|x)}{\pi_\theta(y_l|x)}\right) \cdot R(x, y_w, y_l) \right] \\
&= \mathbb{E} \left[ \log \frac{\pi_\theta(y_w|x)}{\pi_\theta(y_l|x)} \cdot \left( \beta \log \frac{\pi_\text{ref}(y_w|x)}{\pi_\text{ref}(y_l|x)} \right) \right]
\end{align}$$

Where:
- $(y_w, y_l)$ is a preference pair (chosen, rejected)
- $R(x, y_w, y_l)$ is the reward signal
- $\pi_\theta$ is the policy being optimized
- $\pi_\text{ref}$ is the reference policy (frozen)
- $\beta$ controls the deviation from reference

### DPO Simplification

DPO shows this objective can be **simplified** to:

$$\begin{align}
\text{DPO Objective} &= \mathbb{E}_{x,y_w,y_l \sim \mathcal{D}} \left[ \log \sigma \left(\frac{\pi_\theta(y_w|x)}{\pi_\theta(y_l|x)}\right) \cdot \left| \beta \log \frac{\pi_\theta(y_w|x)}{\pi_\text{ref}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_\text{ref}(y_l|x)} \right| \right] \\
&\approx \mathbb{E}\left[ \log \frac{\pi_\theta(y_w|x)}{\pi_\theta(y_l|x)} \cdot \beta \log \frac{\pi_\theta(y_w|x}}{\pi_\text{ref}(y_w|x)} \right]
\end{align}$$

This is **exactly the same** as the RLHF objective, but **simplified**!

### Key Insight

DPO treats the log-ratio as an implicit reward:
$$r(x, y) = \beta \log \frac{\pi_\text{ref}(y|x)}{\pi_\text{ref}(\text{rejected}|x)}$$

### DPO Loss Function

$$\begin{align}
\mathcal{L}_{DPO} &= -\mathbb{E}_{(x,y_w,y_l)\sim\mathcal{D}} \left[ \log \sigma\left(\frac{\pi_\theta(y_w|x)}{\pi_\theta(y_l|x)}\right) \cdot \left| \beta \log \frac{\pi_\theta(y_w|x)}{\pi_\text{ref}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_\text{ref}(y_l|x)} \right| \right] \\
&= -\mathbb{E}\left[ \log \sigma\left(\frac{\pi_\theta(y_w|x)}{\pi_\theta(y_l|x)}\right) \cdot \left[ \beta \log \frac{\pi_\theta(y_w|x)}{\pi_\text{ref}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_\text{ref}(y_l|x)} \right] \right]
\end{align}$$

Where $\sigma$ is the sigmoid function.

## When to Use DPO

### ✅ Use DPO for:

1. **Preference alignment**: Improve model to prefer better responses
2. **Chat model fine-tuning**: Make chatbot responses more helpful
3. **Reducing harmful outputs**: Train to avoid unwanted behaviors
4. **Simpler than RLHF**: When you don't want to train reward model
5. **Stable training**: DPO is more stable than PPO

### ❌ Don't use DPO for:

1. **Task-specific fine-tuning**: Use SFT instead
2. **Knowledge injection**: Use continued pre-training
3. **Format learning**: Use instruction tuning
4. **Factual accuracy**: DPO optimizes preferences, not facts

## Comparison: SFT vs DPO vs RLHF

| Aspect | SFT | DPO | RLHF (PPO) |
|-------|-----|-----|------------|
| **Data** | Instruction-response pairs | Preference pairs (chosen/rejected) | Preference pairs + rewards |
| **Training** | Supervised learning | Direct preference optimization | Policy gradient with reward model |
| **Complexity** | Simple | Moderate | Complex |
| **Stability** | High | High | Low-Medium |
| **Compute** | Low | Low | High |
| **Time** | Short | Short | Long |
| **Reward Model** | None | Reference frozen | Trained separately |

## DPO Hyperparameters

### Beta ($\beta$)

Controls how much to deviate from reference model:

| $\beta$ | Effect | Use Case |
|-------|--------|----------|
| 0.05-0.1 | Very conservative | Minimal change from reference |
| **0.1-0.2** | **Default (recommended)** | Most cases |
| 0.2-0.5 | Moderate | Strong alignment needed |
| 0.5-1.0+ | Aggressive | Strong preference alignment |

**Rule of thumb**: Start with 0.1, increase if model isn't aligned enough.

### Loss Type

| Type | Description | When to Use |
|------|-------------|-------------|
| **sigmoid** | Default | Most cases, smooth gradient |
| **hinge** | Penalizes | When you want strict preferences |
| **ipo** | Inclusive preference | For diverse preferences |
| **pairwise** | Pairwise comparison | For large-scale training |

## Data Format

### Preference Pair Format

```json
{
  "prompt": "What is the capital of France?",
  "chosen": "The capital of France is Paris. It is known for the Eiffel Tower and rich history.",
  "rejected": "Paris is the capital."
}
```

### Alpaca Preference Format

```json
{
  "instruction": "What is the capital of France?",
  "input": "",
  "output": "The capital of France is Paris.",
  "rejected": "Paris is the capital.",
  "chosen": "The capital of France is Paris. It is known for the Eiffel Tower and rich history."
}
```

## Training Workflow

### Typical Pipeline

```
1. Base Model (Pre-trained)
         ↓
2. SFT (Instruction Tuning)
         ↓
3. DPO (Preference Optimization) ← YOU ARE HERE
         ↓
4. Aligned Model
```

### Can Also Use DPO Directly

```
1. Base Model (Pre-trained)
         ↓
2. DPO (Preference Optimization)
         ↓
3. Aligned Model
```

## Training Tips

### 1. Start with SFT First

DPO works best when model already has basic instruction-following capability.

**Recommended**:
1. SFT for 1-3 epochs (instruction following)
2. DPO for alignment (preference optimization)

### 2. Dataset Quality > Quantity

- High-quality preference pairs
- Clear preference signals
- Avoid contradictory pairs

### 3. Beta Tuning

**Start with**: $\beta = 0.1$

**Increase if**: Model doesn't follow preferences

**Decrease if**: Model becomes too conservative

### 4. Reference Model

- Use smaller model than main model (saves VRAM)
- Keep frozen during training
- Can be SFT version of same model

### 5. Memory Optimization for 8GB VRAM

```python
# Use smaller reference model
reference_model = "Qwen/Qwen2.5-0.5B-Instruct"  # 0.5B
main_model = "Qwen/Qwen2.5-1.5B-Instruct"   # 1.5B

# Freeze reference
for param in reference_model.parameters():
    param.requires_grad = False

# Use gradient checkpointing
gradient_checkpointing = True

# Batch size 1, gradient accumulation 8
```

## Common Issues

### Issue: Model becomes conservative

**Symptoms**: Model plays it safe, generic responses

**Solution**: Decrease beta
```python
beta = 0.05  # Decrease from 0.1
```

### Issue: No improvement

**Symptoms**: Loss doesn't decrease, metrics don't improve

**Solutions**:
1. Check preference data quality
2. Increase training epochs
3. Try higher beta
4. Ensure SFT was done first

### Issue: Training unstable

**Symptoms**: Loss spikes, NaN

**Solutions**:
1. Lower learning rate
2. Add gradient clipping
3. Increase warmup
4. Check for contradictory pairs in data

## Advantages Over RLHF

1. **Simpler**: No separate reward model training
2. **Stable**: More stable training
3. **Efficient**: Less compute required
4. **Direct**: Optimizes preferences directly

## Disadvantages

1. **Need preference data**: More expensive than SFT data
2. **Not for facts**: DPO optimizes preferences, not knowledge
3. **Can overfit**: To specific preferences in training data

## Resources

- [DPO Paper](https://arxiv.org/abs/2305.14290) - Original DPO paper
- [TRL DPO Documentation](https://huggingface.co/docs/trl/main/en/dpo_trainer/)
- [Alignment Research](https://huggingface.co/blog/aligned-to-train-august-2023-08-22)

## Next Steps

After implementing DPO, consider:
1. **KTO** (Kahneman-Tversky Optimization): Alternative to DPO
2. **ORPO** (Odds Ratio Preference Optimization): More recent technique
3. **RLAIF** (RL from AI Feedback): Synthetic preferences
4. **Iterative DPO**: Multiple rounds of SFT + DPO
