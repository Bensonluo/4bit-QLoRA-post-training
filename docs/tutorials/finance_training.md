# Finance Domain Training Guide

This guide focuses on fine-tuning LLMs for finance and stock investment applications.

## Why Finance LLMs?

Finance LLMs provide:
- **Domain expertise**: Stock analysis, financial metrics, investment strategies
- **Regulatory awareness**: Compliance, risk management
- **Professional tone**: Business-appropriate communication
- **Real-world value**: Actual investment assistant capabilities

## Use Cases

### 1. Financial Analysis

```
User: "Analyze AAPL's Q3 2024 earnings"
Model: Provides analysis of revenue, EPS, guidance, market reaction
```

### 2. Investment Education

```
User: "Explain P/E ratio in simple terms"
Model: Clear explanation with examples
```

### 3. Portfolio Strategy

```
User: "Should I diversify my tech portfolio?"
Model: Discusses diversification benefits, suggests strategies
```

### 4. Risk Assessment

```
User: "What are the risks of cryptocurrency?"
Model: Explains volatility, regulatory, security risks
```

## Data Preparation

### Public Finance Datasets

#### 1. Finance-Alpaca (Recommended)

```python
from src.data import FinanceDataset

dataset = FinanceDataset(
    data_path="yahma/alpaca-cleaned",
    max_samples=50000,
)
```

This filters the Alpaca dataset for finance-related content.

#### 2. Financial News

- Bloomberg, Reuters headlines
- Earnings call transcripts
- SEC filings (10-K, 10-Q)

#### 3. Financial Q&A

- Stack Exchange Finance
- Reddit r/finance, r/investing
- Financial advisor Q&A

### Custom Data Format

Create `data/custom/finance_qa.jsonl`:

```json
{"instruction": "What is the difference between stocks and bonds?", "input": "", "output": "Stocks represent ownership..."}
{"instruction": "Explain dividend yield", "input": "", "output": "Dividend yield is..."}
{"instruction": "What is a P/E ratio?", "input": "AAPL has a P/E of 25", "output": "A P/E ratio of 25 means..."}
```

### Data Quality Guidelines

✅ **Good Finance Data:**
- Accurate financial information
- Clear explanations
- Real-world examples
- Professional tone
- Covers risk and downsides

❌ **Avoid:**
- Outdated information (check dates!)
- Specific investment advice (regulatory risk)
- Guaranteed returns
- Insider information
- Biased recommendations

## Training Configuration

### Optimized Finance Config

The framework includes a finance-optimized preset:

```python
from config.sft import FINANCE_SFT_CONFIG

# Already optimized for finance domain
print(FINANCE_SFT_CONFIG.model)      # Qwen 1.5B
print(FINANCE_SFT_CONFIG.training)   # Memory-optimized
print(FINANCE_SFT_CONFIG.lora)       # Finance-optimized modules
```

### Key Finance-Specific Settings

```yaml
# Model: Qwen 1.5B (bilingual, strong performance)
model:
  name: "Qwen/Qwen2.5-1.5B-Instruct"
  quantization_bits: 4
  max_length: 1024  # Finance text can be longer

# Training: More epochs for domain specialization
training:
  num_epochs: 3-5  # Finance needs more exposure
  learning_rate: 2e-4
  batch_size: 1
  gradient_accumulation_steps: 8

# LoRA: Balanced expressiveness
lora:
  r: 16
  lora_alpha: 32
  target_modules: all_attention_and_ffn  # Full fine-tuning

# Data: Finance filtering
data:
  dataset_name: "yahma/alpaca-cleaned"
  max_samples: 50000
  format: "alpaca"
```

## Training Pipeline

### Step 1: Prepare Data

```bash
python scripts/download_data.py \
    --dataset yahma/alpaca-cleaned \
    --output-dir ./data/raw \
    --num-samples 50000
```

### Step 2: Train (Local)

```bash
python scripts/train_sft.py --finance-mode
```

### Step 3: Train (Remote Windows)

```bash
python scripts/train_remote.py --finance-mode
```

### Step 4: Monitor Training

```bash
# TensorBoard
tensorboard --logdir ./outputs/logs

# Or check logs
tail -f ./outputs/sft/training.log
```

## Evaluation

### Quantitative Metrics

```python
from src.evaluation import compute_perplexity, generate_samples

# Compute perplexity on finance test set
ppl = compute_perplexity(model, test_dataset, tokenizer)
print(f"Finance perplexity: {ppl:.2f}")
```

### Qualitative Evaluation

Test on finance questions:

```python
test_prompts = [
    "What is the difference between ETF and mutual fund?",
    "Explain the concept of dollar-cost averaging",
    "What are the risks of investing in bonds?",
    "How do I calculate dividend yield?",
    "What is a market cap?",
]

for prompt in test_prompts:
    response = model.generate(prompt)
    print(f"Q: {prompt}")
    print(f"A: {response}\n")
```

### Human Evaluation

Create a test set and evaluate:
1. Accuracy of information
2. Completeness of answer
3. Clarity of explanation
4. Appropriate disclaimers

## Advanced Techniques

### 1. Multi-Stage Training

```python
# Stage 1: General finance (Alpaca finance-filtered)
# Stage 2: Specialized (stocks, or crypto, or taxes)

# Train stage 1
python scripts/train_sft.py \
    --dataset general-finance \
    --output-dir ./outputs/finance-general

# Train stage 2 (load stage 1 as base)
python scripts/train_sft.py \
    --model ./outputs/finance-general \
    --dataset stock-analysis \
    --output-dir ./outputs/finance-stocks
```

### 2. Curriculum Learning

Start with easy examples, progress to hard:

```python
# Easy: "What is a stock?"
# Medium: "Compare stocks and bonds"
# Hard: "Analyze this 10-K filing"
```

### 3. RAG Augmentation

Combine with retrieval for up-to-date info:

```python
# 1. Retrieve recent news
# 2. Add as context
# 3. Generate response with context

prompt = f"""
Based on the following news:
{retrieved_news}

Answer: {user_question}
"""
```

## Deployment

### Local Deployment

```bash
# Merge adapters
python scripts/merge_lora.py \
    --base-model Qwen/Qwen2.5-1.5B-Instruct \
    --adapter-path ./outputs/finance-sft \
    --output-dir ./outputs/finance-merged

# Export to GGUF (optional)
python scripts/merge_lora.py \
    --base-model Qwen/Qwen2.5-1.5B-Instruct \
    --adapter-path ./outputs/finance-sft \
    --output-dir ./outputs/finance-merged \
    --export-gguf
```

### API Server

```python
from fastapi import FastAPI
from transformers import AutoModelForCausalLM, AutoTokenizer

app = FastAPI()

model = AutoModelForCausalLM.from_pretrained("./outputs/finance-merged")
tokenizer = AutoTokenizer.from_pretrained("./outputs/finance-merged")

@app.post("/chat")
def chat(prompt: str):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=512)
    response = tokenizer.decode(outputs[0])
    return {"response": response}
```

## Best Practices

### 1. Add Disclaimers

Always include appropriate disclaimers:

```python
SYSTEM_PROMPT = """You are a helpful financial assistant.
Always include the disclaimer: "This is not financial advice.
Consult a qualified financial advisor for investment decisions."
"""
```

### 2. Know Your Limits

Model should not:
- Provide specific investment advice ("Buy AAPL now")
- Guarantee returns
- Ignore risk factors
- Make predictions with certainty

Model should:
- Explain concepts
- Compare options
- Highlight risks
- Suggest further research
- Recommend professional advice

### 3. Keep Data Fresh

Financial data becomes outdated quickly:
- Regularly update training data
- Use date filters in RAG
- Clearly indicate data recency

## Example Outputs

### Before Finance Fine-tuning

```
Q: What is a P/E ratio?
A: I'm not sure about that specific financial term.
```

### After Finance Fine-tuning

```
Q: What is a P/E ratio?
A: The P/E (Price-to-Earnings) ratio measures a company's
stock price relative to its earnings per share. It's calculated
by dividing the current stock price by EPS. A higher P/E may
indicate investors expect higher growth, while a lower P/E may
suggest the stock is undervalued. However, always compare P/E
ratios within the same industry, as "normal" varies by sector.
```

## Common Issues

### Issue: Hallucinating Numbers

**Solution**: Add more examples with correct numbers in training data

### Issue: Outdated Information

**Solution**: Use RAG with current data, add date context

### Issue: Too Generic

**Solution**: Increase training epochs, use more finance-specific data

### Issue: Regulatory Risk

**Solution**: Add disclaimers, avoid specific advice

## Resources

### Datasets
- [FinanceBench](https://huggingface.co/datasets/PatronusAI/financebench)
- [Financial Phrase Bank](https://www.researchgate.net/profile/Petra-Mikula/publication/)
- [FinQA](https://github.com/czyssrs/FinQA)

### References
- [Investopedia](https://www.investopedia.com/)
- [SEC EDGAR](https://www.sec.gov/edgar)
- [Yahoo Finance](https://finance.yahoo.com/)

## Next Steps

1. **Collect Data**: Gather finance-specific QA pairs
2. **Train Model**: Run `python scripts/train_sft.py --finance-mode`
3. **Evaluate**: Test on finance questions
4. **Deploy**: Create investment assistant API

Happy finance fine-tuning! 📈💰
