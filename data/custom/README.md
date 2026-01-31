# Custom Datasets Guide

This directory is for storing your custom training datasets.

## Supported Formats

### 1. Alpaca Format (for SFT)

File format: JSONL

```json
{"instruction": "Your question here", "input": "", "output": "Your answer here"}
{"instruction": "Another question", "input": "Optional context", "output": "Another answer"}
```

### 2. Preference Format (for DPO)

File format: JSONL

```json
{"prompt": "What is a stock?", "chosen": "A stock represents ownership...", "rejected": "I don't know."}
{"prompt": "How do I invest?", "chosen": "To start investing...", "rejected": "Investing is risky."}
```

## Finance Domain Examples

See `data/custom/examples/sample_finance.jsonl` for example finance-related training data.

## Adding Your Own Data

1. Create a JSONL file in this directory
2. Format according to the examples above
3. Use the dataset in training:

```bash
python scripts/train_sft.py --train-file data/custom/your_data.jsonl
```

## Dataset Tips

- **Quality > Quantity**: Better to have 100 high-quality examples than 1000 poor ones
- **Consistent Format**: Keep instruction format consistent across examples
- **Cover Variety**: Include different types of questions/tasks
- **Avoid Repetition**: Don't repeat similar examples too many times
- **Test First**: Start with 50-100 examples to test training pipeline
