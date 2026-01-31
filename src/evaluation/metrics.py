"""Evaluation metrics for language models."""

import torch
from datasets import Dataset
from transformers import PreTrainedModel, PreTrainedTokenizer

from src.utils import console


def compute_perplexity(
    model: PreTrainedModel,
    dataset: Dataset,
    tokenizer: PreTrainedTokenizer,
    max_length: int = 512,
) -> float:
    """Compute perplexity on a dataset.

    Args:
        model: Model to evaluate
        dataset: Dataset to evaluate on
        tokenizer: Tokenizer
        max_length: Maximum sequence length

    Returns:
        Perplexity score
    """
    console.print("[cyan]Computing perplexity...[/cyan]")

    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for i, example in enumerate(dataset):
            # Tokenize input
            if "text" in example:
                text = example["text"]
            elif "instruction" in example:
                text = example["instruction"]
            else:
                continue

            inputs = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
            ).to(model.device)

            # Forward pass
            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss

            total_loss += loss.item() * inputs["input_ids"].size(1)
            total_tokens += inputs["input_ids"].size(1)

            if (i + 1) % 100 == 0:
                console.print(f"  Processed {i + 1} examples...")

    # Average loss and compute perplexity
    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss)).item()

    console.print(f"[green]✓ Perplexity: {perplexity:.2f}[/green]")

    return perplexity


def compute_accuracy(
    model: PreTrainedModel,
    dataset: Dataset,
    tokenizer: PreTrainedTokenizer,
) -> float:
    """Compute token-level accuracy.

    Args:
        model: Model to evaluate
        dataset: Dataset to evaluate on
        tokenizer: Tokenizer

    Returns:
        Accuracy score
    """
    console.print("[cyan]Computing accuracy...[/cyan]")

    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for example in dataset:
            if "text" not in example:
                continue

            inputs = tokenizer(
                example["text"],
                return_tensors="pt",
            ).to(model.device)

            outputs = model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=-1)

            # Compare with input (next token prediction)
            correct += (predictions[:, :-1] == inputs["input_ids"][:, 1:]).sum().item()
            total += inputs["input_ids"].size(1) - 1

    accuracy = correct / total if total > 0 else 0

    console.print(f"[green]✓ Accuracy: {accuracy:.4f}[/green]")

    return accuracy
