"""Data preprocessing utilities."""


from typing import Any

from datasets import Dataset
from transformers import PreTrainedTokenizer

from src.utils.logging import console


class DataCollator:
    """Custom data collator for batching."""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        padding: str = "max_length",
        max_length: int = 512,
        pad_to_multiple_of: int | None = None,
    ) -> None:
        """Initialize data collator.

        Args:
            tokenizer: Tokenizer to use
            padding: Padding strategy ("longest", "max_length")
            max_length: Maximum sequence length
            pad_to_multiple_of: Pad sequence length to multiple of this value
        """
        self.tokenizer = tokenizer
        self.padding = padding
        self.max_length = max_length
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        """Collate features into a batch.

        Args:
            features: List of feature dictionaries

        Returns:
            Batched features
        """
        # Get all keys from first feature
        keys = features[0].keys()

        batch: dict[str, Any] = {}
        for key in keys:
            # Collect values for this key
            values = [f[key] for f in features]

            # Pad if necessary
            if key in ["input_ids", "attention_mask", "labels"]:
                batch[key] = self._pad_sequences(values)
            else:
                batch[key] = values

        return batch

    def _pad_sequences(self, sequences: list[list[int]]) -> list[list[int]]:
        """Pad sequences to same length.

        Args:
            sequences: List of sequences to pad

        Returns:
            Padded sequences
        """
        import torch

        # Convert to tensors
        tensors = [torch.tensor(s, dtype=torch.long) for s in sequences]

        # Pad
        padded = torch.nn.utils.rnn.pad_sequence(
            tensors,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        )

        # Truncate if too long
        if padded.size(1) > self.max_length:
            padded = padded[:, : self.max_length]

        # Pad to max_length if needed
        if self.padding == "max_length" and padded.size(1) < self.max_length:
            pad_length = self.max_length - padded.size(1)
            padded = torch.nn.functional.pad(
                padded,
                (0, pad_length),
                value=self.tokenizer.pad_token_id,
            )

        return padded.tolist()


def format_instruction(
    instruction: str,
    input_text: str = "",
    output: str = "",
    format_type: str = "alpaca",
) -> str:
    """Format instruction, input, and output into a prompt.

    Args:
        instruction: Instruction text
        input_text: Optional input context
        output: Expected output
        format_type: Format style ("alpaca", "chat")

    Returns:
        Formatted prompt string
    """
    if format_type == "alpaca":
        if input_text:
            return f"""### Instruction:
{instruction}

### Input:
{input_text}

### Response:
{output}"""
        else:
            return f"""### Instruction:
{instruction}

### Response:
{output}"""

    elif format_type == "chat":
        # Chat format for models like GPT
        messages = []
        if input_text:
            messages.append({"role": "user", "content": f"{instruction}\n\n{input_text}"})
        else:
            messages.append({"role": "user", "content": instruction})

        if output:
            messages.append({"role": "assistant", "content": output})

        # Simple string representation
        return "\n".join([f"{m['role']}: {m['content']}" for m in messages])

    else:
        raise ValueError(f"Unknown format type: {format_type}")


def tokenize_function(
    examples: dict[str, list[Any]],
    tokenizer: PreTrainedTokenizer,
    max_length: int = 512,
    text_column: str = "text",
) -> dict[str, Any]:
    """Tokenize a batch of examples.

    Args:
        examples: Batch of examples
        tokenizer: Tokenizer to use
        max_length: Maximum sequence length
        text_column: Column name containing text

    Returns:
        Tokenized batch
    """
    tokenized: dict[str, Any] = tokenizer(
        examples[text_column],
        truncation=True,
        max_length=max_length,
        padding="max_length",
    )
    return tokenized


def compute_statistics(dataset: Dataset) -> dict[str, Any]:
    """Compute dataset statistics.

    Args:
        dataset: Dataset to analyze

    Returns:
        Dictionary with statistics
    """
    stats: dict[str, Any] = {
        "num_samples": len(dataset),
        "columns": dataset.column_names,
    }

    # Compute lengths if text column exists
    if "text" in dataset.column_names:
        lengths = [len(text.split()) for text in dataset["text"]]
        stats["avg_length"] = sum(lengths) / len(lengths)
        stats["max_length"] = max(lengths)
        stats["min_length"] = min(lengths)

    return stats


def print_dataset_statistics(dataset: Dataset, name: str = "Dataset") -> None:
    """Print dataset statistics.

    Args:
        dataset: Dataset to analyze
        name: Dataset name for display
    """
    stats = compute_statistics(dataset)

    console.print(f"\n[bold cyan]{name} Statistics[/bold cyan]")
    console.print(f"  Samples: {stats['num_samples']:,}")
    console.print(f"  Columns: {', '.join(stats['columns'])}")

    if "avg_length" in stats:
        console.print(f"  Avg Length: {stats['avg_length']:.1f} words")
        console.print(f"  Max Length: {stats['max_length']} words")
        console.print(f"  Min Length: {stats['min_length']} words")


if __name__ == "__main__":
    # Test formatting
    console.print("[cyan]Testing data formatting...[/cyan]")

    formatted = format_instruction(
        instruction="What is a stock?",
        input_text="",
        output="A stock represents ownership in a company...",
        format_type="alpaca",
    )

    console.print("\n[bold]Formatted Example:[/bold]")
    console.print(formatted)
