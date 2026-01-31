"""Dataset loaders for various data formats."""

from typing import Optional, Dict, Any

from datasets import Dataset, load_dataset

from src.data.base import BaseDataset
from src.utils.logging import console


class AlpacaDataset(BaseDataset):
    """Dataset in Alpaca format (instruction, input, output).

    Example:
        {
            "instruction": "Explain quantum entanglement",
            "input": "",
            "output": "Quantum entanglement is..."
        }
    """

    def load(self) -> Dataset:
        """Load Alpaca-format dataset."""
        console.print(f"[cyan]Loading Alpaca dataset: {self.data_path}[/cyan]")

        # Try loading from Hugging Face
        try:
            dataset = load_dataset(self.data_path, split="train")
        except Exception as e:
            console.print(f"[yellow]Warning: Could not load from HF: {e}[/yellow]")
            # Try loading as local file
            try:
                dataset = load_dataset("json", data_files=self.data_path, split="train")
            except Exception as e2:
                raise RuntimeError(f"Failed to load dataset: {e2}")

        # Limit samples if specified
        if self.max_samples and len(dataset) > self.max_samples:
            dataset = dataset.select(range(self.max_samples))
            console.print(f"[yellow]Limited to {self.max_samples} samples[/yellow]")

        self.dataset = dataset
        console.print(f"[green]✓ Loaded {len(dataset)} samples[/green]")

        return dataset

    def format_for_training(
        self,
        tokenizer,
        max_length: int = 512,
    ) -> Dataset:
        """Format Alpaca dataset for SFT training."""
        if self.dataset is None:
            self.load()

        def format_prompt(example: dict) -> dict:
            """Format a single example."""
            instruction = example["instruction"]
            input_text = example.get("input", "")
            output = example["output"]

            # Build prompt
            if input_text:
                prompt = f"""### Instruction:
{instruction}

### Input:
{input_text}

### Response:
{output}"""
            else:
                prompt = f"""### Instruction:
{instruction}

### Response:
{output}"""

            # Tokenize
            tokenized = tokenizer(
                prompt,
                truncation=True,
                max_length=max_length,
                padding="max_length",
            )

            # For causal LM, labels are same as input_ids
            tokenized["labels"] = tokenized["input_ids"].copy()

            return tokenized

        console.print("[cyan]Formatting dataset for training...[/cyan]")
        formatted_dataset = self.dataset.map(format_prompt, remove_columns=self.dataset.column_names)
        console.print("[green]✓ Dataset formatted[/green]")

        return formatted_dataset


class FinanceDataset(AlpacaDataset):
    """Finance-specific dataset (extends Alpaca format).

    This class provides finance-specific filtering and preprocessing.
    """

    FINANCE_KEYWORDS = [
        "stock", "investment", "finance", "financial", "trading",
        "portfolio", "dividend", "market", "economy", "business",
        "money", "capital", "asset", "fund", "equity", "bond",
        "currency", "crypto", "bitcoin", "earnings", "revenue",
    ]

    def load(self) -> Dataset:
        """Load and filter dataset for finance content."""
        console.print(f"[cyan]Loading finance dataset: {self.data_path}[/cyan]")

        # Load base dataset
        super().load()

        # Filter for finance-related content
        if self.dataset:
            console.print("[cyan]Filtering for finance-related content...[/cyan]")
            filtered_dataset = self._filter_finance(self.dataset)
            console.print(f"[green]✓ Filtered to {len(filtered_dataset)} finance samples[/green]")
            self.dataset = filtered_dataset

        return self.dataset

    def _filter_finance(self, dataset: Dataset) -> Dataset:
        """Filter dataset to only include finance-related examples."""
        def is_finance_related(example: dict) -> bool:
            """Check if example is finance-related."""
            text = (
                example.get("instruction", "") +
                " " +
                example.get("input", "") +
                " " +
                example.get("output", "")
            ).lower()

            return any(keyword in text for keyword in self.FINANCE_KEYWORDS)

        return dataset.filter(is_finance_related)


class PreferenceDataset(BaseDataset):
    """Dataset for DPO training with chosen/rejected responses.

    Example:
        {
            "prompt": "What is the best investment strategy?",
            "chosen": "The best investment strategy depends...",
            "rejected": "I don't know."
        }
    """

    def load(self) -> Dataset:
        """Load preference dataset."""
        console.print(f"[cyan]Loading preference dataset: {self.data_path}[/cyan]")

        try:
            dataset = load_dataset(self.data_path, split="train")
        except Exception as e:
            console.print(f"[yellow]Warning: Could not load from HF: {e}[/yellow]")
            try:
                dataset = load_dataset("json", data_files=self.data_path, split="train")
            except Exception as e2:
                raise RuntimeError(f"Failed to load dataset: {e2}")

        if self.max_samples and len(dataset) > self.max_samples:
            dataset = dataset.select(range(self.max_samples))

        self.dataset = dataset
        console.print(f"[green]✓ Loaded {len(dataset)} preference pairs[/green]")

        return dataset

    def format_for_training(
        self,
        tokenizer,
        max_length: int = 512,
    ) -> Dataset:
        """Format preference dataset for DPO training."""
        if self.dataset is None:
            self.load()

        def format_preference(example: dict) -> dict:
            """Format a single preference pair."""
            prompt = example["prompt"]
            chosen = example["chosen"]
            rejected = example["rejected"]

            # Tokenize prompt
            tokenized_prompt = tokenizer(
                prompt,
                truncation=True,
                max_length=max_length // 3,
                padding=False,
                return_tensors=None,
            )

            # Tokenize chosen
            tokenized_chosen = tokenizer(
                chosen,
                truncation=True,
                max_length=max_length,
                padding=False,
                return_tensors=None,
            )

            # Tokenize rejected
            tokenized_rejected = tokenizer(
                rejected,
                truncation=True,
                max_length=max_length,
                padding=False,
                return_tensors=None,
            )

            return {
                "prompt_input_ids": tokenized_prompt["input_ids"],
                "prompt_attention_mask": tokenized_prompt["attention_mask"],
                "chosen_input_ids": tokenized_chosen["input_ids"],
                "chosen_attention_mask": tokenized_chosen["attention_mask"],
                "rejected_input_ids": tokenized_rejected["input_ids"],
                "rejected_attention_mask": tokenized_rejected["attention_mask"],
            }

        console.print("[cyan]Formatting preference dataset...[/cyan]")
        formatted_dataset = self.dataset.map(format_preference, remove_columns=self.dataset.column_names)
        console.print("[green]✓ Dataset formatted[/green]")

        return formatted_dataset


def load_custom_dataset(
    file_path: str,
    format_type: str = "alpaca",
    max_samples: Optional[int] = None,
) -> Dataset:
    """Load custom dataset from local file.

    Args:
        file_path: Path to JSONL or JSON file
        format_type: Type of dataset ("alpaca", "preference")
        max_samples: Maximum number of samples to load

    Returns:
        Loaded dataset
    """
    console.print(f"[cyan]Loading custom dataset: {file_path}[/cyan]")
    console.print(f"  Format: {format_type}")

    # Load dataset
    dataset = load_dataset("json", data_files=file_path, split="train")

    if max_samples and len(dataset) > max_samples:
        dataset = dataset.select(range(max_samples))
        console.print(f"[yellow]Limited to {max_samples} samples[/yellow]")

    console.print(f"[green]✓ Loaded {len(dataset)} samples[/green]")

    return dataset


if __name__ == "__main__":
    # Test dataset loading
    print("Testing dataset loaders...")

    # Test with small Alpaca dataset
    alpaca_ds = AlpacaDataset(
        data_path="yahma/alpaca-cleaned",
        max_samples=100,
    )
    alpaca_ds.load()
    print(f"✓ AlpacaDataset: {len(alpaca_ds)} samples")

    # Test finance filtering
    finance_ds = FinanceDataset(
        data_path="yahma/alpaca-cleaned",
        max_samples=1000,
    )
    finance_ds.load()
    print(f"✓ FinanceDataset: {len(finance_ds)} samples")
