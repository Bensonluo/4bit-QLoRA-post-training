"""Dataset loader for GRPO training.

Expected dataset formats:
  1. {"prompt": "...", "answer": "..."}
  2. {"prompt": "...", "reference": "..."}
  3. {"prompt": "..."}
"""

from __future__ import annotations

from typing import Any

from datasets import Dataset, load_dataset

from src.data.base import BaseDataset
from src.utils.logging import console


class GRPODataset(BaseDataset):
    """Dataset for GRPO training.

    Attributes:
        data_path: Path or Hugging Face dataset identifier.
        max_samples: Maximum number of samples to use.
        prompt_key: Column key for prompts.
        answer_key: Column key for reference answers.
        reference_key: Column key for reference responses.
    """

    def __init__(
        self,
        data_path: str,
        max_samples: int | None = None,
        prompt_key: str = "prompt",
        answer_key: str = "answer",
        reference_key: str = "reference",
    ) -> None:
        """Initialize GRPO dataset."""
        super().__init__(data_path=data_path, max_samples=max_samples)
        self.prompt_key = prompt_key
        self.answer_key = answer_key
        self.reference_key = reference_key

    def load(self) -> Dataset:
        """Load GRPO dataset."""
        console.print(f"[cyan]Loading GRPO dataset: {self.data_path}[/cyan]")

        try:
            dataset = load_dataset(self.data_path, split="train")
        except Exception as e:
            console.print(f"[yellow]Warning: Could not load from HF: {e}[/yellow]")
            try:
                dataset = load_dataset("json", data_files=self.data_path, split="train")
            except Exception as e2:
                raise RuntimeError(f"Failed to load dataset: {e2}") from None

        if self.max_samples and len(dataset) > self.max_samples:
            dataset = dataset.select(range(self.max_samples))
            console.print(f"[yellow]Limited to {self.max_samples} samples[/yellow]")

        self.dataset = self._validate_and_rename(dataset)
        console.print(f"[green]✓ Loaded {len(self.dataset)} GRPO samples[/green]")
        return self.dataset

    def _validate_and_rename(self, dataset: Dataset) -> Dataset:
        """Ensure required columns exist and standardize names."""
        column_names = dataset.column_names
        if self.prompt_key not in column_names:
            raise ValueError(
                f"Dataset must contain a '{self.prompt_key}' column. Found columns: {column_names}"
            )

        rename_map: dict[str, str] = {}
        if self.prompt_key != "prompt":
            rename_map[self.prompt_key] = "prompt"
        if self.answer_key in column_names and self.answer_key != "answer":
            rename_map[self.answer_key] = "answer"
        if self.reference_key in column_names and self.reference_key != "reference":
            rename_map[self.reference_key] = "reference"

        if rename_map:
            dataset = dataset.rename_columns(rename_map)

        # Ensure prompt is a string
        def _normalize(example: dict[str, Any]) -> dict[str, Any]:
            prompt = example["prompt"]
            if isinstance(prompt, list):
                # conversation format -> keep as-is
                example["prompt"] = prompt
            else:
                example["prompt"] = str(prompt)
            return example

        return dataset.map(_normalize)

    def format_for_training(
        self,
        tokenizer: Any,
        max_length: int = 512,
    ) -> Dataset:
        """Format dataset for GRPO training.

        For GRPO, the tokenizer is managed by the trainer during generation,
        so we only validate/return the dataset with standardized columns.
        """
        if self.dataset is None:
            self.load()
        _ = tokenizer, max_length  # kept for signature parity with BaseDataset
        return self.dataset
