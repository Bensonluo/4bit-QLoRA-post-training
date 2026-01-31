"""Base dataset classes for different training formats."""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional

from datasets import Dataset
from transformers import PreTrainedTokenizer


class BaseDataset(ABC):
    """Abstract base class for datasets."""

    def __init__(
        self,
        data_path: str,
        max_samples: Optional[int] = None,
    ):
        """Initialize dataset.

        Args:
            data_path: Path or Hugging Face dataset identifier
            max_samples: Maximum number of samples to use
        """
        self.data_path = data_path
        self.max_samples = max_samples
        self.dataset = None

    @abstractmethod
    def load(self) -> Dataset:
        """Load dataset from path or Hugging Face.

        Returns:
            Loaded dataset
        """
        pass

    @abstractmethod
    def format_for_training(
        self,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512,
    ) -> Dataset:
        """Format dataset for training.

        Args:
            tokenizer: Tokenizer to use
            max_length: Maximum sequence length

        Returns:
            Formatted dataset
        """
        pass

    def split_dataset(
        self,
        validation_split: float = 0.1,
        seed: int = 42,
    ) -> tuple[Dataset, Dataset]:
        """Split dataset into train and validation.

        Args:
            validation_split: Fraction of data for validation
            seed: Random seed for reproducibility

        Returns:
            Tuple of (train_dataset, val_dataset)
        """
        if self.dataset is None:
            self.load()

        split_dict = self.dataset.train_test_split(
            test_size=validation_split,
            seed=seed,
        )

        return split_dict["train"], split_dict["test"]

    def __len__(self) -> int:
        """Get dataset length."""
        if self.dataset is None:
            return 0
        return len(self.dataset)

    def __repr__(self) -> str:
        """String representation."""
        return f"{self.__class__.__name__}(path={self.data_path}, samples={self.max_samples})"
