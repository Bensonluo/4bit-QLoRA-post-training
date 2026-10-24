"""Data APIs; protocol inspection does not import the training stack."""

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.data.base import BaseDataset
    from src.data.loaders import (
        AlpacaDataset,
        FinanceDataset,
        PreferenceDataset,
        load_custom_dataset,
    )
    from src.data.medical_dataset import MedicalEntityDataset
    from src.data.preprocessors import (
        DataCollator,
        compute_statistics,
        format_instruction,
        print_dataset_statistics,
        tokenize_function,
    )

__all__ = [
    # Base
    "BaseDataset",
    # Loaders
    "AlpacaDataset",
    "FinanceDataset",
    "MedicalEntityDataset",
    "PreferenceDataset",
    "load_custom_dataset",
    # Preprocessors
    "DataCollator",
    "format_instruction",
    "tokenize_function",
    "compute_statistics",
    "print_dataset_statistics",
]

_MODULES = {
    "BaseDataset": "base",
    "AlpacaDataset": "loaders",
    "FinanceDataset": "loaders",
    "PreferenceDataset": "loaders",
    "load_custom_dataset": "loaders",
    "MedicalEntityDataset": "medical_dataset",
    "DataCollator": "preprocessors",
    "format_instruction": "preprocessors",
    "tokenize_function": "preprocessors",
    "compute_statistics": "preprocessors",
    "print_dataset_statistics": "preprocessors",
}


def __getattr__(name: str) -> Any:
    if name not in _MODULES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f"{__name__}.{_MODULES[name]}"), name)
    globals()[name] = value
    return value
