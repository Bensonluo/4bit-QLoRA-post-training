"""Reproducibility utilities for random seed setting."""

import random
from typing import Optional

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = False) -> None:
    """Set random seeds for reproducibility.

    Args:
        seed: Random seed value
        deterministic: Whether to use deterministic algorithms (slower but fully reproducible)
    """
    random.seed(seed)
    np.random.seed(seed)

    try:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        else:
            torch.backends.cudnn.deterministic = False
            if torch.cuda.is_available():
                torch.backends.cudnn.benchmark = True

    except Exception as e:
        print(f"Warning: Could not fully set seed for PyTorch: {e}")


def get_seed() -> int:
    """Get a random seed value.

    Returns:
        Random seed based on current time
    """
    import time

    return int(time.time()) % (2**32)


if __name__ == "__main__":
    # Test seed setting
    print("Testing seed setting...")

    set_seed(42)

    # Generate some random values
    print(f"Random: {random.random()}")
    print(f"NumPy: {np.random.rand()}")
    print(f"PyTorch: {torch.rand(1).item()}")
