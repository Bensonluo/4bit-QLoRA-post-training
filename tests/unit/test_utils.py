"""Unit tests for utility functions."""

import random

import numpy as np
import torch

from src.utils import estimate_model_vram, set_seed
from src.utils.seed import get_seed


def test_set_seed():
    """Test seed setting for reproducibility."""
    set_seed(42)
    assert True  # If no exception, test passes


def test_set_seed_makes_all_three_generators_deterministic():
    """Same seed → identical draws from random, numpy, and torch."""
    set_seed(123)
    py_a, np_a, torch_a = random.random(), np.random.rand(), torch.rand(3)
    set_seed(123)
    py_b, np_b, torch_b = random.random(), np.random.rand(), torch.rand(3)

    assert py_a == py_b
    assert np_a == np_b
    assert torch.allclose(torch_a, torch_b)


def test_set_seed_deterministic_flag_sets_cudnn_flags():
    set_seed(7, deterministic=True)
    assert torch.backends.cudnn.deterministic is True
    assert torch.backends.cudnn.benchmark is False
    set_seed(7)  # restore non-deterministic defaults
    assert torch.backends.cudnn.deterministic is False


def test_get_seed_is_in_uint32_range():
    assert 0 <= get_seed() < 2**32


def test_estimate_model_vram():
    """Test VRAM estimation."""
    estimate = estimate_model_vram(
        model_name="Qwen/Qwen2.5-1.5B-Instruct",
        quantization_bits=4,
        lora_r=16,
        batch_size=1,
        max_length=1024,
    )

    assert "total_gb" in estimate
    assert estimate["total_gb"] > 0
    assert estimate["total_gb"] < 10  # Should be under 10GB with QLoRA
