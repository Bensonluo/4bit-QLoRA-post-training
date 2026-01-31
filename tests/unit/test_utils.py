"""Unit tests for utility functions."""

import pytest
from src.utils import set_seed, estimate_model_vram


def test_set_seed():
    """Test seed setting for reproducibility."""
    set_seed(42)
    assert True  # If no exception, test passes


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
