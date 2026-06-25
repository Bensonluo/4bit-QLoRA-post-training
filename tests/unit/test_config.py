"""Unit tests for configuration."""

from unittest.mock import MagicMock, patch

import pytest

from config.base import LoRAConfig, ModelConfig, TrainingConfig


def test_model_config():
    """Test ModelConfig creation and validation."""
    mock_platform = MagicMock()
    mock_platform.is_cuda = True
    mock_platform.is_mps = False
    mock_platform.device = "cuda"

    with patch("src.utils.platform_utils.get_platform", return_value=mock_platform):
        config = ModelConfig(
            name="Qwen/Qwen2.5-1.5B-Instruct",
            quantization_bits=4,
        )

    assert config.name == "Qwen/Qwen2.5-1.5B-Instruct"
    assert config.quantization_bits == 4


def test_invalid_quantization():
    """Test that invalid quantization is rejected."""
    with pytest.raises(ValueError):
        ModelConfig(quantization_bits=16)  # Not 4 or 8


def test_training_config():
    """Test TrainingConfig."""
    config = TrainingConfig(
        batch_size=1,
        gradient_accumulation_steps=8,
    )

    assert config.batch_size == 1
    assert config.effective_batch_size == 8


def test_lora_config():
    """Test LoRAConfig."""
    config = LoRAConfig(r=16, lora_alpha=32)

    assert config.r == 16
    assert config.lora_alpha == 32
