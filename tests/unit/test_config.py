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


def test_loftq_defaults_and_validation():
    """LoftQ init knobs: defaults match the QLoRA base, ranges enforced."""
    assert LoRAConfig().loftq_bits == 4
    assert LoRAConfig().loftq_iter == 1

    # "loftq" is accepted at our config layer; peft's dict arrives from trainers.
    cfg = LoRAConfig(init_lora_weights="loftq", loftq_bits=4, loftq_iter=2)
    assert cfg.init_lora_weights == "loftq"

    with pytest.raises(ValueError, match="loftq_bits"):
        LoRAConfig(loftq_bits=0)
    with pytest.raises(ValueError, match="loftq_bits"):
        LoRAConfig(loftq_bits=-4)
    with pytest.raises(ValueError, match="loftq_iter"):
        LoRAConfig(loftq_iter=0)


def test_neftune_noise_alpha_validation():
    """NEFTune alpha must be positive when set; default off."""
    assert TrainingConfig().neftune_noise_alpha is None

    config = TrainingConfig(neftune_noise_alpha=5.0)
    assert config.neftune_noise_alpha == 5.0

    with pytest.raises(ValueError, match="neftune_noise_alpha"):
        TrainingConfig(neftune_noise_alpha=0.0)
    with pytest.raises(ValueError, match="neftune_noise_alpha"):
        TrainingConfig(neftune_noise_alpha=-5.0)


def test_torch_compile_defaults_off():
    """torch.compile is opt-in; default keeps current behavior."""
    assert TrainingConfig().torch_compile is False
    assert TrainingConfig(torch_compile=True).torch_compile is True


def test_gradient_checkpointing_use_reentrant_defaults_false() -> None:
    """Non-reentrant checkpointing is the PyTorch-recommended default (PEFT-safe)."""
    assert TrainingConfig().gradient_checkpointing_use_reentrant is False
    assert TrainingConfig(gradient_checkpointing=False).gradient_checkpointing is False


def test_torch_empty_cache_steps_defaults_off() -> None:
    """Periodic empty_cache is opt-in; None keeps current behavior."""
    assert TrainingConfig().torch_empty_cache_steps is None
    assert TrainingConfig(torch_empty_cache_steps=100).torch_empty_cache_steps == 100


def test_auto_find_batch_size_defaults_off() -> None:
    """OOM auto-retry is opt-in; default keeps current behavior."""
    assert TrainingConfig().auto_find_batch_size is False
    assert TrainingConfig(auto_find_batch_size=True).auto_find_batch_size is True


def test_auto_find_batch_size_rejected_with_zero3() -> None:
    """HF Trainer hard-errors on this combo — fail at config construction."""
    with pytest.raises(ValueError, match="ZeRO-3"):
        TrainingConfig(
            auto_find_batch_size=True,
            deepspeed_config="config/distributed/deepspeed_configs/zero_stage_3_offload.json",
        )


def test_auto_find_batch_size_allowed_with_zero2() -> None:
    """ZeRO-2/FSDP paths keep the auto-retry capability."""
    cfg = TrainingConfig(
        auto_find_batch_size=True,
        deepspeed_config="config/distributed/deepspeed_configs/zero_stage_2.json",
    )
    assert cfg.auto_find_batch_size is True


def test_train_sampling_strategy_defaults_random() -> None:
    """Sampler defaults to plain random shuffling."""
    assert TrainingConfig().train_sampling_strategy == "random"


def test_train_sampling_strategy_invalid_rejected() -> None:
    """Sampler typos fail at config construction, not in the dataloader."""
    with pytest.raises(ValueError, match="train_sampling_strategy"):
        TrainingConfig(train_sampling_strategy="groupbylength")


# ─── LoRA-GA (gradient-approximation init, arXiv:2407.05000) ─────────────


def test_lora_ga_defaults() -> None:
    """LoRA-GA fields default to the paper-recommended settings."""
    cfg = LoRAConfig(init_lora_weights="lora_ga")
    assert cfg.lora_ga_direction == "ArB2r"
    assert cfg.lora_ga_scale == "stable"
    assert cfg.lora_ga_stable_gamma == 16
    assert cfg.lora_ga_calibration_batches == 4
    assert cfg.lora_ga_cache_file is None


def test_lora_ga_field_validation() -> None:
    """Invalid LoRA-GA parameters fail at construction, not in peft."""
    with pytest.raises(ValueError, match="lora_ga_direction"):
        LoRAConfig(init_lora_weights="lora_ga", lora_ga_direction="diag")
    with pytest.raises(ValueError, match="lora_ga_scale"):
        LoRAConfig(init_lora_weights="lora_ga", lora_ga_scale="fast")
    with pytest.raises(ValueError, match="lora_ga_stable_gamma"):
        LoRAConfig(init_lora_weights="lora_ga", lora_ga_stable_gamma=0)
    with pytest.raises(ValueError, match="lora_ga_calibration_batches"):
        LoRAConfig(init_lora_weights="lora_ga", lora_ga_calibration_batches=0)


def _cuda_platform() -> MagicMock:
    platform = MagicMock()
    platform.is_cuda = True
    platform.is_mps = False
    platform.device = "cuda"
    return platform


def test_sft_config_rejects_lora_ga_with_quantized_base() -> None:
    """LoRA-GA needs full-precision gradients — reject 4/8-bit at SFTConfig build."""
    from config.sft import SFTConfig

    with (
        patch("src.utils.platform_utils.get_platform", return_value=_cuda_platform()),
        pytest.raises(ValueError, match="full-precision base"),
    ):
        SFTConfig(
            model=ModelConfig(name="qwen-test", quantization_bits=4),
            lora=LoRAConfig(init_lora_weights="lora_ga"),
        )


def test_sft_config_accepts_lora_ga_full_precision() -> None:
    """Full-precision base + LoRA-GA is the supported combination."""
    from config.sft import SFTConfig

    with patch("src.utils.platform_utils.get_platform", return_value=_cuda_platform()):
        cfg = SFTConfig(
            model=ModelConfig(name="qwen-test", quantization_bits=None),
            lora=LoRAConfig(init_lora_weights="lora_ga"),
        )
    assert cfg.lora.init_lora_weights == "lora_ga"


def test_dpo_config_rejects_lora_ga() -> None:
    """No validated LoRA-GA calibration exists for preference losses."""
    from config.dpo import DPOTrainingConfig

    with pytest.raises(ValueError, match="not supported for DPO"):
        DPOTrainingConfig(lora_config=LoRAConfig(init_lora_weights="lora_ga"))


def test_grpo_config_rejects_lora_ga() -> None:
    """No validated LoRA-GA calibration exists for policy-gradient losses."""
    from config.grpo import GRPOTrainingConfig

    with pytest.raises(ValueError, match="not supported for GRPO"):
        GRPOTrainingConfig(lora_config=LoRAConfig(init_lora_weights="lora_ga"))


# --- LoRA per-module surface (rank_pattern / alpha_pattern / exclude_modules) ---


def test_lora_pattern_defaults() -> None:
    cfg = LoRAConfig()
    assert cfg.rank_pattern is None
    assert cfg.alpha_pattern is None
    assert cfg.exclude_modules is None


def test_rank_pattern_requires_positive_int_values() -> None:
    # Valid shapes pass...
    LoRAConfig(rank_pattern={"^model.layers.0.q_proj": 32})
    LoRAConfig(rank_pattern={"q_proj": 8})
    # ...invalid shapes fail at config construction, not adapter injection.
    for bad in (0, -1, True, "16", 1.5):
        with pytest.raises(ValueError, match="rank_pattern"):
            LoRAConfig(rank_pattern={"q_proj": bad})
    with pytest.raises(ValueError, match="rank_pattern"):
        LoRAConfig(rank_pattern={"": 16})  # empty regex key


def test_alpha_pattern_requires_positive_int_values() -> None:
    LoRAConfig(alpha_pattern={"^model.layers.0.q_proj": 64})
    for bad in (0, -2, True, "32", 2.5):
        with pytest.raises(ValueError, match="alpha_pattern"):
            LoRAConfig(alpha_pattern={"q_proj": bad})
    with pytest.raises(ValueError, match="alpha_pattern"):
        LoRAConfig(alpha_pattern={"": 32})


def test_exclude_modules_type_validated() -> None:
    # Regex string and list of strings are the peft-documented shapes.
    LoRAConfig(exclude_modules="lm_head")
    LoRAConfig(exclude_modules=["lm_head", "score"])
    with pytest.raises(ValueError, match="exclude_modules"):
        LoRAConfig(exclude_modules=42)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="exclude_modules"):
        LoRAConfig(exclude_modules=["lm_head", 7])  # type: ignore[list-item]
    with pytest.raises(ValueError, match="exclude_modules"):
        LoRAConfig(exclude_modules=["lm_head", ""])


def test_peak_flops_per_device_validation() -> None:
    """MFU-calibration knob: default None (TRL H100 default), positive when set."""
    assert TrainingConfig().peak_flops_per_device is None
    cfg = TrainingConfig(peak_flops_per_device=3.12e14)  # A100 bf16 dense
    assert cfg.peak_flops_per_device == 3.12e14
    with pytest.raises(ValueError, match="peak_flops_per_device"):
        TrainingConfig(peak_flops_per_device=-1.0)
    with pytest.raises(ValueError, match="peak_flops_per_device"):
        TrainingConfig(peak_flops_per_device=0.0)
