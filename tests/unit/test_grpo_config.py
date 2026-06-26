"""Unit tests for GRPO configuration."""

import pytest

from config.grpo import (
    FINANCE_GRPO_CONFIG,
    GRPOConfig,
    GRPOTrainingConfig,
    RewardConfig,
)


class TestRewardConfig:
    def test_default_reward_config(self) -> None:
        cfg = RewardConfig()
        assert cfg.reward_funcs == ["format", "accuracy"]
        assert cfg.judge_model is None

    def test_invalid_reward_function(self) -> None:
        with pytest.raises(ValueError, match="Invalid reward functions"):
            RewardConfig(reward_funcs=["unknown"])

    def test_llm_judge_requires_model(self) -> None:
        with pytest.raises(ValueError, match="judge_model must be specified"):
            RewardConfig(reward_funcs=["llm_judge"])


class TestGRPOConfig:
    def test_default_grpo_config(self) -> None:
        cfg = GRPOConfig()
        assert cfg.beta == 0.04
        assert cfg.num_generations == 8
        assert cfg.max_completion_length == 256

    def test_invalid_beta(self) -> None:
        with pytest.raises(ValueError, match="beta must be non-negative"):
            GRPOConfig(beta=-0.1)

    def test_invalid_num_generations(self) -> None:
        with pytest.raises(ValueError, match="num_generations must be at least 2"):
            GRPOConfig(num_generations=1)

    def test_invalid_epsilon(self) -> None:
        with pytest.raises(ValueError, match="epsilon must be between 0 and 1"):
            GRPOConfig(epsilon=1.5)


class TestGRPOTrainingConfig:
    def test_default_reference_config(self) -> None:
        cfg = GRPOTrainingConfig()
        assert cfg.reference_config is not None
        assert cfg.reference_config.name == cfg.model_config.name

    def test_preset_exists(self) -> None:
        assert FINANCE_GRPO_CONFIG.grpo_config.num_generations == 4
