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

    def test_cispo_loss_type_accepted(self) -> None:
        # MiniMax-M1 truncated-IS loss (arXiv:2506.13585) — verified present
        # in the installed TRL 1.4.0 GRPOConfig before adding to the Literal.
        assert GRPOConfig(loss_type="cispo").loss_type == "cispo"

    def test_invalid_loss_type_rejected(self) -> None:
        with pytest.raises(ValueError, match="loss_type must be one of"):
            GRPOConfig(loss_type="vespo")  # exists in newer TRL, not in 1.4.0

    def test_vllm_defaults(self) -> None:
        cfg = GRPOConfig()
        assert cfg.use_vllm is False
        assert cfg.vllm_mode == "colocate"
        assert cfg.vllm_gpu_memory_utilization == pytest.approx(0.3)

    def test_invalid_vllm_gpu_memory_utilization(self) -> None:
        with pytest.raises(ValueError, match="vllm_gpu_memory_utilization"):
            GRPOConfig(vllm_gpu_memory_utilization=0.0)
        with pytest.raises(ValueError, match="vllm_gpu_memory_utilization"):
            GRPOConfig(vllm_gpu_memory_utilization=1.5)

    def test_server_mode_warns_but_allows(self) -> None:
        with pytest.warns(UserWarning, match="external vLLM server"):
            GRPOConfig(use_vllm=True, vllm_mode="server")

    def test_colocate_mode_is_silent(self) -> None:
        GRPOConfig(use_vllm=True, vllm_mode="colocate")  # no warning expected

    def test_top_entropy_quantile_default_keeps_all_tokens(self) -> None:
        assert GRPOConfig().top_entropy_quantile == pytest.approx(1.0)

    def test_top_entropy_quantile_range_validated(self) -> None:
        # TRL itself does not range-check this field (1.5 is silently
        # accepted upstream) — the config must catch it instead.
        with pytest.raises(ValueError, match="top_entropy_quantile"):
            GRPOConfig(top_entropy_quantile=1.5)
        with pytest.raises(ValueError, match="top_entropy_quantile"):
            GRPOConfig(top_entropy_quantile=-0.1)

    def test_top_entropy_quantile_accepts_paper_value(self) -> None:
        assert GRPOConfig(top_entropy_quantile=0.2).top_entropy_quantile == pytest.approx(0.2)

    def test_min_p_defaults_off(self) -> None:
        assert GRPOConfig().min_p is None
        assert GRPOConfig().generation_kwargs is None

    def test_min_p_range_validated(self) -> None:
        # Boundaries are valid...
        GRPOConfig(min_p=0.0)
        GRPOConfig(min_p=1.0)
        # ...values outside [0, 1] are not.
        with pytest.raises(ValueError, match="min_p"):
            GRPOConfig(min_p=-0.01)
        with pytest.raises(ValueError, match="min_p"):
            GRPOConfig(min_p=1.01)

    def test_vllm_truncation_sampling_warns(self) -> None:
        # TRL issue #6789: truncation sampling biases the vLLM IS correction.
        with pytest.warns(UserWarning, match="importance-sampling"):
            GRPOConfig(use_vllm=True, min_p=0.05)
        with pytest.warns(UserWarning, match="importance-sampling"):
            GRPOConfig(use_vllm=True, top_p=0.9)
        with pytest.warns(UserWarning, match="importance-sampling"):
            GRPOConfig(use_vllm=True, top_k=50)

    def test_vllm_untruncated_sampling_is_silent(self) -> None:
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("error")  # any UserWarning fails the test
            GRPOConfig(use_vllm=True)  # top_p=1, top_k=0, min_p=None — exact

    def test_non_vllm_truncation_sampling_is_silent(self) -> None:
        # The #6789 bias is vLLM-specific (sampler-vs-trainer logprob
        # renormalization); transformers-side rollout sampling is unaffected.
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            GRPOConfig(min_p=0.05, top_p=0.95)


class TestGRPOTrainingConfig:
    def test_default_reference_config(self) -> None:
        cfg = GRPOTrainingConfig()
        assert cfg.reference_config is not None
        assert cfg.reference_config.name == cfg.model_config.name

    def test_preset_exists(self) -> None:
        assert FINANCE_GRPO_CONFIG.grpo_config.num_generations == 4
