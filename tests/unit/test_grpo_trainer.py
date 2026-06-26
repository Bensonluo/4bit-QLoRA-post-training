"""Unit tests for GRPO trainer setup (without actual model loading)."""

from unittest.mock import MagicMock, patch

import pytest

from config.grpo import GRPOConfig, GRPOTrainingConfig, RewardConfig
from src.training.grpo_trainer import GRPOTrainer


class TestGRPOTrainer:
    def test_build_reward_funcs(self) -> None:
        cfg = GRPOTrainingConfig(
            reward_config=RewardConfig(reward_funcs=["format", "accuracy"]),
        )
        trainer = GRPOTrainer(cfg)
        funcs = trainer._build_reward_funcs()
        assert len(funcs) == 2

    def test_build_grpo_config(self) -> None:
        cfg = GRPOTrainingConfig(
            grpo_config=GRPOConfig(num_generations=4, beta=0.1),
        )
        trainer = GRPOTrainer(cfg)
        trl_cfg = trainer._build_grpo_config()
        assert trl_cfg.num_generations == 4
        assert trl_cfg.beta == pytest.approx(0.1)

    @patch("src.training.grpo_trainer.get_peft_model")
    @patch("src.training.grpo_trainer.prepare_model_for_kbit_training", return_value=None)
    @patch("src.training.grpo_trainer.load_model_and_tokenizer")
    def test_prepare_model_sets_lora(
        self,
        mock_load: MagicMock,
        mock_kbit: MagicMock,
        mock_peft: MagicMock,
    ) -> None:
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_peft.return_value = mock_model
        # Simulate parameters
        param = MagicMock()
        param.numel.return_value = 1000
        param.requires_grad = True
        mock_model.parameters.return_value = [param]
        mock_load.return_value = (mock_model, mock_tokenizer)

        cfg = GRPOTrainingConfig()
        trainer = GRPOTrainer(cfg)
        trainer.prepare_model()

        assert trainer.model is not None
        assert trainer.tokenizer is mock_tokenizer
        mock_peft.assert_called_once()
