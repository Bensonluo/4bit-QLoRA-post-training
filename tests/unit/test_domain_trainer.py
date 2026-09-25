"""Unit tests for the domain adaptation trainer (src/training/domain_trainer.py).

No model loading — dataset classes are patched at their source module and the
five-stage pipeline is exercised via a stub trainer.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

from config.base import DataConfig, LoggingConfig, LoRAConfig, ModelConfig, TrainingConfig
from src.training.domain_trainer import DomainAdaptationTrainer, run_domain_adaptation
from src.training.sft_trainer import SFTTrainer


def _configs(output_dir: str = "./outputs/test-domain") -> dict[str, Any]:
    return {
        "model_config": ModelConfig(name="Qwen/Qwen2.5-0.5B-Instruct"),
        "training_config": TrainingConfig(output_dir=output_dir),
        "lora_config": LoRAConfig(),
        "data_config": DataConfig(dataset_name="fake/data"),
        "logging_config": LoggingConfig(),
    }


class _StubDataset:
    """Records calls; mirrors the BaseDataset surface prepare_data uses."""

    instances: list[_StubDataset] = []

    def __init__(self, data_path: str, max_samples: int | None = None) -> None:
        self.data_path = data_path
        self.max_samples = max_samples
        self.load_calls = 0
        self.format_calls: list[dict[str, Any]] = []
        _StubDataset.instances.append(self)

    def load(self) -> None:
        self.load_calls += 1

    def split_dataset(
        self, validation_split: float = 0.1, seed: int | None = None
    ) -> tuple[list[int], list[int]]:
        return [1, 2, 3], [4]

    def format_for_training(self, tokenizer: Any, max_length: int = 1024) -> list[int]:
        self.format_calls.append({"tokenizer": tokenizer, "max_length": max_length})
        return [0]


class TestDomainAdaptationTrainer:
    def test_is_sft_trainer_with_domain_name(self, tmp_path: Any) -> None:
        trainer = DomainAdaptationTrainer(domain_name="medical", **_configs(str(tmp_path)))
        assert isinstance(trainer, SFTTrainer)
        assert trainer.domain_name == "medical"
        assert trainer.model is None and trainer.tokenizer is None  # lazy loading

    @patch("src.data.AlpacaDataset", _StubDataset)
    def test_prepare_data_uses_alpaca_for_unknown_domain(self, tmp_path: Any) -> None:
        _StubDataset.instances = []
        cfgs = _configs(str(tmp_path))
        trainer = DomainAdaptationTrainer(domain_name="legal", **cfgs)
        with patch("src.training.domain_trainer.console.print"):
            trainer.prepare_data()

        assert len(_StubDataset.instances) == 1
        stub = _StubDataset.instances[0]
        assert stub.load_calls == 1
        # train + eval both formatted with the model max_length
        assert len(stub.format_calls) == 2
        assert all(c["max_length"] == cfgs["model_config"].max_length for c in stub.format_calls)
        assert trainer.train_dataset == [0] and trainer.eval_dataset == [0]

    @patch("src.data.FinanceDataset", _StubDataset)
    def test_prepare_data_uses_finance_dataset_for_finance(self, tmp_path: Any) -> None:
        _StubDataset.instances = []
        trainer = DomainAdaptationTrainer(domain_name="finance", **_configs(str(tmp_path)))
        with patch("src.training.domain_trainer.console.print"):
            trainer.prepare_data()
        assert len(_StubDataset.instances) == 1
        assert _StubDataset.instances[0].data_path == "fake/data"

    @patch("src.data.AlpacaDataset", _StubDataset)
    def test_prepare_data_passes_max_samples(self, tmp_path: Any) -> None:
        _StubDataset.instances = []
        cfgs = _configs(str(tmp_path))
        cfgs["data_config"] = DataConfig(dataset_name="fake/data", max_samples=50)
        trainer = DomainAdaptationTrainer(domain_name="medical", **cfgs)
        with patch("src.training.domain_trainer.console.print"):
            trainer.prepare_data()
        assert _StubDataset.instances[0].max_samples == 50


class TestRunDomainAdaptation:
    def test_runs_five_stage_pipeline(self) -> None:
        stub = MagicMock(spec=DomainAdaptationTrainer)
        with (
            patch(
                "src.training.domain_trainer.DomainAdaptationTrainer", return_value=stub
            ) as mock_cls,
            patch("src.training.domain_trainer.console.print"),
        ):
            run_domain_adaptation(domain_name="finance", **_configs())

        mock_cls.assert_called_once_with(domain_name="finance", **_configs())
        stub.prepare_model.assert_called_once()
        stub.prepare_data.assert_called_once()
        stub.setup_trainer.assert_called_once()
        stub.train.assert_called_once()
        stub.evaluate.assert_called_once()
