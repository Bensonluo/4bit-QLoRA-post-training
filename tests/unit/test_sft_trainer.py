"""Unit tests for SFT trainer setup (src/training/sft_trainer.py).

setup_trainer() is exercised with the HF Trainer and data collator patched
out; TrainingArguments is constructed for real so the passthrough surface
(liger / neftune / torch_compile / warmup semantics) is pinned against
upstream regressions.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import torch
from transformers import TrainingArguments

from config.base import DataConfig, LoggingConfig, LoRAConfig, ModelConfig, TrainingConfig
from src.training.sft_trainer import MemoryCallback, SFTTrainer, run_sft_training


def _make_trainer(tmp_path: Path, **training_overrides: Any) -> SFTTrainer:
    training_overrides.setdefault("bf16", False)
    return SFTTrainer(
        model_config=ModelConfig(name="Qwen/Qwen2.5-0.5B-Instruct"),
        training_config=TrainingConfig(output_dir=str(tmp_path / "sft"), **training_overrides),
        lora_config=LoRAConfig(),
        data_config=DataConfig(max_samples=10),
        logging_config=LoggingConfig(),
    )


def _stub_stage(trainer: SFTTrainer) -> None:
    """Fill the post-prepare_model/prepare_data attributes with fakes."""
    trainer.model = MagicMock()
    trainer.tokenizer = MagicMock()
    trainer.train_dataset = ["a", "b"]
    trainer.eval_dataset = None


def _as_str(value: Any) -> str:
    return getattr(value, "value", value)


class TestSetupTrainer:
    @patch("src.training.sft_trainer.AttentionMaskCausalCollator")
    @patch("src.training.sft_trainer.Trainer")
    def test_builds_valid_training_arguments(
        self, mock_trainer_cls: MagicMock, mock_collator_cls: MagicMock, tmp_path: Path
    ) -> None:
        trainer = _make_trainer(tmp_path)
        _stub_stage(trainer)
        trainer.setup_trainer()

        kwargs = mock_trainer_cls.call_args.kwargs
        args = kwargs["args"]
        assert isinstance(args, TrainingArguments)
        assert args.output_dir == str(tmp_path / "sft")
        # eval_dataset=None → evaluation disabled, best-model loading off
        assert _as_str(args.eval_strategy) == "no"
        assert args.load_best_model_at_end is False
        # transformers >= 5.x: float warmup_steps keeps ratio semantics
        assert args.warmup_steps == pytest.approx(0.03)
        assert kwargs["train_dataset"] == trainer.train_dataset

    @patch("src.training.sft_trainer.AttentionMaskCausalCollator")
    @patch("src.training.sft_trainer.Trainer")
    def test_accelerator_flags_default_off(
        self, mock_trainer_cls: MagicMock, _mock_collator: MagicMock, tmp_path: Path
    ) -> None:
        trainer = _make_trainer(tmp_path)
        _stub_stage(trainer)
        trainer.setup_trainer()

        args = mock_trainer_cls.call_args.kwargs["args"]
        assert args.use_liger_kernel is False
        assert args.neftune_noise_alpha is None
        assert args.torch_compile is False

    @patch("src.training.sft_trainer.AttentionMaskCausalCollator")
    @patch("src.training.sft_trainer.Trainer")
    def test_forwards_liger_neftune_torch_compile(
        self, mock_trainer_cls: MagicMock, _mock_collator: MagicMock, tmp_path: Path
    ) -> None:
        trainer = _make_trainer(
            tmp_path,
            use_liger_kernel=True,
            neftune_noise_alpha=5.0,
            torch_compile=True,
        )
        _stub_stage(trainer)
        trainer.setup_trainer()

        args = mock_trainer_cls.call_args.kwargs["args"]
        assert args.use_liger_kernel is True
        assert args.neftune_noise_alpha == pytest.approx(5.0)
        assert args.torch_compile is True

    @patch("src.training.sft_trainer.AttentionMaskCausalCollator")
    @patch("src.training.sft_trainer.Trainer")
    def test_torch_compile_with_quantization_warns(
        self, mock_trainer_cls: MagicMock, _mock_collator: MagicMock, tmp_path: Path
    ) -> None:
        trainer = _make_trainer(tmp_path, torch_compile=True)
        # Bypass the MPS auto-disable in ModelConfig.__post_init__ so the
        # guard sees a genuinely quantized model.
        trainer.model_config.quantization_bits = 4
        _stub_stage(trainer)

        with pytest.warns(UserWarning, match="torch_compile"):
            trainer.setup_trainer()

    @patch("src.training.sft_trainer.AttentionMaskCausalCollator")
    @patch("src.training.sft_trainer.Trainer")
    def test_no_compile_warning_on_full_precision(
        self, mock_trainer_cls: MagicMock, _mock_collator: MagicMock, tmp_path: Path
    ) -> None:
        import warnings

        trainer = _make_trainer(tmp_path, torch_compile=True)
        assert trainer.model_config.quantization_bits is None  # MPS auto-disabled
        _stub_stage(trainer)

        with warnings.catch_warnings():
            warnings.simplefilter("error")  # any UserWarning fails the test
            trainer.setup_trainer()


class TestPrepareModel:
    @patch("src.training.sft_trainer.get_peft_model")
    @patch("src.training.sft_trainer.prepare_model_for_kbit_training", return_value=None)
    @patch("src.training.sft_trainer.load_model_and_tokenizer")
    def test_prepare_model_applies_lora_with_flags(
        self,
        mock_load: MagicMock,
        mock_kbit: MagicMock,
        mock_peft: MagicMock,
        tmp_path: Path,
    ) -> None:
        mock_model = MagicMock()
        param = MagicMock()
        param.numel.return_value = 1000
        param.requires_grad = True
        mock_model.parameters.return_value = [param]
        mock_load.return_value = (mock_model, MagicMock())
        mock_peft.return_value = mock_model

        trainer = _make_trainer(tmp_path)
        trainer.lora_config.use_rslora = True
        trainer.lora_config.use_dora = True
        trainer.lora_config.init_lora_weights = "pissa"
        trainer.prepare_model()

        peft_lora_cfg = mock_peft.call_args[0][1]
        assert peft_lora_cfg.use_rslora is True
        assert peft_lora_cfg.use_dora is True
        assert peft_lora_cfg.init_lora_weights == "pissa"
        assert trainer.model is mock_model

    @patch("src.training.sft_trainer.get_peft_model")
    @patch("src.training.sft_trainer.prepare_model_for_kbit_training", return_value=None)
    @patch("src.training.sft_trainer.load_model_and_tokenizer")
    def test_prepare_model_injects_loftq_config(
        self,
        mock_load: MagicMock,
        mock_kbit: MagicMock,
        mock_peft: MagicMock,
        tmp_path: Path,
    ) -> None:
        mock_model = MagicMock()
        param = MagicMock()
        param.numel.return_value = 1000
        param.requires_grad = True
        mock_model.parameters.return_value = [param]
        mock_load.return_value = (mock_model, MagicMock())
        mock_peft.return_value = mock_model

        trainer = _make_trainer(tmp_path)
        trainer.lora_config.init_lora_weights = "loftq"
        trainer.lora_config.loftq_bits = 4
        trainer.lora_config.loftq_iter = 2
        trainer.prepare_model()

        # peft hard-errors on loftq without the dict — trainers must inject it.
        assert mock_peft.call_args[0][1].loftq_config == {
            "loftq_bits": 4,
            "loftq_iter": 2,
        }
        # Non-CUDA platform: k-bit preparation must be skipped.
        mock_kbit.assert_not_called()

    @patch("src.training.sft_trainer.get_peft_model")
    @patch("src.training.sft_trainer.prepare_model_for_kbit_training", return_value=None)
    @patch("src.training.sft_trainer.load_model_and_tokenizer")
    def test_prepare_model_forwards_lora_patterns(
        self,
        mock_load: MagicMock,
        mock_kbit: MagicMock,
        mock_peft: MagicMock,
        tmp_path: Path,
    ) -> None:
        mock_model = MagicMock()
        param = MagicMock()
        param.numel.return_value = 1000
        param.requires_grad = True
        mock_model.parameters.return_value = [param]
        mock_load.return_value = (mock_model, MagicMock())
        mock_peft.return_value = mock_model

        trainer = _make_trainer(tmp_path)
        trainer.lora_config.rank_pattern = {"^model.layers.0.q_proj": 32}
        trainer.lora_config.alpha_pattern = {"^model.layers.0.q_proj": 64}
        trainer.lora_config.exclude_modules = ["lm_head"]
        trainer.prepare_model()

        peft_lora_cfg = mock_peft.call_args[0][1]
        assert peft_lora_cfg.rank_pattern == {"^model.layers.0.q_proj": 32}
        assert peft_lora_cfg.alpha_pattern == {"^model.layers.0.q_proj": 64}
        # peft normalizes the exclude list to a set.
        assert peft_lora_cfg.exclude_modules == {"lm_head"}

    @patch("src.training.sft_trainer.get_peft_model")
    @patch("src.training.sft_trainer.prepare_model_for_kbit_training", return_value=None)
    @patch("src.training.sft_trainer.load_model_and_tokenizer")
    def test_prepare_model_patterns_default_absent(
        self,
        mock_load: MagicMock,
        mock_kbit: MagicMock,
        mock_peft: MagicMock,
        tmp_path: Path,
    ) -> None:
        mock_model = MagicMock()
        param = MagicMock()
        param.numel.return_value = 1000
        param.requires_grad = True
        mock_model.parameters.return_value = [param]
        mock_load.return_value = (mock_model, MagicMock())
        mock_peft.return_value = mock_model

        trainer = _make_trainer(tmp_path)
        trainer.prepare_model()

        peft_lora_cfg = mock_peft.call_args[0][1]
        # None → forwarded as peft defaults (no forced pattern behavior).
        assert peft_lora_cfg.rank_pattern == {}
        assert peft_lora_cfg.alpha_pattern == {}
        assert peft_lora_cfg.exclude_modules is None


class TestPrepareData:
    def _stub_dataset(self) -> MagicMock:
        ds = MagicMock()
        ds.split_dataset.return_value = (["t"], ["v"])
        ds.format_for_training.side_effect = lambda *_a, **_k: ["formatted"]
        return ds

    @patch("src.training.sft_trainer.FinanceDataset")
    @patch("src.training.sft_trainer.AlpacaDataset")
    def test_generic_name_selects_alpaca(
        self, mock_alpaca: MagicMock, mock_finance: MagicMock, tmp_path: Path
    ) -> None:
        ds = self._stub_dataset()
        mock_alpaca.return_value = ds
        trainer = _make_trainer(tmp_path)
        trainer.data_config.dataset_name = "some/other-dataset"
        trainer.tokenizer = MagicMock()

        trainer.prepare_data()

        mock_alpaca.assert_called_once()
        mock_finance.assert_not_called()
        assert trainer.train_dataset == ["formatted"]
        assert trainer.eval_dataset == ["formatted"]

    @patch("src.training.sft_trainer.FinanceDataset")
    @patch("src.training.sft_trainer.AlpacaDataset")
    def test_finance_name_selects_finance_dataset(
        self, mock_alpaca: MagicMock, mock_finance: MagicMock, tmp_path: Path
    ) -> None:
        ds = self._stub_dataset()
        mock_finance.return_value = ds
        trainer = _make_trainer(tmp_path)
        trainer.data_config.dataset_name = "finance-alpaca"
        trainer.tokenizer = MagicMock()

        trainer.prepare_data()

        mock_finance.assert_called_once()
        mock_alpaca.assert_not_called()

    @patch("src.training.sft_trainer.AlpacaDataset")
    def test_validation_file_loaded_when_split_is_none(
        self, mock_alpaca: MagicMock, tmp_path: Path
    ) -> None:
        class _FakeDS:
            instances: list[_FakeDS] = []

            def __init__(self, data_path: str | None = None, max_samples: int | None = None):
                self.load_calls = 0
                self.dataset = ["raw"]
                _FakeDS.instances.append(self)

            def load(self) -> None:
                self.load_calls += 1

            def split_dataset(self, validation_split: float, seed: int) -> tuple[Any, None]:
                return (["t"], None)  # no eval split from train data

            def format_for_training(self, tok: Any, max_length: int) -> list[str]:
                return ["formatted"]

        _FakeDS.instances = []
        primary = _FakeDS()  # the patched constructor hands the trainer this one
        mock_alpaca.return_value = primary
        val_file = tmp_path / "val.json"
        val_file.write_text("[]")

        trainer = _make_trainer(tmp_path)
        trainer.data_config.dataset_name = "some/other-dataset"
        trainer.data_config.validation_file = str(val_file)
        trainer.tokenizer = MagicMock()

        trainer.prepare_data()

        # A second dataset instance was built from the validation file,
        # loaded, and its formatted output assigned as eval_dataset.
        assert len(_FakeDS.instances) == 2
        assert _FakeDS.instances[1].load_calls == 1
        assert trainer.eval_dataset == ["formatted"]


class TestOptimizerOverride:
    @patch("src.training.sft_trainer.AttentionMaskCausalCollator")
    @patch("src.training.sft_trainer.Trainer")
    def test_optim_injected_when_set(
        self, mock_trainer_cls: MagicMock, _mock_collator: MagicMock, tmp_path: Path
    ) -> None:
        trainer = _make_trainer(tmp_path, optim="paged_adamw_8bit")
        _stub_stage(trainer)
        trainer.setup_trainer()

        args = mock_trainer_cls.call_args.kwargs["args"]
        assert args.optim == "paged_adamw_8bit"

    @patch("src.training.sft_trainer.AttentionMaskCausalCollator")
    @patch("src.training.sft_trainer.Trainer")
    def test_optim_defaults_to_library_choice(
        self, mock_trainer_cls: MagicMock, _mock_collator: MagicMock, tmp_path: Path
    ) -> None:
        trainer = _make_trainer(tmp_path)
        _stub_stage(trainer)
        trainer.setup_trainer()

        args = mock_trainer_cls.call_args.kwargs["args"]
        # None → library default (adamw_torch_fused in transformers >= 5.x).
        assert args.optim == "adamw_torch_fused"


class TestTrain:
    def _armed(self, tmp_path: Path) -> tuple[SFTTrainer, MagicMock]:
        trainer = _make_trainer(tmp_path)
        _stub_stage(trainer)
        hf_trainer = MagicMock()
        hf_trainer.train.return_value = SimpleNamespace(metrics={"train_loss": 0.5})
        trainer.trainer = hf_trainer
        return trainer, hf_trainer

    @patch("src.training.sft_trainer.register_trained_model")
    @patch("src.training.sft_trainer.log_metrics")
    @patch("src.training.sft_trainer.setup_tensorboard")
    def test_train_happy_path_saves_and_registers(
        self,
        _mock_tb: MagicMock,
        _mock_logm: MagicMock,
        mock_reg: MagicMock,
        tmp_path: Path,
    ) -> None:
        trainer, hf_trainer = self._armed(tmp_path)

        result = trainer.train()

        hf_trainer.train.assert_called_once_with(resume_from_checkpoint=None)
        hf_trainer.save_model.assert_called_once()
        trainer.tokenizer.save_pretrained.assert_called_once_with(
            trainer.training_config.output_dir
        )
        mock_reg.assert_called_once()
        assert result is hf_trainer.train.return_value

    @patch("src.training.sft_trainer.register_trained_model")
    @patch("src.training.sft_trainer.log_metrics")
    @patch("src.training.sft_trainer.setup_tensorboard")
    def test_train_resume_checkpoint_forwarded(
        self, _mock_tb: MagicMock, _mock_lm: MagicMock, _mock_reg: MagicMock, tmp_path: Path
    ) -> None:
        trainer, hf_trainer = self._armed(tmp_path)

        trainer.train(resume_from_checkpoint="outputs/ckpt-100")

        hf_trainer.train.assert_called_once_with(resume_from_checkpoint="outputs/ckpt-100")

    @patch("src.training.sft_trainer.setup_tensorboard")
    def test_train_failure_ends_run_and_reraises(self, _mock_tb: MagicMock, tmp_path: Path) -> None:
        trainer, hf_trainer = self._armed(tmp_path)
        hf_trainer.train.side_effect = RuntimeError("OOM")

        with (
            patch.object(trainer._tracker, "end_run") as mock_end,
            pytest.raises(RuntimeError, match="OOM"),
        ):
            trainer.train()
        mock_end.assert_called_once()  # finally-block cleanup ran

    @patch("src.training.sft_trainer.register_trained_model")
    @patch("src.training.sft_trainer.log_metrics")
    @patch("src.training.sft_trainer.setup_tensorboard")
    @patch("src.training.sft_trainer.setup_wandb")
    def test_train_with_wandb_finishes_run(
        self,
        mock_wandb: MagicMock,
        _mock_tb: MagicMock,
        _mock_lm: MagicMock,
        _mock_reg: MagicMock,
        tmp_path: Path,
    ) -> None:
        trainer, _ = self._armed(tmp_path)
        trainer.logging_config.use_wandb = True
        wandb_run = MagicMock()
        mock_wandb.return_value = wandb_run

        trainer.train()

        mock_wandb.assert_called_once()
        wandb_run.finish.assert_called_once()

    def test_evaluate_returns_metrics(self, tmp_path: Path) -> None:
        trainer = _make_trainer(tmp_path)
        hf_trainer = MagicMock()
        hf_trainer.evaluate.return_value = {"eval_loss": 0.42}
        trainer.trainer = hf_trainer

        assert trainer.evaluate() == {"eval_loss": 0.42}


class TestRunSFTTraining:
    @patch("src.training.sft_trainer.SFTTrainer")
    def test_pipeline_calls_all_stages(self, mock_cls: MagicMock) -> None:
        instance = mock_cls.return_value
        run_sft_training(
            model_config=ModelConfig(name="qwen-test"),
            training_config=TrainingConfig(output_dir="out"),
            lora_config=LoRAConfig(),
            data_config=DataConfig(),
            logging_config=LoggingConfig(),
            resume_from_checkpoint="ckpt-5",
        )
        mock_cls.assert_called_once()
        instance.prepare_model.assert_called_once()
        instance.prepare_data.assert_called_once()
        instance.setup_trainer.assert_called_once()
        instance.train.assert_called_once_with(resume_from_checkpoint="ckpt-5")


class TestGradientCheckpointingKwargs:
    @patch("src.training.sft_trainer.AttentionMaskCausalCollator")
    @patch("src.training.sft_trainer.Trainer")
    def test_non_reentrant_forwarded_when_checkpointing_on(
        self, mock_trainer_cls: MagicMock, _mock_collator: MagicMock, tmp_path: Path
    ) -> None:
        trainer = _make_trainer(tmp_path)  # gradient_checkpointing defaults True
        _stub_stage(trainer)
        trainer.setup_trainer()

        args = mock_trainer_cls.call_args.kwargs["args"]
        assert args.gradient_checkpointing_kwargs == {"use_reentrant": False}

    @patch("src.training.sft_trainer.AttentionMaskCausalCollator")
    @patch("src.training.sft_trainer.Trainer")
    def test_kwargs_none_when_checkpointing_off(
        self, mock_trainer_cls: MagicMock, _mock_collator: MagicMock, tmp_path: Path
    ) -> None:
        trainer = _make_trainer(tmp_path, gradient_checkpointing=False)
        _stub_stage(trainer)
        trainer.setup_trainer()

        args = mock_trainer_cls.call_args.kwargs["args"]
        assert args.gradient_checkpointing_kwargs is None


class TestTorchEmptyCacheSteps:
    @patch("src.training.sft_trainer.AttentionMaskCausalCollator")
    @patch("src.training.sft_trainer.Trainer")
    def test_value_forwarded(
        self, mock_trainer_cls: MagicMock, _mock_collator: MagicMock, tmp_path: Path
    ) -> None:
        trainer = _make_trainer(tmp_path, torch_empty_cache_steps=100)
        _stub_stage(trainer)
        trainer.setup_trainer()

        args = mock_trainer_cls.call_args.kwargs["args"]
        assert args.torch_empty_cache_steps == 100

    @patch("src.training.sft_trainer.AttentionMaskCausalCollator")
    @patch("src.training.sft_trainer.Trainer")
    def test_auto_find_batch_size_forwarded(
        self, mock_trainer_cls: MagicMock, _mock_collator: MagicMock, tmp_path: Path
    ) -> None:
        trainer = _make_trainer(tmp_path, auto_find_batch_size=True)
        _stub_stage(trainer)
        trainer.setup_trainer()

        args = mock_trainer_cls.call_args.kwargs["args"]
        assert args.auto_find_batch_size is True

    @patch("src.training.sft_trainer.AttentionMaskCausalCollator")
    @patch("src.training.sft_trainer.Trainer")
    def test_train_sampling_strategy_forwarded(
        self, mock_trainer_cls: MagicMock, _mock_collator: MagicMock, tmp_path: Path
    ) -> None:
        trainer = _make_trainer(tmp_path, train_sampling_strategy="group_by_length")
        _stub_stage(trainer)
        trainer.setup_trainer()

        args = mock_trainer_cls.call_args.kwargs["args"]
        assert args.train_sampling_strategy == "group_by_length"


class TestMemoryCallback:
    @patch("src.training.sft_trainer.log_gpu_memory")
    def test_logs_at_interval_steps(self, mock_log: MagicMock) -> None:
        callback = MemoryCallback(log_steps=10)
        control = object()
        state = SimpleNamespace(global_step=20)

        returned = callback.on_step_end(args=MagicMock(), state=state, control=control)

        mock_log.assert_called_once_with(20, wandb_run=None)
        assert returned is control

    @patch("src.training.sft_trainer.log_gpu_memory")
    def test_skips_between_intervals(self, mock_log: MagicMock) -> None:
        callback = MemoryCallback(log_steps=10)
        callback.on_step_end(args=MagicMock(), state=SimpleNamespace(global_step=15), control=None)
        mock_log.assert_not_called()


class TestReportTo:
    def test_wandb_backend_appended_when_enabled(self, tmp_path: Path) -> None:
        trainer = _make_trainer(tmp_path)
        trainer.logging_config.use_wandb = True
        trainer.logging_config.use_tensorboard = False
        assert trainer._get_report_to() == ["wandb"]

    def test_both_backends_ordered(self, tmp_path: Path) -> None:
        trainer = _make_trainer(tmp_path)
        trainer.logging_config.use_wandb = True
        trainer.logging_config.use_tensorboard = True
        assert trainer._get_report_to() == ["wandb", "tensorboard"]


class TestPrepareModelKbit:
    @patch("src.training.sft_trainer.get_platform")
    @patch("src.training.sft_trainer.get_peft_model")
    @patch("src.training.sft_trainer.prepare_model_for_kbit_training")
    @patch("src.training.sft_trainer.load_model_and_tokenizer")
    def test_kbit_preparation_on_cuda_with_quantization(
        self,
        mock_load: MagicMock,
        mock_kbit: MagicMock,
        mock_peft: MagicMock,
        mock_platform: MagicMock,
        tmp_path: Path,
    ) -> None:
        mock_platform.return_value = SimpleNamespace(is_cuda=True)
        mock_model = MagicMock()
        param = MagicMock()
        param.numel.return_value = 10
        param.requires_grad = True
        mock_model.parameters.return_value = [param]
        mock_load.return_value = (mock_model, MagicMock())
        mock_kbit.return_value = mock_model
        mock_peft.return_value = mock_model

        trainer = _make_trainer(tmp_path)
        trainer.model_config.quantization_bits = 4  # bypass MPS auto-disable
        trainer.prepare_model()

        mock_kbit.assert_called_once_with(mock_model)

    @patch("src.training.sft_trainer.get_platform")
    @patch("src.training.sft_trainer.get_peft_model")
    @patch("src.training.sft_trainer.prepare_model_for_kbit_training")
    @patch("src.training.sft_trainer.load_model_and_tokenizer")
    def test_kbit_skipped_on_cuda_full_precision(
        self,
        mock_load: MagicMock,
        mock_kbit: MagicMock,
        mock_peft: MagicMock,
        mock_platform: MagicMock,
        tmp_path: Path,
    ) -> None:
        mock_platform.return_value = SimpleNamespace(is_cuda=True)
        mock_model = MagicMock()
        param = MagicMock()
        param.numel.return_value = 10
        mock_model.parameters.return_value = [param]
        mock_load.return_value = (mock_model, MagicMock())
        mock_peft.return_value = mock_model

        trainer = _make_trainer(tmp_path)
        trainer.model_config.quantization_bits = None
        trainer.prepare_model()

        mock_kbit.assert_not_called()


class TestPrepareDataBranches:
    @patch("src.data.medical_dataset.MedicalEntityDataset")
    @patch("src.training.sft_trainer.AlpacaDataset")
    def test_medical_name_selects_medical_dataset(
        self, mock_alpaca: MagicMock, mock_medical: MagicMock, tmp_path: Path
    ) -> None:
        ds = MagicMock()
        ds.split_dataset.return_value = (["t"], ["v"])
        ds.format_for_training.side_effect = lambda *_a, **_k: ["formatted"]
        mock_medical.return_value = ds
        trainer = _make_trainer(tmp_path)
        trainer.data_config.dataset_name = "medical_entity_v2"
        trainer.tokenizer = MagicMock()

        trainer.prepare_data()

        mock_medical.assert_called_once()
        mock_alpaca.assert_not_called()

    @patch("src.training.sft_trainer.AlpacaDataset")
    def test_missing_validation_file_warns(
        self, mock_alpaca: MagicMock, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        ds = MagicMock()
        ds.split_dataset.return_value = (["t"], None)  # no eval split
        ds.format_for_training.side_effect = lambda *_a, **_k: ["formatted"]
        mock_alpaca.return_value = ds
        trainer = _make_trainer(tmp_path)
        trainer.data_config.dataset_name = "some/other-dataset"
        trainer.data_config.validation_file = str(tmp_path / "nope.json")
        trainer.tokenizer = MagicMock()

        trainer.prepare_data()

        assert trainer.eval_dataset is None
        assert "No validation data found" in capsys.readouterr().out


class TestDistributedInjection:
    @patch("src.training.sft_trainer.get_distributed_info")
    @patch("src.training.sft_trainer.AttentionMaskCausalCollator")
    @patch("src.training.sft_trainer.Trainer")
    def test_fsdp_injected(
        self,
        mock_trainer_cls: MagicMock,
        _mock_collator: MagicMock,
        _mock_dist: MagicMock,
        tmp_path: Path,
    ) -> None:
        trainer = _make_trainer(tmp_path, fsdp="full_shard", fsdp_config={"fsdp_strategy": 1})
        _stub_stage(trainer)
        trainer.setup_trainer()

        args = mock_trainer_cls.call_args.kwargs["args"]
        # TrainingArguments normalizes the string into FSDPOption enums and
        # canonicalizes fsdp_config keys ("fsdp_strategy" -> "strategy").
        assert [str(getattr(opt, "value", opt)) for opt in args.fsdp] == ["full_shard"]
        assert args.fsdp_config["strategy"] == 1
        assert args.fsdp_config["min_num_params"] == 0

    @patch("src.training.sft_trainer.get_distributed_info")
    @patch("src.training.sft_trainer.TrainingArguments")
    @patch("src.training.sft_trainer.AttentionMaskCausalCollator")
    @patch("src.training.sft_trainer.Trainer")
    def test_deepspeed_injected(
        self,
        mock_trainer_cls: MagicMock,
        _mock_collator: MagicMock,
        mock_ta: MagicMock,
        _mock_dist: MagicMock,
        tmp_path: Path,
    ) -> None:
        # TrainingArguments itself would require the deepspeed package to
        # accept the kwarg; patch it and pin the forwarding of the raw dict.
        ds_config = str(tmp_path / "zero2.json")
        Path(ds_config).write_text("{}")
        trainer = _make_trainer(tmp_path, deepspeed_config=ds_config)
        _stub_stage(trainer)
        trainer.setup_trainer()

        kwargs = mock_ta.call_args.kwargs
        assert kwargs["deepspeed"] == ds_config

    @patch("src.training.sft_trainer.get_distributed_info")
    @patch("src.training.sft_trainer.AttentionMaskCausalCollator")
    @patch("src.training.sft_trainer.Trainer")
    def test_distributed_banner_printed(
        self,
        mock_trainer_cls: MagicMock,
        _mock_collator: MagicMock,
        mock_dist: MagicMock,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        mock_dist.return_value = SimpleNamespace(is_distributed=True, world_size=4)
        trainer = _make_trainer(tmp_path, fsdp="full_shard")
        _stub_stage(trainer)
        trainer.setup_trainer()

        out = capsys.readouterr().out
        assert "world_size=4" in out
        assert "strategy=full_shard" in out


class TestTrainMlflowStart:
    @patch("src.training.sft_trainer.register_trained_model")
    @patch("src.training.sft_trainer.log_metrics")
    @patch("src.training.sft_trainer.setup_tensorboard")
    def test_active_tracker_starts_mlflow_run(
        self,
        _mock_tb: MagicMock,
        _mock_lm: MagicMock,
        _mock_reg: MagicMock,
        tmp_path: Path,
    ) -> None:
        trainer, hf_trainer = TestTrain()._armed(tmp_path)
        tracker = MagicMock()
        tracker.active = True
        tracker.start_run.return_value = "run-123"
        tracker.end_run.return_value = None
        tracker.log_metrics.return_value = None
        tracker.log_params.return_value = None
        trainer._tracker = tracker

        trainer.train()

        tracker.start_run.assert_called_once()
        call = tracker.start_run.call_args
        assert call.kwargs["run_name"] == trainer.logging_config.mlflow_run_name
        assert "model" in call.kwargs["config"]
        hf_trainer.train.assert_called_once()


class TestPrepareModelLoRAGA:
    """LoRA-GA init (arXiv:2407.05000): config injection + calibration wiring.

    peft silently falls back to gaussian init when preprocess_loraga has not
    run, so these tests pin that the trainer always calls it with a working
    train_step closure before attaching adapters.
    """

    def _ga_model(self) -> MagicMock:
        # Real torch Parameter so .device resolves (tensors move to it).
        # parameters() must yield a FRESH iterator per call (like a real
        # generator) — prepare_model consumes it several times.
        model = MagicMock()
        real_param = torch.nn.Parameter(torch.zeros(1))
        model.parameters.side_effect = lambda: iter([real_param])
        return model

    @patch("src.training.sft_trainer._dataset_class_for")
    @patch("src.training.sft_trainer.preprocess_loraga")
    @patch("src.training.sft_trainer.get_peft_model")
    @patch("src.training.sft_trainer.prepare_model_for_kbit_training", return_value=None)
    @patch("src.training.sft_trainer.load_model_and_tokenizer")
    def test_prepare_model_injects_lora_ga_config_and_runs_calibration(
        self,
        mock_load: MagicMock,
        mock_kbit: MagicMock,
        mock_peft: MagicMock,
        mock_preprocess: MagicMock,
        mock_ds_cls: MagicMock,
        tmp_path: Path,
    ) -> None:
        mock_model = self._ga_model()
        mock_load.return_value = (mock_model, MagicMock())
        mock_peft.return_value = mock_model

        rows = [
            {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]},
            {"input_ids": [4, 5, 6], "attention_mask": [1, 1, 1]},
        ]

        class _FakeCalibDataset:
            created: list[_FakeCalibDataset] = []

            def __init__(self, **kwargs: Any) -> None:
                self.init_kwargs = kwargs
                _FakeCalibDataset.created.append(self)

            def load(self) -> None:
                pass

            def format_for_training(self, *args: Any, **kwargs: Any) -> list[dict[str, Any]]:
                self.fmt_kwargs = kwargs
                return rows

        mock_ds_cls.return_value = _FakeCalibDataset

        trainer = _make_trainer(tmp_path)
        trainer.lora_config.init_lora_weights = "lora_ga"
        trainer.prepare_model()

        # Calibration runs BEFORE the adapter is attached, with the injected
        # LoraGAConfig and the configured cache path (None by default).
        mock_preprocess.assert_called_once()
        preprocess_args = mock_preprocess.call_args
        lora_cfg = preprocess_args[0][1]
        assert preprocess_args.kwargs["cache_file"] is None
        assert lora_cfg.lora_ga_config.direction == "ArB2r"
        assert lora_cfg.lora_ga_config.scale == "stable"
        assert lora_cfg.lora_ga_config.stable_gamma == 16
        # get_peft_model receives the same config object peft preprocessed.
        assert mock_peft.call_args[0][1] is lora_cfg

        # The calibration slice is sized lora_ga_calibration_batches * batch_size
        # and formatted with the training max_length.
        fake = _FakeCalibDataset.created[0]
        assert fake.init_kwargs["max_samples"] == 4 * 1
        assert fake.fmt_kwargs["max_length"] == trainer.model_config.max_length

        # The captured train_step runs forward+backward over the batches.
        train_step = preprocess_args[0][2]
        assert callable(train_step)
        train_step()
        assert mock_model.called
        forward_kwargs = mock_model.call_args.kwargs
        assert "labels" in forward_kwargs
        assert mock_model.return_value.loss.backward.called

    @patch("src.training.sft_trainer._dataset_class_for")
    @patch("src.training.sft_trainer.preprocess_loraga")
    @patch("src.training.sft_trainer.get_peft_model")
    @patch("src.training.sft_trainer.prepare_model_for_kbit_training", return_value=None)
    @patch("src.training.sft_trainer.load_model_and_tokenizer")
    def test_prepare_model_forwards_lora_ga_overrides_and_cache(
        self,
        mock_load: MagicMock,
        mock_kbit: MagicMock,
        mock_peft: MagicMock,
        mock_preprocess: MagicMock,
        mock_ds_cls: MagicMock,
        tmp_path: Path,
    ) -> None:
        mock_model = self._ga_model()
        mock_load.return_value = (mock_model, MagicMock())
        mock_peft.return_value = mock_model

        class _FakeEmptyDataset:
            def __init__(self, **kwargs: Any) -> None:
                pass

            def load(self) -> None:
                pass

            def format_for_training(self, *args: Any, **kwargs: Any) -> list[dict[str, Any]]:
                return []

        mock_ds_cls.return_value = _FakeEmptyDataset

        trainer = _make_trainer(tmp_path)
        trainer.lora_config.init_lora_weights = "lora_ga"
        trainer.lora_config.lora_ga_direction = "ArBr"
        trainer.lora_config.lora_ga_scale = "gd_scale"
        trainer.lora_config.lora_ga_stable_gamma = 32
        trainer.lora_config.lora_ga_cache_file = str(tmp_path / "ga_grads.pt")
        trainer.prepare_model()

        lora_cfg = mock_preprocess.call_args[0][1]
        assert lora_cfg.lora_ga_config.direction == "ArBr"
        assert lora_cfg.lora_ga_config.scale == "gd_scale"
        assert lora_cfg.lora_ga_config.stable_gamma == 32
        assert mock_preprocess.call_args.kwargs["cache_file"] == str(tmp_path / "ga_grads.pt")

    @patch("src.training.sft_trainer.get_peft_model")
    @patch("src.training.sft_trainer.prepare_model_for_kbit_training", return_value=None)
    @patch("src.training.sft_trainer.load_model_and_tokenizer")
    def test_prepare_model_rejects_lora_ga_on_quantized_base(
        self,
        mock_load: MagicMock,
        mock_kbit: MagicMock,
        mock_peft: MagicMock,
        tmp_path: Path,
    ) -> None:
        # Post-construction mutation bypasses SFTConfig's guard — the trainer
        # re-checks so the failure is a clear message, not a peft stack trace.
        mock_model = self._ga_model()
        mock_load.return_value = (mock_model, MagicMock())

        trainer = _make_trainer(tmp_path)
        trainer.model_config.quantization_bits = 4
        trainer.lora_config.init_lora_weights = "lora_ga"

        with pytest.raises(ValueError, match="full-precision"):
            trainer.prepare_model()
        mock_peft.assert_not_called()


class TestMFUCallbackAttachment:
    @patch("src.training.sft_trainer.AttentionMaskCausalCollator")
    @patch("src.training.sft_trainer.Trainer")
    def test_setup_trainer_attaches_mfu_callback(
        self, mock_trainer_cls: MagicMock, _mock_collator: MagicMock, tmp_path: Path
    ) -> None:
        from src.training.callbacks import MFUCallback

        trainer = _make_trainer(tmp_path)
        _stub_stage(trainer)
        trainer.setup_trainer()

        callbacks = mock_trainer_cls.call_args.kwargs["callbacks"]
        assert any(isinstance(c, MFUCallback) for c in callbacks)
        # The attached instance is gracefully disabled: a MagicMock model
        # config cannot satisfy compute_flops_per_token.
        mfu_cb = next(c for c in callbacks if isinstance(c, MFUCallback))
        assert mfu_cb._fpt is None
