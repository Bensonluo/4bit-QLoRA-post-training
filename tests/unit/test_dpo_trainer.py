"""Unit tests for the DPO trainer (TRL 1.4 API surface)."""

import sys
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from datasets import Dataset
from trl import DPOConfig as TRLDPOConfig

from config.base import LoggingConfig, LoRAConfig, ModelConfig, TrainingConfig
from config.dpo import DPOConfig, PreferenceDataConfig, ReferenceModelConfig
from src.training.dpo_trainer import DPOTrainer


def _make_trainer(tmp_path: Any) -> DPOTrainer:
    """Build a DPOTrainer with light configs (no model loading)."""
    return DPOTrainer(
        model_config=ModelConfig(name="Qwen/Qwen2.5-0.5B-Instruct"),
        training_config=TrainingConfig(output_dir=str(tmp_path / "dpo")),
        lora_config=LoRAConfig(),
        dpo_config=DPOConfig(),
        data_config=PreferenceDataConfig(max_samples=10),
        reference_config=ReferenceModelConfig(name="Qwen/Qwen2.5-0.5B-Instruct"),
        logging_config=LoggingConfig(),
    )


def _fake_dataset(n: int = 2) -> Dataset:
    return Dataset.from_dict(
        {
            "prompt": [f"prompt {j}" for j in range(n)],
            "chosen": [f"chosen {j}" for j in range(n)],
            "rejected": [f"rejected {j}" for j in range(n)],
        }
    )


class TestDPOConfigValidation:
    def test_defaults(self) -> None:
        cfg = DPOConfig()
        assert cfg.beta == 0.1
        assert cfg.loss_type == "sigmoid"
        assert cfg.max_length == 512

    def test_invalid_loss_type(self) -> None:
        with pytest.raises(ValueError, match="Invalid loss_type"):
            DPOConfig(loss_type="pairwise")  # removed in TRL >= 1.0

    def test_nonpositive_beta(self) -> None:
        with pytest.raises(ValueError, match="beta must be positive"):
            DPOConfig(beta=0.0)

    def test_label_smoothing_range(self) -> None:
        with pytest.raises(ValueError, match="label_smoothing"):
            DPOConfig(label_smoothing=1.5)


class TestDPOTrainerInit:
    def test_init_stores_configs(self, tmp_path: Any) -> None:
        trainer = _make_trainer(tmp_path)
        assert trainer.dpo_config.beta == 0.1
        assert trainer.model is None


class TestFilterFinance:
    def test_filter_finance_predicate(self, tmp_path: Any) -> None:
        trainer = _make_trainer(tmp_path)
        dataset = MagicMock()
        trainer._filter_finance(dataset)
        # The predicate is what matters — capture and exercise it directly.
        predicate = dataset.filter.call_args[0][0]

        finance = {"prompt": "stock", "chosen": "ok", "rejected": "no"}
        other = {"prompt": "recipe", "chosen": "ok", "rejected": "no"}
        assert predicate(finance) is True
        assert predicate(other) is False


class TestPrepareModels:
    @patch("src.training.dpo_trainer.get_peft_model")
    @patch("src.training.dpo_trainer.prepare_model_for_kbit_training", return_value=None)
    @patch("src.training.dpo_trainer.load_model_and_tokenizer")
    def test_prepare_models_freezes_reference(
        self,
        mock_load: MagicMock,
        mock_kbit: MagicMock,
        mock_peft: MagicMock,
        tmp_path: Any,
    ) -> None:
        policy_param = MagicMock()
        policy_param.numel.return_value = 100
        policy = MagicMock()
        policy.parameters.return_value = [policy_param]
        tokenizer = MagicMock()

        ref_param = MagicMock()
        ref_param.numel.return_value = 100
        ref = MagicMock()
        ref.parameters.return_value = [ref_param]

        # First call loads the policy, second the reference model.
        mock_load.side_effect = [(policy, tokenizer), (ref, MagicMock())]
        mock_peft.side_effect = lambda model, cfg: model

        trainer = _make_trainer(tmp_path)
        trainer.prepare_models()

        assert mock_load.call_count == 2
        mock_peft.assert_called_once()  # LoRA on policy only, never the reference
        assert ref_param.requires_grad is False
        assert trainer.tokenizer is tokenizer


class TestSetupTrainer:
    """Regression tests for the TRL >= 1.4 config surface.

    DPOTrainer used to construct TRLDPOConfig with kwargs that no longer
    exist (max_prompt_length, reference_model_free, ...) and splat
    ``trl_dpo_config.__dict__`` into the trainer constructor. These tests
    pin the corrected API usage.
    """

    @patch("src.training.dpo_trainer.TRLDPOTrainer")
    def test_setup_trainer_builds_valid_trl_config(
        self, mock_trl_trainer: MagicMock, tmp_path: Any
    ) -> None:
        trainer = _make_trainer(tmp_path)
        trainer.model = MagicMock()
        trainer.ref_model = MagicMock()
        trainer.tokenizer = MagicMock()
        trainer.train_dataset = _fake_dataset()
        trainer.eval_dataset = _fake_dataset()

        trainer.setup_trainer()

        # TRLDPOConfig(**training_kwargs) constructed for real (no TypeError).
        kwargs = mock_trl_trainer.call_args.kwargs
        assert isinstance(kwargs["args"], TRLDPOConfig)
        assert kwargs["args"].beta == 0.1
        assert kwargs["args"].loss_type == ["sigmoid"]
        assert kwargs["args"].max_length == 512
        assert kwargs["args"].eval_strategy is not None
        # TRL >= 1.0 API: processing_class, not tokenizer=.
        assert kwargs["processing_class"] is trainer.tokenizer
        assert kwargs["ref_model"] is trainer.ref_model
        # No beta= collision kwarg on the trainer itself.
        assert "beta" not in kwargs
        assert "tokenizer" not in kwargs

    @patch("src.training.dpo_trainer.TRLDPOTrainer")
    def test_setup_trainer_forwards_dpo_fields(
        self, mock_trl_trainer: MagicMock, tmp_path: Any
    ) -> None:
        trainer = _make_trainer(tmp_path)
        trainer.dpo_config = DPOConfig(beta=0.2, loss_type="ipo", max_length=256)
        trainer.model = MagicMock()
        trainer.ref_model = MagicMock()
        trainer.tokenizer = MagicMock()
        trainer.train_dataset = _fake_dataset()
        trainer.eval_dataset = _fake_dataset()

        trainer.setup_trainer()

        args = mock_trl_trainer.call_args.kwargs["args"]
        assert args.beta == 0.2
        assert args.loss_type == ["ipo"]
        assert args.max_length == 256

    @patch("src.training.dpo_trainer.TRLDPOTrainer")
    def test_setup_trainer_forwards_torch_compile(
        self, mock_trl_trainer: MagicMock, tmp_path: Any
    ) -> None:
        trainer = _make_trainer(tmp_path)
        trainer.training_config.torch_compile = True
        trainer.model = MagicMock()
        trainer.ref_model = MagicMock()
        trainer.tokenizer = MagicMock()
        trainer.train_dataset = _fake_dataset()
        trainer.eval_dataset = _fake_dataset()

        trainer.setup_trainer()

        args = mock_trl_trainer.call_args.kwargs["args"]
        assert args.torch_compile is True
        # NEFTune is SFT-only — DPO must not forward it even if set.
        trainer.training_config.neftune_noise_alpha = 5.0
        trainer.setup_trainer()
        args = mock_trl_trainer.call_args.kwargs["args"]
        assert args.neftune_noise_alpha is None

    @patch("src.training.dpo_trainer.TRLDPOTrainer")
    def test_setup_trainer_injects_optim(self, mock_trl_trainer: MagicMock, tmp_path: Any) -> None:
        trainer = _make_trainer(tmp_path)
        trainer.training_config.optim = "paged_adamw_8bit"
        trainer.model = MagicMock()
        trainer.ref_model = MagicMock()
        trainer.tokenizer = MagicMock()
        trainer.train_dataset = _fake_dataset()
        trainer.eval_dataset = _fake_dataset()

        trainer.setup_trainer()

        args = mock_trl_trainer.call_args.kwargs["args"]
        assert args.optim == "paged_adamw_8bit"

    @patch("src.training.dpo_trainer.get_peft_model")
    @patch("src.training.dpo_trainer.prepare_model_for_kbit_training", return_value=None)
    @patch("src.training.dpo_trainer.load_model_and_tokenizer")
    def test_prepare_models_passes_init_strategy(
        self, mock_load: MagicMock, mock_kbit: MagicMock, mock_peft: MagicMock, tmp_path: Any
    ) -> None:
        model = MagicMock()
        param = MagicMock()
        param.numel.return_value = 10
        param.requires_grad = False
        model.parameters.return_value = [param]
        mock_load.return_value = (model, MagicMock())
        mock_peft.return_value = model

        trainer = _make_trainer(tmp_path)
        trainer.lora_config.init_lora_weights = "pissa"
        trainer.prepare_models()

        assert mock_peft.call_args[0][1].init_lora_weights == "pissa"

    @patch("src.training.dpo_trainer.get_peft_model")
    @patch("src.training.dpo_trainer.prepare_model_for_kbit_training", return_value=None)
    @patch("src.training.dpo_trainer.load_model_and_tokenizer")
    def test_prepare_models_injects_loftq_config(
        self, mock_load: MagicMock, mock_kbit: MagicMock, mock_peft: MagicMock, tmp_path: Any
    ) -> None:
        model = MagicMock()
        param = MagicMock()
        param.numel.return_value = 10
        param.requires_grad = False
        model.parameters.return_value = [param]
        mock_load.return_value = (model, MagicMock())
        mock_peft.return_value = model

        trainer = _make_trainer(tmp_path)
        trainer.lora_config.init_lora_weights = "loftq"
        trainer.lora_config.loftq_bits = 4
        trainer.lora_config.loftq_iter = 2
        trainer.prepare_models()

        # peft hard-errors on loftq without the dict — trainers must inject it.
        assert mock_peft.call_args[0][1].loftq_config == {
            "loftq_bits": 4,
            "loftq_iter": 2,
        }

    @patch("src.training.dpo_trainer.get_peft_model")
    @patch("src.training.dpo_trainer.prepare_model_for_kbit_training", return_value=None)
    @patch("src.training.dpo_trainer.load_model_and_tokenizer")
    def test_prepare_models_forwards_lora_patterns(
        self, mock_load: MagicMock, mock_kbit: MagicMock, mock_peft: MagicMock, tmp_path: Any
    ) -> None:
        model = MagicMock()
        param = MagicMock()
        param.numel.return_value = 10
        param.requires_grad = False
        model.parameters.return_value = [param]
        mock_load.return_value = (model, MagicMock())
        mock_peft.return_value = model

        trainer = _make_trainer(tmp_path)
        trainer.lora_config.rank_pattern = {"^model.layers.0.q_proj": 32}
        trainer.lora_config.alpha_pattern = {"^model.layers.0.q_proj": 64}
        trainer.lora_config.exclude_modules = "lm_head"
        trainer.prepare_models()

        peft_lora_cfg = mock_peft.call_args[0][1]
        assert peft_lora_cfg.rank_pattern == {"^model.layers.0.q_proj": 32}
        assert peft_lora_cfg.alpha_pattern == {"^model.layers.0.q_proj": 64}
        # peft keeps a regex string exclude as the raw string.
        assert peft_lora_cfg.exclude_modules == "lm_head"

    @patch("src.training.dpo_trainer.get_peft_model")
    @patch("src.training.dpo_trainer.prepare_model_for_kbit_training", return_value=None)
    @patch("src.training.dpo_trainer.load_model_and_tokenizer")
    def test_prepare_models_omits_loftq_config_for_other_inits(
        self, mock_load: MagicMock, mock_kbit: MagicMock, mock_peft: MagicMock, tmp_path: Any
    ) -> None:
        model = MagicMock()
        param = MagicMock()
        param.numel.return_value = 10
        param.requires_grad = False
        model.parameters.return_value = [param]
        mock_load.return_value = (model, MagicMock())
        mock_peft.return_value = model

        trainer = _make_trainer(tmp_path)
        trainer.prepare_models()

        # peft normalizes absent loftq_config to an empty dict.
        assert mock_peft.call_args[0][1].loftq_config == {}


class TestPrepareData:
    @patch("src.training.dpo_trainer.PreferenceDataset")
    def test_prepare_data_loads_and_splits(self, mock_ds_cls: MagicMock, tmp_path: Any) -> None:
        ds = MagicMock()
        train_ds, eval_ds = MagicMock(), MagicMock()
        ds.split_dataset.return_value = (train_ds, eval_ds)
        mock_ds_cls.return_value = ds

        trainer = _make_trainer(tmp_path)
        trainer.data_config.auto_filter = False
        trainer.prepare_data()

        mock_ds_cls.assert_called_once_with(
            data_path=trainer.data_config.dataset_name,
            max_samples=trainer.data_config.max_samples,
        )
        ds.load.assert_called_once()
        assert trainer.train_dataset is train_ds
        assert trainer.eval_dataset is eval_ds

    @patch("src.training.dpo_trainer.PreferenceDataset")
    def test_prepare_data_applies_finance_filter_when_enabled(
        self, mock_ds_cls: MagicMock, tmp_path: Any
    ) -> None:
        ds = MagicMock()
        filtered = MagicMock()
        filtered.split_dataset.return_value = (MagicMock(), MagicMock())
        ds.filter.return_value = filtered
        mock_ds_cls.return_value = ds

        trainer = _make_trainer(tmp_path)
        trainer.data_config.auto_filter = True
        trainer.prepare_data()

        ds.filter.assert_called_once()
        # Split runs on the filtered dataset, not the raw one.
        filtered.split_dataset.assert_called_once()
        ds.split_dataset.assert_not_called()


class TestDistributedInjection:
    @patch("src.training.dpo_trainer.TRLDPOTrainer")
    def test_setup_trainer_injects_fsdp(self, mock_trl_trainer: MagicMock, tmp_path: Any) -> None:
        trainer = _make_trainer(tmp_path)
        trainer.training_config.fsdp = "full_shard"
        trainer.model = MagicMock()
        trainer.ref_model = MagicMock()
        trainer.tokenizer = MagicMock()
        trainer.train_dataset = _fake_dataset()
        trainer.eval_dataset = _fake_dataset()

        trainer.setup_trainer()

        args = mock_trl_trainer.call_args.kwargs["args"]
        # TRL parses fsdp into a list of FSDPOption enums — compare the values.
        assert [getattr(o, "value", o) for o in args.fsdp] == ["full_shard"]


class TestPrecomputeRefLogProbs:
    def test_config_default_off(self) -> None:
        assert DPOConfig().precompute_ref_log_probs is False

    @patch("src.training.dpo_trainer.TRLDPOTrainer")
    def test_forwarded_to_trl_config(self, mock_trl_trainer: MagicMock, tmp_path: Any) -> None:
        trainer = _make_trainer(tmp_path)
        trainer.dpo_config = DPOConfig(precompute_ref_log_probs=True)
        trainer.model = MagicMock()
        trainer.ref_model = MagicMock()
        trainer.tokenizer = MagicMock()
        trainer.train_dataset = _fake_dataset()
        trainer.eval_dataset = _fake_dataset()

        trainer.setup_trainer()

        args = mock_trl_trainer.call_args.kwargs["args"]
        assert args.precompute_ref_log_probs is True


class TestActivationOffloading:
    def test_config_default_off(self) -> None:
        assert DPOConfig().activation_offloading is False

    @patch("src.training.dpo_trainer.TRLDPOTrainer")
    def test_forwarded_to_trl_config(self, mock_trl_trainer: MagicMock, tmp_path: Any) -> None:
        trainer = _make_trainer(tmp_path)
        trainer.dpo_config = DPOConfig(activation_offloading=True)
        trainer.model = MagicMock()
        trainer.ref_model = MagicMock()
        trainer.tokenizer = MagicMock()
        trainer.train_dataset = _fake_dataset()
        trainer.eval_dataset = _fake_dataset()

        trainer.setup_trainer()

        args = mock_trl_trainer.call_args.kwargs["args"]
        assert args.activation_offloading is True


class TestTrain:
    def _armed(self, tmp_path: Any) -> tuple[DPOTrainer, MagicMock]:
        trainer = _make_trainer(tmp_path)
        trainer.model = MagicMock()
        trainer.ref_model = MagicMock()
        trainer.tokenizer = MagicMock()
        hf = MagicMock()
        hf.train.return_value = MagicMock(metrics={"train_loss": 0.5})
        trainer.trainer = hf
        return trainer, hf

    @patch("src.training.dpo_trainer.register_trained_model")
    def test_train_happy_path_saves_and_registers(self, mock_reg: MagicMock, tmp_path: Any) -> None:
        trainer, hf = self._armed(tmp_path)

        result = trainer.train()

        hf.train.assert_called_once()
        hf.save_model.assert_called_once()
        trainer.tokenizer.save_pretrained.assert_called_once_with(
            trainer.training_config.output_dir
        )
        mock_reg.assert_called_once()
        assert result is hf.train.return_value

    def test_train_failure_ends_run_and_reraises(self, tmp_path: Any) -> None:
        trainer, hf = self._armed(tmp_path)
        hf.train.side_effect = RuntimeError("OOM")

        with (
            patch.object(trainer._tracker, "end_run") as mock_end,
            pytest.raises(RuntimeError, match="OOM"),
        ):
            trainer.train()
        mock_end.assert_called_once()

    @patch("src.training.dpo_trainer.register_trained_model")
    def test_train_with_wandb_finishes_run(self, _mock_reg: MagicMock, tmp_path: Any) -> None:
        # wandb is an optional dep (local import in train()) — inject a fake
        # module so `import wandb` resolves without the package installed.
        fake_wandb = MagicMock()
        trainer, _ = self._armed(tmp_path)
        trainer.logging_config.use_wandb = True

        with patch.dict(sys.modules, {"wandb": fake_wandb}):
            trainer.train()

        fake_wandb.init.assert_called_once()
        fake_wandb.finish.assert_called_once()

    def test_evaluate_returns_metrics(self, tmp_path: Any) -> None:
        trainer, hf = self._armed(tmp_path)
        hf.evaluate.return_value = {"eval_loss": 0.33}

        assert trainer.evaluate() == {"eval_loss": 0.33}


class TestRunDPOTraining:
    @patch("src.training.dpo_trainer.DPOTrainer")
    def test_pipeline_calls_all_stages(self, mock_cls: MagicMock, tmp_path: Any) -> None:
        from src.training.dpo_trainer import run_dpo_training

        instance = mock_cls.return_value
        run_dpo_training(
            model_config=ModelConfig(name="qwen-test"),
            training_config=TrainingConfig(output_dir=str(tmp_path / "dpo")),
            lora_config=LoRAConfig(),
            dpo_config=DPOConfig(),
            data_config=PreferenceDataConfig(max_samples=10),
            reference_config=ReferenceModelConfig(name="qwen-test"),
            logging_config=LoggingConfig(),
        )
        mock_cls.assert_called_once()
        instance.prepare_models.assert_called_once()
        instance.prepare_data.assert_called_once()
        instance.setup_trainer.assert_called_once()
        instance.train.assert_called_once()
        instance.evaluate.assert_called_once()


class TestGradientCheckpointingKwargs:
    @patch("src.training.dpo_trainer.TRLDPOTrainer")
    def test_non_reentrant_forwarded_when_checkpointing_on(
        self, mock_trl_trainer: MagicMock, tmp_path: Any
    ) -> None:
        trainer = _make_trainer(tmp_path)  # gradient_checkpointing defaults True
        trainer.model = MagicMock()
        trainer.ref_model = MagicMock()
        trainer.tokenizer = MagicMock()
        trainer.train_dataset = _fake_dataset()
        trainer.eval_dataset = _fake_dataset()

        trainer.setup_trainer()

        args = mock_trl_trainer.call_args.kwargs["args"]
        assert args.gradient_checkpointing_kwargs == {"use_reentrant": False}


class TestTorchEmptyCacheSteps:
    @patch("src.training.dpo_trainer.TRLDPOTrainer")
    def test_value_forwarded(self, mock_trl_trainer: MagicMock, tmp_path: Any) -> None:
        trainer = _make_trainer(tmp_path)
        trainer.training_config.torch_empty_cache_steps = 100
        trainer.model = MagicMock()
        trainer.ref_model = MagicMock()
        trainer.tokenizer = MagicMock()
        trainer.train_dataset = _fake_dataset()
        trainer.eval_dataset = _fake_dataset()

        trainer.setup_trainer()

        args = mock_trl_trainer.call_args.kwargs["args"]
        assert args.torch_empty_cache_steps == 100

    @patch("src.training.dpo_trainer.TRLDPOTrainer")
    def test_auto_find_batch_size_forwarded(
        self, mock_trl_trainer: MagicMock, tmp_path: Any
    ) -> None:
        trainer = _make_trainer(tmp_path)
        trainer.training_config.auto_find_batch_size = True
        trainer.model = MagicMock()
        trainer.ref_model = MagicMock()
        trainer.tokenizer = MagicMock()
        trainer.train_dataset = _fake_dataset()
        trainer.eval_dataset = _fake_dataset()

        trainer.setup_trainer()

        args = mock_trl_trainer.call_args.kwargs["args"]
        assert args.auto_find_batch_size is True

    @patch("src.training.dpo_trainer.TRLDPOTrainer")
    def test_train_sampling_strategy_forwarded(
        self, mock_trl_trainer: MagicMock, tmp_path: Any
    ) -> None:
        trainer = _make_trainer(tmp_path)
        trainer.training_config.train_sampling_strategy = "group_by_length"
        trainer.model = MagicMock()
        trainer.ref_model = MagicMock()
        trainer.tokenizer = MagicMock()
        trainer.train_dataset = _fake_dataset()
        trainer.eval_dataset = _fake_dataset()

        trainer.setup_trainer()

        args = mock_trl_trainer.call_args.kwargs["args"]
        assert args.train_sampling_strategy == "group_by_length"


class TestKBitPreparation:
    @patch("src.training.dpo_trainer.get_platform")
    @patch("src.training.dpo_trainer.get_peft_model")
    @patch("src.training.dpo_trainer.prepare_model_for_kbit_training", return_value=None)
    @patch("src.training.dpo_trainer.load_model_and_tokenizer")
    def test_cuda_quantized_policy_gets_kbit_prep(
        self,
        mock_load: MagicMock,
        mock_kbit: MagicMock,
        mock_peft: MagicMock,
        mock_platform: MagicMock,
        tmp_path: Any,
    ) -> None:
        policy = MagicMock()
        policy_param = MagicMock()
        policy_param.numel.return_value = 100
        policy.parameters.return_value = [policy_param]
        ref = MagicMock()
        ref.parameters.return_value = []

        # First call loads the policy, second the reference model.
        mock_load.side_effect = [(policy, MagicMock()), (ref, MagicMock())]
        mock_peft.return_value = policy  # kbit prep returns None; peft restores the model
        mock_platform.return_value.is_cuda = True

        trainer = _make_trainer(tmp_path)
        trainer.model_config.quantization_bits = 4  # set post-construction (MPS default None)
        trainer.prepare_models()

        mock_kbit.assert_called_once_with(policy)


class TestDeepSpeedInjection:
    # Patch TRLDPOConfig: real construction would require the deepspeed package
    # (TrainingArguments require_versions it on deepspeed= paths).
    @patch("src.training.dpo_trainer.TRLDPOConfig")
    @patch("src.training.dpo_trainer.TRLDPOTrainer")
    def test_setup_trainer_injects_deepspeed(
        self, mock_trl_trainer: MagicMock, mock_cfg_cls: MagicMock, tmp_path: Any
    ) -> None:
        ds_path = "config/distributed/deepspeed_configs/zero_stage_2.json"
        trainer = _make_trainer(tmp_path)
        # Post-construction assignment bypasses the FSDP/DeepSpeed mutual-exclusion check.
        trainer.training_config.deepspeed_config = ds_path
        trainer.model = MagicMock()
        trainer.ref_model = MagicMock()
        trainer.tokenizer = MagicMock()
        trainer.train_dataset = _fake_dataset()
        trainer.eval_dataset = _fake_dataset()

        trainer.setup_trainer()

        kwargs = mock_cfg_cls.call_args.kwargs
        assert kwargs["deepspeed"] == ds_path


class TestFSDPConfigForwarding:
    @patch("src.training.dpo_trainer.TRLDPOConfig")
    @patch("src.training.dpo_trainer.TRLDPOTrainer")
    def test_setup_trainer_forwards_fsdp_config(
        self, mock_trl_trainer: MagicMock, mock_cfg_cls: MagicMock, tmp_path: Any
    ) -> None:
        trainer = _make_trainer(tmp_path)
        trainer.training_config.fsdp = "full_shard"
        trainer.training_config.fsdp_config = {"transformer_layer_cls_to_wrap": "Qwen2DecoderLayer"}
        trainer.model = MagicMock()
        trainer.ref_model = MagicMock()
        trainer.tokenizer = MagicMock()
        trainer.train_dataset = _fake_dataset()
        trainer.eval_dataset = _fake_dataset()

        trainer.setup_trainer()

        kwargs = mock_cfg_cls.call_args.kwargs
        assert kwargs["fsdp"] == "full_shard"
        assert kwargs["fsdp_config"]["transformer_layer_cls_to_wrap"] == "Qwen2DecoderLayer"


class TestDistributedBanner:
    @patch("src.training.dpo_trainer.get_distributed_info")
    @patch("src.training.dpo_trainer.TRLDPOTrainer")
    def test_distributed_banner_shows_world_size(
        self, mock_trl_trainer: MagicMock, mock_dist: MagicMock, tmp_path: Any
    ) -> None:
        mock_dist.return_value = MagicMock(is_distributed=True, world_size=4)
        trainer = _make_trainer(tmp_path)
        trainer.model = MagicMock()
        trainer.ref_model = MagicMock()
        trainer.tokenizer = MagicMock()
        trainer.train_dataset = _fake_dataset()
        trainer.eval_dataset = _fake_dataset()

        trainer.setup_trainer()

        mock_dist.assert_called_once()


class TestTrainMlflowStart:
    @patch("src.training.dpo_trainer.register_trained_model")
    def test_mlflow_run_started_when_tracker_active(
        self, mock_reg: MagicMock, tmp_path: Any
    ) -> None:
        trainer = _make_trainer(tmp_path)
        trainer.trainer = MagicMock()
        trainer.trainer.train.return_value = MagicMock(metrics={"train_loss": 0.5})
        trainer.tokenizer = MagicMock()
        tracker = MagicMock()
        tracker.active = True
        trainer._tracker = tracker

        trainer.train()

        tracker.start_run.assert_called_once()
        tracker.end_run.assert_called_once()


class TestMFUCallbackAttachment:
    @patch("src.training.dpo_trainer.TRLDPOTrainer")
    def test_setup_trainer_attaches_mfu_callback(
        self, mock_trl_trainer: MagicMock, tmp_path: Any
    ) -> None:
        from src.training.callbacks import MFUCallback

        trainer = _make_trainer(tmp_path)
        trainer.model = MagicMock()
        trainer.ref_model = MagicMock()
        trainer.tokenizer = MagicMock()
        trainer.train_dataset = _fake_dataset()
        trainer.eval_dataset = _fake_dataset()

        trainer.setup_trainer()

        callbacks = mock_trl_trainer.call_args.kwargs["callbacks"]
        assert any(isinstance(c, MFUCallback) for c in callbacks)
        mfu_cb = next(c for c in callbacks if isinstance(c, MFUCallback))
        assert mfu_cb._fpt is None  # MagicMock config → graceful disable
        # DPO counts both sides of each preference pair.
        assert mfu_cb.tokens_per_step == (
            2
            * trainer.training_config.batch_size
            * trainer.training_config.gradient_accumulation_steps
            * trainer.dpo_config.max_length
        )
