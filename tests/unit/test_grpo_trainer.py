"""Unit tests for GRPO trainer setup (without actual model loading)."""

from unittest.mock import MagicMock, patch

import pytest

from config.base import LoggingConfig, TrainingConfig
from config.grpo import GRPOConfig, GRPOTrainingConfig, RewardConfig
from src.training.grpo_trainer import GRPOTrainer, MemoryCallback, run_grpo_training


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

    def test_vllm_fields_pass_through(self) -> None:
        cfg = GRPOTrainingConfig(
            grpo_config=GRPOConfig(
                use_vllm=True,
                vllm_mode="colocate",
                vllm_gpu_memory_utilization=0.25,
            ),
        )
        trl_cfg = GRPOTrainer(cfg)._build_grpo_config()
        assert trl_cfg.use_vllm is True
        assert trl_cfg.vllm_mode == "colocate"
        assert trl_cfg.vllm_gpu_memory_utilization == pytest.approx(0.25)

    def test_use_liger_kernel_pass_through(self) -> None:
        cfg = GRPOTrainingConfig(
            training_config=TrainingConfig(use_liger_kernel=True),
        )
        trl_cfg = GRPOTrainer(cfg)._build_grpo_config()
        assert trl_cfg.use_liger_kernel is True

    def test_liger_defaults_off(self) -> None:
        trl_cfg = GRPOTrainer(GRPOTrainingConfig())._build_grpo_config()
        assert trl_cfg.use_liger_kernel is False

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

    @patch("src.training.grpo_trainer.get_peft_model")
    @patch("src.training.grpo_trainer.prepare_model_for_kbit_training", return_value=None)
    @patch("src.training.grpo_trainer.load_model_and_tokenizer")
    def test_prepare_model_passes_use_dora(
        self,
        mock_load: MagicMock,
        mock_kbit: MagicMock,
        mock_peft: MagicMock,
    ) -> None:
        mock_model = MagicMock()
        mock_load.return_value = (mock_model, MagicMock())
        mock_peft.return_value = mock_model
        param = MagicMock()
        param.numel.return_value = 1000
        param.requires_grad = True
        mock_model.parameters.return_value = [param]

        cfg = GRPOTrainingConfig()
        cfg.lora_config.use_dora = True
        GRPOTrainer(cfg).prepare_model()
        peft_lora_cfg = mock_peft.call_args[0][1]
        assert peft_lora_cfg.use_dora is True

    def test_torch_compile_pass_through(self) -> None:
        cfg = GRPOTrainingConfig(training_config=TrainingConfig(torch_compile=True))
        trl_cfg = GRPOTrainer(cfg)._build_grpo_config()
        assert trl_cfg.torch_compile is True

    def test_neftune_not_forwarded_to_grpo(self) -> None:
        # NEFTune is validated for instruction SFT only — GRPO must keep the
        # inherited field at its default even when the training config sets it.
        cfg = GRPOTrainingConfig(training_config=TrainingConfig(neftune_noise_alpha=5.0))
        trl_cfg = GRPOTrainer(cfg)._build_grpo_config()
        assert trl_cfg.neftune_noise_alpha is None

    def test_optim_pass_through(self) -> None:
        cfg = GRPOTrainingConfig(training_config=TrainingConfig(optim="paged_adamw_8bit"))
        trl_cfg = GRPOTrainer(cfg)._build_grpo_config()
        assert trl_cfg.optim == "paged_adamw_8bit"

    def test_optim_default_not_injected(self) -> None:
        trl_cfg = GRPOTrainer(GRPOTrainingConfig())._build_grpo_config()
        # Library default flows through untouched.
        assert trl_cfg.optim == "adamw_torch_fused"

    @patch("src.training.grpo_trainer.get_peft_model")
    @patch("src.training.grpo_trainer.prepare_model_for_kbit_training", return_value=None)
    @patch("src.training.grpo_trainer.load_model_and_tokenizer")
    def test_prepare_model_passes_init_strategy(
        self, mock_load: MagicMock, mock_kbit: MagicMock, mock_peft: MagicMock
    ) -> None:
        mock_model = MagicMock()
        param = MagicMock()
        param.numel.return_value = 1000
        param.requires_grad = True
        mock_model.parameters.return_value = [param]
        mock_load.return_value = (mock_model, MagicMock())
        mock_peft.return_value = mock_model

        cfg = GRPOTrainingConfig()
        cfg.lora_config.init_lora_weights = "pissa"
        GRPOTrainer(cfg).prepare_model()

        assert mock_peft.call_args[0][1].init_lora_weights == "pissa"

    @patch("src.training.grpo_trainer.get_peft_model")
    @patch("src.training.grpo_trainer.prepare_model_for_kbit_training", return_value=None)
    @patch("src.training.grpo_trainer.load_model_and_tokenizer")
    def test_prepare_model_injects_loftq_config(
        self, mock_load: MagicMock, mock_kbit: MagicMock, mock_peft: MagicMock
    ) -> None:
        mock_model = MagicMock()
        param = MagicMock()
        param.numel.return_value = 1000
        param.requires_grad = True
        mock_model.parameters.return_value = [param]
        mock_load.return_value = (mock_model, MagicMock())
        mock_peft.return_value = mock_model

        cfg = GRPOTrainingConfig()
        cfg.lora_config.init_lora_weights = "loftq"
        cfg.lora_config.loftq_bits = 4
        cfg.lora_config.loftq_iter = 2
        GRPOTrainer(cfg).prepare_model()

        # peft hard-errors on loftq without the dict — trainers must inject it.
        assert mock_peft.call_args[0][1].loftq_config == {
            "loftq_bits": 4,
            "loftq_iter": 2,
        }

    @patch("src.training.grpo_trainer.get_peft_model")
    @patch("src.training.grpo_trainer.prepare_model_for_kbit_training", return_value=None)
    @patch("src.training.grpo_trainer.load_model_and_tokenizer")
    def test_prepare_model_forwards_lora_patterns(
        self, mock_load: MagicMock, mock_kbit: MagicMock, mock_peft: MagicMock
    ) -> None:
        mock_model = MagicMock()
        param = MagicMock()
        param.numel.return_value = 1000
        param.requires_grad = True
        mock_model.parameters.return_value = [param]
        mock_load.return_value = (mock_model, MagicMock())
        mock_peft.return_value = mock_model

        cfg = GRPOTrainingConfig()
        cfg.lora_config.rank_pattern = {"experts.gate_up_proj": 32}
        cfg.lora_config.alpha_pattern = {"experts.gate_up_proj": 64}
        cfg.lora_config.exclude_modules = ["lm_head"]
        GRPOTrainer(cfg).prepare_model()

        peft_lora_cfg = mock_peft.call_args[0][1]
        assert peft_lora_cfg.rank_pattern == {"experts.gate_up_proj": 32}
        assert peft_lora_cfg.alpha_pattern == {"experts.gate_up_proj": 64}
        # peft normalizes the exclude list to a set.
        assert peft_lora_cfg.exclude_modules == {"lm_head"}


class TestGradientCheckpointingKwargs:
    def test_non_reentrant_forwarded_when_checkpointing_on(self) -> None:
        # gradient_checkpointing defaults True in TrainingConfig
        trl_cfg = GRPOTrainer(GRPOTrainingConfig())._build_grpo_config()
        assert trl_cfg.gradient_checkpointing_kwargs == {"use_reentrant": False}

    def test_kwargs_none_when_checkpointing_off(self) -> None:
        cfg = GRPOTrainingConfig()
        cfg.training_config.gradient_checkpointing = False
        trl_cfg = GRPOTrainer(cfg)._build_grpo_config()
        assert trl_cfg.gradient_checkpointing_kwargs is None


class TestDapoFlags:
    def test_defaults_match_trl(self) -> None:
        from config.grpo import GRPOConfig

        cfg = GRPOConfig()
        assert cfg.epsilon_high is None  # symmetric clipping by default
        assert cfg.mask_truncated_completions is False

    def test_epsilon_high_validation(self) -> None:
        from config.grpo import GRPOConfig

        with pytest.raises(ValueError, match="epsilon_high"):
            GRPOConfig(epsilon_high=1.5)

    def test_dapo_flags_forwarded(self) -> None:
        from config.grpo import GRPOConfig

        cfg = GRPOTrainingConfig(
            grpo_config=GRPOConfig(epsilon=0.2, epsilon_high=0.28, mask_truncated_completions=True)
        )
        trl_cfg = GRPOTrainer(cfg)._build_grpo_config()

        assert trl_cfg.epsilon == 0.2
        assert trl_cfg.epsilon_high == 0.28
        assert trl_cfg.mask_truncated_completions is True

    def test_top_entropy_quantile_forwarded(self) -> None:
        from config.grpo import GRPOConfig

        cfg = GRPOTrainingConfig(grpo_config=GRPOConfig(top_entropy_quantile=0.2))
        trl_cfg = GRPOTrainer(cfg)._build_grpo_config()

        assert trl_cfg.top_entropy_quantile == pytest.approx(0.2)

    def test_top_entropy_quantile_default_passthrough(self) -> None:
        trl_cfg = GRPOTrainer(GRPOTrainingConfig())._build_grpo_config()
        assert trl_cfg.top_entropy_quantile == pytest.approx(1.0)

    def test_min_p_and_generation_kwargs_forwarded(self) -> None:
        from config.grpo import GRPOConfig

        cfg = GRPOTrainingConfig(
            grpo_config=GRPOConfig(
                min_p=0.05,
                generation_kwargs={"no_repeat_ngram_size": 4},
            )
        )
        trl_cfg = GRPOTrainer(cfg)._build_grpo_config()

        assert trl_cfg.min_p == pytest.approx(0.05)
        assert trl_cfg.generation_kwargs == {"no_repeat_ngram_size": 4}

    def test_min_p_and_generation_kwargs_default_passthrough(self) -> None:
        trl_cfg = GRPOTrainer(GRPOTrainingConfig())._build_grpo_config()
        # Library defaults flow through untouched (no forced sampling).
        assert trl_cfg.min_p is None
        assert trl_cfg.generation_kwargs is None


class TestImportanceSamplingLevel:
    def test_default_token(self) -> None:
        from config.grpo import GRPOConfig

        assert GRPOConfig().importance_sampling_level == "token"

    def test_invalid_level_rejected(self) -> None:
        from config.grpo import GRPOConfig

        with pytest.raises(Exception, match="importance_sampling_level"):
            GRPOConfig(importance_sampling_level="sentence")

    def test_sequence_level_forwarded(self) -> None:
        from config.grpo import GRPOConfig

        cfg = GRPOTrainingConfig(grpo_config=GRPOConfig(importance_sampling_level="sequence"))
        trl_cfg = GRPOTrainer(cfg)._build_grpo_config()

        assert trl_cfg.importance_sampling_level == "sequence"


class TestTorchEmptyCacheSteps:
    def test_default_none(self) -> None:
        trl_cfg = GRPOTrainer(GRPOTrainingConfig())._build_grpo_config()
        assert trl_cfg.torch_empty_cache_steps is None

    def test_value_forwarded(self) -> None:
        cfg = GRPOTrainingConfig(training_config=TrainingConfig(torch_empty_cache_steps=100))
        trl_cfg = GRPOTrainer(cfg)._build_grpo_config()
        assert trl_cfg.torch_empty_cache_steps == 100

    def test_auto_find_batch_size_forwarded(self) -> None:
        cfg = GRPOTrainingConfig(training_config=TrainingConfig(auto_find_batch_size=True))
        trl_cfg = GRPOTrainer(cfg)._build_grpo_config()
        assert trl_cfg.auto_find_batch_size is True

    def test_train_sampling_strategy_forwarded(self) -> None:
        cfg = GRPOTrainingConfig(
            training_config=TrainingConfig(train_sampling_strategy="group_by_length")
        )
        trl_cfg = GRPOTrainer(cfg)._build_grpo_config()
        assert trl_cfg.train_sampling_strategy == "group_by_length"


class TestMemoryCallback:
    def test_logs_at_interval(self) -> None:
        cb = MemoryCallback(log_steps=10)
        state = MagicMock()
        state.global_step = 20
        control = MagicMock()
        with patch("src.utils.log_gpu_memory") as mock_log:
            result = cb.on_step_end(MagicMock(), state, control)
        mock_log.assert_called_once_with(20, wandb_run=None)
        assert result is control

    def test_silent_between_intervals(self) -> None:
        cb = MemoryCallback(log_steps=10)
        state = MagicMock()
        state.global_step = 23
        with patch("src.utils.log_gpu_memory") as mock_log:
            cb.on_step_end(MagicMock(), state, MagicMock())
        mock_log.assert_not_called()


class TestPrepareData:
    @patch("src.training.grpo_trainer.GRPODataset")
    def test_loads_and_splits(self, mock_ds_cls: MagicMock) -> None:
        ds = mock_ds_cls.return_value
        train_ds = MagicMock()
        train_ds.__len__.return_value = 8
        eval_ds = MagicMock()
        eval_ds.__len__.return_value = 2
        ds.split_dataset.return_value = (train_ds, eval_ds)

        trainer = GRPOTrainer(GRPOTrainingConfig())
        trainer.prepare_data()

        mock_ds_cls.assert_called_once_with(
            data_path="yahma/alpaca-cleaned",
            max_samples=None,
            prompt_key="prompt",
            answer_key="answer",
            reference_key="reference",
        )
        ds.load.assert_called_once()
        assert trainer.train_dataset is train_ds
        assert trainer.eval_dataset is eval_ds


class TestTrain:
    @patch("src.training.grpo_trainer.register_trained_model")
    def test_happy_path_saves_registers_and_returns_result(self, mock_register: MagicMock) -> None:
        cfg = GRPOTrainingConfig()
        trainer = GRPOTrainer(cfg)
        trainer.trainer = MagicMock()
        trainer.trainer.train.return_value = "train_result"
        trainer.tokenizer = MagicMock()
        trainer._tracker = MagicMock()
        trainer._tracker.active = False

        result = trainer.train()

        assert result == "train_result"
        trainer.trainer.save_model.assert_called_once()
        trainer.tokenizer.save_pretrained.assert_called_once_with(cfg.training_config.output_dir)
        mock_register.assert_called_once()
        trainer._tracker.end_run.assert_called_once()

    @patch("src.training.grpo_trainer.register_trained_model")
    def test_failure_ends_run_and_reraises(self, mock_register: MagicMock) -> None:
        trainer = GRPOTrainer(GRPOTrainingConfig())
        trainer.trainer = MagicMock()
        trainer.trainer.train.side_effect = RuntimeError("boom")
        trainer.tokenizer = MagicMock()
        trainer._tracker = MagicMock()
        trainer._tracker.active = False

        with pytest.raises(RuntimeError, match="boom"):
            trainer.train()

        trainer._tracker.end_run.assert_called_once()
        trainer.trainer.save_model.assert_not_called()
        mock_register.assert_not_called()

    @patch("src.training.grpo_trainer.register_trained_model")
    def test_wandb_lifecycle(self, _mock_register: MagicMock) -> None:
        cfg = GRPOTrainingConfig(logging_config=LoggingConfig(use_wandb=True))
        fake_run = MagicMock()
        trainer = GRPOTrainer(cfg)
        trainer.trainer = MagicMock()
        trainer.tokenizer = MagicMock()
        trainer._tracker = MagicMock()
        trainer._tracker.active = False

        with patch("src.utils.setup_wandb", return_value=fake_run) as mock_setup:
            trainer.train()

        mock_setup.assert_called_once()
        fake_run.finish.assert_called_once()

    @patch("src.training.grpo_trainer.register_trained_model")
    def test_mlflow_run_started_when_active(self, mock_register: MagicMock) -> None:
        trainer = GRPOTrainer(GRPOTrainingConfig())
        trainer.trainer = MagicMock()
        trainer.tokenizer = MagicMock()
        trainer._tracker = MagicMock()
        trainer._tracker.active = True

        trainer.train()

        trainer._tracker.start_run.assert_called_once()
        mock_register.assert_called_once()


class TestEvaluate:
    def test_returns_metrics(self) -> None:
        trainer = GRPOTrainer(GRPOTrainingConfig())
        trainer.trainer = MagicMock()
        trainer.trainer.evaluate.return_value = {"eval_reward": 0.5}

        metrics = trainer.evaluate()

        assert metrics == {"eval_reward": 0.5}


class TestRunGRPOTraining:
    @patch("src.training.grpo_trainer.GRPOTrainer")
    def test_five_stage_pipeline(self, mock_cls: MagicMock) -> None:
        instance = mock_cls.return_value

        run_grpo_training(GRPOTrainingConfig())

        mock_cls.assert_called_once()
        instance.prepare_model.assert_called_once()
        instance.prepare_data.assert_called_once()
        instance.setup_trainer.assert_called_once()
        instance.train.assert_called_once()
        instance.evaluate.assert_called_once()


class TestKBitPreparation:
    def _model_double(self) -> MagicMock:
        mock_model = MagicMock()
        param = MagicMock()
        param.numel.return_value = 1000
        param.requires_grad = True
        mock_model.parameters.return_value = [param]
        return mock_model

    @patch("src.training.grpo_trainer.get_platform")
    @patch("src.training.grpo_trainer.get_peft_model")
    @patch("src.training.grpo_trainer.prepare_model_for_kbit_training", return_value=None)
    @patch("src.training.grpo_trainer.load_model_and_tokenizer")
    def test_cuda_quantized_model_gets_kbit_prep(
        self,
        mock_load: MagicMock,
        mock_kbit: MagicMock,
        mock_peft: MagicMock,
        mock_platform: MagicMock,
    ) -> None:
        mock_model = self._model_double()
        mock_load.return_value = (mock_model, MagicMock())
        mock_peft.return_value = mock_model
        mock_platform.return_value.is_cuda = True

        cfg = GRPOTrainingConfig()
        cfg.model_config.quantization_bits = 4
        GRPOTrainer(cfg).prepare_model()

        mock_kbit.assert_called_once_with(mock_model)

    @patch("src.training.grpo_trainer.get_platform")
    @patch("src.training.grpo_trainer.get_peft_model")
    @patch("src.training.grpo_trainer.prepare_model_for_kbit_training", return_value=None)
    @patch("src.training.grpo_trainer.load_model_and_tokenizer")
    def test_non_cuda_skips_kbit_prep(
        self,
        mock_load: MagicMock,
        mock_kbit: MagicMock,
        mock_peft: MagicMock,
        mock_platform: MagicMock,
    ) -> None:
        mock_model = self._model_double()
        mock_load.return_value = (mock_model, MagicMock())
        mock_peft.return_value = mock_model
        mock_platform.return_value.is_cuda = False

        cfg = GRPOTrainingConfig()
        cfg.model_config.quantization_bits = 4
        GRPOTrainer(cfg).prepare_model()

        mock_kbit.assert_not_called()


class TestRewardWeights:
    def test_weights_aligned_to_func_order_with_default_1(self) -> None:
        cfg = GRPOTrainingConfig(
            reward_config=RewardConfig(
                reward_funcs=["format", "accuracy"],
                reward_weights={"format": 2.0},
            ),
        )
        trl_cfg = GRPOTrainer(cfg)._build_grpo_config()
        assert trl_cfg.reward_weights == [2.0, 1.0]

    def test_absent_weights_not_injected(self) -> None:
        trl_cfg = GRPOTrainer(GRPOTrainingConfig())._build_grpo_config()
        assert trl_cfg.reward_weights is None


class TestDistributedInjection:
    @patch("src.training.grpo_trainer.TRLGRPOConfig")
    def test_fsdp_and_fsdp_config_forwarded(self, mock_cfg_cls: MagicMock) -> None:
        cfg = GRPOTrainingConfig()
        cfg.training_config.fsdp = "full_shard"
        cfg.training_config.fsdp_config = {"transformer_layer_cls_to_wrap": "Qwen2DecoderLayer"}

        GRPOTrainer(cfg)._build_grpo_config()

        kwargs = mock_cfg_cls.call_args.kwargs
        assert kwargs["fsdp"] == "full_shard"
        assert kwargs["fsdp_config"]["transformer_layer_cls_to_wrap"] == "Qwen2DecoderLayer"

    @patch("src.training.grpo_trainer.TRLGRPOConfig")
    def test_deepspeed_forwarded(self, mock_cfg_cls: MagicMock) -> None:
        ds_path = "config/distributed/deepspeed_configs/zero_stage_2.json"
        cfg = GRPOTrainingConfig()
        cfg.training_config.deepspeed_config = ds_path

        GRPOTrainer(cfg)._build_grpo_config()

        assert mock_cfg_cls.call_args.kwargs["deepspeed"] == ds_path

    def test_fsdp_real_config_parses_options(self) -> None:
        # Real TRL construction parses the string into FSDPOption enums —
        # mirroring the DPO-side contract.
        cfg = GRPOTrainingConfig()
        cfg.training_config.fsdp = "full_shard"
        trl_cfg = GRPOTrainer(cfg)._build_grpo_config()
        assert [getattr(o, "value", o) for o in trl_cfg.fsdp] == ["full_shard"]

    @patch("src.training.grpo_trainer.get_distributed_info")
    def test_distributed_banner_shows_world_size(self, mock_dist: MagicMock) -> None:
        mock_dist.return_value = MagicMock(is_distributed=True, world_size=4)
        trl_cfg = GRPOTrainer(GRPOTrainingConfig())._build_grpo_config()
        assert trl_cfg is not None
        mock_dist.assert_called_once()


class TestSetupTrainer:
    @patch("src.training.grpo_trainer.TRLGRPOTrainer")
    def test_constructs_trl_trainer_with_components(self, mock_trl: MagicMock) -> None:
        cfg = GRPOTrainingConfig(
            reward_config=RewardConfig(reward_funcs=["format"]),
        )
        trainer = GRPOTrainer(cfg)
        trainer.model = MagicMock()
        trainer.tokenizer = MagicMock()
        trainer.train_dataset = MagicMock()
        trainer.eval_dataset = MagicMock()
        trainer._tracker = MagicMock()

        trainer.setup_trainer()

        kwargs = mock_trl.call_args.kwargs
        assert kwargs["model"] is trainer.model
        assert kwargs["processing_class"] is trainer.tokenizer
        assert kwargs["train_dataset"] is trainer.train_dataset
        assert kwargs["eval_dataset"] is trainer.eval_dataset
        assert len(kwargs["reward_funcs"]) == 1
        assert len(kwargs["callbacks"]) == 2  # MemoryCallback + MLflowTrainCallback
        assert kwargs["args"].num_generations == cfg.grpo_config.num_generations
        assert trainer.trainer is mock_trl.return_value


class TestPrepareDataNoEval:
    @patch("src.training.grpo_trainer.GRPODataset")
    def test_no_eval_dataset_prints_train_only(self, mock_ds_cls: MagicMock) -> None:
        ds = mock_ds_cls.return_value
        train_ds = MagicMock()
        train_ds.__len__.return_value = 8
        ds.split_dataset.return_value = (train_ds, None)

        trainer = GRPOTrainer(GRPOTrainingConfig())
        trainer.prepare_data()

        assert trainer.train_dataset is train_ds
        assert trainer.eval_dataset is None
