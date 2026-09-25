"""GRPO (Group Relative Policy Optimization) trainer implementation.

Wraps TRL's GRPOTrainer and integrates with the project's QLoRA + tracking stack.
"""

from __future__ import annotations

import os
from typing import Any

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import PreTrainedModel, TrainerCallback
from trl import GRPOConfig as TRLGRPOConfig
from trl import GRPOTrainer as TRLGRPOTrainer

from config.grpo import GRPOTrainingConfig
from src.data.grpo_dataset import GRPODataset
from src.models import load_model_and_tokenizer
from src.tracking import MLflowTrainCallback, get_tracker, register_trained_model
from src.training.distributed import get_distributed_info
from src.training.reward_engine import RewardFunction, build_reward_functions
from src.utils import console, set_seed, setup_logging
from src.utils.platform_utils import get_platform

# Element type accepted by TRL GRPOTrainer's `reward_funcs` argument (functions
# may also be given as model id strings or PreTrainedModel judge instances).
TRLRewardFunc = str | PreTrainedModel | RewardFunction


class MemoryCallback(TrainerCallback):
    """Callback to log GPU memory usage during GRPO training."""

    def __init__(self, log_steps: int = 100) -> None:
        """Initialize memory callback."""
        super().__init__()
        self.log_steps = log_steps

    def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
        """Log memory at end of step."""
        if state.global_step % self.log_steps == 0:
            from src.utils import log_gpu_memory

            log_gpu_memory(state.global_step, wandb_run=None)
        return control


class GRPOTrainer:
    """GRPO trainer with QLoRA and memory optimizations."""

    def __init__(self, config: GRPOTrainingConfig) -> None:
        """Initialize GRPO trainer.

        Args:
            config: Complete GRPO training configuration.
        """
        self.config = config

        # Set random seed
        set_seed(config.training_config.seed)

        # Setup logging
        self.logger = setup_logging(
            log_file=os.path.join(config.training_config.output_dir, "grpo_training.log"),
            level=config.logging_config.console_level,
        )

        # Models and tokenizer (loaded later)
        self.model: Any = None
        self.ref_model: Any = None
        self.tokenizer: Any = None
        self.trainer: TRLGRPOTrainer | None = None

        # MLflow tracker (no-op if use_mlflow=False)
        self._tracker = get_tracker(config.logging_config)

    def prepare_model(self) -> None:
        """Load policy model, apply LoRA, and load reference model."""
        console.print("\n[bold cyan]=== Preparing GRPO Models ===[/bold cyan]\n")

        # Load policy model and tokenizer
        console.print(f"[cyan]Loading policy model: {self.config.model_config.name}[/cyan]")
        self.model, self.tokenizer = load_model_and_tokenizer(self.config.model_config)

        # Prepare for k-bit training (CUDA only)
        platform_info = get_platform()
        if platform_info.is_cuda and self.config.model_config.quantization_bits in (4, 8):
            console.print("[cyan]Preparing model for k-bit training...[/cyan]")
            self.model = prepare_model_for_kbit_training(self.model)
        else:
            console.print("[cyan]Skipping k-bit preparation (not needed on this platform)[/cyan]")

        # Apply LoRA
        lora_kwargs: dict[str, Any] = dict(
            r=self.config.lora_config.r,
            lora_alpha=self.config.lora_config.lora_alpha,
            lora_dropout=self.config.lora_config.lora_dropout,
            target_modules=self.config.lora_config.target_modules,
            bias=self.config.lora_config.bias,
            task_type=self.config.lora_config.task_type,
            use_dora=self.config.lora_config.use_dora,
            use_rslora=self.config.lora_config.use_rslora,
            init_lora_weights=self.config.lora_config.init_lora_weights,
        )
        if self.config.lora_config.init_lora_weights == "loftq":
            # peft hard-errors on loftq without the dict — inject it here.
            lora_kwargs["loftq_config"] = {
                "loftq_bits": self.config.lora_config.loftq_bits,
                "loftq_iter": self.config.lora_config.loftq_iter,
            }
        # Per-module rank/alpha overrides and module exclusions (peft
        # regex-keyed patterns; None = peft defaults, no-op).
        if self.config.lora_config.rank_pattern is not None:
            lora_kwargs["rank_pattern"] = self.config.lora_config.rank_pattern
        if self.config.lora_config.alpha_pattern is not None:
            lora_kwargs["alpha_pattern"] = self.config.lora_config.alpha_pattern
        if self.config.lora_config.exclude_modules is not None:
            lora_kwargs["exclude_modules"] = self.config.lora_config.exclude_modules
        lora_cfg = LoraConfig(**lora_kwargs)
        console.print(
            f"[cyan]Applying LoRA (r={self.config.lora_config.r}, "
            f"alpha={self.config.lora_config.lora_alpha})...[/cyan]"
        )
        self.model = get_peft_model(self.model, lora_cfg)

        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        console.print(
            f"[green]✓ Policy model ready "
            f"({trainable_params:,} trainable, {trainable_params / total_params:.2%})[/green]"
        )

        # Reference policy: TRL >= 1.4 manages it internally (adapter-disable
        # path for PEFT policies, auto-created reference otherwise; see
        # GRPOConfig.beta / sync_ref_model). Pre-loading a frozen copy would
        # double VRAM usage for nothing, so reference_config is not loaded.
        console.print(
            "[cyan]Reference policy managed by TRL internally "
            "(reference_config unused — saves ~1x model VRAM)[/cyan]\n"
        )

    def prepare_data(self) -> None:
        """Load and prepare GRPO dataset."""
        console.print("\n[bold cyan]=== Preparing GRPO Data ===[/bold cyan]\n")

        data_cfg = self.config.data_config
        dataset = GRPODataset(
            data_path=data_cfg.dataset_name,
            max_samples=data_cfg.max_samples,
            prompt_key=getattr(data_cfg, "prompt_key", "prompt"),
            answer_key=getattr(data_cfg, "answer_key", "answer"),
            reference_key=getattr(data_cfg, "reference_key", "reference"),
        )

        dataset.load()
        self.train_dataset, self.eval_dataset = dataset.split_dataset(
            validation_split=data_cfg.validation_split,
            seed=self.config.training_config.seed,
        )

        console.print(f"[green]✓ Train samples: {len(self.train_dataset):,}[/green]")
        if self.eval_dataset is not None:
            console.print(f"[green]✓ Validation samples: {len(self.eval_dataset):,}[/green]")
        console.print()

    def _build_reward_funcs(self) -> list[TRLRewardFunc]:
        """Build reward functions from config."""
        reward_kwargs: dict[str, Any] = {
            "judge_model": self.config.reward_config.judge_model,
            "judge_prompt_template": self.config.reward_config.judge_prompt_template,
            "answer_key": "answer",
            "reference_key": "reference",
        }

        funcs_with_weights = build_reward_functions(
            names=self.config.reward_config.reward_funcs,
            weights=self.config.reward_config.reward_weights,
            **reward_kwargs,
        )

        # Return only the functions; weights are handled by TRL's reward_weights arg.
        funcs: list[TRLRewardFunc] = [fn for fn, _ in funcs_with_weights]
        return funcs

    def _build_grpo_config(self) -> TRLGRPOConfig:
        """Build TRL GRPOConfig from project config."""
        dist_info = get_distributed_info()
        cfg = self.config.grpo_config
        training = self.config.training_config

        kwargs: dict[str, Any] = dict(
            output_dir=training.output_dir,
            # Opt-in Triton fused kernels — see SFTTrainer note ([liger] extra).
            use_liger_kernel=training.use_liger_kernel,
            # Opt-in torch.compile — bf16 full-precision path only (Dynamo
            # cannot trace bitsandbytes 4-bit ops; see TrainingConfig).
            torch_compile=training.torch_compile,
            num_train_epochs=training.num_epochs,
            per_device_train_batch_size=training.batch_size,
            per_device_eval_batch_size=training.batch_size,
            gradient_accumulation_steps=training.gradient_accumulation_steps,
            learning_rate=training.learning_rate,
            weight_decay=training.weight_decay,
            # transformers >= 5.x: warmup_ratio merged into warmup_steps
            # (a float value keeps ratio semantics).
            warmup_steps=training.warmup_ratio,
            lr_scheduler_type=training.lr_scheduler_type,
            logging_steps=training.logging_steps,
            save_steps=training.save_steps,
            eval_steps=training.eval_steps,
            save_total_limit=training.save_total_limit,
            gradient_checkpointing=training.gradient_checkpointing,
            # Non-reentrant checkpointing — PyTorch-recommended and the
            # reliable path for frozen-base (LoRA/QLoRA) fine-tunes.
            gradient_checkpointing_kwargs=(
                {"use_reentrant": training.gradient_checkpointing_use_reentrant}
                if training.gradient_checkpointing
                else None
            ),
            # Periodic torch.cuda.empty_cache() — opt-in fragmentation relief
            # for 8 GB cards with variable-length batches (None = off).
            torch_empty_cache_steps=training.torch_empty_cache_steps,
            # On OOM, auto-restart with halved batch size (ZeRO-3 excluded
            # at config validation; see TrainingConfig).
            auto_find_batch_size=training.auto_find_batch_size,
            # Length-grouped sampling — padding-waste reduction for
            # variable-length prompts (opt-in).
            train_sampling_strategy=training.train_sampling_strategy,
            fp16=training.fp16,
            bf16=training.bf16,
            max_grad_norm=training.max_grad_norm,
            seed=training.seed,
            report_to=["tensorboard"] if self.config.logging_config.use_tensorboard else [],
            logging_dir=self.config.logging_config.log_dir
            if self.config.logging_config.use_tensorboard
            else None,
            remove_unused_columns=False,
            # GRPO-specific
            beta=cfg.beta,
            num_generations=cfg.num_generations,
            num_generations_eval=cfg.num_generations,
            max_completion_length=cfg.max_completion_length,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            top_k=cfg.top_k,
            repetition_penalty=cfg.repetition_penalty,
            # Min-p sampling floor (None = off). Config warns when combined
            # with use_vllm — truncation biases the IS correction (#6789).
            min_p=cfg.min_p,
            # Escape-hatch sampling knobs TRL lacks first-class fields for;
            # conflicting keys override the top-level params (TRL semantics).
            generation_kwargs=cfg.generation_kwargs,
            use_vllm=cfg.use_vllm,
            vllm_mode=cfg.vllm_mode,
            vllm_gpu_memory_utilization=cfg.vllm_gpu_memory_utilization,
            scale_rewards=cfg.scale_rewards,
            num_iterations=cfg.num_iterations,
            epsilon=cfg.epsilon,
            # DAPO Clip-Higher (arXiv:2503.14476): asymmetric clip range
            # [epsilon, epsilon_high]; None → symmetric (TRL default).
            epsilon_high=cfg.epsilon_high,
            # DAPO Overlong Filtering: exclude max-length-truncated
            # completions from the loss.
            mask_truncated_completions=cfg.mask_truncated_completions,
            # GSPO (arXiv:2507.18071): "sequence" computes importance
            # ratios per sequence instead of per token.
            importance_sampling_level=cfg.importance_sampling_level,
            # Entropy-based token filtering (arXiv:2506.01939, Beyond the
            # 80/20 Rule): 1.0 keeps all tokens; paper recommends 0.2.
            top_entropy_quantile=cfg.top_entropy_quantile,
            loss_type=cfg.loss_type,
        )

        if self.config.reward_config.reward_weights:
            kwargs["reward_weights"] = [
                self.config.reward_config.reward_weights.get(name, 1.0)
                for name in self.config.reward_config.reward_funcs
            ]

        # Distributed strategy
        if training.fsdp:
            kwargs["fsdp"] = training.fsdp
            if training.fsdp_config:
                kwargs["fsdp_config"] = training.fsdp_config
            console.print(f"[green]✓ FSDP enabled (GRPO): {training.fsdp}[/green]")
        elif training.deepspeed_config:
            kwargs["deepspeed"] = training.deepspeed_config
            console.print(
                f"[green]✓ DeepSpeed config injected (GRPO): {training.deepspeed_config}[/green]"
            )

        if dist_info.is_distributed:
            console.print(
                f"[cyan]Distributed GRPO engaged: world_size={dist_info.world_size}[/cyan]"
            )

        # Optional optimizer override (e.g. "paged_adamw_8bit" — the QLoRA-paper
        # recipe: 8-bit states paged to CPU RAM to survive VRAM spikes).
        if training.optim:
            kwargs["optim"] = training.optim

        return TRLGRPOConfig(**kwargs)

    def setup_trainer(self) -> None:
        """Setup TRL GRPO trainer."""
        console.print("\n[bold cyan]=== Setting Up GRPO Trainer ===[/bold cyan]\n")

        grpo_args = self._build_grpo_config()
        reward_funcs = self._build_reward_funcs()

        console.print(f"[cyan]Reward functions: {self.config.reward_config.reward_funcs}[/cyan]")
        console.print(f"[cyan]Beta: {grpo_args.beta}[/cyan]")
        console.print(f"[cyan]Num generations: {grpo_args.num_generations}[/cyan]")
        console.print(
            f"[cyan]Effective batch size: "
            f"{self.config.training_config.effective_batch_size}[/cyan]\n"
        )

        self.trainer = TRLGRPOTrainer(
            model=self.model,
            reward_funcs=reward_funcs,
            args=grpo_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            processing_class=self.tokenizer,
            callbacks=[
                MemoryCallback(log_steps=self.config.training_config.logging_steps),
                MLflowTrainCallback(self._tracker),
            ],
        )

        console.print("[green]✓ GRPO Trainer configured[/green]\n")

    def train(self) -> Any:
        """Run GRPO training."""
        console.print("\n[bold green]=== Starting GRPO Training ===[/bold green]\n")

        # Setup W&B
        wandb_run = None
        if self.config.logging_config.use_wandb:
            from src.utils import setup_wandb

            wandb_run = setup_wandb(
                project=self.config.logging_config.wandb_project,
                config={
                    "model": self.config.model_config.__dict__,
                    "training": self.config.training_config.__dict__,
                    "lora": self.config.lora_config.__dict__,
                    "grpo": self.config.grpo_config.__dict__,
                    "reward": self.config.reward_config.__dict__,
                },
                entity=self.config.logging_config.wandb_entity,
                run_name=self.config.logging_config.wandb_run_name,
            )

        # MLflow run
        if self._tracker.active:
            self._tracker.start_run(
                run_name=self.config.logging_config.mlflow_run_name,
                config={
                    "model": self.config.model_config.__dict__,
                    "training": self.config.training_config.__dict__,
                    "lora": self.config.lora_config.__dict__,
                    "grpo": self.config.grpo_config.__dict__,
                    "reward": self.config.reward_config.__dict__,
                },
            )

        assert self.trainer is not None
        try:
            train_result = self.trainer.train()

            # Save final model
            console.print(
                f"\n[cyan]Saving model to: {self.config.training_config.output_dir}[/cyan]"
            )
            self.trainer.save_model()
            self.tokenizer.save_pretrained(self.config.training_config.output_dir)

            # Register to MLflow Model Registry
            register_trained_model(
                adapter_dir=self.config.training_config.output_dir,
                tracker=self._tracker,
                logging_config=self.config.logging_config,
                base_model_name=self.config.model_config.name,
                model_config=self.config.model_config,
            )

            console.print("\n[bold green]=== GRPO Training Complete! ===[/bold green]\n")
            return train_result

        except Exception as e:
            console.print(f"\n[red]✗ GRPO training failed: {e}[/red]\n")
            raise
        finally:
            if wandb_run is not None:
                wandb_run.finish()
            self._tracker.end_run()

    def evaluate(self) -> dict[str, float]:
        """Evaluate GRPO-trained model."""
        console.print("\n[bold cyan]=== Evaluating GRPO Model ===[/bold cyan]\n")
        assert self.trainer is not None
        metrics: dict[str, float] = self.trainer.evaluate()
        console.print("[green]Evaluation Results:[/green]")
        for key, value in metrics.items():
            console.print(f"  {key}: {value}")
        return metrics


def run_grpo_training(config: GRPOTrainingConfig) -> None:
    """Run complete GRPO training pipeline."""
    console.print("\n[bold magenta]=== GRPO Training ===[/bold magenta]\n")

    trainer = GRPOTrainer(config)
    trainer.prepare_model()
    trainer.prepare_data()
    trainer.setup_trainer()
    trainer.train()
    trainer.evaluate()
