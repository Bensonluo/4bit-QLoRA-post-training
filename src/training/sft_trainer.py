"""Supervised Fine-Tuning (SFT) trainer with QLoRA."""

import os
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from peft.tuners.lora.loraga import preprocess_loraga
from transformers import (
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
from transformers.trainer_callback import TrainerCallback

from config.base import DataConfig, LoggingConfig, LoRAConfig, ModelConfig, TrainingConfig
from src.data import AlpacaDataset, BaseDataset, FinanceDataset
from src.models import load_model_and_tokenizer
from src.tracking import MLflowTrainCallback, get_tracker, register_trained_model
from src.training.callbacks import MFUCallback
from src.training.distributed import get_distributed_info
from src.utils import (
    console,
    log_gpu_memory,
    log_metrics,
    set_seed,
    setup_logging,
    setup_tensorboard,
    setup_wandb,
)
from src.utils.platform_utils import get_platform

# peft exports LoraGAConfig at the top level only in recent versions — import
# from its defining module so this works across the supported peft range.
from peft.tuners.lora.config import LoraGAConfig  # isort: skip


def _dataset_class_for(dataset_name: str) -> type[BaseDataset]:
    """Select the dataset class for a dataset name.

    Shared by prepare_data (full training set) and LoRA-GA calibration
    (a small slice) so both see the exact same data-formatting path.
    """
    if "finance" in dataset_name.lower() or dataset_name == "yahma/alpaca-cleaned":
        return FinanceDataset
    if "medical_entity" in dataset_name.lower():
        from src.data.medical_dataset import MedicalEntityDataset

        return MedicalEntityDataset
    return AlpacaDataset


class MemoryCallback(TrainerCallback):
    """Callback to log GPU memory usage during training."""

    def __init__(self, log_steps: int = 100) -> None:
        """Initialize memory callback.

        Args:
            log_steps: Log memory every N steps
        """
        super().__init__()
        self.log_steps = log_steps

    def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
        """Log memory at end of step."""
        if state.global_step % self.log_steps == 0:
            log_gpu_memory(state.global_step, wandb_run=None)
        return control


class SFTTrainer:
    """Supervised Fine-Tuning trainer with QLoRA."""

    def __init__(
        self,
        model_config: ModelConfig,
        training_config: TrainingConfig,
        lora_config: LoRAConfig,
        data_config: DataConfig,
        logging_config: LoggingConfig,
    ) -> None:
        """Initialize SFT trainer.

        Args:
            model_config: Model configuration
            training_config: Training configuration
            lora_config: LoRA configuration
            data_config: Data configuration
            logging_config: Logging configuration
        """
        self.model_config = model_config
        self.training_config = training_config
        self.lora_config = lora_config
        self.data_config = data_config
        self.logging_config = logging_config

        # Set random seed
        set_seed(training_config.seed)

        # Setup logging
        self.logger = setup_logging(
            log_file=os.path.join(training_config.output_dir, "training.log"),
            level=logging_config.console_level,
        )

        # Model and tokenizer (loaded later)
        self.model: Any = None
        self.tokenizer: Any = None
        self.trainer: Trainer | None = None

        # MLflow tracker (no-op if use_mlflow=False)
        self._tracker = get_tracker(logging_config)

    def _get_report_to(self) -> list[str]:
        """Determine reporting backends based on logging config.

        NOTE: MLflow is intentionally NOT added here. We use our own
        `MLflowTrainCallback` (mounted in setup_trainer) for finer control over
        what gets logged. Adding "mlflow" here would activate HF's built-in
        MLflowCallback too, causing double writes.
        """
        backends = []
        if self.logging_config.use_wandb:
            backends.append("wandb")
        if self.logging_config.use_tensorboard:
            backends.append("tensorboard")
        return backends if backends else ["none"]

    def prepare_model(self) -> None:
        """Load model and apply LoRA adapters."""
        console.print("\n[bold cyan]=== Preparing Model ===[/bold cyan]\n")

        # Load model and tokenizer
        self.model, self.tokenizer = load_model_and_tokenizer(self.model_config)

        # Prepare model for k-bit training (only needed with bitsandbytes quantization)
        platform_info = get_platform()
        if platform_info.is_cuda and self.model_config.quantization_bits in (4, 8):
            console.print("[cyan]Preparing model for k-bit training...[/cyan]")
            self.model = prepare_model_for_kbit_training(self.model)
        else:
            console.print("[cyan]Skipping k-bit preparation (not needed on this platform)[/cyan]")

        # Get LoRA configuration
        lora_kwargs: dict[str, Any] = dict(
            r=self.lora_config.r,
            lora_alpha=self.lora_config.lora_alpha,
            lora_dropout=self.lora_config.lora_dropout,
            target_modules=self.lora_config.target_modules,
            bias=self.lora_config.bias,
            task_type=self.lora_config.task_type,
            use_dora=self.lora_config.use_dora,
            use_rslora=self.lora_config.use_rslora,
            init_lora_weights=self.lora_config.init_lora_weights,
        )
        if self.lora_config.init_lora_weights == "loftq":
            # peft hard-errors on loftq without the dict — inject it here.
            lora_kwargs["loftq_config"] = {
                "loftq_bits": self.lora_config.loftq_bits,
                "loftq_iter": self.lora_config.loftq_iter,
            }
        # Per-module rank/alpha overrides and module exclusions (peft
        # regex-keyed patterns; None = peft defaults, no-op).
        if self.lora_config.rank_pattern is not None:
            lora_kwargs["rank_pattern"] = self.lora_config.rank_pattern
        if self.lora_config.alpha_pattern is not None:
            lora_kwargs["alpha_pattern"] = self.lora_config.alpha_pattern
        if self.lora_config.exclude_modules is not None:
            lora_kwargs["exclude_modules"] = self.lora_config.exclude_modules
        if self.lora_config.init_lora_weights == "lora_ga":
            lora_kwargs["lora_ga_config"] = LoraGAConfig(
                direction=self.lora_config.lora_ga_direction,
                scale=self.lora_config.lora_ga_scale,
                stable_gamma=self.lora_config.lora_ga_stable_gamma,
            )
        lora_cfg = LoraConfig(**lora_kwargs)

        # LoRA-GA: peft consumes the gradient estimate ONLY if preprocess_loraga
        # ran first — otherwise it silently falls back to gaussian init. Run it
        # on a calibration slice of the (same) training data before get_peft_model.
        if self.lora_config.init_lora_weights == "lora_ga":
            if self.model_config.quantization_bits in (4, 8):
                # Catches configs mutated after SFTConfig construction (the
                # composite config validates this at build time too).
                raise ValueError(
                    "init_lora_weights='lora_ga' (LoRA-GA) requires a full-precision "
                    f"base — quantization_bits={self.model_config.quantization_bits} "
                    "is rejected by peft's gradient estimation. Use "
                    "quantization_bits=None or a different init strategy."
                )
            console.print(
                "[cyan]LoRA-GA: estimating full-finetuning gradient on calibration "
                f"data ({self.lora_config.lora_ga_calibration_batches} batches)...[/cyan]"
            )
            preprocess_loraga(
                self.model,
                lora_cfg,
                self._build_lora_ga_train_step(),
                cache_file=self.lora_config.lora_ga_cache_file,
            )

        # Apply LoRA
        console.print(
            f"[cyan]Applying LoRA (r={self.lora_config.r}, alpha={self.lora_config.lora_alpha})...[/cyan]"
        )
        self.model = get_peft_model(self.model, lora_cfg)

        # Print trainable parameters
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_percent = 100 * trainable_params / total_params

        console.print(
            f"[green]✓ Trainable parameters: {trainable_params:,} ({trainable_percent:.2f}%)[/green]"
        )
        console.print(f"[green]✓ Total parameters: {total_params:,}[/green]")

    def _build_lora_ga_train_step(self) -> Callable[[], None]:
        """Build the calibration callback peft's preprocess_loraga consumes.

        Loads a small slice of the training data via the same dataset class
        prepare_data would select, tokenizes it with the same tokenizer and
        max_length, and returns a closure that runs forward+backward over
        the calibration batches. peft accumulates the target-module weight
        gradients and initializes A/B from their SVD (LoRA-GA,
        arXiv:2407.05000).
        """
        dataset_cls = _dataset_class_for(self.data_config.dataset_name)
        calib_dataset = dataset_cls(
            data_path=self.data_config.dataset_name,
            max_samples=self.lora_config.lora_ga_calibration_batches
            * self.training_config.batch_size,
        )
        calib_dataset.load()
        tokenized = calib_dataset.format_for_training(
            self.tokenizer,
            max_length=self.model_config.max_length,
        )

        # tokenize_function emits input_ids/attention_mask (no labels) — pass
        # the input_ids as labels so the model computes shifted CE loss itself.
        batch_size = max(1, self.training_config.batch_size)
        batches: list[dict[str, Any]] = []
        for start in range(0, len(tokenized), batch_size):
            rows = tokenized[start : start + batch_size]
            batches.append(
                {
                    key: torch.tensor([row[key] for row in rows], dtype=torch.long)
                    for key in ("input_ids", "attention_mask")
                    if key in rows[0]
                }
            )

        model = self.model
        device = next(model.parameters()).device

        def train_step() -> None:
            for batch in batches:
                moved = {key: tensor.to(device) for key, tensor in batch.items()}
                loss = model(**moved, labels=moved["input_ids"]).loss
                loss.backward()

        return train_step

    def prepare_data(self) -> None:
        """Load and prepare dataset."""
        console.print("\n[bold cyan]=== Preparing Data ===[/bold cyan]\n")

        # Choose dataset type based on config (shared with LoRA-GA calibration)
        dataset: BaseDataset
        dataset_cls = _dataset_class_for(self.data_config.dataset_name)
        dataset = dataset_cls(
            data_path=self.data_config.dataset_name,
            max_samples=self.data_config.max_samples,
        )

        # Load dataset
        dataset.load()

        # Split into train/validation
        self.train_dataset, self.eval_dataset = dataset.split_dataset(
            validation_split=self.data_config.validation_split,
            seed=self.training_config.seed,
        )

        console.print(f"[green]✓ Train samples: {len(self.train_dataset):,}[/green]")

        # Load separate validation file if no split was done
        _val_dataset_obj: BaseDataset | None = None
        if self.eval_dataset is None and self.data_config.validation_file:
            val_path = Path(self.data_config.validation_file)
            if val_path.exists():
                val_cls = type(dataset)
                _val_dataset_obj = val_cls(
                    data_path=self.data_config.validation_file,
                    max_samples=self.data_config.max_samples,
                )
                _val_dataset_obj.load()
                self.eval_dataset = _val_dataset_obj.dataset
                console.print(f"[green]✓ Validation samples: {len(self.eval_dataset):,}[/green]\n")
            else:
                console.print("[yellow]⚠ No validation data found[/yellow]\n")
        else:
            val_count = len(self.eval_dataset) if self.eval_dataset else 0
            console.print(f"[green]✓ Validation samples: {val_count:,}[/green]\n")

        # Format datasets for training
        self.train_dataset = dataset.format_for_training(
            self.tokenizer,
            max_length=self.model_config.max_length,
        )
        if self.eval_dataset is not None:
            if _val_dataset_obj is not None:
                self.eval_dataset = _val_dataset_obj.format_for_training(
                    self.tokenizer,
                    max_length=self.model_config.max_length,
                )
            else:
                self.eval_dataset = dataset.format_for_training(
                    self.tokenizer,
                    max_length=self.model_config.max_length,
                )

    def setup_trainer(self) -> None:
        """Setup Hugging Face Trainer."""
        console.print("\n[bold cyan]=== Setting Up Trainer ===[/bold cyan]\n")

        # Detect distributed context (torchrun/accelerate set LOCAL_RANK/WORLD_SIZE).
        dist_info = get_distributed_info()

        # torch.compile cannot trace bitsandbytes 4/8-bit custom autograd ops —
        # warn loudly instead of letting the run die deep inside Dynamo.
        if self.training_config.torch_compile and self.model_config.quantization_bits in (4, 8):
            warnings.warn(
                "torch_compile=True is incompatible with bitsandbytes "
                f"{self.model_config.quantization_bits}-bit quantization (Dynamo cannot "
                "trace Params4bit/Linear4bit). Use the bf16 full-precision path "
                "(quantization_bits=None) or disable torch_compile.",
                stacklevel=2,
            )

        # Training arguments — build as a dict first so we can conditionally inject DeepSpeed.
        training_kwargs: dict[str, Any] = dict(
            output_dir=self.training_config.output_dir,
            # Opt-in Triton fused kernels (RMSNorm/RoPE/SwiGLU/fused CE) —
            # ~20% throughput, up to 60% activation-memory reduction on
            # supported CUDA models. Requires the optional [liger] extra.
            use_liger_kernel=self.training_config.use_liger_kernel,
            # NEFTune noisy embeddings (α/√(L·d)) — SFT-only regularizer;
            # HF Trainer auto-disables the noise at eval. None = off.
            neftune_noise_alpha=self.training_config.neftune_noise_alpha,
            # Opt-in torch.compile — bf16 full-precision path only (see
            # TrainingConfig.torch_compile docstring).
            torch_compile=self.training_config.torch_compile,
            num_train_epochs=self.training_config.num_epochs,
            per_device_train_batch_size=self.training_config.batch_size,
            per_device_eval_batch_size=self.training_config.batch_size,
            gradient_accumulation_steps=self.training_config.gradient_accumulation_steps,
            learning_rate=self.training_config.learning_rate,
            weight_decay=self.training_config.weight_decay,
            # transformers >= 5.x: warmup_ratio merged into warmup_steps
            # (a float value keeps ratio semantics).
            warmup_steps=self.training_config.warmup_ratio,
            lr_scheduler_type=self.training_config.lr_scheduler_type,
            logging_steps=self.training_config.logging_steps,
            save_steps=self.training_config.save_steps,
            eval_steps=self.training_config.eval_steps,
            save_total_limit=self.training_config.save_total_limit,
            gradient_checkpointing=self.training_config.gradient_checkpointing,
            # Non-reentrant checkpointing — PyTorch-recommended and the
            # reliable path for frozen-base (LoRA/QLoRA) fine-tunes.
            gradient_checkpointing_kwargs=(
                {"use_reentrant": self.training_config.gradient_checkpointing_use_reentrant}
                if self.training_config.gradient_checkpointing
                else None
            ),
            # Periodic torch.cuda.empty_cache() — opt-in fragmentation relief
            # for 8 GB cards with variable-length batches (None = off).
            torch_empty_cache_steps=self.training_config.torch_empty_cache_steps,
            # On OOM, auto-restart with halved batch size (ZeRO-3 excluded
            # at config validation; see TrainingConfig).
            auto_find_batch_size=self.training_config.auto_find_batch_size,
            # Length-grouped sampling — padding-waste reduction for
            # variable-length instruction data (opt-in).
            train_sampling_strategy=self.training_config.train_sampling_strategy,
            fp16=self.training_config.fp16,
            bf16=self.training_config.bf16,
            max_grad_norm=self.training_config.max_grad_norm,
            report_to=self._get_report_to(),
            run_name=self.logging_config.wandb_run_name,
            logging_dir=self.logging_config.log_dir
            if self.logging_config.use_tensorboard
            else None,
            save_strategy="steps",
            eval_strategy="steps" if self.eval_dataset is not None else "no",
            load_best_model_at_end=self.eval_dataset is not None,
            metric_for_best_model="eval_loss" if self.eval_dataset is not None else None,
            greater_is_better=False if self.eval_dataset is not None else None,
            seed=self.training_config.seed,
            data_seed=self.training_config.seed,
            ddp_find_unused_parameters=False,
            dataloader_num_workers=0,
            dataloader_pin_memory=False,
        )

        # Inject distributed strategy: FSDP (PyTorch-native, default) or DeepSpeed.
        # HF Trainer natively understands both `fsdp=`/`fsdp_config=` and `deepspeed=`.
        # Exactly one is set — TrainingConfig.__post_init__ enforces mutual exclusion.
        if self.training_config.fsdp:
            training_kwargs["fsdp"] = self.training_config.fsdp
            if self.training_config.fsdp_config:
                training_kwargs["fsdp_config"] = self.training_config.fsdp_config
            console.print(f"[green]✓ FSDP enabled: {self.training_config.fsdp}[/green]")
        elif self.training_config.deepspeed_config:
            training_kwargs["deepspeed"] = self.training_config.deepspeed_config
            console.print(
                f"[green]✓ DeepSpeed config injected: {self.training_config.deepspeed_config}[/green]"
            )

        if dist_info.is_distributed:
            strategy = (
                self.training_config.fsdp
                or os.path.basename(self.training_config.deepspeed_config or "")
                or "DDP"
            )
            console.print(
                f"[cyan]Distributed training engaged: world_size={dist_info.world_size}, "
                f"strategy={strategy}[/cyan]"
            )

        # Optional optimizer override (e.g. "paged_adamw_8bit" — the QLoRA-paper
        # recipe: 8-bit states paged to CPU RAM to survive VRAM spikes).
        if self.training_config.optim:
            training_kwargs["optim"] = self.training_config.optim

        training_args = TrainingArguments(**training_kwargs)

        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,  # Causal LM, not masked LM
            pad_to_multiple_of=8,
        )

        # Create trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            data_collator=data_collator,
            callbacks=[
                MemoryCallback(log_steps=self.training_config.logging_steps),
                MLflowTrainCallback(self._tracker),
                MFUCallback(
                    model_config=getattr(self.model, "config", None),
                    seq_len=self.model_config.max_length,
                    tokens_per_step=(
                        self.training_config.batch_size
                        * self.training_config.gradient_accumulation_steps
                        * dist_info.world_size
                        * self.model_config.max_length
                    ),
                    world_size=dist_info.world_size,
                    peak_flops_per_device=self.training_config.peak_flops_per_device,
                ),
            ],
        )

        console.print("[green]✓ Trainer configured[/green]")
        console.print(f"  Effective batch size: {self.training_config.effective_batch_size}")
        console.print(
            f"  Training steps: {len(self.train_dataset) // self.training_config.effective_batch_size * self.training_config.num_epochs}\n"
        )

    def train(self, resume_from_checkpoint: str | None = None) -> Any:
        """Run training."""
        if resume_from_checkpoint:
            console.print(
                f"\n[bold green]=== Resuming Training from {resume_from_checkpoint} ===[/bold green]\n"
            )
        else:
            console.print("\n[bold green]=== Starting Training ===[/bold green]\n")

        # Setup W&B
        if self.logging_config.use_wandb:
            wandb_run = setup_wandb(
                project=self.logging_config.wandb_project,
                config={
                    "model": self.model_config.__dict__,
                    "training": self.training_config.__dict__,
                    "lora": self.lora_config.__dict__,
                    "data": self.data_config.__dict__,
                },
                entity=self.logging_config.wandb_entity,
                run_name=self.logging_config.wandb_run_name,
            )
        else:
            wandb_run = None

        # Setup TensorBoard
        setup_tensorboard(
            log_dir=self.logging_config.log_dir,
            enabled=self.logging_config.use_tensorboard,
        )

        # MLflow run
        if self._tracker.active:
            self._tracker.start_run(
                run_name=self.logging_config.mlflow_run_name,
                config={
                    "model": self.model_config.__dict__,
                    "training": self.training_config.__dict__,
                    "lora": self.lora_config.__dict__,
                    "data": self.data_config.__dict__,
                },
            )

        # Train
        try:
            assert self.trainer is not None
            train_result = self.trainer.train(resume_from_checkpoint=resume_from_checkpoint)

            # Save final model
            console.print(f"\n[cyan]Saving model to: {self.training_config.output_dir}[/cyan]")
            self.trainer.save_model()
            self.tokenizer.save_pretrained(self.training_config.output_dir)

            # Register to MLflow Model Registry (no-op unless register_model=True).
            # This closes the lineage loop: model version ← run ← params + metrics.
            register_trained_model(
                adapter_dir=self.training_config.output_dir,
                tracker=self._tracker,
                logging_config=self.logging_config,
                base_model_name=self.model_config.name,
                model_config=self.model_config,
            )

            # Log final metrics
            metrics = train_result.metrics
            log_metrics(metrics, prefix="train_", wandb_run=wandb_run)

            console.print("\n[bold green]=== Training Complete! ===[/bold green]\n")

            return train_result

        except Exception as e:
            console.print(f"\n[red]✗ Training failed: {e}[/red]\n")
            raise

        finally:
            if wandb_run is not None:
                wandb_run.finish()
            self._tracker.end_run()

    def evaluate(self) -> dict[str, float]:
        """Evaluate model."""
        console.print("\n[bold cyan]=== Evaluating Model ===[/bold cyan]\n")

        assert self.trainer is not None
        metrics = self.trainer.evaluate()

        console.print("[green]Evaluation Results:[/green]")
        for key, value in metrics.items():
            console.print(f"  {key}: {value:.4f}")

        return metrics


def run_sft_training(
    model_config: ModelConfig,
    training_config: TrainingConfig,
    lora_config: LoRAConfig,
    data_config: DataConfig,
    logging_config: LoggingConfig,
    resume_from_checkpoint: str | None = None,
) -> SFTTrainer:
    """Run complete SFT training pipeline.

    Args:
        model_config: Model configuration
        training_config: Training configuration
        lora_config: LoRA configuration
        data_config: Data configuration
        logging_config: Logging configuration
        resume_from_checkpoint: Optional path to checkpoint for resuming

    Returns:
        The SFTTrainer after training — callers can post-process ``trainer.model``
        (e.g. write a DCP checkpoint under FSDP) without re-instantiating it.
    """
    # Create trainer
    trainer = SFTTrainer(
        model_config=model_config,
        training_config=training_config,
        lora_config=lora_config,
        data_config=data_config,
        logging_config=logging_config,
    )

    # Prepare model
    trainer.prepare_model()

    # Prepare data
    trainer.prepare_data()

    # Setup trainer
    trainer.setup_trainer()

    # Train (Trainer runs final eval automatically if eval_dataset exists)
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    return trainer


if __name__ == "__main__":
    # Test with minimal config
    from config.sft import FINANCE_SFT_CONFIG

    # Use smaller config for testing
    test_config = FINANCE_SFT_CONFIG
    test_config.data.max_samples = 100
    test_config.training.num_epochs = 1
    test_config.training.output_dir = "./outputs/test_sft"

    console.print("[yellow]Running SFT training test...[/yellow]")
    console.print("[yellow]This will download Qwen 1.5B and train on 100 samples[/yellow]\n")

    try:
        run_sft_training(
            model_config=test_config.model,
            training_config=test_config.training,
            lora_config=test_config.lora,
            data_config=test_config.data,
            logging_config=test_config.logging,
        )
        console.print("\n[green]✓ SFT training test successful![/green]")
    except Exception as e:
        console.print(f"\n[red]✗ Test failed: {e}[/red]")
        import traceback

        traceback.print_exc()
