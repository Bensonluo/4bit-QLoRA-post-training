"""Supervised Fine-Tuning (SFT) trainer with QLoRA."""

import os
from pathlib import Path
from typing import Optional

import torch
from transformers import (
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from transformers.trainer_callback import TrainerCallback
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

from config.base import ModelConfig, TrainingConfig, LoRAConfig, DataConfig, LoggingConfig
from src.models import load_model_and_tokenizer, print_model_info
from src.data import AlpacaDataset, FinanceDataset
from src.utils import set_seed, setup_logging, setup_wandb, setup_tensorboard, log_metrics, log_gpu_memory, console
from src.utils.platform_utils import get_platform
from src.tracking import get_tracker, MLflowTrainCallback


class MemoryCallback(TrainerCallback):
    """Callback to log GPU memory usage during training."""

    def __init__(self, log_steps: int = 100):
        """Initialize memory callback.

        Args:
            log_steps: Log memory every N steps
        """
        super().__init__()
        self.log_steps = log_steps

    def on_step_end(self, args, state, control, **kwargs):
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
    ):
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
        self.model = None
        self.tokenizer = None
        self.trainer = None

        # MLflow tracker (no-op if use_mlflow=False)
        self._tracker = get_tracker(logging_config)

    def _get_report_to(self) -> list[str]:
        """Determine reporting backends based on logging config."""
        backends = []
        if self.logging_config.use_wandb:
            backends.append("wandb")
        if self.logging_config.use_tensorboard:
            backends.append("tensorboard")
        if self.logging_config.use_mlflow:
            backends.append("mlflow")
        return backends if backends else ["none"]

    def prepare_model(self):
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
        lora_cfg = LoraConfig(
            r=self.lora_config.r,
            lora_alpha=self.lora_config.lora_alpha,
            lora_dropout=self.lora_config.lora_dropout,
            target_modules=self.lora_config.target_modules,
            bias=self.lora_config.bias,
            task_type=self.lora_config.task_type,
        )

        # Apply LoRA
        console.print(f"[cyan]Applying LoRA (r={self.lora_config.r}, alpha={self.lora_config.lora_alpha})...[/cyan]")
        self.model = get_peft_model(self.model, lora_cfg)

        # Print trainable parameters
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_percent = 100 * trainable_params / total_params

        console.print(f"[green]✓ Trainable parameters: {trainable_params:,} ({trainable_percent:.2f}%)[/green]")
        console.print(f"[green]✓ Total parameters: {total_params:,}[/green]")

    def prepare_data(self):
        """Load and prepare dataset."""
        console.print("\n[bold cyan]=== Preparing Data ===[/bold cyan]\n")

        # Choose dataset type based on config
        if "finance" in self.data_config.dataset_name.lower() or self.data_config.dataset_name == "yahma/alpaca-cleaned":
            dataset = FinanceDataset(
                data_path=self.data_config.dataset_name,
                max_samples=self.data_config.max_samples,
            )
        elif "medical_entity" in self.data_config.dataset_name.lower():
            from src.data.medical_dataset import MedicalEntityDataset
            dataset = MedicalEntityDataset(
                data_path=self.data_config.dataset_name,
                max_samples=self.data_config.max_samples,
            )
        else:
            dataset = AlpacaDataset(
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
        _val_dataset_obj = None
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

    def setup_trainer(self):
        """Setup Hugging Face Trainer."""
        console.print("\n[bold cyan]=== Setting Up Trainer ===[/bold cyan]\n")

        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.training_config.output_dir,
            num_train_epochs=self.training_config.num_epochs,
            per_device_train_batch_size=self.training_config.batch_size,
            per_device_eval_batch_size=self.training_config.batch_size,
            gradient_accumulation_steps=self.training_config.gradient_accumulation_steps,
            learning_rate=self.training_config.learning_rate,
            weight_decay=self.training_config.weight_decay,
            warmup_ratio=self.training_config.warmup_ratio,
            lr_scheduler_type=self.training_config.lr_scheduler_type,
            logging_steps=self.training_config.logging_steps,
            save_steps=self.training_config.save_steps,
            eval_steps=self.training_config.eval_steps,
            save_total_limit=self.training_config.save_total_limit,
            gradient_checkpointing=self.training_config.gradient_checkpointing,
            fp16=self.training_config.fp16,
            bf16=self.training_config.bf16,
            max_grad_norm=self.training_config.max_grad_norm,
            report_to=self._get_report_to(),
            run_name=self.logging_config.wandb_run_name,
            logging_dir=self.logging_config.log_dir if self.logging_config.use_tensorboard else None,
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
            callbacks=[MemoryCallback(log_steps=self.training_config.logging_steps), MLflowTrainCallback(self._tracker)],
        )

        console.print("[green]✓ Trainer configured[/green]")
        console.print(f"  Effective batch size: {self.training_config.effective_batch_size}")
        console.print(f"  Training steps: {len(self.train_dataset) // self.training_config.effective_batch_size * self.training_config.num_epochs}\n")

    def train(self, resume_from_checkpoint: str | None = None):
        """Run training."""
        if resume_from_checkpoint:
            console.print(f"\n[bold green]=== Resuming Training from {resume_from_checkpoint} ===[/bold green]\n")
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
            train_result = self.trainer.train(resume_from_checkpoint=resume_from_checkpoint)

            # Save final model
            console.print(f"\n[cyan]Saving model to: {self.training_config.output_dir}[/cyan]")
            self.trainer.save_model()
            self.tokenizer.save_pretrained(self.training_config.output_dir)

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

    def evaluate(self):
        """Evaluate model."""
        console.print("\n[bold cyan]=== Evaluating Model ===[/bold cyan]\n")

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
) -> None:
    """Run complete SFT training pipeline.

    Args:
        model_config: Model configuration
        training_config: Training configuration
        lora_config: LoRA configuration
        data_config: Data configuration
        logging_config: Logging configuration
        resume_from_checkpoint: Optional path to checkpoint for resuming
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
