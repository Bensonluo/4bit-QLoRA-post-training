"""DPO (Direct Preference Optimization) trainer implementation.

This module implements DPO training using TRL's DPOTrainer with
memory optimizations for 8GB VRAM.
"""

import os
from typing import Optional

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

from transformers.trainer import Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import DPOConfig as TRLDPOConfig, DPOTrainer as TRLDPOTrainer

from config.base import ModelConfig, TrainingConfig, LoRAConfig, DataConfig, LoggingConfig
from config.dpo import DPOConfig, ReferenceModelConfig, PreferenceDataConfig, DPOTrainingConfig
from src.models import load_model_and_tokenizer
from src.data.loaders import PreferenceDataset
from src.utils import set_seed, console, setup_logging


class DPOTrainer:
    """DPO trainer with memory optimizations.

    This class wraps TRL's DPOTrainer and handles:
    - Loading main and reference models
    - Preparing preference datasets
    - Training with DPO loss
    - Memory optimization for 8GB VRAM
    """

    def __init__(
        self,
        model_config: ModelConfig,
        training_config: TrainingConfig,
        lora_config: LoRAConfig,
        dpo_config: DPOConfig,
        data_config: PreferenceDataConfig,
        reference_config: ReferenceModelConfig,
        logging_config: LoggingConfig,
    ):
        """Initialize DPO trainer.

        Args:
            model_config: Main model configuration
            training_config: Training configuration
            lora_config: LoRA configuration
            dpo_config: DPO-specific configuration
            data_config: Preference dataset configuration
            reference_config: Reference model configuration
            logging_config: Logging configuration
        """
        self.model_config = model_config
        self.training_config = training_config
        self.lora_config = lora_config
        self.dpo_config = dpo_config
        self.data_config = data_config
        self.reference_config = reference_config
        self.logging_config = logging_config

        # Set random seed
        set_seed(training_config.seed)

        # Setup logging
        self.logger = setup_logging(
            log_file=os.path.join(training_config.output_dir, "dpo_training.log"),
            level=logging_config.console_level,
        )

        # Models and tokenizer (loaded later)
        self.model = None
        self.ref_model = None
        self.tokenizer = None
        self.trainer = None

    def prepare_models(self):
        """Load main model, reference model, and tokenizer."""
        console.print("\n[bold cyan]=== Preparing Models ===[/bold cyan]\n")

        # Load main model with LoRA
        console.print(f"[cyan]Loading main model: {self.model_config.name}[/cyan]")
        self.model, self.tokenizer = load_model_and_tokenizer(self.model_config)

        # Prepare for k-bit training
        self.model = prepare_model_for_kbit_training(self.model)

        # Apply LoRA
        console.print(f"[cyan]Applying LoRA (r={self.lora_config.r})...[/cyan]")
        lora_cfg = LoraConfig(
            r=self.lora_config.r,
            lora_alpha=self.lora_config.lora_alpha,
            lora_dropout=self.lora_config.lora_dropout,
            target_modules=self.lora_config.target_modules,
            bias=self.lora_config.bias,
            task_type=self.lora_config.task_type,
        )
        self.model = get_peft_model(self.model, lora_cfg)

        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        console.print(f"[green]✓ Main model ready ({trainable_params:,} trainable params, {trainable_params/total_params:.2%})[/green]")

        # Load reference model
        console.print(f"\n[cyan]Loading reference model: {self.reference_config.name}[/cyan]")
        self.ref_model, _ = load_model_and_tokenizer(self.reference_config)

        # Freeze reference model
        console.print("[cyan]Freezing reference model...[/cyan]")
        for param in self.ref_model.parameters():
            param.requires_grad = False

        ref_params = sum(p.numel() for p in self.ref_model.parameters())
        console.print(f"[green]✓ Reference model ready ({ref_params:,} params, frozen)[/green]")

        console.print()

    def prepare_data(self):
        """Load and prepare preference dataset."""
        console.print("[bold cyan]=== Preparing Preference Data ===[/bold cyan]\n")

        # Load dataset
        dataset = PreferenceDataset(
            data_path=self.data_config.dataset_name,
            max_samples=self.data_config.max_samples,
        )
        dataset.load()

        # Filter for finance if enabled
        if self.data_config.auto_filter:
            console.print("[yellow]Filtering for finance-related content...[/yellow]")
            dataset = self._filter_finance(dataset)
            console.print(f"[green]✓ Filtered to {len(dataset)} finance samples[/green]\n")

        # Split
        train_dataset, eval_dataset = dataset.split_dataset(
            validation_split=self.data_config.validation_split,
            seed=self.training_config.seed,
        )

        console.print(f"[green]✓ Train samples: {len(train_dataset)}[/green]")
        console.print(f"[green]✓ Validation samples: {len(eval_dataset)}[/green]\n")

        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset

    def _filter_finance(self, dataset):
        """Filter dataset for finance content."""
        finance_keywords = [
            "stock", "investment", "finance", "financial", "trading",
            "portfolio", "dividend", "market", "business",
            "money", "capital", "asset", "fund", "equity", "bond",
            "currency", "crypto", "earnings", "revenue", "profit",
        ]

        def is_finance_related(example):
            text = (
                example.get("prompt", "") +
                " " +
                example.get("chosen", "") +
                " " +
                example.get("rejected", "")
            ).lower()
            return any(keyword in text for keyword in finance_keywords)

        return dataset.filter(is_finance_related)

    def setup_trainer(self):
        """Setup TRL DPO trainer."""
        console.print("[bold cyan]=== Setting Up DPO Trainer ===[/bold cyan]\n")

        # TRL DPO configuration
        trl_dpo_config = TRLDPOConfig(
            beta=self.dpo_config.beta,
            max_length=self.dpo_config.max_length,
            max_prompt_length=self.dpo_config.max_prompt_length,
            max_target_length=self.dpo_config.max_target_length,
            reference_model_free=self.dpo_config.reference_model_free,
            loss_type=self.dpo_config.loss_type,
            label_smoothing=self.dpo_config.label_smoothing,
            padding_value=self.dpo_config.padding_value,
        )

        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.training_config.output_dir,
            num_train_epochs=self.training_config.num_epochs,
            per_device_train_batch_size=self.training_config.batch_size,
            per_device_eval_batch_size=self.training_config.batch_size,
            gradient_accumulation_steps=self.training_config.gradient_accumulation_steps,
            learning_rate=self.training_config.learning_rate,
            warmup_ratio=self.training_config.warmup_ratio,
            lr_scheduler_type=self.training_config.lr_scheduler_type,
            logging_steps=self.training_config.logging_steps,
            save_steps=self.training_config.save_steps,
            eval_steps=self.training_config.eval_steps,
            save_total_limit=self.training_config.save_total_limit,
            gradient_checkpointing=self.training_config.gradient_checkpointing,
            fp16=self.training_config.fp16,
            bf16=self.training_config.bf16,
            seed=self.training_config.seed,
            report_to=["tensorboard"] if self.logging_config.use_tensorboard else [],
            logging_dir=self.logging_config.log_dir if self.logging_config.use_tensorboard else None,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            remove_unused_columns=False,
        )

        console.print("[green]✓ DPO Trainer configured[/green]")
        console.print(f"  Beta: {trl_dpo_config.beta}")
        console.print(f"  Loss type: {trl_dpo_config.loss_type}")
        console.print(f"  Max length: {trl_dpo_config.max_length}")
        console.print(f"  Effective batch size: {self.training_config.effective_batch_size}")
        console.print()

        # Create DPO trainer
        self.trainer = TRLDPOTrainer(
            model=self.model,
            ref_model=self.ref_model,
            args=training_args,
            beta=trl_dpo_config.beta,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            tokenizer=self.tokenizer,
            **trl_dpo_config.__dict__,
        )

    def train(self):
        """Run DPO training."""
        console.print("\n[bold green]=== Starting DPO Training ===[/bold green]\n")

        # Setup W&B
        if self.logging_config.use_wandb:
            import wandb
            wandb.init(
                project=self.logging_config.wandb_project,
                name=self.logging_config.wandb_run_name,
                config={
                    "model": self.model_config.name,
                    "dpo_beta": self.dpo_config.beta,
                    "dataset": self.data_config.dataset_name,
                    "max_samples": self.data_config.max_samples,
                },
            )

        try:
            self.trainer.train()

            console.print("\n[bold green]✅ DPO Training Complete![/bold green]")
            console.print(f"[cyan]Model saved to: {self.training_config.output_dir}[/cyan]\n")

        except Exception as e:
            console.print(f"\n[red]DPO training failed: {e}[/red]\n")
            raise
        finally:
            if self.logging_config.use_wandb:
                import wandb
                wandb.finish()

    def evaluate(self):
        """Evaluate DPO-trained model."""
        console.print("\n[bold cyan]=== Evaluating DPO Model ===[/bold cyan]\n")

        metrics = self.trainer.evaluate()

        console.print("[green]Evaluation Results:[/green]")
        for key, value in metrics.items():
            console.print(f"  {key}: {value}")

        return metrics


def run_dpo_training(
    model_config: ModelConfig,
    training_config: TrainingConfig,
    lora_config: LoRAConfig,
    dpo_config: DPOConfig,
    data_config: PreferenceDataConfig,
    reference_config: ReferenceModelConfig,
    logging_config: LoggingConfig,
) -> None:
    """Run complete DPO training pipeline.

    Args:
        model_config: Main model configuration
        training_config: Training configuration
        lora_config: LoRA configuration
        dpo_config: DPO configuration
        data_config: Preference data configuration
        reference_config: Reference model configuration
        logging_config: Logging configuration
    """
    console.print("\n[bold magenta]=== DPO Training ===[/bold magenta]\n")

    trainer = DPOTrainer(
        model_config=model_config,
        training_config=training_config,
        lora_config=lora_config,
        dpo_config=dpo_config,
        data_config=data_config,
        reference_config=reference_config,
        logging_config=logging_config,
    )

    trainer.prepare_models()
    trainer.prepare_data()
    trainer.setup_trainer()
    trainer.train()
    trainer.evaluate()


if __name__ == "__main__":
    # Test with minimal config
    print("Testing DPO trainer...")
    print("This module is typically used via scripts/train_dpo.py")
