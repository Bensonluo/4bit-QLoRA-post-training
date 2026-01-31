#!/usr/bin/env python3
"""CLI script for SFT training.

Usage:
    python scripts/train_sft.py --model-name Qwen/Qwen2.5-1.5B-Instruct --dataset finance-alpaca
"""

import os
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel

# Add parent directory to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.base import ModelConfig, TrainingConfig, LoRAConfig, DataConfig, LoggingConfig
from config.sft import FINANCE_SFT_CONFIG
from src.training import run_sft_training

app = typer.Typer(
    name="train-sft",
    help="Train LLM with Supervised Fine-Tuning (SFT) using QLoRA",
    add_completion=False,
)

console = Console()


@app.command()
def main(
    # Model arguments
    model_name: str = typer.Option(
        "Qwen/Qwen2.5-1.5B-Instruct",
        "--model-name",
        "-m",
        help="Hugging Face model name or path",
    ),
    quantization_bits: int = typer.Option(
        4,
        "--quantization-bits",
        "-q",
        help="Quantization bits (4 or 8)",
    ),
    max_length: int = typer.Option(
        1024,
        "--max-length",
        help="Maximum sequence length",
    ),

    # Data arguments
    dataset: str = typer.Option(
        "yahma/alpaca-cleaned",
        "--dataset",
        "-d",
        help="Dataset name or path",
    ),
    max_samples: Optional[int] = typer.Option(
        None,
        "--max-samples",
        "-n",
        help="Maximum number of samples (None for all)",
    ),
    validation_split: float = typer.Option(
        0.1,
        "--validation-split",
        help="Fraction of data for validation",
    ),

    # Training arguments
    output_dir: str = typer.Option(
        "./outputs/sft",
        "--output-dir",
        "-o",
        help="Output directory for checkpoints",
    ),
    num_epochs: int = typer.Option(
        3,
        "--epochs",
        "-e",
        help="Number of training epochs",
    ),
    batch_size: int = typer.Option(
        1,
        "--batch-size",
        "-b",
        help="Training batch size",
    ),
    gradient_accumulation_steps: int = typer.Option(
        8,
        "--gradient-accumulation-steps",
        "-g",
        help="Gradient accumulation steps",
    ),
    learning_rate: float = typer.Option(
        2e-4,
        "--learning-rate",
        "--lr",
        help="Learning rate",
    ),
    warmup_ratio: float = typer.Option(
        0.03,
        "--warmup-ratio",
        help="Warmup ratio",
    ),

    # LoRA arguments
    lora_r: int = typer.Option(
        16,
        "--lora-r",
        help="LoRA rank",
    ),
    lora_alpha: int = typer.Option(
        32,
        "--lora-alpha",
        help="LoRA alpha",
    ),
    lora_dropout: float = typer.Option(
        0.05,
        "--lora-dropout",
        help="LoRA dropout",
    ),

    # Logging arguments
    use_wandb: bool = typer.Option(
        False,
        "--use-wandb",
        help="Use Weights & Biases logging",
    ),
    wandb_project: str = typer.Option(
        "qlora-post-training",
        "--wandb-project",
        help="W&B project name",
    ),
    wandb_run_name: Optional[str] = typer.Option(
        None,
        "--wandb-run-name",
        help="W&B run name",
    ),

    # Other arguments
    seed: int = typer.Option(
        42,
        "--seed",
        help="Random seed",
    ),
    config_file: Optional[str] = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to YAML config file (overrides other args)",
    ),
    finance_mode: bool = typer.Option(
        False,
        "--finance-mode",
        "-f",
        help="Use finance preset configuration",
    ),
):
    """Run SFT training with QLoRA."""

    # Print header
    console.print(Panel.fit(
        "[bold cyan]QLoRA SFT Training[/bold cyan]\n"
        f"Model: {model_name}\n"
        f"Dataset: {dataset}",
        border_style="cyan",
    ))
    console.print()

    # Load from config file if provided
    if config_file:
        console.print(f"[cyan]Loading config from: {config_file}[/cyan]")
        config = FINANCE_SFT_CONFIG.from_yaml(config_file)

        model_config = config.model
        training_config = config.training
        lora_config = config.lora
        data_config = config.data
        logging_config = config.logging

    elif finance_mode:
        # Use finance preset
        console.print("[yellow]Using finance preset configuration[/yellow]")
        model_config = FINANCE_SFT_CONFIG.model
        training_config = FINANCE_SFT_CONFIG.training
        lora_config = FINANCE_SFT_CONFIG.lora
        data_config = FINANCE_SFT_CONFIG.data
        logging_config = FINANCE_SFT_CONFIG.logging

        # Override with CLI args
        model_config.name = model_name
        model_config.quantization_bits = quantization_bits
        model_config.max_length = max_length
        data_config.dataset_name = dataset
        if max_samples:
            data_config.max_samples = max_samples
        data_config.validation_split = validation_split
        training_config.output_dir = output_dir
        training_config.num_epochs = num_epochs
        training_config.batch_size = batch_size
        training_config.gradient_accumulation_steps = gradient_accumulation_steps
        training_config.learning_rate = learning_rate
        training_config.warmup_ratio = warmup_ratio
        training_config.seed = seed
        lora_config.r = lora_r
        lora_config.lora_alpha = lora_alpha
        lora_config.lora_dropout = lora_dropout
        logging_config.use_wandb = use_wandb
        if use_wandb:
            logging_config.wandb_project = wandb_project
            logging_config.wandb_run_name = wandb_run_name

    else:
        # Create configs from CLI args
        model_config = ModelConfig(
            name=model_name,
            quantization_bits=quantization_bits,
            max_length=max_length,
        )

        lora_config = LoRAConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
        )

        training_config = TrainingConfig(
            output_dir=output_dir,
            num_epochs=num_epochs,
            batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            warmup_ratio=warmup_ratio,
            seed=seed,
        )

        data_config = DataConfig(
            dataset_name=dataset,
            max_samples=max_samples,
            validation_split=validation_split,
        )

        logging_config = LoggingConfig(
            use_wandb=use_wandb,
            wandb_project=wandb_project,
            wandb_run_name=wandb_run_name,
        )

    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Print configuration summary
    console.print(Panel.fit(
        f"""[bold]Configuration:[/bold]

Model: {model_config.name}
Quantization: {model_config.quantization_bits}-bit
Max Length: {model_config.max_length}

Dataset: {data_config.dataset_name}
Max Samples: {data_config.max_samples or 'All'}
Validation Split: {data_config.validation_split}

Epochs: {training_config.num_epochs}
Batch Size: {training_config.batch_size}
Grad Accum: {training_config.gradient_accumulation_steps}
Effective BS: {training_config.effective_batch_size}
Learning Rate: {training_config.learning_rate}

LoRA r: {lora_config.r}
LoRA alpha: {lora_config.lora_alpha}
LoRA dropout: {lora_config.lora_dropout}

Output: {training_config.output_dir}
Seed: {training_config.seed}""",
        border_style="green",
    ))
    console.print()

    # Confirm
    console.print("[yellow]Starting training in 3 seconds... (Ctrl+C to cancel)[/yellow]")
    import time
    time.sleep(3)

    # Run training
    try:
        run_sft_training(
            model_config=model_config,
            training_config=training_config,
            lora_config=lora_config,
            data_config=data_config,
            logging_config=logging_config,
        )

        console.print("\n[bold green]✓ Training completed successfully![/bold green]")
        console.print(f"[cyan]Model saved to: {output_dir}[/cyan]\n")

    except KeyboardInterrupt:
        console.print("\n[yellow]Training interrupted by user[/yellow]")
        raise
    except Exception as e:
        console.print(f"\n[red]✗ Training failed: {e}[/red]")
        raise


if __name__ == "__main__":
    app()
