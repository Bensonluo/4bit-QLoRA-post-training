#!/usr/bin/env python3
"""DPO (Direct Preference Optimization) training script.

This script trains a language model using DPO with preference pairs.
Optimized for 8GB VRAM (RTX 4060).

Usage:
    # Quick test (100 samples, 1 epoch)
    python scripts/train_dpo.py --quick-test

    # Full training with default config
    python scripts/train_dpo.py

    # Custom dataset
    python scripts/train_dpo.py --dataset-name path/to/data.json

    # Resume from checkpoint
    python scripts/train_dpo.py --resume outputs/dpo/checkpoint-1000
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import argparse
from config.dpo import DPOTrainingConfig, FINANCE_DPO_CONFIG
from src.training.dpo_trainer import run_dpo_training
from src.utils.logging import console


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="DPO training script for preference optimization"
    )

    # Mode selection
    parser.add_argument(
        "--quick-test",
        action="store_true",
        help="Run quick test (100 samples, 1 epoch)",
    )

    # Model config
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Model name or path (default: from config)",
    )
    parser.add_argument(
        "--quantization-bits",
        type=int,
        choices=[4, 8],
        default=None,
        help="Quantization bits (default: from config)",
    )

    # Training config
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: from config)",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=None,
        help="Number of training epochs (default: from config)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size per device (default: from config)",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=None,
        help="Gradient accumulation steps (default: from config)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Learning rate (default: from config)",
    )

    # LoRA config
    parser.add_argument(
        "--lora-r",
        type=int,
        default=None,
        help="LoRA rank (default: from config)",
    )
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=None,
        help="LoRA alpha (default: from config)",
    )

    # DPO config
    parser.add_argument(
        "--beta",
        type=float,
        default=None,
        help="DPO beta parameter (default: 0.1)",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=None,
        help="Maximum sequence length (default: 512)",
    )

    # Data config
    parser.add_argument(
        "--dataset-name",
        type=str,
        default=None,
        help="Dataset name or path (default: from config)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples (default: from config)",
    )
    parser.add_argument(
        "--validation-split",
        type=float,
        default=None,
        help="Validation split fraction (default: 0.1)",
    )
    parser.add_argument(
        "--auto-filter",
        action="store_true",
        help="Enable auto-filtering for finance content",
    )

    # Reference model config
    parser.add_argument(
        "--reference-model",
        type=str,
        default=None,
        help="Reference model name or path (default: from config)",
    )

    # Logging
    parser.add_argument(
        "--use-wandb",
        action="store_true",
        help="Enable Weights & Biases logging",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default=None,
        help="W&B project name",
    )
    parser.add_argument(
        "--wandb-run-name",
        type=str,
        default=None,
        help="W&B run name",
    )

    # Resume
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume from checkpoint path",
    )

    return parser.parse_args()


def apply_args_to_config(config, args):
    """Apply command line arguments to configuration."""

    # Quick test mode
    if args.quick_test:
        console.print("[yellow]Quick test mode: 100 samples, 1 epoch[/yellow]")
        config.data_config.max_samples = 100
        config.training_config.num_epochs = 1
        config.training_config.output_dir = "./outputs/dpo-quick-test"
        config.logging_config.use_wandb = False

    # Model config
    if args.model_name:
        config.model_config.name = args.model_name
    if args.quantization_bits:
        config.model_config.quantization_bits = args.quantization_bits

    # Training config
    if args.output_dir:
        config.training_config.output_dir = args.output_dir
    if args.num_epochs:
        config.training_config.num_epochs = args.num_epochs
    if args.batch_size:
        config.training_config.batch_size = args.batch_size
    if args.gradient_accumulation_steps:
        config.training_config.gradient_accumulation_steps = args.gradient_accumulation_steps
    if args.learning_rate:
        config.training_config.learning_rate = args.learning_rate

    # LoRA config
    if args.lora_r:
        config.lora_config.r = args.lora_r
    if args.lora_alpha:
        config.lora_config.lora_alpha = args.lora_alpha

    # DPO config
    if args.beta:
        config.dpo_config.beta = args.beta
    if args.max_length:
        config.dpo_config.max_length = args.max_length

    # Data config
    if args.dataset_name:
        config.data_config.dataset_name = args.dataset_name
    if args.max_samples:
        config.data_config.max_samples = args.max_samples
    if args.validation_split:
        config.data_config.validation_split = args.validation_split
    if args.auto_filter:
        config.data_config.auto_filter = True

    # Reference model
    if args.reference_model:
        config.reference_config.name = args.reference_model

    # Logging
    if args.use_wandb:
        config.logging_config.use_wandb = True
    if args.wandb_project:
        config.logging_config.wandb_project = args.wandb_project
    if args.wandb_run_name:
        config.logging_config.wandb_run_name = args.wandb_run_name

    return config


def main():
    """Main training function."""
    args = parse_args()

    # Print banner
    console.print("\n[bold cyan]=== DPO Training Script ===[/bold cyan]\n")

    # Load configuration
    if args.quick_test:
        config = FINANCE_DPO_CONFIG
        console.print("[yellow]Using FINANCE_DPO_CONFIG with quick test overrides[/yellow]\n")
    else:
        config = FINANCE_DPO_CONFIG
        console.print("[yellow]Using FINANCE_DPO_CONFIG[/yellow]\n")

    # Apply command line arguments
    config = apply_args_to_config(config, args)

    # Print configuration
    console.print("[bold]Configuration:[/bold]")
    console.print(f"  Model: {config.model_config.name}")
    console.print(f"  Quantization: {config.model_config.quantization_bits}-bit")
    console.print(f"  Reference Model: {config.reference_config.name}")
    console.print(f"  Dataset: {config.data_config.dataset_name}")
    console.print(f"  Max Samples: {config.data_config.max_samples}")
    console.print(f"  DPO Beta: {config.dpo_config.beta}")
    console.print(f"  LoRA Rank: {config.lora_config.r}")
    console.print(f"  Epochs: {config.training_config.num_epochs}")
    console.print(f"  Batch Size: {config.training_config.batch_size}")
    console.print(f"  Gradient Accumulation: {config.training_config.gradient_accumulation_steps}")
    console.print(f"  Effective Batch Size: {config.training_config.effective_batch_size}")
    console.print(f"  Output Dir: {config.training_config.output_dir}")
    console.print(f"  W&B: {config.logging_config.use_wandb}")
    console.print()

    # Create output directory
    os.makedirs(config.training_config.output_dir, exist_ok=True)

    # Run training
    try:
        run_dpo_training(
            model_config=config.model_config,
            training_config=config.training_config,
            lora_config=config.lora_config,
            dpo_config=config.dpo_config,
            data_config=config.data_config,
            reference_config=config.reference_config,
            logging_config=config.logging_config,
        )

        console.print("\n[bold green]✅ DPO training completed successfully![/bold green]\n")

    except KeyboardInterrupt:
        console.print("\n[yellow]Training interrupted by user[/yellow]\n")
        sys.exit(1)

    except Exception as e:
        console.print(f"\n[red]Training failed: {e}[/red]\n")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
