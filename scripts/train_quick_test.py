#!/usr/bin/env python3
"""
Quick Training Test Script

Tests the complete training pipeline with minimal data.
Use this to verify your setup before running full training.

Model: Qwen 2.5-0.5B-Instruct (4-bit)
Dataset: 100 finance-filtered samples
Epochs: 1
Expected time: 5-10 minutes

Usage:
    python scripts/train_quick_test.py

Or on Windows:
    ssh windows
    cd ~/4bit-QLoRA-post-training
    source venv/bin/activate
    python scripts/train_quick_test.py
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Set Hugging Face mirror (required in China)
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from config.base import ModelConfig, TrainingConfig, LoRAConfig, DataConfig, LoggingConfig
from src.models import load_model_and_tokenizer
from src.data import AlpacaDataset
from src.training.sft_trainer import SFTTrainer
from src.utils import set_seed, console


def main():
    """Run quick training test."""

    console.print("\n[bold cyan]=== Quick Training Test ===[/bold cyan]\n")
    console.print("[yellow]This will download Qwen 0.5B and train for ~5-10 minutes[/yellow]\n")

    # Model configuration - smaller model for testing
    model_config = ModelConfig(
        name="Qwen/Qwen2.5-0.5B-Instruct",
        quantization_bits=4,
        use_flash_attention=False,  # Flash Attention not installed
        max_length=512,
        torch_dtype="bfloat16",
    )

    # Training configuration
    training_config = TrainingConfig(
        output_dir="./outputs/test-quick",
        num_epochs=1,
        batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        gradient_checkpointing=True,
        bf16=True,
        seed=42,
    )

    # LoRA configuration
    lora_config = LoRAConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
    )

    # Data configuration - small dataset
    data_config = DataConfig(
        dataset_name="yahma/alpaca-cleaned",
        max_samples=100,
        validation_split=0.1,
        format="alpaca",
    )

    # Logging configuration
    logging_config = LoggingConfig(
        use_wandb=False,
        use_tensorboard=False,
        console_level="INFO",
    )

    # Print configuration
    console.print("[bold]Test Configuration:[/bold]")
    console.print(f"  Model: {model_config.name}")
    console.print(f"  Quantization: {model_config.quantization_bits}-bit")
    console.print(f"  Samples: {data_config.max_samples}")
    console.print(f"  Epochs: {training_config.num_epochs}")
    console.print(f"  Effective batch: {training_config.effective_batch_size}")
    console.print()

    # Set seed
    set_seed(training_config.seed)

    # Create trainer
    trainer = SFTTrainer(
        model_config=model_config,
        training_config=training_config,
        lora_config=lora_config,
        data_config=data_config,
        logging_config=logging_config,
    )

    # Run training
    try:
        console.print("[cyan]Starting training...[/cyan]\n")

        trainer.prepare_model()
        trainer.prepare_data()
        trainer.setup_trainer()
        trainer.train()
        trainer.evaluate()

        console.print("\n[bold green]✅ Test Training Complete![/bold green]")
        console.print(f"[cyan]Model saved to: {training_config.output_dir}[/cyan]\n")

        console.print("[bold]Next Steps:[/bold]")
        console.print("1. If test passed, run full training:")
        console.print("   python scripts/train_finance_full.py")
        console.print("2. View results:")
        console.print(f"   ls {training_config.output_dir}")
        console.print()

    except KeyboardInterrupt:
        console.print("\n[yellow]Test interrupted by user[/yellow]")
        raise
    except Exception as e:
        console.print(f"\n[red]Test failed: {e}[/red]\n")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
EOF

echo "✓ Created scripts/train_quick_test.py"
