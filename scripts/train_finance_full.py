#!/usr/bin/env python3
"""
Full Finance Domain Training Script

This script performs complete supervised fine-tuning on finance data.
Optimized for RTX 4060 8GB VRAM using 4-bit QLoRA.

Usage:
    python scripts/train_finance_full.py

Or on Windows via SSH:
    ssh windows "cd ~/4bit-QLoRA-post-training && source venv/bin/activate && python scripts/train_finance_full.py"

Expected runtime: 2-3 hours on RTX 4060
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Set Hugging Face mirror for China (required for fast downloads)
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from config.base import ModelConfig, TrainingConfig, LoRAConfig, DataConfig, LoggingConfig
from src.models import load_model_and_tokenizer
from src.data import FinanceDataset
from src.training.sft_trainer import SFTTrainer
from src.utils import set_seed, console, setup_logging


def main():
    """Run complete finance domain training."""
    
    console.print("\n[bold cyan]=== Finance Domain Training ===[/bold cyan]\n")
    
    # Model configuration - Qwen 1.5B with 4-bit quantization
    model_config = ModelConfig(
        name="Qwen/Qwen2.5-1.5B-Instruct",
        quantization_bits=4,
        use_flash_attention=False,  # Flash Attention 2 not installed
        max_length=1024,
        torch_dtype="bfloat16",
    )
    
    # Training configuration - optimized for 8GB VRAM
    training_config = TrainingConfig(
        output_dir="./outputs/finance-sft",
        num_epochs=3,
        batch_size=1,  # Small batch size for 8GB VRAM
        gradient_accumulation_steps=8,  # Effective batch size = 8
        learning_rate=2e-4,
        weight_decay=0.01,
        warmup_ratio=0.03,
        gradient_checkpointing=True,  # Required for 8GB VRAM
        bf16=True,
        seed=42,
    )
    
    # LoRA configuration
    lora_config = LoRAConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],  # Minimal for memory
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    # Data configuration
    data_config = DataConfig(
        dataset_name="yahma/alpaca-cleaned",
        max_samples=50000,  # Large finance dataset
        validation_split=0.1,
        format="alpaca",
    )
    
    # Logging configuration
    logging_config = LoggingConfig(
        use_wandb=False,
        use_tensorboard=True,
        log_dir="./outputs/logs",
        log_memory=True,
        console_level="INFO",
    )
    
    # Print configuration summary
    console.print("[bold]Configuration:[/bold]")
    console.print(f"  Model: {model_config.name}")
    console.print(f"  Quantization: {model_config.quantization_bits}-bit")
    console.print(f"  Max samples: {data_config.max_samples:,}")
    console.print(f"  Epochs: {training_config.num_epochs}")
    console.print(f"  Effective batch size: {training_config.effective_batch_size}")
    console.print(f"  Gradient checkpointing: {training_config.gradient_checkpointing}")
    console.print(f"  LoRA rank: {lora_config.r}")
    console.print()
    
    # Set random seed for reproducibility
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
        trainer.prepare_model()
        trainer.prepare_data()
        trainer.setup_trainer()
        trainer.train()
        trainer.evaluate()
        
        console.print("\n[bold green]✅ Training Complete![/bold green]")
        console.print(f"[cyan]Model saved to: {training_config.output_dir}[/cyan]\n")
        
    except KeyboardInterrupt:
        console.print("\n[yellow]Training interrupted by user[/yellow]")
        raise
    except Exception as e:
        console.print(f"\n[red]Training failed: {e}[/red]\n")
        raise


if __name__ == "__main__":
    main()
