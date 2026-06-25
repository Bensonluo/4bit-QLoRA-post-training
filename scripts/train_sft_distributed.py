#!/usr/bin/env python3
"""CLI script for distributed SFT training (DDP / DeepSpeed).

This is the torchrun-compatible counterpart to `train_sft.py`. The only
differences vs single-GPU training:

  1. It accepts `--distributed-preset` (ddp | zero_stage_1 | zero_stage_2 |
     zero_stage_3_offload) or `--deepspeed-config <path>`, and injects the
     resolved DeepSpeed JSON into TrainingConfig.deepspeed_config.
  2. Only rank 0 prints the banner / sleeps / final success message. The
     actual training loop is identical — HF Trainer handles process-group
     init, DistributedSampler, and gradient all-reduce internally.

Launch it with torchrun (or accelerate launch). Never invoke directly for
multi-GPU — `python scripts/train_sft_distributed.py` will run single-GPU.

Examples:
    # Pure DDP on 4 GPUs
    torchrun --nproc_per_node=4 scripts/train_sft_distributed.py \\
        --model-name Qwen/Qwen3-0.6B --distributed-preset ddp

    # DeepSpeed ZeRO-2 (recommended for QLoRA) on 2 GPUs
    torchrun --nproc_per_node=2 scripts/train_sft_distributed.py \\
        --model-name Qwen/Qwen3-1.7B --distributed-preset zero_stage_2

    # Custom DeepSpeed config
    torchrun --nproc_per_node=4 scripts/train_sft_distributed.py \\
        --deepspeed-config /path/to/my_zero.json
"""

import os
import sys
import time
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.base import DataConfig, LoggingConfig, LoRAConfig, ModelConfig, TrainingConfig
from config.distributed import resolve_distributed_config
from src.training import run_sft_training
from src.training.distributed import get_distributed_info

app = typer.Typer(
    name="train-sft-distributed",
    help="Distributed SFT training (DDP / DeepSpeed) — launch via torchrun",
    add_completion=False,
)

console = Console()


@app.command()
def main(
    # Model arguments
    model_name: str = typer.Option(
        "Qwen/Qwen3-0.6B",
        "--model-name",
        "-m",
        help="Hugging Face model name or path",
    ),
    quantization_bits: int = typer.Option(
        4,
        "--quantization-bits",
        "-q",
        help="Quantization bits (4 or 8). Set 0 for full precision (recommended with ZeRO-3).",
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
    max_samples: int | None = typer.Option(
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
        "./outputs/sft_distributed",
        "--output-dir",
        "-o",
        help="Output directory for checkpoints",
    ),
    num_epochs: int = typer.Option(3, "--epochs", "-e", help="Number of training epochs"),
    batch_size: int = typer.Option(
        1,
        "--batch-size",
        "-b",
        help="PER-DEVICE batch size (total = this × world_size × grad_accum)",
    ),
    gradient_accumulation_steps: int = typer.Option(
        8, "--gradient-accumulation-steps", "-g", help="Gradient accumulation steps"
    ),
    learning_rate: float = typer.Option(2e-4, "--learning-rate", "--lr", help="Learning rate"),
    warmup_ratio: float = typer.Option(0.03, "--warmup-ratio", help="Warmup ratio"),
    # LoRA arguments
    lora_r: int = typer.Option(16, "--lora-r", help="LoRA rank"),
    lora_alpha: int = typer.Option(32, "--lora-alpha", help="LoRA alpha"),
    lora_dropout: float = typer.Option(0.05, "--lora-dropout", help="LoRA dropout"),
    # 🆕 Distributed arguments
    distributed_preset: str | None = typer.Option(
        None,
        "--distributed-preset",
        "-p",
        help=(
            "Named distributed strategy. FSDP is the PyTorch-native default: "
            "fsdp_full (≈ZeRO-3), fsdp_grad (≈ZeRO-2). DeepSpeed for offload: "
            "zero_stage_1/2/3_offload. ddp = plain replication. "
            "Mutually exclusive with --deepspeed-config / --fsdp-mode."
        ),
    ),
    deepspeed_config: str | None = typer.Option(
        None,
        "--deepspeed-config",
        help="Direct path to a DeepSpeed JSON (overrides --distributed-preset).",
    ),
    fsdp_mode: str | None = typer.Option(
        None,
        "--fsdp-mode",
        help=(
            "Direct HF Trainer fsdp string: 'full_shard' (params+grads+optim) or "
            "'sharded_grad_scaled' (grads+optim). Overrides --distributed-preset."
        ),
    ),
    # Logging arguments
    use_wandb: bool = typer.Option(False, "--use-wandb", help="Use Weights & Biases logging"),
    wandb_project: str = typer.Option(
        "qlora-post-training", "--wandb-project", help="W&B project name"
    ),
    wandb_run_name: str | None = typer.Option(None, "--wandb-run-name", help="W&B run name"),
    seed: int = typer.Option(42, "--seed", help="Random seed"),
):
    """Run distributed SFT training with QLoRA (launch via torchrun)."""

    # 🆕 Detect distributed context + resolve DeepSpeed config.
    dist_info = get_distributed_info()
    # 🆕 Resolve the full distributed config (FSDP or DeepSpeed) via the unified resolver.
    dist_cfg = resolve_distributed_config(
        preset_name=distributed_preset,
        deepspeed_path=deepspeed_config,
        fsdp_mode=fsdp_mode,
    )

    # 🆕 Only rank 0 prints banner / sleeps / final message — otherwise N copies flood the log.
    is_rank0 = dist_info.is_main_process

    if is_rank0:
        if dist_cfg.fsdp:
            strategy_label = f"FSDP ({dist_cfg.fsdp})"
        elif dist_cfg.deepspeed_config:
            strategy_label = f"DeepSpeed ({dist_cfg.preset.value})"
        else:
            strategy_label = "Pure DDP (no sharding)"
        console.print(Panel.fit(
            "[bold cyan]Distributed SFT Training[/bold cyan]\n"
            f"Model: {model_name}\n"
            f"Dataset: {dataset}\n"
            f"World size: {dist_info.world_size}\n"
            f"Strategy: {strategy_label}",
            border_style="cyan",
        ))
        console.print()

        if not dist_info.is_distributed:
            console.print(
                "[yellow]⚠ WORLD_SIZE=1 — running single-GPU. "
                "For multi-GPU, launch with: torchrun --nproc_per_node=N "
                f"{' '.join(sys.argv)}[/yellow]\n"
            )

    # Sanity: FSDP/ZeRO-3 + bnb quantization can be unstable. We default to bf16 full
    # precision in the launch scripts, but warn if the user combines sharding with QLoRA.
    uses_param_sharding = dist_cfg.fsdp == "full_shard" or (
        dist_cfg.deepspeed_config and "zero_stage_3" in dist_cfg.deepspeed_config
    )
    if uses_param_sharding and quantization_bits in (4, 8) and is_rank0:
        console.print(
            "[bold red]⚠ WARNING: full parameter sharding (FSDP full_shard / ZeRO-3) + "
            "bitsandbytes quantization can be unstable. The standard 2026 practice is "
            "bf16 full precision (--quantization-bits 0) with FSDP. Proceeding, but "
            "switch to bf16 if you see NaN losses.[/bold red]\n"
        )

    # Build configs.
    model_config = ModelConfig(
        name=model_name,
        quantization_bits=quantization_bits,
        max_length=max_length,
    )
    lora_config = LoRAConfig(r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout)
    training_config = TrainingConfig(
        output_dir=output_dir,
        num_epochs=num_epochs,
        batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        warmup_ratio=warmup_ratio,
        seed=seed,
        # 🆕 Inject the resolved distributed strategy (exactly one of fsdp/deepspeed is set).
        fsdp=dist_cfg.fsdp,
        fsdp_config=dist_cfg.fsdp_config,
        deepspeed_config=dist_cfg.deepspeed_config,
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

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    if is_rank0:
        total_batch = batch_size * dist_info.world_size * gradient_accumulation_steps
        if training_config.fsdp:
            dist_label = f"FSDP {training_config.fsdp}"
        elif training_config.deepspeed_config:
            dist_label = f"DeepSpeed {os.path.basename(training_config.deepspeed_config)}"
        else:
            dist_label = "DDP"
        console.print(Panel.fit(
            f"""[bold]Configuration:[/bold]

Model: {model_config.name}
Quantization: {model_config.quantization_bits or 'none (full bf16)'}-bit
World size: {dist_info.world_size}

Dataset: {data_config.dataset_name}
Max Samples: {data_config.max_samples or 'All'}

Epochs: {training_config.num_epochs}
Per-device BS: {training_config.batch_size}
Grad Accum: {training_config.gradient_accumulation_steps}
Total effective BS: {total_batch}  (per_device × world_size × grad_accum)
Learning Rate: {training_config.learning_rate}
Distributed: {dist_label}

LoRA r: {lora_config.r}  alpha: {lora_config.lora_alpha}

Output: {training_config.output_dir}""",
            border_style="green",
        ))
        console.print()
        console.print("[yellow]Starting in 3s... (Ctrl+C to cancel)[/yellow]")
        time.sleep(3)

    # Run training — identical call to single-GPU; HF Trainer handles the rest.
    try:
        run_sft_training(
            model_config=model_config,
            training_config=training_config,
            lora_config=lora_config,
            data_config=data_config,
            logging_config=logging_config,
        )
        if is_rank0:
            console.print("\n[bold green]✓ Distributed training completed![/bold green]")
            console.print(f"[cyan]Model saved to: {output_dir}[/cyan]\n")
    except KeyboardInterrupt:
        if is_rank0:
            console.print("\n[yellow]Training interrupted[/yellow]")
        raise
    except Exception as e:
        if is_rank0:
            console.print(f"\n[red]✗ Training failed: {e}[/red]")
        raise


if __name__ == "__main__":
    app()
