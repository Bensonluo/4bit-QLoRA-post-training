"""CLI for GRPO training."""

from __future__ import annotations

from pathlib import Path

import typer
import yaml

from config.base import DataConfig, LoggingConfig, LoRAConfig, ModelConfig, TrainingConfig
from config.grpo import GRPOConfig, GRPOTrainingConfig, RewardConfig
from src.training.grpo_trainer import run_grpo_training
from src.utils import console

app = typer.Typer(help="GRPO training for LLMs")


@app.command()
def train(
    config: str | None = typer.Option(None, "--config", "-c", help="Path to YAML config file"),
    model_name: str = typer.Option("Qwen/Qwen2.5-1.5B-Instruct", "--model", help="Base model name"),
    dataset: str = typer.Option(
        "yahma/alpaca-cleaned", "--dataset", help="Dataset path or HF name"
    ),
    output_dir: str = typer.Option("./outputs/grpo", "--output-dir", help="Output directory"),
    num_epochs: int = typer.Option(1, "--epochs", help="Number of training epochs"),
    batch_size: int = typer.Option(1, "--batch-size", help="Per-device batch size"),
    grad_accum: int = typer.Option(8, "--grad-accum", help="Gradient accumulation steps"),
    lr: float = typer.Option(5e-6, "--lr", help="Learning rate"),
    lora_r: int = typer.Option(16, "--lora-r", help="LoRA rank"),
    beta: float = typer.Option(0.04, "--beta", help="KL penalty coefficient"),
    num_generations: int = typer.Option(4, "--num-generations", help="Group size"),
    max_completion_length: int = typer.Option(
        256, "--max-completion-length", help="Max generation length"
    ),
    reward_funcs: str = typer.Option(
        "format,accuracy", "--reward-funcs", help="Comma-separated reward functions"
    ),
    max_samples: int | None = typer.Option(1000, "--max-samples", help="Max training samples"),
    use_wandb: bool = typer.Option(False, "--use-wandb", help="Enable W&B logging"),
    use_mlflow: bool = typer.Option(False, "--use-mlflow", help="Enable MLflow tracking"),
) -> None:
    """Run GRPO training."""
    if config:
        cfg = _load_config_from_yaml(config)
    else:
        cfg = GRPOTrainingConfig(
            model_config=ModelConfig(name=model_name),
            training_config=TrainingConfig(
                output_dir=output_dir,
                num_epochs=num_epochs,
                batch_size=batch_size,
                gradient_accumulation_steps=grad_accum,
                learning_rate=lr,
                gradient_checkpointing=True,
                bf16=True,
            ),
            lora_config=LoRAConfig(r=lora_r, lora_alpha=lora_r * 2),
            grpo_config=GRPOConfig(
                beta=beta,
                num_generations=num_generations,
                max_completion_length=max_completion_length,
            ),
            reward_config=RewardConfig(
                reward_funcs=[r.strip() for r in reward_funcs.split(",")],
            ),
            data_config=DataConfig(
                dataset_name=dataset,
                max_samples=max_samples,
                format="grpo",
            ),
            logging_config=LoggingConfig(
                use_wandb=use_wandb,
                use_mlflow=use_mlflow,
            ),
        )

    console.print(cfg)
    run_grpo_training(cfg)


def _load_config_from_yaml(path: str) -> GRPOTrainingConfig:
    """Load GRPO config from YAML file."""
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(config_path) as f:
        data = yaml.safe_load(f)

    return GRPOTrainingConfig(
        model_config=ModelConfig(**data.get("model", {})),
        training_config=TrainingConfig(**data.get("training", {})),
        lora_config=LoRAConfig(**data.get("lora", {})),
        grpo_config=GRPOConfig(**data.get("grpo", {})),
        reward_config=RewardConfig(**data.get("reward", {})),
        data_config=DataConfig(**data.get("data", {})),
        logging_config=LoggingConfig(**data.get("logging", {})),
    )


if __name__ == "__main__":
    app()
