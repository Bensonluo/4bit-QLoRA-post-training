"""SFT (Supervised Fine-Tuning) specific configurations."""

import yaml
from pathlib import Path
from typing import Optional

from config.base import DataConfig, LoRAConfig, LoggingConfig, ModelConfig, TrainingConfig


class SFTConfig:
    """Complete configuration for SFT training."""

    def __init__(
        self,
        model: Optional[ModelConfig] = None,
        lora: Optional[LoRAConfig] = None,
        training: Optional[TrainingConfig] = None,
        data: Optional[DataConfig] = None,
        logging: Optional[LoggingConfig] = None,
    ):
        """Initialize SFT configuration.

        Args:
            model: Model configuration
            lora: LoRA configuration
            training: Training configuration
            data: Data configuration
            logging: Logging configuration
        """
        self.model = model or ModelConfig()
        self.lora = lora or LoRAConfig()
        self.training = training or TrainingConfig()
        self.data = data or DataConfig()
        self.logging = logging or LoggingConfig()

    @classmethod
    def from_yaml(cls, path: str) -> "SFTConfig":
        """Load configuration from YAML file.

        Args:
            path: Path to YAML file

        Returns:
            SFTConfig instance
        """
        config_path = Path(path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(config_path, "r") as f:
            config_dict = yaml.safe_load(f)

        return cls(
            model=ModelConfig(**config_dict.get("model", {})),
            lora=LoRAConfig(**config_dict.get("lora", {})),
            training=TrainingConfig(**config_dict.get("training", {})),
            data=DataConfig(**config_dict.get("data", {})),
            logging=LoggingConfig(**config_dict.get("logging", {})),
        )

    def to_yaml(self, path: str) -> None:
        """Save configuration to YAML file.

        Args:
            path: Path to save YAML file
        """
        config_path = Path(path)
        config_path.parent.mkdir(parents=True, exist_ok=True)

        config_dict = {
            "model": self.model.__dict__,
            "lora": self.lora.__dict__,
            "training": self.training.__dict__,
            "data": self.data.__dict__,
            "logging": self.logging.__dict__,
        }

        with open(config_path, "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False)

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"SFTConfig(\n"
            f"  model={self.model.name},\n"
            f"  quantization={self.model.quantization_bits}bit,\n"
            f"  epochs={self.training.num_epochs},\n"
            f"  batch_size={self.training.batch_size},\n"
            f"  grad_accum={self.training.gradient_accumulation_steps},\n"
            f"  lr={self.training.learning_rate},\n"
            f"  lora_r={self.lora.r},\n"
            f"  dataset={self.data.dataset_name}\n"
            f")"
        )


# Finance-specific preset
FINANCE_SFT_CONFIG = SFTConfig(
    model=ModelConfig(
        name="Qwen/Qwen2.5-1.5B-Instruct",
        quantization_bits=4,
        max_length=1024,
        torch_dtype="bfloat16",
    ),
    lora=LoRAConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    ),
    training=TrainingConfig(
        output_dir="./outputs/finance-sft",
        num_epochs=3,
        batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        gradient_checkpointing=True,
        bf16=True,
    ),
    data=DataConfig(
        dataset_name="yahma/alpaca-cleaned",  # Will override with finance dataset
        max_samples=50000,
        validation_split=0.1,
        format="alpaca",
    ),
    logging=LoggingConfig(
        use_wandb=True,
        wandb_project="finance-qlora",
        use_tensorboard=True,
        log_memory=True,
    ),
)
