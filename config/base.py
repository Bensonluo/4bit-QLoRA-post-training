"""Configuration base classes for QLoRA post-training."""

from dataclasses import dataclass, field
from typing import List, Literal, Optional


@dataclass
class ModelConfig:
    """Model configuration for loading and quantization.

    Attributes:
        name: Hugging Face model name or path
        quantization_bits: Number of bits for quantization (4 or 8)
        load_in_8bit: Whether to use 8-bit quantization instead of 4-bit
        trust_remote_code: Whether to trust remote code from HF
        use_flash_attention: Whether to use Flash Attention 2
        max_length: Maximum sequence length
        device_map: Device mapping strategy ("auto", "balanced", or manual)
        torch_dtype: Data type for model weights
    """

    name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    quantization_bits: int = 4
    load_in_8bit: bool = False
    trust_remote_code: bool = True
    use_flash_attention: bool = True
    max_length: int = 512
    device_map: str = "auto"
    torch_dtype: str = "bfloat16"

    def __post_init__(self):
        """Validate configuration."""
        if self.quantization_bits not in [4, 8]:
            raise ValueError("quantization_bits must be 4 or 8")

        if self.torch_dtype not in ["float32", "float16", "bfloat16"]:
            raise ValueError(
                "torch_dtype must be one of: float32, float16, bfloat16"
            )


@dataclass
class LoRAConfig:
    """LoRA (Low-Rank Adaptation) configuration.

    Attributes:
        r: LoRA rank (higher = more parameters but more expressive)
        lora_alpha: LoRA scaling factor (typically 2x r)
        lora_dropout: Dropout probability for LoRA layers
        target_modules: Which modules to apply LoRA to
        bias: Whether to train bias parameters ("none", "all", or "lora_only")
        task_type: Task type for LoRA (CAUSAL_LM for most LLMs)
    """

    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj"]
    )
    bias: str = "none"
    task_type: str = "CAUSAL_LM"

    def __post_init__(self):
        """Validate configuration."""
        if self.r <= 0:
            raise ValueError("LoRA rank r must be positive")

        if self.lora_alpha <= 0:
            raise ValueError("LoRA alpha must be positive")

        if not 0 <= self.lora_dropout <= 1:
            raise ValueError("lora_dropout must be between 0 and 1")

        if self.bias not in ["none", "all", "lora_only"]:
            raise ValueError('bias must be one of: "none", "all", "lora_only"')


@dataclass
class TrainingConfig:
    """Training configuration for fine-tuning.

    Attributes:
        output_dir: Directory to save checkpoints and logs
        num_epochs: Number of training epochs
        batch_size: Training batch size (typically 1-2 for 8GB VRAM)
        gradient_accumulation_steps: Steps to accumulate gradients
        learning_rate: Learning rate (typically 1e-4 to 5e-4 for LoRA)
        weight_decay: Weight decay for regularization
        warmup_ratio: Fraction of training steps for warmup
        lr_scheduler_type: Learning rate scheduler type
        logging_steps: Log training metrics every N steps
        save_steps: Save checkpoint every N steps
        eval_steps: Evaluate every N steps
        save_total_limit: Maximum number of checkpoints to keep
        gradient_checkpointing: Whether to use gradient checkpointing
        fp16: Use mixed precision (fp16)
        bf16: Use bfloat16 mixed precision (preferred on RTX 30xx+)
        max_grad_norm: Maximum gradient norm for clipping
        seed: Random seed for reproducibility
    """

    output_dir: str = "./outputs/checkpoints"
    num_epochs: int = 3
    batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    logging_steps: int = 10
    save_steps: int = 100
    eval_steps: int = 100
    save_total_limit: int = 3
    gradient_checkpointing: bool = True
    fp16: bool = False
    bf16: bool = True
    max_grad_norm: float = 1.0
    seed: int = 42

    def __post_init__(self):
        """Validate configuration."""
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")

        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")

        if not 0 < self.warmup_ratio <= 1:
            raise ValueError("warmup_ratio must be between 0 and 1")

        if self.fp16 and self.bf16:
            raise ValueError("Cannot enable both fp16 and bf16")

    @property
    def effective_batch_size(self) -> int:
        """Calculate effective batch size including gradient accumulation."""
        return self.batch_size * self.gradient_accumulation_steps


@dataclass
class DataConfig:
    """Data configuration for training.

    Attributes:
        dataset_name: Hugging Face dataset name or local path
        dataset_split: Dataset split to use ("train", "test", etc.)
        max_samples: Maximum number of samples to use (None for all)
        validation_split: Fraction of data for validation (0 to 1)
        preprocessing_num_workers: Number of workers for preprocessing
        train_file: Path to custom training data
        validation_file: Path to custom validation data
        format: Data format ("alpaca", "chat", "sharegpt", etc.)
    """

    dataset_name: str = "yahma/alpaca-cleaned"
    dataset_split: str = "train"
    max_samples: Optional[int] = None
    validation_split: float = 0.1
    preprocessing_num_workers: int = 4
    train_file: Optional[str] = None
    validation_file: Optional[str] = None
    format: str = "alpaca"

    def __post_init__(self):
        """Validate configuration."""
        if not 0 <= self.validation_split < 1:
            raise ValueError("validation_split must be between 0 and 1")

        if self.format not in ["alpaca", "chat", "sharegpt", "dpo"]:
            raise ValueError(
                'format must be one of: "alpaca", "chat", "sharegpt", "dpo"'
            )


@dataclass
class LoggingConfig:
    """Logging and monitoring configuration.

    Attributes:
        use_wandb: Whether to use Weights & Biases
        wandb_project: W&B project name
        wandb_entity: W&B entity/team
        wandb_run_name: Custom W&B run name
        use_tensorboard: Whether to use TensorBoard
        log_dir: Directory for TensorBoard logs
        log_memory: Whether to log GPU memory usage
        console_level: Console logging level
    """

    use_wandb: bool = False
    wandb_project: str = "qlora-post-training"
    wandb_entity: Optional[str] = None
    wandb_run_name: Optional[str] = None
    use_tensorboard: bool = True
    log_dir: str = "./outputs/logs"
    log_memory: bool = True
    console_level: str = "INFO"


@dataclass
class DPOConfig:
    """DPO-specific configuration.

    Attributes:
        beta: DPO beta parameter (controls preference strength)
        max_length: Maximum sequence length for DPO
        max_prompt_length: Maximum prompt length
        max_target_length: Maximum target length
        reference_model_free: Whether to freeze reference model
        loss_type: DPO loss type ("sigmoid", "hinge", "ipo", etc.)
    """

    beta: float = 0.1
    max_length: int = 512
    max_prompt_length: int = 128
    max_target_length: int = 384
    reference_model_free: bool = False
    loss_type: str = "sigmoid"

    def __post_init__(self):
        """Validate configuration."""
        if self.beta <= 0:
            raise ValueError("beta must be positive")

        if self.loss_type not in ["sigmoid", "hinge", "ipo", "pairwise"]:
            raise ValueError(
                'loss_type must be one of: "sigmoid", "hinge", "ipo", "pairwise"'
            )
