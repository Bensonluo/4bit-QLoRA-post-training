"""DPO (Direct Preference Optimization) configuration.

DPO trains on preference pairs to align models with human preferences
without training a separate reward model.
"""

from dataclasses import dataclass
from typing import Literal, Optional


@dataclass
class DPOConfig:
    """Configuration for DPO training.

    DPO directly optimizes policy using preference pairs, avoiding the need
    for a separate reward model (required in RLHF).

    Attributes:
        beta: DPO temperature parameter. Lower = more conservative
            - 0.1-0.2: Conservative (default)
            - 0.2-0.5: Moderate
            - 0.5-1.0: Aggressive
        max_length: Maximum sequence length for DPO
        max_prompt_length: Maximum prompt length
        max_target_length: Maximum target length
        reference_model_free: Whether to freeze reference model
        loss_type: DPO loss function ("sigmoid", "hinge", "ipo", "pairwise")
        label_smoothing: Apply smoothing to labels
        padding_value: Padding token ID
    """

    beta: float = 0.1
    max_length: int = 512
    max_prompt_length: int = 128
    max_target_length: int = 384
    reference_model_free: bool = False
    loss_type: Literal["sigmoid", "hinge", "ipo", "pairwise"] = "sigmoid"
    label_smoothing: float = 0.0
    padding_value: int = -100

    def __post_init__(self):
        """Validate DPO configuration."""
        if self.beta <= 0:
            raise ValueError("beta must be positive")

        if self.max_prompt_length + self.max_target_length > self.max_length:
            raise ValueError(
                f"Prompt ({self.max_prompt_length}) + target "
                f"({self.max_target_length}) exceeds max_length ({self.max_length})"
            )

        if self.loss_type not in ["sigmoid", "hinge", "ipo", "pairwise"]:
            raise ValueError(
                f"Invalid loss_type: {self.loss_type}. "
                f"Must be one of: sigmoid, hinge, ipo, pairwise"
            )

        if not 0 <= self.label_smoothing <= 1:
            raise ValueError("label_smoothing must be between 0 and 1")


@dataclass
class ReferenceModelConfig:
    """Configuration for the reference model in DPO.

    The reference model is frozen and used to compute rewards for preference
    optimization.

    Attributes:
        name: Reference model name or path
        quantization_bits: Quantization bits (4 or 8)
        use_flash_attention: Whether to use Flash Attention
        device_map: Device mapping strategy
    """

    name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    quantization_bits: int = 4
    use_flash_attention: bool = False
    device_map: str = "auto"
    torch_dtype: str = "bfloat16"

    def __post_init__(self):
        """Validate reference model configuration."""
        if self.quantization_bits not in [4, 8]:
            raise ValueError("quantization_bits must be 4 or 8")


@dataclass
class PreferenceDataConfig:
    """Configuration for preference datasets.

    DPO requires datasets with preference pairs: (prompt, chosen, rejected).

    Attributes:
        dataset_name: Hugging Face dataset name or local path
        split: Dataset split to use
        max_samples: Maximum number of samples (None for all)
        format: Data format ("preference" or custom)
        validation_split: Fraction of data for validation
        auto_filter: Whether to filter for finance content
        preprocess_num_workers: Number of workers for preprocessing
    """

    dataset_name: str = "HuggingFaceH4/argilla-dpo-mix-7k"
    split: str = "train"
    max_samples: Optional[int] = 5000  # Start with smaller dataset
    validation_split: float = 0.1
    format: str = "preference"
    auto_filter: bool = False
    preprocess_num_workers: int = 4

    def __post_init__(self):
        """Validate preference data configuration."""
        if not 0 <= self.validation_split < 1:
            raise ValueError("validation_split must be between 0 and 1")


@dataclass
class DPOTrainingConfig:
    """Complete DPO training configuration.

    This combines all configurations needed for DPO training.
    """

    def __init__(
        self,
        model_config: Optional['ModelConfig'] = None,
        training_config: Optional['TrainingConfig'] = None,
        lora_config: Optional['LoRAConfig'] = None,
        dpo_config: Optional[DPOConfig] = None,
        data_config: Optional[PreferenceDataConfig] = None,
        reference_config: Optional[ReferenceModelConfig] = None,
        logging_config: Optional['LoggingConfig'] = None,
    ):
        """Initialize DPO training configuration.

        Args:
            model_config: Main model configuration
            training_config: Training configuration
            lora_config: LoRA configuration
            dpo_config: DPO-specific configuration
            data_config: Preference dataset configuration
            reference_config: Reference model configuration
            logging_config: Logging configuration
        """
        from config.base import ModelConfig, TrainingConfig, LoRAConfig, LoggingConfig

        self.model_config = model_config or ModelConfig(
            name="Qwen/Qwen2.5-1.5B-Instruct",
            quantization_bits=4,
            use_flash_attention=False,
            max_length=512,
        )

        self.training_config = training_config or TrainingConfig(
            output_dir="./outputs/dpo",
            num_epochs=3,
            batch_size=1,
            gradient_accumulation_steps=8,
            learning_rate=1e-4,  # DPO typically uses lower LR
            gradient_checkpointing=True,
            bf16=True,
        )

        self.lora_config = lora_config or LoRAConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=["q_proj", "v_proj"],
        )

        self.dpo_config = dpo_config or DPOConfig()

        self.data_config = data_config or PreferenceDataConfig()

        self.reference_config = reference_config or ReferenceModelConfig(
            name="Qwen/Qwen2.5-0.5B-Instruct",  # Smaller reference model
            quantization_bits=4,
            use_flash_attention=False,
        )

        self.logging_config = logging_config or LoggingConfig()

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"DPOTrainingConfig(\n"
            f"  model={self.model_config.name},\n"
            f"  dpo_beta={self.dpo_config.beta},\n"
            f"  reference_model={self.reference_config.name},\n"
            f"  dataset={self.data_config.dataset_name}\n"
            f")"
        )


# Pre-configured DPO presets
FINANCE_DPO_CONFIG = DPOTrainingConfig()
FINANCE_DPO_CONFIG.model_config.quantization_bits = 4
FINANCE_DPO_CONFIG.model_config.use_flash_attention = False
FINANCE_DPO_CONFIG.dpo_config.beta = 0.1
FINANCE_DPO_CONFIG.data_config.auto_filter = True  # Filter for finance
FINANCE_DPO_CONFIG.training_config.num_epochs = 3
FINANCE_DPO_CONFIG.training_config.gradient_accumulation_steps = 8
