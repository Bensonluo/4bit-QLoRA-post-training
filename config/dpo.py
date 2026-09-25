"""DPO (Direct Preference Optimization) configuration.

DPO trains on preference pairs to align models with human preferences
without training a separate reward model.
"""

from __future__ import annotations

from dataclasses import dataclass

from config.base import LoggingConfig, LoRAConfig, ModelConfig, TrainingConfig

# Loss variants supported by TRL >= 1.0 DPOConfig. TRL takes loss_type as a
# LIST (multiple losses combined via loss_weights); we expose a single
# selection here and forward it as a one-element list in the trainer.
DPO_LOSS_TYPES = (
    "sigmoid",
    "hinge",
    "ipo",
    "exo_pair",
    "nca_pair",
    "robust",
    "bco_pair",
    "sppo_hard",
    "aot",
    "aot_unpaired",
    "apo_zero",
    "apo_down",
    "discopop",
    "sft",
    "sigmoid_norm",
)


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
        max_length: Maximum sequence length for DPO (prompt + completion).
            TRL >= 1.0 removed separate prompt/target truncation — this single
            value governs truncation end to end.
        loss_type: DPO loss function (see DPO_LOSS_TYPES for valid values)
        label_smoothing: Apply smoothing to labels
        precompute_ref_log_probs: TRL memory optimization — compute reference
            log-probs in one upfront pass so the reference model is not kept
            in VRAM during the optimization loop (DPO otherwise needs roughly
            two models' worth of memory). Trade-off: a one-time precompute
            pass over the dataset before training starts.
        activation_offloading: TRL memory optimization — offload activation
            tensors to CPU RAM during the forward pass and bring them back
            only for the backward pass (implemented via PyTorch
            saved_tensors_hooks; CUDA streams overlap the transfers by
            default). Significantly cuts PEAK VRAM at the cost of slightly
            slower training. Complements precompute_ref_log_probs on 8 GB
            cards: no resident ref model + offloaded activations. DPO-only —
            TRL does not expose it on GRPOConfig and our SFT path uses plain
            HF TrainingArguments (which lacks the field).

    Note:
        TRL's `sync_ref_model` (TR-DPO, arXiv:2404.09656 — EMA-style sync of
        the reference to the policy every ref_model_sync_steps) is
        deliberately NOT exposed: TRL hard-errors on `sync_ref_model=True`
        with a PEFT policy, and this framework's DPO trainer always trains a
        LoRA adapter. It also conflicts with precompute_ref_log_probs. Do
        not forward it without first supporting full fine-tuning DPO.
        TRL's `padding_free` (flattened padding-less forward) is likewise
        NOT exposed: it requires FlashAttention 2/3 (absent on this repo's
        8 GB RTX 4060 target), and TRL 1.4.0 hard-disables the feature
        anyway ("temporarily unavailable after a refactor", silently
        falling back to standard padding).
    """

    beta: float = 0.1
    max_length: int = 512
    loss_type: str = "sigmoid"
    label_smoothing: float = 0.0
    precompute_ref_log_probs: bool = False
    activation_offloading: bool = False

    def __post_init__(self) -> None:
        """Validate DPO configuration."""
        if self.beta <= 0:
            raise ValueError("beta must be positive")

        if self.loss_type not in DPO_LOSS_TYPES:
            raise ValueError(
                f"Invalid loss_type: {self.loss_type}. Must be one of: {', '.join(DPO_LOSS_TYPES)}"
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

    def __post_init__(self) -> None:
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
    max_samples: int | None = 5000  # Start with smaller dataset
    validation_split: float = 0.1
    format: str = "preference"
    auto_filter: bool = False
    preprocess_num_workers: int = 4

    def __post_init__(self) -> None:
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
        model_config: ModelConfig | None = None,
        training_config: TrainingConfig | None = None,
        lora_config: LoRAConfig | None = None,
        dpo_config: DPOConfig | None = None,
        data_config: PreferenceDataConfig | None = None,
        reference_config: ReferenceModelConfig | None = None,
        logging_config: LoggingConfig | None = None,
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

        if self.lora_config.init_lora_weights == "lora_ga":
            raise ValueError(
                "init_lora_weights='lora_ga' (LoRA-GA) is not supported for DPO: the "
                "technique calibrates adapters against a supervised (LM) loss gradient; "
                "no validated calibration exists for preference losses. Use "
                "another init strategy (e.g. 'pissa' or the default)."
            )

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
