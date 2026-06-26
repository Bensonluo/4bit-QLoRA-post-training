"""GRPO (Group Relative Policy Optimization) configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from config.base import DataConfig, LoggingConfig, LoRAConfig, ModelConfig, TrainingConfig


@dataclass
class RewardConfig:
    """Configuration for reward functions used in GRPO training.

    Attributes:
        reward_funcs: List of reward function names to use.
            Built-in: "format", "accuracy", "llm_judge", "length", "cosine".
        reward_weights: Optional per-reward weights. If None, all rewards are
            summed with equal weight.
        judge_model: Model name or path for LLM-as-a-Judge reward.
        judge_prompt_template: Optional custom prompt template for judge.
        answer_key: Dataset column key containing the reference answer for
            accuracy reward.
        reference_key: Dataset column key containing reference response for
            judge/ similarity rewards.
    """

    reward_funcs: list[str] = field(default_factory=lambda: ["format", "accuracy"])
    reward_weights: dict[str, float] | None = None
    judge_model: str | None = None
    judge_prompt_template: str | None = None
    answer_key: str = "answer"
    reference_key: str = "reference"

    def __post_init__(self) -> None:
        """Validate reward configuration."""
        if not self.reward_funcs:
            raise ValueError("At least one reward function must be specified")

        valid_funcs = {"format", "accuracy", "llm_judge", "length", "cosine"}
        invalid = set(self.reward_funcs) - valid_funcs
        if invalid:
            raise ValueError(f"Invalid reward functions: {invalid}. Valid options: {valid_funcs}")

        if "llm_judge" in self.reward_funcs and not self.judge_model:
            raise ValueError("judge_model must be specified when using llm_judge reward")


@dataclass
class GRPOConfig:
    """GRPO-specific configuration.

    Attributes:
        beta: KL penalty coefficient.
        num_generations: Number of completions sampled per prompt (group size).
        max_completion_length: Maximum generation length for completions.
        temperature: Sampling temperature for generation.
        top_p: Nucleus sampling top_p.
        top_k: Top-k sampling.
        repetition_penalty: Repetition penalty.
        use_vllm: Whether to use vLLM for faster group sampling.
        scale_rewards: How to normalize rewards ("group" or "global").
        num_iterations: Number of policy update iterations per generated batch.
        epsilon: Clipping parameter for GRPO surrogate loss.
        loss_type: GRPO loss variant ("grpo", "dapo", "drgrpo").
    """

    beta: float = 0.04
    num_generations: int = 8
    max_completion_length: int = 256
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0
    repetition_penalty: float = 1.0
    use_vllm: bool = False
    scale_rewards: Literal["group", "global", "none"] = "group"
    num_iterations: int = 1
    epsilon: float = 0.2
    loss_type: Literal["grpo", "dapo", "drgrpo"] = "grpo"

    def __post_init__(self) -> None:
        """Validate GRPO configuration."""
        if self.beta < 0:
            raise ValueError("beta must be non-negative")

        if self.num_generations < 2:
            raise ValueError("num_generations must be at least 2")

        if self.max_completion_length <= 0:
            raise ValueError("max_completion_length must be positive")

        if not 0 <= self.epsilon <= 1:
            raise ValueError("epsilon must be between 0 and 1")


@dataclass
class GRPOTrainingConfig:
    """Complete configuration for GRPO training."""

    model_config: ModelConfig = field(default_factory=ModelConfig)
    training_config: TrainingConfig = field(default_factory=TrainingConfig)
    lora_config: LoRAConfig = field(default_factory=LoRAConfig)
    grpo_config: GRPOConfig = field(default_factory=GRPOConfig)
    reward_config: RewardConfig = field(default_factory=RewardConfig)
    data_config: DataConfig = field(default_factory=DataConfig)
    reference_config: ModelConfig | None = None
    logging_config: LoggingConfig = field(default_factory=LoggingConfig)

    def __post_init__(self) -> None:
        """Set default reference config to model config if not provided."""
        if self.reference_config is None:
            self.reference_config = ModelConfig(
                name=self.model_config.name,
                quantization_bits=self.model_config.quantization_bits,
                load_in_8bit=self.model_config.load_in_8bit,
                trust_remote_code=self.model_config.trust_remote_code,
                use_flash_attention=self.model_config.use_flash_attention,
                max_length=self.model_config.max_length,
                device_map=self.model_config.device_map,
                torch_dtype=self.model_config.torch_dtype,
                merge_dtype=self.model_config.merge_dtype,
            )

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"GRPOTrainingConfig(\n"
            f"  model={self.model_config.name},\n"
            f"  beta={self.grpo_config.beta},\n"
            f"  num_generations={self.grpo_config.num_generations},\n"
            f"  max_completion_length={self.grpo_config.max_completion_length},\n"
            f"  reward_funcs={self.reward_config.reward_funcs},\n"
            f"  dataset={self.data_config.dataset_name},\n"
            f"  lora_r={self.lora_config.r}\n"
            f")"
        )


# Finance-specific preset
FINANCE_GRPO_CONFIG = GRPOTrainingConfig(
    model_config=ModelConfig(
        name="Qwen/Qwen2.5-1.5B-Instruct",
        quantization_bits=4,
        max_length=1024,
        torch_dtype="bfloat16",
    ),
    lora_config=LoRAConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    ),
    training_config=TrainingConfig(
        output_dir="./outputs/grpo-finance",
        num_epochs=1,
        batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=5e-6,
        gradient_checkpointing=True,
        bf16=True,
    ),
    grpo_config=GRPOConfig(
        beta=0.04,
        num_generations=4,
        max_completion_length=256,
    ),
    data_config=DataConfig(
        dataset_name="yahma/alpaca-cleaned",
        max_samples=1000,
        validation_split=0.1,
        format="grpo",
    ),
    reward_config=RewardConfig(
        reward_funcs=["format", "accuracy"],
        answer_key="answer",
    ),
    logging_config=LoggingConfig(
        use_wandb=True,
        wandb_project="finance-grpo",
        use_tensorboard=True,
        log_memory=True,
    ),
)
