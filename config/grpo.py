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
        min_p: Min-p sampling — minimum token probability, scaled by the
            probability of the most likely token (TRL GRPOConfig field;
            typical values 0.01-0.2). Unlike top_p/top_k, min-p keeps a
            dynamic floor relative to the top token, which filters long-tail
            garbage without capping a confident model's diversity. None
            (default) disables it. CAVEAT (TRL issue #6789): ANY truncation
            sampling (min_p set, top_p < 1, or top_k > 0) under
            use_vllm=True biases the importance-sampling correction — vLLM
            renormalizes log-probs over the truncated distribution while
            the trainer recomputes over the full vocabulary. The config
            warns on that combination; untruncated sampling
            (top_p=1, top_k=0, min_p=None) is exact.
        generation_kwargs: Optional dict passed through verbatim to
            GenerationConfig (transformers) / SamplingParams (vLLM) at
            rollout time — the escape hatch for sampling knobs TRL does not
            surface as first-class fields (suppress_tokens, stop_strings,
            no_repeat_ngram_size, ...). Keys conflicting with the top-level
            generation parameters (min_p, top_p, ...) OVERRIDE them — TRL
            documented semantics; prefer the first-class fields when one
            exists.
        use_vllm: Whether to use vLLM for faster group sampling. Requires the
            optional `vllm` extra (`pip install -e ".[vllm]"`) and a CUDA GPU
            with enough VRAM to share between training and the vLLM engine —
            8 GB consumer cards are below the practical floor.
        vllm_mode: "colocate" (vLLM shares the training process; single-node
            default) or "server" (external vLLM server, overlaps generation
            with training but needs extra infra).
        vllm_gpu_memory_utilization: Fraction of GPU memory vLLM may claim for
            weights + KV cache. TRL default 0.3; lower it (0.2-0.25) if
            training OOMs while vLLM holds its cache (classic colocate
            failure mode).
        scale_rewards: How to normalize rewards ("group" or "global").
        num_iterations: Number of policy update iterations per generated batch.
        epsilon: Clipping parameter for GRPO surrogate loss.
        epsilon_high: Upper clipping bound for DAPO Clip-Higher
            (arXiv:2503.14476). None = symmetric clipping (falls back to
            epsilon). Setting e.g. epsilon=0.2 + epsilon_high=0.28 widens
            the upper bound to encourage exploration on low-probability
            tokens — now common practice in GRPO-based reasoning RL.
        mask_truncated_completions: DAPO Overlong Filtering — mask the loss
            of completions truncated by max_completion_length so cutoff
            artifacts don't pollute training. Recommended when training on
            long chain-of-thought.
            Version caveat: TRL < 1.9.1 normalizes the DAPO-family loss
            incorrectly when steps_per_generation differs from
            gradient_accumulation_steps (fixed in TRL #6024). This config
            does not expose steps_per_generation, so TRL keeps it equal to
            gradient_accumulation_steps and the bug cannot trigger here —
            but keep them equal if you ever hand-build TRL args.
        importance_sampling_level: Where importance ratios are computed —
            "token" (standard GRPO, TRL default) or "sequence" (GSPO,
            arXiv:2507.18071, Qwen team). Sequence-level ratios (one per
            sequence instead of per token) are more stable for LLM RL;
            TRL labels its implementation experimental.
        loss_type: GRPO loss variant ("grpo", "dapo", "bnpo", "dr_grpo", "cispo").
            Defaults to "dapo" — TRL >= 1.4's recommended default, which
            eliminates the length bias of token-level normalization.
            "cispo" (MiniMax-M1, arXiv:2506.13585) truncates the importance
            sampling weights themselves instead of clipping the
            advantage-scaled ratios: out-of-distribution tokens get their IS
            weight capped but keep contributing gradient (PPO/DAPO-style
            clipping zeroes them entirely). In TRL, epsilon_high plays the
            paper's ε_max role when loss_type="cispo".
        top_entropy_quantile: Entropy-based token filtering (arXiv:2506.01939,
            "Beyond the 80/20 Rule"): keep only the top-rho quantile of tokens
            by per-position entropy in the policy loss — high-entropy
            "minority" tokens carry most of the RL signal. 1.0 (default)
            keeps all tokens; the paper recommends 0.2. Composes with
            mask_truncated_completions (only non-truncated completions
            contribute entropy).
    """

    beta: float = 0.04
    num_generations: int = 8
    max_completion_length: int = 256
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0
    repetition_penalty: float = 1.0
    min_p: float | None = None
    generation_kwargs: dict | None = None
    use_vllm: bool = False
    vllm_mode: Literal["colocate", "server"] = "colocate"
    vllm_gpu_memory_utilization: float = 0.3
    scale_rewards: Literal["group", "global", "none"] = "group"
    num_iterations: int = 1
    epsilon: float = 0.2
    epsilon_high: float | None = None
    mask_truncated_completions: bool = False
    importance_sampling_level: Literal["token", "sequence"] = "token"
    loss_type: Literal["grpo", "dapo", "bnpo", "dr_grpo", "cispo"] = "dapo"
    top_entropy_quantile: float = 1.0

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

        if self.epsilon_high is not None and not 0 <= self.epsilon_high <= 1:
            raise ValueError("epsilon_high must be between 0 and 1 when set")

        if self.min_p is not None and not 0 <= self.min_p <= 1:
            raise ValueError("min_p must be between 0 and 1 when set")

        # Truncation sampling under vLLM biases the importance-sampling
        # correction (TRL issue #6789): vLLM renormalizes log-probs over the
        # truncated distribution, the trainer recomputes over the full
        # vocabulary. TRL's own config-level warning postdates 1.4.0 — this
        # guard surfaces the caveat before any vLLM engine launches.
        if self.use_vllm and (self.min_p is not None or self.top_p < 1 or self.top_k > 0):
            import warnings

            warnings.warn(
                "Truncation sampling (min_p set, top_p < 1, or top_k > 0) with "
                "use_vllm=True biases the vLLM importance-sampling correction "
                "(TRL issue #6789): sampler log-probs are renormalized over the "
                "truncated distribution while the trainer recomputes over the "
                "full vocabulary. Keep top_p=1, top_k=0, min_p=None for exact "
                "ratios, or accept a per-token bias in the correction term.",
                stacklevel=2,
            )

        # TRL does not range-check this field (1.5 is silently accepted) —
        # validate here so a typo fails at config construction.
        if not 0 <= self.top_entropy_quantile <= 1:
            raise ValueError("top_entropy_quantile must be between 0 and 1")

        # Literal fields are static-typed only — validate at runtime so a
        # typo fails at config construction, not deep inside TRL.
        if self.loss_type not in ("grpo", "dapo", "bnpo", "dr_grpo", "cispo"):
            raise ValueError("loss_type must be one of: grpo, dapo, bnpo, dr_grpo, cispo")

        if self.importance_sampling_level not in ("token", "sequence"):
            raise ValueError("importance_sampling_level must be 'token' or 'sequence'")

        if self.scale_rewards not in ("group", "global", "none"):
            raise ValueError("scale_rewards must be 'group', 'global', or 'none'")

        if self.vllm_mode not in ("colocate", "server"):
            raise ValueError("vllm_mode must be 'colocate' or 'server'")

        if not 0 < self.vllm_gpu_memory_utilization <= 1:
            raise ValueError("vllm_gpu_memory_utilization must be in (0, 1]")

        if self.use_vllm and self.vllm_mode == "server":
            # Server mode requires an external vLLM server; colocate is the
            # only mode this repo launches automatically. Allowed but loud.
            import warnings

            warnings.warn(
                "vllm_mode='server' expects an external vLLM server "
                "(vllm serve <model>) — the trainer will not start one.",
                stacklevel=2,
            )


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

        if self.lora_config.init_lora_weights == "lora_ga":
            raise ValueError(
                "init_lora_weights='lora_ga' (LoRA-GA) is not supported for GRPO: the "
                "technique calibrates adapters against a supervised (LM) loss gradient; "
                "no validated calibration exists for policy-gradient losses. Use another "
                "init strategy (e.g. 'pissa' or the default)."
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
