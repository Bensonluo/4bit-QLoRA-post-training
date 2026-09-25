"""Configuration base classes for QLoRA post-training."""

from dataclasses import dataclass, field
from typing import Literal


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
    quantization_bits: int | None = 4
    load_in_8bit: bool = False
    trust_remote_code: bool = True
    use_flash_attention: bool = True
    max_length: int = 512
    device_map: str | None = "auto"
    torch_dtype: str = "bfloat16"
    # 🆕 Precision used when merging LoRA adapter back into the base model for
    # registration. Defaults to bf16 (full-precision merge for best quality).
    merge_dtype: str = "bfloat16"

    def __post_init__(self) -> None:
        """Validate configuration and auto-adjust for platform."""
        if self.quantization_bits not in [4, 8, None]:
            raise ValueError("quantization_bits must be 4, 8, or None")

        if self.torch_dtype not in ["float32", "float16", "bfloat16"]:
            raise ValueError("torch_dtype must be one of: float32, float16, bfloat16")

        # Auto-adjust for platform
        from src.utils.platform_utils import get_platform

        platform_info = get_platform()

        if platform_info.is_mps or platform_info.device == "cpu":
            # Quantization not supported on MPS/CPU — disable it
            if self.quantization_bits is not None:
                import warnings

                warnings.warn(
                    f"Quantization (bitsandbytes) is not available on {platform_info.device}. "
                    "Setting quantization_bits=None. Model will load in full precision.",
                    stacklevel=2,
                )
                self.quantization_bits = None

            # device_map is not supported on MPS
            if self.device_map == "auto":
                self.device_map = None


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
        use_dora: Opt in to DoRA (Weight-Decomposed Low-Rank Adaptation, peft
            >= 0.10). Closes roughly half of the LoRA-to-full-finetuning
            quality gap for ~5-10% extra VRAM, but the peft implementation
            trains noticeably slower than plain LoRA — keep off for speed-
            critical runs. Works with 4-bit quantized (QLoRA) bases.
        use_rslora: Opt in to Rank-Stabilized LoRA scaling (arXiv:2312.03732):
            scales the adapter by α/√r instead of α/r, which prevents the
            gradient vanishing that stunts standard LoRA at high ranks.
            Meaningful for r >= 32-64 (near-identical at low ranks).
        init_lora_weights: peft initialization strategy. True (default) keeps
            the standard Kaiming/zeros init; a string selects a research
            variant (validated by peft's LoraConfig at model-prep time):
            "pissa" — SVD init from the principal components of W
              (arXiv:2404.02948, NeurIPS 2024): faster convergence AND
              reduced quantization error, a good fit for 4-bit QLoRA;
            "pissa_niter_<N>" — fast iterative-SVD variant;
            "loftq" — quantization-aware init (arXiv:2310.08659, LoftQ):
              initializes A/B from the SVD of (W - quantize(W)) so the
              adapter compensates quantization error from step 0 — designed
              for 4-bit QLoRA bases. Trainers inject the required
              loftq_config dict from loftq_bits/loftq_iter below;
            "lora_ga" — gradient-approximation init (LoRA-GA, arXiv:2407.05000):
              initializes A/B from the SVD of an estimated full-finetuning
              gradient so the adapter starts aligned with the FT direction
              (paper: matches full FT loss from step ~1 instead of hundreds
              of steps). FULL PRECISION ONLY — peft hard-rejects quantized
              bases (LoRA-GA needs full-precision gradients during
              preprocessing), so SFTConfig rejects init_lora_weights="lora_ga"
              combined with quantization_bits in (4, 8), and DPO/GRPO reject
              it outright (the calibration targets a supervised loss; no
              validated calibration exists for preference/policy-gradient
              losses). SFTTrainer runs peft's preprocess_loraga on a small
              calibration slice of the training data before attaching the
              adapter — without that step peft silently falls back to
              gaussian init, which is why this must be trainer-driven;
            "gaussian", "orthogonal", "eva", "olora", "corda" — other
              documented strategies.
        loftq_bits: Quantization bit-width for LoftQ init (default 4,
            matching the QLoRA base). Only used when
            init_lora_weights="loftq".
        loftq_iter: Alternating SVD iterations for LoftQ init (default 1;
            higher = closer approximation, slower init). Only used when
            init_lora_weights="loftq".
        lora_ga_direction: LoRA-GA subspace selection — how A/B rows are
            drawn from the SVD factors of the estimated gradient
            ("ArB2r" default, "ArBr", "A2rBr", "random"; paper Table 5
            finds ArB2r most stable). Only used when
            init_lora_weights="lora_ga".
        lora_ga_scale: LoRA-GA output scaling ("stable" default, "gd_scale",
            "unit", "weight_svd"). Only used when init_lora_weights="lora_ga".
        lora_ga_stable_gamma: Stability scaling factor for
            lora_ga_scale="stable" (paper default 16).
        lora_ga_calibration_batches: Number of calibration batches the SFT
            trainer feeds to peft's gradient estimation (default 4). More
            batches = a less noisy gradient estimate, at linear calibration
            cost. Only used when init_lora_weights="lora_ga".
        lora_ga_cache_file: Optional path for caching estimated LoRA-GA
            gradients (peft preprocess_loraga cache_file). If the file
            exists, gradients load from it and calibration is skipped
            entirely — useful when re-running the same base+data combo.
            Only used when init_lora_weights="lora_ga".
        rank_pattern: Per-module rank overrides (peft LoraConfig field) —
            a dict mapping module-name regex to a rank, e.g.
            {"^model.layers.0.self_attn.q_proj": 32}. Modules matching a
            key use that rank instead of r; unlisted modules keep r. peft
            regex semantics: keys are matched against the full module name
            with "$" auto-appended; "^" anchors to the start of the module
            name; unanchored keys match name suffixes ("q_proj" matches
            model.layers.0.q_proj but not q_proj_deep). Use case: spend
            adapter capacity where it matters — e.g. higher rank on early
            attention projections, lower on MLP; MoE expert matrices
            (peft docs recommend rank_pattern for Mixtral/Qwen3-MoE
            experts). This is also the substrate adaptive-rank methods
            build on (e.g. peft's EVA, and the FIM-guided allocation
            proposal #3203). Values must be positive ints.
        alpha_pattern: Per-module lora_alpha overrides — same regex
            semantics as rank_pattern. Set alongside rank_pattern to keep a
            constant alpha/r scaling ratio per module (e.g. rank 32 alpha 64
            + rank 8 alpha 16 both keep 2x). Values must be positive ints.
        exclude_modules: Module names (regex string or list) to exclude
            from LoRA targeting — subtracted AFTER target_modules is
            applied. E.g. target_modules="all-linear" +
            exclude_modules=["lm_head", "score"] targets every linear
            except the output head. Same regex semantics as
            target_modules.
    """

    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj"]
    )
    bias: Literal["none", "all", "lora_only"] = "none"
    task_type: str = "CAUSAL_LM"
    use_dora: bool = False
    use_rslora: bool = False
    init_lora_weights: bool | str = True
    loftq_bits: int = 4
    loftq_iter: int = 1
    lora_ga_direction: Literal["ArBr", "A2rBr", "ArB2r", "random"] = "ArB2r"
    lora_ga_scale: Literal["stable", "weight_svd", "gd_scale", "unit"] = "stable"
    lora_ga_stable_gamma: int = 16
    lora_ga_calibration_batches: int = 4
    lora_ga_cache_file: str | None = None
    rank_pattern: dict[str, int] | None = None
    alpha_pattern: dict[str, int] | None = None
    exclude_modules: str | list[str] | None = None

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.r <= 0:
            raise ValueError("LoRA rank r must be positive")

        if self.lora_alpha <= 0:
            raise ValueError("LoRA alpha must be positive")

        if not 0 <= self.lora_dropout <= 1:
            raise ValueError("lora_dropout must be between 0 and 1")

        if self.bias not in ["none", "all", "lora_only"]:
            raise ValueError('bias must be one of: "none", "all", "lora_only"')

        if self.loftq_bits <= 0:
            raise ValueError("loftq_bits must be a positive integer")

        if self.loftq_iter < 1:
            raise ValueError("loftq_iter must be at least 1")

        if self.lora_ga_direction not in ("ArBr", "A2rBr", "ArB2r", "random"):
            raise ValueError('lora_ga_direction must be one of: "ArBr", "A2rBr", "ArB2r", "random"')

        if self.lora_ga_scale not in ("stable", "weight_svd", "gd_scale", "unit"):
            raise ValueError(
                'lora_ga_scale must be one of: "stable", "weight_svd", "gd_scale", "unit"'
            )

        if self.lora_ga_stable_gamma < 1:
            raise ValueError("lora_ga_stable_gamma must be a positive integer")

        if self.lora_ga_calibration_batches < 1:
            raise ValueError("lora_ga_calibration_batches must be at least 1")

        # Per-module rank/alpha overrides — peft consumes these as regex
        # keys with positive-int values; validate shape here so a typo fails
        # at config construction, not at adapter injection.
        if self.rank_pattern is not None:
            for key, value in self.rank_pattern.items():
                if not key or not isinstance(value, int) or isinstance(value, bool) or value < 1:
                    raise ValueError(
                        f"rank_pattern entries must map non-empty regex keys to positive ints; got {key!r}: {value!r}"
                    )
        if self.alpha_pattern is not None:
            for key, value in self.alpha_pattern.items():
                if not key or not isinstance(value, int) or isinstance(value, bool) or value < 1:
                    raise ValueError(
                        f"alpha_pattern entries must map non-empty regex keys to positive ints; got {key!r}: {value!r}"
                    )
        if self.exclude_modules is not None and not isinstance(self.exclude_modules, (str, list)):
            raise ValueError(
                "exclude_modules must be a regex string or a list of module-name strings"
            )
        if isinstance(self.exclude_modules, list):
            for entry in self.exclude_modules:
                if not isinstance(entry, str) or not entry:
                    raise ValueError("exclude_modules list entries must be non-empty strings")


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
        gradient_checkpointing_use_reentrant: Which torch.utils.checkpoint
            implementation to use when gradient_checkpointing is on. False
            (non-reentrant) is the PyTorch-recommended setting and is
            required for reliable operation with frozen-base fine-tunes
            (LoRA/QLoRA): the reentrant variant historically raises
            "element 0 of tensors does not require grad" on detached
            checkpoint inputs and is being deprecated upstream.
        torch_empty_cache_steps: Run torch.cuda.empty_cache() every N steps
            (None = off). Mitigates allocator FRAGMENTATION OOMs ("memory
            reports free but allocation fails") common with variable-length
            batches; does not add usable memory and each call forces a
            GPU-CPU sync, so keep N ≥ 50.
        auto_find_batch_size: On CUDA OOM, automatically restart training
            with a halved batch size (exponential decay, starting from
            batch_size) until one fits. Avoids manual OOM-retry loops on
            shared 8 GB cards. Caveats: (1) incompatible with DeepSpeed
            ZeRO-3 — HF Trainer raises; rejected at config validation when
            the deepspeed_config path names a zero_stage_3 preset;
            (2) each failed attempt restarts the loop from scratch, so a
            wildly oversized start wastes time; (3) halving the batch size
            HALVES the effective batch (gradient_accumulation_steps is not
            rescaled) — scale it manually if a constant effective batch
            matters.
        train_sampling_strategy: Training dataloader sampler — "random"
            (default), "sequential", or "group_by_length". "group_by_length"
            uses HF's LengthGroupedSampler to batch similar-length samples
            together, minimizing padding waste on variable-length data
            (finance/medical instruction sets vary 10x in length — this is
            a free throughput win, no hardware requirement). Caveats:
            (1) a one-time length pass runs at startup unless the dataset
            has a precomputed "length" column (HF default
            length_column_name="length"); (2) length-correlated batching
            mildly reduces shuffling randomness — megabatches are still
            shuffled internally; (3) ignored for IterableDataset.
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
    gradient_checkpointing_use_reentrant: bool = False
    torch_empty_cache_steps: int | None = None
    auto_find_batch_size: bool = False
    train_sampling_strategy: str = "random"
    fp16: bool = False
    bf16: bool = True
    max_grad_norm: float = 1.0
    seed: int = 42
    # 🆕 Distributed training fields (all default to off — single-GPU behavior is unchanged).
    #
    # Resource-unconstrained, industry-standard 2026 stack:
    #   - FSDP (PyTorch-native) is the DEFAULT multi-GPU strategy. Prefer bf16 full
    #     precision + FSDP over QLoRA + DeepSpeed when GPU memory allows.
    #   - DeepSpeed remains available for cases needing CPU/NVMe offload.
    #
    # Exactly one of (deepspeed_config, fsdp) should be set. The CLI resolver in
    # config.distributed.resolve_distributed_config() enforces mutual exclusion.
    #
    # FSDP: HF Trainer `fsdp=` string. Set to "full_shard" (params+grads+optim
    #   sharded, ≈ ZeRO-3) or "sharded_grad_scaled" (≈ ZeRO-2).
    fsdp: str | None = None
    # FSDP auto-wrap config dict (transformer_layer_cls_to_wrap, min_num_params, ...).
    # Built by config.distributed when an FSDP preset is selected.
    fsdp_config: dict | None = None
    # Path to a DeepSpeed JSON config. When set, HF Trainer is launched with
    # `deepspeed=<path>` and torchrun is expected to wrap the process.
    deepspeed_config: str | None = None
    # Whether to auto-derive `per_device_train_batch_size` from world_size (kept total
    # effective batch constant). Off by default — explicit per-device batch is clearer.
    auto_scale_batch: bool = False
    # Opt-in Liger fused Triton kernels (RMSNorm/RoPE/SwiGLU/fused linear CE):
    # ~20% throughput gain and up to 60% activation-memory reduction on supported
    # CUDA models. Requires `pip install -e ".[liger]"`. No effect on MPS/CPU.
    use_liger_kernel: bool = False
    # NEFTune noisy-embedding regularizer (arXiv:2310.05914, ICLR 2024). Adds
    # uniform noise scaled α/√(L·d) to input embeddings during training only
    # (HF Trainer auto-disables the hook at eval). α=5 is the community default
    # (typical window 5-15); on LLaMA-2-7B/Alpaca this lifted AlpacaEval from
    # 29.79% to 64.69%. Validated for instruction SFT — SFTTrainer forwards it;
    # DPO/GRPO deliberately do not (unvalidated for preference/RL objectives).
    # Note: NEFTune's noise is computed incorrectly under sequence packing —
    # this framework does not pack, so the field is safe here.
    neftune_noise_alpha: float | None = None
    # Opt-in torch.compile of the training model (TrainingArguments.torch_compile).
    # Only sane on the bf16 full-precision path (FSDP/DDP with
    # quantization_bits=None): Dynamo cannot trace bitsandbytes
    # Params4bit/Linear4bit custom autograd ops, so compiling a 4-bit QLoRA run
    # fails or silently degrades. SFTTrainer warns at setup when combined
    # with 4/8-bit quantization.
    torch_compile: bool = False
    # Optional optimizer override, forwarded to HF TrainingArguments.optim.
    # None keeps the library default. Recommended on CUDA QLoRA:
    # "paged_adamw_8bit" — the QLoRA-paper recipe (arXiv:2305.14314): 8-bit
    # optimizer states with NVIDIA unified-memory paging that swaps to CPU
    # RAM during VRAM spikes, preventing OOM on 8 GB cards. Valid names are
    # validated by HF Trainer at runtime (OptimizerNames).
    optim: str | None = None
    # Peak tensor-core throughput of ONE training device in FLOP/s, used by
    # the MFU observability callback (TRL >= 1.4 compute_mfu helpers). None
    # keeps TRL's default (9.895e14 ≈ H100 bf16 dense). Set your device's
    # real peak for meaningful MFU on other hardware, e.g. A100 bf16 dense
    # ≈ 3.12e14.
    peak_flops_per_device: float | None = None

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")

        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")

        if not 0 < self.warmup_ratio <= 1:
            raise ValueError("warmup_ratio must be between 0 and 1")

        if self.fp16 and self.bf16:
            raise ValueError("Cannot enable both fp16 and bf16")

        if self.neftune_noise_alpha is not None and self.neftune_noise_alpha <= 0:
            raise ValueError("neftune_noise_alpha must be positive (typical range 5-15)")

        if self.peak_flops_per_device is not None and self.peak_flops_per_device <= 0:
            raise ValueError("peak_flops_per_device must be positive (FLOP/s of one device)")

        # FSDP and DeepSpeed are mutually exclusive (both try to own sharding).
        if self.fsdp and self.deepspeed_config:
            raise ValueError(
                "Cannot enable both fsdp and deepspeed_config — pick one sharding strategy."
            )

        # HF Trainer hard-errors on auto_find_batch_size under ZeRO-3 — fail
        # at config construction instead of deep inside the training loop.
        if (
            self.auto_find_batch_size
            and self.deepspeed_config
            and "zero_stage_3" in str(self.deepspeed_config)
        ):
            raise ValueError(
                "auto_find_batch_size is incompatible with DeepSpeed ZeRO-3 "
                "(HF Trainer raises). Use ZeRO-2, ZeRO-1, or FSDP instead."
            )

        # Sampler choice — validated locally so a typo fails at config
        # construction, not deep inside the Trainer's dataloader setup.
        if self.train_sampling_strategy not in ("random", "sequential", "group_by_length"):
            raise ValueError(
                "train_sampling_strategy must be one of: random, sequential, group_by_length"
            )

    @property
    def effective_batch_size(self) -> int:
        """Calculate effective batch size including gradient accumulation."""
        return self.batch_size * self.gradient_accumulation_steps

    @property
    def is_distributed(self) -> bool:
        """True when a sharding strategy (FSDP/DeepSpeed) is configured OR torchrun is active."""
        import os

        return (
            os.environ.get("WORLD_SIZE", "1") != "1"
            or self.deepspeed_config is not None
            or self.fsdp is not None
        )


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
    max_samples: int | None = None
    validation_split: float = 0.1
    preprocessing_num_workers: int = 4
    train_file: str | None = None
    validation_file: str | None = None
    format: str = "alpaca"

    def __post_init__(self) -> None:
        """Validate configuration."""
        if not 0 <= self.validation_split < 1:
            raise ValueError("validation_split must be between 0 and 1")

        if self.format not in ["alpaca", "chat", "sharegpt", "dpo", "grpo"]:
            raise ValueError('format must be one of: "alpaca", "chat", "sharegpt", "dpo", "grpo"')


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
    wandb_entity: str | None = None
    wandb_run_name: str | None = None
    use_tensorboard: bool = True
    log_dir: str = "./outputs/logs"
    log_memory: bool = True
    console_level: str = "INFO"
    use_mlflow: bool = False
    mlflow_tracking_uri: str = "./outputs/mlruns"
    mlflow_experiment_name: str = "qlora-post-training"
    mlflow_run_name: str | None = None
    # 🆕 Model Registry configuration (all default off — backward compatible).
    # When register_model=True AND use_mlflow=True, training auto-merges the LoRA
    # adapter into the base model and registers it to the MLflow Model Registry,
    # creating a model version with lineage back to the training run.
    register_model: bool = False
    registry_model_name: str | None = None  # defaults to model name if None
    merge_before_register: bool = True  # merge adapter into base before logging
    registry_stage: str = "Staging"  # initial stage: Staging/Production/Archived


# NOTE: DPOConfig lives in config/dpo.py (single source of truth). The copy that
# used to live here was an unused duplicate and was removed.
