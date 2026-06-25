"""Distributed training presets: FSDP (PyTorch-native) + DeepSpeed.

Industry-standard open-source strategies for 2026:

    +-------------------+--------------------------------------------------+
    | Strategy          | When to use                                      |
    +-------------------+--------------------------------------------------+
    | FSDP (full_shard) | DEFAULT. PyTorch-native, no extra deps.          |
    | sharded_grad_scal | FSDP variant — shards grads+optimizer only.      |
    | DDP               | Plain replication, no sharding. Baseline.        |
    | DeepSpeed ZeRO-2  | Same idea as FSDP full_shard, via DeepSpeed.     |
    | DeepSpeed ZeRO-3  | Extreme scale — params/grads/optim all sharded + |
    | + offload         | optional CPU/NVMe offload. Use when FSDP OOMs.   |
    +-------------------+--------------------------------------------------+

Design notes:
- Resource constraints no longer apply: prefer bf16 full precision + FSDP over
  QLoRA + ZeRO. QLoRA remains available (the underlying SFT/DPO loop keeps it),
  but the distributed layer is now precision-agnostic and FSDP-first.
- FSDP needs the model loaded WITHOUT device_map (FSDP moves shards itself).
  The loader's distributed-mode pinning already handles this.
- HF Trainer accepts `fsdp=` and `fsdp_config=` directly — same injection pattern
  as DeepSpeed.
"""

import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

# Directory holding the JSON configs (DeepSpeed + FSDP auto-wrap settings).
_CONFIGS_DIR = Path(__file__).parent / "deepspeed_configs"


class DistributedPreset(str, Enum):
    """Named distributed strategies.

    `str` mixin so values serialize as strings (handy for CLI / YAML).
    Order roughly goes "simplest → most memory-saving".
    """

    # --- DDP: pure replication, no sharding ---
    DDP = "ddp"

    # --- FSDP (PyTorch-native, the DEFAULT for multi-GPU) ---
    # full_shard ≈ DeepSpeed ZeRO-3: shards params + grads + optimizer state.
    FSDP_FULL = "fsdp_full"
    # sharded_grad_scal ≈ DeepSpeed ZeRO-2: shards grads + optimizer, params stay whole.
    FSDP_GRAD = "fsdp_grad"

    # --- DeepSpeed (third-party, use when FSDP is insufficient) ---
    ZERO_1 = "zero_stage_1"
    ZERO_2 = "zero_stage_2"
    ZERO_3_OFFLOAD = "zero_stage_3_offload"

    @property
    def family(self) -> str:
        """Group label: 'ddp' | 'fsdp' | 'deepspeed'."""
        if self == DistributedPreset.DDP:
            return "ddp"
        if self.value.startswith("fsdp"):
            return "fsdp"
        return "deepspeed"

    @property
    def is_deepspeed(self) -> bool:
        return self.family == "deepspeed"

    @property
    def is_fsdp(self) -> bool:
        return self.family == "fsdp"


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ResolvedDistributedConfig:
    """The fully resolved distributed strategy, ready to inject into TrainingArguments.

    Exactly one of (deepspeed_config, fsdp) is active. `None/None` means pure DDP.

    Attributes:
        preset: The source preset (or DDP if none given).
        deepspeed_config: Absolute path to DeepSpeed JSON, or None.
        fsdp: HF Trainer `fsdp` string value, or None. One of:
            "full_shard" / "sharded_grad_scaled" / "shard_grad_offload" / _AUTO.
        fsdp_config: Dict passed to HF Trainer's `fsdp_config=` (auto-wrap settings).
    """

    preset: DistributedPreset
    deepspeed_config: str | None = None
    fsdp: str | None = None
    fsdp_config: dict[str, Any] | None = None

    @property
    def is_distributed(self) -> bool:
        """True if any non-DDP sharding is configured.

        Note: this reflects the *strategy*, not whether torchrun is active.
        Callers wanting "am I in a multi-process run?" should use
        src.training.distributed.get_distributed_info().is_distributed instead.
        """
        return self.preset != DistributedPreset.DDP


# FSDP auto-wrap config: wrap by transformer_layer_cls so FSDP shards per layer
# group. HF Trainer merges this with its own defaults. Adjust transformer_layer_names
# per model family — common defaults cover Llama/Qwen/Mistral architecture.
def _default_fsdp_wrap_config() -> dict[str, Any]:
    """Default FSDP auto-wrap config (transformer-layer-based wrapping)."""
    return {
        # Wrap at the decoder-block granularity. For Llama/Qwen/Mistral-style
        # decoder-only models this is typically the transformer layer class.
        # Extend the list for other architectures.
        "transformer_layer_cls_to_wrap": [
            "LlamaDecoderLayer",
            "Qwen2DecoderLayer",
            "Qwen3DecoderLayer",
            "MistralDecoderLayer",
            "GemmaDecoderLayer",
        ],
        "min_num_params": 1_000_000,  # don't bother wrapping tiny layers
        # Full-shard by default: params + grads + optimizer state sharded.
        # This matches the `fsdp="full_shard"` setting.
        "xla": False,
        "xla_fsdp_v2": False,
        "xla_fsdp_grad_ckpt": False,
    }


# HF Trainer `fsdp=` string values for each FSDP preset.
_FSDP_MODE = {
    DistributedPreset.FSDP_FULL: "full_shard",
    DistributedPreset.FSDP_GRAD: "sharded_grad_scaled",
}


def _deepspeed_path(preset: DistributedPreset) -> str:
    """Locate the DeepSpeed JSON for a ZeRO preset. Fails loudly if missing."""
    filename = f"{preset.value}.json"
    path = _CONFIGS_DIR / filename
    if not path.exists():
        raise FileNotFoundError(
            f"DeepSpeed config '{filename}' for preset {preset.name} not found at {path}. "
            "Packaging bug — JSONs should ship under config/distributed/deepspeed_configs/."
        )
    return str(path)


def resolve_distributed_config(
    preset_name: str | None = None,
    *,
    deepspeed_path: str | None = None,
    fsdp_mode: str | None = None,
) -> ResolvedDistributedConfig:
    """Resolve a distributed strategy from a preset name or explicit overrides.

    Resolution order:
      1. If `deepspeed_path` (a JSON path) is given, use DeepSpeed directly.
      2. Else if `fsdp_mode` (a HF `fsdp=` string) is given, use FSDP directly.
      3. Else if `preset_name` is given, map via DistributedPreset.
      4. Else default to DDP (no sharding).

    Args:
        preset_name: A DistributedPreset value (e.g. "fsdp_full", "zero_stage_2").
        deepspeed_path: Direct path to a DeepSpeed JSON.
        fsdp_mode: Direct HF Trainer fsdp string ("full_shard" / "sharded_grad_scaled" / ...).

    Returns:
        ResolvedDistributedConfig ready to feed into TrainingArguments.
    """
    # 1. Explicit DeepSpeed path wins.
    if deepspeed_path:
        if not os.path.exists(deepspeed_path):
            raise FileNotFoundError(f"DeepSpeed config not found: {deepspeed_path}")
        return ResolvedDistributedConfig(
            preset=DistributedPreset.ZERO_2,  # closest semantic match
            deepspeed_config=deepspeed_path,
        )

    # 2. Explicit FSDP mode.
    if fsdp_mode:
        preset = (
            DistributedPreset.FSDP_FULL
            if fsdp_mode == "full_shard"
            else DistributedPreset.FSDP_GRAD
        )
        return ResolvedDistributedConfig(
            preset=preset,
            fsdp=fsdp_mode,
            fsdp_config=_default_fsdp_wrap_config(),
        )

    # 3. Preset name.
    if preset_name:
        try:
            preset = DistributedPreset(preset_name)
        except ValueError:
            valid = ", ".join(p.value for p in DistributedPreset)
            raise ValueError(
                f"Unknown distributed preset '{preset_name}'. Valid: {valid}"
            ) from None

        if preset == DistributedPreset.DDP:
            return ResolvedDistributedConfig(preset=preset)
        if preset.is_fsdp:
            return ResolvedDistributedConfig(
                preset=preset,
                fsdp=_FSDP_MODE[preset],
                fsdp_config=_default_fsdp_wrap_config(),
            )
        # DeepSpeed ZeRO presets
        return ResolvedDistributedConfig(
            preset=preset,
            deepspeed_config=_deepspeed_path(preset),
        )

    # 4. Default: pure DDP.
    return ResolvedDistributedConfig(preset=DistributedPreset.DDP)


# Back-compat shim for the previous single-purpose API.
def get_deepspeed_config_path(preset: DistributedPreset) -> str | None:
    """Legacy helper: return DeepSpeed JSON path for a preset, or None for non-DS presets."""
    if not preset.is_deepspeed:
        return None
    return _deepspeed_path(preset)


def resolve_deepspeed_config(
    preset_name: str | None = None,
    explicit_path: str | None = None,
) -> str | None:
    """Legacy helper: resolve a DeepSpeed JSON path (None for DDP/FSDP)."""
    resolved = resolve_distributed_config(
        preset_name=preset_name,
        deepspeed_path=explicit_path,
    )
    return resolved.deepspeed_config
