"""Distributed training configuration: FSDP + DeepSpeed presets and resolver.

Industry-standard 2026 stack (resource-unconstrained, open-source):
  - FSDP (PyTorch-native) is the DEFAULT multi-GPU strategy.
  - DeepSpeed ZeRO-3 + offload for extreme scale beyond FSDP.
  - DDP as the simple replication baseline.

Public API:
    DistributedPreset            — enum: ddp / fsdp_full / fsdp_grad / zero_1/2/3
    ResolvedDistributedConfig    — frozen result: {deepspeed_config | fsdp | fsdp_config}
    resolve_distributed_config() — main resolver (preset name OR explicit overrides)
    get_deepspeed_config_path()  — legacy: DeepSpeed JSON path only
    resolve_deepspeed_config()   — legacy: DeepSpeed JSON path only
"""

from config.distributed.presets import (
    DistributedPreset,
    ResolvedDistributedConfig,
    get_deepspeed_config_path,
    resolve_deepspeed_config,
    resolve_distributed_config,
)

__all__ = [
    "DistributedPreset",
    "ResolvedDistributedConfig",
    "resolve_distributed_config",
    "get_deepspeed_config_path",
    "resolve_deepspeed_config",
]
