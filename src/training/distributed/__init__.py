"""Distributed training helpers (DDP / DeepSpeed) for the QLoRA platform.

This package provides a thin detection + logging layer on top of what the
HuggingFace `Trainer` already does natively:

- `get_distributed_info()` inspects the environment variables that `torchrun`
  / `accelerate launch` inject (`LOCAL_RANK`, `RANK`, `WORLD_SIZE`) so callers
  can branch on "am I running distributed?" without importing torch.distributed.
- `rank_zero_only` / `is_rank_zero` make console output clean under multi-GPU.
- `setup_distributed()` pins the current CUDA device to `LOCAL_RANK` (idempotent;
  HF Trainer handles `init_process_group` itself).

Nothing here changes the training loop — it only informs *how* models are placed
and *what* gets printed. All functions are safe to call in single-GPU / CPU mode.
"""

from src.training.distributed.env import (
    DistributedInfo,
    get_distributed_info,
    is_rank_zero,
    rank_zero_only,
    setup_distributed,
)
from src.training.distributed.logger import get_rank_zero_console

__all__ = [
    "DistributedInfo",
    "get_distributed_info",
    "is_rank_zero",
    "rank_zero_only",
    "setup_distributed",
    "get_rank_zero_console",
]
