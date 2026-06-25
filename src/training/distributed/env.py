"""Distributed environment detection for torchrun / accelerate launch.

`torchrun` (and `accelerate launch`) inject these environment variables into
every worker process:

    LOCAL_RANK  — device index of this process (0..N-1)
    RANK        — global rank of this process
    WORLD_SIZE  — total number of processes
    MASTER_ADDR / MASTER_PORT — rendezvous endpoint

When none of these are set, we are in plain single-process mode (LOCAL_RANK=-1,
WORLD_SIZE=1). All functions here degrade gracefully to that case, so callers
can write one code path for both single-GPU and multi-GPU.
"""

import functools
import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class DistributedInfo:
    """Snapshot of the distributed runtime context.

    Attributes:
        is_distributed: True when WORLD_SIZE > 1 (i.e. launched via torchrun/accelerate
            with more than one worker). A single-process `python script.py` run is False.
        local_rank: Device index for this process. -1 when not distributed.
        rank: Global rank. 0 for the main/coordinator process; 0 when not distributed.
        world_size: Total worker count. 1 when not distributed.
        is_main_process: Convenience flag — True on rank 0 (the only process that
            should print to stdout, save checkpoints, run evaluation, etc.).
    """

    is_distributed: bool
    local_rank: int
    rank: int
    world_size: int
    is_main_process: bool


def get_distributed_info() -> DistributedInfo:
    """Inspect env vars to detect the distributed context.

    This is intentionally env-var based (not `torch.distributed.is_initialized()`)
    so it works *before* `init_process_group` runs — e.g. at model-loading time.
    """
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    # Single-process run: torchrun with --nproc_per_node=1 still sets LOCAL_RANK=0,
    # but HF Trainer treats world_size==1 as non-distributed. Match that convention
    # so device_map="auto" is preserved for the common single-GPU case.
    is_distributed = world_size > 1

    return DistributedInfo(
        is_distributed=is_distributed,
        local_rank=local_rank,
        rank=rank,
        world_size=world_size,
        is_main_process=(rank == 0),
    )


def is_rank_zero() -> bool:
    """True on the main process (rank 0), or always True when not distributed."""
    return get_distributed_info().is_main_process


def rank_zero_only(func: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator: only run `func` on rank 0; no-op on other ranks.

    Useful for wrapping expensive logging or one-time setup that would otherwise
    be redundantly executed N times across GPUs.
    """

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        if is_rank_zero():
            return func(*args, **kwargs)
        return None

    return wrapper


def setup_distributed() -> DistributedInfo:
    """Pin the current process to its CUDA device.

    HF Trainer / DeepSpeed handle `init_process_group` internally — this function
    does NOT call it. Its only job is `torch.cuda.set_device(local_rank)` so that
    subsequent `tensor.to("cuda")` lands on the right GPU under DDP/ZeRO.

    Safe to call in single-GPU mode (no-op when local_rank < 0).
    """
    info = get_distributed_info()
    if info.is_distributed and info.local_rank >= 0 and torch.cuda.is_available():
        torch.cuda.set_device(info.local_rank)
    return info
