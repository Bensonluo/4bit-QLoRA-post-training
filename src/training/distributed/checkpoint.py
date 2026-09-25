"""Distributed Checkpoint (DCP) save/load for FSDP training.

The HF Trainer (transformers 5.9) has no native DCP switch — TrainingArguments
only carries FSDP1-style ``fsdp``/``fsdp_config``, and FSDP checkpoints in the
HF format have a long history of fragile resumes (per-rank shards, world-size
sensitivity). This module provides the TorchTitan-style decoupled path instead:

- ``save_dcp_checkpoint`` — writes a reshardable sharded checkpoint via
  ``torch.distributed.checkpoint.async_save`` (stages state to CPU, writes from
  a background thread, so the training loop is not blocked on disk I/O).
- ``load_dcp_checkpoint`` — loads in place; DCP resharding on load means a
  checkpoint saved on N GPUs can be resumed on M GPUs.

The HF-format artifacts the Trainer already writes are untouched: DCP
checkpoints are for mid-training resume, HF format for final export — the same
separation TorchTitan uses. Requires torch >= 2.4 (``async_save`` landed in
2.3 but the ``no_dist`` kwarg our calls rely on only exists from 2.4);
verified against 2.11.
"""

from __future__ import annotations

from concurrent.futures import Future
from pathlib import Path
from typing import Any

_DCP_AVAILABLE: bool | None = None


def dcp_available() -> bool:
    """True when ``torch.distributed.checkpoint`` is importable (torch >= 2.4).

    Result is cached — this is probed before every guarded call site.
    """
    global _DCP_AVAILABLE
    if _DCP_AVAILABLE is None:
        try:
            import torch.distributed.checkpoint  # noqa: F401

            _DCP_AVAILABLE = True
        except ImportError:
            _DCP_AVAILABLE = False
    return _DCP_AVAILABLE


def save_dcp_checkpoint(
    model: Any,
    path: str | Path,
    *,
    optimizer: Any = None,
    extra_state: dict[str, Any] | None = None,
    asynchronous: bool = True,
) -> Future | None:
    """Write a reshardable DCP checkpoint of model (and optionally optimizer).

    Uses FSDP-aware ``get_state_dict`` so sharded parameters stay sharded
    (no rank gathers the full model). ``extra_state`` (epoch, RNG state, ...)
    rides along under the ``"extra"`` key.

    Args:
        model: The (possibly FSDP-wrapped) module to checkpoint.
        path: Checkpoint directory (created by DCP).
        optimizer: Optional optimizer whose state should be resumable.
        extra_state: Optional small non-tensor state saved alongside.
        asynchronous: True (default) stages to CPU and writes from a
            background thread — returns a Future to finalize. False writes
            synchronously (blocks) and returns None.

    Returns:
        The :class:`concurrent.futures.Future` from ``async_save`` when
        asynchronous, else None.

    Raises:
        RuntimeError: If torch lacks DCP support.
    """
    if not dcp_available():
        raise RuntimeError(
            "torch.distributed.checkpoint unavailable (needs torch >= 2.4); "
            "disable DCP checkpointing to use the Trainer's standard saves."
        )
    import torch
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import get_state_dict

    model_sd, optim_sd = get_state_dict(model, optimizer if optimizer is not None else ())
    state: dict[str, Any] = {"model": model_sd}
    if optimizer is not None:
        state["optimizer"] = optim_sd
    if extra_state is not None:
        state["extra"] = extra_state

    # Single-process contexts (tests, laptop dev) have no process group to
    # collectivize over — DCP's no_dist mode writes directly.
    no_dist = not torch.distributed.is_available() or not torch.distributed.is_initialized()
    if asynchronous:
        response = dcp.async_save(state, checkpoint_id=str(path), no_dist=no_dist)
        # The default THREAD checkpointer yields a plain Future; a PROCESS
        # checkpointer yields an AsyncSaveResponse instead — normalize to the
        # future that signals the write landed (upload_completion).
        if not isinstance(response, Future):
            return response.upload_completion
        return response
    dcp.save(state, checkpoint_id=str(path), no_dist=no_dist)
    return None


def finalize_dcp_save(future: Future | None) -> None:
    """Block until an async DCP save lands; surface write failures loudly."""
    if future is not None:
        future.result()


def load_dcp_checkpoint(
    model: Any,
    path: str | Path,
    *,
    optimizer: Any = None,
    extra_state: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Load a DCP checkpoint in place into ``model`` (and ``optimizer``).

    The state dict acts as a load template: DCP fills tensors in place and
    resharding (N save ranks -> M load ranks) is handled internally.
    When ``extra_state`` is given as a template (same keys as saved), the
    loaded values are returned.

    Raises:
        RuntimeError: If torch lacks DCP support.
    """
    if not dcp_available():
        raise RuntimeError(
            "torch.distributed.checkpoint unavailable (needs torch >= 2.4); "
            "disable DCP checkpointing to use the Trainer's standard saves."
        )
    import torch
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict

    model_sd, optim_sd = get_state_dict(model, optimizer if optimizer is not None else ())
    state: dict[str, Any] = {"model": model_sd}
    if optimizer is not None:
        state["optimizer"] = optim_sd
    if extra_state is not None:
        state["extra"] = extra_state

    no_dist = not torch.distributed.is_available() or not torch.distributed.is_initialized()
    dcp.load(state, checkpoint_id=str(path), no_dist=no_dist)
    set_state_dict(
        model,
        optimizer if optimizer is not None else (),
        model_state_dict=state["model"],
        optim_state_dict=state.get("optimizer", {}),
    )
    return state.get("extra")
