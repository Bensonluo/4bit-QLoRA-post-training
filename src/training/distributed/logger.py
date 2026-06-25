"""Rank-aware console logging for distributed runs.

Under DDP/DeepSpeed, every rank prints identical messages, which floods the log
and makes benchmark timing unreadable. `get_rank_zero_console()` returns a wrapper
that only prints on rank 0; on other ranks the calls are silent no-ops.

We wrap the project's existing `rich` Console rather than replacing it, so all
existing `[green]...[/green]` markup keeps working.
"""

from typing import Any

from src.training.distributed.env import is_rank_zero
from src.utils.logging import console as _global_console


class _RankZeroConsole:
    """Proxy that forwards calls to the rich Console only on rank 0.

    Implements just enough of the Console interface (`.print()`) for our usage.
    Any other attribute access is forwarded to the underlying console so that,
    e.g., `console.status(...)` still works on rank 0 (and is skipped elsewhere).
    """

    def __init__(self, real_console: Any):
        self._real = real_console

    def print(self, *args, **kwargs):
        if is_rank_zero():
            return self._real.print(*args, **kwargs)
        return None

    def __getattr__(self, name: str) -> Any:
        # Forward non-print access (e.g. .status, .rule) to rank 0 only.
        # On non-zero ranks, return a callable no-op / the attribute itself.
        if is_rank_zero():
            return getattr(self._real, name)

        attr = getattr(self._real, name)
        if callable(attr):

            def _noop(*_args, **_kwargs):
                return None

            return _noop
        return attr


def get_rank_zero_console() -> _RankZeroConsole:
    """Return a console proxy that prints only on rank 0.

    Use this in trainer/loader code paths that may run under torchrun:

        from src.training.distributed import get_rank_zero_console
        console = get_rank_zero_console()
        console.print("[green]Training started[/green]")
    """
    return _RankZeroConsole(_global_console)
