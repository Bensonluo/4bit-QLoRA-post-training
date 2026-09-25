"""Shared metric-formatting helpers for dashboard pages and adapters.

Policy: never fabricate data points. Eval JSON can hold None/strings/bools —
non-numeric values render as "—" (or a gap in charts), never as a defaulted 0.
"""

from __future__ import annotations


def fmt_pct(value: object) -> str:
    """Format a percentage metric. Eval JSON can hold None/strings — never crash."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return f"{value:.1%}"
    return "—"


def fmt_num(value: object) -> str:
    """Format a plain numeric metric, "—" for non-numeric values."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        fmt = ".3f" if abs(value) < 1 else ".1f"
        return f"{value:{fmt}}"
    return "—"


def delta(a: object, b: object, pct: bool) -> str | None:
    """Signed delta only when both sides are numeric — never fabricate from a default 0."""
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if pct:
            return f"{b - a:+.1%}"
        fmt = ".3f" if abs(b - a) < 1 else ".1f"
        return f"{b - a:+{fmt}}"
    return None
