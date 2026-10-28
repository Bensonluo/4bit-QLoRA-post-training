"""Shared metric-formatting helpers for dashboard pages and adapters.

Policy: never fabricate data points. Eval JSON can hold None/strings/bools —
non-numeric values render as "—" (or a gap in charts), never as a defaulted 0.
"""

from __future__ import annotations

import math


def finite_number(value: object) -> bool:
    """Accept finite real JSON numbers, without coercing bools or huge integers."""
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def fmt_pct(value: object) -> str:
    """Format a percentage metric. Eval JSON can hold None/strings — never crash."""
    if finite_number(value) and isinstance(value, (int, float)) and finite_number(value * 100):
        return f"{value:.1%}"
    return "—"


def fmt_num(value: object) -> str:
    """Format a plain numeric metric, "—" for non-numeric values."""
    if finite_number(value) and isinstance(value, (int, float)):
        fmt = ".3f" if abs(value) < 1 else ".1f"
        return f"{value:{fmt}}"
    return "—"


def delta(a: object, b: object, pct: bool) -> str | None:
    """Signed delta only when both sides are numeric — never fabricate from a default 0."""
    if (
        finite_number(a)
        and finite_number(b)
        and isinstance(a, (int, float))
        and isinstance(b, (int, float))
    ):
        difference = b - a
        if not finite_number(difference):
            return None
        if pct:
            return f"{difference:+.1%}" if finite_number(difference * 100) else None
        fmt = ".3f" if abs(difference) < 1 else ".1f"
        return f"{difference:+{fmt}}"
    return None
