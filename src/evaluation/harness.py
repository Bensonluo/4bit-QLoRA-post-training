"""EleutherAI lm-evaluation-harness integration (guarded option #19).

Runs industry-standard benchmarks (HellaSwag, ARC, MMLU, GSM8K, ...) on a
merged model through ``lm_eval.simple_evaluate`` — the harness's recommended
programmatic entry point (see its ``docs/python-api.md``).

The dependency is opt-in: lm-eval 0.4.10 (2026-01) split model backends out
of the base wheel, so ``pip install -e ".[eval]"`` (``lm_eval[hf]``) adds the
evaluation stack while reusing the pinned torch/transformers/accelerate —
the base install is unchanged. Imports here stay lazy so the module (and
everything downstream) works without the extra installed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_INSTALL_HINT = (
    "lm-evaluation-harness not installed — install the evaluation stack: "
    'pip install -e ".[eval]"  (provides lm_eval[hf] >= 0.4.10)'
)


def harness_available() -> bool:
    """True when the lm-eval harness is importable."""
    try:
        import lm_eval  # noqa: F401
    except ImportError:
        return False
    return True


def run_harness_eval(
    model_path: str,
    tasks: list[str],
    *,
    num_fewshot: int | None = None,
    limit: int | float | None = None,
    batch_size: int | str = "auto",
    device: str | None = None,
    dtype: str = "auto",
    apply_chat_template: bool = False,
    output_path: Path | None = None,
) -> dict[str, Any]:
    """Evaluate ``model_path`` on standard benchmark ``tasks``.

    Thin wrapper over ``lm_eval.simple_evaluate`` with the ``hf`` model
    backend. Returns the harness's full result dict (``results`` /
    ``configs`` / ``versions`` / ``n-shot`` / ...) and, when ``output_path``
    is given, writes it as JSON (non-serializable values fall back to ``str``,
    mirroring the harness CLI's own serializer).
    """
    try:
        import lm_eval
    except ImportError as exc:
        raise RuntimeError(_INSTALL_HINT) from exc

    results: dict[str, Any] = lm_eval.simple_evaluate(
        model="hf",
        model_args=f"pretrained={model_path},dtype={dtype}",
        tasks=tasks,
        num_fewshot=num_fewshot,
        batch_size=batch_size,
        device=device,
        limit=limit,
        apply_chat_template=apply_chat_template,
    )
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(results, indent=2, default=str))
    return results


def summarize_results(results: dict[str, Any]) -> dict[str, dict[str, float | None]]:
    """Flatten harness results to ``{task: {metric: value, ...}}``.

    The raw harness keys pair each metric with a ``"<metric>,stderr"`` entry;
    those become ``<metric>_stderr`` siblings (``None`` when absent). Only
    numeric metrics are kept — bookkeeping keys such as ``alias`` are dropped,
    and a missing stderr stays ``None`` instead of being fabricated as 0.
    """
    summary: dict[str, dict[str, float | None]] = {}
    for task, metrics in results.get("results", {}).items():
        entry: dict[str, float | None] = {}
        for key, value in metrics.items():
            if key.endswith(",stderr"):
                continue
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                entry[key] = float(value)
                stderr = metrics.get(f"{key},stderr")
                entry[f"{key}_stderr"] = float(stderr) if isinstance(stderr, (int, float)) else None
        summary[task] = entry
    return summary
