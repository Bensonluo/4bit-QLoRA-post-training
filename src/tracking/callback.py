"""MLflow training callback — passively logs metrics from HF Trainer."""

from __future__ import annotations

from typing import Any

try:
    from transformers.trainer_callback import TrainerCallback
except ImportError:
    TrainerCallback = object  # type: ignore[assignment,misc]


class MLflowTrainCallback(TrainerCallback):
    """Forwards HF Trainer events to MLflow.

    Passive: never sets control.should_* flags.
    Delegates all MLflow calls to the tracker (which no-ops if disabled).
    """

    def __init__(self, tracker: Any) -> None:
        super().__init__()
        self._tracker = tracker

    def on_train_begin(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
        model = kwargs.get("model")
        if model is not None:
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total = sum(p.numel() for p in model.parameters())
            self._tracker.log_metrics({
                "params/trainable": float(trainable),
                "params/total": float(total),
                "params/trainable_pct": 100.0 * trainable / total if total > 0 else 0.0,
            })
        return control

    def on_log(self, args: Any, state: Any, control: Any, logs: dict[str, Any] | None = None, **kwargs: Any) -> Any:
        if logs is None or not self._tracker.active:
            return control
        step = state.global_step
        metrics: dict[str, float] = {}
        for k, v in logs.items():
            if isinstance(v, (int, float)):
                metrics[k] = float(v)
        if metrics:
            self._tracker.log_metrics(metrics, step=step)
        return control

    def on_evaluate(self, args: Any, state: Any, control: Any, metrics: dict[str, Any] | None = None, **kwargs: Any) -> Any:
        if metrics is None or not self._tracker.active:
            return control
        eval_metrics: dict[str, float] = {}
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                eval_metrics[f"eval/{k}"] = float(v)
        if eval_metrics:
            self._tracker.log_metrics(eval_metrics, step=state.global_step)
        return control
