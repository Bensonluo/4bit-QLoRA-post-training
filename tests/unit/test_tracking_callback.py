"""Unit tests for the MLflow training callback (src/tracking/callback.py).

The tracker is a recording double — no MLflow backend needed.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any


class _FakeTracker:
    def __init__(self, active: bool = True) -> None:
        self.active = active
        self.metric_calls: list[tuple[dict[str, float], int | None]] = []

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        self.metric_calls.append((metrics, step))


class _FakeParam:
    def __init__(self, n: int, requires_grad: bool = True) -> None:
        self._n = n
        self.requires_grad = requires_grad

    def numel(self) -> int:
        return self._n


class _FakeModel:
    def __init__(self) -> None:
        self.parameters_return = [
            _FakeParam(100, requires_grad=True),
            _FakeParam(300, requires_grad=False),
        ]

    def parameters(self) -> list[_FakeParam]:
        return self.parameters_return


def _cb(tracker: _FakeTracker) -> Any:
    from src.tracking.callback import MLflowTrainCallback

    return MLflowTrainCallback(tracker)


class TestOnLog:
    def test_numeric_metrics_forwarded_with_step(self) -> None:
        tracker = _FakeTracker()
        control = SimpleNamespace(should_save=False)
        result = _cb(tracker).on_log(
            None,
            SimpleNamespace(global_step=12),
            control,
            logs={"loss": 0.5, "lr": 1e-4},
        )
        assert result is control
        assert tracker.metric_calls == [({"loss": 0.5, "lr": 0.0001}, 12)]

    def test_non_numeric_values_are_filtered(self) -> None:
        tracker = _FakeTracker()
        _cb(tracker).on_log(
            None,
            SimpleNamespace(global_step=1),
            SimpleNamespace(),
            logs={"loss": 0.5, "status": "ok", "tokens": [1, 2]},
        )
        assert tracker.metric_calls == [({"loss": 0.5}, 1)]

    def test_all_non_numeric_logs_nothing(self) -> None:
        tracker = _FakeTracker()
        _cb(tracker).on_log(
            None, SimpleNamespace(global_step=1), SimpleNamespace(), logs={"s": "x"}
        )
        assert tracker.metric_calls == []

    def test_inactive_tracker_is_silent(self) -> None:
        tracker = _FakeTracker(active=False)
        _cb(tracker).on_log(
            None, SimpleNamespace(global_step=1), SimpleNamespace(), logs={"loss": 0.5}
        )
        assert tracker.metric_calls == []

    def test_none_logs_is_safe(self) -> None:
        tracker = _FakeTracker()
        _cb(tracker).on_log(None, SimpleNamespace(global_step=1), SimpleNamespace())
        assert tracker.metric_calls == []


class TestOnEvaluate:
    def test_metrics_prefixed_with_eval(self) -> None:
        tracker = _FakeTracker()
        _cb(tracker).on_evaluate(
            None,
            SimpleNamespace(global_step=20),
            SimpleNamespace(),
            metrics={"eval_loss": 0.7, "epoch": 1.0},
        )
        assert tracker.metric_calls == [({"eval/eval_loss": 0.7, "eval/epoch": 1.0}, 20)]

    def test_guards_mirror_on_log(self) -> None:
        for kwargs in (
            {},  # metrics=None
            {"metrics": {"eval_loss": 0.5}},
        ):
            tracker = _FakeTracker(active=False)
            _cb(tracker).on_evaluate(
                None, SimpleNamespace(global_step=1), SimpleNamespace(), **kwargs
            )
            assert tracker.metric_calls == []

    def test_all_non_numeric_metrics_logs_nothing(self) -> None:
        tracker = _FakeTracker()
        _cb(tracker).on_evaluate(
            None,
            SimpleNamespace(global_step=1),
            SimpleNamespace(),
            metrics={"report": "text"},
        )
        assert tracker.metric_calls == []


class TestOnTrainBegin:
    def test_param_counts_logged(self) -> None:
        tracker = _FakeTracker()
        _cb(tracker).on_train_begin(None, SimpleNamespace(), SimpleNamespace(), model=_FakeModel())
        ((metrics, step),) = tracker.metric_calls
        assert metrics["params/trainable"] == 100.0
        assert metrics["params/total"] == 400.0
        assert metrics["params/trainable_pct"] == 25.0
        assert step is None

    def test_zero_total_params_is_safe(self) -> None:
        tracker = _FakeTracker()
        model = _FakeModel()
        model.parameters_return = []  # no params → total 0
        _cb(tracker).on_train_begin(None, SimpleNamespace(), SimpleNamespace(), model=model)
        ((metrics, _),) = tracker.metric_calls
        assert metrics["params/trainable_pct"] == 0.0

    def test_missing_model_logs_nothing(self) -> None:
        tracker = _FakeTracker()
        _cb(tracker).on_train_begin(None, SimpleNamespace(), SimpleNamespace())
        assert tracker.metric_calls == []

    def test_callback_is_passive_never_touches_control(self) -> None:
        tracker = _FakeTracker()
        control = SimpleNamespace(should_save=False, should_training_stop=False)
        _cb(tracker).on_log(None, SimpleNamespace(global_step=1), control, logs={"loss": 0.5})
        _cb(tracker).on_evaluate(
            None, SimpleNamespace(global_step=1), control, metrics={"eval_loss": 0.5}
        )
        _cb(tracker).on_train_begin(None, SimpleNamespace(), control, model=_FakeModel())
        assert control.should_save is False and control.should_training_stop is False


class TestTransformersImportFallback:
    def test_missing_trainer_callback_falls_back_to_object(self) -> None:
        """ImportError fallback: MLflowTrainCallback subclasses plain object."""
        import importlib

        import transformers.trainer_callback as tc

        import src.tracking.callback as callback_mod

        saved = tc.TrainerCallback
        try:
            del tc.TrainerCallback  # `from ... import TrainerCallback` → ImportError
            importlib.reload(callback_mod)
            assert callback_mod.TrainerCallback is object
        finally:
            tc.TrainerCallback = saved
            importlib.reload(callback_mod)  # restore normal state for other tests
        assert callback_mod.TrainerCallback is not object
