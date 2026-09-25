"""Unit tests for custom training callbacks (src/training/callbacks.py).

No GPU, no real Trainer — state/control are lightweight stand-ins satisfying
the attribute surface the callbacks touch.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

from src.training.callbacks import (
    CheckpointCallback,
    EarlyStoppingCallback,
    LossCallback,
    MemoryMonitorCallback,
    ProgressCallback,
)


def make_state(
    global_step: int = 0, log_history: list[Any] | None = None, output_dir: str = "/tmp/out"
) -> SimpleNamespace:
    return SimpleNamespace(
        global_step=global_step,
        log_history=log_history if log_history is not None else [],
        output_dir=output_dir,
    )


class TestProgressCallback:
    def test_step_end_prints_every_50_steps(self) -> None:
        cb = ProgressCallback()
        with (
            patch.object(cb, "start_time", 0.0),
            patch("src.training.callbacks.console.print") as mock_print,
        ):
            cb.on_step_end(None, make_state(global_step=50, log_history=[{"loss": 2.0}]), None)
        assert mock_print.called
        assert cb.last_step == 50

    def test_step_end_skips_small_deltas(self) -> None:
        cb = ProgressCallback()
        cb.last_step = 50
        with patch("src.training.callbacks.console.print") as mock_print:
            cb.on_step_end(None, make_state(global_step=60), None)
        mock_print.assert_not_called()
        assert cb.last_step == 50  # not updated when skipped

    def test_train_begin_records_start_time(self) -> None:
        cb = ProgressCallback()
        with patch("src.training.callbacks.console.print"):
            cb.on_train_begin(None, make_state(), None)
        assert cb.start_time is not None


class TestLossCallback:
    def test_appends_on_matching_step(self) -> None:
        cb = LossCallback(log_steps=10)
        cb.on_log(None, make_state(global_step=10), None, logs={"loss": 1.5})
        assert cb.get_losses() == [(10, 1.5)]

    def test_skips_non_matching_step(self) -> None:
        cb = LossCallback(log_steps=10)
        cb.on_log(None, make_state(global_step=15), None, logs={"loss": 1.5})
        assert cb.get_losses() == []

    def test_none_logs_is_safe(self) -> None:
        cb = LossCallback()
        cb.on_log(None, make_state(global_step=10), None, logs=None)
        assert cb.get_losses() == []


class TestEarlyStoppingCallback:
    def test_first_eval_sets_baseline_without_stopping(self) -> None:
        cb = EarlyStoppingCallback()
        control = SimpleNamespace(should_training_stop=False)
        cb.on_evaluate(None, make_state(), control, metrics={"eval_loss": 1.0})
        assert cb.best_metric == 1.0
        assert control.should_training_stop is False

    def test_stops_after_patience_exhausted(self) -> None:
        cb = EarlyStoppingCallback(early_stopping_patience=2)
        control = SimpleNamespace(should_training_stop=False)
        cb.on_evaluate(None, make_state(), control, metrics={"eval_loss": 1.0})
        cb.on_evaluate(None, make_state(), control, metrics={"eval_loss": 1.05})  # no imp. 1
        assert control.should_training_stop is False
        cb.on_evaluate(None, make_state(), control, metrics={"eval_loss": 1.10})  # no imp. 2
        assert control.should_training_stop is True
        assert cb.early_stopping_counter == 2

    def test_improvement_resets_counter(self) -> None:
        cb = EarlyStoppingCallback(early_stopping_patience=2)
        control = SimpleNamespace(should_training_stop=False)
        cb.on_evaluate(None, make_state(), control, metrics={"eval_loss": 1.0})
        cb.on_evaluate(None, make_state(), control, metrics={"eval_loss": 1.1})
        cb.on_evaluate(None, make_state(), control, metrics={"eval_loss": 0.5})  # improve
        assert cb.early_stopping_counter == 0
        assert cb.best_metric == 0.5

    def test_threshold_blocks_marginal_gains(self) -> None:
        cb = EarlyStoppingCallback(early_stopping_threshold=0.1)
        control = SimpleNamespace(should_training_stop=False)
        cb.on_evaluate(None, make_state(), control, metrics={"eval_loss": 1.0})
        cb.on_evaluate(None, make_state(), control, metrics={"eval_loss": 0.95})  # < threshold
        assert cb.early_stopping_counter == 1
        assert cb.best_metric == 1.0  # unchanged

    def test_missing_eval_loss_is_ignored(self) -> None:
        cb = EarlyStoppingCallback()
        cb.on_evaluate(
            None,
            make_state(),
            SimpleNamespace(should_training_stop=False),
            metrics={"accuracy": 0.9},
        )
        assert cb.best_metric is None


class TestMemoryMonitorCallback:
    def test_non_matching_step_is_noop(self) -> None:
        cb = MemoryMonitorCallback(log_steps=100)
        control = SimpleNamespace()
        with patch("src.training.callbacks.console.print") as mock_print:
            result = cb.on_step_end(None, make_state(global_step=50), control)
        mock_print.assert_not_called()
        assert result is control

    def test_matching_step_does_not_crash_on_cpu(self) -> None:
        cb = MemoryMonitorCallback(log_steps=100)
        control = SimpleNamespace()
        with patch("src.training.callbacks.console.print"):
            result = cb.on_step_end(None, make_state(global_step=100), control)
        assert result is control


class TestCheckpointCallback:
    def test_first_metric_requests_save(self) -> None:
        cb = CheckpointCallback()
        control = SimpleNamespace(should_save=False)
        with patch("src.training.callbacks.console.print"):
            cb.on_evaluate(None, make_state(), control, metrics={"eval_loss": 1.0})
        assert control.should_save is True
        assert cb.best_metric == 1.0

    def test_improvement_requests_save_again(self) -> None:
        cb = CheckpointCallback()
        with patch("src.training.callbacks.console.print"):
            control1 = SimpleNamespace(should_save=False)
            cb.on_evaluate(None, make_state(), control1, metrics={"eval_loss": 1.0})
            control2 = SimpleNamespace(should_save=False)
            cb.on_evaluate(None, make_state(), control2, metrics={"eval_loss": 0.8})
        assert control2.should_save is True

    def test_no_improvement_leaves_save_off(self) -> None:
        cb = CheckpointCallback()
        with patch("src.training.callbacks.console.print"):
            cb.on_evaluate(
                None, make_state(), SimpleNamespace(should_save=False), metrics={"eval_loss": 1.0}
            )
            control = SimpleNamespace(should_save=False)
            cb.on_evaluate(None, make_state(), control, metrics={"eval_loss": 1.2})
        assert control.should_save is False
        assert cb.best_metric == 1.0  # baseline kept

    def test_greater_is_better_flips_comparison(self) -> None:
        cb = CheckpointCallback(metric_for_best="eval_acc", greater_is_better=True)
        with patch("src.training.callbacks.console.print"):
            cb.on_evaluate(
                None, make_state(), SimpleNamespace(should_save=False), metrics={"eval_acc": 0.5}
            )
            control = SimpleNamespace(should_save=False)
            cb.on_evaluate(None, make_state(), control, metrics={"eval_acc": 0.9})
        assert control.should_save is True

    def test_missing_metric_is_ignored(self) -> None:
        cb = CheckpointCallback()
        control = SimpleNamespace(should_save=False)
        with patch("src.training.callbacks.console.print"):
            cb.on_evaluate(None, make_state(), control, metrics={"other": 1.0})
        assert control.should_save is False
        assert cb.best_metric is None

    def test_last_strategy_is_explicit_noop(self) -> None:
        cb = CheckpointCallback(save_strategy="last")
        control = SimpleNamespace(should_save=False)
        with patch("src.training.callbacks.console.print"):
            cb.on_evaluate(None, make_state(), control, metrics={"eval_loss": 0.5})
        assert control.should_save is False
        assert cb.best_metric is None


class TestProgressCallbackTrainEnd:
    def test_prints_total_time(self) -> None:
        cb = ProgressCallback()
        cb.start_time = None  # never began — elapsed falls back to 0
        with patch("src.training.callbacks.console.print") as mock_print:
            cb.on_train_end(None, make_state(), None)
        printed = " ".join(str(c.args[0]) for c in mock_print.call_args_list)
        assert "Training Complete" in printed
        assert "0.0 minutes" in printed

    def test_elapsed_measured_from_start_time(self) -> None:
        import time as time_mod

        cb = ProgressCallback()
        cb.start_time = time_mod.time() - 120  # "2 minutes ago"
        with patch("src.training.callbacks.console.print") as mock_print:
            cb.on_train_end(None, make_state(), None)
        printed = " ".join(str(c.args[0]) for c in mock_print.call_args_list)
        assert "2.0 minutes" in printed


class TestMemoryMonitorPlatforms:
    def test_mps_branch_reports_unified_memory(self) -> None:
        cb = MemoryMonitorCallback(log_steps=1)
        with (
            patch("torch.cuda.is_available", return_value=False),
            patch("torch.backends.mps.is_available", return_value=True),
            patch("torch.mps.current_allocated_memory", return_value=2 * 1024**3),
            patch("src.training.callbacks.console.print") as mock_print,
        ):
            cb.on_step_end(None, make_state(global_step=1), None)
        printed = " ".join(str(c.args[0]) for c in mock_print.call_args_list)
        assert "MPS Memory: 2.00GB" in printed

    def test_probe_failure_is_silently_ignored(self) -> None:
        cb = MemoryMonitorCallback(log_steps=1)
        with (
            patch("torch.cuda.is_available", side_effect=RuntimeError("driver gone")),
            patch("src.training.callbacks.console.print") as mock_print,
        ):
            cb.on_step_end(None, make_state(global_step=1), None)
        mock_print.assert_not_called()  # exception swallowed, nothing printed

    def test_cuda_branch_reports_vram(self) -> None:
        cb = MemoryMonitorCallback(log_steps=1)
        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.memory_allocated", return_value=2 * 1024**3),
            patch("torch.cuda.memory_reserved", return_value=3 * 1024**3),
            patch("src.training.callbacks.console.print") as mock_print,
        ):
            cb.on_step_end(None, make_state(global_step=1), None)
        printed = " ".join(str(c.args[0]) for c in mock_print.call_args_list)
        assert "VRAM: 2.00GB allocated, 3.00GB reserved" in printed


class TestMFUCallback:
    """MFU observability via TRL >= 1.4 helpers (compute_flops_per_token et al.)."""

    @staticmethod
    def _tiny_qwen3() -> Any:
        from transformers import Qwen3Config

        return Qwen3Config(
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            vocab_size=128,
        )

    def test_init_enabled_on_supported_config(self) -> None:
        from src.training.callbacks import MFUCallback

        cb = MFUCallback(self._tiny_qwen3(), seq_len=512, tokens_per_step=1000)
        assert cb._fpt is not None and cb._fpt > 0
        assert cb.history == []

    def test_init_disabled_gracefully_on_unsupported_config(self) -> None:
        # Qwen2-family configs carry no head_dim — the helper raises, the
        # callback must disable itself, not crash the training run.
        from src.training.callbacks import MFUCallback

        cb = MFUCallback(SimpleNamespace(), seq_len=512, tokens_per_step=1000)
        assert cb._fpt is None
        # on_log is a no-op when disabled.
        logs: dict[str, Any] = {}
        cb.on_log(None, SimpleNamespace(global_step=10), None, logs=logs)
        assert logs == {} and cb.history == []

    def test_on_log_records_windowed_mfu(self) -> None:
        from src.training.callbacks import MFUCallback

        cb = MFUCallback(self._tiny_qwen3(), seq_len=512, tokens_per_step=1000)
        # t0=0 at train begin; on_log reads now=10 → elapsed 10s, 10 steps.
        with patch("src.training.callbacks.time.monotonic", side_effect=[0, 10, 10]):
            cb.on_train_begin(None, SimpleNamespace(global_step=0), None)
            logs: dict[str, Any] = {}
            cb.on_log(None, SimpleNamespace(global_step=10), None, logs=logs)

        assert len(cb.history) == 1
        step, mfu = cb.history[0]
        assert step == 10
        assert 0 < mfu < 100  # percent, computed from 1000 padded tok/s
        assert logs["train_mfu_percent"] == mfu

    def test_on_log_skips_zero_step_delta(self) -> None:
        from src.training.callbacks import MFUCallback

        cb = MFUCallback(self._tiny_qwen3(), seq_len=512, tokens_per_step=1000)
        with patch("src.training.callbacks.time.monotonic", side_effect=[0, 5]):
            cb.on_train_begin(None, SimpleNamespace(global_step=0), None)
            cb.on_log(None, SimpleNamespace(global_step=0), None, logs={})
        assert cb.history == []  # no division by zero, no bogus entry

    def test_custom_peak_lowers_mfu_denominator_effect(self) -> None:
        # Same throughput on a 1e14 peak vs TRL's 9.895e14 H100 default must
        # yield a proportionally higher MFU percent.
        from src.training.callbacks import MFUCallback

        default_cb = MFUCallback(self._tiny_qwen3(), seq_len=512, tokens_per_step=1000)
        slow_gpu_cb = MFUCallback(
            self._tiny_qwen3(), seq_len=512, tokens_per_step=1000, peak_flops_per_device=1e14
        )
        for cb in (default_cb, slow_gpu_cb):
            with patch("src.training.callbacks.time.monotonic", side_effect=[0, 10, 10]):
                cb.on_train_begin(None, SimpleNamespace(global_step=0), None)
                cb.on_log(None, SimpleNamespace(global_step=10), None, logs={})

        assert slow_gpu_cb.history[0][1] > default_cb.history[0][1]

    def test_window_resets_between_log_points(self) -> None:
        from src.training.callbacks import MFUCallback

        cb = MFUCallback(self._tiny_qwen3(), seq_len=512, tokens_per_step=1000)
        # Two windows: 0→10s/10 steps, then 10→20s/10 steps.
        with patch("src.training.callbacks.time.monotonic", side_effect=[0, 10, 10, 20, 20]):
            cb.on_train_begin(None, SimpleNamespace(global_step=0), None)
            cb.on_log(None, SimpleNamespace(global_step=10), None, logs={})
            cb.on_log(None, SimpleNamespace(global_step=20), None, logs={})

        assert [s for s, _ in cb.history] == [10, 20]


class TestMemoryMonitorHostFallback:
    def test_neither_cuda_nor_mps_is_silent(self, monkeypatch: Any) -> None:
        import torch

        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        if hasattr(torch.backends, "mps"):
            monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)

        cb = MemoryMonitorCallback(log_steps=10)
        control = SimpleNamespace()

        result = cb.on_step_end(args=None, state=make_state(global_step=10), control=control)

        # CPU-only host: neither memory branch fires; callback stays silent.
        assert result is control
