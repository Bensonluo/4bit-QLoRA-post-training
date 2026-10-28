"""Exercise the real Trainer logging callback and live-curve publication."""

import json
from types import SimpleNamespace

from transformers import Trainer
from transformers.trainer_callback import CallbackHandler, TrainerControl, TrainerState

from src.workbench.live_loss import LiveLossWriter
from src.workbench.training_progress import LOSS_HISTORY_FILENAME, load_loss_history_with_health


def test_trainer_log_publishes_actual_global_step_without_step_in_logs(tmp_path):
    writer = LiveLossWriter(tmp_path)
    trainer = Trainer.__new__(Trainer)
    trainer.state = TrainerState(global_step=7)
    trainer.args = SimpleNamespace(include_num_input_tokens_seen="no")
    trainer.control = TrainerControl()
    trainer.callback_handler = CallbackHandler([writer], None, None, None, None)

    # Trainer stores the step in log_history, but passes logs without it to callbacks.
    trainer.log({"loss": 1.25, "learning_rate": 0.0001})
    trainer.state.global_step = 8
    trainer.log({"eval_loss": 0.9})
    trainer.state.global_step = 9
    trainer.log({"loss": 1.1})

    assert [row["step"] for row in trainer.state.log_history] == [7, 8, 9]
    assert json.loads((tmp_path / LOSS_HISTORY_FILENAME).read_text()) == [
        {"step": 7.0, "loss": 1.25, "learning_rate": 0.0001},
        {"step": 9.0, "loss": 1.1},
    ]
    assert load_loss_history_with_health(tmp_path) == (
        [{"step": 7.0, "loss": 1.25}, {"step": 9.0, "loss": 1.1}],
        False,
    )


def test_failed_live_update_preserves_last_complete_curve_and_training_continues(
    tmp_path, monkeypatch
):
    from pathlib import Path

    writer = LiveLossWriter(tmp_path)
    writer.on_log(None, SimpleNamespace(global_step=1), None, logs={"loss": 2.0})
    original = (tmp_path / LOSS_HISTORY_FILENAME).read_bytes()
    real_replace = Path.replace

    def fail_replace(path, target):
        if target == tmp_path / LOSS_HISTORY_FILENAME:
            raise OSError("simulated write failure")
        return real_replace(path, target)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "replace", fail_replace)
        writer.on_log(None, SimpleNamespace(global_step=2), None, logs={"loss": 1.5})
    assert (tmp_path / LOSS_HISTORY_FILENAME).read_bytes() == original
    assert sorted(path.name for path in tmp_path.iterdir()) == [LOSS_HISTORY_FILENAME]

    writer.on_log(None, SimpleNamespace(global_step=3), None, logs={"loss": 1.0})
    assert [p["step"] for p in json.loads((tmp_path / LOSS_HISTORY_FILENAME).read_text())] == [
        1.0,
        2.0,
        3.0,
    ]


def test_bad_optional_logging_does_not_interrupt_training(tmp_path):
    writer = LiveLossWriter(tmp_path)
    writer.on_log(None, SimpleNamespace(global_step=1), None, logs={"loss": "invalid"})
    writer.on_log(
        None, SimpleNamespace(global_step=2), None, logs={"loss": 1.0, "learning_rate": {}}
    )
    assert load_loss_history_with_health(tmp_path) == ([{"step": 2.0, "loss": 1.0}], False)
