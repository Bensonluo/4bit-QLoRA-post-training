"""Damaged timing and curve evidence must not crash task browsing."""

import json
import math
import sys

import pytest

from src.workbench.cost_summary import cost_lines, summarize_run_cost
from src.workbench.training_progress import (
    LOSS_HISTORY_FILENAME,
    _segment_mean,
    load_loss_history_with_health,
    loss_trend_lines,
)
from tests.unit.test_full_data import FULL, approved
from tests.unit.test_intake_display_readonly_ui import snapshot, unchanged

pytest_plugins = ["tests.unit.test_data_intake_ui"]


def test_loss_reader_distinguishes_absence_empty_and_bad_points(tmp_path):
    assert load_loss_history_with_health(tmp_path) == ([], False)
    path = tmp_path / LOSS_HISTORY_FILENAME
    path.write_text("")
    assert load_loss_history_with_health(tmp_path) == ([], True)
    path.write_text("[]")
    assert load_loss_history_with_health(tmp_path) == ([], False)
    path.write_text(
        json.dumps(
            [
                {"step": 1, "loss": 2.0},
                {"step": "broken", "loss": 1.0},
                {"step": 3, "loss": float("nan")},
                {"step": False, "loss": 1.0},
                {"step": 1e308, "loss": 1.0},
                {"step": 4, "loss": 0.5},
            ]
        )
    )
    assert load_loss_history_with_health(tmp_path) == (
        [{"step": 1.0, "loss": 2.0}, {"step": 4.0, "loss": 0.5}],
        True,
    )


def test_extreme_finite_loss_points_do_not_overflow_the_trend():
    points = [{"step": float(i), "loss": sys.float_info.max} for i in range(12)]
    assert math.isfinite(_segment_mean(points))
    lines = loss_trend_lines(points)
    assert "整体基本持平" in lines[0]
    assert "inf" not in " ".join(lines).lower()


@pytest.mark.parametrize(
    ("start", "end"),
    [
        ("2026-01-01T00:00:00", "2026-01-01T00:00:01+00:00"),
        (123, "2026-01-01T00:00:01+00:00"),
        ("2026-01-01T00:00:02+00:00", "2026-01-01T00:00:01+00:00"),
    ],
)
def test_incomparable_timing_does_not_invent_cost(start, end):
    result = summarize_run_cost({"started_at": start, "finished_at": end})
    assert result["duration_hours"] is None
    assert result["electricity_cost_estimated"] is None
    assert "无法比较" in "".join(cost_lines(result))


def test_display_page_isolates_corrupt_curve_and_incomparable_time(data_page, monkeypatch):
    import ui.runtime
    from ui.components import agent_settings

    service, _, page = data_page
    session = approved(service)
    session = service.validate_full_data(session.session_id, session.revision, "full.csv", FULL)
    session = service.confirm_full_data(session.session_id, session.revision)
    session = service.materialize_dataset(session.session_id, session.revision)
    root = service.root.parent / "training"
    run_id = "wb-" + "a" * 32
    directory = root / run_id
    output = directory / "model"
    output.mkdir(parents=True)
    (output / LOSS_HISTORY_FILENAME).write_text(
        json.dumps(
            [{"step": 1, "loss": 2.0}, {"step": "bad", "loss": 1.0}, {"step": 3, "loss": 0.5}]
        )
    )
    manifest = directory / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "run_id": run_id,
                "dataset_version": session.dataset.version,
                "config_digest": "fixture",
                "files": {},
            }
        )
    )
    (directory / "run.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "session_id": session.session_id,
                "status": "succeeded",
                "dataset_version": session.dataset.version,
                "model_path": "/tmp/base",
                "output_dir": str(output),
                "config_digest": "fixture",
                "config": {},
                "metrics": {"train_loss": 0.5},
                "artifacts": {"manifest": str(manifest)},
                "started_at": "2026-01-01T00:00:00",
                "finished_at": "2026-01-01T00:00:01+00:00",
            }
        )
    )
    monkeypatch.setattr(ui.runtime, "DISPLAY_ONLY", True)
    monkeypatch.setattr(agent_settings, "DISPLAY_ONLY", True)
    before = snapshot(service.root.parent.parent.parent)
    page.run()
    page.selectbox(key="intake_select").select(session.session_id).run()
    assert not page.exception, [item.message for item in page.exception]
    assert any("保留 2 个有效点" in item.value for item in page.warning)
    assert any("无法比较" in item.value for item in page.markdown)
    assert not any("整体在下降" in item.value for item in page.caption)
    assert any("本轮训练指标" in item.value for item in page.markdown)
    unchanged(service.root.parent.parent.parent, before)
