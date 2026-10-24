"""Show actual task state rather than treating unrelated global artifacts as a completed workflow."""

import sqlite3
from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest

from src.workbench.baseline_analysis import propose_baseline_analysis
from src.workbench.intake_service import IntakeService

ROOT = Path(__file__).resolve().parents[2]


def task(root):
    service = IntakeService(root / "outputs/workbench/intake")
    session = service.create(
        "识别客户咨询类别", "tickets.csv", "描述,答案\n杯子破损,质量\n何时发货,物流\n".encode()
    )
    session = service.apply_analysis(
        session,
        propose_baseline_analysis(session, target_column="答案"),
        model="deterministic-baseline",
    )
    return service, session


def boot(monkeypatch, tmp_path, display=False):
    from ui import config, queries, runtime

    monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(queries, "fetch_runs", lambda *a, **k: pd.DataFrame())
    monkeypatch.setattr(runtime, "DISPLAY_ONLY", display)
    return AppTest.from_file(str(ROOT / "ui/app.py"), default_timeout=45).run()


def test_unrelated_artifacts_do_not_certify_a_completed_business_task(monkeypatch, tmp_path):
    for name in [
        "outputs/wizard/set/train.json",
        "outputs/run/adapter_config.json",
        "domains/entity_matching/data/results/result.json",
    ]:
        p = tmp_path / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("{}")
    page = boot(monkeypatch, tmp_path)
    assert not page.exception
    assert not any("全链路已跑通" in x.value for x in page.success)
    assert not any("产物就绪" in x.value for x in page.markdown)
    assert not any(b.label == "继续任务" for b in page.button)


def test_existing_task_keeps_its_identity_when_opened(monkeypatch, tmp_path):
    service, session = task(tmp_path)
    page = boot(monkeypatch, tmp_path)
    next(b for b in page.button if b.label == "继续任务").click().run()
    assert not page.exception
    assert page.session_state["intake_id"] == session.session_id
    assert any(t.value == session.goal for t in page.title)
    assert service.load(session.session_id).revision == session.revision


def test_display_mode_browses_the_same_task_without_advancing_it(monkeypatch, tmp_path):
    service, session = task(tmp_path)
    page = boot(monkeypatch, tmp_path, display=True)
    next(b for b in page.button if b.label == "浏览示例任务").click().run()
    assert not page.exception
    assert any(t.value == session.goal for t in page.title)
    assert service.load(session.session_id).revision == session.revision
    assert any(b.disabled for b in page.button)


def test_corrupt_snapshot_does_not_hide_other_tasks(monkeypatch, tmp_path):
    service, _ = task(tmp_path)
    with sqlite3.connect(service.database) as db:
        db.execute("INSERT INTO sessions VALUES (?, ?, ?)", ("broken", 0, "not-json"))
    page = boot(monkeypatch, tmp_path)
    assert not page.exception
    assert any("1 个任务快照" in w.value for w in page.warning)
    assert any(b.label == "继续任务" for b in page.button)
