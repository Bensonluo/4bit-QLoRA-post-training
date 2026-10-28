"""Damaged runs are visible as warnings without hiding healthy task evidence."""

import json
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from src.workbench.training_runs import TrainingRunService

ROOT = Path(__file__).resolve().parents[2]


def put_run(root, key, session, **extra):
    identity = "wb-" + key * 32
    directory = root / identity
    directory.mkdir(parents=True)
    record = {
        "run_id": identity,
        "session_id": session,
        "status": "blocked",
        "model_path": "/missing/model",
        "issues": [{"severity": "blocking", "message": "缺少本地模型"}],
        "preflight": None,
        "config": None,
        "metrics": {},
        "artifacts": {},
        **extra,
    }
    (directory / "run.json").write_text(json.dumps(record, ensure_ascii=False))
    return identity


def snapshot(root):
    return {
        str(path.relative_to(root)): path.read_bytes() if path.is_file() else None
        for path in root.rglob("*")
    }


def test_unknown_owner_damage_does_not_hide_other_runs_and_readonly_does_not_write(tmp_path):
    root = tmp_path / "training"
    healthy = put_run(root, "a", "task-a")
    damaged = put_run(root, "b", "task-b")
    (root / damaged / "run.json").write_text("{damaged")
    before = snapshot(tmp_path)
    service = TrainingRunService(root, read_only=True)
    with pytest.raises(ValueError):
        service.list_runs("task-a")
    runs, bad = service.list_runs_tolerant("task-a")
    assert [run["run_id"] for run in runs] == [healthy]
    assert bad == 1
    assert snapshot(tmp_path) == before


def test_unrelated_known_owner_is_skipped_before_reading_a_bad_receipt(tmp_path):
    root = tmp_path / "training"
    healthy = put_run(root, "a", "task-a")
    other = put_run(root, "b", "task-b")
    (root / other / "result.json").write_text("{damaged")
    service = TrainingRunService(root, read_only=True)
    runs, bad = service.list_runs_tolerant("task-a")
    assert [run["run_id"] for run in runs] == [healthy]
    assert bad == 0
    all_runs, bad = service.list_runs_tolerant()
    assert [run["run_id"] for run in all_runs] == [healthy]
    assert bad == 1


@pytest.mark.parametrize(
    "corrupt",
    [
        {"run_id": "wrong"},
        {"session_id": None},
        {"status": "invented"},
        {"issues": {"severity": "warning"}},
        {"preflight": []},
        {"preflight": {"issues": ["invalid"]}},
        {"preflight": {"splits": {"train": {"rows": "unknown"}}}},
        {"config": {"training": None}},
        {"metrics": {"train_loss": "unknown"}},
        {"metrics": {"train_loss": 10**400}},
        {"failure": "unreadable"},
    ],
)
def test_malformed_identity_or_rendered_fields_are_reported(tmp_path, corrupt):
    root = tmp_path / "training"
    healthy = put_run(root, "a", "task-a")
    damaged = put_run(root, "b", "task-a", **corrupt)
    service = TrainingRunService(root, read_only=True)
    runs, bad = service.list_runs_tolerant("task-a")
    assert [run["run_id"] for run in runs] == [healthy]
    assert bad == 1
    assert (root / damaged / "run.json").is_file()


def test_bad_result_cannot_replace_task_identity(tmp_path):
    root = tmp_path / "training"
    identity = put_run(root, "a", "task-a")
    (root / identity / "result.json").write_text(
        json.dumps({"run_id": identity, "session_id": "task-b", "status": "blocked"})
    )
    assert TrainingRunService(root, read_only=True).list_runs_tolerant("task-a") == ([], 1)


def test_malformed_recovery_sidecar_is_reported(tmp_path):
    root = tmp_path / "training"
    identity = put_run(root, "a", "task-a")
    (root / identity / "recovery.json").write_text(json.dumps({"status": []}))
    assert TrainingRunService(root, read_only=True).list_runs_tolerant("task-a") == ([], 1)


def test_artifact_validation_retains_failed_state_instead_of_hiding_run(tmp_path):
    root = tmp_path / "training"
    identity = put_run(
        root, "a", "task-a", status="succeeded", artifacts={"manifest": str(tmp_path / "missing")}
    )
    runs, bad = TrainingRunService(root, read_only=True).list_runs_tolerant("task-a")
    assert bad == 0
    assert runs[0]["run_id"] == identity and runs[0]["status"] == "failed"
    assert runs[0]["failure"]["type"] == "ArtifactMismatch"


def test_legacy_optional_fields_and_unknown_fields_remain_browsable(tmp_path):
    root = tmp_path / "training"
    identity = put_run(
        root,
        "a",
        "task-a",
        issues=None,
        preflight={"status": "passed", "issues": None, "splits": None},
        config={},
        metrics=None,
        future_field={"format": "preserve"},
    )
    runs, bad = TrainingRunService(root, read_only=True).list_runs_tolerant("task-a")
    assert bad == 0 and runs[0]["run_id"] == identity
    assert runs[0]["future_field"] == {"format": "preserve"}


def test_display_task_keeps_healthy_training_record_with_warning(monkeypatch, tmp_path):
    from tests.unit.test_business_evaluation import task
    from ui import config, runtime
    from ui.components import agent_settings

    root = tmp_path / "workspace"
    _, session = task.__wrapped__(root / "outputs/workbench")
    training = root / "outputs/workbench/training"
    healthy = put_run(training, "a", session.session_id)
    damaged = put_run(training, "b", "unknown")
    (training / damaged / "run.json").write_text("{damaged")
    before = snapshot(root)
    monkeypatch.setattr(config, "PROJECT_ROOT", root)
    monkeypatch.setattr(runtime, "DISPLAY_ONLY", True)
    monkeypatch.setattr(agent_settings, "DISPLAY_ONLY", True)
    page = AppTest.from_file(str(ROOT / "ui/views/07_Data_Intake.py"), default_timeout=45).run()
    assert not page.exception, [item.message for item in page.exception]
    assert any(item.value == session.goal for item in page.title)
    assert any(healthy in item.label and "需要修正" in item.label for item in page.expander)
    assert any("1 条训练记录" in item.value and "归属不明" in item.value for item in page.warning)
    assert snapshot(root) == before
