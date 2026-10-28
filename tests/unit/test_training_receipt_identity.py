"""A worker result cannot redefine the identity or authorization of its run."""

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import PropertyMock, patch

import pytest
from streamlit.testing.v1 import AppTest

from src.workbench.training_runs import TrainingRunService

ROOT = Path(__file__).resolve().parents[2]


def tree(root):
    return {
        str(path.relative_to(root)): path.read_bytes() if path.is_file() else None
        for path in root.rglob("*")
    }


def make_run(root, key="a", session_id="task-a"):
    identity = "wb-" + key * 32
    directory = root / identity
    directory.mkdir(parents=True)
    record = {
        "run_id": identity,
        "session_id": session_id,
        "session_revision": 7,
        "status": "running",
        "dataset_version": "dataset-a",
        "dataset": {"source": "actual-source"},
        "model_path": "actual-base",
        "model_identity": {"sha256": "actual"},
        "output_dir": "actual-output",
        "config": {"model": "actual", "flag": True},
        "config_digest": "config-a",
        "preflight": {"status": "passed"},
        "issues": [],
        "plan_trace": [],
        "execution": {"project_root": "/real/root"},
        "acknowledge_warnings": False,
        "recover_technical_failures": True,
        "created_at": "time-a",
        "started_at": "time-b",
    }
    (directory / "run.json").write_text(json.dumps(record))
    return identity, directory, record


@pytest.mark.parametrize("phase", ["initial", "after_poll"])
@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("session_id", "task-b"),
        ("session_revision", 8),
        ("dataset_version", "dataset-b"),
        ("dataset", {"source": "other"}),
        ("model_path", "other-base"),
        ("model_identity", {"sha256": "other"}),
        ("output_dir", "other-output"),
        ("config", {"model": "other"}),
        ("config", {"model": "actual", "flag": 1}),
        ("config_digest", "config-b"),
        ("preflight", {"status": "blocked"}),
        ("issues", ["other"]),
        ("plan_trace", ["other"]),
        ("execution", {"project_root": "/other/private/path"}),
        ("acknowledge_warnings", True),
        ("recover_technical_failures", False),
        ("recover_technical_failures", 1),
        ("recovery_parent_run_id", "wb-" + "b" * 32),
        ("recovery_binding_digest", "forged"),
        ("created_at", "other"),
    ],
)
def test_conflicting_handoff_fields_are_rejected_at_both_receipt_reads(
    tmp_path, phase, field, value
):
    root = tmp_path / "training"
    identity, directory, _ = make_run(root)
    receipt = directory / "result.json"
    forged = {"run_id": identity, "status": "failed", field: value}
    if phase == "initial":
        receipt.write_text(json.dumps(forged))
    else:
        receipt.write_text(json.dumps({"run_id": identity, "status": "running"}))
    before = tree(tmp_path)

    def exit_with_final(_):
        receipt.write_text(json.dumps(forged))  # Worker publishes concurrently.
        return "failed"

    service = TrainingRunService(root, project_root=tmp_path, read_only=True)
    with (
        patch.object(
            TrainingRunService,
            "runner",
            new_callable=PropertyMock,
            return_value=SimpleNamespace(get_status=exit_with_final),
        ),
        pytest.raises(ValueError, match="身份或配置不一致") as error,
    ):
        service.get_status(identity)
    assert str(tmp_path) not in str(error.value) and "/other/private/path" not in str(error.value)
    after = tree(tmp_path)
    before.pop(str(receipt.relative_to(tmp_path)))
    after.pop(str(receipt.relative_to(tmp_path)))
    assert after == before


def test_matching_historical_fields_and_future_result_metadata_remain_readable(tmp_path):
    root = tmp_path / "training"
    identity, directory, record = make_run(root)
    receipt = {
        "run_id": identity,
        "status": "failed",
        "session_id": record["session_id"],
        "dataset": record["dataset"],
        "config": record["config"],
        "failure": {"stage": "training", "message": "recorded"},
        "future_worker_field": "preserved",
    }
    (directory / "result.json").write_text(json.dumps(receipt))
    before = tree(tmp_path)
    actual = TrainingRunService(root, read_only=True).get_status(identity)
    assert actual["status"] == "failed" and actual["failure"]["message"] == "recorded"
    assert actual["future_worker_field"] == "preserved"
    assert actual["session_id"] == record["session_id"] and actual["dataset"] == record["dataset"]
    assert tree(tmp_path) == before


def test_forged_success_and_matching_forged_manifest_are_rejected(tmp_path):
    root = tmp_path / "training"
    identity, directory, _ = make_run(root)
    artifact = directory / "artifact"
    artifact.write_text("evidence")
    manifest = directory / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "run_id": identity,
                "dataset_version": "dataset-b",
                "config_digest": "config-b",
                "files": {
                    "adapter": {
                        "path": str(artifact),
                        "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
                    }
                },
            }
        )
    )
    (directory / "result.json").write_text(
        json.dumps(
            {
                "run_id": identity,
                "status": "succeeded",
                "dataset_version": "dataset-b",
                "config_digest": "config-b",
                "artifacts": {"manifest": str(manifest)},
            }
        )
    )
    before = tree(tmp_path)
    service = TrainingRunService(root, read_only=True)
    with pytest.raises(ValueError, match="身份或配置不一致"):
        service.get_status(identity)
    assert service.list_runs_tolerant("task-a") == ([], 1)
    assert tree(tmp_path) == before


def test_cli_status_reports_conflict_without_forged_success(monkeypatch, tmp_path, capsys):
    from scripts import data_intake

    root = tmp_path / "training"
    identity, directory, _ = make_run(root)
    (directory / "result.json").write_text(
        json.dumps({"run_id": identity, "status": "failed", "session_id": "other"})
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "data_intake.py",
            "--store",
            str(tmp_path / "intake"),
            "--training-root",
            str(root),
            "train-status",
            identity,
        ],
    )
    assert data_intake.main() == 2
    captured = capsys.readouterr()
    assert "身份或配置不一致" in captured.err and str(tmp_path) not in captured.err
    assert captured.out == ""


def test_display_task_keeps_healthy_run_when_receipt_conflicts(monkeypatch, tmp_path):
    from tests.unit.test_business_evaluation import task
    from ui import config, runtime
    from ui.components import agent_settings

    workspace = tmp_path / "workspace"
    _, session = task.__wrapped__(workspace / "outputs/workbench")
    root = workspace / "outputs/workbench/training"
    identity, directory, record = make_run(root, "a", session.session_id)
    record.update(status="blocked", issues=[], preflight=None, config=None)
    (directory / "run.json").write_text(json.dumps(record))
    damaged, bad_dir, bad_record = make_run(root, "b", session.session_id)
    bad_record.update(status="blocked", issues=[], preflight=None, config=None)
    (bad_dir / "run.json").write_text(json.dumps(bad_record))
    (bad_dir / "result.json").write_text(
        json.dumps({"run_id": damaged, "status": "failed", "dataset_version": "forged"})
    )
    before = tree(workspace)
    monkeypatch.setattr(config, "PROJECT_ROOT", workspace)
    monkeypatch.setattr(runtime, "DISPLAY_ONLY", True)
    monkeypatch.setattr(agent_settings, "DISPLAY_ONLY", True)
    page = AppTest.from_file(str(ROOT / "ui/views/07_Data_Intake.py"), default_timeout=45).run()
    assert not page.exception, [item.message for item in page.exception]
    assert any(identity in item.label for item in page.expander)
    assert not any(damaged in item.label for item in page.expander)
    assert any("1 条训练记录" in item.value for item in page.warning)
    assert tree(workspace) == before


def test_unknown_result_field_may_change_between_running_and_terminal_receipts(tmp_path):
    root = tmp_path / "training"
    identity, directory, _ = make_run(root)
    receipt = directory / "result.json"
    receipt.write_text(
        json.dumps({"run_id": identity, "status": "running", "future_worker_field": "pending"})
    )

    def exit_with_final(_):
        receipt.write_text(
            json.dumps({"run_id": identity, "status": "failed", "future_worker_field": "complete"})
        )
        return "failed"

    with patch.object(
        TrainingRunService,
        "runner",
        new_callable=PropertyMock,
        return_value=SimpleNamespace(get_status=exit_with_final),
    ):
        result = TrainingRunService(root, read_only=True).get_status(identity)
    assert result["status"] == "failed" and result["future_worker_field"] == "complete"


@pytest.mark.parametrize("phase", ["initial", "after_poll"])
def test_mlflow_uri_must_match_prepared_logging_at_each_receipt_read(tmp_path, phase):
    root = tmp_path / "training"
    identity, directory, record = make_run(root)
    record["config"] = {"logging": {"mlflow_tracking_uri": "sqlite:///original"}}
    (directory / "run.json").write_text(json.dumps(record))
    receipt = directory / "result.json"
    forged = {"run_id": identity, "status": "failed", "mlflow_tracking_uri": "sqlite:///other"}
    if phase == "initial":
        receipt.write_text(json.dumps(forged))
    else:
        receipt.write_text(json.dumps({"run_id": identity, "status": "running"}))

    def exit_with_final(_):
        receipt.write_text(json.dumps(forged))
        return "failed"

    with (
        patch.object(
            TrainingRunService,
            "runner",
            new_callable=PropertyMock,
            return_value=SimpleNamespace(get_status=exit_with_final),
        ),
        pytest.raises(ValueError, match="身份或配置不一致") as error,
    ):
        TrainingRunService(root, read_only=True).get_status(identity)
    assert "sqlite:///" not in str(error.value)


@pytest.mark.parametrize("value", [None, "", "sqlite:///original", "missing"])
def test_matching_or_missing_mlflow_uri_remains_compatible(tmp_path, value):
    root = tmp_path / "training"
    identity, directory, record = make_run(root)
    record["config"] = {"logging": {"mlflow_tracking_uri": "sqlite:///original"}}
    (directory / "run.json").write_text(json.dumps(record))
    # The actual worker includes these outcome fields; old receipts may omit
    # the tracking URI, or record None when the lookup was unavailable.
    receipt = {
        "run_id": identity,
        "status": "failed",
        "pid": 123,
        "artifacts": {},
        "metrics": {},
        "failure": {"stage": "training", "message": "actual failure"},
        "mlflow_run_id": "real-run" if value == "sqlite:///original" else None,
        "mlflow_status": "recorded" if value == "sqlite:///original" else "unavailable",
    }
    if value != "missing":
        receipt["mlflow_tracking_uri"] = value
    (directory / "result.json").write_text(json.dumps(receipt))
    before = tree(tmp_path)
    result = TrainingRunService(root, read_only=True).get_status(identity)
    assert result["status"] == "failed" and result["failure"]["message"] == "actual failure"
    assert tree(tmp_path) == before
