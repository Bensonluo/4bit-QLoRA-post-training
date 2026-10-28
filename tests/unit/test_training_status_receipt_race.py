"""Training status observes the final receipt published during process exit."""

import hashlib
import json
from types import SimpleNamespace
from unittest.mock import PropertyMock, patch

import pytest

from src.workbench.training_runs import TrainingRunService


def snapshot(root):
    return {
        str(path.relative_to(root)): path.read_bytes() if path.is_file() else None
        for path in root.rglob("*")
    }


@pytest.fixture
def running(tmp_path):
    identity = "wb-" + "a" * 32
    directory = tmp_path / "training" / identity
    directory.mkdir(parents=True)
    record = {
        "run_id": identity,
        "session_id": "task-a",
        "status": "running",
        "dataset_version": "dataset-v1",
        "config_digest": "config-v1",
    }
    (directory / "run.json").write_text(json.dumps(record))
    return (
        TrainingRunService(directory.parent, project_root=tmp_path, read_only=True),
        directory,
        identity,
    )


def valid_success(directory, identity):
    output = directory / "model"
    output.mkdir(exist_ok=True)
    paths = {
        "adapter_config": output / "adapter_config.json",
        "adapter_weights": output / "adapter_model.bin",
        "tokenizer_config": output / "tokenizer_config.json",
    }
    for artifact in paths.values():
        artifact.write_bytes(b"valid artifact")
    manifest = {
        "run_id": identity,
        "dataset_version": "dataset-v1",
        "config_digest": "config-v1",
        "files": {
            key: {
                "path": str(artifact),
                "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
            }
            for key, artifact in paths.items()
        },
    }
    path = output / "workbench_manifest.json"
    path.write_text(json.dumps(manifest))
    return {
        "run_id": identity,
        "status": "succeeded",
        "artifacts": {**{key: str(value) for key, value in paths.items()}, "manifest": str(path)},
    }


@pytest.mark.parametrize("initial", ["missing", "running"])
@pytest.mark.parametrize("terminal", ["succeeded", "failed", "stopped"])
def test_final_receipt_wins_if_published_between_first_read_and_process_exit(
    running, initial, terminal
):
    service, directory, identity = running
    receipt = directory / "result.json"
    if initial == "running":
        receipt.write_text(json.dumps({"run_id": identity, "status": "running"}))
    final = (
        valid_success(directory, identity)
        if terminal == "succeeded"
        else {
            "run_id": identity,
            "status": terminal,
            "failure": {"stage": "training", "type": "RealResult", "message": "actual evidence"},
        }
    )
    before = snapshot(directory.parent.parent)

    def child_finishes(_run_id):
        assert _run_id == identity
        receipt.write_text(json.dumps(final))  # Simulates the worker, not the reader.
        return "finished" if terminal == "succeeded" else "failed"

    with patch.object(
        TrainingRunService,
        "runner",
        new_callable=PropertyMock,
        return_value=SimpleNamespace(get_status=child_finishes),
    ):
        result = service.get_status(identity)
    assert result["status"] == terminal
    if terminal != "succeeded":
        assert result["failure"]["type"] == "RealResult"
    after = snapshot(directory.parent.parent)
    before.pop(str(receipt.relative_to(directory.parent.parent)), None)
    after.pop(str(receipt.relative_to(directory.parent.parent)))
    assert after == before


@pytest.mark.parametrize(
    "damage", ["missing_entry", "missing_weight", "redirected_weight", "bad_files"]
)
def test_succeeded_receipt_requires_bound_adapter_evidence(running, damage, tmp_path):
    service, directory, identity = running
    result = valid_success(directory, identity)
    manifest_path = directory / "model" / "workbench_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if damage == "missing_entry":
        del manifest["files"]["adapter_weights"]
    elif damage == "missing_weight":
        (directory / "model" / "adapter_model.bin").unlink()
    elif damage == "redirected_weight":
        other = tmp_path / "adapter_model.bin"
        other.write_bytes(b"valid artifact")
        manifest["files"]["adapter_weights"]["path"] = str(other)
        result["artifacts"]["adapter_weights"] = str(other)
    else:
        manifest["files"] = []
    manifest_path.write_text(json.dumps(manifest))
    (directory / "result.json").write_text(json.dumps(result))
    before = snapshot(tmp_path)
    status = service.get_status(identity)
    assert status["status"] == "failed"
    assert status["failure"]["stage"] == "artifacts"
    assert status["failure"]["type"] == "ArtifactMismatch"
    assert snapshot(tmp_path) == before


def test_newly_published_receipt_still_checks_identity(running):
    service, directory, identity = running

    def child_finishes(_run_id):
        (directory / "result.json").write_text(json.dumps({"run_id": "other", "status": "failed"}))
        return "failed"

    with (
        patch.object(
            TrainingRunService,
            "runner",
            new_callable=PropertyMock,
            return_value=SimpleNamespace(get_status=child_finishes),
        ),
        pytest.raises(ValueError, match="身份不匹配"),
    ):
        service.get_status(identity)


def test_newly_published_success_still_verifies_artifacts(running):
    service, directory, identity = running
    final = valid_success(directory, identity)
    (directory / "model" / "adapter_model.bin").write_bytes(b"modified adapter")

    def child_finishes(_run_id):
        (directory / "result.json").write_text(json.dumps(final))
        return "finished"

    with patch.object(
        TrainingRunService,
        "runner",
        new_callable=PropertyMock,
        return_value=SimpleNamespace(get_status=child_finishes),
    ):
        result = service.get_status(identity)
    assert result["status"] == "failed" and result["failure"]["type"] == "ArtifactMismatch"


@pytest.mark.parametrize("existing", [False, True])
@pytest.mark.parametrize("requested_stop", [False, True])
def test_exited_process_without_terminal_receipt_remains_truthful(
    running, existing, requested_stop
):
    service, directory, identity = running
    if existing:
        (directory / "result.json").write_text(
            json.dumps({"run_id": identity, "status": "running"})
        )
    if requested_stop:
        (directory / "stop.request").touch()
    before = snapshot(directory.parent.parent)
    with patch.object(
        TrainingRunService,
        "runner",
        new_callable=PropertyMock,
        return_value=SimpleNamespace(get_status=lambda _: "finished"),
    ):
        result = service.get_status(identity)
    assert result["status"] == ("stopped" if requested_stop else "failed")
    assert result["failure"]["type"] == ("UserStopped" if requested_stop else "MissingResult")
    assert snapshot(directory.parent.parent) == before


def test_real_terminal_receipt_wins_over_a_late_stop_request(running):
    service, directory, identity = running
    final = valid_success(directory, identity)
    (directory / "stop.request").touch()

    def child_finishes(_run_id):
        (directory / "result.json").write_text(json.dumps(final))
        return "finished"

    with patch.object(
        TrainingRunService,
        "runner",
        new_callable=PropertyMock,
        return_value=SimpleNamespace(get_status=child_finishes),
    ):
        result = service.get_status(identity)
    assert result["status"] == "succeeded"
