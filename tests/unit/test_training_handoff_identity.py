"""Training must use the identity, data, model and output that were prepared."""

import json
import time
from copy import deepcopy
from unittest.mock import patch

import pytest

from src.workbench.sources import content_digest
from tests.unit.test_workbench_training_runs import _prepare, _wait, environment  # noqa: F401


@pytest.mark.parametrize(
    "damage", ["other_run", "output", "model", "train_file", "dataset_split", "digest", "tracking"]
)
def test_damaged_prepared_run_cannot_be_launched(request, damage):
    fixture = request.getfixturevalue("environment")
    _, session, service, _ = fixture
    first, second = _prepare(fixture), _prepare(fixture)
    directory = service._directory(first["run_id"])
    destination = service._directory(second["run_id"]) / "model"
    destination.mkdir()
    sentinel = destination / "existing-model.safetensors"
    sentinel.write_bytes(b"existing model")
    record = json.loads((directory / "run.json").read_text())
    if damage == "other_run":
        record = second
    elif damage == "output":
        record["output_dir"] = str(destination)
        record["config"]["training"]["output_dir"] = str(destination)
    elif damage == "model":
        record["config"]["model"]["name"] = str(destination)
    elif damage == "train_file":
        record["config"]["data"]["train_file"] = str(destination / "data.jsonl")
    elif damage == "dataset_split":
        record["config"]["data"]["dataset_split"] = "test"
    elif damage == "digest":
        record["config_digest"] = "wrong"
    elif damage == "tracking":
        record["config"]["logging"]["mlflow_run_name"] = second["run_id"]
    if damage not in {"other_run", "digest"}:
        record["config_digest"] = content_digest(record["config"])
    path = directory / "run.json"
    path.write_text(json.dumps(record))
    before = path.read_bytes()
    runner = service.runner
    with (
        patch.object(runner, "launch_training") as launch,
        pytest.raises(ValueError, match="交接|身份"),
    ):
        service.start(first["run_id"], session)
    launch.assert_not_called()
    assert not (directory / "started").exists()
    assert path.read_bytes() == before
    assert sentinel.read_bytes() == b"existing model"


def test_worker_rechecks_run_identity_after_parent_launch_check(request, monkeypatch):
    fixture = request.getfixturevalue("environment")
    _, session, service, _ = fixture
    record = _prepare(fixture)
    other = _prepare(fixture)
    directory = service._directory(record["run_id"])
    runner = service.runner
    launch = runner.launch_training

    def replace_before_worker(*args, **kwargs):
        path = directory / "run.json"
        changed = json.loads(path.read_text())
        changed["run_id"] = other["run_id"]
        path.write_text(json.dumps(changed))
        launched = launch(*args, **kwargs)
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            if runner._active[record["run_id"]].poll() is not None:
                return launched
            time.sleep(0.05)
        pytest.fail("worker did not exit")

    monkeypatch.setattr(runner, "launch_training", replace_before_worker)
    service.start(record["run_id"], session)
    result = service.get_status(record["run_id"])
    assert result["status"] == "failed"
    assert result["failure"]["type"] == "MissingResult"
    assert "训练交接" in service.read_logs(record["run_id"], tail=100)
    assert not (directory / "result.json").exists()
    assert not (directory / "model" / "adapter_model.safetensors").exists()


@pytest.mark.parametrize("damage", ["directory", "directory_and_id", "complete_other_handoff"])
def test_wrong_worker_directory_cannot_overwrite_other_receipt(request, monkeypatch, damage):
    fixture = request.getfixturevalue("environment")
    _, session, service, _ = fixture
    first, second = _prepare(fixture), _prepare(fixture)
    directory = service._directory(first["run_id"])
    other_directory = service._directory(second["run_id"])
    other_receipt = other_directory / "result.json"
    other_receipt.write_text(json.dumps({"run_id": second["run_id"], "status": "succeeded"}))
    before_receipt = other_receipt.read_bytes()
    before_record = (other_directory / "run.json").read_bytes()
    runner = service.runner
    launch = runner.launch_training

    def redirect_worker(technique, config, run_name, **kwargs):
        edited = deepcopy(config)
        edited["workbench"]["run_dir"] = str(other_directory)
        if damage != "directory":
            edited["workbench"]["run_id"] = second["run_id"]
        if damage == "complete_other_handoff":
            edited["training"]["output_dir"] = second["output_dir"]
            edited["logging"]["mlflow_run_name"] = second["run_id"]
        return launch(technique, edited, run_name, **kwargs)

    monkeypatch.setattr(runner, "launch_training", redirect_worker)
    service.start(first["run_id"], session)
    status = _wait(service, first["run_id"])
    assert status["status"] == "failed"
    assert status["failure"]["type"] == "MissingResult"
    assert "训练交接" in service.read_logs(first["run_id"], tail=100)
    assert not (directory / "result.json").exists()
    assert other_receipt.read_bytes() == before_receipt
    assert (other_directory / "run.json").read_bytes() == before_record


def test_output_symlink_to_another_run_cannot_be_started(request):
    fixture = request.getfixturevalue("environment")
    _, session, service, _ = fixture
    first, second = _prepare(fixture), _prepare(fixture)
    directory = service._directory(first["run_id"])
    other_model = service._directory(second["run_id"]) / "model"
    other_model.mkdir()
    sentinel = other_model / "adapter_model.safetensors"
    sentinel.write_bytes(b"existing model")
    (directory / "model").symlink_to(other_model, target_is_directory=True)
    runner = service.runner
    with patch.object(runner, "launch_training") as launch, pytest.raises(ValueError):
        service.start(first["run_id"], session)
    launch.assert_not_called()
    assert not (directory / "started").exists()
    assert sentinel.read_bytes() == b"existing model"


def test_run_directory_symlink_cannot_be_started(request):
    fixture = request.getfixturevalue("environment")
    _, session, service, _ = fixture
    first = _prepare(fixture)
    directory = service._directory(first["run_id"])
    moved = directory.with_name("moved-run")
    directory.rename(moved)
    directory.symlink_to(moved, target_is_directory=True)
    runner = service.runner
    with patch.object(runner, "launch_training") as launch, pytest.raises(ValueError):
        service.start(first["run_id"], session)
    launch.assert_not_called()
    assert not (moved / "started").exists()


@pytest.mark.parametrize("existing", ["artifact", "nested_symlink"])
def test_preexisting_output_artifact_cannot_be_overwritten(request, existing):
    fixture = request.getfixturevalue("environment")
    _, session, service, _ = fixture
    first = _prepare(fixture)
    directory = service._directory(first["run_id"])
    output = directory / "model"
    output.mkdir()
    sentinel = directory / "protected.bin"
    sentinel.write_bytes(b"existing model")
    target = output / "adapter_model.safetensors"
    if existing == "nested_symlink":
        target.symlink_to(sentinel)
    else:
        target.write_bytes(b"previous artifact")
    runner = service.runner
    with patch.object(runner, "launch_training") as launch, pytest.raises(ValueError):
        service.start(first["run_id"], session)
    launch.assert_not_called()
    assert not (directory / "started").exists()
    assert sentinel.read_bytes() == b"existing model"
    if existing == "artifact":
        assert target.read_bytes() == b"previous artifact"


def test_empty_output_directory_remains_startable(request):
    fixture = request.getfixturevalue("environment")
    _, session, service, _ = fixture
    first = _prepare(fixture)
    (service._directory(first["run_id"]) / "model").mkdir()
    runner = service.runner
    with patch.object(runner, "launch_training") as launch:
        result = service.start(first["run_id"], session)
    assert result["status"] == "running"
    launch.assert_called_once()


@pytest.mark.parametrize("existing", ["directory_symlink", "nested_symlink", "artifact"])
def test_worker_rechecks_output_symlink_before_touching_any_receipt(request, monkeypatch, existing):
    fixture = request.getfixturevalue("environment")
    _, session, service, _ = fixture
    first, second = _prepare(fixture), _prepare(fixture)
    directory = service._directory(first["run_id"])
    other_model = service._directory(second["run_id"]) / "model"
    other_model.mkdir()
    sentinel = other_model / "adapter_model.safetensors"
    sentinel.write_bytes(b"existing model")
    runner = service.runner
    launch = runner.launch_training

    def redirect_output(*args, **kwargs):
        output = directory / "model"
        if existing == "directory_symlink":
            output.symlink_to(other_model, target_is_directory=True)
        else:
            output.mkdir()
            target = output / "adapter_model.safetensors"
            if existing == "nested_symlink":
                target.symlink_to(sentinel)
            else:
                target.write_bytes(b"previous artifact")
        return launch(*args, **kwargs)

    monkeypatch.setattr(runner, "launch_training", redirect_output)
    service.start(first["run_id"], session)
    status = _wait(service, first["run_id"])
    assert status["status"] == "failed"
    assert status["failure"]["type"] == "MissingResult"
    assert "训练交接" in service.read_logs(first["run_id"], tail=100)
    assert not (directory / "result.json").exists()
    assert sentinel.read_bytes() == b"existing model"
    if existing == "artifact":
        assert (
            directory / "model" / "adapter_model.safetensors"
        ).read_bytes() == b"previous artifact"
