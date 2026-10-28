"""The worker must preflight the same prepared task and data it will train."""

import json
from pathlib import Path

import pytest

from tests.unit.test_data_materialize import FULL, _full
from tests.unit.test_workbench_training_runs import _prepare, _wait, environment  # noqa: F401


@pytest.mark.parametrize("damage", ["other_task", "other_dataset_same_identity", "revision"])
def test_worker_rejects_other_task_snapshot_before_training_unverified_data(
    request, monkeypatch, damage
):
    fixture = request.getfixturevalue("environment")
    intake, session, service, _ = fixture
    other = _full(intake, data=FULL.replace(b",yes", b",no"))
    other = intake.materialize_dataset(other.session_id, other.revision)
    assert other.dataset.version != session.dataset.version
    record = _prepare(fixture)
    directory = service.root / record["run_id"]
    train = Path(session.dataset.paths["train"])
    original_train = train.read_bytes()
    launch = service.runner.launch_training

    def swap_before_worker(*args, **kwargs):
        # This occurs after the parent has checked the original session.
        replacement = (
            session.model_copy(update={"revision": session.revision + 1})
            if damage == "revision"
            else other
            if damage == "other_task"
            else other.model_copy(
                update={"session_id": session.session_id, "revision": session.revision}
            )
        )
        (directory / "session.json").write_text(replacement.model_dump_json())
        train.write_bytes(original_train.replace(b"yes", b"no"))
        assert train.read_bytes() != original_train
        return launch(*args, **kwargs)

    monkeypatch.setattr(service.runner, "launch_training", swap_before_worker)
    service.start(record["run_id"], session)
    result = _wait(service, record["run_id"])
    assert result["status"] == "failed", service.read_logs(record["run_id"], tail=100)
    assert result["failure"]["stage"] == "preflight"
    assert "任务快照" in result["failure"]["message"]
    assert not (directory / "model" / "adapter_model.safetensors").exists()
    saved = json.loads((directory / "session.json").read_text())
    if damage == "other_task":
        assert saved["session_id"] == other.session_id
    elif damage == "revision":
        assert saved["revision"] == session.revision + 1
    else:
        assert saved["session_id"] == session.session_id
        assert saved["dataset"]["version"] == other.dataset.version
