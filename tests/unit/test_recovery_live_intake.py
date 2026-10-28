"""Automatic recovery reloads its original task and never trusts an old snapshot."""

import json
from unittest.mock import patch

import pytest

from src.workbench.intake_service import IntakeService
from src.workbench.training_recovery import dispatch_after_exit
from src.workbench.training_runs import write_json
from tests.unit import test_training_recovery as recovery_tests
from tests.unit import test_workbench_training_runs as training_tests


@pytest.fixture
def case(tmp_path, monkeypatch):
    environment = training_tests.environment.__wrapped__(tmp_path, monkeypatch)
    original, training, recovery, parent = recovery_tests.failed.__wrapped__(environment)
    intake = environment[0]
    return intake, original, training, recovery, parent


def dispatch(training, parent):
    with patch("src.tracking.runner.TrainingRunner.launch_training", return_value="stub") as launch:
        result = dispatch_after_exit(
            training._directory(parent["run_id"]), 2147483647, 0, timeout=1
        )
    return result, launch.call_count


@pytest.mark.parametrize(
    "state",
    ["pending", "failed", "missing", "wrong_task", "legacy", "missing_snapshot", "bad_snapshot"],
)
def test_dispatch_does_not_launch_if_live_task_cannot_authorize_retry(case, tmp_path, state):
    intake, original, training, _, parent = case
    directory = training._directory(parent["run_id"])
    if state in {"pending", "failed"}:
        check = intake.start_label_verification(
            original.session_id, original.revision, sample_size=1
        )
        if state == "failed":
            verdict = intake.submit_label_verification(
                original.session_id,
                check["verification_id"],
                {item["row_id"]: "与标签不一致" for item in check["items"]},
            )
            assert verdict["verdict"] == "insufficient_agreement"
    elif state == "missing":
        intake.database.unlink()
    elif state == "wrong_task":
        wrong_store = IntakeService(tmp_path / "other-store")
        write_json(
            directory / "intake-store.json",
            {
                "root": str(wrong_store.root.resolve()),
                "session_id": original.session_id,
                "run_id": parent["run_id"],
            },
        )
    elif state == "legacy":
        (directory / "intake-store.json").unlink()
    elif state == "missing_snapshot":
        (directory / "session.json").unlink()
    else:
        (directory / "session.json").write_text("{corrupt")
    result, launches = dispatch(training, parent)
    assert result["status"] == "retry_blocked"
    assert any(word in result["reason"] for word in ("盲标", "存储"))
    assert launches == 0
    assert len(list(training.root.glob("wb-*/run.json"))) == 1
    if state == "missing":
        assert not intake.database.exists()


def test_dispatch_reloads_verified_task_and_starts_one_retry(case):
    intake, original, training, _, parent = case
    assert json.loads(
        (training._directory(parent["run_id"]) / "intake-store.json").read_text()
    ) == {
        "root": str(intake.root.resolve()),
        "session_id": original.session_id,
        "run_id": parent["run_id"],
    }
    assert intake.load(original.session_id).label_verification["verdict"] == "verified"
    result, launches = dispatch(training, parent)
    assert result["status"] == "retry_started" and launches == 1
    child = training._load(result["child_run_id"])
    assert child["recovery_parent_run_id"] == parent["run_id"]


def test_legacy_unbound_run_can_start_normally_but_cannot_authorize_auto_recovery(
    tmp_path, monkeypatch
):
    environment = training_tests.environment.__wrapped__(tmp_path, monkeypatch)
    _, session, training, _ = environment
    first = training_tests._prepare(environment)
    directory = training._directory(first["run_id"])
    with patch("src.tracking.runner.TrainingRunner.launch_training", return_value="stub") as launch:
        with pytest.raises(ValueError, match="存储未绑定"):
            training.start(first["run_id"], session, recover_technical_failures=True)
        launch.assert_not_called()
        assert not (directory / "started").exists()
        result = training.start(first["run_id"], session)
        assert result["status"] == "running"
        launch.assert_called_once()


def test_store_binding_is_not_serialized_into_task_or_run(case):
    _, session, training, _, parent = case
    assert "intake_store" not in session.model_dump()
    assert session == type(session).model_validate_json(session.model_dump_json())
    assert session == session.model_copy(deep=True)
    assert "intake_store" not in parent
    assert "intake_store" not in json.loads(
        (training._directory(parent["run_id"]) / "run.json").read_text()
    )


def test_start_with_explicit_store_binds_only_private_sidecar(tmp_path, monkeypatch):
    environment = training_tests.environment.__wrapped__(tmp_path, monkeypatch)
    intake, session, training, _ = environment
    record = training_tests._prepare(environment)
    directory = training._directory(record["run_id"])
    with patch("src.tracking.runner.TrainingRunner.launch_training", return_value="stub") as launch:
        result = training.start(
            record["run_id"],
            session,
            recover_technical_failures=True,
            recovery_intake_root=intake.root,
        )
    assert result["status"] == "running"
    launch.assert_called_once()
    assert json.loads((directory / "intake-store.json").read_text()) == {
        "root": str(intake.root.resolve()),
        "session_id": session.session_id,
        "run_id": record["run_id"],
    }
    assert "intake_store" not in result
    assert "intake_store" not in json.loads((directory / "run.json").read_text())
    assert "intake_store" not in json.loads((directory / "session.json").read_text())


def test_start_loser_cannot_overwrite_existing_private_binding(tmp_path, monkeypatch):
    environment = training_tests.environment.__wrapped__(tmp_path, monkeypatch)
    intake, session, training, _ = environment
    record = training_tests._prepare(environment)
    directory = training._directory(record["run_id"])
    (directory / "started").touch()
    binding = directory / "intake-store.json"
    binding.write_text('{"root":"original-owner","session_id":"original"}')
    before = binding.read_bytes()
    with patch("src.tracking.runner.TrainingRunner.launch_training") as launch:
        with pytest.raises(ValueError, match="已经提交启动"):
            training.start(
                record["run_id"],
                session,
                recover_technical_failures=True,
                recovery_intake_root=intake.root,
            )
        launch.assert_not_called()
    assert binding.read_bytes() == before


def test_failed_binding_write_does_not_start_and_marker_retains_failed_state(tmp_path, monkeypatch):
    from src.workbench import training_runs

    environment = training_tests.environment.__wrapped__(tmp_path, monkeypatch)
    intake, session, training, _ = environment
    record = training_tests._prepare(environment)
    directory = training._directory(record["run_id"])
    original = training_runs.write_json

    def broken(path, value):
        if path.name == "intake-store.json":
            raise OSError("binding write denied")
        return original(path, value)

    monkeypatch.setattr(training_runs, "write_json", broken)
    with patch("src.tracking.runner.TrainingRunner.launch_training") as launch:
        result = training.start(
            record["run_id"],
            session,
            recover_technical_failures=True,
            recovery_intake_root=intake.root,
        )
        launch.assert_not_called()
    assert result["status"] == "failed" and result["failure"]["stage"] == "launch"
    assert (directory / "started").exists()
    assert not (directory / "intake-store.json").exists()
    assert training.get_status(record["run_id"])["status"] == "failed"
    with pytest.raises(ValueError, match="只有已准备"):
        training.start(record["run_id"], session)


def test_dispatch_uses_explicit_custom_store_not_another_copy(tmp_path, monkeypatch):
    import shutil

    original_environment = training_tests.environment.__wrapped__(tmp_path, monkeypatch)
    original_intake, original_session, training, model = original_environment
    custom_root = tmp_path / "custom 任务 #100%"
    shutil.copytree(original_intake.root, custom_root)
    custom_intake = IntakeService(custom_root)
    custom_session = custom_intake.load(original_session.session_id)
    environment = (custom_intake, custom_session, training, model)
    _, _, _, parent = recovery_tests.failed.__wrapped__(environment)
    original_intake.start_label_verification(
        original_session.session_id, original_session.revision, sample_size=1
    )
    assert (
        original_intake.load(original_session.session_id).label_verification["status"] == "pending"
    )
    assert custom_intake.load(custom_session.session_id).label_verification["verdict"] == "verified"
    result, launches = dispatch(training, parent)
    assert result["status"] == "retry_started" and launches == 1


def test_modified_binding_to_identical_store_is_rejected(case, tmp_path):
    import shutil

    intake, session, training, _, parent = case
    copy = tmp_path / "identical-store"
    shutil.copytree(intake.root, copy)
    assert (
        IntakeService(copy, read_only=True).load(session.session_id).model_dump()
        == session.model_dump()
    )
    binding = training._directory(parent["run_id"]) / "intake-store.json"
    write_json(
        binding,
        {"root": str(copy.resolve()), "session_id": session.session_id, "run_id": parent["run_id"]},
    )
    result, launches = dispatch(training, parent)
    assert result["status"] == "retry_blocked" and launches == 0
    assert len(list(training.root.glob("wb-*/run.json"))) == 1


def test_dispatch_persists_blocked_status_if_authorization_was_revoked(case):
    _, _, training, _, parent = case
    directory = training._directory(parent["run_id"])
    record = json.loads((directory / "run.json").read_text())
    record["recover_technical_failures"] = False
    write_json(directory / "run.json", record)
    write_json(
        directory / "recovery.json",
        {
            "status": "dispatch_waiting",
            "parent_run_id": parent["run_id"],
            "child_run_id": None,
            "dispatcher_pid": 123,
        },
    )
    result, launches = dispatch(training, parent)
    saved = json.loads((directory / "recovery.json").read_text())
    assert result["status"] == saved["status"] == "retry_blocked"
    assert "没有授权" in saved["reason"] and saved["dispatcher_pid"] == 123
    assert launches == 0 and not (directory / "recovery.claim").exists()


def test_dispatch_persists_cancelled_if_stop_arrives_after_parent_exits(case, monkeypatch):
    from src.workbench import training_recovery

    _, _, training, _, parent = case
    directory = training._directory(parent["run_id"])
    write_json(directory / "recovery.json", {"status": "dispatch_waiting", "child_run_id": None})
    original = training_recovery.load_bound_intake_session

    def concurrent_stop(*args, **kwargs):
        live = original(*args, **kwargs)
        (directory / "stop.request").touch()
        return live

    monkeypatch.setattr(training_recovery, "load_bound_intake_session", concurrent_stop)
    result, launches = dispatch(training, parent)
    saved = json.loads((directory / "recovery.json").read_text())
    assert result["status"] == saved["status"] == "cancelled"
    assert launches == 0 and not (directory / "recovery.claim").exists()


@pytest.mark.parametrize("existing", ["claim", "child"])
def test_dispatch_does_not_overwrite_existing_recovery_claim_or_child(case, existing):
    _, _, training, _, parent = case
    directory = training._directory(parent["run_id"])
    saved = {
        "status": "retry_started" if existing == "child" else "preparing_retry",
        "parent_run_id": parent["run_id"],
        "child_run_id": "wb-" + "a" * 32 if existing == "child" else None,
        "reason": "保持已有恢复",
    }
    write_json(directory / "recovery.json", saved)
    if existing == "claim":
        (directory / "recovery.claim").touch()
    # Even a now-unreadable task must not overwrite an already claimed attempt.
    (directory / "intake-store.json").unlink()
    before = (directory / "recovery.json").read_bytes()
    result, launches = dispatch(training, parent)
    assert result == saved and launches == 0
    assert (directory / "recovery.json").read_bytes() == before


def test_dispatch_timeout_does_not_overwrite_an_existing_claim(case):
    import os

    import psutil

    _, _, training, _, parent = case
    directory = training._directory(parent["run_id"])
    saved = {
        "status": "retry_started",
        "parent_run_id": parent["run_id"],
        "child_run_id": "wb-" + "a" * 32,
        "reason": "已有子运行",
    }
    write_json(directory / "recovery.json", saved)
    (directory / "recovery.claim").touch()
    before = (directory / "recovery.json").read_bytes()
    result = dispatch_after_exit(directory, os.getpid(), psutil.Process().create_time(), timeout=0)
    assert result == saved
    assert (directory / "recovery.json").read_bytes() == before


def test_dispatch_stop_keeps_child_identity_when_cancelling(case):
    import os

    import psutil

    _, _, training, _, parent = case
    directory = training._directory(parent["run_id"])
    child = "wb-" + "a" * 32
    write_json(directory / "recovery.json", {"status": "retry_started", "child_run_id": child})
    (directory / "recovery.claim").touch()
    (directory / "stop.request").touch()
    result = dispatch_after_exit(directory, os.getpid(), psutil.Process().create_time(), timeout=0)
    assert result["status"] == "cancelled" and result["child_run_id"] == child
    assert json.loads((directory / "recovery.json").read_text())["child_run_id"] == child
