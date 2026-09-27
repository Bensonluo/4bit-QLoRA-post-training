"""Service-level guards: authorization scope, duplicate submission, stop, resume."""

import json

import pytest

from src.workbench import iteration_execution as execution_module
from src.workbench.intake_models import FullDataValidation, IntakeSession
from src.workbench.iteration_execution import IterationExecutionService, authorization_valid

IDENTITY = "it-" + "a" * 32


def _session(**fields):
    base = dict(
        session_id="s" * 32,
        goal="判断类别",
        revision=4,
        source=type("Source", (), {"digest": "source-digest"})(),
        sources={"main": type("Source", (), {"digest": "source-digest"})()},
        analysis=None,
        full_data=None,
        dataset=None,
        label_verification={"verdict": "verified"},
    )
    base.update(fields)
    return IntakeSession.model_construct(**base)


def test_authorization_binds_goal_sources_analysis_and_full_data():
    full = FullDataValidation.model_construct(status="confirmed", confirmed_revision=3)
    snapshot = _session(full_data=full)
    assert authorization_valid(_session(full_data=full), snapshot)
    # The worker's own materialization only touches the dataset; scope stays valid.
    assert authorization_valid(_session(full_data=full, dataset=object()), snapshot)
    assert not authorization_valid(_session(goal="另一个目标", full_data=full), snapshot)
    changed_source = type("Source", (), {"digest": "other"})()
    assert not authorization_valid(
        _session(
            full_data=full,
            sources={"main": changed_source, "extra": changed_source},
        ),
        snapshot,
    )
    assert not authorization_valid(_session(full_data=None), snapshot)
    other_full = FullDataValidation.model_construct(status="confirmed", confirmed_revision=4)
    assert not authorization_valid(_session(full_data=other_full), snapshot)


@pytest.fixture()
def harness(tmp_path, monkeypatch):
    live = _session(
        full_data=FullDataValidation.model_construct(status="confirmed", confirmed_revision=3)
    )
    iterations = {
        IDENTITY: {
            "iteration_id": IDENTITY,
            "session_id": live.session_id,
            "goal": live.goal,
            "status": "confirmed",
        }
    }
    claims = []
    launches = []

    class IterationsFake:
        def __init__(self, iteration_root, training_root, evaluation_root):
            pass

        def get(self, iteration_id):
            return iterations[iteration_id]

        def claim_execution(self, iteration_id, execution_id):
            claims.append((iteration_id, execution_id))
            return iterations[iteration_id]

    class IntakeFake:
        def __init__(self, root):
            pass

        def load(self, session_id):
            return live

    monkeypatch.setattr(execution_module, "IterationService", IterationsFake)
    monkeypatch.setattr(execution_module, "IntakeService", IntakeFake)
    monkeypatch.setattr(
        execution_module.IterationExecutionService,
        "_launch",
        lambda self, record: launches.append(record) or dict(record),
    )
    service = IterationExecutionService(
        tmp_path / "executions",
        tmp_path / "intake",
        tmp_path / "iterations",
        tmp_path / "training",
        tmp_path / "eval",
    )
    return {
        "service": service,
        "live": live,
        "iterations": iterations,
        "claims": claims,
        "launches": launches,
        "monkeypatch": monkeypatch,
    }


def test_fresh_authorization_snapshots_session_and_claims_iteration(harness):
    started = harness["service"].start(IDENTITY, harness["live"])
    assert started["status"] == "queued"
    assert started["session_revision"] == harness["live"].revision
    assert started["options"] == {
        "acknowledge_warnings": False,
        "independent_rows_confirmed": False,
    }
    assert harness["claims"] == [(IDENTITY, IDENTITY)]
    assert len(harness["launches"]) == 1
    directory = harness["service"]._directory(IDENTITY)
    snapshot = json.loads((directory / "session.json").read_text(encoding="utf-8"))
    assert snapshot["session_id"] == harness["live"].session_id
    assert snapshot["revision"] == harness["live"].revision


def test_authorization_rejects_stale_or_unconfirmed_iterations(harness):
    stale = _session(revision=5, full_data=harness["live"].full_data)
    with pytest.raises(ValueError, match="任务已更新"):
        harness["service"].start(IDENTITY, stale)
    harness["live"].full_data = None
    with pytest.raises(ValueError, match="确认全量数据"):
        harness["service"].start(IDENTITY, harness["live"])
    harness["iterations"][IDENTITY]["status"] = "prepared"
    with pytest.raises(ValueError, match="已确认范围"):
        harness["service"].start(IDENTITY, harness["live"])
    assert harness["service"].get(IDENTITY) is None


def test_duplicate_submission_never_relaunches_or_retrains(harness):
    harness["service"].start(IDENTITY, harness["live"])
    again = harness["service"].start(IDENTITY, harness["live"])
    assert again["status"] == "queued"
    assert len(harness["launches"]) == 1


def test_dead_worker_is_reported_not_silently_requeued(harness, harness2=None):
    service = harness["service"]
    record = service.start(IDENTITY, harness["live"])
    record["worker_pid"] = 999999
    service._write(record)
    harness["monkeypatch"].setattr(execution_module, "_pid_alive", lambda pid: False)
    result = service.start(IDENTITY, harness["live"])
    assert result["status"] == "failed"
    assert "worker.log" in result["message"] or "后台执行进程" in result["message"]
    assert harness["launches"] and len(harness["launches"]) == 1


def test_warning_ack_resume_requires_explicit_acknowledgement(harness):
    service = harness["service"]
    record = service.start(IDENTITY, harness["live"])
    record["status"] = "awaiting_warning_ack"
    record["run_id"] = "wb-" + "1" * 32
    service._write(record)
    unchanged = service.start(IDENTITY, harness["live"])
    assert unchanged["status"] == "awaiting_warning_ack"
    assert len(harness["launches"]) == 1
    resumed = service.start(IDENTITY, harness["live"], acknowledge_warnings=True)
    assert resumed["status"] == "queued"
    assert resumed["options"]["acknowledge_warnings"] is True
    assert len(harness["launches"]) == 2


def test_terminal_states_are_final(harness):
    service = harness["service"]
    record = service.start(IDENTITY, harness["live"])
    record["status"] = "completed"
    record["evaluation_id"] = "e" * 32
    service._write(record)
    duplicate = service.start(IDENTITY, harness["live"])
    assert duplicate["evaluation_id"] == record["evaluation_id"]
    assert len(harness["launches"]) == 1
    blocked = service.start(IDENTITY, harness["live"])  # completed stays idempotent
    assert blocked["status"] == "completed"
    for terminal in ("blocked", "failed", "stopped"):
        record["status"] = terminal
        service._write(record)
        with pytest.raises(ValueError, match="不能重复提交"):
            service.start(IDENTITY, harness["live"])


def test_stop_touches_request_stops_training_and_is_final(harness):
    service = harness["service"]
    record = service.start(IDENTITY, harness["live"])
    record["status"] = "training"
    record["run_id"] = "wb-" + "2" * 32
    service._write(record)
    stopped = []
    training_instances = []

    class TrainingFake:
        def __init__(self, training_root, project_root=None, python_executable=None):
            training_instances.append(self)

        def stop(self, run_id):
            stopped.append(run_id)

    harness["monkeypatch"].setattr(execution_module, "TrainingRunService", TrainingFake)
    result = service.stop(IDENTITY)
    assert result["status"] == "stopped"
    assert stopped == [record["run_id"]]
    assert (service._directory(IDENTITY) / "stop.request").exists()
    with pytest.raises(ValueError, match="不能重复提交"):
        service.start(IDENTITY, harness["live"])
    assert service.stop(IDENTITY)["status"] == "stopped"


def test_stop_unknown_iteration_is_an_error(harness):
    with pytest.raises(ValueError, match="尚未提交自动执行"):
        harness["service"].stop("it-" + "b" * 32)


def test_worker_honors_preexisting_stop_request_without_advancing(harness):
    service = harness["service"]
    record = service.start(IDENTITY, harness["live"])
    record["status"] = "queued"
    service._write(record)
    (service._directory(IDENTITY) / "stop.request").touch()
    assert service.run_worker(IDENTITY) is False
    assert service.get(IDENTITY)["status"] == "stopped"


def test_authorize_requires_verified_label_verification(harness):
    """数据修订后盲标核验失效时,自动执行在授权入口即被拒绝,不起后台进程。"""
    service = harness["service"]
    live = harness["live"]
    live.label_verification = None
    identity = "it-" + "b" * 32
    harness["iterations"][identity] = {
        "iteration_id": identity,
        "session_id": live.session_id,
        "goal": live.goal,
        "status": "confirmed",
    }
    with pytest.raises(ValueError, match="盲标核验"):
        service.start(identity, live)
    assert len(harness["launches"]) == 0
