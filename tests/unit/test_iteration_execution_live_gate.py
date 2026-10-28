"""Automatic iteration must not launch from a stale blind-label snapshot."""

import json
import sqlite3
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from src.workbench.iteration_execution import IterationExecutionService
from tests.unit.test_workbench_training_runs import environment  # noqa: F401


@pytest.fixture
def world(request, tmp_path, monkeypatch):
    intake, session, training, model = request.getfixturevalue("environment")
    identity = "it-" + "e" * 32
    iteration = {
        "iteration_id": identity,
        "session_id": session.session_id,
        "goal": session.goal,
        "status": "confirmed",
        "evaluation_suite": session.dataset.evaluation_suite,
        "parent_run_id": "wb-" + "f" * 32,
        "model_path": str(model),
        "evaluation_protocol": {
            "scorer": "classification_exact",
            "fields": (),
            "max_new_tokens": 32,
            "strip_whitespace": True,
        },
    }
    calls = []

    class Iterations:
        def __init__(self):
            self.training = training
            self.evaluations = SimpleNamespace(compare=self.compare)

        def get(self, _):
            return iteration

        def claim_execution(self, *_):
            return iteration

        def prepare(self, _, live, *, execution_id):
            calls.append(("prepare", live.label_verification))
            run = training.prepare(
                live,
                model,
                max_length=32,
                training_options={
                    "num_epochs": 1,
                    "batch_size": 4,
                    "gradient_accumulation_steps": 1,
                },
            )
            iteration.update(status="prepared", new_run_id=run["run_id"])
            return iteration

        def start(self, _, live, *, acknowledge_warnings, execution_id):
            calls.append(("start", live.label_verification))
            training.start(iteration["new_run_id"], live, acknowledge_warnings=True)
            iteration["status"] = "running"
            return iteration

        def _effective_training_run(self, run_id):
            return {
                "status": "succeeded",
                "model_path": str(model),
                "output_dir": str(tmp_path / "adapter"),
            }

        def compare(self, *_):
            calls.append(("compare", None))
            return SimpleNamespace(status="completed", notes=[], evaluation_id="evaluation")

        def bind_evaluation(self, *_args, **_kwargs):
            return None

    fake = Iterations()
    service = IterationExecutionService(
        tmp_path / "executions",
        intake.root,
        tmp_path / "iterations",
        training.root,
        tmp_path / "evaluations",
        project_root=training.project_root,
    )
    monkeypatch.setattr(service, "_iterations", lambda: fake)
    monkeypatch.setattr(service, "_launch", lambda record: dict(record))
    monkeypatch.setattr("src.workbench.iteration_execution.RELEASE_SETTLE_SECONDS", 0)
    return service, identity, intake, session, training, iteration, calls


@pytest.mark.parametrize("moment", ["preparing", "training"])
def test_new_challenge_during_execution_blocks_before_prepare_or_launch(world, monkeypatch, moment):
    service, identity, intake, session, training, iteration, calls = world
    service.start(identity, session)
    original = service._write
    injected = []

    def write_then_begin_challenge(record):
        original(record)
        if record["status"] == moment and not injected:
            check = intake.start_label_verification(
                session.session_id, session.revision, sample_size=1
            )
            injected.append(check["verification_id"])

    monkeypatch.setattr(service, "_write", write_then_begin_challenge)
    with patch("src.tracking.runner.TrainingRunner.launch_training") as launch:
        result = service.run_worker(identity)
    assert injected
    assert intake.load(session.session_id).label_verification["status"] == "pending"
    assert not launch.called
    assert result is False
    assert service.get(identity)["status"] == "blocked"
    assert "盲标核验" in service.get(identity)["message"]
    if moment == "preparing":
        assert not calls
        assert not list(training.root.glob("wb-*/started"))
    else:
        assert [call[0] for call in calls] == ["prepare"]
        assert not (training._directory(iteration["new_run_id"]) / "started").exists()


@pytest.mark.parametrize("state", ["pending", "stale", "contradictory"])
def test_inconsistent_or_stale_verdict_cannot_authorize(world, state):
    service, identity, intake, session, _, _, _ = world
    if state in {"pending", "stale"}:
        pending = intake.start_label_verification(
            session.session_id, session.revision, sample_size=1
        )
        identity_to_change = pending["verification_id"]
    else:
        identity_to_change = intake.load(session.session_id).label_verification["verification_id"]
    with sqlite3.connect(intake.database) as connection:
        if state == "contradictory":
            result = json.loads(
                connection.execute(
                    "SELECT result FROM label_verifications WHERE verification_id=?",
                    (identity_to_change,),
                ).fetchone()[0]
            )
            result["stale"] = True
            connection.execute(
                "UPDATE label_verifications SET result=? WHERE verification_id=?",
                (json.dumps(result), identity_to_change),
            )
        else:
            connection.execute(
                "UPDATE label_verifications SET status=?, verdict='verified' WHERE verification_id=?",
                (state, identity_to_change),
            )
    current = intake.load(session.session_id)
    if state == "pending":
        assert current.label_verification["status"] == "pending"
        assert current.label_verification["verdict"] == "verified"
    else:
        assert current.label_verification["stale"] is True
    with pytest.raises(ValueError, match="盲标核验"):
        service.start(identity, current)
    assert not (service._directory(identity) / "execution.json").exists()


def test_data_change_after_training_blocks_evaluation(world, monkeypatch):
    service, identity, intake, session, training, iteration, calls = world
    service.start(identity, session)
    original = service._write
    injected = []

    def change_before_evaluation(record):
        original(record)
        if record["status"] == "evaluating" and not injected:
            intake.answer(session.session_id, "重新确认业务要求")
            injected.append(True)

    monkeypatch.setattr(service, "_write", change_before_evaluation)
    real_status = training.get_status
    monkeypatch.setattr(
        training,
        "get_status",
        lambda run_id: (
            {"status": "succeeded", "model_path": iteration["model_path"], "output_dir": "parent"}
            if run_id == iteration["parent_run_id"]
            else real_status(run_id)
        ),
    )
    with patch("src.tracking.runner.TrainingRunner.launch_training") as launch:
        assert service.run_worker(identity) is False
    assert launch.called
    assert injected
    assert service.get(identity)["status"] == "blocked"
    assert "授权后" in service.get(identity)["message"]
    assert [call[0] for call in calls] == ["prepare", "start"]
