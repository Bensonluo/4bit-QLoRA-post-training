"""A prepared run must not bypass a newer blind-label check."""

from pathlib import Path
from unittest.mock import patch

import pytest

from src.workbench.iterations import IterationService
from tests.unit import test_workbench_training_runs as training_tests


@pytest.fixture
def training_environment(tmp_path, monkeypatch):
    return training_tests.environment.__wrapped__(tmp_path, monkeypatch)


def snapshot(root):
    return {
        str(path.relative_to(root)): path.read_bytes() if path.is_file() else None
        for path in Path(root).rglob("*")
    }


def pending_check(intake, session):
    check = intake.start_label_verification(session.session_id, session.revision, sample_size=1)
    return check, intake.load(session.session_id)


@pytest.mark.parametrize("state", ["pending", "failed", "stale", "missing"])
def test_prepared_training_rechecks_current_blind_verdict_before_any_launch(
    training_environment, state
):
    intake, verified, training, _ = training_environment
    record = training_tests._prepare(training_environment)
    assert record["status"] == "prepared"
    if state in {"pending", "failed"}:
        check, session = pending_check(intake, verified)
        if state == "failed":
            answers = {item["row_id"]: "与样例标签不一致的答案" for item in check["items"]}
            result = intake.submit_label_verification(
                session.session_id, check["verification_id"], answers
            )
            assert result["verdict"] == "insufficient_agreement"
            session = intake.load(session.session_id)
        else:
            assert session.label_verification["status"] == "pending"
    else:
        session = verified.model_copy(
            update={
                "label_verification": {"stale": True, "previous_verdict": "verified"}
                if state == "stale"
                else None
            },
            deep=True,
        )
    before = snapshot(training.root)
    with patch("src.tracking.runner.TrainingRunner.launch_training") as launch:
        with pytest.raises(ValueError, match="盲标核验"):
            training.start(record["run_id"], session)
        launch.assert_not_called()
    assert snapshot(training.root) == before
    assert not (training.root / record["run_id"] / "started").exists()


def test_prepared_training_still_starts_when_current_blind_verdict_is_verified(
    training_environment,
):
    _, session, training, _ = training_environment
    record = training_tests._prepare(training_environment)
    with patch("src.tracking.runner.TrainingRunner.launch_training", return_value="stub") as launch:
        result = training.start(record["run_id"], session)
    assert result["status"] == "running"
    launch.assert_called_once()


@pytest.mark.parametrize(
    "verification",
    [
        {"verdict": "verified", "status": "pending"},
        {"verdict": "verified", "status": "stale"},
        {"verdict": "verified", "status": None},
        {"verdict": "verified", "status": "completed", "stale": True},
        {"verdict": "verified", "status": "completed", "stale": "unknown"},
    ],
)
def test_contradictory_blind_verification_cannot_prepare_or_start(
    training_environment, verification
):
    _, verified, training, model = training_environment
    prepared = training_tests._prepare(training_environment)
    current = verified.model_copy(update={"label_verification": verification}, deep=True)
    blocked = training.prepare(current, model, max_length=32)
    assert blocked["status"] == "blocked"
    assert any("盲标核验" in issue["message"] for issue in blocked["issues"])
    before = snapshot(training.root)
    with patch("src.tracking.runner.TrainingRunner.launch_training") as launch:
        with pytest.raises(ValueError, match="盲标核验"):
            training.start(prepared["run_id"], current)
        launch.assert_not_called()
    assert snapshot(training.root) == before
    assert not (training.root / prepared["run_id"] / "started").exists()


def test_legacy_verdict_only_blind_verification_remains_accepted(training_environment):
    _, verified, training, _ = training_environment
    prepared = training_tests._prepare(training_environment)
    legacy = verified.model_copy(update={"label_verification": {"verdict": "verified"}}, deep=True)
    with patch("src.tracking.runner.TrainingRunner.launch_training", return_value="stub") as launch:
        result = training.start(prepared["run_id"], legacy)
    assert result["status"] == "running"
    launch.assert_called_once()


def test_iteration_cannot_start_prepared_run_after_new_unverified_check(
    training_environment, tmp_path
):
    intake, verified, training, _ = training_environment
    run = training_tests._prepare(training_environment)
    iterations = IterationService(tmp_path / "iterations", training.root, tmp_path / "evaluations")
    iterations.training = training
    identity = "it-" + "a" * 32
    before_iteration = iterations._save(
        {
            "iteration_id": identity,
            "session_id": verified.session_id,
            "goal": verified.goal,
            "status": "prepared",
            "new_run_id": run["run_id"],
        }
    )
    _, current = pending_check(intake, verified)
    before = snapshot(training.root)
    with patch("src.tracking.runner.TrainingRunner.launch_training") as launch:
        with pytest.raises(ValueError, match="盲标核验"):
            iterations.start(identity, current)
        launch.assert_not_called()
    assert iterations.get(identity) == before_iteration
    assert snapshot(training.root) == before
    assert not (training.root / run["run_id"] / "started").exists()
