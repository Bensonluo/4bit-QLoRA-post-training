"""Read-only workbench services inspect existing records without side effects."""

import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.tracking.runner import _ACTIVE_PROCS_BY_ROOT, TrainingRunner
from src.workbench.business_evaluation import BusinessEvaluationService
from src.workbench.evaluation_suites import EvalSuiteService
from src.workbench.readonly import ReadOnlyWalUnsupported, open_database
from src.workbench.training_plans import TrainingPlanService
from src.workbench.training_runs import TrainingRunService


def snapshot(root):
    return {
        str(path.relative_to(root)): path.read_bytes() if path.is_file() else None
        for path in root.rglob("*")
    }


def test_readonly_sqlite_uses_existing_path_with_special_characters(tmp_path):
    path = tmp_path / "存储 #100% ?.sqlite"
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE items (value TEXT)")
        connection.execute("INSERT INTO items VALUES ('before')")
    before = path.read_bytes()
    with open_database(path, read_only=True) as connection:
        assert connection.execute("SELECT value FROM items").fetchall() == [("before",)]
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            connection.execute("INSERT INTO items VALUES ('after')")
    assert path.read_bytes() == before
    absent = tmp_path / "missing.sqlite"
    with pytest.raises(sqlite3.OperationalError):
        open_database(absent, read_only=True)
    assert not absent.exists()


def test_wal_database_is_not_read_as_stale_or_allowed_to_write_sidecars(tmp_path):
    path = tmp_path / "wal.sqlite"
    writer = sqlite3.connect(path)
    try:
        writer.execute("PRAGMA journal_mode=WAL")
        writer.execute("CREATE TABLE items (value TEXT)")
        writer.execute("INSERT INTO items VALUES ('latest uncheckpointed value')")
        writer.commit()
        before = snapshot(tmp_path)
        assert "wal.sqlite-wal" in before
        with pytest.raises(ReadOnlyWalUnsupported, match="WAL"):
            open_database(path, read_only=True)
        assert snapshot(tmp_path) == before
        assert writer.execute("SELECT value FROM items").fetchall() == [
            ("latest uncheckpointed value",)
        ]
    finally:
        writer.close()


@pytest.mark.parametrize("suffix", ["-wal", "-shm"])
def test_retired_wal_sidecar_is_not_repaired_or_removed(tmp_path, suffix):
    path = tmp_path / "ordinary.sqlite"
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE items (value TEXT)")
    Path(str(path) + suffix).write_bytes(b"retained log")
    before = snapshot(tmp_path)
    with pytest.raises(ReadOnlyWalUnsupported, match="WAL"):
        open_database(path, read_only=True)
    assert snapshot(tmp_path) == before


def test_readonly_services_do_not_initialize_or_run_work(tmp_path):
    root = tmp_path / "empty"
    training = TrainingRunService(root / "training", root, read_only=True)
    evaluations = BusinessEvaluationService(root / "evaluations", read_only=True)
    suites = EvalSuiteService(root / "suites", read_only=True)
    plans = TrainingPlanService(root / "plans", root / "training", read_only=True)
    assert training.list_runs() == []
    assert evaluations.list_reports() == []
    assert suites.list_suites() == []
    with pytest.raises(PermissionError):
        training.prepare(None, "unused")
    with pytest.raises(PermissionError):
        training.start("unused", None)
    with pytest.raises(PermissionError):
        training.stop("unused")
    with pytest.raises(PermissionError):
        evaluations.compare(None, [], None)
    with pytest.raises(PermissionError):
        evaluations.evaluate_final(None, None, None)
    with pytest.raises(PermissionError):
        suites.freeze(None)
    with pytest.raises(PermissionError):
        plans.context(None, [])
    with pytest.raises(PermissionError):
        plans.probe(None, "unused", 1)
    assert not root.exists()


def test_readonly_runner_observes_finished_process_without_persisting_exit(tmp_path, monkeypatch):
    root = tmp_path / "workspace"
    root.mkdir()
    active = {"demo-run": SimpleNamespace(poll=lambda: 0)}
    monkeypatch.setitem(_ACTIVE_PROCS_BY_ROOT, str(root.resolve()), active)
    before = snapshot(root)
    runner = TrainingRunner(str(root), read_only=True)
    assert runner.get_status("demo-run") == "finished"
    assert runner.list_active() == []
    with pytest.raises(PermissionError):
        runner.launch_training("sft", {}, "blocked")
    with pytest.raises(PermissionError):
        runner.launch_eval("demo-run", "unused")
    with pytest.raises(PermissionError):
        runner.launch_entity_eval("demo-run", "unused", "unused")
    with pytest.raises(PermissionError):
        runner.stop_training("demo-run")
    with pytest.raises(PermissionError):
        runner.delete_run("demo-run")
    with pytest.raises(PermissionError):
        runner._record_exit("demo-run", 0)
    with pytest.raises(PermissionError):
        runner._cleanup_finished()
    assert snapshot(root) == before
    assert active == {"demo-run": active["demo-run"]}


def test_readonly_training_status_does_not_initialize_runner_or_rewrite_metadata(
    tmp_path, monkeypatch
):
    import json

    root = tmp_path / "workspace"
    run_id = "wb-" + "a" * 32
    directory = root / "training" / run_id
    directory.mkdir(parents=True)
    (directory / "run.json").write_text(
        json.dumps({"run_id": run_id, "session_id": "s1", "status": "running"})
    )
    active = {run_id: SimpleNamespace(poll=lambda: 1)}
    monkeypatch.setitem(_ACTIVE_PROCS_BY_ROOT, str(root.resolve()), active)
    before = snapshot(root)
    service = TrainingRunService(root / "training", root, read_only=True)
    status = service.get_status(run_id)
    assert status["status"] == "failed"
    assert status["failure"]["type"] == "MissingResult"
    assert snapshot(root) == before
    assert not (root / "outputs/configs").exists()
