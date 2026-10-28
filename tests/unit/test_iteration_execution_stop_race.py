"""A stop concurrent with training handoff must not launch new work."""

import json
from pathlib import Path
from threading import Event, Thread
from unittest.mock import patch

import pytest

pytest_plugins = ["tests.unit.test_iteration_execution_live_gate"]


def test_stop_after_check_before_training_transition_does_not_start_worker(world, monkeypatch):
    service, identity, _, session, training, iteration, calls = world
    service.start(identity, session)
    stop_path = service._directory(identity) / "stop.request"
    actual_exists = Path.exists
    injected = []

    def stop_during_check(path):
        exists = actual_exists(path)
        if path == stop_path and not exists and iteration["status"] == "prepared" and not injected:
            injected.append(True)
            # The check has already observed False; stop is committed before the
            # worker acquires the transition lock. The execution row still has no run ID.
            stopped = service.stop(identity)
            assert stopped["status"] == "stopped" and not stopped.get("run_id")
        return exists

    monkeypatch.setattr(Path, "exists", stop_during_check)
    with patch("src.tracking.runner.TrainingRunner.launch_training") as launch:
        assert service.run_worker(identity) is False
    assert injected
    assert not launch.called
    assert [name for name, *_ in calls] == ["prepare"]
    assert service.get(identity)["status"] == "stopped"
    assert service.get(identity)["run_id"] == iteration["new_run_id"]
    assert not (training._directory(iteration["new_run_id"]) / "started").exists()


def test_stop_before_prepare_prevents_new_run(world, monkeypatch):
    service, identity, _, session, training, iteration, calls = world
    service.start(identity, session)
    stop_path = service._directory(identity) / "stop.request"
    actual_exists = Path.exists
    injected = []

    def stop_after_check(path):
        exists = actual_exists(path)
        if path == stop_path and not exists and not injected:
            injected.append(True)
            service.stop(identity)
        return exists

    monkeypatch.setattr(Path, "exists", stop_after_check)
    with patch("src.tracking.runner.TrainingRunner.launch_training") as launch:
        assert service.run_worker(identity) is False
    assert injected and not launch.called and not calls
    assert service.get(identity)["status"] == "stopped"
    assert not list(training.root.glob("wb-*/started"))
    assert iteration["status"] == "confirmed"


def test_stop_during_prepare_retains_and_cancels_prepared_run(world, monkeypatch):
    service, identity, _, session, training, iteration, calls = world
    service.start(identity, session)
    iterations = service._iterations()
    original = iterations.prepare

    def stop_during_prepare(*args, **kwargs):
        result = original(*args, **kwargs)
        assert service.stop(identity)["status"] == "stopped"
        return result

    monkeypatch.setattr(iterations, "prepare", stop_during_prepare)
    with patch("src.tracking.runner.TrainingRunner.launch_training") as launch:
        assert service.run_worker(identity) is False
    run_id = iteration["new_run_id"]
    assert [name for name, *_ in calls] == ["prepare"]
    assert not launch.called
    assert service.get(identity)["run_id"] == run_id
    assert (training._directory(run_id) / "stop.request").exists()
    assert not (training._directory(run_id) / "started").exists()


def test_stop_failure_is_recorded_instead_of_claiming_child_stopped(world, monkeypatch):
    service, identity, _, session, training, iteration, _ = world
    service.start(identity, session)
    iterations = service._iterations()
    original = iterations.prepare

    def stop_during_prepare(*args, **kwargs):
        result = original(*args, **kwargs)
        service.stop(identity)
        return result

    def cannot_stop(_):
        raise ValueError("simulated failure")

    monkeypatch.setattr(iterations, "prepare", stop_during_prepare)
    monkeypatch.setattr(training, "stop", cannot_stop)
    assert service.run_worker(identity) is False
    result = service.get(identity)
    assert result["status"] == "stopped"
    assert result["run_id"] == iteration["new_run_id"]
    assert "停止关联训练时出现问题：simulated failure" in result["message"]


def test_stop_returns_during_slow_preflight_and_prevents_launch(request, monkeypatch):
    from src.workbench import training_runs
    from tests.unit.test_workbench_training_runs import _prepare

    fixture = request.getfixturevalue("environment")
    _, session, training, _ = fixture
    run = _prepare(fixture)
    actual_preflight = training_runs.preflight_dataset
    entered, release = Event(), Event()
    errors = []

    def slow_preflight(*args, **kwargs):
        entered.set()
        assert release.wait(10)
        return actual_preflight(*args, **kwargs)

    def start():
        try:
            training.start(run["run_id"], session, acknowledge_warnings=True)
        except Exception as exc:
            errors.append(exc)

    monkeypatch.setattr(training_runs, "preflight_dataset", slow_preflight)
    with patch("src.tracking.runner.TrainingRunner.launch_training") as launch:
        thread = Thread(target=start, daemon=True)
        thread.start()
        try:
            assert entered.wait(10)
            stopped_run = training.stop(run["run_id"])
            assert stopped_run["status"] == "stopped"
            assert stopped_run["failure"]["type"] == "UserStopped"
        finally:
            release.set()
            thread.join(15)
        assert not thread.is_alive()
        assert len(errors) == 1 and "取消" in str(errors[0])
        assert not launch.called
        assert not (training._directory(run["run_id"]) / "started").exists()


def test_automatic_stop_does_not_wait_for_training_preflight(world, monkeypatch):
    from src.workbench import training_runs

    service, identity, _, session, training, iteration, calls = world
    service.start(identity, session)
    actual_preflight = training_runs.preflight_dataset
    entered, release, stopped = Event(), Event(), Event()
    results, errors = [], []

    def slow_start_preflight(*args, **kwargs):
        if calls and calls[-1][0] == "start":
            entered.set()
            assert release.wait(10)
        return actual_preflight(*args, **kwargs)

    def worker():
        try:
            results.append(service.run_worker(identity))
        except Exception as exc:
            errors.append(exc)

    def stop():
        try:
            results.append(service.stop(identity))
        except Exception as exc:
            errors.append(exc)
        finally:
            stopped.set()

    monkeypatch.setattr(training_runs, "preflight_dataset", slow_start_preflight)
    worker_thread = Thread(target=worker, daemon=True)
    stop_thread = Thread(target=stop, daemon=True)
    with patch("src.tracking.runner.TrainingRunner.launch_training") as launch:
        worker_thread.start()
        try:
            assert entered.wait(15)
            stop_thread.start()
            assert stopped.wait(5), "stop waited for slow preflight"
        finally:
            release.set()
            worker_thread.join(20)
            if stop_thread.ident is not None:
                stop_thread.join(20)
        assert not worker_thread.is_alive() and not stop_thread.is_alive()
        assert not errors and not launch.called
    assert False in results
    record = service.get(identity)
    assert record["status"] == "stopped"
    assert record["run_id"] == iteration["new_run_id"]
    assert not (training._directory(record["run_id"]) / "started").exists()


def test_stop_during_evaluation_does_not_bind_a_later_report(world, monkeypatch):
    service, identity, _, session, training, iteration, _ = world
    service.start(identity, session)
    iterations = service._iterations()
    original_compare = iterations.evaluations.compare
    original_status = training.get_status
    bound = []

    def training_status(run_id):
        if run_id == iteration["parent_run_id"]:
            return {
                "status": "succeeded",
                "model_path": iteration["model_path"],
                "output_dir": None,
            }
        return original_status(run_id)

    def stop_while_comparing(*args, **kwargs):
        stopped = service.stop(identity)
        assert "当前评测调用结束后不会继续" in stopped["message"]
        return original_compare(*args, **kwargs)

    monkeypatch.setattr(iterations.evaluations, "compare", stop_while_comparing)
    monkeypatch.setattr(iterations, "bind_evaluation", lambda *a, **k: bound.append(a))
    monkeypatch.setattr(training, "get_status", training_status)
    with (
        patch("src.tracking.runner.TrainingRunner.launch_training"),
        patch("src.tracking.runner.TrainingRunner.stop_training"),
    ):
        assert service.run_worker(identity) is False
    record = service.get(identity)
    assert record["status"] == "stopped", record["message"]
    assert record["run_id"] == iteration["new_run_id"]
    assert record.get("evaluation_id") is None and not bound


def test_stop_waits_only_for_inflight_launch_then_signals_started_worker(request, monkeypatch):
    from tests.unit.test_workbench_training_runs import _prepare

    fixture = request.getfixturevalue("environment")
    _, session, training, _ = fixture
    run = _prepare(fixture)
    entered, release, stopped = Event(), Event(), Event()
    errors = []
    runner = training.runner

    def launch(*_args, **_kwargs):
        entered.set()
        assert release.wait(10)

    def start():
        try:
            training.start(run["run_id"], session, acknowledge_warnings=True)
        except Exception as exc:
            errors.append(exc)

    def stop():
        try:
            training.stop(run["run_id"])
        except Exception as exc:
            errors.append(exc)
        finally:
            stopped.set()

    monkeypatch.setattr(runner, "launch_training", launch)
    monkeypatch.setattr(runner, "get_status", lambda _: "running")
    signals = []
    monkeypatch.setattr(runner, "stop_training", lambda run_id: signals.append(run_id))
    starter = Thread(target=start, daemon=True)
    stopper = Thread(target=stop, daemon=True)
    starter.start()
    try:
        assert entered.wait(10)
        stopper.start()
        assert not stopped.wait(0.15)
    finally:
        release.set()
        starter.join(15)
        if stopper.ident is not None:
            stopper.join(15)
    assert not starter.is_alive() and not stopper.is_alive()
    assert stopped.is_set() and not errors
    assert signals == [run["run_id"]]


def test_cancelled_prepared_run_is_stopped_in_ui_and_cannot_restart(request, tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest

    import ui.config
    import ui.runtime
    from src.workbench.training_runs import TrainingRunService
    from tests.unit.test_workbench_training_runs import _prepare

    fixture = request.getfixturevalue("environment")
    intake, session, training, _ = fixture
    run = _prepare(fixture)
    cancelled = training.stop(run["run_id"])
    assert cancelled["status"] == "stopped"
    assert cancelled["failure"]["type"] == "UserStopped"
    fresh = TrainingRunService(training.root, training.project_root, read_only=True)
    assert fresh.get_status(run["run_id"])["status"] == "stopped"
    assert fresh.list_runs(session.session_id)[0]["status"] == "stopped"
    with patch("src.tracking.runner.TrainingRunner.launch_training") as launch:
        with pytest.raises(ValueError, match="已取消"):
            training.start(run["run_id"], session, acknowledge_warnings=True)
        assert not launch.called

    project = tmp_path / "preview"
    workspace = project / "outputs/workbench"
    workspace.mkdir(parents=True)
    (workspace / "intake").symlink_to(intake.root, target_is_directory=True)
    (workspace / "training").symlink_to(training.root, target_is_directory=True)
    monkeypatch.setattr(ui.config, "PROJECT_ROOT", project)
    monkeypatch.setattr(ui.runtime, "DISPLAY_ONLY", False)
    page_file = Path(__file__).resolve().parents[2] / "ui/views/07_Data_Intake.py"
    page = AppTest.from_file(str(page_file), default_timeout=30).run()
    page.selectbox(key="intake_select").select(session.session_id).run()
    assert not page.exception, [error.message for error in page.exception]
    assert any(run["run_id"] in box.label and "已停止" in box.label for box in page.expander)
    assert not any(button.key == f"start_{run['run_id']}" for button in page.button)


def test_terminal_receipt_takes_precedence_over_cancelled_prepared_state(request):
    from tests.unit.test_workbench_training_runs import _prepare

    fixture = request.getfixturevalue("environment")
    _, _, training, _ = fixture
    run = _prepare(fixture)
    directory = training._directory(run["run_id"])
    (directory / "stop.request").touch()
    (directory / "result.json").write_text(
        json.dumps(
            {
                "run_id": run["run_id"],
                "status": "failed",
                "failure": {"stage": "worker", "type": "ActualFailure", "message": "real error"},
            }
        )
    )
    result = training.get_status(run["run_id"])
    assert result["status"] == "failed"
    assert result["failure"]["type"] == "ActualFailure"


@pytest.mark.parametrize(
    "runner_status,expected",
    [("unknown", "unknown"), ("running", "stopping"), ("finished", "stopped")],
)
def test_cancelled_start_marker_uses_process_evidence(
    request, monkeypatch, runner_status, expected
):
    from tests.unit.test_workbench_training_runs import _prepare

    fixture = request.getfixturevalue("environment")
    _, _, training, _ = fixture
    run = _prepare(fixture)
    directory = training._directory(run["run_id"])
    (directory / "started").touch()
    (directory / "stop.request").touch()
    monkeypatch.setattr(training.runner, "get_status", lambda _: runner_status)
    result = training.get_status(run["run_id"])
    assert result["status"] == expected
    if expected == "stopped":
        assert result["failure"]["type"] == "UserStopped"
