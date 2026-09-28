"""Recovery preserves business configuration and never fabricates a host OOM."""

import json
import subprocess
import sys
import time
from pathlib import Path
from unittest.mock import patch

import psutil
import pytest

from src.workbench.training_recovery import (
    TrainingRecoveryService,
    dispatch_after_exit,
    failure_evidence,
)
from src.workbench.training_runs import write_json
from tests.unit.test_workbench_training_runs import _wait, environment  # noqa: F401


def _exception(message="CUDA out of memory. Tried to allocate 10 MiB", frame="training_step"):
    namespace = {"__name__": "transformers.trainer"}
    exec(
        compile(f"def {frame}():\n    raise RuntimeError({message!r})\n", "fixture.py", "exec"),
        namespace,
    )
    try:
        namespace[frame]()
    except RuntimeError as exc:
        return exc


@pytest.fixture
def failed(environment):  # noqa: F811
    _, session, training, model = environment
    parent = training.prepare(
        session,
        model,
        max_length=32,
        training_options={
            "batch_size": 4,
            "gradient_accumulation_steps": 1,
            "gradient_checkpointing": False,
            "num_epochs": 1,
            "logging_steps": 1,
        },
        model_options={"quantization_bits": None, "torch_dtype": "float32"},
        plan_trace=[{"tool": "training_context", "ok": True}],
    )
    parent.update(
        status="failed",
        recover_technical_failures=True,
        acknowledge_warnings=True,
        execution={
            "project_root": str(training.project_root),
            "python_executable": training.python_executable,
        },
    )
    directory = training._directory(parent["run_id"])
    write_json(directory / "run.json", parent)
    write_json(
        directory / "result.json",
        {
            "run_id": parent["run_id"],
            "status": "failed",
            "failure": {
                "stage": "training",
                "type": "RuntimeError",
                "message": str(_exception()),
                "technical": failure_evidence(_exception()),
            },
        },
    )
    recovery = TrainingRecoveryService(
        training.root, training.project_root, training.python_executable
    )
    return session, training, recovery, parent


@pytest.mark.parametrize(
    "message,frame,eligible",
    [
        ("CUDA out of memory. Tried to allocate", "training_step", True),
        ("MPS backend out of memory (MPS allocated...)", "_inner_training_loop", True),
        ("CUDA out of memory", "prepare_data", False),
        ("CUDA out of memory", "save_model", False),
        ("unrelated invalid data: CUDA out of memory", "training_step", False),
        ("tensor shape mismatch", "training_step", False),
    ],
)
def test_exception_and_actual_training_trace_are_both_required(message, frame, eligible):
    assert bool(failure_evidence(_exception(message, frame))) is eligible


def test_inspect_is_read_only_and_changes_keep_effective_batch(failed):
    session, training, recovery, parent = failed
    result = recovery.inspect(parent["run_id"], session)
    assert result["status"] == "eligible"
    assert result["changes"] == {
        "batch_size": {"before": 4, "after": 2},
        "gradient_accumulation_steps": {"before": 1, "after": 2},
    }
    assert not (training._directory(parent["run_id"]) / "recovery.claim").exists()
    assert len(training.list_runs()) == 1


@pytest.mark.parametrize(
    "batch,accum,checkpoint,expected",
    [(3, 2, False, "eligible"), (1, 2, False, "eligible"), (1, 2, True, "not_eligible")],
)
def test_prime_microbatch_and_checkpoint_fallback(failed, batch, accum, checkpoint, expected):
    session, training, recovery, parent = failed
    parent["config"]["training"].update(
        batch_size=batch, gradient_accumulation_steps=accum, gradient_checkpointing=checkpoint
    )
    write_json(training._directory(parent["run_id"]) / "run.json", parent)
    result = recovery.inspect(parent["run_id"], session)
    assert result["status"] == expected
    if batch == 3:
        assert result["changes"]["batch_size"]["after"] == 1
        assert result["changes"]["gradient_accumulation_steps"]["after"] == 6
    elif not checkpoint:
        assert result["changes"] == {"gradient_checkpointing": {"before": False, "after": True}}


@pytest.mark.parametrize(
    "change", ["unapproved", "child", "data", "artifacts", "stopped", "no_evidence"]
)
def test_non_eligible_failures_never_create_child(failed, change):
    session, training, recovery, parent = failed
    directory = training._directory(parent["run_id"])
    receipt = json.loads((directory / "result.json").read_text())
    if change == "unapproved":
        parent["recover_technical_failures"] = False
    elif change == "child":
        parent["recovery_parent_run_id"] = "another-parent"
    elif change == "data":
        session.dataset = None
    elif change == "artifacts":
        receipt["failure"]["stage"] = "artifacts"
    elif change == "stopped":
        receipt["status"] = "stopped"
    else:
        receipt["failure"].pop("technical")
    write_json(directory / "run.json", parent)
    write_json(directory / "result.json", receipt)
    assert recovery.recover(parent["run_id"], session)["status"] == "not_eligible"
    assert len(training.list_runs()) == 1


def test_single_child_preserves_all_unapproved_config_and_cannot_chain(failed):
    session, training, recovery, parent = failed
    with patch.object(recovery.training.runner, "launch_training", return_value="started"):
        result = recovery.recover(parent["run_id"], session, acknowledge_warnings=True)
    assert result["status"] == "retry_started"
    child = training._load(result["child_run_id"])
    assert child["recovery_parent_run_id"] == parent["run_id"]
    assert child["recover_technical_failures"] is False
    assert child["plan_trace"] == parent["plan_trace"]
    for section in ("model", "data", "lora"):
        assert child["config"][section] == parent["config"][section]
    for key, value in parent["config"]["training"].items():
        if key not in {"output_dir", "batch_size", "gradient_accumulation_steps"}:
            assert child["config"]["training"][key] == value
    assert recovery.recover(parent["run_id"], session)["child_run_id"] == child["run_id"]
    assert len(training.list_runs()) == 2
    assert recovery.inspect(child["run_id"], session)["status"] == "not_eligible"
    assert training.get_status(parent["run_id"])["status"] == "failed"


def test_user_stop_cancels_pending_retry_and_stops_existing_child(failed):
    session, training, recovery, parent = failed
    with patch.object(recovery.training.runner, "launch_training", return_value="started"):
        child_id = recovery.recover(parent["run_id"], session, acknowledge_warnings=True)[
            "child_run_id"
        ]
    with patch.object(training.runner, "stop_training") as stop:
        training.stop(parent["run_id"])
    stop.assert_called_once_with(child_id)
    assert (training._directory(child_id) / "stop.request").exists()
    assert recovery.recover(parent["run_id"], session)["status"] == "cancelled"
    assert len(training.list_runs()) == 2


def test_stop_before_recovery_and_dispatch_timeout_start_nothing(failed):
    session, training, recovery, parent = failed
    directory = training._directory(parent["run_id"])
    write_json(directory / "recovery.json", {"status": "dispatch_waiting"})
    result = dispatch_after_exit(
        directory, psutil.Process().pid, psutil.Process().create_time(), timeout=0
    )
    assert result["status"] == "dispatch_failed"
    training.stop(parent["run_id"])
    assert recovery.recover(parent["run_id"], session)["status"] == "cancelled"
    assert len(training.list_runs()) == 1


def test_real_dispatcher_waits_for_parent_exit_then_runs_one_real_tiny_child(failed):
    session, training, recovery, parent = failed
    directory = training._directory(parent["run_id"])
    sleeper = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(1)"])
    created = psutil.Process(sleeper.pid).create_time()
    script = Path(__file__).resolve().parents[2] / "scripts" / "workbench_recover.py"
    dispatcher = subprocess.Popen(
        [
            sys.executable,
            str(script),
            "--run-dir",
            str(directory),
            "--parent-pid",
            str(sleeper.pid),
            "--parent-created",
            str(created),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        time.sleep(0.2)
        assert len(training.list_runs()) == 1
        sleeper.wait(timeout=5)
        out, err = dispatcher.communicate(timeout=60)
        assert dispatcher.returncode == 0, (out, err)
        report = training.get_status(parent["run_id"])["recovery"]
        assert report["status"] == "retry_started"
        child = _wait(training, report["child_run_id"])
        assert child["status"] == "succeeded", training.read_logs(child["run_id"])
        assert Path(child["artifacts"]["adapter_weights"]).is_file()
        assert recovery.recover(parent["run_id"], session)["child_run_id"] == child["run_id"]
        assert len(training.list_runs()) == 2
    finally:
        if sleeper.poll() is None:
            sleeper.terminate()
            sleeper.wait(timeout=5)
        if dispatcher.poll() is None:
            dispatcher.terminate()
            dispatcher.wait(timeout=5)


def test_worker_persists_proven_failure_before_dispatching(failed, monkeypatch):
    import importlib.util
    import signal

    import yaml

    _, training, _, parent = failed
    directory = training._directory(parent["run_id"])
    config = {**parent["config"], "workbench": {"run_dir": str(directory)}}
    config_path = directory / "worker-test.yaml"
    config_path.write_text(yaml.safe_dump(config))
    worker_path = Path(__file__).resolve().parents[2] / "scripts" / "workbench_train.py"
    spec = importlib.util.spec_from_file_location("test_workbench_worker", worker_path)
    worker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker)
    monkeypatch.setattr(sys, "argv", [str(worker_path), "--config", str(config_path)])
    dispatched = []

    def observe(run_dir, record):
        receipt = json.loads((run_dir / "result.json").read_text())
        assert receipt["status"] == "failed"
        assert receipt["failure"]["technical"]["kind"] == "training_oom"
        assert record["recover_technical_failures"] is True
        dispatched.append(record["run_id"])

    handler = signal.getsignal(signal.SIGTERM)
    try:
        with (
            patch("src.training.sft_trainer.run_sft_training", side_effect=_exception()),
            patch("src.workbench.training_recovery.launch_dispatcher", side_effect=observe),
            patch.object(worker, "mlflow_reference", return_value={"mlflow_status": "unavailable"}),
        ):
            assert worker.main() == 1
    finally:
        signal.signal(signal.SIGTERM, handler)
    assert dispatched == [parent["run_id"]]


@pytest.mark.parametrize(
    "module,inner_name,eligible",
    [
        ("torch.utils.checkpoint", "checkpoint", True),
        ("transformers.trainer", "_save_checkpoint", False),
        ("torch.utils.data.dataloader", "__next__", False),
    ],
)
def test_autograd_checkpoint_oom_can_reduce_batch_but_checkpoint_saving_cannot(
    module, inner_name, eligible
):
    import torch

    inner = {"__name__": module, "oom": torch.OutOfMemoryError}
    exec(
        compile(
            f"def {inner_name}():\n    raise oom('CUDA out of memory')\n",
            "checkpoint-fixture.py",
            "exec",
        ),
        inner,
    )
    trainer = {"__name__": "transformers.trainer", "call_inner": inner[inner_name]}
    exec(compile("def training_step():\n    call_inner()\n", "trainer-fixture.py", "exec"), trainer)
    try:
        trainer["training_step"]()
    except torch.OutOfMemoryError as exc:
        assert bool(failure_evidence(exc)) is eligible
