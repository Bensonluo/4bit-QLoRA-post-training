"""One opt-in technical retry for proven training-step CUDA/MPS memory failures."""

from __future__ import annotations

import json
import math
from pathlib import Path

from src.tracking.runner import _metadata_lock
from src.workbench.materialize import dataset_is_current
from src.workbench.training_runs import TrainingRunService, write_json


def failure_evidence(exc):
    """Use the exception and traceback, never user data or a log substring alone."""
    message = str(exc).lower()
    backend = (
        "cuda"
        if message.startswith("cuda out of memory")
        else "mps"
        if message.startswith("mps backend out of memory")
        else "torch"
        if type(exc).__name__ == "OutOfMemoryError" and type(exc).__module__.startswith("torch")
        else None
    )
    frames, excluded = [], []
    cursor = exc.__traceback__
    while cursor:
        module = cursor.tb_frame.f_globals.get("__name__", "")
        name = cursor.tb_frame.f_code.co_name
        if module == "transformers.trainer" and name in ("training_step", "_inner_training_loop"):
            frames.append(name)
        if (
            name
            in {
                "save_model",
                "save_pretrained",
                "_save_checkpoint",
                "_save",
                "_save_optimizer_and_scheduler",
                "_save_rng_state",
                "_save_scaler",
                "evaluation_loop",
                "prediction_step",
                "prepare_data",
            }
            or (module == "torch.serialization" and name == "save")
            or "dataloader" in name.lower()
            or module.startswith(("torch.utils.data", "datasets.", "src.data."))
        ):
            excluded.append(name)
        cursor = cursor.tb_next
    if backend and frames and not excluded:
        return {"kind": "training_oom", "backend": backend, "training_frames": frames}
    return None


class TrainingRecoveryService:
    def __init__(self, training_root, project_root=None, python_executable=None):
        self.training = TrainingRunService(training_root, project_root, python_executable)

    def inspect(self, run_id, session):
        run = self.training.get_status(run_id)
        directory = self.training._directory(run_id)
        evidence = (run.get("failure") or {}).get("technical")
        result = {
            "status": "not_eligible",
            "parent_run_id": run_id,
            "child_run_id": None,
            "reason": "",
            "changes": {},
            "evidence": evidence,
        }
        if (directory / "stop.request").exists():
            result.update(status="cancelled", reason="用户停止请求已取消技术恢复。")
        elif run.get("recovery_parent_run_id"):
            result["reason"] = "技术重试运行不能再创建重试，最多恢复一次。"
        elif run.get("recover_technical_failures") is not True:
            result["reason"] = "本次启动没有授权自动技术恢复。"
        elif (
            run["status"] != "failed"
            or (run.get("failure") or {}).get("stage") != "training"
            or not evidence
            or evidence.get("kind") != "training_oom"
            or not evidence.get("training_frames")
        ):
            result["reason"] = "没有可证明的训练步骤内存不足证据；需查看失败原因。"
        elif (
            not dataset_is_current(session)
            or session.session_id != run["session_id"]
            or session.dataset.model_dump() != run["dataset"]
        ):
            result["reason"] = "数据或业务方案已变化，不能沿用原授权恢复。"
        else:
            config = run["config"]["training"]
            batch, accumulation = config["batch_size"], config["gradient_accumulation_steps"]
            if (
                type(batch) is not int
                or type(accumulation) is not int
                or batch < 1
                or accumulation < 1
            ):
                result["reason"] = "原批次配置无效，不能安全调整。"
            elif batch > 1:
                divisor = next(
                    (n for n in range(2, math.isqrt(batch) + 1) if batch % n == 0), batch
                )
                result.update(
                    status="eligible",
                    reason="降低微批次并增加梯度累积，保持有效批次大小。",
                    changes={
                        "batch_size": {"before": batch, "after": batch // divisor},
                        "gradient_accumulation_steps": {
                            "before": accumulation,
                            "after": accumulation * divisor,
                        },
                    },
                )
            elif not config["gradient_checkpointing"]:
                result.update(
                    status="eligible",
                    reason="微批次已为 1，尝试开启梯度检查点。",
                    changes={"gradient_checkpointing": {"before": False, "after": True}},
                )
            else:
                result["reason"] = "微批次已为 1 且启用梯度检查点，没有本轮允许的自动调整。"
        existing = directory / "recovery.json"
        if existing.exists():
            stored = json.loads(existing.read_text())
            if stored.get("child_run_id") or (directory / "recovery.claim").exists():
                return {
                    **stored,
                    **(
                        {"status": "cancelled", "reason": result["reason"]}
                        if result["status"] == "cancelled"
                        else {}
                    ),
                }
        return result

    def recover(self, run_id, session, acknowledge_warnings=False):
        directory = self.training._directory(run_id)
        with _metadata_lock(directory / "recovery.lock"):
            report = self.inspect(run_id, session)
            if report["status"] != "eligible":
                return report
            try:
                (directory / "recovery.claim").touch(exist_ok=False)
            except FileExistsError:
                return json.loads((directory / "recovery.json").read_text())
            report["status"] = "preparing_retry"
            write_json(directory / "recovery.json", report)
            try:
                parent = self.training.get_status(run_id)
                config = parent["config"]
                training = {
                    key: value for key, value in config["training"].items() if key != "output_dir"
                }
                training.update({key: delta["after"] for key, delta in report["changes"].items()})
                child = self.training.prepare(
                    session,
                    parent["model_path"],
                    max_length=config["model"]["max_length"],
                    training_options=training,
                    lora_options=dict(config["lora"]),
                    model_options={
                        key: value
                        for key, value in config["model"].items()
                        if key not in {"name", "max_length", "trust_remote_code"}
                    },
                )
                child["recovery_parent_run_id"] = run_id
                child["recover_technical_failures"] = False
                child["recovery_changes"] = report["changes"]
                write_json(self.training._directory(child["run_id"]) / "run.json", child)
                report["child_run_id"] = child["run_id"]
                write_json(directory / "recovery.json", report)
                if child["status"] != "prepared":
                    report.update(status="retry_blocked", reason="技术重试准备未通过，未启动。")
                elif (
                    child["model_identity"] != parent["model_identity"]
                    or child["config"]["model"] != config["model"]
                    or child["config"]["lora"] != config["lora"]
                    or child["config"]["data"] != config["data"]
                    or any(
                        child["config"]["training"][key] != value
                        for key, value in config["training"].items()
                        if key not in {"output_dir", *report["changes"]}
                    )
                ):
                    report.update(
                        status="retry_blocked",
                        reason="准备过程中模型、数据或未授权训练参数发生变化，未启动。",
                    )
                elif (directory / "stop.request").exists():
                    self.training.stop(child["run_id"])
                    report.update(status="cancelled", reason="用户已停止，未启动技术重试。")
                else:
                    child = self.training.start(
                        child["run_id"],
                        session,
                        acknowledge_warnings=acknowledge_warnings,
                        recover_technical_failures=False,
                    )
                    report["status"] = (
                        "retry_started" if child["status"] == "running" else "retry_blocked"
                    )
            except (ValueError, TypeError, OSError, ImportError) as exc:
                report.update(status="retry_blocked", reason=f"技术恢复无法继续：{exc}")
            write_json(directory / "recovery.json", report)
            return report


def launch_dispatcher(directory, record):
    """Launch a lightweight observer; the observer waits for this worker to release memory."""
    import os
    import subprocess
    import sys

    import psutil

    directory = Path(directory)
    failure = json.loads((directory / "result.json").read_text()).get("failure") or {}
    if (
        failure.get("stage") != "training"
        or (failure.get("technical") or {}).get("kind") != "training_oom"
    ):
        return None
    with _metadata_lock(directory / "recovery.lock"):
        if (directory / "stop.request").exists() or (directory / "recovery.dispatch").exists():
            return None
        (directory / "recovery.dispatch").touch(exist_ok=False)
        report = {
            "status": "dispatch_waiting",
            "parent_run_id": record["run_id"],
            "child_run_id": None,
            "reason": "等待失败工作进程退出并释放模型内存，再检查一次技术恢复。",
            "changes": {},
            "evidence": failure["technical"],
        }
        write_json(directory / "recovery.json", report)
        try:
            process = subprocess.Popen(
                [
                    sys.executable,
                    str(Path(__file__).resolve().parents[2] / "scripts" / "workbench_recover.py"),
                    "--run-dir",
                    str(directory),
                    "--parent-pid",
                    str(os.getpid()),
                    "--parent-created",
                    str(psutil.Process().create_time()),
                ],
                start_new_session=True,
            )
            report["dispatcher_pid"] = process.pid
            write_json(directory / "recovery.json", report)
        except (OSError, ValueError) as exc:
            report.update(status="dispatch_failed", reason=f"技术恢复调度器未能启动：{exc}")
            write_json(directory / "recovery.json", report)
        return report


def dispatch_after_exit(run_dir, parent_pid, parent_created, timeout=120):
    """Wait without importing torch; PID reuse or uncertain exit never starts training."""
    import os
    import time

    import psutil

    directory = Path(run_dir).resolve()
    deadline = time.monotonic() + timeout
    while True:
        if (directory / "stop.request").exists():
            status, reason = "cancelled", "用户已取消自动技术恢复。"
            break
        try:
            if os.name == "posix":
                os.kill(parent_pid, 0)
            parent = psutil.Process(parent_pid)
            if parent.create_time() != parent_created:
                status, reason = "dispatch_failed", "父工作进程身份已变化，无法确认安全恢复。"
                break
            if parent.status() == psutil.STATUS_ZOMBIE:
                break
        except (ProcessLookupError, psutil.NoSuchProcess):
            break
        except (PermissionError, psutil.AccessDenied):
            status, reason = "dispatch_failed", "无法确认父工作进程已退出，未启动技术重试。"
            break
        if time.monotonic() >= deadline:
            status, reason = "dispatch_failed", "等待父工作进程退出超时，未启动技术重试。"
            break
        time.sleep(0.1)
    else:
        raise AssertionError("unreachable")
    if "status" in locals():
        with _metadata_lock(directory / "recovery.lock"):
            path = directory / "recovery.json"
            report = json.loads(path.read_text())
            report.update(status=status, reason=reason)
            write_json(path, report)
            return report
    from src.workbench.intake_models import IntakeSession

    record = json.loads((directory / "run.json").read_text())
    execution = record["execution"]
    service = TrainingRecoveryService(
        directory.parent, execution["project_root"], execution["python_executable"]
    )
    session = IntakeSession.model_validate_json((directory / "session.json").read_text())
    return service.recover(
        record["run_id"], session, acknowledge_warnings=record.get("acknowledge_warnings", False)
    )
