"""One explicit authorization executes a confirmed iteration end to end.

The service owns orchestration facts only (status, ids, authorization options,
worker process). Business rules stay in IntakeService / IterationService /
TrainingRunService / BusinessEvaluationService — this module re-uses them and
never re-implements a rule. The background loop lives here so tests can drive
it in-process; ``scripts/workbench_iterate.py`` is a thin subprocess wrapper.

Authorization is bound to the session snapshot taken at ``start()``. If the
business goal, sources, analysis, or confirmed full data change afterwards,
the worker blocks instead of silently reusing the old authorization.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
import traceback
from contextlib import suppress
from datetime import datetime, timezone
from pathlib import Path

from src.tracking.runner import _metadata_lock, _pid_alive
from src.workbench.business_evaluation import EvaluationProtocol
from src.workbench.intake_models import IntakeSession
from src.workbench.intake_service import IntakeService
from src.workbench.iterations import IterationService
from src.workbench.sources import content_digest
from src.workbench.training_runs import TrainingRunService, write_json

MANAGED = {
    "queued",
    "materializing",
    "preparing",
    "awaiting_warning_ack",
    "training",
    "waiting_for_release",
    "evaluating",
}
TERMINAL = {"completed", "blocked", "failed", "stopped"}
POLL_SECONDS = 0.5
# The training worker writes result.json in its finally block; give the process
# a moment to fully exit so evaluation starts with released model memory.
RELEASE_SETTLE_SECONDS = 1.0


def _protocol_from_iteration(record: dict) -> EvaluationProtocol:
    """Rebuild the parent-round evaluation protocol stored at proposal time."""
    stored = record["evaluation_protocol"]
    return EvaluationProtocol(
        scorer=stored["scorer"],
        fields=tuple(stored.get("fields") or ()),
        max_new_tokens=stored["max_new_tokens"],
        strip_whitespace=stored["strip_whitespace"],
        custom_scoring=stored.get("custom_scoring"),
    )


def _sources_digest(session: IntakeSession) -> str | None:
    if not session.sources:
        return None
    return content_digest(
        {alias: source.digest for alias, source in sorted(session.sources.items())}
    )


def _attribute_digest(value: object) -> str | None:
    dump = getattr(value, "model_dump", None)
    return content_digest(dump()) if dump is not None else None


def authorization_valid(live: IntakeSession, snapshot: IntakeSession) -> bool:
    """Authorization scope: goal, sources, analysis and confirmed full data."""
    if live.session_id != snapshot.session_id or live.goal != snapshot.goal:
        return False
    if _sources_digest(live) != _sources_digest(snapshot):
        return False
    return all(
        _attribute_digest(getattr(live, name)) == _attribute_digest(getattr(snapshot, name))
        for name in ("analysis", "full_data")
    )


class IterationExecutionService:
    def __init__(
        self,
        root: str | Path,
        intake_root: str | Path,
        iteration_root: str | Path,
        training_root: str | Path,
        evaluation_root: str | Path,
        project_root: str | Path | None = None,
        python_executable: str | None = None,
    ):
        self.root = Path(root).resolve()
        self.intake_root = Path(intake_root).resolve()
        self.iteration_root = Path(iteration_root).resolve()
        self.training_root = Path(training_root).resolve()
        self.evaluation_root = Path(evaluation_root).resolve()
        self.project_root = (
            Path(project_root).resolve() if project_root else Path(__file__).resolve().parents[2]
        )
        self.python_executable = python_executable or sys.executable

    # ------------------------------------------------------------------ io

    def _directory(self, iteration_id: str) -> Path:
        if not re.fullmatch(r"it-[0-9a-f]{32}", iteration_id):
            raise ValueError("无效改进轮次 ID。")
        return self.root / iteration_id

    def _read(self, iteration_id: str) -> dict | None:
        path = self._directory(iteration_id) / "execution.json"
        if not path.exists():
            return None
        data: object = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None

    def _write(self, record: dict) -> None:
        record["updated_at"] = datetime.now(timezone.utc).isoformat()
        write_json(self._directory(record["iteration_id"]) / "execution.json", record)

    def _update(self, record: dict, **fields: object) -> dict:
        record.update(**fields)
        self._write(record)
        return record

    # ------------------------------------------------------------ services

    def _training(self) -> TrainingRunService:
        return TrainingRunService(self.training_root, self.project_root, self.python_executable)

    def _iterations(self) -> IterationService:
        service = IterationService(self.iteration_root, self.training_root, self.evaluation_root)
        service.training = self._training()
        return service

    # ------------------------------------------------------------- public

    def get(self, iteration_id: str) -> dict | None:
        """Pure read of the execution record; never advances execution."""
        return self._read(iteration_id)

    def start(
        self,
        iteration_id: str,
        session: IntakeSession,
        *,
        acknowledge_warnings: bool = False,
        independent_rows_confirmed: bool = False,
    ) -> dict:
        """Authorize (or resume) the automatic handoff and launch the worker."""
        if type(acknowledge_warnings) is not bool or type(independent_rows_confirmed) is not bool:
            raise ValueError("确认项必须明确为布尔值。")
        directory = self._directory(iteration_id)
        directory.mkdir(parents=True, exist_ok=True)
        with _metadata_lock(directory / "claim.lock"):
            record = self._read(iteration_id)
            if record is None:
                record = self._authorize(
                    iteration_id,
                    session,
                    acknowledge_warnings=acknowledge_warnings,
                    independent_rows_confirmed=independent_rows_confirmed,
                )
            elif record["status"] == "awaiting_warning_ack":
                if not acknowledge_warnings:
                    # Still waiting for the user's explicit review; no relaunch.
                    return record
                record = self._update(
                    record,
                    status="queued",
                    options={
                        **record["options"],
                        "acknowledge_warnings": True,
                    },
                    message="已确认预检提示，继续执行到开发集对照。",
                )
            elif record["status"] in MANAGED:
                pid = record.get("worker_pid")
                if pid and not _pid_alive(int(pid)):
                    return self._update(
                        record,
                        status="failed",
                        message=(
                            "后台执行进程已退出且未完成；请查看执行日志 worker.log，"
                            "处理问题后提出新的改进轮次。"
                        ),
                    )
                # Duplicate submission while in progress: never retrain.
                return record
            elif record["status"] == "completed":
                # A finished handoff is idempotent: the same report comes back
                # and no second training run is ever created.
                return record
            else:
                raise ValueError(
                    f"本轮自动执行已结束（{record['status']}），不能重复提交；"
                    "请查看结果或提出新的改进轮次。"
                )
            return self._launch(record)

    def stop(self, iteration_id: str) -> dict:
        """Stop the handoff and the related training; further starts are refused."""
        directory = self._directory(iteration_id)
        if not (directory / "execution.json").exists():
            raise ValueError("本轮尚未提交自动执行。")
        with _metadata_lock(directory / "claim.lock"):
            record = self._read(iteration_id)
            if record is None:
                raise ValueError("本轮尚未提交自动执行。")
            if record["status"] in TERMINAL:
                return record
            (directory / "stop.request").touch()
            message = "已请求停止自动执行。"
            if record.get("run_id"):
                try:
                    self._training().stop(record["run_id"])
                    message = "已请求停止自动执行，并停止关联训练。"
                except (ValueError, OSError) as exc:
                    message = f"已请求停止自动执行；停止关联训练时出现问题：{exc}"
            return self._update(record, status="stopped", message=message)

    # ------------------------------------------------------------ private

    def _authorize(
        self,
        iteration_id: str,
        session: IntakeSession,
        *,
        acknowledge_warnings: bool,
        independent_rows_confirmed: bool,
    ) -> dict:
        iterations = self._iterations()
        iteration = iterations.get(iteration_id)
        if iteration["session_id"] != session.session_id or iteration["goal"] != session.goal:
            raise ValueError("改进轮次与当前任务不匹配。")
        if iteration["status"] != "confirmed":
            raise ValueError("只有已确认范围的改进轮次可以提交自动执行。")
        live = IntakeService(self.intake_root).load(session.session_id)
        if live.revision != session.revision:
            raise ValueError("任务已更新，请读取当前版本后再提交自动执行。")
        if not (
            live.full_data
            and live.full_data.status == "confirmed"
            and live.full_data.confirmed_revision is not None
        ):
            raise ValueError("请先核对并确认全量数据，再提交自动执行。")
        iterations.claim_execution(iteration_id, iteration_id)
        record = {
            "iteration_id": iteration_id,
            "session_id": session.session_id,
            "session_revision": session.revision,
            "status": "queued",
            "message": "已受理，等待后台执行。",
            "run_id": None,
            "evaluation_id": None,
            "issues": [],
            "options": {
                "acknowledge_warnings": acknowledge_warnings,
                "independent_rows_confirmed": independent_rows_confirmed,
            },
            "worker_pid": None,
            "intake_root": str(self.intake_root),
            "iteration_root": str(self.iteration_root),
            "training_root": str(self.training_root),
            "evaluation_root": str(self.evaluation_root),
            "project_root": str(self.project_root),
            "python_executable": str(self.python_executable),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        self._write(record)
        (self._directory(iteration_id) / "session.json").write_text(
            session.model_dump_json(), encoding="utf-8"
        )
        return record

    def _launch(self, record: dict) -> dict:
        script = Path(record["project_root"]) / "scripts" / "workbench_iterate.py"
        if not script.is_file():
            return self._update(record, status="failed", message=f"缺少后台执行脚本：{script}。")
        log_path = self._directory(record["iteration_id"]) / "worker.log"
        with open(log_path, "a", encoding="utf-8") as log_handle:
            process = subprocess.Popen(
                [
                    record["python_executable"],
                    str(script),
                    "--execution-id",
                    record["iteration_id"],
                    "--execution-root",
                    str(self.root),
                ],
                cwd=record["project_root"],
                env=os.environ.copy(),
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        return self._update(
            record,
            status="queued",
            worker_pid=process.pid,
            message="后台执行已启动；关闭页面不影响执行，可随时查看状态或停止。",
        )

    # ------------------------------------------------------------- worker

    def run_worker(self, iteration_id: str) -> bool:
        """Advance one authorized execution to a terminal state in this process."""
        directory = self._directory(iteration_id)
        record = self._read(iteration_id)
        if record is None:
            raise ValueError("本轮尚未提交自动执行。")

        def stop_requested() -> bool:
            return (directory / "stop.request").exists()

        def write_transition(status: str, terminal: bool, **fields: object) -> dict | None:
            """Write a state change; a stop or terminal state that already landed wins."""
            with _metadata_lock(directory / "claim.lock"):
                current = self._read(iteration_id)
                if current is None:
                    return None
                if current["status"] in TERMINAL and current["status"] != status:
                    return current
                if terminal and stop_requested() and status != "stopped":
                    return current
                current.update(status=status, **fields)
                self._write(current)
                return current

        iterations = self._iterations()
        training = iterations.training
        intake = IntakeService(self.intake_root)

        def stop_now(reason: str) -> bool:
            if record_local.get("run_id"):
                with suppress(ValueError, OSError):
                    training.stop(record_local["run_id"])
            write_transition("stopped", terminal=True, message=reason)
            return False

        record_local = dict(record)
        try:
            if stop_requested():
                record_local.update(run_id=record.get("run_id"))
                return stop_now("停止请求在执行开始前已存在，本轮未继续。")
            iteration = iterations.get(iteration_id)
            live = intake.load(iteration["session_id"])
            snapshot = IntakeSession.model_validate_json(
                (directory / "session.json").read_text(encoding="utf-8")
            )
            if not authorization_valid(live, snapshot):
                write_transition(
                    "blocked",
                    terminal=True,
                    message=(
                        "授权后业务目标、资料或数据方案已变化，原授权不能继续；"
                        "请重新核对后提出新的改进轮次。"
                    ),
                )
                return False

            needs_materialization = iteration["status"] == "confirmed" and (
                live.dataset is None
                or live.dataset.evaluation_suite != iteration["evaluation_suite"]
            )
            if needs_materialization:
                if live.analysis is None or live.analysis.recipe is None:
                    write_transition(
                        "blocked",
                        terminal=True,
                        message="当前任务缺少已确认的数据方案，不能准备固定题集数据。",
                    )
                    return False
                write_transition(
                    "materializing",
                    terminal=False,
                    message="正在用本轮固定题集准备数据版本。",
                )
                independent = (
                    bool(live.analysis.recipe.group_columns)
                    or record["options"]["independent_rows_confirmed"]
                )
                live = intake.materialize_dataset(
                    live.session_id,
                    live.revision,
                    evaluation_suite=iteration["evaluation_suite"],
                    independent_rows_confirmed=independent,
                )
                iteration = iterations.get(iteration_id)
                if stop_requested():
                    return stop_now("已按请求停止自动执行。")

            if iteration["status"] == "confirmed":
                write_transition(
                    "preparing",
                    terminal=False,
                    run_id=iteration.get("new_run_id"),
                    message="正在按已确认方案准备训练。",
                )
                iteration = iterations.prepare(iteration_id, live, execution_id=iteration_id)
                record_local["run_id"] = iteration.get("new_run_id")

            if iteration["status"] == "prepared":
                run = training.get_status(iteration["new_run_id"])
                record_local["run_id"] = iteration["new_run_id"]
                if run["status"] == "blocked":
                    write_transition(
                        "blocked",
                        terminal=True,
                        run_id=iteration["new_run_id"],
                        issues=run["issues"],
                        message="训练准备未通过，请查看问题后处理。",
                    )
                    return False
                preflight = run.get("preflight") or {}
                if preflight.get("status") == "blocked":
                    write_transition(
                        "blocked",
                        terminal=True,
                        run_id=iteration["new_run_id"],
                        issues=run["issues"],
                        message="启动前数据校验未通过，请查看问题后处理。",
                    )
                    return False
                warnings = [
                    issue
                    for issue in (preflight.get("issues") or [])
                    if issue.get("severity") == "warning"
                ]
                if warnings and not record["options"]["acknowledge_warnings"]:
                    # New preflight warnings are shown first; the user must review
                    # them before the first submission proceeds.
                    write_transition(
                        "awaiting_warning_ack",
                        terminal=False,
                        run_id=iteration["new_run_id"],
                        issues=preflight["issues"],
                        message="预检存在需核对的提示；请查看后在原入口确认继续。",
                    )
                    return False
                if stop_requested():
                    return stop_now("已按请求停止自动执行。")
                write_transition(
                    "training",
                    terminal=False,
                    run_id=iteration["new_run_id"],
                    message="训练已按确认方案启动。",
                )
                iteration = iterations.start(
                    iteration_id,
                    live,
                    acknowledge_warnings=record["options"]["acknowledge_warnings"],
                    execution_id=iteration_id,
                )

            run = iterations._effective_training_run(iteration["new_run_id"])
            while True:
                if stop_requested():
                    return stop_now("已按请求停止自动执行及关联训练。")
                if run["status"] == "succeeded":
                    break
                if run["status"] in {"failed", "stopped", "unknown"}:
                    failure = run.get("failure") or {}
                    write_transition(
                        "blocked",
                        terminal=True,
                        run_id=iteration["new_run_id"],
                        issues=run.get("issues") or [],
                        message=(
                            f"训练未成功（{run['status']}）："
                            f"{failure.get('message') or '请查看训练日志。'}"
                            "请处理问题后提出新的改进轮次。"
                        ),
                    )
                    return False
                time.sleep(POLL_SECONDS)
                run = iterations._effective_training_run(iteration["new_run_id"])

            write_transition(
                "waiting_for_release",
                terminal=False,
                run_id=iteration["new_run_id"],
                message="训练进程已退出，等待释放资源后开始开发集对照。",
            )
            while training.runner.get_status(iteration["new_run_id"]) == "running":
                if stop_requested():
                    return stop_now("已按请求停止自动执行。")
                time.sleep(POLL_SECONDS)
            time.sleep(RELEASE_SETTLE_SECONDS)
            if stop_requested():
                return stop_now("已按请求停止自动执行。")

            write_transition(
                "evaluating",
                terminal=False,
                run_id=iteration["new_run_id"],
                message="正在按父轮协议比较基座、父轮与本轮模型。",
            )
            live = intake.load(live.session_id)
            iteration = iterations.get(iteration_id)
            parent = training.get_status(iteration["parent_run_id"])
            if parent["status"] != "succeeded":
                write_transition(
                    "blocked",
                    terminal=True,
                    run_id=iteration["new_run_id"],
                    message="父轮训练产物已不是成功状态，不能完成同题对照。",
                )
                return False
            current = iterations._effective_training_run(iteration["new_run_id"])
            if current["status"] != "succeeded":
                write_transition(
                    "blocked",
                    terminal=True,
                    run_id=iteration["new_run_id"],
                    message="本轮训练未成功产出模型，不能完成同题对照。",
                )
                return False
            from src.workbench.business_evaluation import EvaluationModel

            models = [
                EvaluationModel("基座", iteration["model_path"]),
                EvaluationModel(
                    "父轮模型", parent["model_path"], adapter_path=parent["output_dir"]
                ),
                EvaluationModel(
                    "本轮微调",
                    current["model_path"],
                    adapter_path=current["output_dir"],
                ),
            ]
            report = iterations.evaluations.compare(
                live, models, _protocol_from_iteration(iteration)
            )
            if report.status not in {"completed", "completed_with_failures"}:
                write_transition(
                    "blocked",
                    terminal=True,
                    run_id=iteration["new_run_id"],
                    issues=[{"severity": "warning", "message": note} for note in report.notes],
                    message="开发集对照未完整完成，请查看评测报告。",
                )
                return False
            iterations.bind_evaluation(
                iteration_id, live, report.evaluation_id, execution_id=iteration_id
            )
            write_transition(
                "completed",
                terminal=True,
                run_id=iteration["new_run_id"],
                evaluation_id=report.evaluation_id,
                message="开发集对照已完成；请核对三模型结果后决定采用、继续或停止。",
            )
            return True
        except KeyboardInterrupt:
            return stop_now("执行进程收到停止信号，已停止。")
        except (ValueError, TypeError, KeyError, OSError, ImportError) as exc:
            write_transition(
                "blocked",
                terminal=True,
                message=f"自动执行被阻断：{exc}",
            )
            return False
        except BaseException as exc:
            traceback.print_exc()
            write_transition(
                "failed",
                terminal=True,
                message=f"自动执行失败：{type(exc).__name__}：{exc}",
            )
            return False
