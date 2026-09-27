"""Confirmed improvement hypotheses linked to real training and fixed-case comparisons."""

from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from src.workbench.business_evaluation import BusinessEvaluationService, _model_identity
from src.workbench.evaluation_diagnostics import EvaluationDiagnostics
from src.workbench.evaluation_suites import EvalSuiteService, assert_compatible
from src.workbench.intake_models import IntakeSession
from src.workbench.sources import canonical, content_digest
from src.workbench.training_runs import TrainingRunService


class IterationService:
    def __init__(self, root, training_root, evaluation_root):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.database = self.root / "iterations.sqlite"
        self.training = TrainingRunService(training_root)
        self.evaluations = BusinessEvaluationService(evaluation_root)
        self.suites = EvalSuiteService(self.root / "suites")
        with sqlite3.connect(self.database) as connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS iterations (id TEXT PRIMARY KEY, revision INTEGER NOT NULL, snapshot TEXT NOT NULL)"
            )

    def _save(self, record, expected=None):
        record = json.loads(canonical(record))
        record["revision"] = 0 if expected is None else expected + 1
        record["updated_at"] = datetime.now(timezone.utc).isoformat()
        with sqlite3.connect(self.database) as connection:
            if expected is None:
                connection.execute(
                    "INSERT INTO iterations VALUES (?, ?, ?)",
                    (record["iteration_id"], record["revision"], canonical(record)),
                )
            else:
                result = connection.execute(
                    "UPDATE iterations SET revision=?, snapshot=? WHERE id=? AND revision=?",
                    (record["revision"], canonical(record), record["iteration_id"], expected),
                )
                if result.rowcount != 1:
                    raise ValueError("改进轮次已更新，请重新读取。")
        return record

    def get(self, iteration_id):
        if not re.fullmatch(r"it-[0-9a-f]{32}", iteration_id):
            raise ValueError("无效改进轮次 ID。")
        with sqlite3.connect(self.database) as connection:
            row = connection.execute(
                "SELECT snapshot FROM iterations WHERE id=?", (iteration_id,)
            ).fetchone()
        if row is None:
            raise ValueError("找不到改进轮次。")
        return json.loads(row[0])

    def list_iterations(self, session_id=None):
        with sqlite3.connect(self.database) as connection:
            records = [
                json.loads(row[0])
                for row in connection.execute("SELECT snapshot FROM iterations ORDER BY rowid DESC")
            ]
        return [
            record for record in records if session_id is None or record["session_id"] == session_id
        ]

    def _snapshot(self, run_id):
        return IntakeSession.model_validate_json(
            (self.training._directory(run_id) / "session.json").read_text()
        )

    @staticmethod
    def _has_model(report, run, base=None):
        identity = _model_identity(Path(run["output_dir"]).resolve())
        base = base or _model_identity(Path(run["model_path"]).resolve())
        if Path(base["directory"]).resolve() != Path(run["model_path"]).resolve():
            return False
        trained_base = run.get("model_identity")
        if trained_base and (
            Path(trained_base["path"]).resolve() != Path(base["directory"]).resolve()
            or any(
                base["files"].get(name) != digest
                for name, digest in trained_base["config_and_tokenizer_hashes"].items()
            )
            or any(
                base["files"].get(weight["name"]) != weight["sha256"]
                for weight in trained_base["weights"]
                if "sha256" in weight
            )
        ):
            return False
        return any(
            model.get("identity", {}).get("adapter") == identity
            and model.get("identity", {}).get("base") == base
            for model in report.models
        )

    def _parent_report(self, record):
        report = self.evaluations.get_report(record["parent_evaluation_id"])
        expected = record.get("parent_report_digest")
        if expected and content_digest(asdict(report)) != expected:
            raise ValueError("父轮评测证据在提案后发生变化，请重新提出并确认改进方案。")
        return report

    def propose(
        self,
        session,
        *,
        parent_run_id,
        evaluation_id,
        hypothesis,
        expected_outcome,
        changes,
        data_change=False,
        training_options=None,
        lora_options=None,
        model_options=None,
        max_length=None,
    ):
        if session.dataset is None:
            raise ValueError("请先物化当前确认的数据。")
        if type(data_change) is not bool:
            raise ValueError("数据修改范围必须明确为布尔值。")
        if max_length is not None and (type(max_length) is not int or max_length <= 0):
            raise ValueError("max_length 必须是正整数。")
        for value in (hypothesis, expected_outcome, changes):
            if not isinstance(value, str) or not value.strip():
                raise ValueError("请写清改进假设、预期业务结果与具体修改。")
        parent = self.training.get_status(parent_run_id)
        if parent["status"] != "succeeded" or parent["session_id"] != session.session_id:
            raise ValueError("父轮必须是当前业务任务已成功产出模型的训练。")
        snapshot = self._snapshot(parent_run_id)
        report = self.evaluations.get_report(evaluation_id)
        # A later report may already compare this parent on a compatible revised dataset.
        evidence_session = (
            session if report.dataset["version"] == session.dataset.version else snapshot
        )
        EvaluationDiagnostics(report, evidence_session)
        if not self._has_model(report, parent):
            raise ValueError("所选评测没有包含父轮实际模型产物。")
        config = parent["config"]
        options = {
            "training_options": {k: v for k, v in config["training"].items() if k != "output_dir"},
            "lora_options": dict(config["lora"]),
            "model_options": {
                k: v
                for k, v in config["model"].items()
                if k not in {"name", "max_length", "trust_remote_code"}
            },
            "max_length": config["model"]["max_length"] if max_length is None else max_length,
        }
        overrides = {
            "training_options": training_options,
            "lora_options": lora_options,
            "model_options": model_options,
        }
        forbidden = {
            "training_options": {"output_dir"},
            "model_options": {"name", "max_length", "trust_remote_code"},
            "lora_options": set(),
        }
        for key, values in overrides.items():
            if values is not None and (
                not isinstance(values, dict)
                or set(values) - set(options[key])
                or set(values) & forbidden[key]
            ):
                raise ValueError("改进配置包含不支持或不允许覆盖的字段。")
            options[key].update(values or {})
        reference = self.suites.freeze(evidence_session)
        assert_compatible(session, reference)
        record = {
            "iteration_id": "it-" + uuid4().hex,
            "session_id": session.session_id,
            "goal": session.goal,
            "parent_run_id": parent_run_id,
            "parent_evaluation_id": evaluation_id,
            "parent_report_digest": content_digest(asdict(report)),
            "evaluation_protocol": report.protocol,
            "parent_evidence_binding": "content_verified",
            "evaluation_suite": reference,
            "hypothesis": hypothesis.strip(),
            "expected_outcome": expected_outcome.strip(),
            "changes": changes.strip(),
            "data_change": bool(data_change),
            "training_start": "base",
            "model_path": parent["model_path"],
            "options": options,
            "parent_dataset": snapshot.dataset.model_dump(),
            "proposal_dataset_digest": content_digest(session.dataset.model_dump()),
            "new_run_id": None,
            "evaluation_id": None,
            "status": "proposed",
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        return self._save(record)

    @staticmethod
    def _task(record, session):
        if record["session_id"] != session.session_id or record["goal"] != session.goal:
            raise ValueError("改进轮次必须继续同一业务目标；改变目标需另建任务。")

    def confirm(self, iteration_id, session):
        record = self.get(iteration_id)
        self._task(record, session)
        if (
            record["status"] != "proposed"
            or session.dataset is None
            or content_digest(session.dataset.model_dump()) != record["proposal_dataset_digest"]
        ):
            raise ValueError("提案后数据或轮次状态已变化，请重新核对提案。")
        self._parent_report(record)
        record["confirmed_session_revision"] = session.revision
        record["status"] = "confirmed"
        return self._save(record, record["revision"])

    @staticmethod
    def _execution_owner(record, execution_id):
        if record.get("execution_id") and record["execution_id"] != execution_id:
            raise ValueError("本轮已交给自动执行，请在该执行入口查看或停止，不能重复手动交接。")

    def claim_execution(self, iteration_id, execution_id):
        """Bind the iteration to one automatic execution; manual handoffs are refused."""
        record = self.get(iteration_id)
        self._execution_owner(record, execution_id)
        if record.get("execution_id") == execution_id:
            return record
        if record["status"] != "confirmed":
            raise ValueError("只有已确认范围的改进轮次可以交给自动执行。")
        record["execution_id"] = execution_id
        return self._save(record, record["revision"])

    def prepare(self, iteration_id, session, *, execution_id=None):
        record = self.get(iteration_id)
        self._execution_owner(record, execution_id)
        self._task(record, session)
        if record["status"] != "confirmed":
            raise ValueError("请先确认改进假设和具体配置；每轮只准备一次训练。")
        if (
            session.dataset is None
            or session.dataset.evaluation_suite != record["evaluation_suite"]
        ):
            raise ValueError("请用本轮固定评测套件重新物化确认的数据。")
        assert_compatible(session, record["evaluation_suite"])
        parent_data = record["parent_dataset"]
        changed = any(
            getattr(session.dataset, key) != parent_data[key]
            for key in ("source_digest", "recipe_digest")
        )
        if changed != record["data_change"]:
            raise ValueError("实际数据修改与确认的改进范围不一致，请重新提出并确认方案。")
        if (
            changed
            and "confirmed_session_revision" in record
            and (session.dataset.full_confirmed_revision <= record["confirmed_session_revision"])
        ):
            raise ValueError("请在确认本轮改进后重新核对并确认实际全量数据预览。")
        parent_report = self._parent_report(record)
        parent = self.training.get_status(record["parent_run_id"])
        if parent["status"] != "succeeded" or not self._has_model(parent_report, parent):
            raise ValueError("父轮模型或基座已改变，不能沿用原改进证据开始训练。")
        # Claim once before creating a subprocess handoff, including competing UI/CLI callers.
        record["status"] = "preparing"
        record = self._save(record, record["revision"])
        try:
            run = self.training.prepare(session, record["model_path"], **record["options"])
            record.update(
                new_run_id=run["run_id"],
                run_id=run["run_id"],
                training_run=run,
                status="prepared" if run["status"] == "prepared" else "blocked",
            )
            record["actual_dataset"] = session.dataset.model_dump()
            record["actual_data_changes"] = {
                key: {"before": parent_data[key], "after": getattr(session.dataset, key)}
                for key in ("version", "source_digest", "recipe_digest")
                if getattr(session.dataset, key) != parent_data[key]
            }
        except Exception as exc:
            record.update(status="blocked", failure=str(exc))
            self._save(record, record["revision"])
            raise
        return self._save(record, record["revision"])

    def start(
        self,
        iteration_id,
        session,
        *,
        acknowledge_warnings=False,
        recover_technical_failures=False,
        execution_id=None,
    ):
        record = self.get(iteration_id)
        self._execution_owner(record, execution_id)
        self._task(record, session)
        if record["status"] != "prepared":
            raise ValueError("只有准备完成的改进轮次可以启动。")
        options = {"acknowledge_warnings": acknowledge_warnings}
        if recover_technical_failures:
            options["recover_technical_failures"] = True
        run = self.training.start(record["new_run_id"], session, **options)
        record.update(status="running", training_run=run)
        return self._save(record, record["revision"])

    def _effective_training_run(self, run_id):
        parent = self.training.get_status(run_id)
        child_id = (parent.get("recovery") or {}).get("child_run_id")
        if not child_id:
            return parent
        child = self.training.get_status(child_id)
        if (
            parent["status"] != "failed"
            or parent.get("recover_technical_failures") is not True
            or child.get("recovery_parent_run_id") != run_id
            or child.get("session_id") != parent.get("session_id")
            or child.get("dataset") != parent.get("dataset")
            or child.get("model_identity") != parent.get("model_identity")
        ):
            raise ValueError("技术重试与原业务轮次的数据或模型身份不一致。")
        original, retried = parent["config"], child["config"]
        if any(original[key] != retried[key] for key in ("model", "lora", "data")):
            raise ValueError("技术重试不能改变基座、LoRA方案或数据。")
        before, after = original["training"], retried["training"]
        changed = {key for key in set(before) | set(after) if before.get(key) != after.get(key)}
        if (
            changed
            - {"output_dir", "batch_size", "gradient_accumulation_steps", "gradient_checkpointing"}
            or before["batch_size"] * before["gradient_accumulation_steps"]
            != after["batch_size"] * after["gradient_accumulation_steps"]
            or after["batch_size"] > before["batch_size"]
            or before["gradient_checkpointing"]
            and not after["gradient_checkpointing"]
        ):
            raise ValueError("技术重试修改超出已确认的执行参数范围。")
        return child

    def bind_evaluation(self, iteration_id, session, evaluation_id, *, execution_id=None):
        record = self.get(iteration_id)
        self._execution_owner(record, execution_id)
        self._task(record, session)
        if record["status"] not in {"running", "prepared"} or not record["new_run_id"]:
            raise ValueError("本轮尚未启动训练或已经完成决策。")
        run = self._effective_training_run(record["new_run_id"])
        if run["status"] != "succeeded":
            raise ValueError("本轮训练尚未成功产出模型。")
        report = self.evaluations.get_report(evaluation_id)
        if report.status not in {"completed", "completed_with_failures"}:
            raise ValueError("同题对照尚未完整完成，不能作本轮业务决策。")
        EvaluationDiagnostics(report, session)
        suite = report.dataset.get("evaluation_suite", {})
        if suite.get("suite_id") != record["evaluation_suite"]["suite_id"]:
            raise ValueError("本轮对照必须使用已确认的固定评测套件。")
        parent = self.training.get_status(record["parent_run_id"])
        parent_report = self._parent_report(record)

        def scoring_protocol(value):
            # Old pre-suite reports may lack answers_digest/evaluation_key; the
            # fixed-suite binding (case and answer identity) is separately enforced
            # by the suite reference. Scoring/generation rules cannot drift.
            protocol = {
                key: item
                for key, item in value.items()
                if key not in {"prompt_digest", "answers_digest", "evaluation_key"}
            }
            protocol.setdefault("custom_scoring", None)
            return protocol

        if scoring_protocol(parent_report.protocol) != scoring_protocol(report.protocol):
            raise ValueError("本轮评分规则或生成协议与父轮依据不同，不能把换规则当作模型改善。")
        base = _model_identity(Path(record["model_path"]).resolve())
        if parent["status"] != "succeeded" or not self._has_model(parent_report, parent, base):
            raise ValueError("父轮实际产物或基座与提案依据已经不一致。")
        if not self._has_model(report, run, base) or not self._has_model(report, parent, base):
            raise ValueError("请在同一报告中对照父轮模型与本轮实际模型。")
        if not any(
            model.get("identity", {}).get("base") == base
            and model.get("identity", {}).get("adapter") is None
            for model in report.models
        ):
            raise ValueError("同题对照还必须包含实际基座模型。")
        record.update(
            status="evaluated",
            effective_run_id=run["run_id"],
            technical_recovery_parent_run_id=run.get("recovery_parent_run_id"),
            evaluation_id=evaluation_id,
            evaluation_report_digest=content_digest(asdict(report)),
            parent_evidence_binding=(
                "content_verified" if record.get("parent_report_digest") else "legacy_unbound"
            ),
            comparison_key=report.comparison_key,
            decision_metric="pass_rate"
            if report.protocol["scorer"] == "custom_rules"
            else "exact_match",
            results=[
                {"label": model["label"], "metrics": model["metrics"]} for model in report.models
            ],
        )
        return self._save(record, record["revision"])

    def decide(self, iteration_id, decision, reason):
        record = self.get(iteration_id)
        if (
            record["status"] != "evaluated"
            or decision not in {"adopt", "continue", "stop", "insufficient_evidence"}
            or not isinstance(reason, str)
            or not reason.strip()
        ):
            raise ValueError("完成同题对照后，请选择采用、继续、停止或证据不足并说明业务理由。")
        if (
            record.get("evaluation_report_digest")
            and content_digest(asdict(self.evaluations.get_report(record["evaluation_id"])))
            != record["evaluation_report_digest"]
        ):
            raise ValueError("本轮对照证据在绑定后已改变，请重新核对真实报告。")
        record.update(status="decided", decision=decision, decision_reason=reason.strip())
        return self._save(record, record["revision"])
