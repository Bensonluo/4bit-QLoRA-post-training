"""Frozen business criteria and one-shot, independently held-out acceptance attempts."""

from __future__ import annotations

import json
import math
import re
import sqlite3
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from src.workbench.business_evaluation import (
    BusinessEvaluationService,
    EvaluationModel,
    EvaluationProtocol,
    _model_directory,
    _model_identity,
    confirmed_scoring_identity,
    protocol_settings,
)
from src.workbench.evaluation_suites import EvalSuiteService, evaluation_cases
from src.workbench.intake_models import IntakeSession
from src.workbench.materialize import dataset_is_current
from src.workbench.sources import canonical, content_digest

_FROZEN = (
    "session_id",
    "session_revision",
    "original_session_digest",
    "snapshot_digest",
    "model",
    "model_identity",
    "protocol",
    "criteria",
    "criteria_digest",
    "evaluation_suite",
    "exposure_keys",
    "case_count",
    "training_provenance",
)


def _frozen_payload(record):
    payload = {key: record[key] for key in _FROZEN}
    if record["protocol"]["scorer"] == "custom_rules":
        payload["scoring_identity"] = record["scoring_identity"]
    return payload


def _criteria(value, protocol):
    fields = {"metric", "minimum_score", "minimum_cases", "business_standard"}
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError("请明确填写业务标准、指标、最低分数及最少题数；不使用默认达标门槛。")
    if value["metric"] not in {"exact_match", "manual_acceptance_rate", "pass_rate"}:
        raise ValueError(
            "验收指标必须为 exact_match、manual_acceptance_rate 或自定义规则 pass_rate。"
        )
    score, count = value["minimum_score"], value["minimum_cases"]
    if type(score) not in (int, float) or not math.isfinite(score) or not 0 <= score <= 1:
        raise ValueError("最低分数必须是 0 到 1 的有限数值。")
    if type(count) is not int or count < 1:
        raise ValueError("最少题数必须是正整数。")
    if not isinstance(value["business_standard"], str) or not value["business_standard"].strip():
        raise ValueError("请写清本次业务验收标准。")
    expected_metric = {"open_review": "manual_acceptance_rate", "custom_rules": "pass_rate"}.get(
        protocol.scorer, "exact_match"
    )
    if value["metric"] != expected_metric:
        raise ValueError(
            "开放任务使用 manual_acceptance_rate，自定义规则使用 pass_rate，精确评分使用 exact_match。"
        )
    return {**value, "business_standard": value["business_standard"].strip()}


_SPEC_ELEMENT_KEYS = ("goal", "answer_semantics", "scoring", "temporal_split")


class AcceptanceService:
    def __init__(self, root, evaluation_root):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.database = self.root / "acceptances.sqlite"
        self.evaluations = BusinessEvaluationService(evaluation_root)
        self.suites = EvalSuiteService(self.root / "suites")
        with sqlite3.connect(self.database) as connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS acceptances (id TEXT PRIMARY KEY, revision INTEGER NOT NULL, snapshot TEXT NOT NULL)"
            )
            connection.execute(
                "CREATE TABLE IF NOT EXISTS suite_criteria (suite_id TEXT PRIMARY KEY, digest TEXT NOT NULL)"
            )
            connection.execute(
                "CREATE TABLE IF NOT EXISTS revealed_cases (key TEXT PRIMARY KEY, acceptance_id TEXT NOT NULL)"
            )

    @staticmethod
    def _verify(record):
        if content_digest(_frozen_payload(record)) != record["frozen_digest"]:
            raise ValueError("已冻结验收模型、数据或业务标准被修改。")
        if content_digest(record["criteria"]) != record["criteria_digest"]:
            raise ValueError("业务验收标准指纹不一致。")

    def _write(self, connection, record, *, new=False):
        record = json.loads(canonical(record))
        previous = record.get("revision", 0)
        record["revision"] = 0 if new else previous + 1
        record["updated_at"] = datetime.now(timezone.utc).isoformat()
        if new:
            connection.execute(
                "INSERT INTO acceptances VALUES (?, ?, ?)",
                (record["acceptance_id"], 0, canonical(record)),
            )
        else:
            result = connection.execute(
                "UPDATE acceptances SET revision=?,snapshot=? WHERE id=? AND revision=?",
                (record["revision"], canonical(record), record["acceptance_id"], previous),
            )
            if result.rowcount != 1:
                raise ValueError("验收记录已更新，请重新读取。")
        return record

    def get(self, acceptance_id):
        if not isinstance(acceptance_id, str) or not re.fullmatch(
            r"ac-[0-9a-f]{32}", acceptance_id
        ):
            raise ValueError("无效验收 ID。")
        with sqlite3.connect(self.database) as connection:
            row = connection.execute(
                "SELECT snapshot FROM acceptances WHERE id=?", (acceptance_id,)
            ).fetchone()
        if row is None:
            raise ValueError("找不到这次业务验收。")
        record = json.loads(row[0])
        self._verify(record)
        if record.get("evaluation_id"):
            report = self.evaluations.get_report(record["evaluation_id"])
            if content_digest(asdict(report)) != record.get("report_digest") or content_digest(
                record.get("report")
            ) != record.get("report_digest"):
                raise ValueError("已保存的最终验收报告被修改，不能继续采用或人工评分。")
            record["report"] = asdict(report)
        return record

    def list_acceptances(self, session_id=None):
        with sqlite3.connect(self.database) as connection:
            ids = [
                row[0]
                for row in connection.execute("SELECT id FROM acceptances ORDER BY rowid DESC")
            ]
        return [
            record
            for record in (self.get(identity) for identity in ids)
            if session_id is None or record["session_id"] == session_id
        ]

    def _snapshot(self, record):
        path = self.root / record["acceptance_id"] / "session.json"
        snapshot = IntakeSession.model_validate_json(path.read_text(encoding="utf-8"))
        if content_digest(snapshot.model_dump()) != record["snapshot_digest"]:
            raise ValueError("验收时冻结的数据快照已被修改。")
        return snapshot

    @staticmethod
    def _identity(model):
        return {
            "base": _model_identity(Path(model["base_model"])),
            "adapter": _model_identity(Path(model["adapter_path"]))
            if model["adapter_path"]
            else None,
        }

    @staticmethod
    def _training_provenance(snapshot, model, identity):
        if not model["adapter_path"]:
            return {
                "status": "base_model",
                "scope": "本次没有微调适配器；留出隔离针对当前业务任务，不声称已审计外部预训练语料。",
            }
        from src.workbench.evaluation_diagnostics import inspect_model_training_evidence

        return inspect_model_training_evidence(
            {"requested_model": model, "identity": identity},
            snapshot,
            {"evaluation_suite": snapshot.dataset.evaluation_suite},
        )

    @staticmethod
    def _exposure_keys(session, records):
        keys = set()
        for row in records:
            # Answers cannot be changed to relabel an already revealed question as unseen.
            keys.add(
                content_digest(
                    {"kind": "question", "instruction": row["instruction"], "input": row["input"]}
                )
            )
            for name, value in row["metadata"].get("group", {}).items():
                keys.add(
                    content_digest(
                        {
                            "kind": "business_group",
                            "session_id": session.session_id,
                            "column": name,
                            "value": value,
                        }
                    )
                )
        return sorted(keys)

    @staticmethod
    def _revealed(connection, keys):
        return any(
            connection.execute("SELECT 1 FROM revealed_cases WHERE key=?", (key,)).fetchone()
            for key in keys
        )

    def prepare(
        self,
        session,
        model: EvaluationModel,
        protocol: EvaluationProtocol,
        criteria,
        *,
        task_spec: dict | None = None,
    ):
        if not dataset_is_current(session):
            raise ValueError("请先确认全量资料并物化独立数据分区。")
        if not isinstance(model, EvaluationModel) or not model.label.strip():
            raise ValueError("请明确选择一个待验收的实际模型。")
        protocol.validate()
        criteria = _criteria(criteria, protocol)
        _, scoring_identity = confirmed_scoring_identity(protocol, session)
        recipe = session.analysis.recipe
        if protocol.scorer == "classification_exact" and (
            recipe.output_format != "text"
            or len(recipe.targets) != 1
            or recipe.targets[0].value_kind != "categorical"
        ):
            raise ValueError("分类精确验收需要已确认的单类别答案。")
        if protocol.scorer == "json_fields_exact" and (
            recipe.output_format != "json"
            or not set(protocol.fields) <= {field.label for field in recipe.targets}
        ):
            raise ValueError("结构化验收字段必须来自已确认 JSON 答案方案。")
        resolved = EvaluationModel(
            model.label,
            str(_model_directory(model.base_model, model.revision)),
            str(_model_directory(model.adapter_path)) if model.adapter_path else None,
            model.revision,
        )
        snapshot = session.model_copy(deep=True)
        reference = self.suites.freeze(snapshot)
        snapshot.dataset.evaluation_suite = reference
        cases = evaluation_cases(snapshot, split="test")["records"]
        record = {
            "acceptance_id": "ac-" + uuid4().hex,
            "session_id": session.session_id,
            "session_revision": session.revision,
            "original_session_digest": content_digest(session.model_dump()),
            "snapshot_digest": content_digest(snapshot.model_dump()),
            "model": asdict(resolved),
            "model_identity": self._identity(asdict(resolved)),
            "protocol": protocol_settings(protocol),
            "criteria": criteria,
            "criteria_digest": content_digest(criteria),
            # 规约要素快照(R63 plan_trace 同例):冻结时点四要素的信息性视图,不进
            # frozen_digest——session.json 快照与 snapshot_digest 已使 goal/recipe
            # 可防篡改;旧记录无该键,摘要如实不渲染该段。
            "task_spec": {
                key: task_spec[key] for key in _SPEC_ELEMENT_KEYS if key in (task_spec or {})
            },
            "evaluation_suite": reference,
            "exposure_keys": self._exposure_keys(session, cases),
            "case_count": len(cases),
            "status": "prepared",
            "evaluation_id": None,
            "report_digest": None,
            "report": None,
            "decisions": [],
            "result": {"decision": "pending_run"},
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        record["training_provenance"] = self._training_provenance(
            snapshot, record["model"], record["model_identity"]
        )
        if scoring_identity:
            record["scoring_identity"] = scoring_identity
        record["frozen_digest"] = content_digest(_frozen_payload(record))
        contract = content_digest(
            {"criteria": criteria, "protocol": protocol_settings(protocol), **scoring_identity}
        )
        with sqlite3.connect(self.database) as connection:
            connection.execute("BEGIN IMMEDIATE")
            previous = connection.execute(
                "SELECT digest FROM suite_criteria WHERE suite_id=?", (reference["suite_id"],)
            ).fetchone()
            if previous and previous[0] != contract:
                raise ValueError(
                    "此固定套件已冻结业务标准及评分/生成协议，不能换阈值或协议伪装首次盲测。"
                )
            connection.execute(
                "INSERT OR IGNORE INTO suite_criteria VALUES (?, ?)",
                (reference["suite_id"], contract),
            )
            record["blind_test"] = not self._revealed(connection, record["exposure_keys"])
            record["exposure_note"] = (
                "尚未执行；运行时再次核验题目是否已揭示。"
                if record["blind_test"]
                else "存在已揭示题目或同任务业务对象，不能再作为独立首次验收。"
            )
            directory = self.root / record["acceptance_id"]
            directory.mkdir()
            (directory / "session.json").write_text(
                canonical(snapshot.model_dump()), encoding="utf-8"
            )
            return self._write(connection, record, new=True)

    def _verify_report(self, record, report):
        snapshot = self._snapshot(record)
        if (
            report.dataset.get("purpose") != "final_acceptance"
            or report.dataset.get("split") != "test"
            or report.dataset.get("version") != snapshot.dataset.version
            or len(report.models) != 1
            or report.models[0].get("identity") != record["model_identity"]
        ):
            raise ValueError("最终报告与已冻结模型、数据或最终测试边界不一致。")
        if any(
            canonical(report.protocol.get(key)) != canonical(value)
            for key, value in record["protocol"].items()
        ):
            raise ValueError("最终报告没有使用已确认的评分或生成协议。")
        if any(
            report.protocol.get(key) != value
            for key, value in record.get("scoring_identity", {}).items()
        ):
            raise ValueError("最终报告没有使用冻结的自定义评分规则内容。")
        from src.data.loaders import render_alpaca_prompt

        rows = report.models[0]["rows"]
        cases = evaluation_cases(snapshot, split="test")["records"]
        if len(rows) != len(cases) or {row.get("index") for row in rows} != set(range(len(cases))):
            raise ValueError("最终验收报告遗漏或重复了固定测试题。")
        for row in rows:
            source = cases[row["index"]]
            if (
                row["prompt"] != render_alpaca_prompt({**source, "output": ""})
                or row["expected"] != source["output"]
                or row["source"] != source["metadata"]
            ):
                raise ValueError("最终验收题目或答案与固定留出资料不一致。")

    @staticmethod
    def _result(record):
        criteria = record["criteria"]
        rows = record["report"]["models"][0]["rows"]
        total = len(rows)
        manual = criteria["metric"] == "manual_acceptance_rate"
        if manual:
            judged = {item["index"]: item for item in record["decisions"]}
            if len(judged) != total:
                return {
                    "decision": "pending_review",
                    "total_cases": total,
                    "reviewed_cases": len(judged),
                    "score": None,
                }
            accepted = sum(item["decision"] == "accepted" for item in judged.values())
            usable = sum(
                row["status"] == "needs_business_review" and not row["truncated"] for row in rows
            )
        else:
            accepted = sum(
                row.get("correct") is True and row["status"] == "scored" and not row["truncated"]
                for row in rows
            )
            usable = sum(row["status"] == "scored" and not row["truncated"] for row in rows)
        score = accepted / total if total else 0
        decision = "passed" if score >= criteria["minimum_score"] else "failed"
        reason = "按运行前冻结的业务标准计算，失败及截断保留在全部题目分母中。"
        if total < criteria["minimum_cases"] or not usable:
            decision, reason = "insufficient_evidence", "题目数量不足，或没有可完整核查的有效输出。"
        observed_decision = decision
        if record["training_provenance"]["status"] not in {"available", "base_model"}:
            decision, reason = (
                "insufficient_evidence",
                "未能核验适配器训练资料与本次固定留出题的隔离，数值结果不能作为独立业务验收通过。",
            )
        result = {
            "decision": decision,
            "observed_decision": observed_decision,
            "metric": criteria["metric"],
            "score": score,
            "accepted_cases": accepted,
            "total_cases": total,
            "usable_cases": usable,
            "minimum_score": criteria["minimum_score"],
            "minimum_cases": criteria["minimum_cases"],
            "reason": reason,
        }
        if criteria["metric"] == "pass_rate":
            result["business_score"] = (
                sum(row["business_score"] for row in rows) / total if total else 0
            )
            result["business_score_note"] = (
                "业务均分仅作描述；验收结论按冻结的逐题通过门槛及整体 pass_rate 计算。"
            )
        return result

    def run(self, acceptance_id, session, *, runtime_factory=None):
        record = self.get(acceptance_id)
        if record["status"] != "prepared":
            return record
        if content_digest(session.model_dump()) != record["original_session_digest"]:
            raise ValueError("确认验收后任务快照已变化，请重新核对当前模型、资料与标准。")
        snapshot = self._snapshot(record)
        _, scoring_identity = confirmed_scoring_identity(
            EvaluationProtocol(**record["protocol"]), snapshot
        )
        if scoring_identity != record.get("scoring_identity", {}):
            raise ValueError("确认后自定义评分规则内容发生变化，不能沿用冻结验收。")
        if self._identity(record["model"]) != record["model_identity"]:
            raise ValueError("已选择的模型内容发生变化，不能沿用冻结验收。")
        if (
            self._training_provenance(snapshot, record["model"], record["model_identity"])
            != record["training_provenance"]
        ):
            raise ValueError("验收前训练来源证据已发生变化，请重新核对留出隔离依据。")
        cases = evaluation_cases(snapshot, split="test")["records"]
        if self._exposure_keys(snapshot, cases) != record["exposure_keys"]:
            raise ValueError("固定测试题身份在确认后发生变化。")
        with sqlite3.connect(self.database) as connection:
            connection.execute("BEGIN IMMEDIATE")
            latest = json.loads(
                connection.execute(
                    "SELECT snapshot FROM acceptances WHERE id=?", (acceptance_id,)
                ).fetchone()[0]
            )
            self._verify(latest)
            if latest["status"] != "prepared":
                return latest
            record = latest
            if self._revealed(connection, record["exposure_keys"]):
                record.update(
                    status="blocked",
                    blind_test=False,
                    exposure_note="这些实际题目或同任务业务对象已经揭示；换模型、重上传或换套件 ID 都不能重新变成盲测。请准备新的独立保留资料。",
                    result={
                        "decision": "insufficient_evidence",
                        "reason": "heldout_already_revealed",
                    },
                )
                return self._write(connection, record)
            for key in record["exposure_keys"]:
                connection.execute("INSERT INTO revealed_cases VALUES (?, ?)", (key, acceptance_id))
            record.update(
                status="running",
                blind_test=True,
                exposure_note="本次尝试已占用这些留出题；即使加载或生成失败，也不会再次伪装首次盲测。",
            )
            record = self._write(connection, record)
        try:
            protocol = EvaluationProtocol(**record["protocol"])
            report = self.evaluations.evaluate_final(
                snapshot,
                EvaluationModel(**record["model"]),
                protocol,
                runtime_factory=runtime_factory,
            )
            record.update(
                evaluation_id=report.evaluation_id,
                report_digest=content_digest(asdict(report)),
                report=asdict(report),
            )
            self._verify_report(record, report)
            if self._identity(record["model"]) != record["model_identity"]:
                raise ValueError("验收过程中模型内容发生变化，结果不能绑定为已选择模型的业务验收。")
            if report.status == "release_failed":
                raise ValueError("模型释放失败，最终验收没有完整完成。")
            if record["criteria"]["metric"] == "manual_acceptance_rate":
                record["decisions"] = [
                    {
                        "index": row["index"],
                        "decision": "rejected",
                        "reason": row.get("error") or "生成失败或截断，不允许覆盖为通过。",
                        "locked": True,
                    }
                    for row in report.models[0]["rows"]
                    if row["status"] != "needs_business_review" or row["truncated"]
                ]
            record["result"] = self._result(record)
            record["status"] = (
                "needs_business_review"
                if record["result"]["decision"] == "pending_review"
                else "completed"
            )
        except Exception as exc:
            record.update(
                status="failed",
                error=str(exc),
                result={"decision": "insufficient_evidence", "reason": str(exc)},
            )
        with sqlite3.connect(self.database) as connection:
            return self._write(connection, record)

    def review(self, acceptance_id, decisions):
        record = self.get(acceptance_id)
        if (
            record["status"] != "needs_business_review"
            or record["criteria"]["metric"] != "manual_acceptance_rate"
        ):
            raise ValueError("只有等待逐题人工核查的开放任务可提交判断。")
        if not isinstance(decisions, list) or not decisions:
            raise ValueError("请提交至少一条 accepted/rejected 判断及业务理由。")
        if self._identity(record["model"]) != record["model_identity"]:
            raise ValueError("模型内容已改变，不能将旧报告作为当前模型的人工验收。")
        snapshot = self._snapshot(record)
        if (
            self._training_provenance(snapshot, record["model"], record["model_identity"])
            != record["training_provenance"]
        ):
            raise ValueError("训练来源证据已改变，不能沿用原独立性依据人工验收。")
        judged = {item["index"]: item for item in record["decisions"]}
        rows = {row["index"]: row for row in record["report"]["models"][0]["rows"]}
        seen = set()
        for item in decisions:
            if not isinstance(item, dict) or set(item) != {"index", "decision", "reason"}:
                raise ValueError("逐题判断仅接受 index、decision、reason。")
            index = item["index"]
            if type(index) is not int or index not in rows or index in seen:
                raise ValueError("人工判断题号不存在或重复。")
            seen.add(index)
            if (
                item["decision"] not in {"accepted", "rejected"}
                or not isinstance(item["reason"], str)
                or not item["reason"].strip()
            ):
                raise ValueError("每题必须明确接受或拒绝，并填写具体业务理由。")
            row = rows[index]
            if item["decision"] == "accepted" and (
                row["status"] != "needs_business_review" or row["truncated"]
            ):
                raise ValueError("生成失败或截断的输出不能被人工覆盖成通过。")
            if (
                index in judged
                and not judged[index].get("locked")
                and any(judged[index][key] != item[key] for key in item)
            ):
                raise ValueError("已记录的逐题判断不能覆盖修改；保留原验收证据。")
            if not judged.get(index, {}).get("locked"):
                judged[index] = {**item, "reason": item["reason"].strip(), "locked": False}
        record["decisions"] = [judged[index] for index in sorted(judged)]
        record["result"] = self._result(record)
        if record["result"]["decision"] != "pending_review":
            record["status"] = "completed"
        with sqlite3.connect(self.database) as connection:
            return self._write(connection, record)
