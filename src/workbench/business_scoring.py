"""Business-confirmed deterministic scoring through the existing real OS sandbox."""

from __future__ import annotations

import json
import math
import re
import sqlite3
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from pydantic import Field, field_validator, model_validator

from src.workbench.intake_models import Contract
from src.workbench.materialize import dataset_is_current
from src.workbench.sandbox import TransformCase, TransformSandbox
from src.workbench.sources import canonical, content_digest

SCOPE = (
    "这是对已声明可执行业务规则的评分，不替代主观专家判断；只用开发题设计规则，不读取最终测试题。"
)


def _score(value):
    if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("score/pass_threshold 必须是 0 到 1 的有限数字，不能是布尔值。")
    return float(value)


class ScoringExample(Contract):
    name: str = Field(min_length=1)
    input: str
    expected: str
    output: str
    score: float = Field(ge=0, le=1)
    reason: str = Field(min_length=1)
    kind: Literal["business", "counterexample"]

    @field_validator("score", mode="before")
    @classmethod
    def finite_score(cls, value):
        return _score(value)

    @field_validator("name", "reason")
    @classmethod
    def nonblank(cls, value):
        if not value.strip():
            raise ValueError("样例名称和评分理由不能空白。")
        return value


class ScoringRecipe(Contract):
    business_standard: str = Field(min_length=1)
    source_code: str = Field(
        max_length=128 * 1024,
        description="定义纯确定性 transform(rows,config)，只保留所有原行/原字段并新增score/reason；不能读取时间、环境或随机源。可用re/json/math，执行仅在真实OS隔离中。",
    )
    config: dict[str, Any] = Field(default_factory=dict)
    pass_threshold: float = Field(ge=0, le=1)
    examples: list[ScoringExample] = Field(
        min_length=2,
        description="至少一条真实开发题正例与同题错误输出反例。input为Alpaca input，expected为标准答案；不是完整prompt。",
    )

    @field_validator("pass_threshold", mode="before")
    @classmethod
    def finite_threshold(cls, value):
        return _score(value)

    @model_validator(mode="after")
    def business_contract(self):
        if not self.business_standard.strip():
            raise ValueError("请先明确可执行的业务评分标准。")
        if len({example.name for example in self.examples}) != len(self.examples):
            raise ValueError("评分样例名称不能重复。")
        positive = [example for example in self.examples if example.kind == "business"]
        negative = [example for example in self.examples if example.kind == "counterexample"]
        if not positive or not negative:
            raise ValueError("评分规则必须有业务正例及独立反例。")
        if any(example.score < self.pass_threshold for example in positive) or any(
            example.score >= self.pass_threshold for example in negative
        ):
            raise ValueError("业务正例必须达到通过阈值，反例必须低于阈值。")
        triples = {(example.input, example.expected, example.output) for example in positive}
        if any(
            (example.input, example.expected, example.output) in triples for example in negative
        ):
            raise ValueError("反例不能重复正例后仅修改期望分数。")
        return self


def score_outputs(recipe: ScoringRecipe, rows: list[dict], *, sandbox=None) -> list[dict]:
    recipe = ScoringRecipe.model_validate(
        recipe.model_dump() if isinstance(recipe, ScoringRecipe) else recipe
    )
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise ValueError("评分输入必须是记录列表。")
    ids = [row.get("__row_id") for row in rows]
    if any(not isinstance(identity, str) or not identity for identity in ids) or len(ids) != len(
        set(ids)
    ):
        raise ValueError("每个评分输入必须带唯一非空 __row_id。")
    if any(
        not {"__row_id", "input", "expected", "output"} <= set(row)
        or {"score", "reason"} & set(row)
        or any(not isinstance(row.get(key), str) for key in ("input", "expected", "output"))
        for row in rows
    ):
        raise ValueError("评分输入需有input/expected/output文本，不能预带score/reason。")
    if not rows:
        return []
    execution = (sandbox or TransformSandbox()).run(recipe.source_code, rows, recipe.config)
    if execution.status != "passed" or execution.rows is None:
        raise ValueError(
            f"真实隔离评分未通过（{execution.status}）：{execution.error}；未在宿主执行评分代码。"
        )
    outputs = execution.rows
    if len(outputs) != len(rows) or any(not isinstance(row, dict) for row in outputs):
        raise ValueError("评分不能丢行或复制行。")
    output_ids = [row.get("__row_id") for row in outputs]
    if (
        any(not isinstance(identity, str) for identity in output_ids)
        or len(set(output_ids)) != len(ids)
        or set(output_ids) != set(ids)
    ):
        raise ValueError("评分不能改写来源标识、丢行或复制行。")
    by_id, result = {row["__row_id"]: row for row in outputs}, []
    for original in rows:
        output = by_id[original["__row_id"]]
        if set(output) != set(original) | {"score", "reason"} or any(
            canonical(output[key]) != canonical(value) for key, value in original.items()
        ):
            raise ValueError("评分必须保留全部原字段，仅能新增score和reason。")
        score = _score(output["score"])
        if not isinstance(output["reason"], str) or not output["reason"].strip():
            raise ValueError("每条实际评分必须给出非空理由。")
        result.append({**output, "score": score})
    return result


def _binding(session):
    if session.analysis is None or session.analysis.recipe is None:
        raise ValueError("请先确认业务目标和实际数据处理方案。")
    recipe = session.analysis.recipe
    return {
        "session_id": session.session_id,
        "goal": session.goal,
        "task": session.analysis.task.model_dump(),
        "instruction": recipe.instruction,
        "inputs": [item.model_dump() for item in recipe.inputs],
        "targets": [item.model_dump() for item in recipe.targets],
        "output_format": recipe.output_format,
        "composition": session.analysis.composition,
        "adapter": session.analysis.adapter,
    }


class ScoringService:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.database = self.root / "scoring.sqlite"
        with sqlite3.connect(self.database) as connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS scoring_specs (id TEXT PRIMARY KEY, snapshot TEXT NOT NULL)"
            )

    def _development_records(self, session):
        if not dataset_is_current(session):
            raise ValueError("请先确认并物化当前数据，评分规则只能基于真实开发题起草。")
        from src.workbench.evaluation_suites import evaluation_cases
        from src.workbench.training_preflight import _verified_partitions

        check = {"issues": []}
        records = _verified_partitions(session, check)["validation"]
        if any(issue["severity"] == "blocking" for issue in check["issues"]):
            raise ValueError("当前分区存在交叉泄漏，不能据此起草业务评分。")
        if session.dataset.evaluation_suite:
            records = evaluation_cases(session, split="validation")["records"]
        return records

    def context(self, session, business_standard, *, offset=0, limit=20):
        if not isinstance(business_standard, str) or not business_standard.strip():
            raise ValueError("请明确业务评分标准，不能由程序预设业务达标含义。")
        if type(offset) is not int or offset < 0 or type(limit) is not int or not 1 <= limit <= 20:
            raise ValueError("开发题分页需offset非负整数、limit为1到20。")
        records = self._development_records(session)
        page = []
        characters = 0
        for index, row in enumerate(records[offset : offset + limit], start=offset):
            size = len(row["input"]) + len(row["output"])
            identity = row["metadata"].get("evaluation_case_id", str(index))
            if size > 16000:
                page.append(
                    {
                        "case_id": identity,
                        "content_status": "too_large",
                        "characters": size,
                        "reason": "此题过长，未发送或截断内容；请选择完整可读取的开发题。",
                    }
                )
                continue
            if characters + size > 32000:
                break
            page.append(
                {
                    "case_id": identity,
                    "input": row["input"],
                    "expected": row["output"],
                    "content_status": "complete",
                }
            )
            characters += size
        return {
            "session_id": session.session_id,
            "session_revision": session.revision,
            "goal": session.goal,
            "recipe": session.analysis.recipe.model_dump(),
            "business_standard": business_standard.strip(),
            "development_cases": page,
            "total": len(records),
            "returned": len(page),
            "offset": offset,
            "has_more": offset + len(page) < len(records),
            "next_offset": offset + len(page) if offset + len(page) < len(records) else None,
            "scope_note": SCOPE,
        }

    def draft(self, session, recipe, *, trace=None):
        recipe = ScoringRecipe.model_validate(
            recipe.model_dump() if isinstance(recipe, ScoringRecipe) else recipe
        )
        real = {(row["input"], row["output"]) for row in self._development_records(session)}
        anchored = [
            example
            for example in recipe.examples
            if example.kind == "business" and (example.input, example.expected) in real
        ]
        if not anchored:
            raise ValueError("至少一个评分业务正例必须引用当前真实开发题的input与expected。")
        if not any(
            example.kind == "counterexample"
            and any(
                (example.input, example.expected) == (positive.input, positive.expected)
                and example.output != positive.output
                for positive in anchored
            )
            for example in recipe.examples
        ):
            raise ValueError(
                "至少一个反例必须对应真实开发正例的同题错误输出，不能只改标准答案制造反例。"
            )
        rows, expected, cases = [], [], []
        for index, example in enumerate(recipe.examples):
            row = {
                "__row_id": f"example-{index}",
                "input": example.input,
                "expected": example.expected,
                "output": example.output,
            }
            target = {**row, "score": example.score, "reason": example.reason}
            rows.append(row)
            expected.append(target)
            cases.append(TransformCase(example.name, [row], [target], recipe.config, example.kind))
        runner = TransformSandbox()
        validation = runner.validate(recipe.source_code, cases)
        if validation.status != "passed":
            raise ValueError(
                f"评分正反例的真实隔离验证未通过（{validation.status}）：{validation.cases}"
            )
        actual = score_outputs(recipe, rows, sandbox=runner)
        replay = score_outputs(recipe, rows, sandbox=runner)
        if actual != expected or replay != actual:
            raise ValueError("评分实际输出与声明样例不符或重放不一致；只允许纯确定性规则。")
        binding = _binding(session)
        record = {
            "scoring_id": "sc-" + uuid4().hex,
            "session_id": session.session_id,
            "session_revision": session.revision,
            "session_digest": content_digest(session.model_dump()),
            "binding": binding,
            "recipe": recipe.model_dump(),
            "status": "draft",
            "spec_digest": content_digest({"recipe": recipe.model_dump(), "binding": binding}),
            "validation": {
                **asdict(validation),
                "determinism_checked": True,
                "determinism_scope": "声明样例逐例与批量两次重放一致；仍要求规则对所有输入纯确定性。",
            },
            "example_results": [
                {
                    "name": example.name,
                    "kind": example.kind,
                    **{key: value for key, value in output.items() if key != "__row_id"},
                }
                for example, output in zip(recipe.examples, actual)
            ],
            "trace": trace or [],
            "created_at": datetime.now(timezone.utc).isoformat(),
            "scope_note": SCOPE,
        }
        record["record_digest"] = content_digest(record)
        with sqlite3.connect(self.database) as connection:
            connection.execute(
                "INSERT INTO scoring_specs VALUES (?,?)", (record["scoring_id"], canonical(record))
            )
        return record

    def get(self, scoring_id):
        if not isinstance(scoring_id, str) or not re.fullmatch(r"sc-[0-9a-f]{32}", scoring_id):
            raise ValueError("无效评分规则 ID。")
        with sqlite3.connect(self.database) as connection:
            row = connection.execute(
                "SELECT snapshot FROM scoring_specs WHERE id=?", (scoring_id,)
            ).fetchone()
        if row is None:
            raise ValueError("找不到这份评分规则。")
        record = json.loads(row[0])
        if record.get("record_digest") != content_digest(
            {key: value for key, value in record.items() if key != "record_digest"}
        ):
            raise ValueError("评分规则、确认或验证报告内容已被修改。")
        if record["spec_digest"] != content_digest(
            {"recipe": record["recipe"], "binding": record["binding"]}
        ):
            raise ValueError("评分规则内容与身份不一致。")
        ScoringRecipe.model_validate(record["recipe"])
        return record

    def list_specs(self, session_id=None):
        with sqlite3.connect(self.database) as connection:
            ids = [
                row[0]
                for row in connection.execute("SELECT id FROM scoring_specs ORDER BY rowid DESC")
            ]
        return [
            record
            for record in (self.get(identity) for identity in ids)
            if session_id is None or record["session_id"] == session_id
        ]

    def confirm(self, scoring_id, session):
        record = self.get(scoring_id)
        if (
            record["session_revision"] != session.revision
            or record["session_digest"] != content_digest(session.model_dump())
            or record["binding"] != _binding(session)
        ):
            raise ValueError("评分草稿对应资料或业务方案已更新，请重新基于当前开发题验证并确认。")
        if (
            record["status"] not in {"draft", "confirmed"}
            or record["validation"]["status"] != "passed"
        ):
            raise ValueError("只允许确认真实隔离验证通过的评分草稿。")
        record.update(status="confirmed")
        record["record_digest"] = content_digest(
            {key: value for key, value in record.items() if key != "record_digest"}
        )
        with sqlite3.connect(self.database) as connection:
            connection.execute(
                "UPDATE scoring_specs SET snapshot=? WHERE id=?", (canonical(record), scoring_id)
            )
        return {
            "root": str(self.root),
            "scoring_id": scoring_id,
            "spec_digest": record["spec_digest"],
        }


def load_confirmed(reference, session) -> ScoringRecipe:
    if not isinstance(reference, dict) or set(reference) != {"root", "scoring_id", "spec_digest"}:
        raise ValueError("已确认评分规则引用无效。")
    record = ScoringService(reference["root"]).get(reference["scoring_id"])
    if record["status"] != "confirmed" or record["spec_digest"] != reference["spec_digest"]:
        raise ValueError("评分规则尚未确认或引用的内容指纹不一致。")
    if record["binding"] != _binding(session):
        raise ValueError("当前业务目标、输入或答案语义已变化，评分规则需要重新确认。")
    return ScoringRecipe.model_validate(record["recipe"])
