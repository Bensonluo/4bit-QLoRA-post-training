"""Scenario matrix: the north-star measurement instrument.

北极星的度量是「任意场景,非专家独立走完且语义判断不出错的概率」。本模块
把「任意场景」变成可运行的分布样本:每个场景是一份夹具数据+业务目标+期望
结局(应通过 / 应被某道关卡诚实拦下),跑台用零密钥路径自动走完整旅程,
输出分段结果。它不测模型效果,只测产品在场景多样性下的判断与拦截。
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

STAGES = (
    "create",
    "baseline_analysis",
    "contrast_check",
    "confirm_sample",
    "validate_full",
    "confirm_full",
    "blind_verification",
    "materialize",
)


@dataclass
class ScenarioSpec:
    """One scenario: fixture data + goal + expected outcome."""

    scenario_id: str
    goal: str
    sample: bytes
    sample_name: str
    full: bytes
    full_name: str = "full.csv"
    target_column: str = "类别"
    group_columns: tuple[str, ...] = ()
    excluded_columns: tuple[str, ...] = ()
    # Excel 场景专用:sheet 选择(名称或 1 起始序号);None 沿用默认(第一个 sheet)。
    sample_sheet: str | int | None = None
    full_sheet: str | int | None = None
    expect: str = "passes"  # "passes" | "blocked_at:<stage>"
    expect_note: str = ""
    user_answers: Callable[[dict], dict[str, str]] | None = None
    tags: tuple[str, ...] = ()


@dataclass
class ScenarioResult:
    scenario_id: str
    expect: str
    stages: dict[str, str] = field(default_factory=dict)  # stage -> passed|blocked|error
    blocked_at: str | None = None
    blocked_message: str = ""
    verdict: str = "unresolved"  # as_expected | unexpected_pass | unexpected_block | error
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "expect": self.expect,
            "stages": self.stages,
            "blocked_at": self.blocked_at,
            "blocked_message": self.blocked_message[:200],
            "verdict": self.verdict,
            "detail": self.detail[:200],
        }


def _answers_from_preview(session, source: str = "full_data") -> dict[str, str]:
    """模拟「真懂业务的用户」:按数据的真实标签作答(盲标/对比的正确基线)。"""

    rows = (
        session.full_data.preview.rows
        if source == "full_data" and session.full_data is not None
        else session.preview.rows
        if session.preview is not None
        else []
    )
    return {row.row_id: row.target for row in rows if row.target is not None}


def run_scenario(spec: ScenarioSpec, root: str | Path) -> ScenarioResult:
    """Walk one scenario through the zero-agent journey; record stage-by-stage truth."""
    from src.workbench.baseline_analysis import propose_baseline_analysis
    from src.workbench.intake_service import IntakeService

    result = ScenarioResult(scenario_id=spec.scenario_id, expect=spec.expect)
    service = IntakeService(Path(root) / "intake")
    session = None
    try:
        session = service.create(spec.goal, spec.sample_name, spec.sample, sheet=spec.sample_sheet)
        result.stages["create"] = "passed"

        analysis = propose_baseline_analysis(
            session,
            target_column=spec.target_column,
            group_columns=spec.group_columns,
            excluded_columns=spec.excluded_columns,
        )
        session = service.apply_analysis(session, analysis, model="scenario-matrix")
        if session.analysis is None or session.preview is None:
            result.stages["baseline_analysis"] = "blocked"
            result.blocked_at, result.blocked_message = "baseline_analysis", "分析未生成预览"
        else:
            result.stages["baseline_analysis"] = "passed"

        if result.blocked_at is None:
            pending = service.start_contrast_check(session.session_id, session.revision)
            targets = _answers_from_preview(session, source="sample")
            mapping = {item["row_id"]: targets[item["row_id"]] for item in pending["items"]}
            submitted = service.submit_contrast_check(
                session.session_id, pending["check_id"], mapping
            )
            result.stages["contrast_check"] = (
                "passed" if submitted["verdict"] == "verified" else "blocked"
            )
            if submitted["verdict"] != "verified":
                result.blocked_at, result.blocked_message = "contrast_check", "配对错误"

        if result.blocked_at is None:
            session = service.confirm(session.session_id, session.revision)
            result.stages["confirm_sample"] = "passed"

        if result.blocked_at is None:
            session = service.validate_full_data(
                session.session_id,
                session.revision,
                spec.full_name,
                spec.full,
                sheet=spec.full_sheet,
            )
            blocking = [issue for issue in session.full_data.issues if issue.severity == "blocking"]
            result.stages["validate_full"] = "blocked" if blocking else "passed"
            if blocking:
                result.blocked_at = "validate_full"
                result.blocked_message = blocking[0].message

        if result.blocked_at is None:
            session = service.confirm_full_data(session.session_id, session.revision)
            result.stages["confirm_full"] = "passed"

        if result.blocked_at is None:
            pending = service.start_label_verification(session.session_id, session.revision)
            answer_source = (
                spec.user_answers(session) if spec.user_answers else _answers_from_preview(session)
            )
            answers = {
                item["row_id"]: answer_source.get(item["row_id"], item["row_id"])
                for item in pending["items"]
            }
            verdict = service.submit_label_verification(
                session.session_id, pending["verification_id"], answers
            )
            result.stages["blind_verification"] = (
                "passed" if verdict["verdict"] == "verified" else "blocked"
            )
            if verdict["verdict"] != "verified":
                result.blocked_at, result.blocked_message = (
                    "blind_verification",
                    f"盲标不一致 {verdict['matched']}/{verdict['sample_size']}",
                )

        if result.blocked_at is None:
            independent = not session.analysis.recipe.group_columns
            session = service.materialize_dataset(
                session.session_id,
                session.revision,
                independent_rows_confirmed=independent,
            )
            result.stages["materialize"] = "passed"
    except ValueError as exc:
        # ValueError 是服务的常规业务阻断通道:记为 blocked 而非程序错误。
        stage = next((s for s in STAGES if s not in result.stages), "create")
        result.stages[stage] = "blocked"
        result.blocked_at = stage
        result.blocked_message = str(exc)
    except Exception as exc:  # 程序错误与业务阻断分开统计
        stage = next((s for s in STAGES if s not in result.stages), "create")
        result.stages[stage] = "error"
        result.blocked_at = stage
        result.blocked_message = f"{type(exc).__name__}: {exc}"

    expected_block = spec.expect.startswith("blocked_at:")
    expected_stage = spec.expect.split(":", 1)[1] if expected_block else None
    if result.blocked_at is None:
        result.verdict = "unexpected_block" if expected_block else "as_expected"
    elif result.stages.get(result.blocked_at) == "error":
        result.verdict = "error"
    elif expected_block and result.blocked_at == expected_stage:
        result.verdict = "as_expected"
    else:
        result.verdict = "unexpected_pass" if not expected_block else "unexpected_block"
        result.detail = f"在 {result.blocked_at} 被拦:{result.blocked_message}"
    return result


def run_matrix(specs: list[ScenarioSpec], root: str | Path) -> dict[str, Any]:
    """Run all scenarios and summarize coverage×independence truthfully."""
    results = [run_scenario(spec, Path(root) / spec.scenario_id) for spec in specs]
    verdicts = [result.verdict for result in results]
    return {
        "kind": "scenario_matrix",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scenarios": [result.to_dict() for result in results],
        "summary": {
            "total": len(results),
            "as_expected": verdicts.count("as_expected"),
            "unexpected_pass": verdicts.count("unexpected_pass"),
            "unexpected_block": verdicts.count("unexpected_block"),
            "error": verdicts.count("error"),
            "note": (
                "矩阵测的是产品在场景多样性下的判断与拦截(零密钥路径),"
                "不测模型效果;unexpected_* 是需要修复的产品缺陷。"
            ),
        },
    }


def save_matrix_report(report: dict[str, Any], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
