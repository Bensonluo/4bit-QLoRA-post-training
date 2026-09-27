"""Execute a confirmed improvement direction through the existing real data tools."""

from src.agent.intake import analyze_intake
from src.workbench.evaluation_diagnostics import EvaluationDiagnostics
from src.workbench.intake_service import next_action
from src.workbench.sources import content_digest


def revise_data_for_iteration(intake, iterations, iteration_id, client, *, expected_revision):
    record = iterations.get(iteration_id)
    if record["status"] != "confirmed" or not record["data_change"]:
        raise ValueError("只有已确认且包含数据修改的轮次可以执行数据改进。")
    session = intake.load(record["session_id"])
    iterations._task(record, session)
    if session.revision != expected_revision:
        raise ValueError("任务已更新，请查看最新资料后再执行修改。")
    report = iterations._parent_report(record)
    evidence_session = intake.dataset_snapshot(session.session_id, report.dataset["version"])
    diagnostics = EvaluationDiagnostics(report, evidence_session)
    direction = {
        key: record[key] for key in ("iteration_id", "hypothesis", "expected_outcome", "changes")
    }
    direction["evaluation_suite_id"] = record["evaluation_suite"]["suite_id"]
    analysis, trace = analyze_intake(
        session, client, revision_context=direction, diagnostics=diagnostics
    )
    if iterations.get(iteration_id)["revision"] != record["revision"]:
        raise ValueError("改进方向已更新，生成方案未覆盖当前资料。")
    previous = session.analysis or session.previous_analysis
    updated = intake.apply_analysis(session, analysis, model=client.model, trace=trace)
    keys = ("recipe", "composition", "adapter")
    before = {key: getattr(previous, key) for key in keys} if previous else {}
    after = {key: getattr(analysis, key) for key in keys}
    before = {
        key: value.model_dump() if hasattr(value, "model_dump") else value
        for key, value in before.items()
    }
    after = {
        key: value.model_dump() if hasattr(value, "model_dump") else value
        for key, value in after.items()
    }
    record["data_revision"] = {
        "from_revision": expected_revision,
        "to_revision": updated.revision,
        "analysis_digest": content_digest(analysis.model_dump()),
        "changed_components": [key for key in keys if before.get(key) != after[key]],
        "before": before,
        "after": after,
        "next_action": next_action(updated),
        "model": client.model,
        "tool_trace": trace,
        "scope": "executed_preview_requires_business_confirmation",
    }
    iterations._save(record, record["revision"])
    return updated
