"""旅程漏斗快照：全部数据任务在闭环各环节的当前停点计数。

北极星「度量体系·过程漏斗」的落地：上传 → 分析 → 预览确认 → 全量验证 →
分区 → 预检 → 训练 → 对照 → 决策，每段停着多少任务、停点最集中在哪一段
——停点清单就是下一轮开发的输入。数据全部来自本机 workbench 记录目录，
只读、不写、不发起任何计算。

诚实边界：这是「当前停点」的快照，不是历史通过率。旅程可以回退（数据
修订后停点会变），回退后按新停点重新计数；计数只描述进度停点，不代表
业务效果。某段记录读取失败时如实点名跳过，不假装为零。
"""

from __future__ import annotations

from pathlib import Path

# next_action 枚举 → 闭环前段（数据准备侧）的旅程段映射；顺序即旅程顺序。
_STAGE_NAMES: dict[str, str] = {
    "analysis": "分析与方案",
    "preview_confirm": "样例预览确认",
    "full_validation": "全量验证",
    "split": "独立分区",
    "preflight_ready": "预检就绪",
}

_NEXT_ACTION_STAGES: dict[str, str] = {
    "awaiting_analysis": "analysis",
    "needs_business_answers": "analysis",
    "needs_capability": "analysis",
    "needs_recipe": "analysis",
    "needs_data_revision": "analysis",
    "needs_labels": "analysis",
    "review_preview": "preview_confirm",
    "awaiting_full_data": "full_validation",
    "awaiting_full_validation": "full_validation",
    "needs_full_data_revision": "full_validation",
    "review_full_data": "full_validation",
    "awaiting_dataset_split": "split",
    "ready_for_training_preflight": "preflight_ready",
}

# 训练/验收/轮次状态与决策的人话对照（与各 summarize_* 同风格）。
_TRAINING_NAMES: dict[str, str] = {
    "prepared": "已准备未启动",
    "running": "训练中",
    "stopping": "停止中",
    "succeeded": "成功",
    "failed": "失败",
    "stopped": "已停止",
}
_EVALUATION_NAMES: dict[str, str] = {
    "development_only": "开发集对照",
    "final_acceptance": "最终验收测试",
}
_ACCEPTANCE_NAMES: dict[str, str] = {
    "pending_run": "待运行",
    "pending_review": "待复核",
    "passed": "通过",
    "failed": "未通过",
    "insufficient_evidence": "证据不足",
}
_ITERATION_DECISION_NAMES: dict[str, str] = {
    "adopt": "采用",
    "continue": "继续",
    "stop": "停止",
    "insufficient_evidence": "证据不足",
}


def _count(values: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def _join_counts(counts: dict[str, int], names: dict[str, str]) -> str:
    return "、".join(f"{names.get(key, key)} {value}" for key, value in counts.items())


def build_funnel(
    *,
    session_next_actions: list[str],
    training_statuses: list[str],
    evaluation_purposes: list[str],
    acceptance_decisions: list[str],
    iteration_statuses: list[str],
    iteration_decisions: list[str],
) -> dict:
    """由各记录的状态清单组装漏斗快照（纯函数，页面与 CLI 共用）。

    会话停点按 next_action 枚举映射到旅程段；未收录的枚举进 unmapped
    如实保留，不硬塞进相近段。
    """
    stages = dict.fromkeys(_STAGE_NAMES, 0)
    unmapped: list[str] = []
    for status in session_next_actions:
        stage = _NEXT_ACTION_STAGES.get(status)
        if stage is None:
            unmapped.append(status)
        else:
            stages[stage] += 1
    return {
        "sessions": {
            "total": len(session_next_actions),
            "stages": stages,
            "unmapped": sorted(set(unmapped)),
        },
        "training": _count(training_statuses),
        "evaluations": _count(evaluation_purposes),
        "acceptances": _count(acceptance_decisions),
        "iterations": {
            "total": len(iteration_statuses),
            "statuses": _count(iteration_statuses),
            "decisions": _count(iteration_decisions),
        },
    }


def collect_funnel(
    store_root: str | Path,
    training_root: str | Path,
    evaluation_root: str | Path,
    acceptance_root: str | Path,
    iteration_root: str | Path,
) -> dict:
    """从本机 workbench 记录目录读取各段状态并组装漏斗快照。

    每段独立容错：某段记录损坏只点名跳过（errors 列表），其余段照常统计
    ——只读报告不能因一段坏档全盘失败。
    """
    from src.workbench.intake_service import IntakeService, next_action

    errors: list[str] = []
    session_statuses: list[str] = []
    training_statuses: list[str] = []
    evaluation_purposes: list[str] = []
    acceptance_decisions: list[str] = []
    iteration_statuses: list[str] = []
    iteration_decisions: list[str] = []
    try:
        session_statuses = [
            next_action(session) for session in IntakeService(store_root).list_sessions()
        ]
    except (ValueError, OSError):
        errors.append("数据任务记录")
    try:
        from src.workbench.training_runs import TrainingRunService

        training_statuses = [run["status"] for run in TrainingRunService(training_root).list_runs()]
    except (ValueError, OSError):
        errors.append("训练记录")
    try:
        from src.workbench.business_evaluation import BusinessEvaluationService

        evaluation_purposes = [
            report.dataset.get("purpose", "development_only")
            for report in BusinessEvaluationService(evaluation_root).list_reports(purpose=None)
        ]
    except (ValueError, OSError):
        errors.append("对照评测记录")
    try:
        from src.workbench.acceptance import AcceptanceService

        acceptance_decisions = [
            (record.get("result") or {}).get("decision", "unknown")
            for record in AcceptanceService(acceptance_root, evaluation_root).list_acceptances()
        ]
    except (ValueError, OSError):
        errors.append("业务验收记录")
    try:
        from src.workbench.iterations import IterationService

        iterations = IterationService(iteration_root, training_root, evaluation_root)
        iteration_statuses = [record["status"] for record in iterations.list_iterations()]
        iteration_decisions = [
            record["decision"] for record in iterations.list_iterations() if record.get("decision")
        ]
    except (ValueError, OSError):
        errors.append("改进轮次记录")
    report = build_funnel(
        session_next_actions=session_statuses,
        training_statuses=training_statuses,
        evaluation_purposes=evaluation_purposes,
        acceptance_decisions=acceptance_decisions,
        iteration_statuses=iteration_statuses,
        iteration_decisions=iteration_decisions,
    )
    report["errors"] = errors
    return report


def summarize_funnel(report: dict) -> list[str]:
    """漏斗快照的人话摘要（单一来源，页面与 CLI 同源同词汇）。"""
    sessions = report.get("sessions") or {}
    total = sessions.get("total", 0)
    lines: list[str] = []
    if total == 0:
        lines.append("还没有任何数据任务记录，先从上传业务表格开始。")
    else:
        stages = sessions.get("stages") or {}
        parts = [f"{_STAGE_NAMES[stage]} {count}" for stage, count in stages.items() if count]
        stops = "、".join(parts) if parts else "全部已越过数据准备段"
        lines.append(f"共 {total} 个数据任务，当前停点：{stops}。")
        unmapped = sessions.get("unmapped") or []
        if unmapped:
            names = "、".join(unmapped)
            lines.append(f"另有 {len(unmapped)} 种未收录的停点状态（{names}），按原样列出。")
        busiest = max(stages, key=lambda stage: stages[stage]) if any(stages.values()) else None
        if busiest is not None and stages[busiest]:
            lines.append(
                f"停得最多的是「{_STAGE_NAMES[busiest]}」（{stages[busiest]} 个）"
                "——这里就是下一个最值得查看具体卡点的入口。"
            )
    training = report.get("training") or {}
    if training:
        lines.append(
            f"训练运行共 {sum(training.values())} 个：{_join_counts(training, _TRAINING_NAMES)}。"
        )
    evaluations = report.get("evaluations") or {}
    if evaluations:
        lines.append(
            f"对照评测共 {sum(evaluations.values())} 份：{_join_counts(evaluations, _EVALUATION_NAMES)}。"
        )
    acceptances = report.get("acceptances") or {}
    if acceptances:
        lines.append(
            f"最终业务验收共 {sum(acceptances.values())} 次："
            f"{_join_counts(acceptances, _ACCEPTANCE_NAMES)}。"
        )
    iterations = report.get("iterations") or {}
    if iterations.get("total"):
        decisions = iterations.get("decisions") or {}
        head = f"改进轮次共 {iterations['total']} 轮"
        if decisions:
            head += f"，已决策 {sum(decisions.values())} 轮（{_join_counts(decisions, _ITERATION_DECISION_NAMES)}）"
        lines.append(head + "。")
    errors = report.get("errors") or []
    if errors:
        lines.append(f"{'、'.join(errors)}读取失败，该段未计入（其余段照常统计）。")
    lines.append(
        "以上是各任务当前停点的快照，不是历史通过率；旅程可以回退，回退后按新停点重新计数。"
        "计数只描述进度，不代表业务效果。"
    )
    return lines
