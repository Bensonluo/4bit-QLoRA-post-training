"""任务规约投影：把既有确认记录汇编成训练启动前对齐用的只读视图。

设计依据 docs/plans/task-spec-and-agent-adaptation-design.md 第 3 节与 ADR-1：
任务规约 = 业务目标 ＋ 答案语义（监督信号的业务定义与盲标核验结论）＋
评分口径（已确认的自定义规则）＋ 验收标准（冻结条款或未冻结状态）＋
时间约束（时间预测任务的字段与窗口）。规约是只读投影，不是新实体——
全部内容由既有记录汇编（session + analysis + scoring + acceptance + 最新
iteration），不新增存储、不引入第二事实来源。

诚实边界：投影只反映已确认的事实。草稿（如待确认评分规则）不进入规约，
这是特性不是缺陷；未冻结的验收如实显示「未冻结」，不假装存在标准。
投影回答「我们在教模型什么、按什么口径验收」，不代表模型效果达标。
"""

from __future__ import annotations

from pathlib import Path


def collect_task_spec(
    session_id: str,
    store: Path | str,
    scoring_root: Path | str,
    acceptance_root: Path | str,
    evaluation_root: Path | str,
    iteration_root: Path | str,
    training_root: Path | str,
) -> dict:
    """只读汇编任务规约；不写任何状态，草稿不进入投影。"""
    from src.workbench.acceptance import AcceptanceService
    from src.workbench.business_scoring import ScoringService
    from src.workbench.intake_service import IntakeService
    from src.workbench.iterations import IterationService

    session = IntakeService(store).load(session_id)
    analysis = session.analysis
    recipe = analysis.recipe if analysis else None

    specs = ScoringService(scoring_root).list_specs(session_id=session_id)
    confirmed = [record for record in specs if record["status"] == "confirmed"]

    acceptances = AcceptanceService(acceptance_root, evaluation_root).list_acceptances(
        session_id=session_id
    )
    iterations = IterationService(iteration_root, training_root, evaluation_root).list_iterations(
        session_id=session_id
    )
    latest = iterations[0] if iterations else None

    return {
        "session_id": session.session_id,
        "revision": session.revision,
        "goal": {
            "goal": session.goal,
            "data_description": session.data_description,
            "usage_input": analysis.task.usage_input if analysis else None,
            "desired_output": analysis.task.desired_output if analysis else None,
            "row_meaning": analysis.task.row_meaning if analysis else None,
            "success_criteria": list(analysis.task.success_criteria) if analysis else [],
            "training_approach": analysis.training_approach if analysis else None,
        },
        "answer_semantics": {
            "output_format": recipe.output_format if recipe else None,
            "instruction": recipe.instruction if recipe else None,
            "targets": [
                {
                    "label": binding.label,
                    "column": binding.column,
                    "value_kind": binding.value_kind,
                }
                for binding in (recipe.targets if recipe else [])
            ],
            "supervision_source": analysis.task.supervision_source if analysis else None,
            "label_verification": session.label_verification,
        },
        "scoring": {
            "confirmed": [
                {
                    "scoring_id": record["scoring_id"],
                    "business_standard": record["recipe"]["business_standard"],
                    "pass_threshold": record["recipe"]["pass_threshold"],
                    "example_count": len(record["recipe"]["examples"]),
                }
                for record in confirmed
            ],
            "draft_count": sum(record["status"] == "draft" for record in specs),
        },
        "acceptance": {
            "state": "已冻结" if acceptances else "未冻结",
            "records": [
                {
                    "acceptance_id": record["acceptance_id"],
                    "status": record["status"],
                    "protocol": record["protocol"],
                    "criteria": record["criteria"],
                    "result_decision": record["result"]["decision"],
                }
                for record in acceptances
            ],
        },
        "temporal_split": (
            recipe.temporal_split.model_dump() if recipe and recipe.temporal_split else None
        ),
        "latest_iteration": (
            {
                "iteration_id": latest["iteration_id"],
                "hypothesis": latest["hypothesis"],
                "expected_outcome": latest["expected_outcome"],
                "status": latest["status"],
            }
            if latest
            else None
        ),
    }


def spec_anchor_lines(elements: dict) -> list[str]:
    """任务规约四要素的人话行(单一来源):summarize_task_spec 与验收记录的冻结时
    口径引用共用;空输入返回空列表,旧记录如实没有该段。"""
    if not elements:
        return []
    lines: list[str] = []
    goal = elements.get("goal")
    if goal:
        lines.append(f"业务目标：{goal['goal']}")
        if goal["training_approach"]:
            lines.append(f"训练路径：{goal['training_approach']}")

    semantics = elements.get("answer_semantics")
    if semantics:
        if semantics["instruction"]:
            names = (
                "、".join(target["label"] for target in semantics["targets"])
                or "（未定义答案字段）"
            )
            lines.append(
                f"答案语义：{semantics['output_format']} 输出「{names}」，"
                f"监督来自{semantics['supervision_source']}。"
            )
        else:
            lines.append("答案语义：业务方案尚未生成，规约暂缺这一段。")
        verification = semantics["label_verification"]
        if verification is None:
            lines.append("盲标核验：尚未做过。")
        elif verification.get("stale"):
            lines.append(
                f"盲标核验：已失效（此前结论 {verification.get('previous_verdict', '未知')}），"
                "数据或方案更新后需重新核验。"
            )
        else:
            lines.append(
                f"盲标核验：{verification.get('verdict', '未知')}"
                f"（{verification.get('status', '未知状态')}）。"
            )

    scoring = elements.get("scoring")
    if scoring:
        if scoring["confirmed"]:
            first = scoring["confirmed"][0]
            extra = f"等共 {len(scoring['confirmed'])} 份" if len(scoring["confirmed"]) > 1 else ""
            lines.append(
                f"评分口径：已确认自定义规则（{first['business_standard']}，"
                f"通过阈值 {first['pass_threshold']}）{extra}。"
            )
        else:
            lines.append("评分口径：默认严格匹配（暂无已确认的自定义规则）。")
        if scoring["draft_count"]:
            lines.append(
                f"另有 {scoring['draft_count']} 份草稿评分规则未纳入——投影只反映已确认的事实。"
            )

    if elements.get("temporal_split"):
        split = elements["temporal_split"]
        lines.append(
            f"时间约束：时间预测任务；信息可用字段 {split['available_at_column']}、"
            f"预测时点 {split['prediction_at_column']}、标签成熟 {split['label_end_at_column']}。"
        )
    else:
        lines.append("时间约束：无（非时间预测任务）。")
    return lines


def summarize_task_spec(spec: dict) -> list[str]:
    """人话摘要：与页面规约卡同源同词汇（spec_anchor_lines 是四要素的单一来源）。"""
    lines = [
        f"任务 {spec['session_id'][:12]}（revision {spec['revision']}）的任务规约："
        "由既有确认记录只读汇编，不新增状态。"
    ]
    lines.extend(spec_anchor_lines(spec))

    acceptance = spec["acceptance"]
    if acceptance["records"]:
        newest = acceptance["records"][0]
        criteria = newest["criteria"]
        lines.append(
            f"验收标准：已冻结（{criteria['metric']} 最低 {criteria['minimum_score']}、"
            f"最少 {criteria['minimum_cases']} 例）；最近一次结果 {newest['result_decision']}。"
        )
    else:
        lines.append("验收标准：未冻结——当前没有可对照的独立验收条款。")

    if spec["latest_iteration"]:
        iteration = spec["latest_iteration"]
        lines.append(f"最新改进轮：{iteration['hypothesis']}（{iteration['status']}）。")
    else:
        lines.append("最新改进轮：尚无。")
    lines.append("以上是训练启动前的对齐视图：只汇编已确认事实，不代表模型效果达标。")
    return lines
