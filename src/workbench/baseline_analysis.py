"""Deterministic baseline analysis — the product's built-in algorithm-engineer judgment.

A new user without any Agent key can still start the core path: they pick the
answer column (and optional business-group columns), and this module produces a
valid IntakeAnalysis through the same deterministic pipeline the Agent path
uses (apply_analysis → real preview → business confirmation). It encodes only
what can be derived from the data plus the user's explicit choices, and says so
in findings — it never invents business meaning.
"""

from __future__ import annotations

from collections import Counter

from src.workbench.intake_models import (
    DataRecipe,
    FieldRole,
    Finding,
    IntakeAnalysis,
    IntakeSession,
    TaskSpec,
)

_CATEGORICAL_MAX_DISTINCT = 20
_OPEN_TEXT_MIN_AVG_LENGTH = 40.0


def _column_values(session: IntakeSession, column: str) -> list[str]:
    return [row.values.get(column, "") for row in session.source.rows]


def _value_kind(session: IntakeSession, column: str) -> str:
    values = _column_values(session, column)
    if not values:
        return "unspecified"
    distinct = {value for value in values if value != ""}
    if len(distinct) <= _CATEGORICAL_MAX_DISTINCT:
        return "categorical"
    avg = sum(len(value) for value in values) / len(values)
    return "open_text" if avg >= _OPEN_TEXT_MIN_AVG_LENGTH else "unspecified"


def propose_baseline_analysis(
    session: IntakeSession,
    *,
    target_column: str,
    group_columns: tuple[str, ...] | list[str] = (),
    excluded_columns: tuple[str, ...] | list[str] = (),
    instruction: str | None = None,
) -> IntakeAnalysis:
    """Build a valid baseline analysis from the user's column choices."""
    columns = list(session.source.columns)
    if target_column not in columns:
        raise ValueError(f"答案列「{target_column}」不在数据字段中（可用：{columns}）。")
    groups = list(dict.fromkeys(group_columns))
    excluded = set(excluded_columns)
    unknown = set(groups) | excluded - {target_column}
    if unknown - set(columns):
        raise ValueError(f"选择的字段不存在：{sorted(unknown - set(columns))}。")
    if target_column in groups or target_column in excluded:
        raise ValueError("答案列不能同时作为分组或排除字段。")

    input_columns = [
        column
        for column in columns
        if column != target_column and column not in groups and column not in excluded
    ]
    if not input_columns:
        raise ValueError("至少保留一个输入字段（除答案与分组外不能全部排除）。")
    target_label = f"{target_column}（答案）"
    input_labels = "、".join(input_columns)
    instruction_text = (
        instruction.strip()
        if instruction and instruction.strip()
        else (
            f"{session.goal.strip()}。输入字段：{input_labels}。"
            f"请严格只输出「{target_column}」的答案值本身，不要复述题目，不要附加解释。"
        )
    )

    roles = [
        FieldRole(
            column=target_column,
            role="target",
            reason="用户指定为监督答案列。",
            evidence_row_ids=[row.row_id for row in session.source.rows[:2]],
        )
    ]
    roles.extend(
        FieldRole(
            column=column,
            role="group",
            reason="用户指定为业务分组字段，同一取值保持同分区以防泄漏。",
            evidence_row_ids=[row.row_id for row in session.source.rows[:2]],
        )
        for column in groups
    )
    roles.extend(
        FieldRole(
            column=column,
            role="input",
            reason="未排除的普通字段，单表静态数据按预测时可获得处理；请核对预览确认业务含义。",
            available_at_prediction=True,
            evidence_row_ids=[row.row_id for row in session.source.rows[:2]],
        )
        for column in input_columns
    )
    roles.extend(
        FieldRole(
            column=column,
            role="metadata",
            reason="用户指定排除在模型输入之外，仅作记录。",
        )
        for column in sorted(excluded)
    )

    target_values = [value for value in _column_values(session, target_column) if value != ""]
    distribution = Counter(target_values)
    findings = [
        Finding(
            kind="observed",
            message=(
                f"答案列「{target_column}」非空取值 {len(target_values)}/{len(session.source.rows)} 行，"
                f"共 {len(distribution)} 类："
                + "、".join(f"{value}×{count}" for value, count in distribution.most_common(6))
            ),
            evidence_row_ids=[row.row_id for row in session.source.rows[:2]],
        ),
        Finding(
            kind="needs_business_input",
            message=(
                "基础分析只依据字段选择和数据事实，不判断业务含义；"
                "请在真实转换预览中逐行核对输入与答案的对应关系后再确认。"
            ),
        ),
    ]
    if not groups:
        findings.append(
            Finding(
                kind="needs_business_input",
                message=(
                    "未选择业务分组字段：若同一客户/会话/对象存在多行，请补充分组，"
                    "否则生成分区前需确认每行是独立业务对象。"
                ),
            )
        )

    findings.append(
        Finding(
            kind="observed",
            message=(
                "基础分析范围说明：未做多源组合、时间分区、受限适配与业务问答，"
                "本方案仅覆盖单表字段映射；任务确需这些能力时请配置 Agent 或补充说明后重新分析。"
            ),
        )
    )
    recipe = DataRecipe.model_validate(
        {
            "instruction": instruction_text,
            "inputs": [
                {
                    "column": column,
                    "label": column,
                    "value_kind": _value_kind(session, column),
                    "transforms": [{"operation": "strip"}],
                }
                for column in input_columns
            ],
            "targets": [
                {
                    "column": target_column,
                    "label": target_label,
                    "value_kind": _value_kind(session, target_column),
                    "transforms": [],
                }
            ],
            "group_columns": groups,
            "split_rationale": (
                "按业务分组隔离切分，同一对象不跨训练与评测分区；"
                "未配置 Agent，未做多源组合与时间分区。"
            ),
        }
    )
    task = TaskSpec.model_validate(
        {
            "goal": session.goal.strip(),
            "usage_input": input_labels,
            "desired_output": target_column,
            "row_meaning": (
                "一行一条业务记录；同一分组字段取值视为同一业务对象。"
                if groups
                else "一行一条业务记录；用户未指定业务分组，按独立行处理前需确认。"
            ),
            "supervision_source": f"答案列「{target_column}」由用户指定；其业务来源与含义由用户核对确认。",
            "success_criteria": ["固定开发集上对照基座的同题比较给出可解释差异（不预设方向）"],
            "field_roles": [role.model_dump() for role in roles],
        }
    )
    return IntakeAnalysis.model_validate(
        {
            "task": task.model_dump(),
            "findings": [finding.model_dump() for finding in findings],
            "questions": [],
            "recipe": recipe.model_dump(),
            "training_approach": (
                "单表监督微调（SFT）：按分组隔离切分，先做固定开发集基座对照，"
                "再决定是否迭代；未配置 Agent，训练方案由后续推荐步骤确认。"
            ),
            "next_steps": [
                "核对真实转换预览并确认业务含义，随后提供全量数据并验证。",
            ],
            # 范围说明放 findings（如实但不阻断）：capability_gaps 仅用于
            # 「任务需要而当前能力缺失」的阻断场景；单表映射没有缺失。
            "capability_gaps": [],
        }
    )
