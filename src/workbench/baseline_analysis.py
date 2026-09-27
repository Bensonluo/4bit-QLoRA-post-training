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
    TemporalSplitPolicy,
)

_CATEGORICAL_MAX_DISTINCT = 20
_OPEN_TEXT_MIN_AVG_LENGTH = 40.0
_TIME_PROBLEM_LIMIT = 3


def _require_time_values(session: IntakeSession, policy: TemporalSplitPolicy) -> None:
    """核验三个时间字段的样例值确为带时区时间且满足行内先后顺序。

    与预览/物化使用同一套 parse_timestamp 规则：选错字段（如把编号当时间）
    在生成基础分析时就地报错，而不是等到全量验证或物化才失败。
    只核验格式与顺序，不判断字段的业务时间含义。
    """
    from src.workbench.temporal_split import parse_timestamp

    problems: list[str] = []
    for row in session.source.rows:
        try:
            times = {
                name: parse_timestamp(row.values.get(column), label=f"第{row.row_id}行{column}")
                for name, column in (
                    ("信息可得时间", policy.available_at_column),
                    ("预测时间", policy.prediction_at_column),
                    ("标签窗口结束时间", policy.label_end_at_column),
                )
            }
        except ValueError as exc:
            problems.append(str(exc))
        else:
            if not times["信息可得时间"] <= times["预测时间"] < times["标签窗口结束时间"]:
                problems.append(
                    f"第{row.row_id}行需满足 信息可得时间 ≤ 预测时间 < 标签窗口结束时间；"
                    "不能使用预测之后才可获得的信息。"
                )
        if len(problems) >= _TIME_PROBLEM_LIMIT:
            break
    if problems:
        raise ValueError(
            "时间分区字段的样例值不是可用的带时区ISO时间："
            + "；".join(problems)
            + "。请改选真正的时间字段，或修正数据中的时间格式后重试；不会退回随机切分。"
        )


def _column_values(session: IntakeSession, column: str) -> list[str]:
    return [row.values.get(column, "") for row in session.source.rows]


def _value_kind(session: IntakeSession, column: str) -> str:
    values = _column_values(session, column)
    if not values:
        return "unspecified"
    # 长度优先:平均长度达到开放文本阈值的列即使取值不多也是开放任务
    # (少样本的开放答复不应因去重数小被误判为类别)。
    avg = sum(len(value) for value in values) / len(values)
    if avg >= _OPEN_TEXT_MIN_AVG_LENGTH:
        return "open_text"
    distinct = {value for value in values if value != ""}
    if len(distinct) <= _CATEGORICAL_MAX_DISTINCT:
        return "categorical"
    return "unspecified"


def propose_baseline_analysis(
    session: IntakeSession,
    *,
    target_column: str,
    group_columns: tuple[str, ...] | list[str] = (),
    excluded_columns: tuple[str, ...] | list[str] = (),
    instruction: str | None = None,
    temporal_policy: TemporalSplitPolicy | dict | None = None,
) -> IntakeAnalysis:
    """Build a valid baseline analysis from the user's column choices.

    temporal_policy：用户显式指定的时间分区字段与边界；提供后按时间分区隔离切分，
    基础分析核验字段存在、样例值时间格式与行内先后顺序，不判断业务时间含义。
    """
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

    policy = None
    if temporal_policy is not None:
        policy = TemporalSplitPolicy.model_validate(
            temporal_policy.model_dump()
            if isinstance(temporal_policy, TemporalSplitPolicy)
            else temporal_policy
        )
        time_columns = {
            policy.available_at_column,
            policy.prediction_at_column,
            policy.label_end_at_column,
        }
        missing = time_columns - set(columns)
        if missing:
            raise ValueError(f"时间分区字段不在数据字段中：{sorted(missing)}（可用：{columns}）。")
        _require_time_values(session, policy)
        # 标签窗口结束时间只能用于分区，绝不能进入模型输入（协议同样强制）。
        excluded.add(policy.label_end_at_column)

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
            reason=(
                "时间分区引用的时间字段，保留为模型输入；请核对预测时确实可获得。"
                if policy and column in {policy.available_at_column, policy.prediction_at_column}
                else "未排除的普通字段，单表静态数据按预测时可获得处理；请核对预览确认业务含义。"
            ),
            available_at_prediction=True,
            evidence_row_ids=[row.row_id for row in session.source.rows[:2]],
        )
        for column in input_columns
    )
    roles.extend(
        FieldRole(
            column=column,
            role="metadata",
            reason=(
                "标签窗口结束时间：仅用于时间分区，不得作为模型输入。"
                if policy and column == policy.label_end_at_column
                else "用户指定排除在模型输入之外，仅作记录。"
            ),
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

    # 标签变体检出:同一业务的多种写法(strip/末尾标点归一后相同但原文不同)
    def _normalize_label(value: str) -> str:
        return value.strip().rstrip("。.!！?？;；,，").strip()

    normalized_groups: dict[str, set[str]] = {}
    for value in distribution:
        normalized_groups.setdefault(_normalize_label(value), set()).add(value)
    variants = {key: values for key, values in normalized_groups.items() if len(values) > 1}
    # 归一草案:每组变体映射到该组出现次数最多的写法(平票取数据中先出现的写法,
    # 重复运行结果一致)。草案预置进方案的答案列转换,用户在预览确认时裁决或否决
    # ——归一写法的选择始终由用户决定,但草案已就位,不用手写。
    draft_mapping: dict[str, str] = {}
    if variants:
        first_seen = {value: index for index, value in enumerate(distribution)}
        variant_targets: dict[str, str] = {}
        tie_notes: list[str] = []
        for group_values in variants.values():
            ordered = sorted(group_values, key=first_seen.__getitem__)
            top = max(distribution[value] for value in ordered)
            tied = [value for value in ordered if distribution[value] == top]
            canonical = tied[0]  # 平票时 max 之外也明确取数据中先出现的写法(确定性)。
            variant_targets.update(dict.fromkeys(group_values, canonical))
            if len(tied) > 1:
                tie_notes.append(
                    f"「{'」「'.join(tied)}」出现次数相同（各 {top} 行），"
                    f"草案暂取数据中先出现的「{canonical}」，请在预览确认时改选你认定的规范写法"
                )
        # map_values 是严格映射:映射之外的值会在预览报错,草案必须覆盖答案列
        # 全部已见取值(变体映射到规范写法,其余原样保留)。
        draft_mapping = {value: variant_targets.get(value, value) for value in distribution}
        rules = [
            f"{value}→{normalized}"
            for value, normalized in draft_mapping.items()
            if value != normalized
        ]
        rules_display = "、".join(rules[:6]) + (" 等" if len(rules) > 6 else "")
        shown = ";".join("/".join(sorted(values)) for values in list(variants.values())[:3])
        # 证据行:变体组内每种写法各引用其首次出现的真实样例行,核对不必自己翻数据。
        evidence: list[str] = []
        cited: set[str] = set()
        for row in session.source.rows:
            value = row.values.get(target_column, "")
            if value in variant_targets and value not in cited:
                cited.add(value)
                evidence.append(row.row_id)
        message = (
            f"答案列存在同一业务含义的多种写法（{shown}）。模型会把它们当不同答案学习，"
            "评测也会被判错；建议在原始数据中统一写法，或用转换规则(map_values)归一。"
            f"已生成归一规则草案：{rules_display}"
        )
        if tie_notes:
            message += "；" + "；".join(tie_notes[:2]) + ("等" if len(tie_notes) > 2 else "")
        message += (
            "。草案已预置到下方方案的答案列转换里，预览确认前可修改或删除，"
            "最终采用哪种写法由你裁决。"
        )
        findings.append(
            Finding(kind="needs_business_input", message=message, evidence_row_ids=evidence[:6])
        )
    # 重复输入检出:完全相同的输入会让样本量虚高、训练重复
    from collections import Counter as _Counter

    input_signatures = [
        tuple(row.values.get(column, "") for column in input_columns) for row in session.source.rows
    ]
    duplicate_total = sum(count - 1 for count in _Counter(input_signatures).values() if count > 1)
    if input_signatures and duplicate_total / len(input_signatures) >= 0.1:
        findings.append(
            Finding(
                kind="needs_business_input",
                message=(
                    f"有 {duplicate_total}/{len(input_signatures)} 条输入字段完全重复的行。"
                    "重复会让样本量虚高、训练时同一内容被反复学习;请确认是否为重复录入,"
                    "必要时在原始数据中去除(答案不同的重复尤其危险——同一输入对应多个标签会让模型无法学习)。"
                ),
            )
        )
    if _value_kind(session, target_column) == "open_text":
        findings.append(
            Finding(
                kind="needs_business_input",
                message=(
                    f"答案列「{target_column}」是开放文本。开放任务没有可执行的自动评分规则："
                    "训练和对照可以正常进行,但对照只保留各模型的完整输出供你逐条人工核对,"
                    "不会自动给出好坏分数。需要自动评分时,须先定义并确认业务评分规则。"
                ),
            )
        )
    # 类别不均衡检出:多数类占比过高时,准确率会被「全猜多数类」撑起来
    if target_values and len(target_values) >= 10:
        majority_share = max(distribution.values()) / len(target_values)
        if majority_share >= 0.8:
            majority_label = distribution.most_common(1)[0][0]
            findings.append(
                Finding(
                    kind="needs_business_input",
                    message=(
                        f"答案分布严重不均衡：「{majority_label}」占 {majority_share:.0%}"
                        f"（{len(target_values)} 行）。模型只要全猜这一类就有 "
                        f"{majority_share:.0%} 准确率——总体准确率会骗人。"
                        "后续对照请看每一类的分别表现；少数类恰恰通常是业务上重要的类。"
                    ),
                )
            )
    target_distinct = len(distribution)
    if len(target_values) >= 8 and target_distinct / max(len(target_values), 1) >= 0.9:
        findings.append(
            Finding(
                kind="needs_business_input",
                message=(
                    f"答案列「{target_column}」几乎每行唯一（{target_distinct}/{len(target_values)}）。"
                    "模型难以从逐行唯一的标签学到可泛化规律——常见原因是把编号/ID 类字段选成了答案列。"
                    "如果这不是抽取类任务，请改选真正的业务答案列。"
                ),
            )
        )
    if policy:
        findings.append(
            Finding(
                kind="needs_business_input",
                message=(
                    f"时间分区方案由用户指定：信息可得「{policy.available_at_column}」、"
                    f"预测「{policy.prediction_at_column}」、标签窗口结束「{policy.label_end_at_column}」；"
                    f"验证起点 {policy.validation_start}、测试起点 {policy.test_start}、"
                    f"观察截止 {policy.observation_end}。"
                    "基础分析只核验字段存在、时间格式与行内先后顺序，不判断业务时间含义；"
                    "请确认边界符合真实业务节奏，且标签窗口结束时间在预测时确实未知。"
                ),
            )
        )
    elif any(marker in session.goal for marker in ("预测", "未来", "走势", "行情", "涨跌", "收益")):
        findings.append(
            Finding(
                kind="needs_business_input",
                message=(
                    "目标像是对未来结果的预测。基础分析没有做时间分区：随机切分会把"
                    "未来信息泄漏进训练，得到虚高的假效果。可在下方基础分析里选择时间分区字段，"
                    "或配置 Agent 建立时间方案，或确认这确实不是预测任务后再继续。"
                ),
            )
        )
    findings.append(
        Finding(
            kind="observed",
            message=(
                "基础分析范围说明：未做多源组合、受限适配与业务问答，本方案仅覆盖单表字段映射；"
                + ("时间分区按用户指定的字段与边界执行。" if policy else "未做时间分区。")
                + "任务确需这些能力时请配置 Agent 或补充说明后重新分析。"
            ),
        )
    )
    recipe_payload = {
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
                # 变体检出时预置归一草案;干净标签不添加任何转换,用户可否决。
                "transforms": (
                    [{"operation": "map_values", "mapping": draft_mapping}] if draft_mapping else []
                ),
            }
        ],
        "group_columns": groups,
        "split_rationale": (
            "按时间分区与业务分组隔离切分：训练/验证标签须在下一分区起点前成熟，"
            "同一对象不跨分区；时间字段与边界由用户指定，基础分析不判断业务时间含义。"
            if policy
            else "按业务分组隔离切分，同一对象不跨训练与评测分区；"
            "未配置 Agent，未做多源组合与时间分区。"
        ),
    }
    if policy:
        recipe_payload["temporal_split"] = policy.model_dump()
    recipe = DataRecipe.model_validate(recipe_payload)
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
                "单表监督微调（SFT）："
                + ("按时间分区与分组隔离切分，" if policy else "按分组隔离切分，")
                + "先做固定开发集基座对照，"
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
