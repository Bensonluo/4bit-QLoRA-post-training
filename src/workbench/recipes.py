"""Small, explicit data transforms and previews executed on real source rows."""

from __future__ import annotations

from collections import Counter
from typing import Any

from src.workbench.intake_models import (
    BusinessExample,
    DataRecipe,
    FieldBinding,
    IntakeAnalysis,
    PreviewResult,
    PreviewRow,
    SampleSource,
)
from src.workbench.sources import canonical, content_digest, is_missing, parse_json_value


def _transform(value: Any, binding: FieldBinding) -> Any:
    for step in binding.transforms:
        if is_missing(value):
            return value
        if not isinstance(value, str):
            raise ValueError(f"{binding.column} 的 {step.operation} 需要文本，未自动改变原始类型。")
        if step.operation == "strip":
            value = value.strip()
        elif step.operation == "replace":
            value = value.replace(step.old, step.new)
        elif step.operation == "map_values":
            if value not in step.mapping:
                raise ValueError(f"{binding.column} 的值不在已声明映射中：{value}")
            value = step.mapping[value]
        elif step.operation == "parse_json":
            value = parse_json_value(value)
    return value


def _text(value: Any) -> str:
    return value if isinstance(value, str) else canonical(value)


def validate_recipe(source: SampleSource, recipe: DataRecipe) -> None:
    referenced = {f.column for f in recipe.inputs + recipe.targets} | set(recipe.group_columns)
    if recipe.temporal_split:
        policy = recipe.temporal_split
        referenced |= {
            policy.available_at_column,
            policy.prediction_at_column,
            policy.label_end_at_column,
        }
        if policy.label_end_at_column in {field.column for field in recipe.inputs}:
            raise ValueError("标签窗口结束时间不得作为模型输入。")
    unknown = referenced - set(source.columns)
    if unknown:
        raise ValueError(f"方案引用不存在的字段：{sorted(unknown)}")


def preview_recipe(source: SampleSource, recipe: DataRecipe) -> PreviewResult:
    validate_recipe(source, recipe)
    rows: list[PreviewRow] = []
    for row in source.rows:
        issues: list[str] = []
        inputs: dict[str, Any] = {}
        targets: dict[str, Any] = {}
        try:
            inputs = {b.label: _transform(row.values.get(b.column), b) for b in recipe.inputs}
            targets = {b.label: _transform(row.values.get(b.column), b) for b in recipe.targets}
        except (ValueError, TypeError) as exc:
            issues.append(str(exc))
        if any(is_missing(v) for v in inputs.values()):
            issues.append("模型输入有缺失；未自动补值或删除该行。")
        input_text = "\n".join(f"{label}: {_text(value)}" for label, value in inputs.items())
        missing_target = not targets or any(is_missing(v) for v in targets.values())
        if missing_target:
            issues.append("缺少监督答案，需要补充或确认标签；没有自动生成真值。")
        target = None
        if not missing_target:
            target = (
                canonical(targets)
                if recipe.output_format == "json"
                else _text(next(iter(targets.values())))
            )
        invalid = bool(issues and (not missing_target or len(issues) > 1))
        rows.append(
            PreviewRow(
                row_id=row.row_id,
                original=row.values,
                input=input_text,
                target=target,
                group={c: row.values.get(c) for c in recipe.group_columns},
                status="invalid" if invalid else "needs_label" if missing_target else "ready",
                issues=issues,
            )
        )
    if recipe.temporal_split:
        from src.workbench.temporal_split import is_pending_label, temporal_row_times

        for row in rows:
            try:
                temporal_row_times(row, recipe.temporal_split)
            except ValueError as exc:
                row.status = "invalid"
                row.issues.append(str(exc))
            else:
                if is_pending_label(row, recipe.temporal_split):
                    row.issues = [
                        "标签观察窗口尚未成熟；保留原行并在时间分区明确排除，不补造真值。"
                    ]
    by_input: dict[str, list[PreviewRow]] = {}
    for row in rows:
        if row.status == "ready":
            by_input.setdefault(row.input, []).append(row)
    for peers in by_input.values():
        if not recipe.allow_multiple_targets and len({r.target for r in peers}) > 1:
            for row in peers:
                row.status = "conflict"
                row.issues.append("相同模型输入对应不同答案；请澄清标签或补充输入信息。")
    counts = dict.fromkeys(("ready", "needs_label", "invalid", "conflict"), 0)
    counts.update(Counter(r.status for r in rows))
    return PreviewResult(
        source_digest=source.digest,
        recipe_digest=content_digest(recipe.model_dump()),
        instruction=recipe.instruction,
        rows=rows,
        counts=counts,
        scope_note="这是所提供数据的转换预览，不代表已完成全量数据与独立评测验收。",
    )


def validate_analysis(
    source: SampleSource,
    analysis: IntakeAnalysis,
    *,
    full_source: SampleSource | None = None,
    other_sources: dict[str, SampleSource] | None = None,
) -> None:
    known_rows = {r.row_id for r in source.rows}
    roles = {role.column: role for role in analysis.task.field_roles}
    if len(roles) != len(analysis.task.field_roles):
        raise ValueError("字段角色不能重复。")
    if set(roles) != set(source.columns):
        raise ValueError("必须为所有现有字段给出用途，不明字段标为 unknown，不能添加不存在的列。")
    for item in analysis.task.field_roles:
        if set(item.evidence_row_ids) - known_rows:
            raise ValueError(
                f"字段{item.column}的证据行必须使用当前数据短行ID，不能使用source:或full:前缀。"
                f"有效示例：{sorted(known_rows)[:12]}；请调用inspect_rows核实。"
            )
    if full_source:
        known_rows |= {f"full:{full_source.digest}:{row.row_id}" for row in full_source.rows}
    for other in (other_sources or {}).values():
        known_rows |= {f"source:{other.digest}:{row.row_id}" for row in other.rows}
    for item in analysis.findings:
        if set(item.evidence_row_ids) - known_rows:
            raise ValueError("分析引用了不存在或来源不匹配的证据行。")
    ids = [q.question_id for q in analysis.questions]
    if len(ids) != len(set(ids)):
        raise ValueError("澄清问题 ID 不能重复。")
    if analysis.recipe is not None:
        omitted_groups = {column for column, role in roles.items() if role.role == "group"} - set(
            analysis.recipe.group_columns
        )
        if omitted_groups:
            raise ValueError(
                f"已识别的业务分组字段未写入recipe.group_columns：{sorted(omitted_groups)}。"
                "请保留这些分组并重新预览，避免同一业务对象跨训练与评测分区。"
            )
        if analysis.composition:
            for step in analysis.composition.get("steps", []):
                if step.get("operation") == "forecast_labels":
                    prefix = step.get("prefix", "forecast_")
                    forbidden = {
                        prefix + name
                        for name in (
                            "target_price",
                            "realized_return",
                            "direction",
                            "label_end_at",
                            "status",
                        )
                    }
                    if forbidden & {field.column for field in analysis.recipe.inputs}:
                        raise ValueError(
                            "预测标签组合的未来价格、收益、方向、成熟状态和窗口结束时间不得作为模型输入。"
                        )
                    policy = analysis.recipe.temporal_split
                    if policy is None:
                        raise ValueError("预测标签组合必须确认时间分区规则，不能随机切分。")
                    from src.workbench.temporal_split import parse_timestamp

                    if (
                        policy.prediction_at_column != prefix + "prediction_at"
                        or policy.label_end_at_column != prefix + "label_end_at"
                        or policy.available_at_column != step.get("available_at_column")
                        or parse_timestamp(policy.observation_end)
                        != parse_timestamp(step.get("observation_end"))
                    ):
                        raise ValueError("时间分区的时间字段和观察截止必须与预测标签组合一致。")
                if step.get("operation") == "conversation" and not set(
                    step["group_columns"]
                ) <= set(analysis.recipe.group_columns):
                    raise ValueError("会话样本必须保留会话分组字段，避免同一会话进入不同分区。")
        validate_recipe(source, analysis.recipe)
        for field in analysis.recipe.inputs:
            role = roles[field.column]
            if role.role != "input" or role.available_at_prediction is not True:
                raise ValueError(f"{field.column} 尚未明确为预测时可获得的输入，不能放入输入配方。")
        for field in analysis.recipe.targets:
            if roles[field.column].role != "target":
                raise ValueError(f"{field.column} 尚未声明为监督答案。")


def check_examples(preview: PreviewResult, examples: list[BusinessExample]) -> list[str]:
    rows = {r.row_id: r for r in preview.rows}
    failures = []
    for example in examples:
        row = rows.get(example.row_id)
        if (
            row is None
            or row.input != example.expected_input
            or row.target != example.expected_target
        ):
            failures.append(f"{example.row_id} 的转换结果与已认可业务样例不一致。")
    return failures
