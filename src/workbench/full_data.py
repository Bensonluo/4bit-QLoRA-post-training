"""Execute an approved sample recipe against independently identified full data."""

from __future__ import annotations

import json
from typing import Any

from src.workbench.intake_models import (
    FullDataIssue,
    FullDataValidation,
    IntakeSession,
    PreviewResult,
    SampleSource,
)
from src.workbench.recipes import preview_recipe
from src.workbench.sources import canonical, content_digest, is_missing, profile_source


def full_data_is_current(session: IntakeSession) -> bool:
    report = session.full_data
    recipe = session.analysis.recipe if session.analysis else None
    return bool(
        report
        and report.status != "stale"
        and recipe
        and session.confirmed_revision is not None
        and report.sample_source_digest == session.source.digest
        and report.sample_confirmed_revision == session.confirmed_revision
        and report.approved_recipe_digest == content_digest(recipe.model_dump())
    )


def _target_values(
    preview: PreviewResult, output_format: str, label: str
) -> dict[str, tuple[Any, list[str]]]:
    values: dict[str, tuple[Any, list[str]]] = {}
    for row in preview.rows:
        if row.target is None:
            continue
        value = json.loads(row.target)[label] if output_format == "json" else row.target
        identity = canonical(value)
        if identity not in values:
            values[identity] = (value, [])
        values[identity][1].append(row.row_id)
    return values


def validate_full_source(session: IntakeSession, source: SampleSource) -> FullDataValidation:
    """Return facts and blockers without changing the sample, recipe, or source records."""
    if (
        session.confirmed_revision is None
        or not session.analysis
        or not session.analysis.recipe
        or session.preview is None
    ):
        raise ValueError("请先确认样例的业务含义与转换预览，再验证全量数据。")
    if source.scope != "full":
        raise ValueError("全量验证必须使用明确声明为全量的数据。")
    recipe = session.analysis.recipe
    profile = profile_source(source)
    required = {f.column for f in recipe.inputs + recipe.targets} | set(recipe.group_columns)
    if recipe.temporal_split:
        policy = recipe.temporal_split
        required |= {
            policy.available_at_column,
            policy.prediction_at_column,
            policy.label_end_at_column,
        }
    actual = set(source.columns)
    original = set(session.source.columns)
    missing_required = sorted(required - actual)
    missing_other = sorted(original - actual - required)
    extra = sorted(actual - original)
    type_changes = {
        name: {
            "sample": session.profile["columns"][name]["value_types"],
            "full": profile["columns"][name]["value_types"],
        }
        for name in sorted(original & actual)
        if set(profile["columns"][name]["value_types"])
        - set(session.profile["columns"][name]["value_types"])
    }
    issues: list[FullDataIssue] = []
    if missing_required:
        issues.append(
            FullDataIssue(
                code="missing_required_columns",
                severity="blocking",
                columns=missing_required,
                message="全量文件缺少当前方案必需字段："
                + "、".join(missing_required)
                + "。请补充字段或修订方案；没有覆盖原样例方案。",
            )
        )
    if extra:
        issues.append(
            FullDataIssue(
                code="extra_columns",
                severity="review",
                columns=extra,
                message="全量出现新字段："
                + "、".join(extra)
                + "。当前方案未使用这些字段，请核对其是否改变记录含义或提供必要信息。",
            )
        )
    if missing_other:
        issues.append(
            FullDataIssue(
                code="missing_unused_columns",
                severity="info",
                columns=missing_other,
                message="样例中的这些字段未在全量出现，但当前转换不依赖它们："
                + "、".join(missing_other),
            )
        )
    if type_changes:
        issues.append(
            FullDataIssue(
                code="changed_value_types",
                severity="review",
                columns=list(type_changes),
                message="部分字段出现样例未覆盖的原始值类型，请核对真实转换结果；没有自动强制转换。",
            )
        )
    if source.digest == session.source.digest and session.source.scope == "sample":
        issues.append(
            FullDataIssue(
                code="same_as_sample",
                severity="review",
                message="本次文件与原样例完全相同，请确认它确实代表本次任务的全量资料，而非仅用于理解结构的样例。",
            )
        )
    preview = None if missing_required else preview_recipe(source, recipe)
    new_targets: dict[str, list[Any]] = {}
    if preview:
        from src.workbench.temporal_split import is_pending_label, temporal_assignment

        if recipe.temporal_split:
            try:
                temporal = temporal_assignment(
                    preview.rows, recipe.temporal_split, recipe.group_columns
                )
                if temporal["excluded_rows"]:
                    issues.append(
                        FullDataIssue(
                            code="temporal_exclusions",
                            severity="review",
                            row_ids=[row["row_id"] for row in temporal["excluded_rows"]],
                            message="时间规则明确排除跨窗口、标签尚未成熟或关联排除行；原行保留在物化清单，不补标签、不随机换分区。",
                        )
                    )
            except ValueError as exc:
                issues.append(
                    FullDataIssue(code="temporal_invalid", severity="blocking", message=str(exc))
                )
        if not recipe.allow_multiple_targets:
            approved_answers: dict[str, set[str | None]] = {}
            for example in session.confirmed_examples:
                approved_answers.setdefault(example.expected_input, set()).add(
                    example.expected_target
                )
            changed_answers = [
                row.row_id
                for row in preview.rows
                if row.target is not None
                and row.input in approved_answers
                and row.target not in approved_answers[row.input]
            ]
            if changed_answers:
                issues.append(
                    FullDataIssue(
                        code="sample_answer_disagreement",
                        severity="blocking",
                        row_ids=changed_answers,
                        message="部分全量记录与已认可业务样例具有相同模型输入，但答案不同。请澄清标签或修改业务规则；按实际输入比对，未混用两份文件的行 ID。",
                    )
                )
        for status, message in (
            ("needs_label", "全量存在缺少监督答案的记录，需要补充标签；没有自动生成真值。"),
            ("invalid", "全量存在无法按已确认规则转换的记录，请查看问题行并修订规则或资料。"),
            ("conflict", "全量存在相同模型输入对应不同答案的冲突，需要澄清业务规则。"),
        ):
            rows = [
                r.row_id
                for r in preview.rows
                if r.status == status
                and not (status == "needs_label" and is_pending_label(r, recipe.temporal_split))
            ]
            if rows:
                issues.append(
                    FullDataIssue(
                        code=status,
                        severity="blocking",
                        row_ids=rows,
                        message=f"{message}（{len(rows)} 条）",
                    )
                )
        for field in recipe.targets:
            if field.value_kind != "categorical":
                continue
            sample_values = _target_values(session.preview, recipe.output_format, field.label)
            full_values = _target_values(preview, recipe.output_format, field.label)
            added = [
                value for identity, value in full_values.items() if identity not in sample_values
            ]
            if added:
                new_targets[field.column] = [value for value, _ in added]
                issues.append(
                    FullDataIssue(
                        code="new_categories",
                        severity="review",
                        columns=[field.column],
                        row_ids=[row_id for _, rows in added for row_id in rows],
                        message=f"类别字段 {field.column} 出现样例未覆盖的 {len(added)} 种答案，请核对是否属于目标类别；未自动合并或删除。",
                    )
                )
        missing_group = [
            row.row_id
            for row in preview.rows
            if any(is_missing(value) for value in row.group.values())
        ]
        if missing_group:
            issues.append(
                FullDataIssue(
                    code="missing_group_values",
                    severity="blocking",
                    columns=recipe.group_columns,
                    row_ids=missing_group,
                    message=f"{len(missing_group)} 条记录缺少方案声明的分组标识，无法据此隔离同一对象；先补充分组或澄清切分单位。",
                )
            )
    issues.append(
        FullDataIssue(
            code="split_not_validated",
            severity="info",
            message="本报告只验证全量转换与已发现的数据问题；独立分区、评测资料及训练消费尚未验收。",
        )
    )
    return FullDataValidation(
        source=source,
        profile=profile,
        preview=preview,
        sample_source_digest=session.source.digest,
        approved_recipe_digest=content_digest(recipe.model_dump()),
        sample_confirmed_revision=session.confirmed_revision,
        issues=issues,
        schema_drift={
            "missing_required": missing_required,
            "missing_other": missing_other,
            "extra": extra,
            "type_changes": type_changes,
        },
        new_target_values=new_targets,
        status="needs_revision" if any(i.severity == "blocking" for i in issues) else "review",
    )
