"""Bounded, inspectable source composition; no generated code or implicit coercion."""

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from decimal import Decimal, InvalidOperation
from typing import Annotated, Any, Literal

from pydantic import Field, model_validator

from src.workbench.forecast_labels import ForecastLabelsStep, build_forecast_labels
from src.workbench.intake_models import Contract, SampleSource, SourceRow
from src.workbench.sources import canonical, content_digest, is_missing


class JoinStep(Contract):
    operation: Literal["join"] = "join"
    right_source: str = Field(min_length=1)
    left_on: list[str] = Field(min_length=1)
    right_on: list[str] = Field(min_length=1)
    how: Literal["left", "inner"] = "left"
    cardinality: Literal["one_to_one", "many_to_one", "one_to_many"]
    prefix: str = Field(min_length=1)

    @model_validator(mode="after")
    def keys_match(self) -> JoinStep:
        if len(self.left_on) != len(self.right_on):
            raise ValueError("左右关联键数量必须相同。")
        return self


class ExplodeStep(Contract):
    operation: Literal["explode"] = "explode"
    column: str
    target_column: str
    empty: Literal["keep", "error"]


class ExtractStep(Contract):
    operation: Literal["extract"] = "extract"
    column: str
    path: list[str | int] = Field(min_length=1)
    target_column: str


class ConversationStep(Contract):
    operation: Literal["conversation"] = "conversation"
    group_columns: list[str] = Field(min_length=1)
    order_column: str
    order_type: Literal["numeric", "text"]
    role_column: str
    content_column: str
    history_column: str = "history"
    answer_column: str = "answer"


CompositionStep = Annotated[
    JoinStep | ExplodeStep | ExtractStep | ConversationStep | ForecastLabelsStep,
    Field(discriminator="operation"),
]


class CompositionRecipe(Contract):
    base_source: str = Field(min_length=1)
    steps: list[CompositionStep] = Field(min_length=1)


def required_sources(recipe: CompositionRecipe | dict) -> set[str]:
    recipe = CompositionRecipe.model_validate(recipe)
    names = {recipe.base_source}
    for step in recipe.steps:
        if isinstance(step, JoinStep):
            names.add(step.right_source)
        elif isinstance(step, ForecastLabelsStep):
            names.update((step.price_source, step.calendar_source))
    return names


class SourceRef(Contract):
    source_digest: str
    row_id: str


class CompositionIssue(Contract):
    severity: Literal["blocking", "review", "info"]
    code: str
    message: str
    step: int
    origins: list[SourceRef] = Field(default_factory=list)
    row_ids: list[str] = Field(default_factory=list)


class CompositionStepResult(Contract):
    operation: str
    input_rows: int
    output_rows: int
    status: Literal["complete", "blocked"]


class CompositionResult(Contract):
    source: SampleSource
    issues: list[CompositionIssue]
    steps: list[CompositionStepResult]
    origins: dict[str, list[SourceRef]]

    @property
    def can_confirm(self) -> bool:
        return bool(self.source.rows) and not any(i.severity == "blocking" for i in self.issues)

    @property
    def requires_review(self) -> bool:
        return any(i.severity == "review" for i in self.issues)


def _refs(refs: list[SourceRef]) -> list[SourceRef]:
    return list({(ref.source_digest, ref.row_id): ref for ref in refs}.values())


def _key(values: dict[str, Any], columns: list[str]) -> str | None:
    values = [values.get(column) for column in columns]
    if any(is_missing(value) or isinstance(value, (dict, list)) for value in values):
        return None
    return canonical(values)


def compose_sources(
    sources: dict[str, SampleSource], recipe: CompositionRecipe
) -> CompositionResult:
    """Run explicit linear operators and keep source-qualified lineage on every result."""
    if recipe.base_source not in sources:
        raise ValueError("基础数据源不存在。")
    base = sources[recipe.base_source]
    used_sources = {recipe.base_source}
    columns = list(base.columns)
    rows = [
        (deepcopy(row.values), [SourceRef(source_digest=base.digest, row_id=row.row_id)])
        for row in base.rows
    ]
    issues: list[CompositionIssue] = []
    summaries: list[CompositionStepResult] = []

    for index, step in enumerate(recipe.steps):
        before = len(rows)
        issue_start = len(issues)

        def issue(
            severity: str,
            code: str,
            message: str,
            refs: list[SourceRef] | tuple = (),
            step_index: int = index,
        ) -> None:
            issues.append(
                CompositionIssue(
                    severity=severity,
                    code=code,
                    message=message,
                    step=step_index,
                    origins=_refs(list(refs)),
                )
            )

        def require(required: list[str], available: list[str], emit=issue) -> bool:
            absent = sorted(set(required) - set(available))
            if absent:
                emit("blocking", "missing_columns", f"缺少字段：{', '.join(absent)}。")
            return not absent

        if isinstance(step, JoinStep):
            right = sources.get(step.right_source)
            if right is not None:
                used_sources.add(step.right_source)
            if right is None:
                issue("blocking", "missing_source", f"关联数据源 {step.right_source} 不存在。")
            elif require(step.left_on, columns) and require(step.right_on, right.columns):
                additions = [step.prefix + column for column in right.columns]
                if set(additions) & set(columns):
                    issue(
                        "blocking", "column_collision", "关联后的列名与现有字段冲突，请修改前缀。"
                    )
                else:
                    left_keys: dict[str, list] = defaultdict(list)
                    right_keys: dict[str, list] = defaultdict(list)
                    for values, refs in rows:
                        key = _key(values, step.left_on)
                        if key is None:
                            issue("blocking", "empty_join_key", "左表关联键为空或不是标量。", refs)
                        else:
                            left_keys[key].append((values, refs))
                    for row in right.rows:
                        ref = SourceRef(source_digest=right.digest, row_id=row.row_id)
                        key = _key(row.values, step.right_on)
                        if key is None:
                            issue("blocking", "empty_join_key", "右表关联键为空或不是标量。", [ref])
                        else:
                            right_keys[key].append((row.values, [ref]))
                    sides = []
                    if step.cardinality in {"one_to_one", "one_to_many"}:
                        sides.append(("左表", left_keys))
                    if step.cardinality in {"one_to_one", "many_to_one"}:
                        sides.append(("右表", right_keys))
                    for side, keyed in sides:
                        for matches in keyed.values():
                            if len(matches) > 1:
                                issue(
                                    "blocking",
                                    "cardinality_violation",
                                    f"{side}关联键重复，不符合声明的关联基数。",
                                    [ref for _, refs in matches for ref in refs],
                                )
                    if not any(i.severity == "blocking" for i in issues[issue_start:]):
                        joined = []
                        for values, refs in rows:
                            matches = right_keys.get(_key(values, step.left_on), [])
                            if not matches:
                                issue(
                                    "review",
                                    "unmatched_left",
                                    "左表行未匹配；left 保留空值，inner 按声明排除该行。",
                                    refs,
                                )
                                if step.how == "left":
                                    joined.append(({**values, **dict.fromkeys(additions)}, refs))
                            for right_values, right_refs in matches:
                                joined.append(
                                    (
                                        {
                                            **values,
                                            **{
                                                step.prefix + col: deepcopy(right_values.get(col))
                                                for col in right.columns
                                            },
                                        },
                                        _refs(refs + right_refs),
                                    )
                                )
                        for key, matches in right_keys.items():
                            if key not in left_keys:
                                issue(
                                    "info",
                                    "unused_right",
                                    "右表这些行没有对应左表行，未参与输出。",
                                    [ref for _, refs in matches for ref in refs],
                                )
                        if len(joined) > before:
                            issue(
                                "review",
                                "join_expansion",
                                f"关联将 {before} 行扩展为 {len(joined)} 行，请确认每行训练含义。",
                            )
                        rows, columns = joined, columns + additions
        elif isinstance(step, (ExplodeStep, ExtractStep)):
            if require([step.column], columns):
                if not step.target_column.strip() or step.target_column in columns:
                    issue("blocking", "column_collision", "派生字段必须是非空新列，不能覆盖原值。")
                else:
                    output = []
                    for values, refs in rows:
                        value = values.get(step.column)
                        if isinstance(step, ExplodeStep):
                            if not isinstance(value, list):
                                issue(
                                    "blocking",
                                    "not_list",
                                    "展开仅接受真实列表，不自动解析字符串。",
                                    refs,
                                )
                                output.append(({**values, step.target_column: None}, refs))
                            elif not value:
                                issue(
                                    "review" if step.empty == "keep" else "blocking",
                                    "empty_list",
                                    "空列表行已保留，派生值为空。",
                                    refs,
                                )
                                output.append(({**values, step.target_column: None}, refs))
                            else:
                                output.extend(
                                    ({**values, step.target_column: deepcopy(item)}, refs)
                                    for item in value
                                )
                        else:
                            try:
                                for part in step.path:
                                    if (isinstance(value, dict) and isinstance(part, str)) or (
                                        isinstance(value, list) and type(part) is int and part >= 0
                                    ):
                                        value = value[part]
                                    else:
                                        raise KeyError(part)
                            except (KeyError, IndexError):
                                issue(
                                    "blocking",
                                    "missing_path",
                                    "嵌套路径不存在或类型不匹配，原行保留。",
                                    refs,
                                )
                                value = None
                            output.append(({**values, step.target_column: deepcopy(value)}, refs))
                    if isinstance(step, ExplodeStep) and len(output) > before:
                        issue(
                            "review",
                            "explode_expansion",
                            f"列表展开将 {before} 行扩展为 {len(output)} 行。",
                        )
                    rows, columns = output, columns + [step.target_column]
        elif isinstance(step, ForecastLabelsStep):
            price_source = sources.get(step.price_source)
            calendar_source = sources.get(step.calendar_source)
            if price_source is None or calendar_source is None:
                issue(
                    "blocking", "missing_source", "需要明确的行情与交易日历来源，不能伪造未来标签。"
                )
            else:
                used_sources.update((step.price_source, step.calendar_source))
                if require(
                    [step.event_id_column, step.symbol_column, step.available_at_column], columns
                ):
                    try:
                        labeled = build_forecast_labels(
                            [values for values, _ in rows], price_source, calendar_source, step
                        )
                        for found in labeled["issues"]:
                            refs = [SourceRef.model_validate(ref) for ref in found["refs"]]
                            if found["event_index"] is not None:
                                refs.extend(rows[found["event_index"]][1])
                            issue(found["severity"], found["code"], found["message"], refs)
                        rows = [
                            (
                                output,
                                _refs(
                                    [
                                        *rows[position][1],
                                        *[
                                            SourceRef.model_validate(ref)
                                            for ref in labeled["origins"][position]
                                        ],
                                    ]
                                ),
                            )
                            for position, output in enumerate(labeled["rows"])
                        ]
                        columns += labeled["columns"]
                    except ValueError as exc:
                        issue("blocking", "invalid_forecast_contract", str(exc))
        else:
            required = step.group_columns + [
                step.order_column,
                step.role_column,
                step.content_column,
            ]
            output_columns = step.group_columns + [step.history_column, step.answer_column]
            if require(required, columns):
                if len(set(output_columns)) != len(output_columns) or any(
                    not col.strip() for col in output_columns
                ):
                    issue("blocking", "column_collision", "会话输出字段名必须非空且互不重复。")
                else:
                    groups: dict[str, list] = defaultdict(list)
                    for values, refs in rows:
                        key = _key(values, step.group_columns)
                        if key is None:
                            issue(
                                "blocking",
                                "missing_conversation",
                                "缺少有效会话标识，该行不能归入任何会话。",
                                refs,
                            )
                        else:
                            groups[key].append((values, refs))
                    output = []
                    for group in groups.values():
                        group_refs = [ref for _, refs in group for ref in refs]
                        ordered = []
                        invalid = False
                        for values, refs in group:
                            order = values.get(step.order_column)
                            try:
                                if is_missing(order) or isinstance(order, bool):
                                    raise ValueError
                                if step.order_type == "numeric":
                                    order = Decimal(str(order))
                                    if not order.is_finite():
                                        raise ValueError
                                elif not isinstance(order, str):
                                    raise ValueError
                            except (ValueError, InvalidOperation):
                                issue(
                                    "blocking",
                                    "invalid_order",
                                    "会话顺序缺失或不符合声明类型，不能猜测顺序。",
                                    refs,
                                )
                                invalid = True
                            role = values.get(step.role_column)
                            if (
                                not isinstance(role, str)
                                or role not in {"system", "user", "assistant"}
                                or not isinstance(values.get(step.content_column), str)
                                or not values[step.content_column].strip()
                            ):
                                issue(
                                    "blocking",
                                    "invalid_message",
                                    "会话角色或正文无效，不能生成可信训练对。",
                                    refs,
                                )
                                invalid = True
                            ordered.append((order, values, refs))
                        if invalid:
                            issue(
                                "blocking",
                                "blocked_conversation",
                                "会话存在无效消息，整组保留在原始来源并等待修正，未生成训练行。",
                                group_refs,
                            )
                            continue
                        if len({order for order, _, _ in ordered}) != len(ordered):
                            issue(
                                "blocking",
                                "duplicate_order",
                                "同一会话存在重复顺序，整个会话等待澄清。",
                                group_refs,
                            )
                            continue
                        history, history_refs = [], []
                        start_count = len(output)
                        for _, values, refs in sorted(ordered, key=lambda item: item[0]):
                            role, content = values[step.role_column], values[step.content_column]
                            if role == "assistant":
                                if not any(message["role"] == "user" for message in history):
                                    issue(
                                        "blocking",
                                        "missing_user_context",
                                        "assistant 回答之前没有用户输入，不能生成训练对。",
                                        refs,
                                    )
                                else:
                                    output.append(
                                        (
                                            {
                                                **{col: values[col] for col in step.group_columns},
                                                step.history_column: canonical(history),
                                                step.answer_column: content,
                                            },
                                            _refs(history_refs + refs),
                                        )
                                    )
                            history.append({"role": role, "content": content})
                            history_refs.extend(refs)
                        if len(output) == start_count:
                            issue(
                                "review",
                                "no_assistant_answer",
                                "该会话没有可生成的 assistant 监督答案，未生成训练行。",
                                group_refs,
                            )
                    rows, columns = output, output_columns
        blocked = any(issue.severity == "blocking" for issue in issues[issue_start:])
        summaries.append(
            CompositionStepResult(
                operation=step.operation,
                input_rows=before,
                output_rows=len(rows),
                status="blocked" if blocked else "complete",
            )
        )
        if blocked:
            break

    origins = {f"r{index:06d}": _refs(refs) for index, (_, refs) in enumerate(rows, 1)}
    output_rows = [
        SourceRow(row_id=f"r{index:06d}", line=index, values=values)
        for index, (values, _) in enumerate(rows, 1)
    ]
    digest = content_digest(
        {
            "recipe": recipe.model_dump(),
            "sources": {name: sources[name].digest for name in sorted(used_sources)},
            "rows": [row.model_dump() for row in output_rows],
            "origins": {key: [ref.model_dump() for ref in refs] for key, refs in origins.items()},
        }
    )
    source = SampleSource(
        name=f"composed-{base.name}.jsonl",
        digest=digest,
        scope="full" if all(sources[name].scope == "full" for name in used_sources) else "sample",
        format="jsonl",
        encoding="utf-8",
        columns=columns,
        rows=output_rows,
    )
    return CompositionResult(source=source, issues=issues, steps=summaries, origins=origins)
