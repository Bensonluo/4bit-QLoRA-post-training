"""Validated, isolated one-to-one enrichment for long-tail source formats."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Literal

from pydantic import Field, model_validator

from src.workbench.intake_models import Contract, SampleSource, SourceRow
from src.workbench.sandbox import TransformCase, TransformSandbox
from src.workbench.sources import canonical, content_digest

ROW_ID = "__row_id"


class AdapterExample(Contract):
    name: str
    rows: list[dict[str, Any]] = Field(min_length=1)
    expected_rows: list[dict[str, Any]] | None = None
    kind: Literal["business", "counterexample"]
    expect_error: bool = False


class AdapterRecipe(Contract):
    source_code: str = Field(
        max_length=128 * 1024,
        description="定义 transform(rows, config)。每行保留所有原字段与 __row_id，仅新增声明字段；可导入 re/json/math。代码仅在真实OS隔离执行。",
    )
    config: dict[str, Any] = Field(default_factory=dict)
    new_columns: list[str] = Field(min_length=1, description="新增的解析字段；不能覆盖原字段。")
    examples: list[AdapterExample] = Field(
        min_length=2, description="至少一条基于真实输入的业务期望样例，及独立反例；每行带__row_id。"
    )

    @model_validator(mode="after")
    def validate_examples(self):
        if {case.kind for case in self.examples} != {"business", "counterexample"}:
            raise ValueError("适配规则必须有业务期望样例和独立反例。")
        if len(set(self.new_columns)) != len(self.new_columns) or any(
            not c.strip() or c == ROW_ID for c in self.new_columns
        ):
            raise ValueError("新增字段名不能为空、重复或占用来源标识。")
        return self


class AdapterResult(Contract):
    source: SampleSource
    validation: dict[str, Any]
    spec_digest: str
    origins: dict[str, list[dict[str, str]]]


def apply_adapter(
    source: SampleSource,
    recipe: AdapterRecipe,
    *,
    sandbox: TransformSandbox | None = None,
    require_source_example: bool = True,
) -> AdapterResult:
    """A successful sandbox result is still checked for row preservation and lineage."""
    if ROW_ID in source.columns or set(recipe.new_columns) & set(source.columns):
        raise ValueError("适配器字段与原始资料冲突，请使用新的解析字段名。")
    inputs = [{**row.values, ROW_ID: row.row_id} for row in source.rows]
    if require_source_example:
        real = {canonical(row) for row in inputs}
        if not any(
            canonical(row) in real
            for case in recipe.examples
            if case.kind == "business"
            for row in case.rows
        ):
            raise ValueError("至少一个业务验收例必须引用当前资料的真实输入及行 ID。")
    runner = sandbox or TransformSandbox()
    validation = runner.validate(
        recipe.source_code,
        [
            TransformCase(
                case.name,
                case.rows,
                case.expected_rows,
                recipe.config,
                case.kind,
                case.expect_error,
            )
            for case in recipe.examples
        ],
    )
    if validation.status != "passed":
        errors = "; ".join(
            case.get("error", "") or case["name"] for case in validation.cases if not case["passed"]
        )
        raise ValueError(f"隔离适配验证未通过（{validation.status}）：{errors}；未在宿主执行。")
    execution = runner.run(recipe.source_code, inputs, recipe.config)
    if execution.status != "passed" or execution.rows is None:
        raise ValueError(f"真实资料的隔离转换未通过：{execution.error}；没有采用转换结果。")
    outputs = execution.rows
    expected_ids = {row.row_id for row in source.rows}
    ids = [row.get(ROW_ID) for row in outputs]
    if (
        any(not isinstance(identity, str) for identity in ids)
        or len(ids) != len(set(ids))
        or set(ids) != expected_ids
    ):
        raise ValueError("当前适配只支持一对一新增字段，不能丢行、复制行或更换来源标识。")
    by_id = {row[ROW_ID]: row for row in outputs}
    columns = [*source.columns, *recipe.new_columns]
    rows = []
    for original in source.rows:
        output = by_id[original.row_id]
        if set(output) != set(original.values) | set(recipe.new_columns) | {ROW_ID}:
            raise ValueError("适配输出必须保留原字段并仅增加声明的新字段。")
        if any(
            canonical(output[key]) != canonical(value) for key, value in original.values.items()
        ):
            raise ValueError("适配代码改变了原始字段；请将处理结果写入新字段供业务预览。")
        rows.append(
            SourceRow(
                row_id=original.row_id,
                line=original.line,
                values={key: value for key, value in output.items() if key != ROW_ID},
            )
        )
    identity = content_digest(
        {
            "source": source.digest,
            "recipe": recipe.model_dump(),
            "rows": [row.model_dump() for row in rows],
        }
    )
    result = SampleSource(
        name=f"adapted-{source.name}.jsonl",
        digest=identity,
        scope=source.scope,
        format="jsonl",
        encoding="utf-8",
        columns=columns,
        rows=rows,
    )
    return AdapterResult(
        source=result,
        validation=asdict(validation),
        spec_digest=content_digest(recipe.model_dump()),
        origins={
            row.row_id: [{"source_digest": source.digest, "row_id": row.row_id}]
            for row in source.rows
        },
    )
