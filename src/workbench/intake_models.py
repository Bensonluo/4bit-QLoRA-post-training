"""Contracts for understanding a user's goal and inspecting their sample data."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SourceRow(Contract):
    row_id: str
    line: int
    values: dict[str, Any]


class SampleSource(Contract):
    name: str
    digest: str
    scope: Literal["sample", "full"] = "sample"
    format: Literal["csv", "jsonl", "xlsx", "xls"]
    encoding: str = ""
    delimiter: str = ""
    sheet: str = Field(
        default="",
        description="Excel 实际读取的 sheet 名称；CSV/JSONL 为空。随来源持久化，"
        "供全量复用原文件时按同一 sheet 重读，也让 profile 标注可从来源本身还原。",
    )
    sheet_note: str = Field(
        default="",
        description="多 Sheet 工作簿的读取范围说明（含多个 sheet 时非空）："
        "如实记录本次读了哪个 sheet、哪些未读取；由读取时按实际选择生成。",
    )
    merged_note: str = Field(
        default="",
        description="Excel 合并单元格的如实说明（读取的 sheet 存在与数据区相交的合并区时非空）："
        "点名合并范围与受影响列、说明合并区除左上角外均读为空值；不自动填充，"
        "是否取消合并由用户决定。当前仅 xlsx 检测，xls 不检测。",
    )
    formula_note: str = Field(
        default="",
        description="Excel 公式单元格无缓存计算结果的如实说明（读取的 sheet 的数据区存在"
        "此类公式格时非空）：点名坐标与受影响列、说明这些公式读为空值；不自动计算，"
        "是否用 Excel 打开保存以生成计算结果由用户决定。带缓存值的公式格正常读取、"
        "不列入。当前仅 xlsx 检测，xls 不检测。",
    )
    hidden_note: str = Field(
        default="",
        description="Excel 隐藏行/列的如实说明（读取的 sheet 的数据区存在隐藏行或隐藏列时"
        "非空）：点名行号/列名、说明隐藏行/列照常读入——Excel 中看不到的行列也会进入"
        "分析与训练；不自动排除，是否取消隐藏、删除不需要的行列由用户决定。"
        "当前仅 xlsx 检测，xls 不检测。",
    )
    blank_note: str = Field(
        default="",
        description="空行/全空行的如实说明（跨格式）：CSV/JSONL 存在被跳过的空行时点名"
        "行号并说明空行不进入分析与训练（读取行为不变）；Excel 数据区存在全空行"
        "（整行无值、照常读入为全空记录）时点名行号并说明会按缺少监督答案与分组"
        "标识处理。不自动补行、不自动排除，处理由用户决定。Excel 尾部空行在解析时"
        "自然消失、无从检测，不列入。xlsx 与 xls 均检测（基于解析后的记录，"
        "不依赖引擎特性）。",
    )
    dup_header_note: str = Field(
        default="",
        description="重复表头行的如实说明（跨格式）：数据区存在与表头完全相同的行"
        "（每个单元格都等于其列名，常见于导出拼接）时点名行号并说明该行按普通"
        "数据行读入——输入与答案都会是列名；样例侧不拦（以「就绪」进入预览、"
        "可能参与对比核验），全量侧会被全量验证硬拦。不自动删行，是否删除由"
        "用户决定。判定口径与全量侧 repeated_header_rows 一致（列数 ≥ 2 且每格"
        "等于列名）。xlsx 与 xls 均检测（基于解析后的记录）。",
    )
    columns: list[str]
    rows: list[SourceRow]


class Transform(Contract):
    operation: Literal["strip", "replace", "map_values", "parse_json"]
    old: str = ""
    new: str = ""
    mapping: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_parameters(self) -> Transform:
        if self.operation == "replace" and not self.old:
            raise ValueError("replace 必须指定非空 old。")
        if self.operation == "map_values" and not self.mapping:
            raise ValueError("map_values 必须提供明确映射。")
        return self


class FieldBinding(Contract):
    column: str = Field(min_length=1)
    label: str = Field(min_length=1)
    value_kind: Literal["unspecified", "categorical", "open_text", "numeric_continuous"] = Field(
        default="unspecified",
        description="根据业务目标明确答案是类别 categorical、开放文本 open_text，带小数的连续数值 numeric_continuous，尚未确定用 unspecified；不能仅根据列名或样例不同值数量猜测。numeric_continuous 只如实标注答案形态——当前训练仍按逐字字符串学习，不是数值回归。",
    )
    transforms: list[Transform] = Field(default_factory=list)


class TemporalSplitPolicy(Contract):
    available_at_column: str = Field(
        min_length=1, description="每行全部模型输入最晚可获得的带时区时间字段。"
    )
    prediction_at_column: str = Field(
        min_length=1, description="业务实际做出预测的带时区时间字段。"
    )
    label_end_at_column: str = Field(
        min_length=1, description="标签观察窗口结束时间；不得作为模型输入。"
    )
    validation_start: str
    test_start: str
    observation_end: str

    @model_validator(mode="after")
    def validate_temporal_policy(self):
        columns = [self.available_at_column, self.prediction_at_column, self.label_end_at_column]
        if any(not column.strip() for column in columns) or len(set(columns)) != 3:
            raise ValueError("三个时间字段必须非空且不同。")
        boundaries = []
        for value in (self.validation_start, self.test_start, self.observation_end):
            try:
                parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            except (TypeError, ValueError) as exc:
                raise ValueError("时间边界必须是带时区的ISO时间。") from exc
            if parsed.tzinfo is None or parsed.utcoffset() is None:
                raise ValueError("时间边界必须明确时区。")
            boundaries.append(parsed)
        if not boundaries[0] < boundaries[1] < boundaries[2]:
            raise ValueError("时间边界需满足validation_start < test_start < observation_end。")
        return self


class DataRecipe(Contract):
    instruction: str = Field(min_length=1)
    inputs: list[FieldBinding] = Field(min_length=1)
    targets: list[FieldBinding] = Field(default_factory=list)
    output_format: Literal["text", "json"] = "text"
    allow_multiple_targets: bool = False
    multiple_targets_reason: str = ""
    group_columns: list[str] = Field(default_factory=list)
    split_rationale: str = ""
    temporal_split: TemporalSplitPolicy | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
        description="时间预测任务的已确认信息可用时间/预测时间/标签成熟时间与分区边界；普通任务留空。",
    )

    @model_validator(mode="after")
    def validate_bindings(self) -> DataRecipe:
        for fields in (self.inputs, self.targets):
            names = [f.column for f in fields]
            labels = [f.label for f in fields]
            if len(names) != len(set(names)) or len(labels) != len(set(labels)):
                raise ValueError("同一组字段的来源和显示名不能重复。")
        if set(f.column for f in self.inputs) & set(f.column for f in self.targets):
            raise ValueError("同一列不能同时作为模型输入和监督答案。")
        if self.output_format == "text" and len(self.targets) > 1:
            raise ValueError("多字段答案请使用 json 输出，不能隐式拼接标签。")
        if self.allow_multiple_targets and not self.multiple_targets_reason.strip():
            raise ValueError("允许同输入多种答案时，必须说明业务依据。")
        if self.temporal_split and self.temporal_split.label_end_at_column in {
            f.column for f in self.inputs
        }:
            raise ValueError("标签窗口结束时间不得作为模型输入。")
        return self


class FieldRole(Contract):
    column: str
    role: Literal["input", "target", "group", "metadata", "unused", "unknown"]
    reason: str = Field(min_length=1)
    available_at_prediction: bool | None = None
    evidence_row_ids: list[str] = Field(default_factory=list)


class TaskSpec(Contract):
    goal: str = Field(min_length=1)
    usage_input: str = Field(min_length=1)
    desired_output: str = Field(min_length=1)
    row_meaning: str = Field(min_length=1)
    supervision_source: str
    success_criteria: list[str] = Field(min_length=1)
    field_roles: list[FieldRole]


class Finding(Contract):
    kind: Literal["observed", "hypothesis", "needs_business_input", "needs_full_data"]
    message: str = Field(min_length=1)
    evidence_row_ids: list[str] = Field(default_factory=list)


class Question(Contract):
    question_id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    why: str = Field(min_length=1)
    options: list[str] = Field(default_factory=list)
    blocks_confirmation: bool = True


class IntakeAnalysis(Contract):
    task: TaskSpec
    findings: list[Finding]
    questions: list[Question] = Field(default_factory=list)
    recipe: DataRecipe | None = None
    composition: dict[str, Any] | None = Field(
        default=None,
        description="可选的多源组合方案，必须与 preview_composition 实际运行的 CompositionRecipe 完全一致；普通单表映射留空。",
    )
    adapter: dict[str, Any] | None = Field(
        default=None,
        description="必须与 preview_adapter 隔离验证通过的 AdapterRecipe 一致；仅新增解析字段并保留来源。",
    )
    capability_gaps: list[str] = Field(
        default_factory=list,
        description="只列完成当前目标实际必需但工具不支持的转换能力。缺标签、需上传全量和无关的外部知识库等不属于工具能力缺口；常规字段映射预览成功时通常为空。",
    )
    training_approach: str = Field(min_length=1)
    next_steps: list[str] = Field(min_length=1)


class PreviewRow(Contract):
    row_id: str
    original: dict[str, Any]
    input: str
    target: str | None
    group: dict[str, Any]
    status: Literal["ready", "needs_label", "invalid", "conflict"]
    issues: list[str] = Field(default_factory=list)


class PreviewResult(Contract):
    source_digest: str
    recipe_digest: str
    instruction: str
    rows: list[PreviewRow]
    counts: dict[str, int]
    scope_note: str


class BusinessExample(Contract):
    row_id: str
    expected_input: str
    expected_target: str | None


class FullDataIssue(Contract):
    code: str
    severity: Literal["blocking", "review", "info"]
    message: str
    columns: list[str] = Field(default_factory=list)
    # These IDs belong exclusively to FullDataValidation.source, never session.source.
    row_ids: list[str] = Field(default_factory=list)


class FullDataValidation(Contract):
    source: SampleSource
    profile: dict[str, Any]
    preview: PreviewResult | None = None
    sample_source_digest: str
    approved_recipe_digest: str
    sample_confirmed_revision: int
    issues: list[FullDataIssue] = Field(default_factory=list)
    schema_drift: dict[str, Any] = Field(default_factory=dict)
    new_target_values: dict[str, list[Any]] = Field(default_factory=dict)
    status: Literal["needs_revision", "review", "confirmed", "stale"]
    confirmed_revision: int | None = None
    confirmed_examples: list[BusinessExample] = Field(default_factory=list)
    sources: dict[str, SampleSource] = Field(default_factory=dict)
    composition_report: dict[str, Any] | None = None
    adapter_report: dict[str, Any] | None = None


class DatasetArtifact(Contract):
    name: str
    version: str
    registry_root: str
    source_digest: str
    recipe_digest: str
    full_confirmed_revision: int
    paths: dict[str, str]
    data_config: dict[str, Any]
    statistics: dict[str, Any]
    evaluation_suite: dict[str, Any] | None = None


class IntakeSession(Contract):
    session_id: str
    revision: int = 0
    goal: str
    data_description: str = ""
    source: SampleSource
    sources: dict[str, SampleSource] = Field(default_factory=dict)
    composition_report: dict[str, Any] | None = None
    adapter_report: dict[str, Any] | None = None
    profile: dict[str, Any]
    answers: list[dict[str, str]] = Field(default_factory=list)
    analysis: IntakeAnalysis | None = None
    previous_analysis: IntakeAnalysis | None = None
    preview: PreviewResult | None = None
    confirmed_revision: int | None = None
    confirmed_examples: list[BusinessExample] = Field(default_factory=list)
    full_data: FullDataValidation | None = None
    dataset: DatasetArtifact | None = None
    training_preflight: dict[str, Any] | None = None
    label_verification: dict[str, Any] | None = Field(
        default=None,
        description=(
            "最新盲标核验结论（由 IntakeService 加载时附加）：用户对抽样行隐藏答案作答，"
            "与数据标签比对一致性。绑定全量来源与配方摘要，verdict=verified 才允许训练准备。"
        ),
    )
    agent_model: str = ""
    tool_trace: list[dict[str, Any]] = Field(default_factory=list)
    updated_at: str = ""
