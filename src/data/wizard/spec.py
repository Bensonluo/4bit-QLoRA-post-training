"""数据向导配置：字段映射与流程规格。

把「原始表格 → 训练集」的专家决策固化成可校验的配置对象：
- FieldMapping 声明原始列到语义角色的映射（哪列是别名、哪列是标准名）；
- WizardSpec 声明生成参数（候选数、切分比例、随机种子）。

校验全部放在 __post_init__ / validate()，与 config/ 下的约定一致。
"""

from dataclasses import dataclass, field

SPLITS = ("train", "val", "test")


class WizardError(Exception):
    """向导流程中的用户可修复错误（消息面向非算法工程师，中文）。"""


@dataclass(frozen=True)
class FieldMapping:
    """原始列 → 语义角色的映射。

    两种典型形态都支持：
    - 配对模式：每行就是一条「查询 → 标准」记录（mapping.query 指向查询列）；
    - 知识库模式：每行一个标准条目，变体写在同一格（mapping.variants，按分隔符切开）。
    两者可同时存在，行内查询取并集。
    """

    standard_name: str = ""
    query: str | None = None
    code: str | None = None
    variants: str | None = None
    entity_type: str | None = None
    entity_type_default: str = "entity"

    def validate(self, columns: list[str]) -> list[str]:
        """校验映射列都存在于实际表格；返回错误消息列表（空 = 通过）。"""
        errors: list[str] = []
        known = set(columns)
        if not self.standard_name:
            errors.append("缺少必填映射：standard_name（标准名列）。")
        elif self.standard_name not in known:
            errors.append(f"标准名列 '{self.standard_name}' 不在表格列中（可用列: {columns}）。")
        for role, col in (
            ("query", self.query),
            ("code", self.code),
            ("variants", self.variants),
            ("entity_type", self.entity_type),
        ):
            if col is not None and col not in known:
                errors.append(f"{role} 列 '{col}' 不在表格列中（可用列: {columns}）。")
        return errors


@dataclass(frozen=True)
class WizardSpec:
    """一次向导运行的完整参数。"""

    mapping: FieldMapping = field(default_factory=FieldMapping)
    template: str = "medical_entity"
    split_ratios: tuple[float, float, float] = (0.8, 0.1, 0.1)
    n_candidates: int = 8
    noise_augment: bool = False
    dedup: bool = True
    seed: int = 42

    def __post_init__(self) -> None:
        if not self.template:
            raise WizardError("template 不能为空。")
        if len(self.split_ratios) != 3:
            raise WizardError("split_ratios 必须是 (train, val, test) 三元组。")
        if any(r <= 0 for r in self.split_ratios):
            raise WizardError(f"split_ratios 每一项必须大于 0，当前: {self.split_ratios}。")
        if sum(self.split_ratios) < 0.99 or sum(self.split_ratios) > 1.01:
            raise WizardError(f"split_ratios 之和应约为 1.0，当前: {self.split_ratios}。")
        if self.n_candidates < 2:
            raise WizardError(
                f"n_candidates 至少为 2（否则无法构成判别任务），当前: {self.n_candidates}。"
            )
