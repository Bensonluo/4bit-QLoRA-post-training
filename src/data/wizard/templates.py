"""垂类模板：把归一化表格变成「实体匹配选择题」样本。

模板是向导里沉淀专家判断的地方。医疗实体匹配模板移植了
domains/medical_entity/prepare_data.py 验证过的生成逻辑：
难度按编辑距离分层、前缀硬负例 + 随机负例、候选随机打乱（防位置捷径）。

新垂类通过 register_template() 注册，向导的其余环节（体检/切分/导出）复用。
"""

import json
import random
import zlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from src.data.wizard.importers import RawTable, split_variants
from src.data.wizard.spec import FieldMapping, WizardError, WizardSpec

INSTRUCTION_TEXT = (
    "从候选列表中选出与输入实体匹配的标准名称。"
    '输出JSON：{"match_index": 序号, "standard_name": "标准名", '
    '"code": "编码", "confidence": 置信度}'
)


@dataclass(frozen=True)
class Candidate:
    name: str
    code: str | None
    label: bool  # True = 正确答案
    spec: str | None = None  # 规格（产品匹配任务用，其余模板忽略）


@dataclass
class MatchingSample:
    """中间格式样本（与文件格式解耦，供体检/切分/导出复用）。"""

    query: str
    standard_name: str
    code: str | None
    entity_type: str
    difficulty: str
    candidates: list[Candidate]
    source_row: int  # 原始表格行号（1 起），体检报错定位用


@dataclass(frozen=True)
class RowIssue:
    """被跳过的原始行及原因。"""

    row: int
    reason: str


@dataclass(frozen=True)
class BuildResult:
    samples: list[MatchingSample] = field(default_factory=list)
    dropped: list[RowIssue] = field(default_factory=list)


class DomainTemplate(ABC):
    """垂类模板基类：build_samples 生成中间样本，format_record 落成训练格式。

    与 src/data/base.py 的 BaseDataset 同款 ABC 模式。
    """

    name: str

    @abstractmethod
    def describe(self) -> str:
        """模板说明（--suggest 与 UI 展示用）。"""
        pass

    @abstractmethod
    def build_samples(
        self, table: RawTable, mapping: FieldMapping, spec: WizardSpec
    ) -> BuildResult:
        """从归一化表格生成中间样本（含难度分层与候选构造）。"""
        pass

    @abstractmethod
    def format_record(self, sample: MatchingSample) -> dict[str, object]:
        """中间样本 → Alpaca 训练记录。"""
        pass


# ─── 难度与文本工具（移植自 prepare_data.py，独立实现避免脚本层反向依赖）──


def edit_distance(s1: str, s2: str) -> int:
    if len(s1) < len(s2):
        return edit_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    prev = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr = [i + 1]
        for j, c2 in enumerate(s2):
            curr.append(min(prev[j + 1] + 1, curr[j] + 1, prev[j] + (c1 != c2)))
        prev = curr
    return prev[-1]


def classify_difficulty(query: str, standard: str) -> str:
    """easy: 完全一致或包含；medium: 编辑距离 ≤ 2；hard: 其余。"""
    if query == standard:
        return "easy"
    q, s = query.lower().replace(" ", ""), standard.lower().replace(" ", "")
    if q == s or q in s or s in q:
        return "easy"
    if edit_distance(q, s) <= 2:
        return "medium"
    return "hard"


# ─── 医疗实体匹配模板 ───────────────────────────────────────────────


class EntityMatchingTemplate(DomainTemplate):
    """医疗实体匹配：别名/变体 → 标准名+编码 的候选列表选择题。"""

    name = "medical_entity"

    def describe(self) -> str:
        return (
            "医疗实体匹配模板：把「查询/别名 → 标准名+编码」表格变成候选列表选择题。\n"
            "默认按标准实体切分 train/val/test（防答案泄漏），候选随机打乱（防位置偏差），\n"
            "难度按编辑距离自动分层（easy/medium/hard）。\n"
            "需要列：标准名（必填）；查询、编码、变体、类型（可选）。"
        )

    def build_samples(
        self, table: RawTable, mapping: FieldMapping, spec: WizardSpec
    ) -> BuildResult:
        errors = mapping.validate(table.columns)
        if errors:
            raise WizardError("字段映射校验失败:\n" + "\n".join(f"  - {e}" for e in errors))

        # 全量标准条目池（负例来源）
        standards: list[tuple[str, str | None]] = []
        seen: set[str] = set()
        for row in table.rows:
            std = row.get(mapping.standard_name)
            if std is None or std in seen:
                continue
            seen.add(std)
            standards.append((std, row.get(mapping.code) if mapping.code else None))

        result = BuildResult()
        for idx, row in enumerate(table.rows, start=1):
            std = row.get(mapping.standard_name)
            if std is None:
                result.dropped.append(RowIssue(idx, "标准名为空"))
                continue

            queries: list[str] = []
            if mapping.query:
                q = row.get(mapping.query)
                if q is not None and q != std and q not in queries:
                    queries.append(q)
                elif q == std:
                    queries.append(q)  # 标准名自身作查询（easy 样本）
            if mapping.variants:
                for v in split_variants(row.get(mapping.variants)):
                    if v not in queries:
                        queries.append(v)
            if not queries:
                queries.append(std)  # 无任何查询值时退回标准名，保证行可用
            queries = queries[:64]  # 单行变体爆炸保护

            code = row.get(mapping.code) if mapping.code else None
            entity_type = (
                row.get(mapping.entity_type) if mapping.entity_type else None
            ) or mapping.entity_type_default

            for query in queries:
                sample = self._make_sample(query, std, code, entity_type, standards, spec, idx)
                result.samples.append(sample)
        return result

    def _make_sample(
        self,
        query: str,
        standard: str,
        code: str | None,
        entity_type: str,
        standards: list[tuple[str, str | None]],
        spec: WizardSpec,
        row: int,
    ) -> MatchingSample:
        # 跨进程稳定（内置 hash 带随机盐，会破坏同种子同产出的可复现承诺）
        rng = random.Random(spec.seed * 1_000_003 + row * 131 + zlib.crc32(query.encode()))
        negatives = self._pick_negatives(standard, standards, spec.n_candidates - 1, rng)
        candidates = [Candidate(standard, code, True)] + [
            Candidate(name, neg_code, False) for name, neg_code in negatives
        ]
        rng.shuffle(candidates)
        return MatchingSample(
            query=query,
            standard_name=standard,
            code=code,
            entity_type=entity_type,
            difficulty=classify_difficulty(query, standard),
            candidates=candidates,
            source_row=row,
        )

    @staticmethod
    def _pick_negatives(
        standard: str,
        standards: list[tuple[str, str | None]],
        n: int,
        rng: random.Random,
    ) -> list[tuple[str, str | None]]:
        """前缀硬负例优先（与标准名最易混淆），其余随机补齐。"""
        prefix = standard[:2]
        hard = [(s, c) for s, c in standards if s != standard and prefix and s.startswith(prefix)]
        rng.shuffle(hard)
        pool = [(s, c) for s, c in standards if s != standard and (s, c) not in hard]
        rng.shuffle(pool)
        picked = hard[:n]
        picked += pool[: max(0, n - len(picked))]
        return picked

    def format_record(self, sample: MatchingSample) -> dict[str, object]:
        match_idx = next((i for i, c in enumerate(sample.candidates) if c.label), None)
        if match_idx is None:
            raise WizardError(f"样本（行 {sample.source_row}）没有标注正确候选，无法格式化。")
        lines = []
        for i, c in enumerate(sample.candidates):
            lines.append(f"{i + 1}. {c.name} ({c.code})" if c.code else f"{i + 1}. {c.name}")
        output: dict[str, object] = {
            "match_index": match_idx + 1,
            "standard_name": sample.standard_name,
        }
        if sample.code is not None:
            output["code"] = sample.code
        output["confidence"] = 0.95
        return {
            "instruction": INSTRUCTION_TEXT,
            "input": f"输入实体: {sample.query}\n候选:\n" + "\n".join(lines),
            "output": json.dumps(output, ensure_ascii=False),
            "metadata": {
                "entity_type": sample.entity_type,
                "difficulty": sample.difficulty,
            },
        }


# ─── 模板注册表（与 ui/components/domain_adapters.py 同款模式）────────


_TEMPLATES: dict[str, DomainTemplate] = {}


def register_template(template: DomainTemplate) -> None:
    if template.name in _TEMPLATES:
        raise WizardError(f"模板 '{template.name}' 已注册。")
    _TEMPLATES[template.name] = template


def get_template(name: str) -> DomainTemplate:
    if name not in _TEMPLATES:
        available = ", ".join(sorted(_TEMPLATES)) or "（无）"
        raise WizardError(f"未知模板 '{name}'。可用模板: {available}。")
    return _TEMPLATES[name]


def available_templates() -> list[str]:
    return sorted(_TEMPLATES)


register_template(EntityMatchingTemplate())
