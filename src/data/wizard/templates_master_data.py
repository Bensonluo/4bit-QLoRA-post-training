"""主数据匹配模板：机构 + 产品双任务，messages chat 格式。

移植 domains/master_data/scripts/generate_data.py 验证过的双任务设计：
同一模型同时学两个任务，任务由 system prompt 区分（多任务 SFT 的标准做法）。
机构任务逐候选输出 {matched, confidence}；产品任务按核心名一致性输出 A/B/D 等级。
与医疗模板（Alpaca 选择题）的差异证明模板插件体系对不同任务格式成立：
导出的是 messages chat 记录（TRL SFTTrainer 原生支持），可直接喂给
domains/master_data/scripts/train.py。
"""

import json
import random
import zlib

from src.data.wizard.importers import RawTable, split_variants
from src.data.wizard.spec import FieldMapping, WizardError, WizardSpec
from src.data.wizard.templates import (
    BuildResult,
    Candidate,
    DomainTemplate,
    MatchingSample,
    RowIssue,
    classify_difficulty,
    register_template,
)

INST_SYSTEM_PROMPT = (
    "角色：专业的医药机构主数据匹配审核员\n"
    "任务：逐一判断【输入机构】与列表中的每个【候选机构】是否代表同一物理实体。\n"
    "严格规则：\n"
    "1. 必须对每个候选机构【独立】进行验证，候选之间绝不能互相干扰。\n"
    "2. 严格按优先级1-6执行短路验证（精确>地理>大学>粒度>修饰词>辅助）。"
    "只要任何高优先级冲突，该候选即为false。\n"
    "3. 强制失败规则：若出现核心冲突，或符合四项严格失败条件之一，直接判false。\n"
    "验证流程（对每个候选独立执行）：\n"
    "Step 1 — 输入清洗：在内心去除人名、联系方式、测试标记、无意义数字。\n"
    "Step 2 — 括号评估：判断核心机构名在括号内还是括号外。\n"
    "Step 3 — 短路匹配验证：\n"
    "- 优先级1：精确匹配（完全一致直接判true）\n"
    "- 优先级2：地理信息层级（街道>区>市>省，存在层级冲突则判false）\n"
    "- 优先级3：大学/研究机构（上下级关系必须明确，不可错配）\n"
    "- 优先级4：最小粒度匹配（院区、分院不可与总院混淆）\n"
    "- 优先级5：精确修饰词（数字、分院、子类型、区域后缀冲突则判false）\n"
    "- 优先级6：辅助信息综合\n"
    "输出要求：\n"
    "严格输出标准JSON数组，数组长度必须与候选列表一致。不要输出任何思考过程或其他字符。\n"
    "格式：\n"
    "[\n"
    '  {"index": 1, "reasoning": "P1(通过)->P2(冲突:输入A区,候选B区)->判定false", '
    '"matched": false, "confidence": "Low"},\n'
    '  {"index": 2, "reasoning": "P1(通过)->P2(通过)->P3(通过)->全通过", '
    '"matched": true, "confidence": "High"}\n'
    "]"
)

PROD_SYSTEM_PROMPT = (
    "角色：专业的产品数据匹配专家\n"
    "任务：逐一判定【输入产品】与列表中的每个【候选产品】是否为同一核心产品，"
    "并评估匹配等级。\n"
    "严格规则：\n"
    "1. 必须对每个候选产品【独立】进行验证，候选之间绝不能互相干扰。\n"
    "2. 核心名称拥有绝对一票否决权。\n"
    "3. 不需要计算具体分数，只需根据差异情况判定匹配等级（A/B/D）。\n"
    "4. 必须先提取核心名，再比对修饰词，最后比对规格。\n"
    "判定流程（对每个候选独立执行）：\n"
    "Step 1 — 一票否决（核心名称一致性）：\n"
    "去除修饰词，提取核心产品名。若核心名不一致，直接判定为 D级。\n"
    "Step 2 — 差异提取与定级（仅在核心名一致时执行）：\n"
    "比对修饰词（材质、方法、品牌、型号）和规格（尺寸、包装数、容量），判定匹配等级：\n"
    "- A级：核心名一致，且修饰词完全一致，规格完全一致。\n"
    "- B级：核心名一致，但修饰词或规格存在差异（如剂型不同、剂量不同、数量不同等）。\n"
    "- D级：核心名不一致。\n"
    "输出要求：\n"
    "严格输出标准JSON数组，数组长度必须与候选列表一致。不要输出任何思考过程或其他字符。\n"
    "格式：\n"
    "[\n"
    '  {"index": 1, "core_name_match": false, "modifier_diff": "无", '
    '"spec_diff": "无", "match_grade": "D"},\n'
    '  {"index": 2, "core_name_match": true, "modifier_diff": "剂型差异", '
    '"spec_diff": "0.25g*24片/盒vs0.5g*20粒/盒", "match_grade": "B"}\n'
    "]"
)

_INST_HINTS = ("机构", "inst", "医院", "hospital", "药房", "药店", "pharmacy", "门店")
_PROD_HINTS = ("产品", "prod", "药品", "drug")


def _detect_task(value: str | None) -> str:
    """类型列取值 → 任务名。识别不了默认机构（基线显示机构匹配是唯一有提升空间的任务）。"""
    v = (value or "").lower()
    if any(h in v for h in _PROD_HINTS):
        return "product"
    if any(h in v for h in _INST_HINTS):
        return "institution"
    return "institution"


def _common_prefix_len(a: str, b: str) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


class MasterDataTemplate(DomainTemplate):
    """主数据匹配：机构 + 产品双任务模板，导出 messages chat 格式。"""

    name = "master_data"

    def describe(self) -> str:
        return (
            "主数据匹配模板（机构 + 产品双任务，messages chat 格式）：\n"
            "同一模型学两个任务，任务由 system prompt 区分——机构任务逐候选输出 "
            "matched/confidence 判定，产品任务按核心名一致性输出 A/B/D 等级。\n"
            "类型列取值含「机构/institution/hospital/pharmacy」→ 机构任务；"
            "含「产品/product/药品/drug」→ 产品任务；未映射类型列时默认机构任务。\n"
            "规格列可选：产品任务的查询与候选会带上规格。\n"
            "导出 messages 格式，可直接用于 domains/master_data/scripts/train.py。\n"
            "需要列：标准名（必填）；查询/别名、编码、变体、类型、规格（可选）。"
        )

    def build_samples(
        self, table: RawTable, mapping: FieldMapping, spec: WizardSpec
    ) -> BuildResult:
        errors = mapping.validate(table.columns)
        if errors:
            raise WizardError("字段映射校验失败:\n" + "\n".join(f"  - {e}" for e in errors))

        # 全量标准条目池（按任务分组，负例绝不跨任务）
        pools: dict[str, list[tuple[str, str | None, str | None]]] = {
            "institution": [],
            "product": [],
        }
        seen: set[str] = set()
        for row in table.rows:
            std = row.get(mapping.standard_name)
            if std is None or std in seen:
                continue
            seen.add(std)
            task = _detect_task(row.get(mapping.entity_type) if mapping.entity_type else None)
            pools[task].append(
                (
                    std,
                    row.get(mapping.code) if mapping.code else None,
                    row.get(mapping.spec) if mapping.spec else None,
                )
            )

        result = BuildResult()
        for idx, row in enumerate(table.rows, start=1):
            std = row.get(mapping.standard_name)
            if std is None:
                result.dropped.append(RowIssue(idx, "标准名为空"))
                continue
            task = _detect_task(row.get(mapping.entity_type) if mapping.entity_type else None)

            queries: list[str] = []
            if mapping.query:
                q = row.get(mapping.query)
                if q is not None:
                    queries.append(q)  # queries 此时必为空，无需查重
            if mapping.variants:
                for v in split_variants(row.get(mapping.variants)):
                    if v not in queries:
                        queries.append(v)
            if not queries:
                queries.append(std)  # 无任何查询值时退回标准名，保证行可用
            queries = queries[:64]  # 单行变体爆炸保护

            code = row.get(mapping.code) if mapping.code else None
            row_spec = row.get(mapping.spec) if mapping.spec else None
            for query in queries:
                result.samples.append(
                    self._make_sample(query, std, code, row_spec, task, pools[task], spec, idx)
                )
        return result

    def _make_sample(
        self,
        query: str,
        standard: str,
        code: str | None,
        row_spec: str | None,
        task: str,
        pool: list[tuple[str, str | None, str | None]],
        spec: WizardSpec,
        row: int,
    ) -> MatchingSample:
        # 产品任务把规格并入查询文本（与 generate_data.py 的 query_name + query_spec 一致）
        full_query = f"{query} {row_spec}" if task == "product" and row_spec else query
        # 跨进程稳定（内置 hash 带随机盐，会破坏同种子同产出的可复现承诺）
        rng = random.Random(spec.seed * 1_000_003 + row * 131 + zlib.crc32(full_query.encode()))
        negatives = self._pick_negatives(standard, pool, spec.n_candidates - 1, rng)
        candidates = [Candidate(standard, code, True, spec=row_spec)] + [
            Candidate(name, neg_code, False, spec=neg_spec)
            for name, neg_code, neg_spec in negatives
        ]
        rng.shuffle(candidates)
        return MatchingSample(
            query=full_query,
            standard_name=standard,
            code=code,
            entity_type=task,
            difficulty=classify_difficulty(query, standard),
            candidates=candidates,
            source_row=row,
        )

    @staticmethod
    def _pick_negatives(
        standard: str,
        pool: list[tuple[str, str | None, str | None]],
        n: int,
        rng: random.Random,
    ) -> list[tuple[str, str | None, str | None]]:
        """前缀硬负例优先（同核心名最易混淆），其余随机补齐；不跨任务池。"""
        prefix = standard[:2]
        hard = [t for t in pool if t[0] != standard and prefix and t[0].startswith(prefix)]
        rest = [t for t in pool if t[0] != standard and t not in hard]
        rng.shuffle(hard)
        rng.shuffle(rest)
        picked = hard[:n]
        picked += rest[: max(0, n - len(picked))]
        return picked

    def format_record(self, sample: MatchingSample) -> dict[str, object]:
        if not any(c.label for c in sample.candidates):
            raise WizardError(f"样本（行 {sample.source_row}）没有标注正确候选，无法格式化。")
        if sample.entity_type == "product":
            return self._format_product(sample)
        return self._format_institution(sample)

    def _format_institution(self, sample: MatchingSample) -> dict[str, object]:
        lines = [
            f"[{i + 1}] 编码: {c.code}, 名称: {c.name}" if c.code else f"[{i + 1}] 名称: {c.name}"
            for i, c in enumerate(sample.candidates)
        ]
        user_content = (
            f"【输入机构】：{sample.query}\n【候选机构列表】：\n"
            + "\n".join(lines)
            + "\n请逐个独立验证并输出JSON数组："
        )
        items: list[dict[str, object]] = []
        for i, c in enumerate(sample.candidates):
            if c.label:
                if sample.difficulty == "easy":
                    reasoning, confidence = "P1(精确匹配)->判定true", "High"
                elif sample.difficulty == "medium":
                    reasoning, confidence = "P1(非精确)->P2(通过,简写/全称差异)->判定true", "High"
                else:
                    reasoning, confidence = "P1(非精确)->P2(通过,简写/全称差异)->判定true", "Medium"
                items.append(
                    {
                        "index": i + 1,
                        "reasoning": reasoning,
                        "matched": True,
                        "confidence": confidence,
                    }
                )
            else:
                items.append(
                    {
                        "index": i + 1,
                        "reasoning": "P1(非精确)->核心名冲突(不同实体)->判定false",
                        "matched": False,
                        "confidence": "Low",
                    }
                )
        return {
            "messages": [
                {"role": "system", "content": INST_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": json.dumps(items, ensure_ascii=False, indent=2)},
            ]
        }

    def _format_product(self, sample: MatchingSample) -> dict[str, object]:
        lines = [
            f"[{i + 1}] 编码: {c.code}, 名称: {c.name}, 规格: {c.spec}"
            if c.code and c.spec
            else f"[{i + 1}] 编码: {c.code}, 名称: {c.name}"
            if c.code
            else f"[{i + 1}] 名称: {c.name}"
            for i, c in enumerate(sample.candidates)
        ]
        user_content = (
            f"【输入产品】：{sample.query}\n【候选产品列表】：\n"
            + "\n".join(lines)
            + "\n请逐个独立验证并输出JSON数组："
        )
        pos_spec = next(c.spec or "" for c in sample.candidates if c.label)
        items: list[dict[str, object]] = []
        for i, c in enumerate(sample.candidates):
            if c.label:
                items.append(
                    {
                        "index": i + 1,
                        "core_name_match": True,
                        "modifier_diff": "无",
                        "spec_diff": "无",
                        "match_grade": "A",
                    }
                )
            elif _common_prefix_len(c.name, sample.standard_name) >= 2:
                # 同核心名的剂型/规格近亲 → B 级硬负例（核心名一致但非同一产品）
                items.append(
                    {
                        "index": i + 1,
                        "core_name_match": True,
                        "modifier_diff": "剂型差异" if c.name != sample.standard_name else "无",
                        "spec_diff": "不同" if (c.spec or "") != pos_spec else "无",
                        "match_grade": "B",
                    }
                )
            else:
                items.append(
                    {
                        "index": i + 1,
                        "core_name_match": False,
                        "modifier_diff": "无",
                        "spec_diff": "无",
                        "match_grade": "D",
                    }
                )
        return {
            "messages": [
                {"role": "system", "content": PROD_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": json.dumps(items, ensure_ascii=False, indent=2)},
            ]
        }


register_template(MasterDataTemplate())
