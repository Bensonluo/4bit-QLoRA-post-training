"""噪音增强：给查询注入错别字/漏字，标准答案不变。

字符级输入扰动 + 固定标签是数据增强的成熟基线（跨领域 NER 增强、
扰动一致性学习都用这一类做法）：对查询做轻量扰动（相邻交换/漏字/重字），
候选列表与标准答案保持不变，模型由此学到「错别字不改变匹配判断」。

设计要点：
- 只扰动查询文本；含规格后缀（首个空格之后）的产品查询只扰动名称部分，
  规格数字保持原样，避免把「规格不同→B级」的标签语义搅浑。
- 噪音副本与原样本同实体同行号，实体组切分时天然落入同一 split；
  val/test 各自获得扰动视图，正好满足「扰动集上测鲁棒性」的评测惯例。
- 随机种子由 spec.seed + 查询 + 行号派生，同配置重跑产出一致
  （与 templates 的种子约定同风格，跨进程稳定）。
"""

import random
import zlib
from dataclasses import replace

from src.data.wizard.spec import WizardSpec
from src.data.wizard.templates import MatchingSample, classify_difficulty

# 可用扰动算子：swap（相邻交换）/ delete（漏字）/ duplicate（重字）
# delete 与 duplicate 对长度 ≥2 的文本必然有效，swap 只在存在相邻不同字符时可用
_OPS_BASE = ("delete", "duplicate")


def _corrupt_once(text: str, rng: random.Random) -> str | None:
    """对 text 施加一次字符扰动；长度不足 2 时返回 None。"""
    if len(text) < 2:
        return None
    pairs = [j for j in range(len(text) - 1) if text[j] != text[j + 1]]
    op = rng.choice((*_OPS_BASE, "swap") if pairs else _OPS_BASE)
    i = rng.randrange(len(text))
    if op == "swap":
        # 只交换相邻且不同的字符，保证扰动后文本一定变化
        j = rng.choice(pairs)
        return text[:j] + text[j + 1] + text[j] + text[j + 2 :]
    if op == "delete":
        return text[:i] + text[i + 1 :]
    return text[:i] + text[i] * 2 + text[i + 1 :]


def corrupt_query(query: str, rng: random.Random) -> str | None:
    """扰动查询文本；首个空格后的部分（规格等修饰段）原样保留。"""
    name, sep, rest = query.partition(" ")
    corrupted = _corrupt_once(name, rng)
    if corrupted is None:
        return None
    return corrupted + sep + rest


def augment_samples(
    samples: list[MatchingSample], spec: WizardSpec
) -> tuple[list[MatchingSample], int]:
    """为每个样本追加一条带错别字的查询副本（标签与候选不变，难度重估）。

    返回 (原样本 + 噪音副本, 副条数)。副本沿用原样本的标准名/编码/候选列表，
    难度按扰动后的查询重新计算——原本 easy 的精确匹配变成带错字的 medium/hard，
    难度分布检查会如实反映。查询太短（<2 字符）扰动不出有效变体的样本跳过。
    """
    augmented: list[MatchingSample] = []
    for s in samples:
        rng = random.Random(
            spec.seed * 1_000_003 + zlib.crc32(s.query.encode()) + s.source_row * 131
        )
        noisy = corrupt_query(s.query, rng)
        if noisy is None:
            continue
        augmented.append(
            replace(
                s,
                query=noisy,
                difficulty=classify_difficulty(noisy, s.standard_name),
            )
        )
    return samples + augmented, len(augmented)
