"""数据体检：训练集生成后的质量门禁。

每项检查返回 CheckResult，severity 约定：
- error   阻断导出——不修就不能训（如 train/test 泄漏）；
- warning 不阻断，但会把风险讲清楚（如答案位置集中）；
- info    仅报告分布，供决策参考。

消息写给「会写代码但不懂 ML 的工程师」：先说结论，再说为什么影响训练，最后给修法。
"""

from dataclasses import dataclass, field
from typing import Literal

from src.data.wizard.spec import SPLITS
from src.data.wizard.templates import MatchingSample, RowIssue

Severity = Literal["error", "warning", "info"]

# 答案位置集中阈值：均匀分布下 8 候选期望占比 12.5%，超过 40% 视为学位置捷径
POSITION_DOMINANCE = 0.4
# 位置偏差检查最少样本数（太少时占比无意义）
POSITION_MIN_SAMPLES = 10


@dataclass(frozen=True)
class CheckResult:
    check_id: str
    severity: Severity
    passed: bool
    message: str
    details: dict[str, object] = field(default_factory=dict)


def check_dropped_rows(dropped: list[RowIssue]) -> CheckResult:
    """原始行缺失（标准名为空等）→ warning（这些行已被跳过，用户应知情并确认）。"""
    if not dropped:
        return CheckResult("dropped_rows", "warning", True, "原始数据行全部可用。")
    examples = "、".join(f"第 {d.row} 行（{d.reason}）" for d in dropped[:3])
    more = f" 等 {len(dropped)} 行" if len(dropped) > 3 else ""
    return CheckResult(
        "dropped_rows",
        "warning",
        False,
        f"有 {len(dropped)} 行缺少必需字段被跳过，例如 {examples}{more}。"
        "这些行不会进入训练集；请补全标准名或从表中删除这些行后重跑。",
        details={"count": len(dropped), "rows": [d.row for d in dropped[:20]]},
    )


def check_duplicates(
    splits: dict[str, list[MatchingSample]],
) -> CheckResult:
    """同一 split 内完全重复的 (query, standard) → warning（浪费训练步，且让样本数虚高）。"""
    dup_total = 0
    dup_examples: list[str] = []
    for split in SPLITS:
        seen: set[tuple[str, str]] = set()
        for s in splits.get(split, []):
            key = (s.query, s.standard_name)
            if key in seen:
                dup_total += 1
                if len(dup_examples) < 3:
                    dup_examples.append(f"[{split}] {s.query} → {s.standard_name}")
            seen.add(key)
    if dup_total == 0:
        return CheckResult("duplicates", "warning", True, "无重复样本。")
    return CheckResult(
        "duplicates",
        "warning",
        False,
        f"存在 {dup_total} 条完全重复的样本（相同查询和标准名），例如: "
        + "；".join(dup_examples)
        + "。重复样本会让模型在这些条目上过拟合、样本量统计虚高；"
        "开启去重（dedup=True，默认开启）可自动去除。",
        details={"count": dup_total, "examples": dup_examples},
    )


def check_split_leakage(
    splits: dict[str, list[MatchingSample]],
) -> CheckResult:
    """train/val/test 之间标准实体重叠 → error（评测分数虚高，最严重的数据问题）。"""

    def entity_keys(samples: list[MatchingSample]) -> set[str]:
        # 优先用编码（更精确），无编码列时退回标准名
        keys = {s.code for s in samples if s.code}
        keys |= {s.standard_name for s in samples}
        return keys

    overlaps: list[tuple[str, str, list[str]]] = []
    for i, a in enumerate(SPLITS):
        for b in SPLITS[i + 1 :]:
            shared = entity_keys(splits.get(a, [])) & entity_keys(splits.get(b, []))
            if shared:
                overlaps.append((a, b, sorted(shared)[:3]))
    if not overlaps:
        return CheckResult("split_leakage", "error", True, "train/val/test 实体无重叠。")
    described = "；".join(f"{a}/{b}: {ex}" for a, b, ex in overlaps)
    return CheckResult(
        "split_leakage",
        "error",
        False,
        f"数据泄漏——同一实体同时出现在不同 split（{described}）。"
        "模型在训练时见过评测实体的答案，分数会虚高且无法反映真实匹配能力。"
        "请按标准实体整体切分（本向导默认按实体切分，出现此错误通常意味着外部注入了数据）。",
        details={"pairs": [(a, b) for a, b, _ in overlaps]},
    )


def check_position_bias(
    splits: dict[str, list[MatchingSample]],
) -> CheckResult:
    """正确答案在候选列表中的位置分布 → warning（位置集中 = 模型学捷径）。"""
    worst_share = 0.0
    worst: tuple[str, int, float] | None = None
    for split in SPLITS:
        samples = splits.get(split, [])
        if len(samples) < POSITION_MIN_SAMPLES:
            continue
        counts: dict[int, int] = {}
        for s in samples:
            idx = next((i for i, c in enumerate(s.candidates) if c.label), None)
            if idx is None:
                continue
            counts[idx] = counts.get(idx, 0) + 1
        if not counts:
            continue
        top_idx = max(counts, key=lambda k: counts[k])
        share = counts[top_idx] / sum(counts.values())
        if share > worst_share:
            worst_share = share
            worst = (split, top_idx + 1, share)
    if worst is None or worst_share <= POSITION_DOMINANCE:
        return CheckResult("position_bias", "warning", True, "答案位置分布正常，无位置捷径风险。")
    split, pos, share = worst
    return CheckResult(
        "position_bias",
        "warning",
        False,
        f"位置偏差——{split} 集中 {share:.0%} 的样本正确答案都在第 {pos} 个候选。"
        f"模型会学会'总是选第 {pos} 个'而不是真正判断相似度，换一批数据就失效。"
        "修法：生成候选列表后随机打乱顺序（本向导默认已打乱；此警告通常来自外部注入的数据）。",
        details={"split": split, "position": pos, "share": round(share, 3)},
    )


def check_difficulty_balance(
    splits: dict[str, list[MatchingSample]],
) -> CheckResult:
    """难度分布报告 → info（全部同一难度时提醒分层评测缺维度）。"""
    all_samples = [s for split in SPLITS for s in splits.get(split, [])]
    if not all_samples:
        return CheckResult("difficulty_balance", "info", True, "无样本可统计。")
    counts: dict[str, int] = {}
    for s in all_samples:
        counts[s.difficulty] = counts.get(s.difficulty, 0) + 1
    total = sum(counts.values())
    summary = ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))
    missing = [d for d in ("easy", "medium", "hard") if d not in counts]
    if missing:
        return CheckResult(
            "difficulty_balance",
            "info",
            True,
            f"难度分布: {summary}（共 {total} 条）。缺少 {('、'.join(missing))} 难度样本——"
            "不影响训练，但评测无法分层报告该难度上的表现；"
            "若原始数据里存在更模糊的别名，建议补充进来。",
            details={"counts": counts, "total": total},
        )
    return CheckResult(
        "difficulty_balance",
        "info",
        True,
        f"难度分布: {summary}（共 {total} 条），三个难度层都有样本，可分层评测。",
        details={"counts": counts, "total": total},
    )


def check_candidate_counts(
    splits: dict[str, list[MatchingSample]],
) -> CheckResult:
    """候选数过少 → error（单候选无法构成判别任务）。"""
    bad: list[str] = []
    for split in SPLITS:
        for s in splits.get(split, []):
            if len(s.candidates) < 2:
                bad.append(f"[{split}] 行 {s.source_row}（{len(s.candidates)} 个候选）")
    if not bad:
        return CheckResult("candidate_counts", "error", True, "所有样本候选数 ≥ 2。")
    return CheckResult(
        "candidate_counts",
        "error",
        False,
        f"有 {len(bad)} 个样本候选数不足 2，例如: {'；'.join(bad[:3])}。"
        "只有一个候选时模型不需要判断'哪个更匹配'，学不到判别能力。"
        "修法：知识库条目太少时无法构造干扰项——至少需要 n_candidates 个不同标准实体。",
        details={"count": len(bad), "examples": bad[:10]},
    )


def run_checks(
    splits: dict[str, list[MatchingSample]], dropped: list[RowIssue]
) -> list[CheckResult]:
    """跑全部体检项，顺序即报告展示顺序。"""
    return [
        check_dropped_rows(dropped),
        check_split_leakage(splits),
        check_position_bias(splits),
        check_candidate_counts(splits),
        check_duplicates(splits),
        check_difficulty_balance(splits),
    ]
