"""切分与导出：按实体分组切分（防泄漏）→ Alpaca JSON 落盘。

切分粒度是「标准实体」而不是「样本行」——同一实体的所有别名/变体样本
必须落在同一个 split，否则评测时模型是在背答案（prepare_data.py 的教训）。
"""

import json
import random
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from src.data.wizard.spec import SPLITS, WizardError, WizardSpec
from src.data.wizard.templates import DomainTemplate, MatchingSample


@dataclass(frozen=True)
class SplitResult:
    train: list[MatchingSample] = field(default_factory=list)
    val: list[MatchingSample] = field(default_factory=list)
    test: list[MatchingSample] = field(default_factory=list)

    def as_dict(self) -> dict[str, list[MatchingSample]]:
        return {"train": self.train, "val": self.val, "test": self.test}


def dedup_samples(samples: list[MatchingSample]) -> tuple[list[MatchingSample], int]:
    """按 (query, standard_name, entity_type) 去重（保留首条），返回 (新列表, 去除数)。"""
    seen: set[tuple[str, str, str]] = set()
    kept: list[MatchingSample] = []
    for s in samples:
        key = (s.query, s.standard_name, s.entity_type)
        if key in seen:
            continue
        seen.add(key)
        kept.append(s)
    return kept, len(samples) - len(kept)


def split_by_entity(samples: list[MatchingSample], spec: WizardSpec) -> SplitResult:
    """按标准实体分组切分：打乱组序后按比例把整组划入各 split。"""
    groups: dict[str, list[MatchingSample]] = defaultdict(list)
    for s in samples:
        groups[s.code or s.standard_name].append(s)

    if len(groups) < 3:
        raise WizardError(
            f"不同标准实体只有 {len(groups)} 个，无法切成 train/val/test 三份且互不泄漏。"
            "至少需要 3 个不同标准实体；建议 30+ 以保证每个 split 有代表性。"
        )

    rng = random.Random(spec.seed)
    keys = sorted(groups)  # 先排序再打乱，保证与 dict 插入顺序无关的可复现性
    rng.shuffle(keys)

    n = len(keys)
    train_keys = set(keys[: int(n * spec.split_ratios[0])])
    val_keys = set(
        keys[int(n * spec.split_ratios[0]) : int(n * (spec.split_ratios[0] + spec.split_ratios[1]))]
    )
    # 其余实体全部归 test（补齐比例舍入余数）

    result = SplitResult()
    for key in keys:
        bucket = (
            result.train if key in train_keys else result.val if key in val_keys else result.test
        )
        bucket.extend(groups[key])
    for bucket in (result.train, result.val, result.test):
        rng.shuffle(bucket)
    return result


@dataclass(frozen=True)
class ExportReport:
    out_dir: Path
    files: dict[str, Path]
    counts: dict[str, int]
    difficulty: dict[str, dict[str, int]]

    def to_dict(self) -> dict[str, object]:
        return {
            "out_dir": str(self.out_dir),
            "files": {k: str(v) for k, v in self.files.items()},
            "counts": self.counts,
            "difficulty": self.difficulty,
        }


def _difficulty_counts(samples: list[MatchingSample]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for s in samples:
        counts[s.difficulty] = counts.get(s.difficulty, 0) + 1
    return counts


def export_splits(
    splits: SplitResult, template: DomainTemplate, out_dir: str | Path
) -> ExportReport:
    """把三个 split 落成 train.json / val.json / test.json（Alpaca 指令格式）。"""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    files: dict[str, Path] = {}
    counts: dict[str, int] = {}
    difficulty: dict[str, dict[str, int]] = {}
    mapping = splits.as_dict()
    for split in SPLITS:
        records = [template.format_record(s) for s in mapping[split]]
        path = out / f"{split}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
        files[split] = path
        counts[split] = len(records)
        difficulty[split] = _difficulty_counts(mapping[split])
    return ExportReport(out_dir=out, files=files, counts=counts, difficulty=difficulty)
