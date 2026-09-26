"""数据向导切分/导出层测试：按实体切分防泄漏、Alpaca 落盘。"""

import json

import pytest

from src.data.wizard.exporter import dedup_samples, export_splits, split_by_entity
from src.data.wizard.spec import FieldMapping, WizardError, WizardSpec
from src.data.wizard.templates import Candidate, EntityMatchingTemplate, MatchingSample


def make_samples(
    n_entities: int, per_entity: int = 3, code_prefix: str = "C"
) -> list[MatchingSample]:
    samples = []
    for i in range(n_entities):
        for j in range(per_entity):
            samples.append(
                MatchingSample(
                    query=f"q{i}_{j}",
                    standard_name=f"S{i}",
                    code=f"{code_prefix}{i}",
                    entity_type="entity",
                    difficulty=["easy", "medium", "hard"][j % 3],
                    candidates=[
                        Candidate(f"S{i}", f"{code_prefix}{i}", True),
                        Candidate(f"S{(i + 1) % n_entities}", None, False),
                    ],
                    source_row=i * per_entity + j + 1,
                )
            )
    return samples


def spec(**overrides) -> WizardSpec:
    return WizardSpec(mapping=FieldMapping(standard_name="标准名"), **overrides)


class TestDedup:
    def test_no_duplicates(self) -> None:
        samples = make_samples(3, per_entity=1)
        kept, removed = dedup_samples(samples)
        assert len(kept) == 3
        assert removed == 0

    def test_removes_duplicates_keeps_first(self) -> None:
        samples = make_samples(1, per_entity=1) * 3
        kept, removed = dedup_samples(samples)
        assert len(kept) == 1
        assert removed == 2


class TestSplitByEntity:
    def test_entity_groups_never_cross_splits(self) -> None:
        samples = make_samples(20, per_entity=3)
        result = split_by_entity(samples, spec())
        # 同一实体（code）的所有样本必须在同一 split
        for bucket in (result.train, result.val, result.test):
            codes = {s.code for s in bucket}
            for other in (result.train, result.val, result.test):
                if other is bucket:
                    continue
                assert not (codes & {s.code for s in other})

    def test_ratios_respected(self) -> None:
        samples = make_samples(100, per_entity=1)
        result = split_by_entity(samples, spec())
        assert len(result.train) == 80
        assert len(result.val) == 10
        assert len(result.test) == 10

    def test_deterministic(self) -> None:
        samples = make_samples(30, per_entity=1)
        a = split_by_entity(samples, spec(seed=7))
        b = split_by_entity(samples, spec(seed=7))
        assert [s.query for s in a.train] == [s.query for s in b.train]
        c = split_by_entity(samples, spec(seed=8))
        assert [s.query for s in a.train] != [s.query for s in c.train]

    def test_splits_by_name_when_no_code(self) -> None:
        samples = make_samples(10, per_entity=2)
        for s in samples:
            s.code = None
        result = split_by_entity(samples, spec())
        assert sum(len(b) for b in (result.train, result.val, result.test)) == 20
        # 同名实体不跨 split
        train_names = {s.standard_name for s in result.train}
        test_names = {s.standard_name for s in result.test}
        assert not train_names & test_names

    def test_too_few_entities(self) -> None:
        with pytest.raises(WizardError, match="无法切成"):
            split_by_entity(make_samples(2, per_entity=2), spec())

    def test_custom_ratios(self) -> None:
        samples = make_samples(50, per_entity=1)
        result = split_by_entity(samples, spec(split_ratios=(0.6, 0.2, 0.2)))
        assert (len(result.train), len(result.val), len(result.test)) == (30, 10, 10)


class TestExportSplits:
    def test_writes_alpaca_files(self, tmp_path) -> None:
        samples = make_samples(10, per_entity=2)
        result = split_by_entity(samples, spec())
        report = export_splits(result, EntityMatchingTemplate(), tmp_path / "out")
        assert set(report.files) == {"train", "val", "test"}
        for split, path in report.files.items():
            with open(path, encoding="utf-8") as f:
                records = json.load(f)
            assert len(records) == report.counts[split]
            for rec in records:
                assert set(rec) == {"instruction", "input", "output", "metadata"}
                json.loads(rec["output"])  # output 是合法 JSON
        # 10 实体按 0.8/0.1/0.1 → 8/1/1 组 × 每组 2 样本
        assert report.counts == {"train": 16, "val": 2, "test": 2}
        # 难度统计按 split 报告
        assert report.difficulty["train"]
        assert report.to_dict()["files"]["train"].endswith("train.json")
