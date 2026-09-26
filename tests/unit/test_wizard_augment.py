"""噪音增强测试：字符扰动算子 / 副本语义 / 流水线集成。"""

import random

from src.data.wizard.augment import _corrupt_once, augment_samples, corrupt_query
from src.data.wizard.importers import RawTable
from src.data.wizard.pipeline import WizardPipeline
from src.data.wizard.spec import FieldMapping, WizardSpec
from src.data.wizard.templates import Candidate, MatchingSample, classify_difficulty, edit_distance

QUERY = "阿莫西林胶囊"


def make_sample(
    query: str = "泰诺林",
    standard: str = "对乙酰氨基酚片",
    code: str | None = "Z5",
    row: int = 1,
) -> MatchingSample:
    return MatchingSample(
        query=query,
        standard_name=standard,
        code=code,
        entity_type="drug",
        difficulty=classify_difficulty(query, standard),
        candidates=[
            Candidate(standard, code, True),
            Candidate("布洛芬缓释胶囊", "Z6", False),
            Candidate("阿莫西林颗粒", "Z7", False),
        ],
        source_row=row,
    )


class TestCorruptOnce:
    def test_variants_stay_within_one_edit(self) -> None:
        # swap=编辑距离2，delete/duplicate=编辑距离1；长度至多变化 1
        for seed in range(40):
            mutated = _corrupt_once(QUERY, random.Random(seed))
            assert mutated is not None
            assert mutated != QUERY
            assert 1 <= edit_distance(mutated, QUERY) <= 2
            assert len(mutated) in {len(QUERY) - 1, len(QUERY), len(QUERY) + 1}

    def test_short_or_empty_returns_none(self) -> None:
        assert _corrupt_once("仁", random.Random(0)) is None
        assert _corrupt_once("", random.Random(0)) is None

    def test_repeated_chars_still_corruptible(self) -> None:
        # 交换相邻同字是无效扰动会被重试；漏字/重字仍可产出变体
        for seed in range(20):
            assert _corrupt_once("啊啊啊", random.Random(seed)) is not None

    def test_same_seed_same_result(self) -> None:
        a = _corrupt_once(QUERY, random.Random(7))
        b = _corrupt_once(QUERY, random.Random(7))
        assert a == b


class TestCorruptQuery:
    def test_spec_suffix_preserved(self) -> None:
        # 产品查询「名称 规格」：只扰动名称部分，规格段原样保留
        for seed in range(30):
            q = corrupt_query("阿莫仙 0.25g*24片/盒", random.Random(seed))
            assert q is None or q.endswith(" 0.25g*24片/盒")

    def test_plain_query_never_gains_space(self) -> None:
        q = corrupt_query(QUERY, random.Random(1))
        assert q is not None and " " not in q


class TestAugmentSamples:
    def test_doubles_with_fixed_labels(self) -> None:
        samples = [
            make_sample(query=f"别名{i}", standard=f"药{i}", code=f"Z{i}", row=i + 1)
            for i in range(3)
        ]
        spec = WizardSpec(mapping=FieldMapping(standard_name="标准名"))
        out, n = augment_samples(samples, spec)
        assert n == 3
        assert len(out) == 6
        for orig, copy in zip(samples, out[3:]):
            assert copy.query != orig.query
            assert copy.standard_name == orig.standard_name
            assert copy.code == orig.code
            assert copy.source_row == orig.source_row
            assert copy.candidates == orig.candidates
            assert copy.difficulty == classify_difficulty(copy.query, copy.standard_name)

    def test_deterministic(self) -> None:
        samples = [make_sample(query="泰诺林", row=1)]
        spec = WizardSpec(mapping=FieldMapping(standard_name="标准名"), seed=42)
        out1, _ = augment_samples(samples, spec)
        out2, _ = augment_samples(samples, spec)
        assert [s.query for s in out1] == [s.query for s in out2]

    def test_uncorruptible_query_skipped(self) -> None:
        # 单字符查询扰动不出变体 → 不追加副本，原样本保留
        samples = [make_sample(query="仁", standard="仁和堂", row=1)]
        out, n = augment_samples(samples, WizardSpec(mapping=FieldMapping(standard_name="标准名")))
        assert n == 0
        assert out == samples


# 查询彼此拉开距离：若用 别名0/别名1/别名11 这类近似串，扰动可能撞出另一实体的
# 查询（别名11 漏字 → 别名1），ambiguous_query 会如实阻断——那是安全网，不是 bug
DRUG_QUERIES = [
    "泰诺林",
    "芬必得",
    "严迪片",
    "世福素",
    "希刻劳",
    "可乐必妥",
    "阿莫仙",
    "感冒灵",
    "布洛芬",
    "罗红素",
    "头孢克",
    "氨咖黄",
]
DRUG_ROWS = [{"标准名": f"标准{i}", "别名": q, "编码": f"Z{i}"} for i, q in enumerate(DRUG_QUERIES)]
DRUG_COLUMNS = ["标准名", "别名", "编码"]


def make_table(rows: list[dict], columns: list[str]) -> RawTable:
    return RawTable(
        source="test", columns=columns, rows=[{c: r.get(c) for c in columns} for r in rows]
    )


class TestPipelineNoise:
    def test_noise_on_doubles_export(self, tmp_path) -> None:
        spec = WizardSpec(
            mapping=FieldMapping(standard_name="标准名", query="别名", code="编码"),
            noise_augment=True,
        )
        report = WizardPipeline(spec).run(make_table(DRUG_ROWS, DRUG_COLUMNS), tmp_path)
        assert report.augmented == 12
        assert report.built_samples == 24
        assert report.passed  # 泄漏/歧义/候选检查全过：副本与原样本同实体同组
        total = sum(report.split_counts.values())
        assert total == 24

    def test_off_by_default(self, tmp_path) -> None:
        spec = WizardSpec(mapping=FieldMapping(standard_name="标准名", query="别名", code="编码"))
        report = WizardPipeline(spec).run(make_table(DRUG_ROWS, DRUG_COLUMNS), tmp_path)
        assert report.augmented == 0
        assert report.built_samples == 12
        assert all("噪音增强" not in line for line in report.summary_lines())

    def test_summary_and_dict_expose_augmented(self, tmp_path) -> None:
        spec = WizardSpec(
            mapping=FieldMapping(standard_name="标准名", query="别名", code="编码"),
            noise_augment=True,
        )
        report = WizardPipeline(spec).run(make_table(DRUG_ROWS, DRUG_COLUMNS), tmp_path)
        assert any("噪音增强 +12" in line for line in report.summary_lines())
        assert report.to_dict()["augmented"] == 12

    def test_deterministic_export(self, tmp_path) -> None:
        spec = WizardSpec(
            mapping=FieldMapping(standard_name="标准名", query="别名", code="编码"),
            noise_augment=True,
        )
        r1 = WizardPipeline(spec).run(make_table(DRUG_ROWS, DRUG_COLUMNS), tmp_path / "a")
        r2 = WizardPipeline(spec).run(make_table(DRUG_ROWS, DRUG_COLUMNS), tmp_path / "b")
        assert r1.augmented == r2.augmented == 12
        assert (tmp_path / "a" / "train.json").read_bytes() == (
            tmp_path / "b" / "train.json"
        ).read_bytes()
