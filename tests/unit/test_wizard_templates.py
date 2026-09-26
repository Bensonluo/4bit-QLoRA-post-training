"""数据向导模板层测试：样本生成 / 难度分层 / 负例 / 格式化 / 注册表。"""

import json

import pytest

from src.data.wizard.importers import RawTable
from src.data.wizard.spec import FieldMapping, WizardError, WizardSpec
from src.data.wizard.templates import (
    BuildResult,
    Candidate,
    EntityMatchingTemplate,
    MatchingSample,
    available_templates,
    classify_difficulty,
    edit_distance,
    get_template,
    register_template,
)


def make_table(columns: list[str], rows: list[dict]) -> RawTable:
    return RawTable(
        source="test",
        columns=columns,
        rows=[{c: r.get(c) for c in columns} for r in rows],
    )


DRUGS = [
    {"标准名": "阿莫西林胶囊", "别名": "阿莫西林", "编码": "Z1"},
    {"标准名": "阿莫西林颗粒", "别名": "阿莫西林干混悬", "编码": "Z2"},
    {"标准名": "布洛芬片", "别名": "芬必得", "编码": "Z3"},
    {"标准名": "布洛芬缓释胶囊", "别名": "芬必得缓释", "编码": "Z4"},
    {"标准名": "对乙酰氨基酚片", "别名": "泰诺林", "编码": "Z5"},
]


def drug_mapping() -> FieldMapping:
    return FieldMapping(standard_name="标准名", query="别名", code="编码")


def drug_spec(**overrides) -> WizardSpec:
    return WizardSpec(mapping=drug_mapping(), **overrides)


class TestBuildSamples:
    def test_pair_mode_basic(self) -> None:
        result = EntityMatchingTemplate().build_samples(
            make_table(["标准名", "别名", "编码"], DRUGS), drug_mapping(), drug_spec()
        )
        assert len(result.samples) == 5
        assert result.dropped == []
        s = result.samples[0]
        assert s.query == "阿莫西林"
        assert s.standard_name == "阿莫西林胶囊"
        assert s.code == "Z1"
        assert s.entity_type == "entity"
        # 每个样本恰好 1 个正例；候选数受池大小限制（5 实体 → 1 正例 + 4 负例）
        assert sum(c.label for c in s.candidates) == 1
        assert len(s.candidates) == 5

    def test_deterministic_with_same_seed(self) -> None:
        table = make_table(["标准名", "别名", "编码"], DRUGS)
        r1 = EntityMatchingTemplate().build_samples(table, drug_mapping(), drug_spec())
        r2 = EntityMatchingTemplate().build_samples(table, drug_mapping(), drug_spec())
        assert [c.name for s in r1.samples for c in s.candidates] == [
            c.name for s in r2.samples for c in s.candidates
        ]

    def test_invalid_mapping_raises(self) -> None:
        with pytest.raises(WizardError, match="字段映射校验失败"):
            EntityMatchingTemplate().build_samples(
                make_table(["标准名"], [{"标准名": "A"}]),
                FieldMapping(standard_name="标准名", query="不存在"),
                drug_spec(),
            )

    def test_missing_standard_row_dropped(self) -> None:
        rows = DRUGS + [{"标准名": None, "别名": "孤儿", "编码": "Z9"}]
        result = EntityMatchingTemplate().build_samples(
            make_table(["标准名", "别名", "编码"], rows), drug_mapping(), drug_spec()
        )
        assert len(result.samples) == 5
        assert len(result.dropped) == 1
        assert result.dropped[0].row == 6
        assert "标准名为空" in result.dropped[0].reason

    def test_query_equals_standard_becomes_easy_sample(self) -> None:
        rows = [{"标准名": "阿莫西林胶囊", "别名": "阿莫西林胶囊", "编码": "Z1"}] + DRUGS[1:]
        result = EntityMatchingTemplate().build_samples(
            make_table(["标准名", "别名", "编码"], rows), drug_mapping(), drug_spec()
        )
        assert result.samples[0].query == "阿莫西林胶囊"
        assert result.samples[0].difficulty == "easy"

    def test_variants_column_expands(self) -> None:
        rows = [
            {"标准名": "阿莫西林胶囊", "变体": "阿莫西林、阿莫仙", "编码": "Z1"},
            {"标准名": "布洛芬片", "变体": "芬必得", "编码": "Z2"},
            {"标准名": "对乙酰氨基酚片", "变体": "泰诺林", "编码": "Z3"},
        ]
        mapping = FieldMapping(standard_name="标准名", variants="变体", code="编码")
        result = EntityMatchingTemplate().build_samples(
            make_table(["标准名", "变体", "编码"], rows), mapping, drug_spec()
        )
        queries = [s.query for s in result.samples]
        assert "阿莫西林" in queries and "阿莫仙" in queries
        assert queries.count("阿莫西林") == 1  # 不与 query 列重复计入

    def test_empty_query_cell_falls_back_to_standard(self) -> None:
        # query 列已映射但单元格为空 → 该行退回标准名自身作查询
        rows = [{"标准名": "阿莫西林胶囊", "别名": None, "编码": "Z1"}] + DRUGS[1:]
        result = EntityMatchingTemplate().build_samples(
            make_table(["标准名", "别名", "编码"], rows), drug_mapping(), drug_spec()
        )
        assert result.samples[0].query == "阿莫西林胶囊"
        assert result.samples[0].difficulty == "easy"

    def test_variant_equal_to_query_not_duplicated(self) -> None:
        # 变体与 query 列值相同 → 不重复计入查询
        rows = [
            {
                "标准名": "阿莫西林胶囊",
                "别名": "阿莫西林",
                "变体": "阿莫西林、阿莫仙",
                "编码": "Z1",
            },
            {"标准名": "布洛芬片", "别名": "芬必得", "变体": "芬必得", "编码": "Z2"},
        ]
        mapping = FieldMapping(standard_name="标准名", query="别名", variants="变体", code="编码")
        result = EntityMatchingTemplate().build_samples(
            make_table(["标准名", "别名", "变体", "编码"], rows), mapping, drug_spec()
        )
        queries = [s.query for s in result.samples]
        assert queries.count("阿莫西林") == 1
        assert queries.count("芬必得") == 1

    def test_row_without_any_query_falls_back_to_standard(self) -> None:
        rows = [
            {"标准名": "A药", "编码": "Z1"},
            {"标准名": "B药", "编码": "Z2"},
            {"标准名": "C药", "编码": "Z3"},
        ]
        mapping = FieldMapping(standard_name="标准名", code="编码")
        result = EntityMatchingTemplate().build_samples(
            make_table(["标准名", "编码"], rows), mapping, drug_spec()
        )
        assert [s.query for s in result.samples] == ["A药", "B药", "C药"]
        assert all(s.difficulty == "easy" for s in result.samples)

    def test_variants_cap_64(self) -> None:
        many = "、".join(f"变体{i}" for i in range(70))
        rows = [{"标准名": f"药{i}", "变体": many, "编码": f"Z{i}"} for i in range(3)]
        mapping = FieldMapping(standard_name="标准名", variants="变体", code="编码")
        result = EntityMatchingTemplate().build_samples(
            make_table(["标准名", "变体", "编码"], rows), mapping, drug_spec()
        )
        # 每行最多 64 个查询（3 行 × 64 = 192）
        assert len(result.samples) == 192

    def test_entity_type_column(self) -> None:
        rows = [dict(d, 类型="drug") for d in DRUGS]
        mapping = FieldMapping(
            standard_name="标准名", query="别名", code="编码", entity_type="类型"
        )
        result = EntityMatchingTemplate().build_samples(
            make_table(["标准名", "别名", "编码", "类型"], rows), mapping, drug_spec()
        )
        assert all(s.entity_type == "drug" for s in result.samples)

    def test_small_kb_fewer_candidates(self) -> None:
        rows = DRUGS[:2]
        result = EntityMatchingTemplate().build_samples(
            make_table(["标准名", "别名", "编码"], rows), drug_mapping(), drug_spec(n_candidates=8)
        )
        # 池里只有 2 个实体 → 正例 1 + 负例 1
        assert all(len(s.candidates) == 2 for s in result.samples)

    def test_duplicate_standards_deduped_in_pool(self) -> None:
        rows = DRUGS + [DRUGS[0]]
        result = EntityMatchingTemplate().build_samples(
            make_table(["标准名", "别名", "编码"], rows), drug_mapping(), drug_spec()
        )
        # 重复标准行不产生新样本（query 相同被去重前的 build 层面：仍生成但内容一致）
        assert len(result.samples) == 6  # 5 + 重复行的 1 个 query
        s0 = result.samples[0]
        # 负例池不含重复条目
        names = [c.name for c in s0.candidates]
        assert len(names) == len(set(names))


class TestDifficulty:
    @pytest.mark.parametrize(
        ("query", "standard", "expected"),
        [
            ("阿莫西林胶囊", "阿莫西林胶囊", "easy"),
            ("阿莫西林", "阿莫西林胶囊", "easy"),  # q in s
            ("阿莫西林胶囊 250mg", "阿莫西林胶囊", "easy"),  # s in q（去空格后）
            ("阿莫西林胶襄", "阿莫西林胶囊", "medium"),  # 编辑距离 1
            ("泰诺林", "对乙酰氨基酚片", "hard"),
        ],
    )
    def test_classification(self, query, standard, expected) -> None:
        assert classify_difficulty(query, standard) == expected

    def test_edit_distance_empty(self) -> None:
        assert edit_distance("abc", "") == 3

    def test_edit_distance_swap_order(self) -> None:
        assert edit_distance("", "abc") == 3


class TestFormatRecord:
    def _sample(self, code: str | None = "Z1") -> MatchingSample:
        return MatchingSample(
            query="阿莫西林",
            standard_name="阿莫西林胶囊",
            code=code,
            entity_type="drug",
            difficulty="medium",
            candidates=[
                Candidate("布洛芬片", "Z3", False),
                Candidate("阿莫西林胶囊", code, True),
            ],
            source_row=1,
        )

    def test_record_with_code(self) -> None:
        record = EntityMatchingTemplate().format_record(self._sample())
        assert record["instruction"].startswith("从候选列表中选出")
        assert "输入实体: 阿莫西林" in record["input"]
        assert "1. 布洛芬片 (Z3)" in record["input"]
        output = json.loads(record["output"])
        assert output["match_index"] == 2
        assert output["code"] == "Z1"
        assert record["metadata"] == {"entity_type": "drug", "difficulty": "medium"}

    def test_record_without_code(self) -> None:
        s = self._sample(code=None)
        s.candidates = [Candidate("布洛芬片", None, False), Candidate("阿莫西林胶囊", None, True)]
        record = EntityMatchingTemplate().format_record(s)
        assert "(Z3)" not in record["input"]
        assert "2. 阿莫西林胶囊" in record["input"]
        assert "code" not in json.loads(record["output"])

    def test_no_labeled_candidate_raises(self) -> None:
        s = self._sample()
        s.candidates = [Candidate("布洛芬片", "Z3", False)]
        with pytest.raises(WizardError, match="没有标注正确候选"):
            EntityMatchingTemplate().format_record(s)


class TestRegistry:
    def test_default_template_registered(self) -> None:
        assert "medical_entity" in available_templates()
        assert get_template("medical_entity").name == "medical_entity"

    def test_unknown_template(self) -> None:
        with pytest.raises(WizardError, match="未知模板"):
            get_template("nope")

    def test_duplicate_registration(self) -> None:
        with pytest.raises(WizardError, match="已注册"):
            register_template(EntityMatchingTemplate())

    def test_describe_mentions_key_columns(self) -> None:
        text = get_template("medical_entity").describe()
        assert "标准名" in text
        assert "泄漏" in text or "打乱" in text


class TestNegativePicking:
    def test_prefix_hard_negatives_preferred(self) -> None:
        import random

        rng = random.Random(0)
        standards = [
            ("阿莫西林胶囊", "Z1"),
            ("阿莫西林颗粒", "Z2"),
            ("阿莫西林片", "Z3"),
            ("布洛芬片", "Z4"),
        ]
        picked = EntityMatchingTemplate._pick_negatives("阿莫西林胶囊", standards, 3, rng)
        names = [n for n, _ in picked]
        assert (
            names[0] == "阿莫西林颗粒" or names[1] == "阿莫西林颗粒" or names[2] == "阿莫西林颗粒"
        )
        assert "阿莫西林胶囊" not in names

    def test_no_prefix_match_falls_back_to_random(self) -> None:
        import random

        rng = random.Random(0)
        standards = [("甲药", "Z1"), ("乙药", "Z2"), ("丙药", "Z3")]
        picked = EntityMatchingTemplate._pick_negatives("丁药", standards, 2, rng)
        assert len(picked) == 2

    def test_more_hard_than_needed(self) -> None:
        import random

        rng = random.Random(0)
        standards = [(f"阿莫西林{i}", f"Z{i}") for i in range(10)]
        picked = EntityMatchingTemplate._pick_negatives("阿莫西林9", standards, 3, rng)
        assert len(picked) == 3
        assert all(n != "阿莫西林9" for n, _ in picked)


class TestBuildResultDefaults:
    def test_empty(self) -> None:
        r = BuildResult()
        assert r.samples == []
        assert r.dropped == []
