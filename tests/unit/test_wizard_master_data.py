"""主数据模板测试：双任务构建 / messages 格式化 / 任务池隔离 / 端到端导出。"""

import json

import pytest

from src.data.wizard import MasterDataTemplate, available_templates  # noqa: F401  注册副作用
from src.data.wizard.exporter import split_by_entity
from src.data.wizard.importers import RawTable, suggest_mapping
from src.data.wizard.pipeline import WizardPipeline
from src.data.wizard.spec import FieldMapping, WizardError, WizardSpec
from src.data.wizard.templates import Candidate, MatchingSample
from src.data.wizard.templates_master_data import (
    INST_SYSTEM_PROMPT,
    PROD_SYSTEM_PROMPT,
    _common_prefix_len,
    _detect_task,
)


def make_table(columns: list[str], rows: list[dict]) -> RawTable:
    return RawTable(
        source="test",
        columns=columns,
        rows=[{c: r.get(c) for c in columns} for r in rows],
    )


INSTITUTIONS = [
    {"标准名": "保和堂(昌平区光明路店)", "别名": "保和堂大药房", "编码": "P000001", "类型": "机构"},
    {"标准名": "益民堂(海淀区中关村店)", "别名": "益民堂药房", "编码": "P000002", "类型": "机构"},
    {"标准名": "仁和药房(朝阳区望京店)", "别名": "仁和", "编码": "P000003", "类型": "机构"},
    {"标准名": "同济堂(浦东新区张江店)", "别名": "同济堂药房", "编码": "P000004", "类型": "机构"},
]

PRODUCTS = [
    {
        "标准名": "阿莫西林胶囊",
        "别名": "阿莫仙",
        "编码": "Z15020414",
        "类型": "产品",
        "规格": "0.25g*24片/盒",
    },
    {
        "标准名": "阿莫西林颗粒",
        "别名": "阿莫西林干混悬",
        "编码": "Z15020415",
        "类型": "产品",
        "规格": "0.125g*12袋/盒",
    },
    {
        "标准名": "布洛芬缓释胶囊",
        "别名": "芬必得缓释",
        "编码": "Z15020416",
        "类型": "产品",
        "规格": "0.3g*20粒/盒",
    },
    {
        "标准名": "草果四味汤散",
        "别名": "草果四味",
        "编码": "Z15020417",
        "类型": "产品",
        "规格": "10g/袋",
    },
]

MD_COLUMNS = ["标准名", "别名", "编码", "类型", "规格"]


def md_rows() -> list[dict]:
    return [dict(r) for r in INSTITUTIONS + PRODUCTS]


def md_mapping() -> FieldMapping:
    return FieldMapping(
        standard_name="标准名", query="别名", code="编码", entity_type="类型", spec="规格"
    )


def md_spec() -> WizardSpec:
    return WizardSpec(mapping=md_mapping(), template="master_data")


class TestTaskDetection:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            ("机构", "institution"),
            ("institution", "institution"),
            ("hospital", "institution"),
            ("pharmacy_chain", "institution"),
            ("门店", "institution"),
            (None, "institution"),
            ("", "institution"),
            ("产品", "product"),
            ("product", "product"),
            ("药品", "product"),
            ("drug", "product"),
            ("unknown_value", "institution"),  # 识别不了默认机构
        ],
    )
    def test_detection(self, value, expected) -> None:
        assert _detect_task(value) == expected

    def test_common_prefix(self) -> None:
        assert _common_prefix_len("阿莫西林胶囊", "阿莫西林颗粒") == 4
        assert _common_prefix_len("abc", "abc") == 3
        assert _common_prefix_len("", "abc") == 0
        assert _common_prefix_len("草果", "布洛芬") == 0


class TestBuildSamples:
    def test_dual_task_pool_isolation(self) -> None:
        result = MasterDataTemplate().build_samples(
            make_table(MD_COLUMNS, md_rows()), md_mapping(), md_spec()
        )
        assert len(result.samples) == 8  # 每行 1 个查询
        inst_names = {r["标准名"] for r in INSTITUTIONS}
        prod_names = {r["标准名"] for r in PRODUCTS}
        for s in result.samples:
            pool = inst_names if s.entity_type == "institution" else prod_names
            other = prod_names if s.entity_type == "institution" else inst_names
            assert {c.name for c in s.candidates} <= pool  # 候选不跨任务池
            assert not ({c.name for c in s.candidates} & other)
            assert sum(c.label for c in s.candidates) == 1

    def test_default_task_is_institution_without_type_column(self) -> None:
        rows = [{"标准名": r["标准名"], "别名": r["别名"], "编码": r["编码"]} for r in INSTITUTIONS]
        mapping = FieldMapping(standard_name="标准名", query="别名", code="编码")
        result = MasterDataTemplate().build_samples(
            make_table(["标准名", "别名", "编码"], rows), mapping, md_spec()
        )
        assert all(s.entity_type == "institution" for s in result.samples)

    def test_product_query_includes_spec(self) -> None:
        result = MasterDataTemplate().build_samples(
            make_table(MD_COLUMNS, md_rows()), md_mapping(), md_spec()
        )
        prod = [s for s in result.samples if s.entity_type == "product"][0]
        assert prod.query == "阿莫仙 0.25g*24片/盒"
        pos = next(c for c in prod.candidates if c.label)
        assert pos.spec == "0.25g*24片/盒"

    def test_product_without_spec_column_keeps_plain_query(self) -> None:
        rows = [
            {"标准名": r["标准名"], "别名": r["别名"], "编码": r["编码"], "类型": r["类型"]}
            for r in PRODUCTS
        ]
        mapping = FieldMapping(
            standard_name="标准名", query="别名", code="编码", entity_type="类型"
        )
        result = MasterDataTemplate().build_samples(
            make_table(["标准名", "别名", "编码", "类型"], rows), mapping, md_spec()
        )
        assert all(" " not in s.query for s in result.samples)
        assert all(c.spec is None for s in result.samples for c in s.candidates)

    def test_institution_query_untouched_by_spec(self) -> None:
        # 机构行即使规格列有值也不并入查询（规格只属于产品任务）
        rows = md_rows()
        for r in rows[:4]:
            r["规格"] = "不应出现"
        result = MasterDataTemplate().build_samples(
            make_table(MD_COLUMNS, rows), md_mapping(), md_spec()
        )
        inst = [s for s in result.samples if s.entity_type == "institution"][0]
        assert inst.query == "保和堂大药房"

    def test_dropped_rows_recorded(self) -> None:
        rows = md_rows() + [
            {"标准名": None, "别名": "孤儿", "编码": "P9", "类型": "机构", "规格": None}
        ]
        result = MasterDataTemplate().build_samples(
            make_table(MD_COLUMNS, rows), md_mapping(), md_spec()
        )
        assert len(result.samples) == 8
        assert len(result.dropped) == 1
        assert result.dropped[0].row == 9
        assert "标准名为空" in result.dropped[0].reason

    def test_duplicate_standard_rows_not_double_pooled(self) -> None:
        rows = md_rows() + [dict(INSTITUTIONS[0])]  # 完全重复的标准行
        result = MasterDataTemplate().build_samples(
            make_table(MD_COLUMNS, rows), md_mapping(), md_spec()
        )
        assert len(result.samples) == 9  # 重复行仍生成样本（去重交给后续 dedup 环节）
        names = [c.name for c in result.samples[0].candidates]
        assert len(names) == len(set(names))  # 但负例池不重复

    def test_empty_query_cell_falls_back_to_standard(self) -> None:
        rows = md_rows()
        rows[0]["别名"] = None
        result = MasterDataTemplate().build_samples(
            make_table(MD_COLUMNS, rows), md_mapping(), md_spec()
        )
        assert result.samples[0].query == INSTITUTIONS[0]["标准名"]

    def test_deterministic_with_same_seed(self) -> None:
        table = make_table(MD_COLUMNS, md_rows())
        r1 = MasterDataTemplate().build_samples(table, md_mapping(), md_spec())
        r2 = MasterDataTemplate().build_samples(table, md_mapping(), md_spec())
        assert [(s.query, [c.name for c in s.candidates]) for s in r1.samples] == [
            (s.query, [c.name for c in s.candidates]) for s in r2.samples
        ]

    def test_invalid_spec_column_raises(self) -> None:
        bad = FieldMapping(
            standard_name="标准名", query="别名", code="编码", entity_type="类型", spec="不存在"
        )
        with pytest.raises(WizardError, match="字段映射校验失败"):
            MasterDataTemplate().build_samples(make_table(MD_COLUMNS, md_rows()), bad, md_spec())

    def test_variants_expand_queries(self) -> None:
        rows = [dict(r) for r in md_rows()]
        # 变体里重复了别名值（保和堂大药房）→ 不重复计入查询
        rows[0]["变体"] = "保和堂大药房、保和堂、昌平保和堂"
        columns = MD_COLUMNS + ["变体"]
        mapping = FieldMapping(
            standard_name="标准名",
            query="别名",
            code="编码",
            entity_type="类型",
            spec="规格",
            variants="变体",
        )
        result = MasterDataTemplate().build_samples(make_table(columns, rows), mapping, md_spec())
        queries = [s.query for s in result.samples]
        assert queries.count("保和堂大药房") == 1
        assert queries.count("保和堂") == 1
        assert queries.count("昌平保和堂") == 1

    def test_no_query_columns_falls_back_to_standard(self) -> None:
        rows = [{"标准名": r["标准名"], "编码": r["编码"], "类型": r["类型"]} for r in INSTITUTIONS]
        mapping = FieldMapping(standard_name="标准名", code="编码", entity_type="类型")
        result = MasterDataTemplate().build_samples(
            make_table(["标准名", "编码", "类型"], rows), mapping, md_spec()
        )
        assert [s.query for s in result.samples] == [r["标准名"] for r in INSTITUTIONS]


class TestFormatInstitution:
    def _inst_sample(self, difficulty: str = "medium", with_code: bool = True) -> MatchingSample:
        code = "P000001" if with_code else None
        return MatchingSample(
            query="保和堂大药房",
            standard_name="保和堂(昌平区光明路店)",
            code=code,
            entity_type="institution",
            difficulty=difficulty,
            candidates=[
                Candidate("益民堂(海淀区中关村店)", "P000002", False),
                Candidate("保和堂(昌平区光明路店)", code, True),
                Candidate("仁和药房(朝阳区望京店)", "P000003", False),
            ],
            source_row=1,
        )

    def test_messages_shape_and_answers(self) -> None:
        record = MasterDataTemplate().format_record(self._inst_sample())
        msgs = record["messages"]
        assert [m["role"] for m in msgs] == ["system", "user", "assistant"]
        assert msgs[0]["content"] == INST_SYSTEM_PROMPT
        assert "【输入机构】：保和堂大药房" in msgs[1]["content"]
        assert "[1] 编码: P000002, 名称: 益民堂(海淀区中关村店)" in msgs[1]["content"]
        items = json.loads(msgs[2]["content"])
        assert len(items) == 3
        assert [it["index"] for it in items] == [1, 2, 3]  # 与候选顺序一一对应
        pos = [it for it in items if it["matched"]][0]
        assert pos["index"] == 2
        assert pos["confidence"] in ("High", "Medium")
        for it in items:
            if not it["matched"]:
                assert it["confidence"] == "Low"
                assert "false" in it["reasoning"]

    @pytest.mark.parametrize(
        ("difficulty", "reasoning_frag", "confidence"),
        [
            ("easy", "精确匹配", "High"),
            ("medium", "简写/全称差异", "High"),
            ("hard", "简写/全称差异", "Medium"),
        ],
    )
    def test_positive_reasoning_by_difficulty(self, difficulty, reasoning_frag, confidence) -> None:
        record = MasterDataTemplate().format_record(self._inst_sample(difficulty=difficulty))
        pos = [it for it in json.loads(record["messages"][2]["content"]) if it["matched"]][0]
        assert reasoning_frag in pos["reasoning"]
        assert pos["confidence"] == confidence

    def test_candidate_without_code_renders_name_only(self) -> None:
        record = MasterDataTemplate().format_record(self._inst_sample(with_code=False))
        user = record["messages"][1]["content"]
        assert "[2] 名称: 保和堂(昌平区光明路店)" in user
        assert "[2] 编码:" not in user

    def test_no_labeled_candidate_raises(self) -> None:
        s = self._inst_sample()
        s.candidates = [Candidate("益民堂(海淀区中关村店)", "P000002", False)]
        with pytest.raises(WizardError, match="没有标注正确候选"):
            MasterDataTemplate().format_record(s)


class TestFormatProduct:
    def _prod_sample(self) -> MatchingSample:
        return MatchingSample(
            query="阿莫仙 0.25g*24片/盒",
            standard_name="阿莫西林胶囊",
            code="Z15020414",
            entity_type="product",
            difficulty="medium",
            candidates=[
                Candidate("阿莫西林颗粒", "Z15020415", False, spec="0.125g*12袋/盒"),
                Candidate("阿莫西林胶囊", "Z15020414", True, spec="0.25g*24片/盒"),
                Candidate("布洛芬缓释胶囊", "Z15020416", False, spec="0.3g*20粒/盒"),
            ],
            source_row=5,
        )

    def test_grades_and_shape(self) -> None:
        record = MasterDataTemplate().format_record(self._prod_sample())
        msgs = record["messages"]
        assert msgs[0]["content"] == PROD_SYSTEM_PROMPT
        assert "【输入产品】：阿莫仙 0.25g*24片/盒" in msgs[1]["content"]
        assert "[1] 编码: Z15020415, 名称: 阿莫西林颗粒, 规格: 0.125g*12袋/盒" in msgs[1]["content"]
        items = json.loads(msgs[2]["content"])
        grades = {it["match_grade"] for it in items}
        assert grades == {"A", "B", "D"}
        a = next(it for it in items if it["match_grade"] == "A")
        assert a["core_name_match"] is True and a["spec_diff"] == "无"
        b = next(it for it in items if it["match_grade"] == "B")
        assert b["core_name_match"] is True
        assert b["modifier_diff"] == "剂型差异"
        assert b["spec_diff"] == "不同"
        d = next(it for it in items if it["match_grade"] == "D")
        assert d["core_name_match"] is False

    def test_same_name_negative_gets_b_without_modifier_diff(self) -> None:
        # 同名不同码（换包装）：名称一致 → modifier_diff 无；规格相同 → spec_diff 无
        s = self._prod_sample()
        s.candidates[0] = Candidate("阿莫西林胶囊", "ZOTHER", False, spec="0.25g*24片/盒")
        record = MasterDataTemplate().format_record(s)
        b = [it for it in json.loads(record["messages"][2]["content"]) if it["match_grade"] == "B"][
            0
        ]
        assert b["modifier_diff"] == "无"
        assert b["spec_diff"] == "无"

    def test_candidate_without_spec_renders_without_spec_segment(self) -> None:
        s = self._prod_sample()
        s.candidates[2] = Candidate("布洛芬缓释胶囊", "Z15020416", False, spec=None)
        user = MasterDataTemplate().format_record(s)["messages"][1]["content"]
        assert "[3] 编码: Z15020416, 名称: 布洛芬缓释胶囊\n" in user
        assert "规格: None" not in user

    def test_candidate_without_code_renders_name_only(self) -> None:
        s = self._prod_sample()
        s.candidates[2] = Candidate("布洛芬缓释胶囊", None, False, spec="0.3g*20粒/盒")
        user = MasterDataTemplate().format_record(s)["messages"][1]["content"]
        assert "[3] 名称: 布洛芬缓释胶囊" in user

    def test_no_labeled_candidate_raises(self) -> None:
        s = self._prod_sample()
        s.candidates = [Candidate("阿莫西林颗粒", "Z15020415", False)]
        with pytest.raises(WizardError, match="没有标注正确候选"):
            MasterDataTemplate().format_record(s)


class TestRegistryAndMapping:
    def test_master_data_registered_via_package_import(self) -> None:
        assert "master_data" in available_templates()

    def test_describe_mentions_dual_task(self) -> None:
        text = MasterDataTemplate().describe()
        assert "机构" in text and "产品" in text
        assert "messages" in text

    def test_suggest_mapping_picks_spec_column(self) -> None:
        mapping = suggest_mapping(["标准名", "别名", "编码", "规格", "类型"])
        assert mapping.spec == "规格"

    def test_candidate_spec_defaults_none(self) -> None:
        c = Candidate("X", None, True)
        assert c.spec is None


class TestPipelineEndToEnd:
    def test_master_data_export_roundtrip(self, tmp_path) -> None:
        table = make_table(MD_COLUMNS, md_rows())
        spec = WizardSpec(
            mapping=md_mapping(),
            template="master_data",
            n_candidates=4,
        )
        report = WizardPipeline(spec).run(table, tmp_path)
        assert report.export is not None
        total = sum(report.split_counts.values())
        assert total == 8
        systems = set()
        for split in ("train", "val", "test"):
            records = json.loads((tmp_path / f"{split}.json").read_text(encoding="utf-8"))
            for rec in records:
                assert set(rec) == {"messages"}
                roles = [m["role"] for m in rec["messages"]]
                assert roles == ["system", "user", "assistant"]
                systems.add(rec["messages"][0]["content"])
                items = json.loads(rec["messages"][2]["content"])
                assert len(items) >= 2
        # 8 实体 0.8/0.1/0.1 切分 → train 必然同时含两个任务
        assert systems == {INST_SYSTEM_PROMPT, PROD_SYSTEM_PROMPT}

    def test_split_respects_task_entity_groups(self) -> None:
        # 同一标准实体的多个变体样本必须落在同一 split（防泄漏），任务间互不影响
        rows = [dict(r) for r in md_rows()]
        rows[0]["变体"] = "保和堂、昌平保和堂"
        columns = MD_COLUMNS + ["变体"]
        mapping = FieldMapping(
            standard_name="标准名",
            query="别名",
            code="编码",
            entity_type="类型",
            spec="规格",
            variants="变体",
        )
        result = MasterDataTemplate().build_samples(make_table(columns, rows), mapping, md_spec())
        splits = split_by_entity(result.samples, md_spec())
        # 泄漏 = 同一实体出现在多个 split；同一 split 内多次出现是正常的（变体样本）
        entity_splits: dict[str, set[str]] = {}
        for split_name, bucket in splits.as_dict().items():
            for s in bucket:
                entity_splits.setdefault(s.code or s.standard_name, set()).add(split_name)
        for entity, where in entity_splits.items():
            assert len(where) == 1, f"实体 {entity} 泄漏到多个 split: {where}"
