"""通用实体匹配评测（R120 评测闭环）纯逻辑测试。

北极星（通用微调台）要求：用户在 Data Wizard 导出的任意领域 test.json 上
评自己的 adapter，结果写 eval_detail_*.json 点亮 02/03 两页。本文件钉住
判分/聚合/落盘的纯逻辑（无 torch 依赖——模型加载在 scripts/ 层懒加载）。
"""

import json

import pytest

from src.evaluation.entity_eval import (
    build_rows,
    extract_query,
    judge,
    normalize_text,
    parse_numbered_candidates,
    summarize,
    write_report,
)

INPUT_WITH_CANDIDATES = """从候选列表中选出与输入实体一致的标准名称。
输入实体: 蓝星科技

候选列表:
1. 云岭数据技术（杭州）有限公司 (SUP-0003)
2. 蓝星智能科技（深圳）有限公司 (SUP-0001)
3. 长风物流（上海）有限公司 (SUP-0004)"""


class TestNormalizeText:
    def test_strips_whitespace_and_trailing_punctuation(self) -> None:
        assert normalize_text('  {"match_index": 2} 。 ') == '{"match_index": 2}'

    def test_strips_code_fences(self) -> None:
        assert normalize_text('```json\n{"match_index": 2}\n```') == '{"match_index": 2}'

    def test_strips_wrapping_quotes(self) -> None:
        assert normalize_text('"蓝星智能科技"') == "蓝星智能科技"

    def test_collapses_internal_whitespace(self) -> None:
        assert normalize_text("蓝星   智能\n科技") == "蓝星 智能 科技"


class TestParseNumberedCandidates:
    def test_extracts_names_in_order(self) -> None:
        names = parse_numbered_candidates(INPUT_WITH_CANDIDATES)
        assert names == [
            "云岭数据技术（杭州）有限公司",
            "蓝星智能科技（深圳）有限公司",
            "长风物流（上海）有限公司",
        ]

    def test_plain_lines_ignored(self) -> None:
        names = parse_numbered_candidates("候选列表:\n1. 甲 (Z1)\n无关行\n2. 乙 (Z2)")
        assert names == ["甲", "乙"]

    def test_empty_input(self) -> None:
        assert parse_numbered_candidates("") == []


class TestExtractQuery:
    def test_halfwidth_colon(self) -> None:
        assert extract_query(INPUT_WITH_CANDIDATES) == "蓝星科技"

    def test_fullwidth_colon(self) -> None:
        assert extract_query("输入实体：华信电气") == "华信电气"

    def test_missing_returns_empty(self) -> None:
        assert extract_query("没有查询行") == ""


class TestJudge:
    def test_both_json_same_index(self) -> None:
        assert judge('{"match_index": 2, "code": "SUP-0001"}', '{"match_index": 2}') is True

    def test_both_json_different_index(self) -> None:
        assert judge('{"match_index": 2}', '{"match_index": 3}') is False

    def test_fenced_prediction_matches_plain_expected(self) -> None:
        assert judge('{"match_index": 2}', '```json\n{"match_index": 2}\n```') is True

    def test_index_semantics_over_code(self) -> None:
        # match_index 是语义答案；index 对即对（code 缺失/不一致不推翻）
        assert judge('{"match_index": 2, "code": "A"}', '{"match_index": 2, "code": "B"}') is True

    def test_string_index_coerced(self) -> None:
        assert judge('{"match_index": 2}', '{"match_index": "2"}') is True

    def test_plain_text_equal(self) -> None:
        assert judge("蓝星智能科技（深圳）有限公司", " 蓝星智能科技（深圳）有限公司。") is True

    def test_plain_text_different(self) -> None:
        assert judge("蓝星智能科技", "云岭数据") is False

    def test_expected_json_predicted_garbage(self) -> None:
        assert judge('{"match_index": 2}', "我认为是第 2 个") is False

    def test_dicts_without_match_index_fall_back_to_equality(self) -> None:
        assert judge('{"name": "甲"}', '{"name": "甲"}') is True
        assert judge('{"name": "甲"}', '{"name": "乙"}') is False


def make_records() -> list[dict]:
    """构造 wizard 形状的 test 记录（与 EntityMatchingTemplate.format_record 同构）。"""
    return [
        {
            "instruction": "从候选列表中选出与输入实体一致的标准名称。",
            "input": INPUT_WITH_CANDIDATES,
            "output": '{"match_index": 2, "code": "SUP-0001"}',
            "metadata": {"entity_type": "supplier", "difficulty": "easy"},
        },
        {
            "instruction": "从候选列表中选出与输入实体一致的标准名称。",
            "input": INPUT_WITH_CANDIDATES,
            "output": '{"match_index": 1, "code": "SUP-0003"}',
            "metadata": {"entity_type": "supplier", "difficulty": "hard"},
        },
    ]


class TestBuildRows:
    def test_row_contract_fields(self) -> None:
        rows = build_rows(
            make_records(), ['{"match_index": 2}', '{"match_index": 3}'], [10.0, 20.0]
        )
        assert len(rows) == 2
        row = rows[0]
        # 页面契约（domains/medical_entity/eval/report.py save_results 同构）
        for key in (
            "query",
            "ground_truth",
            "ground_truth_code",
            "predicted_name",
            "predicted_code",
            "confidence",
            "difficulty",
            "entity_type",
            "correct",
            "latency_ms",
            "error",
        ):
            assert key in row, f"per_sample 契约字段缺失: {key}"
        assert row["query"] == "蓝星科技"
        assert row["ground_truth"] == "蓝星智能科技（深圳）有限公司"
        assert row["ground_truth_code"] == "SUP-0001"
        assert row["predicted_name"] == "蓝星智能科技（深圳）有限公司"
        assert row["predicted_code"] is None  # 预测 JSON 无 code 键
        assert row["correct"] is True
        assert row["entity_type"] == "supplier"
        assert row["difficulty"] == "easy"

    def test_wrong_index_marks_incorrect_with_resolved_name(self) -> None:
        rows = build_rows(
            make_records(), ['{"match_index": 2}', '{"match_index": 3}'], [10.0, 20.0]
        )
        assert rows[1]["correct"] is False
        assert rows[1]["predicted_name"] == "长风物流（上海）有限公司"

    def test_unresolvable_prediction_falls_back_to_raw_text(self) -> None:
        rows = build_rows(make_records(), ['{"match_index": 2}', "不会"], [1.0, 2.0])
        assert rows[1]["correct"] is False
        assert rows[1]["predicted_name"] == "不会"

    def test_missing_metadata_defaults(self) -> None:
        records = make_records()
        del records[0]["metadata"]
        rows = build_rows(records, ['{"match_index": 2}', "x"], [1.0, 2.0])
        assert rows[0]["difficulty"] is None
        assert rows[0]["entity_type"] is None

    def test_length_mismatch_truncates_to_shortest(self) -> None:
        rows = build_rows(make_records()[:1] + make_records(), ['{"match_index": 2}'], [1.0])
        assert len(rows) == 1


class TestSummarize:
    def _rows(self) -> list[dict]:
        return build_rows(
            make_records(), ['{"match_index": 2}', '{"match_index": 3}'], [10.0, 30.0]
        )

    def test_summary_contract(self) -> None:
        rows = self._rows()
        s = summarize("my-adapter", rows)
        for key in (
            "model",
            "total",
            "correct",
            "overall_accuracy",
            "accuracy_by_difficulty",
            "accuracy_by_type",
            "avg_latency_ms",
            "mrr",
            "per_sample",
        ):
            assert key in s, f"eval_detail 条目字段缺失: {key}"
        assert s["model"] == "my-adapter"
        assert s["total"] == 2
        assert s["correct"] == 1
        assert s["overall_accuracy"] == 0.5
        assert s["accuracy_by_difficulty"] == {"easy": 1.0, "hard": 0.0}
        assert s["accuracy_by_type"] == {"supplier": 0.5}
        assert s["avg_latency_ms"] == 20.0
        # 单答案贪心任务：命中即 RR=1，未命中无排名 → MRR 恒等于命中率
        assert s["mrr"] == pytest.approx(0.5)
        assert s["per_sample"] == rows

    def test_empty_rows(self) -> None:
        s = summarize("m", [])
        assert s["total"] == 0
        assert s["overall_accuracy"] is None
        assert s["accuracy_by_difficulty"] == {}
        assert s["accuracy_by_type"] == {}
        assert s["mrr"] is None

    def test_rows_without_metadata_omitted_from_groupings(self) -> None:
        records = make_records()
        del records[0]["metadata"]
        del records[1]["metadata"]
        rows = build_rows(records, ['{"match_index": 2}', '{"match_index": 1}'], [1.0, 2.0])
        s = summarize("m", rows)
        assert s["accuracy_by_difficulty"] == {}
        assert s["accuracy_by_type"] == {}


class TestWriteReport:
    def test_writes_eval_detail_json(self, tmp_path) -> None:
        rows = build_rows(make_records(), ['{"match_index": 2}', '{"match_index": 3}'], [1.0, 2.0])
        summaries = [summarize("my-adapter", rows)]
        path = write_report(summaries, tmp_path)
        assert path.exists()
        assert path.name.startswith("eval_detail_") and path.suffix == ".json"
        loaded = json.loads(path.read_text(encoding="utf-8"))
        assert isinstance(loaded, list) and loaded[0]["model"] == "my-adapter"

    def test_creates_missing_dir(self, tmp_path) -> None:
        target = tmp_path / "domains" / "entity_matching" / "data" / "results"
        path = write_report([summarize("m", [])], target)
        assert path.exists()

    def test_returns_latest_sorted_last(self, tmp_path) -> None:
        # load_eval_data 用 sorted(glob, reverse=True)[0] 取最新——文件名必须
        # 时间戳可排序（medical save_results 同款 %Y%m%d_%H%M%S）
        rows = build_rows(make_records(), ['{"match_index": 2}', '{"match_index": 3}'], [1.0, 2.0])
        p1 = write_report([summarize("first", rows)], tmp_path)
        p2 = write_report([summarize("second", rows)], tmp_path)
        assert sorted(tmp_path.glob("eval_detail_*.json"), reverse=True)[0] in (p1, p2)
