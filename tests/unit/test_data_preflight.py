"""数据集预检（src/data/preflight.py）的单元测试。

覆盖：路径识别、JSON/JSONL 解析、格式判定（alpaca/messages/unknown）、
SFT 消费方解释（messages → 域训练脚本修法、unknown → 字段提示、缺字段阻断）。
"""

from __future__ import annotations

import json

import pytest

from src.data.preflight import (
    check_dataset_for_sft,
    inspect_dataset_file,
    looks_like_local_path,
)


def _write_json(path, records) -> None:
    path.write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")


ALPACA = [
    {"instruction": "从候选中选出标准名", "input": "阿莫西林", "output": "阿莫西林胶囊"},
    {"instruction": "从候选中选出标准名", "input": "芬必得", "output": "布洛芬片"},
]

MESSAGES = [
    {
        "messages": [
            {"role": "system", "content": "你是主数据匹配助手"},
            {"role": "user", "content": "匹配：仁和"},
            {"role": "assistant", "content": "仁和药房(朝阳区望京店)"},
        ]
    }
]


class TestLooksLikeLocalPath:
    def test_hf_name_is_not_local(self) -> None:
        assert looks_like_local_path("yahma/alpaca-cleaned") is False

    def test_relative_path_is_local(self) -> None:
        assert looks_like_local_path("outputs/wizard/demo/train.json") is True

    def test_absolute_path_is_local(self) -> None:
        assert looks_like_local_path("/tmp/anything") is True

    def test_dot_relative_path_is_local(self) -> None:
        assert looks_like_local_path("./mydata") is True

    def test_hf_name_with_slash_is_not_local(self) -> None:
        # HF 名是 org/name 形态，本身含 / —— 不能拿斜杠当判据
        assert looks_like_local_path("yahma/alpaca-cleaned") is False

    def test_bare_json_filename_is_local(self) -> None:
        assert looks_like_local_path("data.json") is True

    def test_bare_jsonl_filename_is_local(self) -> None:
        assert looks_like_local_path("data.jsonl") is True


class TestInspectDatasetFile:
    def test_alpaca_file_ok(self, tmp_path) -> None:
        f = tmp_path / "train.json"
        _write_json(f, ALPACA)
        insp = inspect_dataset_file(f)
        assert insp.ok
        assert insp.fmt == "alpaca"
        assert insp.n_records == 2
        assert insp.warnings == ()

    def test_jsonl_supported(self, tmp_path) -> None:
        f = tmp_path / "train.jsonl"
        f.write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in ALPACA) + "\n",
            encoding="utf-8",
        )
        insp = inspect_dataset_file(f)
        assert insp.ok
        assert insp.fmt == "alpaca"
        assert insp.n_records == 2

    def test_jsonl_blank_lines_skipped(self, tmp_path) -> None:
        f = tmp_path / "train.jsonl"
        f.write_text("\n\n" + json.dumps(ALPACA[0]) + "\n\n", encoding="utf-8")
        insp = inspect_dataset_file(f)
        assert insp.ok
        assert insp.n_records == 1

    def test_jsonl_bad_line_is_error(self, tmp_path) -> None:
        f = tmp_path / "train.jsonl"
        f.write_text('{"instruction": "a", "output": "b"}\nnot json\n', encoding="utf-8")
        insp = inspect_dataset_file(f)
        assert not insp.ok
        assert "第 2 行不是合法 JSON" in insp.errors[0]

    def test_messages_format_detected_without_error(self, tmp_path) -> None:
        f = tmp_path / "train.json"
        _write_json(f, MESSAGES)
        insp = inspect_dataset_file(f)
        assert insp.ok  # 通用体检只陈述事实；是否可用由消费方决定
        assert insp.fmt == "messages"
        assert insp.n_records == 1

    def test_missing_file_is_error_with_fix_hint(self, tmp_path) -> None:
        insp = inspect_dataset_file(tmp_path / "nope.json")
        assert not insp.ok
        assert "不存在" in insp.errors[0]
        assert "outputs/wizard" in insp.errors[0]

    def test_malformed_json_is_error(self, tmp_path) -> None:
        f = tmp_path / "train.json"
        f.write_text("{broken", encoding="utf-8")
        insp = inspect_dataset_file(f)
        assert not insp.ok
        assert "JSON 解析失败" in insp.errors[0]

    def test_non_list_top_level_is_error(self, tmp_path) -> None:
        f = tmp_path / "train.json"
        f.write_text('{"instruction": "a"}', encoding="utf-8")
        insp = inspect_dataset_file(f)
        assert not insp.ok
        assert "顶层不是 JSON 数组" in insp.errors[0]

    def test_empty_records_is_error(self, tmp_path) -> None:
        f = tmp_path / "train.json"
        _write_json(f, [])
        insp = inspect_dataset_file(f)
        assert not insp.ok
        assert "数据集为空" in insp.errors[0]

    def test_non_dict_record_is_error(self, tmp_path) -> None:
        f = tmp_path / "train.json"
        f.write_text('["just a string"]', encoding="utf-8")
        insp = inspect_dataset_file(f)
        assert not insp.ok
        assert "第 1 条记录不是 JSON 对象" in insp.errors[0]

    def test_alpaca_records_missing_keys_warn(self, tmp_path) -> None:
        f = tmp_path / "train.json"
        _write_json(f, [ALPACA[0], {"instruction": "只有指令没有输出"}])
        insp = inspect_dataset_file(f)
        assert insp.fmt == "alpaca"
        assert insp.warnings != ()
        assert "第 2 条" in insp.warnings[0]
        assert "instruction/output" in insp.warnings[0]


class TestCheckDatasetForSft:
    def test_hf_name_skips_preflight(self) -> None:
        errors, insp = check_dataset_for_sft("yahma/alpaca-cleaned")
        assert errors == []
        assert insp is None

    def test_alpaca_local_file_passes(self, tmp_path) -> None:
        f = tmp_path / "train.json"
        _write_json(f, ALPACA)
        errors, insp = check_dataset_for_sft(str(f))
        assert errors == []
        assert insp is not None and insp.fmt == "alpaca"

    def test_messages_file_blocked_with_domain_script_hint(self, tmp_path) -> None:
        f = tmp_path / "train.json"
        _write_json(f, MESSAGES)
        errors, _ = check_dataset_for_sft(str(f))
        assert len(errors) == 1
        assert "messages" in errors[0]
        assert "domains/master_data/scripts/train.py" in errors[0]

    def test_unknown_format_blocked_with_keys_shown(self, tmp_path) -> None:
        f = tmp_path / "train.json"
        _write_json(f, [{"prompt": "p", "response": "r"}])
        errors, _ = check_dataset_for_sft(str(f))
        assert len(errors) == 1
        assert "未识别" in errors[0]
        assert "prompt" in errors[0] and "response" in errors[0]
        assert "instruction/output" in errors[0]

    def test_missing_file_blocked(self, tmp_path) -> None:
        errors, insp = check_dataset_for_sft(str(tmp_path / "nope.json"))
        assert len(errors) == 1
        assert "不存在" in errors[0]
        assert insp is not None and not insp.ok

    def test_partial_missing_keys_promoted_to_error(self, tmp_path) -> None:
        # 体检里是 warning（通用陈述）；SFT 消费方读取缺字段必然 KeyError → 阻断
        f = tmp_path / "train.json"
        _write_json(f, [ALPACA[0], {"instruction": "只有指令没有输出"}])
        errors, _ = check_dataset_for_sft(str(f))
        assert len(errors) == 1
        assert "第 2 条" in errors[0]


@pytest.mark.unit
def test_preflight_module_has_no_heavy_imports() -> None:
    """预检在 UI 提交路径上同步执行——必须保持纯标准库、不拉 torch。"""
    import sys

    before = set(sys.modules)
    import src.data.preflight  # noqa: F401  (already imported above; assert no side effects)

    heavy = {"torch", "transformers", "peft", "datasets"}
    new = set(sys.modules) - before
    assert not (new & heavy), f"preflight 引入了重依赖: {new & heavy}"
