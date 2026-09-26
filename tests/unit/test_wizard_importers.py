"""数据向导导入层测试：CSV / Excel / JSONL → RawTable + 列映射建议。"""

import pandas as pd
import pytest

from src.data.wizard.importers import (
    RawTable,
    _normalize,
    import_table,
    split_variants,
    suggest_mapping,
)
from src.data.wizard.spec import WizardError


@pytest.fixture()
def sample_csv(tmp_path):
    path = tmp_path / "raw.csv"
    path.write_text(
        "标准名,别名,编码\n阿莫西林胶囊,阿莫西林,Z1098\n布洛芬片,芬必得,Z1099\n", encoding="utf-8"
    )
    return path


class TestImportTable:
    def test_csv_roundtrip(self, sample_csv) -> None:
        table = import_table(sample_csv)
        assert table.columns == ["标准名", "别名", "编码"]
        assert len(table.rows) == 2
        assert table.rows[0]["标准名"] == "阿莫西林胶囊"
        assert table.rows[1]["编码"] == "Z1099"

    def test_csv_strips_whitespace_and_empty(self, tmp_path) -> None:
        path = tmp_path / "raw.csv"
        path.write_text("a,b\n x ,\n", encoding="utf-8")
        table = import_table(path)
        assert table.rows[0]["a"] == "x"
        assert table.rows[0]["b"] is None

    def test_xlsx_import(self, tmp_path) -> None:
        from openpyxl import Workbook

        path = tmp_path / "raw.xlsx"
        wb = Workbook()
        ws = wb.active
        ws.append(["标准名", "编号"])
        ws.append(["阿莫西林", 123])  # Excel 数字列
        wb.save(path)
        table = import_table(path)
        assert table.rows[0]["编号"] == "123"  # 123.0 → "123"

    def test_jsonl_import_with_column_union(self, tmp_path) -> None:
        path = tmp_path / "raw.jsonl"
        path.write_text(
            '{"标准名": "A", "别名": "a"}\n\n{"标准名": "B", "编码": "Z2"}\n',
            encoding="utf-8",
        )
        table = import_table(path)
        assert table.columns == ["标准名", "别名", "编码"]
        # 空行被跳过；两行都有全部列的键
        assert len(table.rows) == 2
        assert table.rows[0]["编码"] is None
        assert table.rows[1]["别名"] is None

    def test_jsonl_bad_line(self, tmp_path) -> None:
        path = tmp_path / "raw.jsonl"
        path.write_text('{"a": 1}\nnot json\n', encoding="utf-8")
        with pytest.raises(WizardError, match="第 2 行"):
            import_table(path)

    def test_jsonl_non_object_line(self, tmp_path) -> None:
        path = tmp_path / "raw.jsonl"
        path.write_text("[1, 2]\n", encoding="utf-8")
        with pytest.raises(WizardError, match="JSON 对象"):
            import_table(path)

    def test_jsonl_only_blank_lines(self, tmp_path) -> None:
        path = tmp_path / "raw.jsonl"
        path.write_text("\n \n", encoding="utf-8")
        with pytest.raises(WizardError, match="没有数据行"):
            import_table(path)

    def test_missing_file(self, tmp_path) -> None:
        with pytest.raises(WizardError, match="文件不存在"):
            import_table(tmp_path / "nope.csv")

    def test_unsupported_suffix(self, tmp_path) -> None:
        path = tmp_path / "raw.txt"
        path.write_text("x", encoding="utf-8")
        with pytest.raises(WizardError, match="不支持的文件格式"):
            import_table(path)

    def test_empty_csv_only_header(self, tmp_path) -> None:
        path = tmp_path / "raw.csv"
        path.write_text("a,b\n", encoding="utf-8")
        with pytest.raises(WizardError, match="没有数据行"):
            import_table(path)

    def test_pandas_parse_error_wrapped(self, tmp_path, monkeypatch) -> None:
        def boom(*args, **kwargs):
            raise ValueError("bad cell")

        monkeypatch.setattr(pd, "read_csv", boom)
        path = tmp_path / "raw.csv"
        path.write_text("a\n1\n", encoding="utf-8")
        with pytest.raises(WizardError, match="解析文件失败"):
            import_table(path)

    def test_excel_missing_dependency(self, tmp_path, monkeypatch) -> None:
        def boom(*args, **kwargs):
            raise ImportError("No module named 'openpyxl'")

        monkeypatch.setattr(pd, "read_excel", boom)
        path = tmp_path / "raw.xlsx"
        path.write_bytes(b"fake")
        with pytest.raises(WizardError, match="openpyxl"):
            import_table(path)


class TestNormalize:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (None, None),
            (float("nan"), None),
            (123.0, "123"),
            (1.5, "1.5"),
            ("  x ", "x"),
            ("", None),
            ("   ", None),
            (7, "7"),
        ],
    )
    def test_values(self, value, expected) -> None:
        assert _normalize(value) == expected


class TestSplitVariants:
    def test_none(self) -> None:
        assert split_variants(None) == []

    def test_mixed_delimiters_dedup_order(self) -> None:
        assert split_variants("北医三院、北大三院;协和，北医三院|华西") == [
            "北医三院",
            "北大三院",
            "协和",
            "华西",
        ]


class TestSuggestMapping:
    def test_happy_path(self) -> None:
        mapping = suggest_mapping(["药品标准名", "录入别名", "国药准字编码", "类型"])
        assert mapping.standard_name == "药品标准名"
        assert mapping.query == "录入别名"
        assert mapping.code == "国药准字编码"
        assert mapping.entity_type == "类型"
        assert mapping.variants is None

    def test_english_columns(self) -> None:
        mapping = suggest_mapping(["standard_name", "alias", "variants", "code"])
        assert mapping.standard_name == "standard_name"
        assert mapping.query == "alias"
        assert mapping.variants == "variants"
        assert mapping.code == "code"

    def test_fallback_to_first_column(self) -> None:
        mapping = suggest_mapping(["colA", "colB"])
        assert mapping.standard_name == "colA"
        assert mapping.query is None

    def test_one_column_reused_guard(self) -> None:
        # "查询编码" 同时含查询/编码提示：query 先占，code 不得复用同一列
        mapping = suggest_mapping(["标准名", "查询编码"])
        assert mapping.query == "查询编码"
        assert mapping.code is None

    def test_no_columns_raises(self) -> None:
        with pytest.raises(WizardError, match="没有任何列"):
            suggest_mapping([])


class TestRawTableShape:
    def test_frozen(self) -> None:
        import dataclasses

        table = RawTable(source="x", columns=["a"], rows=[{"a": "1"}])
        with pytest.raises(dataclasses.FrozenInstanceError):
            table.source = "y"  # type: ignore[misc]
