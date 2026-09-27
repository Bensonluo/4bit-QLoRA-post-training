"""sources 入口的来源说明:多 Sheet Excel 必须如实标注读取范围,且不改变读取行为。

sheet 选择(按名称或 1 起始序号指定工作表)随多 Sheet 读取一并覆盖:默认读第一个
sheet 的行为不变,显式指定的读取与标注以实测为准;标注随来源对象携带,不共用摘要。
"""

import json
import sys
from io import BytesIO

import pytest
from openpyxl import Workbook

from src.workbench.sources import profile_source, read_source


def _workbook_bytes(*sheets: tuple[str, tuple[str, ...], list[tuple[str, ...]]]) -> bytes:
    """sheets: (名称, 表头行, 数据行) 三元组,按出现顺序写入同一工作簿。"""

    workbook = Workbook()
    for index, (name, header, rows) in enumerate(sheets):
        sheet = workbook.active if index == 0 else workbook.create_sheet()
        sheet.title = name
        sheet.append(header)
        for row in rows:
            sheet.append(row)
    buffer = BytesIO()
    workbook.save(buffer)
    return buffer.getvalue()


_TWO_SHEET_DATA = _workbook_bytes(
    ("工单表", ("编号", "类别"), [("001", "质量")]),
    ("员工表", ("员工号", "部门"), [("E01", "质检")]),
)


def test_single_sheet_excel_profile_has_no_sheet_note():
    """单 sheet 工作簿读取行为不变,profile 不携带 sheet_note;CSV 同样没有。"""
    data = _workbook_bytes(("工单表", ("编号", "类别"), [("001", "质量")]))
    source = read_source("工单.xlsx", data, scope="sample")
    assert source.format == "xlsx"
    assert source.columns == ["编号", "类别"], source.columns
    assert [row.values for row in source.rows] == [{"编号": "001", "类别": "质量"}]
    assert "sheet_note" not in profile_source(source), profile_source(source)

    csv_source = read_source("工单.csv", "编号,类别\n001,质量\n".encode(), scope="sample")
    assert "sheet_note" not in profile_source(csv_source)


def test_multi_sheet_excel_profile_annotates_first_sheet_only():
    """多 sheet 工作簿:读取行为不变(仍只读第一个 sheet),profile 如实标注读取范围。"""
    data = _workbook_bytes(
        ("工单表", ("编号", "类别"), [("001", "质量"), ("002", "物流")]),
        ("员工表", ("员工号", "部门"), [("E01", "质检")]),
    )
    source = read_source("工作簿.xlsx", data, scope="sample")
    # 读取行为不变:列与行全部来自第一个 sheet,第二个 sheet 的列与行不可见
    assert source.columns == ["编号", "类别"], source.columns
    assert [row.values for row in source.rows] == [
        {"编号": "001", "类别": "质量"},
        {"编号": "002", "类别": "物流"},
    ]
    note = profile_source(source)["sheet_note"]
    assert "2 个 sheet" in note, note
    assert "仅读取第一个「工单表」" in note, note
    assert "员工表" in note, note  # 未读取的 sheet 名称如实列出


def test_multi_sheet_note_covers_full_scope_and_caps_long_sheet_lists():
    """全量侧同样标注;其余 sheet 超过 5 个时不逐一罗列,以「等」收尾。"""
    data = _workbook_bytes(
        ("主表", ("编号", "类别"), [("001", "质量")]),
        *[(f"表{i}", ("列",), [("值",)]) for i in range(1, 9)],
    )
    source = read_source("full.xlsx", data, scope="full")
    assert source.scope == "full"
    note = profile_source(source)["sheet_note"]
    assert "9 个 sheet" in note, note
    assert "仅读取第一个「主表」" in note, note
    assert "其余 8 个" in note, note
    assert "表5" in note and "等" in note, note  # 只列前 5 个
    assert "表8" not in note, note


def test_explicit_sheet_by_name_reads_designated_sheet():
    """按名称指定第二个 sheet:列与行全部来自员工表,标注如实写「按指定读取」。"""
    source = read_source("工作簿.xlsx", _TWO_SHEET_DATA, scope="sample", sheet="员工表")
    assert source.sheet == "员工表", source.sheet
    assert source.columns == ["员工号", "部门"], source.columns
    assert [row.values for row in source.rows] == [{"员工号": "E01", "部门": "质检"}]
    note = profile_source(source)["sheet_note"]
    assert "2 个 sheet" in note, note
    assert "按指定读取「员工表」" in note, note
    assert "工单表" in note and "未读取" in note, note  # 未读取的 sheet 名称如实列出


def test_explicit_sheet_by_ordinal_one_based_and_digit_string():
    """序号 1 起始:2=第二个 sheet;CLI 的字符串数字同义;1=第一个(数据同默认,标注写明按指定)。"""
    by_ordinal = read_source("工作簿.xlsx", _TWO_SHEET_DATA, sheet=2)
    by_string = read_source("工作簿.xlsx", _TWO_SHEET_DATA, sheet="2")
    assert by_ordinal.sheet == by_string.sheet == "员工表"
    assert by_ordinal.columns == by_string.columns == ["员工号", "部门"]
    assert [row.values for row in by_string.rows] == [{"员工号": "E01", "部门": "质检"}]

    first = read_source("工作簿.xlsx", _TWO_SHEET_DATA, sheet=1)
    assert first.sheet == "工单表"
    assert first.columns == ["编号", "类别"]
    assert "按指定读取「工单表」" in profile_source(first)["sheet_note"]


def test_explicit_sheet_errors_list_available_sheets():
    """名称不存在/序号越界都明确拒绝,报错列出全部 sheet 名与序号口径,不静默回退。"""
    with pytest.raises(ValueError, match="找不到 sheet「销售表」") as by_name:
        read_source("工作簿.xlsx", _TWO_SHEET_DATA, sheet="销售表")
    assert "工单表、员工表" in str(by_name.value)
    assert "序号从 1 开始" in str(by_name.value)
    with pytest.raises(ValueError, match="sheet 序号 3 超出范围"):
        read_source("工作簿.xlsx", _TWO_SHEET_DATA, sheet=3)
    with pytest.raises(ValueError, match="sheet 序号 9 超出范围"):
        read_source("工作簿.xlsx", _TWO_SHEET_DATA, sheet="9")


def test_sheet_selection_requires_excel():
    """CSV/JSONL 不接受 sheet 选择:明确拒绝而不是静默忽略。"""
    with pytest.raises(ValueError, match="sheet 选择仅对 Excel 文件有效；当前文件是 csv"):
        read_source("工单.csv", "编号\n001\n".encode(), sheet="工单表")
    with pytest.raises(ValueError, match="当前文件是 jsonl"):
        read_source("工单.jsonl", '{"编号": "001"}\n'.encode(), sheet=1)


def test_single_sheet_explicit_selection_reads_it_without_note():
    """单 sheet 工作簿显式指定:正常读取该表;没有未读取的 sheet,profile 无 sheet_note。"""
    data = _workbook_bytes(("工单表", ("编号", "类别"), [("001", "质量")]))
    source = read_source("工单.xlsx", data, sheet="工单表")
    assert source.sheet == "工单表"
    assert source.columns == ["编号", "类别"]
    assert "sheet_note" not in profile_source(source), profile_source(source)


def test_same_workbook_selections_annotated_independently():
    """同一份字节两种指定,标注各自如实——标注随来源对象走,不按文件摘要共用。"""
    first = read_source("工作簿.xlsx", _TWO_SHEET_DATA, sheet=1)
    second = read_source("工作簿.xlsx", _TWO_SHEET_DATA, sheet=2)
    assert first.digest == second.digest  # 同一份字节
    assert "按指定读取「工单表」" in profile_source(first)["sheet_note"]
    assert "按指定读取「员工表」" in profile_source(second)["sheet_note"]


def test_service_create_honors_sheet_and_survives_roundtrip(tmp_path):
    """创建入口(服务层)透传 sheet 选择:会话建在指定的 sheet 上,存档回读后标注仍在。"""
    from src.workbench.intake_service import IntakeService

    service = IntakeService(tmp_path / "intake")
    session = service.create(
        "根据客户首次描述判断售后类别", "工作簿.xlsx", _TWO_SHEET_DATA, sheet="员工表"
    )
    assert session.source.sheet == "员工表"
    assert session.source.columns == ["员工号", "部门"], session.source.columns
    assert "按指定读取「员工表」" in session.profile["sheet_note"]
    reloaded = service.load(session.session_id)
    assert reloaded.source.sheet == "员工表"
    assert "按指定读取「员工表」" in reloaded.profile["sheet_note"]


def test_cli_create_sheet_flag_reads_designated_sheet(tmp_path, monkeypatch, capsys):
    """CLI create --sheet 接线冒烟:--sheet 2 建出的会话读到第二个 sheet 员工表数据。

    详细 CLI 行为回归归 test_data_intake 域;这里只钉住参数透传到服务层的链路。
    """
    from scripts import data_intake as cli

    path = tmp_path / "工作簿.xlsx"
    path.write_bytes(_TWO_SHEET_DATA)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "data_intake.py",
            "--store",
            str(tmp_path / "store"),
            "create",
            "--input",
            str(path),
            "--goal",
            "根据客户首次描述判断售后类别",
            "--sheet",
            "2",
        ],
    )
    assert cli.main() == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["source"]["sheet"] == "员工表"
    assert payload["source"]["columns"] == ["员工号", "部门"]
    assert "按指定读取「员工表」" in payload["profile"]["sheet_note"]
