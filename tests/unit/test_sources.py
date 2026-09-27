"""sources 入口的来源说明:多 Sheet Excel 必须如实标注读取范围,且不改变读取行为。"""

from io import BytesIO

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
