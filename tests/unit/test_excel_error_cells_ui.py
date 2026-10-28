"""The upload page explains Excel errors without replacing or creating tasks."""

from tests.unit.test_data_intake_ui import (
    SHEET_INPUT_LABEL,
    button,
    patch_uploader,
    uploaded_bytes,
)
from tests.unit.test_excel_error_cells import workbook_with

pytest_plugins = ["tests.unit.test_data_intake_ui"]


def test_upload_displays_the_original_error_cell(data_page, monkeypatch):
    service, existing, page = data_page
    holder = {"file": uploaded_bytes(workbook_with(["破损", "#DIV/0!"]), "bad.xlsx")}
    patch_uploader(monkeypatch, "提供 CSV、Excel 或 JSONL", holder)
    page.run()
    next(item for item in page.text_area if item.label == "希望模型完成什么业务工作？").input(
        "判断售后问题"
    )
    next(item for item in page.text_input if item.label == SHEET_INPUT_LABEL).input(" 业务表 ")
    button(page, "读取数据并开始").click().run()
    assert not page.exception
    assert any(
        " 业务表 " in error.value and "B2" in error.value and "#DIV/0!" in error.value
        for error in page.error
    )
    assert [session.session_id for session in service.list_sessions()] == [existing.session_id]
