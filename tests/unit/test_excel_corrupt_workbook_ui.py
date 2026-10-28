"""The data upload UI remains usable after a malformed Excel file."""

from tests.unit.test_data_intake_ui import button, patch_uploader, uploaded_bytes

pytest_plugins = ["tests.unit.test_data_intake_ui"]


def test_upload_rejects_corrupt_excel_without_creating_task(data_page, monkeypatch):
    service, existing, page = data_page
    holder = {"file": uploaded_bytes(b"PK\x03\x04truncated", "bad.xlsx")}
    patch_uploader(monkeypatch, "提供 CSV、Excel 或 JSONL", holder)
    page.run()
    next(item for item in page.text_area if item.label == "希望模型完成什么业务工作？").input(
        "判断售后问题"
    )
    button(page, "读取数据并开始").click().run()
    assert not page.exception, [item.message for item in page.exception]
    assert any("Excel 文件损坏或格式无效" in item.value for item in page.error)
    assert [session.session_id for session in service.list_sessions()] == [existing.session_id]
