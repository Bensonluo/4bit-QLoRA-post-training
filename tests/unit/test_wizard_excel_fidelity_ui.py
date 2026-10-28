"""The wizard exposes spreadsheet errors without replacing the previous table."""

from pathlib import Path
from types import SimpleNamespace

from streamlit.testing.v1 import AppTest

from tests.unit.test_wizard_excel_fidelity import workbook_path
from tests.unit.test_wizard_upload_session_ui import CSV

ROOT = Path(__file__).resolve().parents[2]


def test_error_cell_upload_keeps_previous_table(monkeypatch, tmp_path):
    from ui import config, runtime

    monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(runtime, "DISPLAY_ONLY", False)
    holder = {"upload": SimpleNamespace(name="good.csv", file_id="1", getvalue=lambda: CSV)}
    monkeypatch.setattr(runtime, "action_upload", lambda *args, **kwargs: holder["upload"])
    page = AppTest.from_file(str(ROOT / "ui/views/05_Data_Wizard.py"), default_timeout=30).run()
    assert not page.exception
    before = page.session_state["wizard_table"]
    path = workbook_path(tmp_path, [["标准名", "别名"], ["甲", "#DIV/0!"]])
    holder["upload"] = SimpleNamespace(name="bad.xlsx", file_id="2", getvalue=path.read_bytes)
    page.run()
    assert not page.exception, [item.message for item in page.exception]
    assert any("原始资料" in item.value and "B2" in item.value for item in page.error)
    assert page.session_state["wizard_table"] == before
