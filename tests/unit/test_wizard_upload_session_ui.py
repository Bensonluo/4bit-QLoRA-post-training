"""Uploaded tables should load once and remain usable through reruns."""

from pathlib import Path
from types import SimpleNamespace

from streamlit.proto.Common_pb2 import FileURLs
from streamlit.runtime.uploaded_file_manager import UploadedFile, UploadedFileRec
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[2]
PAGE = ROOT / "ui/views/05_Data_Wizard.py"
CSV = "标准名,别名,编码\n甲,甲别名,A\n乙,乙别名,B\n丙,丙别名,C\n".encode()


def test_upload_does_not_reload_on_each_rerun(monkeypatch, tmp_path):
    from src.data.wizard import importers
    from ui import config, runtime

    monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(runtime, "DISPLAY_ONLY", False)
    upload = UploadedFile(UploadedFileRec("file-1", "sample.csv", "text/csv", CSV), FileURLs())
    monkeypatch.setattr(runtime, "action_upload", lambda *args, **kwargs: upload)
    original = importers.import_table
    reads = []

    def observed(path):
        reads.append(Path(path))
        return original(path)

    monkeypatch.setattr(importers, "import_table", observed)
    page = AppTest.from_file(str(PAGE), default_timeout=30).run()
    assert not page.exception, [e.message for e in page.exception]
    assert len(reads) == 1
    page.run()
    assert not page.exception, [e.message for e in page.exception]
    assert len(reads) == 1, "the retained uploader must not re-import on every interaction"
    assert not reads[0].exists(), "the uploaded file must not remain in a temporary directory"
    assert page.session_state["wizard_table"].source == "sample.csv"


def test_replacement_demo_and_reupload_preserve_the_chosen_source(monkeypatch, tmp_path):
    from src.data.wizard import importers
    from ui import config, runtime

    monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(runtime, "DISPLAY_ONLY", False)
    holder = {"upload": SimpleNamespace(name="sample.csv", file_id="file-1", getvalue=lambda: CSV)}
    monkeypatch.setattr(runtime, "action_upload", lambda *args, **kwargs: holder["upload"])
    original = importers.import_table
    reads = []

    def observed(path):
        reads.append(Path(path))
        return original(path)

    monkeypatch.setattr(importers, "import_table", observed)
    page = AppTest.from_file(str(PAGE), default_timeout=30).run()
    assert not page.exception, [e.message for e in page.exception]
    next(b for b in page.button if "试生成" in b.label).click().run()
    assert not page.exception, [e.message for e in page.exception]
    assert page.session_state["wizard_preview"]["records"]
    page.run()
    assert page.session_state["wizard_preview"]["records"]
    assert len(reads) == 1

    page.session_state["wizard_report"] = "old report"
    replacement = CSV.replace("甲别名".encode(), "新别名".encode())
    holder["upload"] = SimpleNamespace(
        name="sample.csv", file_id="file-2", getvalue=lambda: replacement
    )
    page.run()
    assert not page.exception, [e.message for e in page.exception]
    assert page.session_state["wizard_table"].rows[0]["别名"] == "新别名"
    assert "wizard_preview" not in page.session_state
    assert "wizard_report" not in page.session_state
    assert len(reads) == 2

    next(b for b in page.button if "通用演示" in b.label).click().run()
    assert not page.exception, [e.message for e in page.exception]
    assert page.session_state["wizard_source"] == "demo_suppliers.csv"
    page.run()
    assert page.session_state["wizard_source"] == "demo_suppliers.csv"
    assert len(reads) == 3

    holder["upload"] = None
    page.run()
    holder["upload"] = SimpleNamespace(
        name="sample.csv", file_id="file-2", getvalue=lambda: replacement
    )
    page.run()
    assert page.session_state["wizard_source"] == "sample.csv"
    assert len(reads) == 4
    assert all(not path.exists() for path in reads)


def test_upload_without_file_identifier_uses_content_identity(monkeypatch, tmp_path):
    from src.data.wizard import importers
    from ui import config, runtime

    monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(runtime, "DISPLAY_ONLY", False)
    holder = {"data": CSV}
    upload = SimpleNamespace(name="sample.csv", getvalue=lambda: holder["data"])
    monkeypatch.setattr(runtime, "action_upload", lambda *args, **kwargs: upload)
    original = importers.import_table
    reads = []

    def observed(path):
        reads.append(path)
        return original(path)

    monkeypatch.setattr(importers, "import_table", observed)
    page = AppTest.from_file(str(PAGE), default_timeout=30).run()
    page.run()
    assert len(reads) == 1
    holder["data"] = CSV.replace("甲别名".encode(), "新别名".encode())
    page.run()
    assert not page.exception
    assert len(reads) == 2
    assert page.session_state["wizard_table"].rows[0]["别名"] == "新别名"


def test_upload_filename_cannot_escape_temporary_directory(monkeypatch, tmp_path):
    import tempfile

    from ui import config, runtime

    monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(runtime, "DISPLAY_ONLY", False)
    upload = SimpleNamespace(name="../outside.csv", file_id="file-1", getvalue=lambda: CSV)
    monkeypatch.setattr(runtime, "action_upload", lambda *args, **kwargs: upload)
    original = tempfile.TemporaryDirectory

    def temporary_directory(*args, **kwargs):
        return original(*args, dir=tmp_path, **kwargs)

    monkeypatch.setattr(tempfile, "TemporaryDirectory", temporary_directory)
    page = AppTest.from_file(str(PAGE), default_timeout=30).run()
    assert not page.exception, [e.message for e in page.exception]
    assert page.session_state["wizard_source"] == "outside.csv"
    assert not (tmp_path / "outside.csv").exists()
    assert not list(tmp_path.glob("tunesmith_wizard_*"))


def test_invalid_upload_reports_error_without_replacing_previous_table(monkeypatch, tmp_path):
    from ui import config, runtime

    monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(runtime, "DISPLAY_ONLY", False)
    holder = {"upload": SimpleNamespace(name="sample.csv", file_id="file-1", getvalue=lambda: CSV)}
    monkeypatch.setattr(runtime, "action_upload", lambda *args, **kwargs: holder["upload"])
    page = AppTest.from_file(str(PAGE), default_timeout=30).run()
    assert not page.exception
    holder["upload"] = SimpleNamespace(
        name="broken.jsonl", file_id="file-2", getvalue=lambda: b"{invalid}"
    )
    page.run()
    assert not page.exception, [e.message for e in page.exception]
    assert any("JSONL 第 1 行" in item.value for item in page.error)
    assert page.session_state["wizard_source"] == "sample.csv"
