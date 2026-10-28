"""Malformed Excel sources fail clearly without changing intake records."""

import subprocess
import sys
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import pandas as pd
import pytest
from openpyxl import Workbook

from src.workbench.intake_service import IntakeService
from src.workbench.sources import read_source
from tests.unit.test_full_data import approved


def workbook_bytes():
    book = Workbook()
    sheet = book.active
    sheet.title = "工单"
    sheet.append(["编号", "客户描述", "类别", "处理结果"])
    sheet.append(["1", "物品损坏", "质量", "补发"])
    buffer = BytesIO()
    book.save(buffer)
    return buffer.getvalue()


def corrupt_xml(name, *, remove=False):
    result = BytesIO()
    with ZipFile(BytesIO(workbook_bytes())) as source, ZipFile(result, "w") as target:
        for member in source.infolist():
            if member.filename == name and remove:
                continue
            value = b"<broken" if member.filename == name else source.read(member)
            target.writestr(member, value)
    return result.getvalue()


@pytest.mark.parametrize(
    "data",
    [
        b"PK\x03\x04truncated",
        corrupt_xml("xl/workbook.xml"),
        corrupt_xml("xl/worksheets/sheet1.xml"),
        corrupt_xml("xl/_rels/workbook.xml.rels", remove=True),
        corrupt_xml("xl/worksheets/sheet1.xml", remove=True),
        corrupt_xml("xl/workbook.xml", remove=True),
    ],
)
def test_corrupt_excel_is_rejected_as_actionable_user_error(data):
    with pytest.raises(ValueError, match="Excel.*(损坏|无效|没有可读取)") as caught:
        read_source("broken.xlsx", data)
    assert "traceback" not in str(caught.value).lower()


def test_invalid_full_upload_does_not_change_existing_session(tmp_path):
    service = IntakeService(tmp_path / "intake")
    before = approved(service)
    with pytest.raises(ValueError, match="Excel.*损坏"):
        service.validate_full_data(
            before.session_id, before.revision, "bad.xlsx", b"PK\x03\x04truncated"
        )
    assert service.load(before.session_id).model_dump() == before.model_dump()
    assert not (service.root / before.session_id / "full").exists()


def test_cli_reports_corrupt_workbook_without_traceback_or_new_task(tmp_path):
    data = tmp_path / "corrupt.xlsx"
    data.write_bytes(b"PK\x03\x04truncated")
    store = tmp_path / "intake"
    script = Path(__file__).resolve().parents[2] / "scripts/data_intake.py"
    process = subprocess.run(
        [
            sys.executable,
            str(script),
            "--store",
            str(store),
            "create",
            "--input",
            str(data),
            "--goal",
            "分类",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert process.returncode == 2
    assert "Excel 文件损坏或格式无效" in process.stderr
    assert "Traceback" not in process.stderr
    assert IntakeService(store).list_sessions() == []


def test_missing_dependency_keeps_distinct_install_guidance(monkeypatch):
    def missing(*args, **kwargs):
        raise ImportError("test missing engine")

    monkeypatch.setattr(pd, "ExcelFile", missing)
    with pytest.raises(ValueError, match="缺少对应引擎.*openpyxl.*xlrd"):
        read_source("bad.xlsx", b"test")


@pytest.mark.parametrize("extension", ["xlsx", "xls"])
def test_empty_excel_is_rejected_before_engine_loading(extension, monkeypatch):
    def must_not_load(*args, **kwargs):
        pytest.fail("empty workbook should be rejected before loading its engine")

    monkeypatch.setattr(pd, "ExcelFile", must_not_load)
    with pytest.raises(ValueError, match="Excel 文件为空"):
        read_source(f"empty.{extension}", b"")


def test_filesystem_or_programming_errors_are_not_misclassified(monkeypatch):
    def denied(*args, **kwargs):
        raise PermissionError("filesystem denied")

    monkeypatch.setattr(pd, "ExcelFile", denied)
    with pytest.raises(PermissionError, match="filesystem denied"):
        read_source("bad.xlsx", b"test")

    def faulty(*args, **kwargs):
        raise TypeError("programming error")

    monkeypatch.setattr(pd, "ExcelFile", faulty)
    with pytest.raises(TypeError, match="programming error"):
        read_source("bad.xlsx", b"test")


def test_nonexistent_sheet_retains_specific_feedback():
    with pytest.raises(ValueError, match="找不到 sheet.*工单"):
        read_source("source.xlsx", workbook_bytes(), sheet="不存在")
