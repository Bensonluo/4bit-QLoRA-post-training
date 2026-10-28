"""Real legacy XLS files: valid data, corruption, and scoped reader warnings."""

import subprocess
import sys
from io import BytesIO
from pathlib import Path

import pytest

from src.workbench.intake_service import IntakeService
from src.workbench.sources import read_source
from tests.unit.test_data_intake_ui import button, patch_uploader, uploaded_bytes

pytest.importorskip("xlrd")
xlwt = pytest.importorskip("xlwt")

pytest_plugins = ["tests.unit.test_data_intake_ui"]


def xls_bytes(*, error=False):
    book = xlwt.Workbook()
    sheet = book.add_sheet("业务数据")
    for column, value in enumerate(("编号", "客户描述", "类别", "处理结果")):
        sheet.write(0, column, value)
    sheet.write(1, 0, "1")
    sheet.write(1, 1, "破损")
    if error:
        sheet.row(1).set_cell_error(2, "#DIV/0!")
    else:
        sheet.write(1, 2, "质量")
    sheet.write(1, 3, "补发")
    result = BytesIO()
    book.save(result)
    return result.getvalue()


def test_real_xls_reads_values_and_detects_error_cells():
    result = read_source("valid.xls", xls_bytes())
    assert result.sheet == "业务数据"
    assert result.rows[0].values["类别"] == "质量"
    with pytest.raises(ValueError, match=r"业务数据.*C2.*#DIV/0!"):
        read_source("error.xls", xls_bytes(error=True))


@pytest.mark.parametrize("kind", ["empty", "junk", "header", "early", "late"])
def test_real_corrupt_xls_fails_cleanly_without_stdout(kind, capsys):
    raw = xls_bytes()
    data = {
        "empty": b"",
        "junk": b"not excel",
        "header": raw[:8],
        "early": raw[:100],
        "late": raw[:-500],
    }[kind]
    with pytest.raises(ValueError, match="Excel 文件(为空|损坏或格式无效)"):
        read_source("bad.xls", data)
    assert capsys.readouterr().out == ""


def test_real_xls_with_reader_warning_keeps_values_and_discloses_issue(capsys):
    source = read_source("has-trailing-byte.xls", xls_bytes() + b"x")
    assert source.rows[0].values["类别"] == "质量"
    assert "读取引擎报告文件结构异常" in source.sheet_note
    assert "WARNING" not in source.sheet_note
    assert capsys.readouterr().out == ""


def test_cli_handles_corrupt_real_xls_without_stdout_noise(tmp_path):
    source = tmp_path / "bad.xls"
    source.write_bytes(xls_bytes()[:100])
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
            str(source),
            "--goal",
            "分类",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert process.returncode == 2
    assert process.stdout == ""
    assert "Excel 文件损坏或格式无效" in process.stderr
    assert "Traceback" not in process.stderr
    assert IntakeService(store).list_sessions() == []


def test_page_rejects_corrupt_real_xls_without_creating_task(data_page, monkeypatch):
    service, existing, page = data_page
    holder = {"file": uploaded_bytes(xls_bytes()[:100], "bad.xls")}
    patch_uploader(monkeypatch, "提供 CSV、Excel 或 JSONL", holder)
    page.run()
    next(item for item in page.text_area if item.label == "希望模型完成什么业务工作？").input(
        "判断售后问题"
    )
    button(page, "读取数据并开始").click().run()
    assert not page.exception, [item.message for item in page.exception]
    assert any("Excel 文件损坏或格式无效" in item.value for item in page.error)
    assert [session.session_id for session in service.list_sessions()] == [existing.session_id]
