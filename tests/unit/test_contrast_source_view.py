"""Contrast checks belong to the parsed preview, not just the workbook bytes."""

from io import BytesIO

import pytest
from openpyxl import Workbook

from src.workbench.intake_service import IntakeService
from tests.unit.test_data_intake import analysis


def workbook():
    book = Workbook()
    first = book.active
    first.title = "一月"
    second = book.create_sheet("二月")
    for sheet, prefix in ((first, "甲"), (second, "乙")):
        sheet.append(["编号", "客户描述", "类别", "处理结果"])
        sheet.append([f"{prefix}1", f"{prefix}破损", "质量", "补发"])
        sheet.append([f"{prefix}2", f"{prefix}延迟", "物流", "查询"])
    output = BytesIO()
    book.save(output)
    return output.getvalue()


def switch_sheet(service, session, data, sheet):
    session = service.add_source(
        session.session_id, session.revision, "main", "records.xlsx", data, sheet=sheet
    )
    return service.apply_analysis(session, analysis())


def complete_round(service, session):
    check = service.start_contrast_check(session.session_id, session.revision)
    targets = {row.row_id: row.target for row in session.preview.rows}
    result = service.submit_contrast_check(
        session.session_id,
        check["check_id"],
        {item["row_id"]: targets[item["row_id"]] for item in check["items"]},
    )
    assert result["verdict"] == "verified"
    return check


def test_same_workbook_different_sheet_does_not_reuse_checks(tmp_path):
    service = IntakeService(tmp_path / "intake")
    data = workbook()
    first = service.create("判断客户问题", "records.xlsx", data, sheet="一月")
    first = service.apply_analysis(first, analysis())
    complete_round(service, first)
    complete_round(service, first)
    pending = service.start_contrast_check(first.session_id, first.revision)
    targets = {row.row_id: row.target for row in first.preview.rows}

    second = switch_sheet(service, first, data, "二月")
    assert second.source.digest == first.source.digest
    assert second.preview.rows[0].input != first.preview.rows[0].input
    assert service.contrast_check_status(second.session_id) is None
    with pytest.raises(ValueError, match="核验失效"):
        service.submit_contrast_check(
            second.session_id,
            pending["check_id"],
            {item["row_id"]: targets[item["row_id"]] for item in pending["items"]},
        )
    complete_round(service, second)
    assert service.contrast_check_status(second.session_id)["streak"] == 1

    restored = switch_sheet(service, second, data, "一月")
    assert service.contrast_check_status(restored.session_id)["streak"] == 2
    assert all(
        "甲" in item["input"]
        for round_ in service.contrast_check_status(restored.session_id)["history"]
        for item in round_["items"]
    )


def test_legacy_raw_only_records_are_not_automatically_reused(tmp_path):
    import sqlite3

    service = IntakeService(tmp_path / "intake")
    session = service.create("判断客户问题", "records.xlsx", workbook(), sheet="一月")
    session = service.apply_analysis(session, analysis())
    complete_round(service, session)
    pending = service.start_contrast_check(session.session_id, session.revision)
    legacy_binding = session.source.digest + session.analysis.recipe.model_dump_json()
    with sqlite3.connect(service.database) as connection:
        connection.execute(
            "UPDATE contrast_checks SET binding=? WHERE session_id=?",
            (legacy_binding, session.session_id),
        )
    assert service.contrast_check_status(session.session_id) is None
    with pytest.raises(ValueError, match="核验失效"):
        service.submit_contrast_check(
            session.session_id,
            pending["check_id"],
            {item["row_id"]: item["input"] for item in pending["items"]},
        )
