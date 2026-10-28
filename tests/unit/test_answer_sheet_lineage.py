"""Composed answer sheets identify only corroborated original source rows."""

import csv
import io
from copy import deepcopy

from src.workbench.answer_sheet import answer_sheet_csv, answer_sheet_lineage
from src.workbench.intake_service import IntakeService
from tests.unit.test_answer_sheet_cli import invoke
from tests.unit.test_composed_intake import _workbook_bytes, analysis_for


def make_task(tmp_path):
    workbook = _workbook_bytes(
        ("工单", ("ticket", "description"), [("1", "破损"), ("2", "延迟")]),
        ("标签", ("id", "category"), [("1", "质量"), ("2", "")]),
    )
    service = IntakeService(tmp_path / "intake")
    session = service.create("判断工单类别", "same.xlsx", workbook, sheet="工单")
    session = service.add_source(
        session.session_id, session.revision, "labels", "same.xlsx", workbook, sheet="标签"
    )
    return service, service.apply_analysis(session, analysis_for(session))


def test_composed_cli_sheet_resolves_both_views_of_same_workbook(tmp_path):
    service, session = make_task(tmp_path)
    target = tmp_path / "answers.csv"
    done = invoke(service, "answer-sheet", session.session_id, "--export-csv", target)
    assert done.returncode == 0, done.stderr
    row = list(csv.reader(io.StringIO(target.read_text(encoding="utf-8-sig"))))[1]
    assert row[0] == "r000002"
    assert "需确认答案字段归属" in row[-1]
    assert "main：文件 same.xlsx · 工作表 工单 · 第 3 行" in row[-1]
    assert "labels：文件 same.xlsx · 工作表 标签 · 第 3 行" in row[-1]


def test_ambiguous_legacy_and_mismatched_sources_are_not_guessed(tmp_path):
    _, session = make_task(tmp_path)
    bad = deepcopy(session)
    refs = bad.composition_report["origins"]["r000002"]
    bad.composition_report["origins"]["r000002"] = [
        {"source_digest": refs[0]["source_digest"], "row_id": refs[0]["row_id"]}
    ]
    assert "r000002" not in answer_sheet_lineage(bad)
    bad = deepcopy(session)
    bad.sources["main"] = bad.sources["labels"]
    assert "r000002" not in answer_sheet_lineage(bad)
    bad = deepcopy(session)
    bad.preview.rows[1].original["description"] = "不一致"
    assert "r000002" not in answer_sheet_lineage(bad)
    data = answer_sheet_csv(
        [bad.preview.rows[1]], ["label_category"], lineage=answer_sheet_lineage(bad)
    )
    assert "未记录或不可核验" in data.decode("utf-8-sig")


def test_full_composition_sheet_uses_full_sources_not_sample(tmp_path):
    sample = _workbook_bytes(
        ("工单", ("ticket", "description"), [("1", "破损"), ("2", "延迟")]),
        ("标签", ("id", "category"), [("1", "质量"), ("2", "物流")]),
    )
    full = _workbook_bytes(
        ("工单", ("ticket", "description"), [("3", "维修"), ("4", "送达")]),
        ("标签", ("id", "category"), [("3", "质量"), ("4", "")]),
    )
    service = IntakeService(tmp_path / "intake")
    session = service.create("判断工单类别", "sample.xlsx", sample, sheet="工单")
    session = service.add_source(
        session.session_id, session.revision, "labels", "sample.xlsx", sample, sheet="标签"
    )
    session = service.apply_analysis(session, analysis_for(session))
    session = service.confirm(session.session_id, session.revision)
    session = service.validate_full_sources(
        session.session_id,
        session.revision,
        {"main": ("full.xlsx", full), "labels": ("full.xlsx", full)},
        sheets={"main": "工单", "labels": "标签"},
    )
    target = tmp_path / "full-answers.csv"
    done = invoke(service, "answer-sheet", session.session_id, "--export-csv", target)
    assert done.returncode == 0, done.stderr
    row = list(csv.reader(io.StringIO(target.read_text(encoding="utf-8-sig"))))[1]
    assert "文件 full.xlsx · 工作表 工单 · 第 3 行" in row[-1]
    assert "文件 full.xlsx · 工作表 标签 · 第 3 行" in row[-1]
    assert "sample.xlsx" not in row[-1]
