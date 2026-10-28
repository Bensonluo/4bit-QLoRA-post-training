"""Excel sheet identity must survive user input and source reuse verbatim."""

import json

import pytest

from src.workbench.intake_service import IntakeService
from src.workbench.sources import read_source
from tests.unit.test_data_intake import analysis
from tests.unit.test_data_intake_ui import (
    SHEET_INPUT_LABEL,
    _workbook_bytes,
    button,
    patch_uploader,
    uploaded_bytes,
)
from tests.unit.test_full_data import approved
from tests.unit.test_full_data_cli import invoke
from tests.unit.test_multisource_cli import FULL_MAIN, composition_session

pytest_plugins = ["tests.unit.test_data_intake_ui"]


@pytest.fixture
def workbook():
    header = ("编号", "客户描述", "类别", "处理结果")
    return _workbook_bytes(
        ("数据", header, [("a", "普通工作表", "物流", "查询")]),
        (
            " 数据 ",
            header,
            [
                ("b", "带空格工作表", "质量", "补发"),
                ("c", "第二条样例", "物流", "查询"),
            ],
        ),
        (" 2 ", header, [("c", "数字名称工作表", "咨询", "回复")]),
        ("   ", header, [("d", "纯空格工作表", "售后", "处理")]),
    )


def test_explicit_sheet_name_exact_match_precedes_trimming_or_ordinal(workbook):
    spaced = read_source("book.xlsx", workbook, sheet=" 数据 ")
    assert spaced.sheet == " 数据 "
    assert spaced.rows[0].values["客户描述"] == "带空格工作表"
    assert (
        read_source("book.xlsx", workbook, sheet="数据").rows[0].values["客户描述"] == "普通工作表"
    )
    assert read_source("book.xlsx", workbook, sheet=" 2 ").sheet == " 2 "
    assert read_source("book.xlsx", workbook, sheet=2).sheet == " 数据 "
    assert read_source("book.xlsx", workbook, sheet="2").sheet == " 数据 "
    assert read_source("book.xlsx", workbook, sheet="  2").sheet == " 数据 "
    assert read_source("book.xlsx", workbook, sheet=" 数据").sheet == "数据"
    assert read_source("book.xlsx", workbook, sheet="   ").sheet == "   "
    assert read_source("book.xlsx", workbook).sheet == "数据"


def test_full_source_reuse_retains_exact_sheet(workbook, tmp_path):
    service = IntakeService(tmp_path / "intake")
    session = service.create(
        "判断售后问题类型", "book.xlsx", workbook, scope="full", sheet=" 数据 "
    )
    assert session.source.sheet == " 数据 "
    session = service.apply_analysis(session, analysis())
    session = service.confirm(session.session_id, session.revision)
    session = service.validate_full_data(session.session_id, session.revision)
    assert session.full_data.source.sheet == " 数据 "
    assert session.full_data.preview.rows[0].target == "质量"
    assert service.load(session.session_id).full_data.source.sheet == " 数据 "


def test_create_page_preserves_exact_sheet_input(data_page, monkeypatch, workbook):
    service, existing, page = data_page
    holder = {"file": uploaded_bytes(workbook, "book.xlsx")}
    patch_uploader(monkeypatch, "提供 CSV、Excel 或 JSONL", holder)
    page.run()
    next(t for t in page.text_area if t.label == "希望模型完成什么业务工作？").input(
        "判断售后问题类型"
    )
    next(t for t in page.text_input if t.label == SHEET_INPUT_LABEL).input(" 数据 ")
    button(page, "读取数据并开始").click().run()
    assert not page.exception
    created = next(s for s in service.list_sessions() if s.session_id != existing.session_id)
    assert created.source.sheet == " 数据 "
    assert created.source.rows[0].values["类别"] == "质量"


def test_add_source_page_preserves_exact_sheet_input(data_page, monkeypatch, workbook):
    service, session, page = data_page
    holder = {"file": uploaded_bytes(workbook, "book.xlsx")}
    patch_uploader(monkeypatch, "上传补充原始资料", holder)
    page.run()
    page.selectbox(key="intake_select").select(session.session_id).run()
    next(t for t in page.text_input if t.label == "补充资料名称").input("labels")
    next(t for t in page.text_input if t.label == SHEET_INPUT_LABEL).input(" 数据 ")
    button(page, "保存补充资料").click().run()
    assert not page.exception
    source = service.load(session.session_id).sources["labels"]
    assert source.sheet == " 数据 "
    assert source.rows[0].values["类别"] == "质量"


def test_full_upload_page_preserves_exact_sheet_input(data_page, monkeypatch, workbook):
    service, _, page = data_page
    session = approved(service)
    holder = {"file": uploaded_bytes(workbook, "book.xlsx")}
    patch_uploader(monkeypatch, "提供本次任务的全量文件", holder)
    page.run()
    page.selectbox(key="intake_select").select(session.session_id).run()
    next(t for t in page.text_input if t.label == SHEET_INPUT_LABEL).input(" 数据 ")
    button(page, "按已确认方案验证全量数据").click().run()
    assert not page.exception
    report = service.load(session.session_id).full_data
    assert report.sources["main"].sheet == " 数据 "
    assert report.preview.rows[0].target == "质量"


def labels_workbook():
    return _workbook_bytes(
        ("类别", ("id", "category"), [("10", "错误"), ("11", "错误"), ("12", "错误")]),
        (" 类别 ", ("id", "category"), [("10", "质量"), ("11", "物流"), ("12", "质量")]),
        ("   ", ("id", "category"), [("10", "质量"), ("11", "物流"), ("12", "质量")]),
    )


def test_composed_full_page_preserves_exact_sheet_input(data_page, monkeypatch):
    service, _, page = data_page
    session = composition_session(service)
    main_holder = {"file": uploaded_bytes(FULL_MAIN, "main.csv")}
    label_holder = {"file": uploaded_bytes(labels_workbook(), "labels.xlsx")}
    patch_uploader(monkeypatch, "全量原始资料：main", main_holder)
    patch_uploader(monkeypatch, "全量原始资料：labels", label_holder)
    page.run()
    page.selectbox(key="intake_select").select(session.session_id).run()
    next(t for t in page.text_input if t.key == f"full_sheet_{session.session_id}_labels").input(
        " 类别 "
    )
    button(page, "按组合方案验证全部全量资料").click().run()
    assert not page.exception
    report = service.load(session.session_id).full_data
    assert report.sources["labels"].sheet == " 类别 "
    assert [row.target for row in report.preview.rows] == ["质量", "物流", "质量"]


@pytest.mark.parametrize("sheet_name", [" 类别 ", "   "])
def test_full_sources_cli_preserves_exact_sheet_value(tmp_path, sheet_name):
    service = IntakeService(tmp_path / "intake")
    session = composition_session(service)
    main_file, labels_file = tmp_path / "main.csv", tmp_path / "labels.xlsx"
    main_file.write_bytes(FULL_MAIN)
    labels_file.write_bytes(labels_workbook())
    result = invoke(
        service,
        "full-sources",
        session.session_id,
        "--revision",
        session.revision,
        "--source",
        f"main={main_file}",
        "--source",
        f"labels={labels_file}",
        "--sheet",
        f"labels={sheet_name}",
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["full_data"]["sources"]["labels"]["sheet"] == sheet_name
    assert [row["target"] for row in payload["full_data"]["preview"]["rows"]] == [
        "质量",
        "物流",
        "质量",
    ]
