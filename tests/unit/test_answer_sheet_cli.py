"""待补答案清单 CLI(介入点 10):把「找人填写」变成可交接的命令。

stdout 纯 JSON(口径/行数/行ID/答案字段),stderr 人话(清单口径行+三条
规则+导出行);无预览如实报错退出,没有缺答案行不写文件——诚实降级,
不为了「命令成功」编造清单。
"""

from __future__ import annotations

import csv
import io
import json
import subprocess
import sys
from pathlib import Path

from src.workbench.intake_service import IntakeService
from tests.unit.test_data_intake import CSV, analysis

CLI = Path(__file__).resolve().parents[2] / "scripts/data_intake.py"

MISSING = "编号,客户描述,类别,处理结果\n001,杯子破损,质量,补发\n002,物流未更新,,查询\n".encode()
FULL_MISSING = "编号,客户描述,类别,处理结果\n100,收到破杯,质量,补发\n101,快递太慢,,查询\n".encode()


def _store(tmp_path, csv=CSV):
    service = IntakeService(tmp_path / "intake")
    session = service.create("根据客户首次描述预测类别", "工单.csv", csv)
    session = service.apply_analysis(session, analysis())
    return service, session


def invoke(service, *args):
    return subprocess.run(
        [sys.executable, str(CLI), "--store", str(service.root), *map(str, args)],
        text=True,
        capture_output=True,
        check=False,
    )


def test_answer_sheet_cli_exports_handoff_sheet_for_business_filler(tmp_path):
    """样例口径:JSON 给行ID 与答案字段,CSV 保留原始值,规则同源。"""
    service, session = _store(tmp_path, MISSING)
    target = tmp_path / "sheet.csv"
    done = invoke(service, "answer-sheet", session.session_id, "--export-csv", target)
    assert done.returncode == 0, done.stderr
    result = json.loads(done.stdout)
    assert result["scope"] == "sample"
    assert result["missing_count"] == 1
    assert result["row_ids"] == ["r000002"]
    assert result["answer_fields"] == ["类别"]
    # stderr:清单口径行先说清按哪份预览、几行;随后三条规则与页面同源。
    assert "清单口径：样例转换预览，待补 1 行。" in done.stderr
    assert "只填「待填答案」列" in done.stderr
    assert "不要凭猜测填" in done.stderr
    assert "比编一个答案更有价值" in done.stderr
    assert "待补答案清单已导出" in done.stderr
    # 填写表:BOM+表头+行ID 与题目输入原样,待填格为空——不含数据答案。
    data = target.read_bytes()
    assert data.startswith(b"\xef\xbb\xbf")
    lines = data.decode("utf-8-sig").strip("\r\n").splitlines()
    assert lines[0] == "行ID,题目输入（模型将看到的内容）,待填答案（类别）,原始记录位置"
    assert lines[1].startswith("r000002,")
    record = list(csv.reader(io.StringIO(data.decode("utf-8-sig"))))[1]
    assert record[2] == ""
    assert record[3] == "文件 工单.csv · 第 3 行（记录起始行）"
    assert "物流未更新" in lines[1]


def test_answer_sheet_cli_no_missing_is_honest_and_writes_no_file(tmp_path):
    """没有缺答案行:不写文件、明说无需导出,不用空清单冒充交接物。"""
    service, session = _store(tmp_path, CSV)
    target = tmp_path / "sheet.csv"
    done = invoke(service, "answer-sheet", session.session_id, "--export-csv", target)
    assert done.returncode == 0, done.stderr
    result = json.loads(done.stdout)
    assert result["missing_count"] == 0 and result["row_ids"] == []
    assert "当前没有缺答案的行，无需导出清单。" in done.stderr
    assert "只填「待填答案」列" not in done.stderr
    assert not target.exists()


def test_answer_sheet_cli_requires_analysis_first(tmp_path):
    """没有真实转换预览:如实报错退出(exit 2),不凭原始表猜缺答案行。"""
    service = IntakeService(tmp_path / "intake")
    session = service.create("根据客户首次描述预测类别", "工单.csv", MISSING)
    done = invoke(service, "answer-sheet", session.session_id)
    assert done.returncode == 2
    assert "先运行 analyze" in done.stderr
    assert "baseline-analyze" in done.stderr, "零密钥路径同样能生成预览,必须一并点名"


def test_answer_sheet_cli_uses_current_full_preview_when_available(tmp_path):
    """全量预览存在且未失效:清单口径切到全量——缺答案行在全量文件里找。"""
    service, session = _store(tmp_path, CSV)
    session = service.confirm(session.session_id, session.revision)
    session = service.validate_full_data(
        session.session_id, session.revision, "full.csv", FULL_MISSING
    )
    assert session.full_data is not None and session.full_data.preview is not None
    target = tmp_path / "full-sheet.csv"
    done = invoke(service, "answer-sheet", session.session_id, "--export-csv", target)
    assert done.returncode == 0, done.stderr
    result = json.loads(done.stdout)
    assert result["scope"] == "full"
    assert result["missing_count"] == 1
    assert result["row_ids"] == ["r000002"]  # 全量文件自己的行ID 空间
    assert "清单口径：全量真实转换预览" in done.stderr
    record = list(csv.reader(io.StringIO(target.read_text(encoding="utf-8-sig"))))[1]
    assert record[-1] == "文件 full.csv · 第 3 行（记录起始行）"


def test_sheet_locates_original_record_after_blank_and_multiline_rows(tmp_path):
    raw = (
        '编号,客户描述,类别,处理结果\n001,"杯子\n破损",质量,补发\n\n002,物流未更新,,查询\n'
    ).encode()
    service, session = _store(tmp_path, raw)
    target = tmp_path / "sheet.csv"
    done = invoke(service, "answer-sheet", session.session_id, "--export-csv", target)
    assert done.returncode == 0, done.stderr
    row = list(csv.reader(io.StringIO(target.read_text(encoding="utf-8-sig"))))[1]
    assert row[0] == "r000002"
    assert row[-1] == "文件 工单.csv · 第 5 行（记录起始行）"
    assert "清单不能直接上传回填" in done.stderr


def test_excel_sheet_location_survives_cli_handoff(tmp_path):
    from openpyxl import Workbook

    workbook = Workbook()
    workbook.active.title = "其他"
    sheet = workbook.create_sheet(" 数据 ")
    sheet.append(["编号", "客户描述", "类别", "处理结果"])
    sheet.append(["001", "杯子破损", "质量", "补发"])
    sheet.append(["002", "物流未更新", "", "查询"])
    data = io.BytesIO()
    workbook.save(data)
    service = IntakeService(tmp_path / "intake")
    session = service.create("判断类别", "book.xlsx", data.getvalue(), sheet=" 数据 ")
    session = service.apply_analysis(session, analysis())
    target = tmp_path / "sheet.csv"
    done = invoke(service, "answer-sheet", session.session_id, "--export-csv", target)
    assert done.returncode == 0, done.stderr
    row = list(csv.reader(io.StringIO(target.read_text(encoding="utf-8-sig"))))[1]
    assert row[-1] == "文件 book.xlsx · 工作表  数据  · 第 3 行（记录起始行）"


def test_composed_preview_does_not_claim_synthetic_row_is_original_location(tmp_path):
    from src.workbench.answer_sheet import answer_sheet_csv, answer_sheet_source

    _, session = _store(tmp_path, MISSING)
    session.composition_report = {"origins": {}}
    source = answer_sheet_source(session)
    assert source is None
    raw = answer_sheet_csv([session.preview.rows[1]], ["类别"], source=source)
    row = list(csv.reader(io.StringIO(raw.decode("utf-8-sig"))))[1]
    assert row[-1] == "未记录或不可核验，请查看来源血缘"


def test_mismatched_preview_does_not_claim_an_original_row(tmp_path):
    from copy import deepcopy

    from src.workbench.answer_sheet import answer_sheet_csv, answer_sheet_source

    _, session = _store(tmp_path, MISSING)
    changed = deepcopy(session.preview.rows[1])
    changed.original["客户描述"] = "另一份资料"
    raw = answer_sheet_csv([changed], ["类别"], source=answer_sheet_source(session))
    row = list(csv.reader(io.StringIO(raw.decode("utf-8-sig"))))[1]
    assert row[-1] == "未记录或不可核验，请查看来源血缘"
