"""待补答案清单(介入点 10):行筛选、填写表与三条规则各有单一来源。

筛选与 needs_labels 判定共用;填写表只含 行ID、题目输入与待填列,结构上
不可能泄露答案;三条规则把「怎么交接、按什么依据填、输入不足怎么办」
编码成确定性文案,页面与 CLI 不各说各话。
"""

from __future__ import annotations

from types import SimpleNamespace

import src.workbench.temporal_split as temporal_split_module
from src.workbench.answer_sheet import (
    answer_sheet_csv,
    answer_sheet_lines,
    missing_answer_rows,
)


def _row(row_id, status):
    return SimpleNamespace(
        row_id=row_id,
        input=f"输入-{row_id}",
        target=None,
        original={},
        group={},
        status=status,
        issues=[],
    )


def test_missing_answer_rows_filters_by_status_and_excludes_pending_labels(monkeypatch):
    """needs_label 且非未成熟标签才进清单;未成熟是既定保留,不是漏填。"""
    calls = []

    def fake_pending(row, policy):
        calls.append((row.row_id, policy))
        return row.row_id == "r000003"  # 只有这行是「标签窗口未结束」的保留行

    monkeypatch.setattr(temporal_split_module, "is_pending_label", fake_pending)
    rows = [
        _row("r000001", "needs_label"),
        _row("r000002", "ready"),
        _row("r000003", "needs_label"),  # 未成熟:排除
        _row("r000004", "invalid"),
        _row("r000005", "conflict"),
    ]
    policy = object()
    missing = missing_answer_rows(rows, policy)
    assert [row.row_id for row in missing] == ["r000001"]
    # policy 原样透传给未成熟判定,不在此处重复时间算术。
    assert ("r000001", policy) in calls


def test_missing_answer_rows_defaults_to_no_policy():
    """无时间方案时(None)照常筛选——常见分类任务没有未成熟概念。"""
    rows = [_row("r000001", "needs_label"), _row("r000002", "ready")]
    assert [row.row_id for row in missing_answer_rows(rows)] == ["r000001"]


def test_answer_sheet_csv_is_bom_excel_csv_with_blank_fill_columns():
    """BOM 保证 Excel 直开;列固定 行ID/题目输入/每字段一个待填列,待填格为空。"""
    rows = [_row("r000001", "needs_label"), _row("r000002", "needs_label")]
    data = answer_sheet_csv(rows, ["类别"])
    assert data.startswith(b"\xef\xbb\xbf")  # utf-8-sig BOM
    text = data.decode("utf-8-sig")
    lines = text.strip("\r\n").splitlines()
    assert lines[0] == "行ID,题目输入（模型将看到的内容）,待填答案（类别）"
    assert lines[1] == "r000001,输入-r000001,"
    assert lines[2] == "r000002,输入-r000002,"
    # 多字段时每字段一列,顺序与 recipe.targets 一致。
    multi = answer_sheet_csv(rows, ["类别", "紧急度"]).decode("utf-8-sig")
    assert multi.splitlines()[0] == (
        "行ID,题目输入（模型将看到的内容）,待填答案（类别）,待填答案（紧急度）"
    )
    assert multi.splitlines()[1].endswith(",,")
    # 结构上不泄露答案:表内没有 target 列,只有输入原文与空待填格。


def test_answer_sheet_lines_pin_handoff_no_guessing_and_insufficient_input():
    """三条规则逐条钉住:交接机制、防编造门槛、输入不足的如实处置。"""
    lines = answer_sheet_lines(7)
    assert len(lines) == 3
    assert lines[0] == (
        "共 7 条待补：交给填写人后只填「待填答案」列，行ID 与题目输入列保持原样——回传后按行ID 对号回填。"
    )
    assert "不要凭猜测填" in lines[1]
    assert "监督信号" in lines[1]
    assert "错一条教错一条" in lines[1]
    assert "留空" in lines[2]
    assert "任务定义问题" in lines[2]
    assert "比编一个答案更有价值" in lines[2]
