"""过程漏斗快照：build_funnel 纯映射 / summarize_funnel 人话 / collect_funnel 容错 / CLI 双流。

北极星「度量体系·过程漏斗」的停点计数只读快照。测试钉住三条诚实边界:
- 快照不是历史通过率(收尾行固定声明);
- 未收录的枚举按原样列出,不硬塞进相近段;
- 某段记录读取失败只点名跳过,其余段照常统计。
"""

import json
import sys
from pathlib import Path

import pytest

from src.workbench.funnel_report import build_funnel, collect_funnel, summarize_funnel

ALL_NEXT_ACTIONS = [
    "awaiting_analysis",
    "needs_business_answers",
    "needs_capability",
    "needs_recipe",
    "needs_data_revision",
    "needs_labels",
    "review_preview",
    "awaiting_full_data",
    "awaiting_full_validation",
    "needs_full_data_revision",
    "review_full_data",
    "awaiting_dataset_split",
    "ready_for_training_preflight",
]


def _five_roots(tmp_path: Path) -> tuple[Path, ...]:
    return tuple(
        tmp_path / name
        for name in ("intake", "training", "evaluations", "acceptance", "iterations")
    )


def test_build_funnel_maps_every_next_action_to_its_journey_stage():
    report = build_funnel(
        session_next_actions=ALL_NEXT_ACTIONS,
        training_statuses=[],
        evaluation_purposes=[],
        acceptance_decisions=[],
        iteration_statuses=[],
        iteration_decisions=[],
    )
    assert report["sessions"] == {
        "total": 13,
        "stages": {
            "analysis": 6,
            "preview_confirm": 1,
            "full_validation": 4,
            "split": 1,
            "preflight_ready": 1,
        },
        "unmapped": [],
    }


def test_build_funnel_keeps_unmapped_verbatim_dedup_and_sorted():
    report = build_funnel(
        session_next_actions=["zeta", "alpha", "zeta"],
        training_statuses=["prepared", "succeeded", "prepared"],
        evaluation_purposes=[],
        acceptance_decisions=[],
        iteration_statuses=[],
        iteration_decisions=[],
    )
    assert report["sessions"]["unmapped"] == ["alpha", "zeta"]
    assert list(report["sessions"]["stages"].values()) == [0] * 5
    assert report["training"] == {"prepared": 2, "succeeded": 1}
    assert list(report["training"]) == ["prepared", "succeeded"], "计数按状态名排序"


def test_summarize_funnel_empty_state_and_closing_boundary():
    report = build_funnel(
        session_next_actions=[],
        training_statuses=[],
        evaluation_purposes=[],
        acceptance_decisions=[],
        iteration_statuses=[],
        iteration_decisions=[],
    )
    lines = summarize_funnel(report)
    assert lines[0] == "还没有任何数据任务记录，先从上传业务表格开始。"
    assert lines[-1].startswith("以上是各任务当前停点的快照，不是历史通过率")
    assert "不代表业务效果" in lines[-1]


def test_summarize_funnel_reports_stops_and_busiest_entry():
    report = build_funnel(
        session_next_actions=["awaiting_analysis", "needs_labels", "ready_for_training_preflight"],
        training_statuses=[],
        evaluation_purposes=[],
        acceptance_decisions=[],
        iteration_statuses=[],
        iteration_decisions=[],
    )
    lines = summarize_funnel(report)
    assert "共 3 个数据任务，当前停点：分析与方案 2、预检就绪 1。" in lines
    assert "停得最多的是「分析与方案」（2 个）——这里就是下一个最值得查看具体卡点的入口。" in lines


def test_summarize_funnel_all_beyond_data_prep_lists_unmapped_instead_of_faking_zero():
    report = build_funnel(
        session_next_actions=["mystery_state"],
        training_statuses=[],
        evaluation_purposes=[],
        acceptance_decisions=[],
        iteration_statuses=[],
        iteration_decisions=[],
    )
    lines = summarize_funnel(report)
    assert "共 1 个数据任务，当前停点：全部已越过数据准备段。" in lines
    assert "另有 1 种未收录的停点状态（mystery_state），按原样列出。" in lines
    assert not any("停得最多" in line for line in lines), "没有任何已收录停点时不给「最集中」结论"


def test_summarize_funnel_renders_each_segment_with_human_names_and_verbatim_unknowns():
    report = build_funnel(
        session_next_actions=["awaiting_analysis"],
        training_statuses=["prepared", "succeeded"],
        evaluation_purposes=["development_only", "final_acceptance"],
        acceptance_decisions=["passed"],
        iteration_statuses=["decided", "evaluated"],
        iteration_decisions=["adopt"],
    )
    lines = summarize_funnel(report)
    assert "训练运行共 2 个：已准备未启动 1、成功 1。" in lines
    assert "对照评测共 2 份：开发集对照 1、最终验收测试 1。" in lines
    assert "最终业务验收共 1 次：通过 1。" in lines
    assert "改进轮次共 2 轮，已决策 1 轮（采用 1）。" in lines


def test_summarize_funnel_iteration_line_without_decisions_and_verbatim_unknown_status():
    report = build_funnel(
        session_next_actions=[],
        training_statuses=["mystery_status"],
        evaluation_purposes=[],
        acceptance_decisions=[],
        iteration_statuses=["proposed"],
        iteration_decisions=[],
    )
    lines = summarize_funnel(report)
    assert "训练运行共 1 个：mystery_status 1。" in lines, "未知状态按原样列出"
    assert "改进轮次共 1 轮。" in lines, "无决策时不出现「已决策」半句"


def test_summarize_funnel_names_failed_segments_and_survives_bare_record():
    report = {"errors": ["训练记录", "业务验收记录"]}
    lines = summarize_funnel(report)
    assert "训练记录、业务验收记录读取失败，该段未计入（其余段照常统计）。" in lines
    assert lines[-1].startswith("以上是各任务当前停点的快照")


def test_collect_funnel_reads_empty_or_missing_roots_as_zero_without_errors(tmp_path):
    roots = _five_roots(tmp_path)
    for root in roots:  # 先测目录不存在,再测空目录存在
        assert not root.exists()
    first = collect_funnel(*roots)
    for root in roots:
        root.mkdir(parents=True, exist_ok=True)  # IntakeService 构造时已建过自己的根
    second = collect_funnel(*roots)
    for report in (first, second):
        assert report["sessions"]["total"] == 0
        assert report["training"] == {}
        assert report["evaluations"] == {}
        assert report["acceptances"] == {}
        assert report["iterations"] == {"total": 0, "statuses": {}, "decisions": {}}
        assert report["errors"] == []


def test_collect_funnel_skips_only_the_failed_segment_and_names_it(tmp_path, monkeypatch):
    from src.workbench.intake_service import IntakeService
    from src.workbench.training_runs import TrainingRunService
    from tests.unit.test_data_intake import CSV

    intake_root, training_root, evaluation_root, acceptance_root, iteration_root = _five_roots(
        tmp_path
    )
    IntakeService(intake_root).create("根据客户首次描述预测类别", "工单.csv", CSV)

    def broken_list_runs(self):
        raise ValueError("训练记录损坏")

    monkeypatch.setattr(TrainingRunService, "list_runs", broken_list_runs)
    report = collect_funnel(
        intake_root, training_root, evaluation_root, acceptance_root, iteration_root
    )
    assert report["errors"] == ["训练记录"]
    assert report["sessions"]["total"] == 1, "未失败的段照常统计"
    assert report["sessions"]["stages"]["analysis"] == 1
    assert "训练记录读取失败，该段未计入（其余段照常统计）。" in summarize_funnel(report)


def test_collect_funnel_names_failed_session_segment(tmp_path, monkeypatch):
    from src.workbench.intake_service import IntakeService

    intake_root, *_ = _five_roots(tmp_path)

    def broken_list_sessions(self):
        raise OSError("数据任务记录不可读")

    monkeypatch.setattr(IntakeService, "list_sessions", broken_list_sessions)
    report = collect_funnel(*_five_roots(tmp_path))
    assert report["errors"] == ["数据任务记录"]
    assert report["sessions"]["total"] == 0


def test_cli_funnel_report_streams_json_stdout_and_human_stderr(tmp_path, monkeypatch, capsys):
    """双流契约:stdout 纯 JSON(脚本可解析),stderr 人话摘要与 summarize_funnel 同源。"""
    from scripts import data_intake

    intake_root, training_root, evaluation_root, acceptance_root, iteration_root = _five_roots(
        tmp_path
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "data_intake.py",
            "--store",
            str(intake_root),
            "--training-root",
            str(training_root),
            "--evaluation-root",
            str(evaluation_root),
            "--acceptance-root",
            str(acceptance_root),
            "--iteration-root",
            str(iteration_root),
            "funnel-report",
        ],
    )
    assert data_intake.main() == 0
    out, err = capsys.readouterr()
    payload = json.loads(out)
    assert payload["sessions"]["total"] == 0
    assert payload["errors"] == []
    assert "还没有任何数据任务记录，先从上传业务表格开始。" in err
    assert "不是历史通过率" in err


def test_ui_sidebar_funnel_snapshot_matches_cli_wording(tmp_path, monkeypatch):
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    import ui.config
    from src.workbench.intake_service import IntakeService
    from tests.unit.test_data_intake import CSV

    PAGE = Path(__file__).resolve().parents[2] / "ui/pages/07_Data_Intake.py"
    for name in ("PROVIDER", "BASE_URL", "MODEL", "API_KEY"):
        monkeypatch.delenv(f"TUNESMITH_AGENT_{name}", raising=False)
    monkeypatch.setattr(ui.config, "PROJECT_ROOT", tmp_path)

    # 空库:侧栏快照显示空状态,页面无异常。
    empty_page = AppTest.from_file(str(PAGE), default_timeout=20)
    empty_page.run()
    assert not empty_page.exception
    assert any(
        "还没有任何数据任务记录，先从上传业务表格开始。" in block.value
        for block in empty_page.markdown
    )

    # 建一个新任务后:与 CLI funnel-report 同一条停点句(单一来源词汇)。
    IntakeService(tmp_path / "outputs/workbench/intake").create(
        "根据客户首次描述预测类别", "工单.csv", CSV
    )
    page = AppTest.from_file(str(PAGE), default_timeout=20)
    page.run()
    assert not page.exception
    assert any(
        "共 1 个数据任务，当前停点：分析与方案 1。" in block.value for block in page.markdown
    )
    assert any("停得最多的是「分析与方案」" in block.value for block in page.markdown)
