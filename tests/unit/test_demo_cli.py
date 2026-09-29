"""零密钥演示旅程的 CLI 平权(R80):create --demo 与 baseline-analyze 行为级钉死。

页面的两个零密钥入口——「第一次使用？用内置演示任务开始」与「没有 Agent 服务？
用基础分析开始」——此前只在页面可用;CLI 用户创建任务后必须配置 Agent 才能继续,
与「零密钥走到训练前检查」的承诺不符。本文件钉死 CLI 侧两条命令与页面同源:
演示冷启动的目标/样例/说明来自 demo_task 单一来源,基础分析与页面共用
propose_baseline_analysis + summarize_analysis,边界(互斥、缺文件、已有分析)
如实报错不静默降级。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from scripts import data_intake
from src.workbench.demo_task import DEMO_DESCRIPTION, DEMO_GOAL, is_demo_session
from src.workbench.intake_service import IntakeService
from tests.unit.test_data_intake import CSV, analysis

ROOT = Path(__file__).resolve().parents[2]


def _run(monkeypatch, capsys, store: Path, *argv: str) -> int:
    monkeypatch.setattr(sys, "argv", ["data_intake.py", "--store", str(store), *argv])
    return data_intake.main()


def test_create_demo_matches_ui_path_and_digest(monkeypatch, capsys, tmp_path):
    """create --demo 与页面一键演示同一来源:目标/说明来自 demo_task,内容摘要可被
    配套全量按钮的门控识别(digest 单一来源),不造内联副本。"""
    assert _run(monkeypatch, capsys, tmp_path / "intake", "create", "--demo") == 0
    captured = capsys.readouterr()
    session = json.loads(captured.out)
    assert session["goal"] == DEMO_GOAL, "演示目标必须来自 demo_task 单一来源"
    loaded = IntakeService(tmp_path / "intake").load(session["session_id"])
    assert loaded.data_description == DEMO_DESCRIPTION
    assert loaded.source.scope == "sample"
    assert is_demo_session(loaded.source.digest, ROOT), (
        "创建出的任务必须能被配套全量数据的摘要门控识别为演示任务"
    )
    assert "已用内置演示任务创建" in captured.err
    assert "baseline-analyze SESSION_ID --target 类别 --group 编号 --exclude 处理结果" in (
        captured.err
    ), "零密钥下一步指引必须在场"
    assert "没有 Agent 服务用 baseline-analyze 零密钥开始" in captured.err, (
        "共享尾行(awaiting_analysis)必须与零密钥指引同方向,不再把用户指去需密钥的 analyze 单一路径"
    )


def test_create_demo_conflict_and_missing_args_exit_2(monkeypatch, capsys, tmp_path):
    """--demo 与全部自定义参数互斥(逐一点名,不静默忽略);两者都缺同样明确报错。"""
    code = _run(
        monkeypatch,
        capsys,
        tmp_path / "intake",
        "create",
        "--demo",
        "--input",
        "x.csv",
        "--goal",
        "测试",
    )
    assert code == 2
    err = capsys.readouterr().err
    assert "不收 --input、--goal" in err, "互斥参数必须逐一点名"
    assert "不静默忽略" in err, "不静默忽略边界必须写明"
    code = _run(monkeypatch, capsys, tmp_path / "intake", "create", "--demo", "--scope", "full")
    assert code == 2
    assert "--scope full" in capsys.readouterr().err, "scope=full 与演示样例矛盾,必须点名"
    code = _run(
        monkeypatch,
        capsys,
        tmp_path / "intake",
        "create",
        "--demo",
        "--description",
        "自定义说明",
    )
    assert code == 2
    assert "--description" in capsys.readouterr().err, "会被静默忽略的参数同样必须点名"
    code = _run(monkeypatch, capsys, tmp_path / "intake", "create")
    assert code == 2
    assert "create 需要 --input 与 --goal；零密钥冷启动可改用 --demo" in capsys.readouterr().err


def test_create_demo_absent_files_honest(monkeypatch, capsys, tmp_path):
    """演示文件不在场时如实报错(与页面隐藏入口同口径),不编造演示数据。"""
    monkeypatch.setattr(data_intake, "PROJECT_ROOT", tmp_path)
    assert _run(monkeypatch, capsys, tmp_path / "intake", "create", "--demo") == 2
    assert "不编造演示数据" in capsys.readouterr().err


def test_baseline_analyze_zero_key_preview_and_summary(monkeypatch, capsys, tmp_path):
    """baseline-analyze 与页面共用同一来源:确定性方案、真实预览与 summarize_analysis
    人话摘要;stdout 仍是纯 JSON,全程零密钥。"""
    sample = tmp_path / "sample.csv"
    sample.write_bytes(CSV)
    assert (
        _run(
            monkeypatch,
            capsys,
            tmp_path / "intake",
            "create",
            "--input",
            str(sample),
            "--goal",
            "根据客户描述判断售后问题类型",
        )
        == 0
    )
    session_id = json.loads(capsys.readouterr().out)["session_id"]
    code = _run(
        monkeypatch,
        capsys,
        tmp_path / "intake",
        "baseline-analyze",
        session_id,
        "--target",
        "类别",
        "--group",
        "编号",
        "--exclude",
        "处理结果",
    )
    assert code == 0
    captured = capsys.readouterr()
    session = json.loads(captured.out)
    assert session["agent_model"] == "baseline-deterministic"
    assert session["preview"] is not None, "基础分析必须产出真实转换预览"
    assert "这份分析给出数据判断与待确认问题" in captured.err, "摘要必须来自 summarize_analysis"
    assert "不代表业务效果达标。" in captured.err
    assert "下一步" in captured.err


def test_baseline_analyze_blocked_when_analysis_exists(monkeypatch, capsys, tmp_path):
    """已有 Agent 分析时拒绝:页面此时隐藏基础分析入口,CLI 同口径不悄悄覆盖——
    把 Agent 方案换成确定性方案是质量降级。已有基础分析可重跑见替换测试。"""
    sample = tmp_path / "sample.csv"
    sample.write_bytes(CSV)
    _run(
        monkeypatch,
        capsys,
        tmp_path / "intake",
        "create",
        "--input",
        str(sample),
        "--goal",
        "目标",
    )
    session_id = json.loads(capsys.readouterr().out)["session_id"]
    # 用 Agent 家族分析占位(model="" 非 baseline-deterministic):
    # 页面同口径——已有 Agent 分析时不再显示基础分析入口。
    service = IntakeService(tmp_path / "intake")
    service.apply_analysis(service.load(session_id), analysis())
    code = _run(
        monkeypatch,
        capsys,
        tmp_path / "intake",
        "baseline-analyze",
        session_id,
        "--target",
        "类别",
    )
    assert code == 2
    err = capsys.readouterr().err
    assert "CLI 同口径" in err
    assert "请用 analyze" in err, "拒绝时必须点名 Agent 重分析的出口,不是死胡同"


def test_baseline_analyze_replaces_prior_baseline_analysis(monkeypatch, capsys, tmp_path):
    """needs_data_revision 的零密钥出口(R82):已有基础分析时调整字段重新生成——
    替换旧方案、预览与确认状态随之失效(与 Agent 重分析同一套失效语义)。"""
    sample = tmp_path / "sample.csv"
    sample.write_bytes(CSV)
    _run(
        monkeypatch,
        capsys,
        tmp_path / "intake",
        "create",
        "--input",
        str(sample),
        "--goal",
        "目标",
    )
    session_id = json.loads(capsys.readouterr().out)["session_id"]
    assert (
        _run(
            monkeypatch,
            capsys,
            tmp_path / "intake",
            "baseline-analyze",
            session_id,
            "--target",
            "类别",
        )
        == 0
    )
    capsys.readouterr()
    # 确认样例后重跑:确认状态被替换打回原形,这是零密钥用户修复问题行后的循环。
    service = IntakeService(tmp_path / "intake")
    confirmed = service.confirm(session_id, service.load(session_id).revision)
    assert confirmed.confirmed_revision is not None, "前置:重跑前确已确认"
    code = _run(
        monkeypatch,
        capsys,
        tmp_path / "intake",
        "baseline-analyze",
        session_id,
        "--target",
        "类别",
        "--instruction",
        "只依据客户原话判断,不要参考处理结果",
    )
    assert code == 0, "同族(基础→基础)重跑必须放行,这是零密钥出口"
    captured = capsys.readouterr()
    assert "已替换此前的基础分析" in captured.err, "替换必须先于摘要明说,不悄悄覆盖"
    loaded = IntakeService(tmp_path / "intake").load(session_id)
    assert loaded.confirmed_revision is None, "替换后确认状态必须失效,重新核对"
    assert loaded.agent_model == "baseline-deterministic"
    assert loaded.analysis.recipe.instruction == "只依据客户原话判断,不要参考处理结果", (
        "调整的字段必须真实进入新方案"
    )


def test_baseline_analyze_temporal_missing_flags_exit_2(monkeypatch, capsys, tmp_path):
    """--temporal 必须同时给全六个参数:缺哪些点名哪些,不静默按随机切分降级。"""
    sample = tmp_path / "sample.csv"
    sample.write_bytes(CSV)
    _run(
        monkeypatch,
        capsys,
        tmp_path / "intake",
        "create",
        "--input",
        str(sample),
        "--goal",
        "目标",
    )
    session_id = json.loads(capsys.readouterr().out)["session_id"]
    code = _run(
        monkeypatch,
        capsys,
        tmp_path / "intake",
        "baseline-analyze",
        session_id,
        "--target",
        "类别",
        "--temporal",
    )
    assert code == 2
    err = capsys.readouterr().err
    for flag in (
        "--available-at-column",
        "--prediction-at-column",
        "--label-end-column",
        "--validation-start",
        "--test-start",
        "--observation-end",
    ):
        assert flag in err, f"缺失参数必须逐一点名: {flag}"


def test_baseline_analyze_temporal_success_path(monkeypatch, capsys, tmp_path):
    """--temporal 六参数齐全的成功路径:键名与 TemporalSplitPolicy 字段一致
    (label_end_at_column),方案真实携带 temporal_split——CLI 时间分区不是只会
    报错的死路;插桩即测,键名再写错这里先红。"""
    from tests.unit.test_baseline_analysis import TEMPORAL_ROWS

    header = "编号,描述,类别,记录时间,决策时间,窗口结束\n"
    body = "".join(",".join(row) + "\n" for row in TEMPORAL_ROWS)
    sample = tmp_path / "temporal.csv"
    sample.write_text(header + body, encoding="utf-8")
    assert (
        _run(
            monkeypatch,
            capsys,
            tmp_path / "intake",
            "create",
            "--input",
            str(sample),
            "--goal",
            "按描述判断类别",
        )
        == 0
    )
    session_id = json.loads(capsys.readouterr().out)["session_id"]
    code = _run(
        monkeypatch,
        capsys,
        tmp_path / "intake",
        "baseline-analyze",
        session_id,
        "--target",
        "类别",
        "--temporal",
        "--available-at-column",
        "记录时间",
        "--prediction-at-column",
        "决策时间",
        "--label-end-column",
        "窗口结束",
        "--validation-start",
        "2026-02-01T00:00:00Z",
        "--test-start",
        "2026-03-01T00:00:00Z",
        "--observation-end",
        "2026-04-01T00:00:00Z",
    )
    assert code == 0, "六参数齐全时时间方案必须成功(键名与模型字段一致)"
    temporal = json.loads(capsys.readouterr().out)["analysis"]["recipe"]["temporal_split"]
    assert temporal["label_end_at_column"] == "窗口结束", "策略必须真实携带时间分区字段"


def test_baseline_analyze_instruction_override_pinned(monkeypatch, capsys, tmp_path):
    """--instruction 覆盖被钉死:自定义指令进方案,agent_model 仍是确定性记名。"""
    sample = tmp_path / "sample.csv"
    sample.write_bytes(CSV)
    _run(
        monkeypatch,
        capsys,
        tmp_path / "intake",
        "create",
        "--input",
        str(sample),
        "--goal",
        "目标",
    )
    session_id = json.loads(capsys.readouterr().out)["session_id"]
    assert (
        _run(
            monkeypatch,
            capsys,
            tmp_path / "intake",
            "baseline-analyze",
            session_id,
            "--target",
            "类别",
            "--instruction",
            "只依据客户原话判断,不要参考处理结果",
        )
        == 0
    )
    session = json.loads(capsys.readouterr().out)
    assert session["analysis"]["recipe"]["instruction"] == "只依据客户原话判断,不要参考处理结果"
    assert session["agent_model"] == "baseline-deterministic"


def test_zero_key_demo_cli_journey_reaches_preview(tmp_path, monkeypatch, capsys):
    """CLI 侧零密钥旅程最小闭环:create --demo → baseline-analyze 直接可续,
    演示样例的字段与指引命令逐字对得上(指引不是空话)。"""
    assert _run(monkeypatch, capsys, tmp_path / "intake", "create", "--demo") == 0
    session_id = json.loads(capsys.readouterr().out)["session_id"]
    code = _run(
        monkeypatch,
        capsys,
        tmp_path / "intake",
        "baseline-analyze",
        session_id,
        "--target",
        "类别",
        "--group",
        "编号",
        "--exclude",
        "处理结果",
    )
    assert code == 0, "create --demo 的指引命令必须对演示数据真实可用"
    assert json.loads(capsys.readouterr().out)["preview"] is not None


@pytest.mark.parametrize(
    "command",
    ["create", "baseline-analyze"],
)
def test_cli_help_documents_zero_key_entries(monkeypatch, capsys, tmp_path, command):
    """--help 与命令面同步:两个零密钥入口在帮助文本里可发现,口径与页面词汇一致。"""
    from tests.unit.test_readme_alignment import _cli_help_text

    help_text = _cli_help_text(monkeypatch, capsys, tmp_path, command)
    if command == "create":
        assert "--demo" in help_text
        assert "第一次使用？用内置演示任务开始" in help_text, "入口文案必须与页面一致"
        assert "不编造演示数据" in help_text
    else:
        assert "--target" in help_text and "--group" in help_text and "--exclude" in help_text
        assert "--temporal" in help_text
        assert "零密钥" in help_text
        assert "没有 Agent 服务？用基础分析开始" in help_text, "入口文案必须与页面一致"
        assert "重新生成" in help_text, "已有基础分析可调整字段重新生成(R82 零密钥出口)必须在场"
        assert "已有 Agent 分析时拒绝" in help_text, "拒绝边界同样写在帮助文本里"
