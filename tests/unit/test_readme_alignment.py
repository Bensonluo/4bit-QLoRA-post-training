"""README 与北极星对齐:本地链接可解析,语义安全层与场景矩阵如实在场。

文档健康度巡检:README 主路径、agent-setup 语义安全层(含盲标 CLI 完整用法)、
north-star 权威版的关键句全部钉死——文档漂移即测试红,改文档必须改测试。
"""

import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
README = ROOT / "README.md"
AGENT_SETUP = ROOT / "docs" / "agent-setup.md"

# 北极星权威版;README 只能指向它,不能另立版本。
NORTH_STAR = "docs/plans/north-star.md"
NORTH_STAR_FILE = ROOT / NORTH_STAR

# 架构权威版:north-star 管目标,本文管协作机制(三层边界/任务规约/多轮状态机)。
TASK_SPEC_DESIGN = ROOT / "docs" / "plans" / "task-spec-and-agent-adaptation-design.md"

# 试用记录:专家介入点清单的现状补充按条钉死,历史原文不改写。
USER_TRIAL_LOG = ROOT / "docs" / "validation" / "user-trial-log.md"

LINK_PATTERN = re.compile(r"\[[^\]]+\]\(([^)]+)\)")
IMG_PATTERN = re.compile(r'<img\s+src="([^"]+)"')


def _section(text, start, end=None):
    """截取 start 标题到 end 标题(缺省到文末)之间的段落,用于局部钉死。"""
    begin = text.index(start)
    finish = text.index(end, begin) if end else len(text)
    return text[begin:finish]


def _local_targets(text):
    for match in LINK_PATTERN.findall(text) + IMG_PATTERN.findall(text):
        target = match.strip()
        if target.startswith(("http://", "https://", "mailto:", "#")):
            continue
        yield target


# hero GIF 曾被列为待录制占位;R79 实物落地后不再有任何豁免——仓库内引用必须真实存在。
DOCUMENTED_PENDING_ASSETS: set[str] = set()


def test_all_local_links_resolve_to_existing_files():
    """README 引用的每个仓库内路径都必须存在——不再引用已删除文档。"""
    broken = [
        target
        for target in _local_targets(README.read_text(encoding="utf-8"))
        if target not in DOCUMENTED_PENDING_ASSETS and not (ROOT / target.split("#")[0]).exists()
    ]
    assert broken == [], f"README 引用了不存在的文件: {broken}"


def test_product_description_covers_semantic_safety_layer():
    """产品主路径描述必须包含语义安全层(盲标/对比/探针)与场景矩阵。"""
    text = README.read_text(encoding="utf-8")
    for term in ("盲标核验", "对比核验", "可学性探针", "场景矩阵", "语义安全"):
        assert term in text, f"README 缺少语义安全层关键词: {term}"
    # 英文主描述同样可见(关键词的英文名或中文名至少其一)。
    assert "semantic safety layer" in text.lower()


def test_readme_points_to_authoritative_north_star():
    assert (ROOT / NORTH_STAR).exists()
    assert README.read_text(encoding="utf-8").count(NORTH_STAR) >= 1


def test_license_file_exists_and_declares_mit():
    """README 与 pyproject 都声明 MIT,许可证文件必须真实存在。"""
    license_text = (ROOT / "LICENSE").read_text(encoding="utf-8")
    assert "MIT License" in license_text
    assert "Permission is hereby granted" in license_text


def test_readme_links_agent_setup_and_both_files_exist():
    """README 主工作流必须链接 agent-setup 配置指南,文档本体真实存在。"""
    text = README.read_text(encoding="utf-8")
    assert "](docs/agent-setup.md)" in text, "README 缺少 agent-setup 链接"
    assert AGENT_SETUP.exists()


def test_agent_setup_semantic_safety_layer_pins_three_gates():
    """语义安全层段:三道关卡条目、口径与确定性边界的关键句钉死。"""
    section = _section(
        AGENT_SETUP.read_text(encoding="utf-8"),
        "## 语义安全层",
        "### 盲标核验的完整 CLI 用法",
    )
    for gate in (
        "**对比核验（样例确认时）**",
        "**盲标核验（训练准备的硬门禁）**",
        "**可学性探针（可选证据，非门禁）**",
    ):
        assert gate in section, f"语义安全层缺少关卡条目: {gate}"
    assert "（1–50 条，默认 5 条）" in section, "盲标样本量口径漂移"
    assert "预览或方案变化后核验自动失效" in section, "对比核验失效规则漂移"
    assert "与「瞎猜多数类」基线对比" in section, "可学性探针基线口径漂移"
    assert "这三道关卡都不使用 LLM 判断、不消耗模型服务额度，判定全部确定性可复现。" in section


def test_contrast_cli_docs_pinned():
    """对比核验 CLI 用法钉死:命令、--answer 格式、二连对口径、confirm 提示与软门禁边界。"""
    section = _section(
        AGENT_SETUP.read_text(encoding="utf-8"),
        "## 语义安全层",
        "### 盲标核验的完整 CLI 用法",
    )
    assert "contrast-check SESSION_ID --revision CURRENT_REVISION" in section
    assert "contrast-check-submit SESSION_ID --check-id CHECK_ID --answer 行ID=候选答案" in section
    assert "stdout 为纯 JSON" in section, "位点必须写明:stdout 纯 JSON、stderr 人读"
    assert "还需再连续配对正确一轮（二连对）才算真正看清" in section, "二连对口径必须写明"
    assert "防瞎蒙靠的是连胜不是单轮" in section, "连胜语义必须写明"
    assert "轮连胜：转换的业务含义经多组不同题目反复配对核对" in section, "三档连胜词汇必须写明"
    assert "`contrast_streak_banner` 单一来源" in section, "连胜词汇单一来源必须写明"
    assert "页面横幅与 CLI 不各说各话" in section, "页面与 CLI 同源必须写明"
    assert "此前的确认可能是盲点头" in section, "配错提示必须写明"
    assert "尚未核验" in section and "建议先运行 `contrast-check`" in section
    assert "软门禁" in section and "不阻断确认" in section, "软门禁边界必须写明"
    assert "提交不收 `--revision`" in section, "提交参数边界必须写明"
    assert "自动失效" in section, "失效规则必须写明"


def test_next_action_tail_docs_pinned():
    """stderr 尾行人话对照钉死:枚举保留+人话翻译+单一来源+未知态不编造。"""
    section = _section(
        AGENT_SETUP.read_text(encoding="utf-8"),
        "## CLI 配置与检查",
        "## 从样例继续到全量数据",
    )
    assert "下一步状态: 枚举（人话对照）" in section, "尾行格式必须写明"
    assert "`next_action_phrase` 单一来源" in section, "人话对照单一来源必须写明"
    assert "13 个状态全覆盖" in section and "未知状态只显枚举不编造" in section
    assert "awaiting_analysis 的尾行同时点名两条路径" in section, (
        "零密钥路径进尾行必须在场(R81):awaiting_analysis 不只指 analyze"
    )
    assert "needs_data_revision 的尾行同样点名两条重分析路径" in section, (
        "重分析路径进尾行必须在场(R82):needs_data_revision 不只指需密钥的 analyze"
    )
    assert "needs_labels 的尾行点名零密钥修复工具链" in section, (
        "缺标签出口进尾行必须在场(R83):needs_labels 不只是提醒,还给零密钥工具"
    )
    assert "`answer-sheet` 导出待补清单" in section, "R83 工具链必须点名 answer-sheet"
    assert "补齐标注」不是一句没有出口的提醒" in section, "R83 诚实定位句必须在场"
    assert "review_preview 的尾行点名配对工具链" in section, (
        "配对工具链进尾行必须在场(R84):review_preview 不能只说完成对比核验不给入口"
    )
    assert "`contrast-check` 抽两道配对题" in section, "R84 工具链必须点名 contrast-check"
    assert "「完成对比核验」不是页面专属动作" in section, "R84 平权定位句必须在场"
    assert "awaiting_full_data／awaiting_full_validation 的尾行点名 `full-validate`" in section, (
        "全量验证入口进尾行必须在场(R85):awaiting_full_* 不能只说提供全量数据不给命令"
    )
    assert "多资料任务用 `full-sources`" in section, "R85 多资料变体必须点名"
    assert "review_full_data 的尾行点名 `full-confirm`" in section, (
        "R85 全量确认命令必须点名"
    )
    assert "needs_full_data_revision 的尾行点名修正后重跑" in section, (
        "R85 阻断态重验出口必须点名"
    )
    assert "全量数据段每一步都有可照抄的命令入口" in section, "R85 平权定位句必须在场"
    assert "零密钥用户在问题行修复后不会走进死胡同" in section, "R82 死胡同根治的诚实边界必须写明"
    assert "不会被指去配置密钥才能跑的命令" in section, "零密钥诚实边界必须写明"
    assert "与页面提示同词汇" in section, "页面与 CLI 同源必须写明"
    for command in ("show", "create", "analyze", "confirm", "materialize"):
        assert f"`{command}`" in section, "尾行命令清单必须列明"


def test_listing_tail_docs_pinned():
    """五个清单命令的清单尾行钉死:空态点名入口、非空计数、summarize_listing 单一来源。"""
    section = _section(
        AGENT_SETUP.read_text(encoding="utf-8"),
        "## CLI 配置与检查",
        "## 从样例继续到全量数据",
    )
    assert "`summarize_listing` 单一来源" in section, "清单尾行单一来源必须写明"
    assert "空清单点名该走的第一步入口" in section, "空态口径必须写明"
    assert "用户分不清「还没有」和「查错了任务」" in section, "空态动机必须写明"
    assert "非空给计数" in section, "计数口径必须写明"
    assert "不逐条灌业务人话" in section, "不灌逐条人话边界必须写明"
    for command in ("train-list", "plan-list", "iteration-list", "acceptance-list", "scoring-list"):
        assert f"`{command}`" in section, "清单命令五员必须列明"


def test_agent_setup_contrast_check_help_matches_documentation(monkeypatch, capsys, tmp_path):
    """对比核验用法与真实 argparse 同步:--revision 必填;--check-id/--answer 提交参数。"""
    help_text = _cli_help_text(monkeypatch, capsys, tmp_path, "contrast-check")
    assert "--revision" in help_text
    submit_help = _cli_help_text(monkeypatch, capsys, tmp_path, "contrast-check-submit")
    assert "--check-id" in submit_help
    assert "--answer" in submit_help


def test_agent_setup_blind_label_cli_section_pins_full_usage():
    """「盲标核验的完整 CLI 用法」小节:命令、参数与统计键说明钉死。"""
    section = _section(
        AGENT_SETUP.read_text(encoding="utf-8"),
        "### 盲标核验的完整 CLI 用法",
        "## 生成独立数据分区与版本",
    )
    assert "label-verify SESSION_ID --revision CURRENT_REVISION --size 5" in section
    assert "label-verify-submit SESSION_ID --verification-id VERIFICATION_ID" in section
    assert "--export-csv" in section
    for key in ("evidence_note", "shortfall_note", "submit_hint", "agreement_lower_bound"):
        assert key in section, f"盲标 CLI 用法缺少统计键说明: {key}"
    assert "不收 `--revision`" in section
    assert "只含「行ID、题目输入、留空待填的盲标答案」三列，不含数据答案" in section
    assert "判定：verified（5/5 一致，95% 置信下界约 57%）" in section


def test_agent_setup_blind_label_sample_size_stats_pinned():
    """「盲标核验的样本量与统计口径」段:自选范围与 Wilson 下界数字钉死。"""
    section = _section(AGENT_SETUP.read_text(encoding="utf-8"), "## 盲标核验的样本量与统计口径")
    assert "页面抽取表单可在 1–50 条之间自选（默认 5 条）" in section
    assert "超出范围的请求在抽题前就被拒绝" in section
    assert "5 条的 95% Wilson 置信下界约 57%，20 条约 84%，30 条约 89%" in section


def test_readme_quick_start_pins_dashboard_main_path():
    """Quick Start 主路径:安装、启动命令、入口页名与 agent-setup 链接钉死。"""
    quickstart = _section(README.read_text(encoding="utf-8"), "## 🚀 Quick Start", "## 🧙")
    assert "### Option 1: Dashboard (recommended)" in quickstart
    assert "python -m venv venv && source venv/bin/activate" in quickstart
    assert 'pip install -e ".[ui]"' in quickstart
    assert "python scripts/launch_dashboard.py" in quickstart
    assert "http://localhost:8501 and choose **目标与数据**" in quickstart
    assert "[Agent setup and workflow](docs/agent-setup.md)" in quickstart


def test_north_star_pins_authority_and_key_sentences():
    """北极星权威版:章节结构、安全层五机制、硬边界与指标口径钉死。"""
    text = NORTH_STAR_FILE.read_text(encoding="utf-8")
    assert "# TuneSmith 北极星目标(权威版本)" in text
    for heading in (
        "## 一句话定义",
        "## 服务对象",
        "## 硬边界",
        "## 核心难点与安全层",
        "## 度量体系",
    ):
        assert heading in text, f"北极星缺少关键章节: {heading}"
    assert "压缩成一个非专家能安全操作、看得懂、可追溯的工位" in text
    for mechanism in (
        "**盲标核验**",
        "**对比预览**",
        "**可学性探针**",
        "**同题对照**",
        "**确定性门禁**",
    ):
        assert mechanism in text, f"北极星安全层缺少机制条目: {mechanism}"
    for boundary in ("**不做推理/serving 层**", "**不替用户宣判业务成败**", "**密钥纪律**"):
        assert boundary in text, f"北极星缺少硬边界条目: {boundary}"
    assert "覆盖率 × 独立通过率" in text
    assert "**诚实红线**" in text
    assert "场景多样性用场景矩阵系统性覆盖" in text


def test_north_star_gap_section_pins_closed_evidence_and_remaining_gaps():
    """差距对照节随轮次演进:已收口三差距带证据,未收口差距如实点名且旧表述不复存在。"""
    text = NORTH_STAR_FILE.read_text(encoding="utf-8")
    # 旧表述(2026-09-27 声称三差距开放)必须已被修订替换。
    assert "差距集中在三点" not in text
    # 已收口:三差距的证据指针在场。
    for phrase in (
        "与现状的对照(2026-09-28",
        "train-export",
        "train-cost",
        "过程漏斗",
        "f24cfbd..3aa6eef",
    ):
        assert phrase in text, f"北极星差距对照缺少收口证据指针: {phrase}"
    # 仍然开放的差距如实点名:北极星指标零真实测量是当前第一差距。
    for phrase in (
        "北极星指标从未被测过",
        "开发者自测",
        "不构成北极星证据",
        "task-spec-and-agent-adaptation-design.md",
        "CLI 平权",
    ):
        assert phrase in text, f"北极星差距对照缺少开放差距点名: {phrase}"


def _cli_help_text(monkeypatch, capsys, tmp_path, *command):
    """进程内跑真实 CLI 的 --help(参考 test_label_verify_cli 的 argv 注入模式)。"""
    from scripts import data_intake

    # CJK 描述在默认 80 列下会被 argparse 折行,把逐字钉死的短语从中间拆断——
    # 统一放宽到 240 列,断言对象是文案本身而不是终端宽度。
    monkeypatch.setenv("COLUMNS", "240")
    monkeypatch.setattr(
        sys,
        "argv",
        ["data_intake.py", "--store", str(tmp_path), *command, "--help"],
    )
    with pytest.raises(SystemExit) as excinfo:
        data_intake.main()
    assert excinfo.value.code == 0
    return capsys.readouterr().out


def test_agent_setup_label_verify_help_matches_documentation(monkeypatch, capsys, tmp_path):
    """agent-setup 盲标抽题用法与真实 argparse 同步:--size 默认 5 上限 50、--export-csv。"""
    help_text = _cli_help_text(monkeypatch, capsys, tmp_path, "label-verify")
    assert "--revision" in help_text
    assert "--size" in help_text
    assert "默认 5，上限 50" in help_text, "样本量口径漂移:文档写 1–50 默认 5"
    assert "--export-csv" in help_text


def test_agent_setup_label_verify_submit_help_matches_documentation(monkeypatch, capsys, tmp_path):
    """label-verify-submit:--verification-id 必填、--answer 可重复、不收 --revision。"""
    help_text = _cli_help_text(monkeypatch, capsys, tmp_path, "label-verify-submit")
    assert "--verification-id" in help_text
    assert "每条抽样行一个" in help_text
    assert "--revision" not in help_text, "文档钉死:label-verify-submit 不收 --revision"


def test_probe_cli_verdict_docs_pinned():
    """可学性探针 CLI 判定行口径钉死:三组数字、三态词汇单一来源、note 原文。"""
    section = _section(
        AGENT_SETUP.read_text(encoding="utf-8"),
        "## 语义安全层",
        "### 盲标核验的完整 CLI 用法",
    )
    assert "先给判定行" in section, "判定行先于候选清单必须写明"
    assert "基座零样本、瞎猜多数类基线与差异三组数字" in section
    assert "`probe_verdict_phrase` 单一来源" in section, "三态词汇单一来源必须写明"
    assert "先核查提示格式与任务定义" in section
    assert "不预测微调效果" in section
    assert "`learnability-probe-show` 回读同样先给判定行" in section
    assert "页面与 CLI 不各说各话" in section


def test_agent_setup_learnability_probe_help_matches_documentation(monkeypatch, capsys, tmp_path):
    """可学性探针用法同步:--revision/--model-path 必填,--size/--export-csv 可选。"""
    help_text = _cli_help_text(monkeypatch, capsys, tmp_path, "learnability-probe")
    assert "--revision" in help_text
    assert "--model-path" in help_text
    assert "--size" in help_text
    assert "--export-csv" in help_text


def test_semantic_safety_gates_are_backed_by_scenario_matrix():
    """agent-setup「三道关卡已接入流程」由场景矩阵背书:41+ 场景含盲标/对比拦截覆盖。"""
    from src.workbench.scenario_specs import builtin_scenarios

    scenarios = builtin_scenarios()
    assert len(scenarios) >= 41, f"场景矩阵应保持 41+ 规模,当前 {len(scenarios)}"
    blind = [s for s in scenarios if "盲标" in s.expect_note]
    contrast = [s for s in scenarios if "对比核验" in s.expect_note]
    assert blind, "场景矩阵缺少盲标核验相关场景,agent-setup 的门禁描述失去背书"
    assert contrast, "场景矩阵缺少对比核验相关场景,agent-setup 的门禁描述失去背书"


def test_sheet_selection_docs_and_cli_help_in_sync(monkeypatch, capsys, tmp_path):
    """sheet 选择文档与真实 CLI 同步:三入口同款 help 串、full-sources 按资料指定、
    agent-setup 的四个入口/标注/拒绝边界关键句在场。"""
    canonical = "读取 Excel 的指定 sheet（名称或序号，1 表示第一个；默认第一个）"
    for command in ("create", "add-source", "full-validate"):
        help_text = _cli_help_text(monkeypatch, capsys, tmp_path, command)
        assert "--sheet" in help_text, f"{command} 缺少 --sheet 参数"
        assert canonical in help_text, f"{command} 的 --sheet help 漂移"
    full_sources_help = _cli_help_text(monkeypatch, capsys, tmp_path, "full-sources")
    assert "--sheet ALIAS=名称或序号" in full_sources_help, "full-sources 缺少按资料的 --sheet"
    assert "1 起始序号" in full_sources_help

    section = _section(
        AGENT_SETUP.read_text(encoding="utf-8"),
        "### 多份资料一起分析",
        "### 长尾字段解析与受限适配",
    )
    assert "四个入口" in section, "sheet 选择文档必须说明四个入口的对称性"
    assert "--sheet ALIAS=名称或序号" in section
    assert "仅读取第一个" in section and "如实标注" in section, "默认读取范围标注口径漂移"
    assert "CSV/JSONL 传 sheet 会被明确拒绝" in section
    single = _section(
        AGENT_SETUP.read_text(encoding="utf-8"), "### 单份资料", "### 连续数值答案的如实边界"
    )
    assert "`full-validate` 支持 `--encoding`、`--delimiter`、`--sheet`" in single


def test_numeric_continuous_boundary_docs_pinned_and_backed_by_matrix():
    """连续数值答案的如实边界文档:value_kind/逐字/回归边界关键句钉死,由场景矩阵背书。"""
    section = _section(
        AGENT_SETUP.read_text(encoding="utf-8"),
        "### 连续数值答案的如实边界",
        "## 语义安全层",
    )
    assert "numeric_continuous" in section
    assert "不是数值回归" in section and "逐字" in section
    assert "「1.0」与「1.00」算两个不同答案" in section
    assert "numeric_new_values" in section and "不阻断" in section
    assert "离散化" in section, "数值误差口径的出路(先离散化)必须写明"
    assert "整数编码（0/1/2）不受此判定影响" in section

    # 场景矩阵背书:numeric-continuous-target 场景真实存在,文档不是空头承诺
    from src.workbench.scenario_specs import builtin_scenarios

    scenarios = {s.scenario_id: s for s in builtin_scenarios()}
    assert "numeric-continuous-target" in scenarios, "场景矩阵缺少连续数值目标场景"


def test_merged_cells_docs_pinned_and_backed_by_matrix():
    """Excel 合并单元格文档:merged_note/不自动填充/xls 边界关键句钉死,由场景矩阵背书。"""
    section = _section(
        AGENT_SETUP.read_text(encoding="utf-8"),
        "### 多份资料一起分析",
        "### 长尾字段解析与受限适配",
    )
    assert "merged_note" in section
    assert "合并区除左上角外均读为空值" in section
    assert "没有自动填充" in section
    assert "缺少监督答案" in section, "空值根因与拦截的关联必须写明"
    assert "未读取 sheet 的合并不列入" in section
    assert "xls 引擎不提供合并范围，不检测" in section

    # 场景矩阵背书:merged-cells-in-target-column 场景真实存在且结局被钉住
    from src.workbench.scenario_specs import builtin_scenarios

    scenarios = {s.scenario_id: s for s in builtin_scenarios()}
    assert "merged-cells-in-target-column" in scenarios, "场景矩阵缺少合并单元格场景"
    assert scenarios["merged-cells-in-target-column"].expect == "blocked_at:confirm_sample"


def test_formula_cells_docs_pinned_and_backed_by_matrix():
    """Excel 公式格无缓存文档:formula_note/不自动计算/带缓存不列入/xls 边界关键句钉死,
    由场景矩阵背书。"""
    section = _section(
        AGENT_SETUP.read_text(encoding="utf-8"),
        "### 多份资料一起分析",
        "### 长尾字段解析与受限适配",
    )
    assert "formula_note" in section
    assert "这些公式读为空值" in section
    assert "没有自动计算" in section
    assert "缺少监督答案" in section, "空值根因与拦截的关联必须写明"
    assert "带缓存值，按缓存值正常读取、不列入" in section
    assert "分组列等其他列的公式同样读空" in section, "影响面不止答案列必须写明"
    assert "xls 引擎不提供公式清单，不检测" in section

    # 场景矩阵背书:formula-cells-in-target-column 场景真实存在且结局被钉住
    from src.workbench.scenario_specs import builtin_scenarios

    scenarios = {s.scenario_id: s for s in builtin_scenarios()}
    assert "formula-cells-in-target-column" in scenarios, "场景矩阵缺少公式格场景"
    assert scenarios["formula-cells-in-target-column"].expect == "blocked_at:confirm_sample"


def test_hidden_rows_docs_pinned_and_backed_by_matrix():
    """Excel 隐藏行/列文档:hidden_note/照常读入/不自动排除/xls 边界关键句钉死,
    由场景矩阵背书。"""
    section = _section(
        AGENT_SETUP.read_text(encoding="utf-8"),
        "### 多份资料一起分析",
        "### 长尾字段解析与受限适配",
    )
    assert "hidden_note" in section
    assert "隐藏行照常读入" in section
    assert "Excel 中看不到的行也会进入分析与训练" in section
    assert "没有自动排除" in section, "不自动排除的语义安全边界必须写明"
    assert "隐藏列仍出现在可用字段中" in section, "隐藏列的影响面必须写明"
    assert "未读取 sheet 的隐藏不列入" in section
    assert "xls 引擎不提供隐藏标志，不检测" in section

    # 场景矩阵背书:hidden-rows-in-sheet 场景真实存在且结局被钉住(披露不阻断)
    from src.workbench.scenario_specs import builtin_scenarios

    scenarios = {s.scenario_id: s for s in builtin_scenarios()}
    assert "hidden-rows-in-sheet" in scenarios, "场景矩阵缺少隐藏行场景"
    assert scenarios["hidden-rows-in-sheet"].expect == "passes"


def test_excel_fact_notes_page_rendering_docs_pinned():
    """来源如实标注的页面渲染位点文档钉死:披露不埋进 JSON,非专家看得见。"""
    section = _section(
        AGENT_SETUP.read_text(encoding="utf-8"),
        "### 多份资料一起分析",
        "### 长尾字段解析与受限适配",
    )
    assert "不埋进 JSON" in section, "披露必须写明在页面渲染,而非只存 JSON"
    assert "「原始资料与补充文件」" in section, "补充资料区的标注位点必须写明"
    assert "全量验证报告的来源行下" in section, "全量报告的标注位点必须写明"
    assert "sheet 级标注用 info" in section
    assert "warning 提示影响数据事实" in section, "severity 分级必须写明"
    assert "一条都不渲染" in section, "CSV/JSONL 负例边界必须写明"
    assert "空行与重复表头事实同样渲染" in section, "blank/dup_header 跨格式渲染边界必须写明"


def test_blank_rows_docs_pinned_and_backed_by_matrix():
    """空行/全空行文档:blank_note 双口径(已跳过/照常读入)/尾部空行边界/xls 覆盖
    关键句钉死,由场景矩阵背书。"""
    section = _section(
        AGENT_SETUP.read_text(encoding="utf-8"),
        "### 多份资料一起分析",
        "### 长尾字段解析与受限适配",
    )
    assert "blank_note" in section
    assert "已跳过" in section, "CSV/JSONL 跳过口径必须写明"
    assert "空行不进入分析与训练" in section, "跳过的下游影响必须写明"
    assert "照常读入为全空记录" in section, "Excel 全空行口径必须写明"
    assert "缺少监督答案与分组标识" in section, "全空行被拦的根因关联必须写明"
    assert "没有自动排除" in section, "不自动排除的语义安全边界必须写明"
    assert "尾部空行在解析时自然消失、无从检测" in section, "如实边界必须写明"
    assert "xls 同样检测" in section, "跨引擎覆盖面(与 xlsx-only 检测的差别)必须写明"

    # 场景矩阵背书:blank-rows-in-sheet 场景真实存在且结局被钉住(样例侧缺标签门拦截)
    from src.workbench.scenario_specs import builtin_scenarios

    scenarios = {s.scenario_id: s for s in builtin_scenarios()}
    assert "blank-rows-in-sheet" in scenarios, "场景矩阵缺少全空行场景"
    assert scenarios["blank-rows-in-sheet"].expect == "blocked_at:confirm_sample"


def test_dup_header_rows_docs_pinned_and_backed_by_matrix():
    """重复表头行文档:dup_header_note 双侧行为分述/判据与全量侧同源/没有自动删行/
    单列不检测/跨引擎覆盖关键句钉死,由场景矩阵背书。"""
    section = _section(
        AGENT_SETUP.read_text(encoding="utf-8"),
        "### 多份资料一起分析",
        "### 长尾字段解析与受限适配",
    )
    assert "dup_header_note" in section
    assert "输入与答案都会是列名" in section, "该行污染输入与答案的根因必须写明"
    assert "样例侧不拦" in section and "全量验证硬拦" in section, "两侧行为分述必须写明"
    assert "没有自动删行" in section, "不自动删行的语义安全边界必须写明"
    assert "判据完全一致" in section, "与全量侧 repeated_header_rows 同口径必须写明"
    assert "单列文件不检测" in section, "单列边界必须写明"
    assert "xls 均检测" in section, "跨引擎覆盖面(与 xlsx-only 检测的差别)必须写明"

    # 场景矩阵背书:样例侧(披露不拦)与全量侧(硬拦)两场景真实存在且结局钉住
    from src.workbench.scenario_specs import builtin_scenarios

    scenarios = {s.scenario_id: s for s in builtin_scenarios()}
    assert "duplicate-header-row-in-sample" in scenarios, "场景矩阵缺少样例侧重复表头场景"
    assert scenarios["duplicate-header-row-in-sample"].expect == "passes"
    assert "duplicate-header-rows-in-full" in scenarios, "场景矩阵缺少全量侧重复表头场景"
    assert scenarios["duplicate-header-rows-in-full"].expect == "blocked_at:validate_full"


def test_answer_coverage_docs_pinned_and_backed_by_matrix():
    """物化分区答案覆盖披露文档:三统计键/逐字学习边界/没有自动重切/20 种门限/
    时间与固定题集同口径关键句钉死,由场景矩阵背书。"""
    section = _section(
        AGENT_SETUP.read_text(encoding="utf-8"),
        "## 生成独立数据分区与版本",
        "## 时间预测任务：先核对来源与标签窗口",
    )
    for key in ("answer_coverage_note", "answer_counts_by_split", "train_missing_answers"):
        assert key in section, f"答案覆盖披露缺少统计键说明: {key}"
    assert "从未出现在训练集" in section, "训练集缺口点名必须写明"
    assert "逐字学习" in section and "照常打分" in section, "逐字学习与照常打分的反差必须写明"
    assert "没有自动重新切分" in section, "不自动重切的语义安全边界必须写明"
    assert "不超过 20 种" in section and "超过 20 种" in section, "20 种门限的双侧口径必须写明"
    assert "时间分区与固定题集沿用同一披露口径" in section, "三种切分方式同口径必须写明"
    assert "页面数据集版本区的摘要与 CLI" in section, "页面与 CLI 两个展示位点必须写明"

    # 场景矩阵背书:rare-category-only-in-holdout 场景真实存在且结局钉住(披露不阻断)
    from src.workbench.scenario_specs import builtin_scenarios

    scenarios = {s.scenario_id: s for s in builtin_scenarios()}
    assert "rare-category-only-in-holdout" in scenarios, "场景矩阵缺少稀有答案落保留分区场景"
    assert scenarios["rare-category-only-in-holdout"].expect == "passes"


def test_duplicate_rows_docs_pinned_and_backed_by_matrix():
    """完全相同例题披露文档:双计数键/两种成因分述/隐式加权/与冲突守卫分界/
    同分区/没有自动去重关键句钉死,由场景矩阵背书。"""
    section = _section(
        AGENT_SETUP.read_text(encoding="utf-8"),
        "## 生成独立数据分区与版本",
        "## 时间预测任务：先核对来源与标签窗口",
    )
    for key in ("rendered_exact_duplicate_rows", "source_exact_duplicate_rows", "duplicate_note"):
        assert key in section, f"重复例题披露缺少统计键说明: {key}"
    assert "渲染后完全相同" in section, "渲染重复的判定口径必须写明"
    assert "等效于给这些例题加权" in section, "隐式加权效应必须写明"
    assert "原始行完全重复" in section and "渲染成同一例题" in section, "两种成因分述必须写明"
    assert "相同输入配不同答案会被全量验证硬拦" in section, "与冲突守卫的分界必须写明"
    assert "永不跨分区" in section, "重复例题同分区事实必须写明"
    assert "没有自动去重" in section, "不自动去重的语义安全边界必须写明"

    # 场景矩阵背书:exact-duplicate-rows-in-full 场景真实存在且结局钉住(披露不阻断)
    from src.workbench.scenario_specs import builtin_scenarios

    scenarios = {s.scenario_id: s for s in builtin_scenarios()}
    assert "exact-duplicate-rows-in-full" in scenarios, "场景矩阵缺少完全相同例题场景"
    assert scenarios["exact-duplicate-rows-in-full"].expect == "passes"


def test_field_accuracy_disclosure_docs_pinned():
    """评测对照逐字段准确率披露文档:逐字段口径/失败截断计错/整题答对口径关键句钉死。"""
    section = _section(
        AGENT_SETUP.read_text(encoding="utf-8"),
        "## 比较基座与本轮微调效果",
        "## 让 Agent 解读结果与下一步",
    )
    assert "field_accuracy" in section
    assert "最弱" in section
    assert "无法按 JSON 解析" in section
    assert "全部字段都对" in section
    assert "保持沉默" in section  # 非 JSON 任务沉默边界
    assert "大白话解读" in section  # 页面位点


def test_dominant_output_disclosure_docs_pinned():
    """输出坍缩披露文档:判定口径/核查方向/复述多数类边界/零分归因关键句钉死。"""
    section = _section(
        AGENT_SETUP.read_text(encoding="utf-8"),
        "## 比较基座与本轮微调效果",
        "## 让 Agent 解读结果与下一步",
    )
    assert "输出高度重复" in section
    assert "`dominant_output_models` 单一来源" in section
    assert "80%" in section and "至少 4 条" in section, "判定口径必须写明"
    assert "对照开发集答案分布" in section, "核查方向必须写明"
    assert "复述多数类" in section, "答案分布集中的假阳性边界必须写明"
    assert "不认定原因" in section, "观察事实边界必须写明"
    assert "反复输出同一答案" in section, "零分归因点名必须写明"
    assert "没有观察到截断、生成失败、复述或重复输出" in section, "缺位句口径必须写明"
    assert "不计入占比分母" in section, "None 输出口径必须写明"
    assert "对照区警告" in section, "位点必须写明"


def test_custom_scoring_summary_docs_pinned():
    """自定义/开放任务对照的人话摘要口径钉死:不把通过数写成答对,开放任务只报生成事实。"""
    section = _section(
        AGENT_SETUP.read_text(encoding="utf-8"),
        "## 定义并确认自定义业务评分",
        "## 用独立测试题做单模型业务验收",
    )
    assert "大白话解读" in section
    assert "业务评分均值" in section and "通过" in section
    assert "不把通过数写成「答对」" in section
    assert "不做自动评分" in section
    assert "生成失败、缺失或截断的回答不能人工标为通过" in section


def test_scoring_summary_docs_pinned():
    """scoring-* CLI stderr 人话摘要口径钉死:澄清态、确认绑定语义与不构成达标判断。"""
    section = _section(
        AGENT_SETUP.read_text(encoding="utf-8"),
        "## 定义并确认自定义业务评分",
        "## 用独立测试题做单模型业务验收",
    )
    assert "stderr" in section, "位点必须写明:stdout 纯 JSON、stderr 追加人话"
    assert "scoring-list` 只追加一行清单尾行" in section, "清单尾行口径必须写明"
    assert "summarize_listing" in section, "清单尾行单一来源必须写明"
    assert "软件不会自动确认评分规则" in section, "草稿态不自动确认边界必须写明"
    assert "绑定当前业务目标与输入/答案语义" in section, "已确认规则的绑定语义必须写明"
    assert "数据修订后兼容规则可继续用" in section, "兼容复用与重新确认的分界必须写明"
    assert "当前没有可确认的评分方案" in section, "澄清态口径必须写明"
    assert "不等于严格准确率" in section, "均值/通过率不作严格准确率必须写明"
    assert "不构成业务达标的判断" in section, "固定边界句必须写明"
    assert "summarize_scoring" in section, "页面渲染位点必须点名摘要函数"
    assert "同源同词汇" in section, "页面与 CLI 同口径承诺必须写明"
    # 评分摘要工具核查轨迹行(R62):与评测解读同格式(summarize_tool_trace 单一来源),由摘要函数自动带上。
    assert "评分摘要现以评测解读同一格式渲染工具核查轨迹" in section, (
        "评分轨迹行必须写明(与评测解读同格式)"
    )


def test_suite_summary_docs_pinned():
    """suite-freeze/show CLI stderr 人话摘要口径钉死:题数、内容锁定与比较基线边界。"""
    section = _section(
        AGENT_SETUP.read_text(encoding="utf-8"),
        "## 从评测结果进入下一轮改进",
        "## 配置文件与优先级",
    )
    assert "stderr" in section, "位点必须写明:stdout 纯 JSON、stderr 追加人话"
    assert "summarize_suite" in section, "摘要函数必须点名"
    assert "原评分题不能修改" in section, "内容锁定边界必须写明"
    assert "不自动扩充评分题" in section, "新增行不扩充原题必须写明"
    assert "--suite-id" in section, "复用方式必须写明"
    assert "锚定数据版本" in section, "完整清单的锚定版本必须写明"
    assert "固定题集只保证各轮比较基线一致" in section, "固定边界句必须写明"
    assert "不代表业务效果达标" in section, "不作业务结论边界必须写明"
    assert "页面「分区设置」选择固定题集后渲染同一份摘要" in section, "页面位点必须写明"
    assert "同源同词汇" in section, "页面与 CLI 同口径承诺必须写明"


def test_assessment_summary_docs_pinned():
    """eval-analyze CLI stderr 人话摘要口径钉死:观察/假设分离、轨迹失败计数与不自动执行边界。"""
    section = _section(
        AGENT_SETUP.read_text(encoding="utf-8"),
        "## 让 Agent 解读结果与下一步",
        "## 定义并确认自定义业务评分",
    )
    assert "stderr" in section, "位点必须写明:stdout 纯 JSON、stderr 追加人话"
    assert "summarize_assessment" in section, "摘要函数必须点名"
    assert "假设不是事实" in section, "观察/假设分离口径必须写明"
    assert "先核查数据" in section, "决策五态人话名必须写明"
    assert "需要业务核对" in section, "决策五态人话名必须写明"
    assert "解读自己声明的局限" in section, "局限原文复述必须写明"
    assert "失败的调用没有取到证据" in section, "轨迹失败计数口径必须写明"
    assert "不会据此自动改标签" in section, "不自动执行边界必须写明"
    assert "不代表业务效果达标" in section, "不作业务结论边界必须写明"
    assert "核查记录下方渲染同一份摘要" in section, "页面位点必须写明"
    assert "同源同词汇" in section, "页面与 CLI 同口径承诺必须写明"


def test_full_report_summary_docs_pinned():
    """full-* CLI stderr 全量报告人话摘要口径钉死:结论四态、逐条问题原文与边界句。"""
    section = _section(
        AGENT_SETUP.read_text(encoding="utf-8"),
        "### 单份资料",
        "### 连续数值答案的如实边界",
    )
    assert "stderr" in section, "位点必须写明:stdout 纯 JSON、stderr 追加人话"
    assert "summarize_full_report" in section, "摘要函数必须点名"
    assert "存在阻断问题需先修正再重验" in section, "阻断态口径必须写明"
    assert "尚未开始训练" in section, "确认态不启动训练边界必须写明"
    assert "报告失效" in section, "失效态口径必须写明"
    assert "阻断在前" in section, "问题排序口径必须写明"
    assert "已生成预览" in section and "缺少答案" in section, "转换四态计数必须写明"
    assert "不代表模型效果或业务达标" in section, "不作业务结论边界必须写明"
    assert "页面全量验证区" in section, "页面同源位点必须写明"
    assert "同源" in section, "页面与 CLI 词汇同源承诺必须写明"


def test_acceptance_summary_docs_pinned():
    """最终验收 CLI stderr 人话摘要口径钉死:冻结标准、五态结论、分母口径与不作数边界。"""
    section = _section(
        AGENT_SETUP.read_text(encoding="utf-8"),
        "## 用独立测试题做单模型业务验收",
        "## 从评测结果进入下一轮改进",
    )
    assert "stderr" in section, "位点必须写明:stdout 纯 JSON、stderr 追加人话"
    assert "运行前冻结的标准" in section, "冻结条款复述必须写明"
    assert "证据不足，不能确认可交付" in section, "结论三态必须写明"
    assert "失败与截断保留在全部题目分母中，按未通过计" in section, "分母口径必须写明"
    assert "数值不能作为独立业务验收的结论" in section, "隔离未核验不作数边界必须写明"
    assert "仅作描述" in section, "自定义规则业务评分均值的口径必须写明"
    assert "不会自动部署模型" in section, "收尾边界必须写明"
    assert "summarize_acceptance" in section, "页面渲染位点必须点名摘要函数"
    assert "页面最终验收区在每条记录的结论下方渲染同一份摘要" in section, "页面位点必须写明"
    assert "同一口径" in section, "页面与 CLI 同口径承诺必须写明"


def test_iteration_execution_summary_docs_pinned():
    """迭代与自动执行 CLI stderr 人话摘要口径钉死:状态机各态、等确认用户动作与诚实边界。"""
    section = _section(
        AGENT_SETUP.read_text(encoding="utf-8"),
        "## 从评测结果进入下一轮改进",
        "## 配置文件与优先级",
    )
    assert "stderr" in section, "位点必须写明:stdout 纯 JSON、stderr 追加人话"
    assert "采用本轮结果" in section, "决定四态回显(采用/继续/停止/证据不足)必须写明"
    assert "不会自动部署模型" in section, "采用记录的不部署边界必须写明"
    assert "不代表业务效果达标" in section, "收尾边界必须写明"
    assert "关闭页面不影响执行" in section, "后台独立性必须写明"
    assert "勾选确认继续才会恢复" in section, "等确认暂停的用户动作必须写明"
    assert "重复提交不会再次训练" in section, "completed 幂等边界必须写明"
    assert "worker.log" in section, "被阻断/失败的排查指向必须写明"
    assert "summarize_iteration" in section, "页面轮次摘要位点必须点名函数"
    assert "summarize_execution" in section, "页面执行摘要位点必须点名函数"
    assert "页面改进轮次区渲染同一对摘要函数" in section, "页面位点必须写明"
    assert "刷新按钮上方" in section, "执行摘要的页面锚点必须写明"
    assert "同源同词汇" in section, "页面与 CLI 同口径承诺必须写明"


def test_materialize_summary_docs_pinned():
    """materialize CLI stderr 分区人话摘要口径钉死:三切分统一、边界句与时间补充行。"""
    section = _section(
        AGENT_SETUP.read_text(encoding="utf-8"),
        "## 生成独立数据分区与版本",
        "## 时间预测任务：先核对来源与标签窗口",
    )
    assert "stderr" in section, "位点必须写明:stdout 纯 JSON、stderr 追加人话"
    assert "summarize_dataset" in section, "与页面同口径的摘要函数必须点名"
    assert "按业务对象隔离划分" in section, "分法句(分组)必须写明"
    assert "按已确认的时间边界划分" in section, "分法句(时间)必须写明"
    assert "沿用固定开发/测试题集" in section, "分法句(固定题集)必须写明"
    assert "比例受分组大小影响" in section, "分组如实边界必须写明"
    assert "分区就绪只说明数据已按规则隔离" in section, "固定边界句必须写明"
    assert "逐原因排除计数" in section, "时间方案逐原因计数补充行必须写明"
    assert "metadata.excluded_rows" in section, "manifest 原行明细指引必须写明"


def test_plan_summary_docs_pinned():
    """plan-* CLI stderr 人话摘要口径钉死:分层位点、状态三态、理由原文与不自动启动边界。"""
    section = _section(
        AGENT_SETUP.read_text(encoding="utf-8"),
        "## 让 Agent 推荐训练方案",
        "## 在同一任务中启动真实训练",
    )
    assert "stderr" in section, "位点必须写明:stdout 纯 JSON、stderr 追加人话"
    assert "plan-list` 只追加一行清单尾行" in section, "清单尾行口径必须写明"
    assert "summarize_listing" in section, "清单尾行单一来源必须写明"
    assert "方案可供确认" in section, "状态三态(ready)必须写明"
    assert "需要先完善数据" in section, "状态三态(needs_data)必须写明"
    assert "当前条件不支持" in section, "状态三态(unsupported)必须写明"
    assert "推荐理由与尚未验证的限制原文" in section, "理由/限制如实复述必须写明"
    assert "需要你先回答的业务问题" in section, "待答业务问题必须写明"
    assert "LoRA rank" in section, "关键参数点名必须写明"
    assert "probe" in section, "预检证据的真实存放位点必须写明"
    assert "不会自动启动" in section, "确认准备的不启动边界必须写明"
    assert "不构成训练效果或业务达标的判断" in section, "固定边界句必须写明"


def test_model_list_tail_docs_pinned():
    """model-list 发现尾行钉死:单一来源、分档计数、issues 指引与页面同词汇。"""
    section = _section(
        AGENT_SETUP.read_text(encoding="utf-8"),
        "## 让 Agent 推荐训练方案",
        "## 在同一任务中启动真实训练",
    )
    assert "`summarize_model_discovery` 单一来源" in section, "发现尾行单一来源必须写明"
    assert "文件完整/不完整分档计数" in section, "分档计数口径必须写明"
    assert "issues 字段" in section, "缺文件明细指引必须写明"
    assert "文件完整只表示可以进一步检查" in section, "文件完整边界必须写明"
    assert "与页面候选模型区同词汇" in section, "页面与 CLI 同源必须写明"


def test_analysis_summary_docs_pinned():
    """analyze stderr 发现与待确认问题摘要钉死:单一来源、kind 四译名、证据行引用、工具核查轨迹行与边界句。"""
    section = _section(
        AGENT_SETUP.read_text(encoding="utf-8"),
        "## CLI 配置与检查",
        "## 从样例继续到全量数据",
    )
    assert "stderr" in section, "位点必须写明:stdout 纯 JSON、stderr 追加人话"
    assert "`summarize_analysis` 单一来源" in section, "摘要单一来源必须写明"
    assert "已观察" in section, "kind 译名(已观察)必须写明"
    assert "待验证推断" in section, "kind 译名(待验证推断)必须写明"
    assert "需要业务解释" in section, "kind 译名(需要业务解释)必须写明"
    assert "需要全量验证" in section, "kind 译名(需要全量验证)必须写明"
    assert "（证据：" in section, "证据行引用必须写明"
    assert "待确认问题" in section, "待确认问题翻译必须写明"
    assert "暂定微调思路" in section, "暂定微调思路行必须写明"
    assert "「数据判断与待确认问题」区同词汇" in section, "与页面同词汇必须写明"
    assert "不代表业务效果达标" in section, "固定边界句必须写明"
    # 工具核查轨迹行与评测解读同格式(summarize_tool_trace 单一来源),页面同位渲染。
    assert "工具核查轨迹" in section, "轨迹行必须写明(与评测解读同格式)"
    assert "`summarize_tool_trace` 单一来源" in section, "轨迹行单一来源必须点名"
    assert "分析只依赖成功的调用" in section, "轨迹失败计数口径必须写明"
    assert "「处理规则与工具记录」区在同一位置渲染同一行" in section, "页面轨迹行位点必须写明"
    assert "同源同词汇" in section, "页面与 CLI 同口径承诺必须写明"


def test_train_lineage_docs_pinned(monkeypatch, capsys, tmp_path):
    """train-lineage 与 registry_cli lineage 的人话口径钉死:双流契约、单一来源、边界句与同源同词汇。"""
    section = _section(
        AGENT_SETUP.read_text(encoding="utf-8"),
        "## 在同一任务中启动真实训练",
        "## 允许一次显存不足技术恢复",
    )
    assert "train-lineage RUN_ID" in section, "CLI 块必须列 train-lineage"
    assert "stderr" in section, "位点必须写明:stdout 纯 JSON、stderr 追加人话"
    assert "`summarize_registration` 单一来源" in section, "正向摘要单一来源必须写明"
    assert "点名全部版本与别名" in section, "已注册口径必须写明"
    assert "可照抄的合并与注册命令" in section, "未注册指引必须写明"
    assert "查询失败如实报告原因，不编造状态" in section, "失败态口径必须写明"
    assert "注册只说明模型库记录了这次训练的产物与血缘，不代表业务效果达标" in section, (
        "注册边界句必须写明"
    )
    assert "registry_cli.py lineage" in section, "反向血缘入口必须写明"
    assert "`summarize_lineage` 单一来源" in section, "反向摘要单一来源必须写明"
    assert "缺项如实显示「-」" in section, "缺项回退口径必须写明"
    assert "同源同词汇" in section, "页面与 CLI 同口径承诺必须写明"
    # 帮助文本与命令面同步:收 RUN_ID,不收 train-logs 的 --tail 与准备类 --revision
    help_text = _cli_help_text(monkeypatch, capsys, tmp_path, "train-lineage")
    assert "run_id" in help_text, "train-lineage 收位置参数 RUN_ID"
    assert "--tail" not in help_text, "--tail 是 train-logs 专属"
    assert "--revision" not in help_text, "血缘是只读查询,不收 --revision"
    assert "注册状态" in help_text, "帮助文本必须说明这是注册状态查询"


def test_train_export_docs_pinned(monkeypatch, capsys, tmp_path):
    """train-export(环节⑨交接)人话口径钉死:默认目录、证据链、幂等、阻塞点名与边界句。"""
    section = _section(
        AGENT_SETUP.read_text(encoding="utf-8"),
        "## 在同一任务中启动真实训练",
        "## 允许一次显存不足技术恢复",
    )
    assert "train-export RUN_ID" in section, "CLI 块必须列 train-export"
    assert "`summarize_export` 单一来源" in section, "摘要单一来源必须写明"
    assert "默认 `outputs/workbench/merged/RUN_ID`" in section, "默认导出目录必须写明"
    assert "`--output-dir` 覆盖" in section, "输出目录覆盖参数必须写明"
    assert "不下载、不更换底座" in section, "复用训练记录基座的边界必须写明"
    assert "export_evidence.json" in section, "证据链文件必须写明"
    assert "重复导出不会改变模型内容" in section, "幂等口径必须写明"
    assert "逐条点名原因，不静默降级" in section, "阻塞态口径必须写明"
    assert "导出只产出模型文件与证据记录，不代表业务效果达标，也不会自动部署" in section, (
        "导出边界句必须写明"
    )
    assert "只读盘点，不在页面执行合并" in section, "页面只读盘点边界必须写明"
    assert "同源同词汇" in section, "页面与 CLI 同口径承诺必须写明"
    # 帮助文本与命令面同步:收 RUN_ID 与 --output-dir,不收只读族没有的 --revision/--tail。
    help_text = _cli_help_text(monkeypatch, capsys, tmp_path, "train-export")
    assert "run_id" in help_text, "train-export 收位置参数 RUN_ID"
    assert "--output-dir" in help_text
    assert "--revision" not in help_text, "导出沿用训练记录,不收 --revision"
    assert "--tail" not in help_text, "--tail 是 train-logs 专属"
    assert "合并导出" in help_text, "帮助文本必须说明这是合并导出"


def test_train_loss_trend_docs_pinned():
    """逐条 loss 曲线与趋势人话口径钉死:数据来源、三态判定、边界句与页面/CLI 同源。"""
    section = _section(
        AGENT_SETUP.read_text(encoding="utf-8"),
        "## 在同一任务中启动真实训练",
        "## 允许一次显存不足技术恢复",
    )
    assert "workbench_loss_history.json" in section, "序列文件名必须写明"
    assert "只剩每个键的末值，曲线必须由这份序列重建" in section, "压平背景必须写明"
    assert "`loss_trend_lines` 单一来源" in section, "趋势单一来源必须写明"
    assert "「本轮训练指标」区" in section, "页面位点必须写明"
    assert "「查看原始指标 JSON」折叠区" in section, "原始 JSON 折叠区必须写明"
    assert "整体在下降" in section and "整体基本持平" in section and "末段反而更高" in section, (
        "三态结论必须写明"
    )
    assert "不代表训练失败" in section, "持平核查方向必须写明"
    assert "学习率过大或数据里有异常样本" in section, "不降反升核查方向必须写明"
    assert "不认定原因" in section, "观察事实边界必须写明"
    assert "不代表业务效果；效果要看同一套开发题上的对照报告" in section, "边界句必须写明"
    assert "没有逐条记录，不编造曲线" in section, "缺位态口径必须写明"
    assert "同源同词汇" in section, "页面与 CLI 同口径承诺必须写明"
    # 实时曲线(R54):写入侧、训练中口径与中断态口径同段钉死。
    assert "LiveLossWriter" in section, "实时写入回调名必须写明"
    assert "`extra_callbacks` 参数" in section, "回调注入参数必须写明"
    assert "训练进行中页面与 CLI 读到的就是已训练部分的曲线" in section, "实时语义必须写明"
    assert "「训练中的 loss 曲线」区" in section, "训练中页面位点必须写明"
    assert "刷新页面查看最新进度" in section, "刷新提示必须写明"
    assert "训练进行中，趋势判定等训练完成后再看" in section, "训练中不做三态判定必须写明"
    assert "「训练未完成时已记录的 loss 曲线」区" in section, "中断态页面位点必须写明"
    assert "只代表已训练的部分" in section, "中断态边界必须写明"


def test_funnel_report_docs_pinned(monkeypatch, capsys, tmp_path):
    """funnel-report(过程漏斗快照)人话口径钉死:停点五段、容错点名、页面同源与边界句。"""
    section = _section(
        AGENT_SETUP.read_text(encoding="utf-8"),
        "## CLI 配置与检查",
        "## 从样例继续到全量数据",
    )
    assert "funnel-report" in section, "命令名必须写明"
    assert "`summarize_funnel` 单一来源" in section, "摘要单一来源必须写明"
    assert "只读快照，不发起计算" in section, "只读边界必须写明"
    assert "分析与方案／样例预览确认／全量验证／独立分区／预检就绪" in section, (
        "数据准备五段停点必须写明"
    )
    assert "下一个最值得查看具体卡点的入口" in section, "最集中停点指引必须写明"
    assert "未收录的停点状态按原样列出，不硬塞进相近段" in section, "未知枚举口径必须写明"
    assert "点名跳过" in section and "不假装为零" in section, "容错口径必须写明"
    assert "`--store` 与四个 `--*-root` 参数" in section, "数据来源参数必须写明"
    assert "「全部任务停点快照」" in section, "页面位点必须写明"
    assert "同源同词汇" in section, "页面与 CLI 同口径承诺必须写明"
    assert "不是历史通过率" in section and "不代表业务效果" in section, "边界句必须写明"
    # 帮助文本与命令面同步:不收 RUN_ID/--revision/--tail,说明这是只读停点计数。
    help_text = _cli_help_text(monkeypatch, capsys, tmp_path, "funnel-report")
    assert "停点计数" in help_text and "只读" in help_text, "帮助文本必须说明只读停点计数"
    assert "run_id" not in help_text, "funnel-report 不收 RUN_ID"
    assert "--revision" not in help_text, "快照不针对具体修订"
    assert "--tail" not in help_text, "--tail 是 train-logs 专属"


def test_task_spec_show_docs_pinned(monkeypatch, capsys, tmp_path):
    """task-spec-show(任务规约投影)口径钉死:五要素、草稿不入投影、页面同源与边界句。"""
    section = _section(
        AGENT_SETUP.read_text(encoding="utf-8"),
        "## 训练启动前对齐：任务规约投影",
        "## 在同一任务中启动真实训练",
    )
    assert "task-spec-show SESSION_ID" in section, "命令与位置参数必须写明"
    assert "`summarize_task_spec` 单一来源" in section, "摘要单一来源必须写明"
    assert "只读汇编" in section and "不写任何状态" in section, "只读边界必须写明"
    for element in ("业务目标", "答案语义", "评分口径", "验收标准", "时间约束"):
        assert element in section, f"规约五要素缺 {element}"
    assert "草稿评分规则不进入规约" in section, "草稿不入投影必须写明"
    assert "未冻结验收不假装存在标准" in section, "未冻结如实边界必须写明"
    assert "「📋 任务规约投影」" in section, "页面位点必须写明"
    assert "同源同词汇" in section, "页面与 CLI 同口径承诺必须写明"
    assert "不代表模型效果达标" in section, "边界句必须写明"
    # 规约确认联动(R60 切片 A):启动决策点同读这份投影,折叠区 label、train-start stderr 与勾选文案同段钉死。
    assert "「📋 任务规约（启动本轮训练前的口径）」" in section, "启动位折叠区 label 必须逐字写明"
    assert "`train-start` 在启动时输出同一份规约摘要" in section, (
        "train-start 启动时输出规约摘要必须写明"
    )
    assert "已核对任务规约与预检提示，按当前方案开始训练。" in section, (
        "预检警告勾选文案必须逐字写明"
    )
    assert "页面与 CLI 同源同词汇" in section, "启动决策点页面与 CLI 同口径承诺必须写明"
    # 帮助文本与命令面同步:收 SESSION_ID,只读投影不收 --revision/--tail。
    help_text = _cli_help_text(monkeypatch, capsys, tmp_path, "task-spec-show")
    assert "session_id" in help_text, "task-spec-show 收位置参数 SESSION_ID"
    assert "只读" in help_text, "帮助文本必须说明这是只读投影"
    assert "--revision" not in help_text, "投影不针对具体修订"
    assert "--tail" not in help_text, "--tail 是 train-logs 专属"


def test_train_cost_docs_pinned(monkeypatch, capsys, tmp_path):
    """train-cost(成本账 CLI 平权)口径钉死:单一来源、0 不对比、估计值边界与页面同源。"""
    section = _section(
        AGENT_SETUP.read_text(encoding="utf-8"),
        "## 在同一任务中启动真实训练",
        "## 允许一次显存不足技术恢复",
    )
    assert "train-cost RUN_ID" in section, "CLI 块必须列 train-cost"
    assert "`summarize_run_cost` 出账" in section and "`cost_lines` 出人话" in section, (
        "单一来源必须写明"
    )
    assert "0 不对比" in section, "API 对比开关口径必须写明"
    assert "默认自动检测本机设备" in section, "设备默认口径必须写明"
    assert "不是电表读数" in section, "估计值边界必须写明"
    assert "口径不同，仅供量级比较" in section, "API 对比口径必须写明"
    assert "页面在成功训练记录下渲染的成本账同一来源" in section, "页面位点必须写明"
    assert "同源同词汇" in section, "页面与 CLI 同口径承诺必须写明"
    # 帮助文本与命令面同步:收 RUN_ID 与三个可选参数,不收其他 train-* 家族参数。
    help_text = _cli_help_text(monkeypatch, capsys, tmp_path, "train-cost")
    assert "run_id" in help_text, "train-cost 收位置参数 RUN_ID"
    for flag in ("--api-price", "--monthly-queries", "--device"):
        assert flag in help_text, f"train-cost 缺少 {flag}"
    assert "--revision" not in help_text, "成本账是只读查询,不收 --revision"
    assert "--tail" not in help_text, "--tail 是 train-logs 专属"
    assert "--output-dir" not in help_text, "--output-dir 是 train-export 专属"
    assert "成本账" in help_text, "帮助文本必须说明这是成本账查询"


def test_task_spec_design_doc_pinned():
    """任务规约与 Agent 协作架构文档钉死:三层边界、规约只读投影决策、环节协议与路线。"""
    doc = TASK_SPEC_DESIGN.read_text(encoding="utf-8")
    assert "# 任务规约与 Agent 全流程协作架构" in doc, "标题必须在场"
    # ADR-1:规约是只读投影,不引入第二事实来源(双写不一致是要消灭的形态)。
    assert "只读投影，不是新实体" in doc, "核心架构决策必须写明"
    assert "不引入第二事实来源" in doc, "决策理由必须写明"
    # 三层边界:内核唯一事实、Agent 产物待确认、状态转移用户专属。
    assert "单一事实来源" in doc, "内核定位必须写明"
    assert "产物一律停在待确认" in doc, "Agent 层边界必须写明"
    assert "状态转移只能由用户显式动作触发" in doc, "用户决策层边界必须写明"
    # 环节协议与多轮状态机的关键口径。
    assert "Agent 介入是设计选择，不是默认" in doc, "环节介入原则必须写明"
    assert "insufficient_evidence" in doc, "证据不足作为合法决定必须写明"
    # 差距与路线如实:Phase 1 投影已实现并点名实现位,外部依赖不纳入自主迭代。
    assert "任务规约投影——已实现（Phase 1 落地，2026-09-28）" in doc, "投影落地必须如实登记"
    assert "task-spec-show SESSION" in doc, "落地入口必须点名"
    assert "src/workbench/task_spec_projection.py" in doc, "实现位点必须可追溯"
    assert "不纳入自主迭代" in doc, "外部依赖边界必须写明"
    # Phase 2 切片 A 与 Phase 3(协作轨迹统一)按同一格式登记完成;
    # Phase 3 如实登记侦察纠正:留痕落盘早已覆盖五入口,本轮补的是呈现统一,不是补落盘。
    assert "已完成（切片 A，2026-09-28）" in doc, "Phase 2 切片 A 落地必须如实登记"
    assert "`train-start` 启动时向 stderr 注入 `summarize_task_spec` 规约摘要" in doc, (
        "CLI 实现位点必须点名"
    )
    assert "「📋 任务规约（启动本轮训练前的口径）」折叠区" in doc, "页面实现位点必须点名"
    assert "协作轨迹统一**~~ **已完成（2026-09-28）**" in doc, (
        "Phase 3 协作轨迹统一落地必须如实登记"
    )
    assert "`summarize_tool_trace` 单一来源、页面与 CLI 同源同词汇" in doc, (
        "轨迹摘要单一来源与同口径承诺必须写明"
    )
    assert "本轮补齐的是呈现统一，不是补落盘" in doc, "侦察纠正必须如实登记"
    assert "协作轨迹呈现已统一" in doc, "差距 3 呈现统一登记必须写明"
    # R63:训练运行记录补齐 plan_trace 快照与同格式轨迹行,差距 3 完全收口。
    assert (
        "训练运行记录也已补齐（R63 落地，2026-09-28）：方案执行时把方案 trace 快照进运行记录" in doc
    ), "训练运行记录轨迹补齐必须如实登记"
    assert "差距 3 完全收口" in doc, "差距收口结论必须写明"


def test_train_run_summary_trace_docs_pinned():
    """训练运行摘要轨迹行钉死(R63):plan_trace 为方案阶段快照,直接启动如实缺席。"""
    section = _section(
        AGENT_SETUP.read_text(encoding="utf-8"),
        "## 在同一任务中启动真实训练",
        "## 允许一次显存不足技术恢复",
    )
    assert "`summarize_training_run` 单一来源" in section, "运行摘要单一来源必须写明"
    assert "`plan_trace` 键" in section, "方案阶段轨迹快照键必须写明"
    assert "`summarize_tool_trace` 单一来源" in section, "轨迹行单一来源必须写明"
    assert "训练方案只依赖成功的调用" in section, "轨迹行句式必须在场"
    assert "如实缺席，不编造轨迹" in section, "直接启动缺位态口径必须写明"


def test_acceptance_spec_citation_docs_pinned():
    """验收冻结引用任务规约钉死(R64 切片 B):四要素快照、规约摘要先行与旧记录缺位态。"""
    design = TASK_SPEC_DESIGN.read_text(encoding="utf-8")
    # 设计文档 Phase 2:切片 B 如实登记,「转入后续候选」欠账句随之撤下。
    assert "（切片 B，2026-09-28）" in design, "切片 B 落地必须如实登记"
    assert "冻结时引用的任务规约口径" in design, "规约口径段名必须点名"
    assert "`spec_anchor_lines` 单一来源" in design, "段渲染单一来源必须写明"
    assert "转入后续候选" not in design, "欠账句必须随切片 B 落地撤下"
    section = _section(
        AGENT_SETUP.read_text(encoding="utf-8"),
        "## 用独立测试题做单模型业务验收",
        "## 从评测结果进入下一轮改进",
    )
    assert "任务规约四要素" in section and "`task_spec` 键" in section, (
        "冻结时四要素快照进验收记录必须写明"
    )
    assert "冻结前先向 stderr 打印同一份规约摘要" in section, (
        "acceptance-prepare 冻结前摘要先行必须写明"
    )
    assert "冻结时引用的任务规约口径" in section, "摘要规约口径段必须写明"
    assert "「📋 任务规约（冻结验收条款前的口径）」" in section, (
        "冻结表单旁折叠区 label 必须逐字写明"
    )
    assert "如实没有该段" in section, "旧记录缺位态必须写明"


def test_training_guidance_docs_pinned():
    """训练参数/分区设置引导单一来源钉死(R65 切片①):三函数、同源同词汇、非本产品实测边界。"""
    text = AGENT_SETUP.read_text(encoding="utf-8")
    assert "src/workbench/training_guidance.py" in text, "单一来源模块路径必须写明"
    assert text.count("src/workbench/training_guidance.py") >= 2, "分区段与训练段各点名单一来源"
    for func in (
        "manual_training_parameter_lines",
        "split_settings_guidance_lines",
        "small_test_set_line",
    ):
        assert func in text, f"引导函数缺少说明: {func}"
    assert "同源同词汇" in text, "页面与 CLI 同口径承诺必须写明"
    assert "非本产品实测" in text, "推荐值与学习率分档的诚实边界必须写明"
    # 分区段:「何时该改」引导与 <30 条条数提醒(1/N 算术,不是统计保证,不替用户决定比例)。
    split_section = _section(
        text, "## 生成独立数据分区与版本", "## 时间预测任务：先核对来源与标签窗口"
    )
    assert "split_settings_guidance_lines" in split_section, "分区引导函数必须在分区段落位"
    assert "small_test_set_line" in split_section, "条数提醒函数必须在分区段落位"
    assert "何时该改" in split_section, "分区引导主题必须写明"
    assert "少于 30 条" in split_section, "条数提醒阈值必须写明"
    assert "不是统计保证" in split_section, "阈值诚实边界必须写明"
    # 训练段:逐参数大白话与推荐起步值,诚实边界照抄模块 docstring 口径。
    train_section = _section(text, "## 在同一任务中启动真实训练", "## 允许一次显存不足技术恢复")
    assert "manual_training_parameter_lines" in train_section, "手工参数函数必须在训练段落位"
    assert "参数大白话" in train_section and "推荐起步值" in train_section, (
        "大白话与起步值主题必须写明"
    )
    assert "外部指南的汇总启发" in train_section, "诚实边界措辞必须照抄模块口径"
    # 试用记录介入点清单第 6/7 条:现状补充逐条在场,历史原文未改写。
    trial = USER_TRIAL_LOG.read_text(encoding="utf-8")
    item6 = _section(trial, "6. - [ ] 零密钥路径", "7. - [ ]")
    item7 = _section(trial, "7. - [ ] 「分区设置」", "8. - [ ]")
    assert "2026-09-28 现状补充" in item6, "介入点清单第 6 条缺现状补充"
    assert "2026-09-28 现状补充" in item7, "介入点清单第 7 条缺现状补充"
    assert "逐参数大白话与推荐起步值" in item6, "第 6 条补充必须点名大白话与起步值"
    assert "src/workbench/training_guidance.py 单一来源" in item6, "第 6 条补充必须点名单一来源"
    assert "何时该改" in item7 and "少于 30 条" in item7, "第 7 条补充必须点名引导与条数提醒"


def test_echo_triage_docs_pinned():
    """回声预警事实分流单一来源钉死(R66):echo_triage_lines、按报告内事实分流、
    不认定原因边界与试用记录第 1 条现状补充。"""
    section = _section(
        AGENT_SETUP.read_text(encoding="utf-8"),
        "## 比较基座与本轮微调效果",
        "## 让 Agent 解读结果与下一步",
    )
    assert "src/workbench/evaluation_diagnostics.py" in section, "单一来源模块路径必须写明"
    assert "`echo_triage_lines` 单一来源" in section, "回声分流单一来源必须点名函数"
    assert "`summarize_comparison`" in section, "CLI 摘要取词位点必须写明"
    assert "同源同词汇" in section, "页面与 CLI 同口径承诺必须写明"
    assert "按报告内事实分流" in section, "分流主题必须写明"
    assert "回声题是否同时触及生成长度上限" in section, "截断分流方向必须写明"
    assert "1.5 倍" in section, "输入长度分流阈值必须写明"
    assert "两个平均值同时给出供人自行核对" in section, "核对口径必须写明"
    assert "补全式（Alpaca）" in section, "模板方向只陈述已记录事实必须写明"
    assert "无法用报告内事实分流" in section, "模板缺位态口径必须写明"
    assert "以上是按报告内事实排出的核查顺序，不认定原因" in section, "固定收尾句必须写明"
    assert "每改一项后用同一题集复测一次" in section, "复测口径必须写明"
    assert "不认定原因" in section, "观察事实边界必须写明"
    # 试用记录介入点清单第 1 条:现状补充在场,历史原文未改写。
    trial = USER_TRIAL_LOG.read_text(encoding="utf-8")
    item1 = _section(trial, "1. - [ ] 对照预警", "2. - [ ]")
    assert "2026-09-28 现状补充" in item1, "介入点清单第 1 条缺现状补充"
    for phrase in ("核查顺序", "同源", "evaluation_diagnostics"):
        assert phrase in item1, f"第 1 条补充必须点名 {phrase}"
    assert "echo_triage_lines" in item1, "第 1 条补充必须点名单一来源函数"
    assert "能否独立选出正确的核查方向" in item1, "第 1 条历史原文必须保留"


def test_three_model_delta_docs_pinned():
    """三模型对照题数差单一来源钉死(R67):three_model_delta_lines、分数差换算题数差、
    单题分辨率与不作统计结论边界,试用记录第 11 条现状补充在场。"""
    section = _section(
        AGENT_SETUP.read_text(encoding="utf-8"),
        "## 从评测结果进入下一轮改进",
        "## 配置文件与优先级",
    )
    assert "src/workbench/report_summary.py" in section, "单一来源模块路径必须写明"
    assert "`three_model_delta_lines` 单一来源" in section, "题数差单一来源必须点名函数"
    assert "`summarize_iteration`" in section, "轮次摘要取词位点必须写明"
    assert "同源同词汇" in section, "页面与 CLI 同口径承诺必须写明"
    assert "题数差" in section, "分数差换算题数差主题必须写明"
    assert "每题约占" in section, "单题分辨率措辞必须写明"
    assert "1 题量级" in section, "1 题量级参考价值边界必须写明"
    assert "不作统计结论" in section, "统计边界必须写明"
    assert "对应行不出现" in section, "缺位态口径必须写明"
    # 试用记录介入点清单第 11 条:现状补充在场,历史原文未改写。
    trial = USER_TRIAL_LOG.read_text(encoding="utf-8")
    item11 = _section(trial, "11. - [ ] 三模型对照表", "12. - [ ]")
    assert "2026-09-28 现状补充" in item11, "介入点清单第 11 条缺现状补充"
    assert "three_model_delta_lines" in item11, "第 11 条补充必须点名单一来源函数"
    assert "题数差" in item11 and "单题分辨率" in item11, "第 11 条补充必须点名换算与分辨率"
    assert "能否看懂" in item11, "第 11 条历史原文必须保留"


def test_acceptance_gate_docs_pinned():
    """验收门槛分辨率算术单一来源钉死(R68):acceptance_gate_lines、通过率换算需通过/容错、
    每题分辨率与不作统计结论边界,试用记录第 9 条现状补充在场。"""
    section = _section(
        AGENT_SETUP.read_text(encoding="utf-8"),
        "## 用独立测试题做单模型业务验收",
        "## 从评测结果进入下一轮改进",
    )
    assert "src/workbench/report_summary.py" in section, "单一来源模块路径必须写明"
    assert "`acceptance_gate_lines` 单一来源" in section, "门槛算术单一来源必须点名函数"
    assert "同源同词汇" in section, "页面与 CLI 同口径承诺必须写明"
    assert "需通过" in section and "容错" in section, "通过率换算需通过/容错题数必须写明"
    assert "每题占通过率" in section, "每题分辨率措辞必须写明"
    assert "证据不足" in section, "最低题数超标的预告口径必须写明"
    assert "不作统计结论" in section, "统计边界必须写明"
    # 试用记录介入点清单第 9 条:现状补充在场,历史原文未改写。
    trial = USER_TRIAL_LOG.read_text(encoding="utf-8")
    item9 = _section(trial, "9. - [ ] 最终验收门槛", "10. - [ ]")
    assert "2026-09-28 现状补充" in item9, "介入点清单第 9 条缺现状补充"
    assert "acceptance_gate_lines" in item9, "第 9 条补充必须点名单一来源函数"
    assert "需通过" in item9 and "容错" in item9, "第 9 条补充必须点名换算口径"
    assert "同一式" in item9, "第 9 条补充必须点名算术与执行判定同一式"
    assert "软件不替你设定业务门槛" in item9, "第 9 条历史原文必须保留"
    assert "数值本身是否需要专家意见" in item9, "第 9 条历史原文必须保留"


def test_zero_score_remedy_docs_pinned():
    """零分对照失败原因对号处理钉死(R69):summarize_comparison 零分分支、对号行、
    补数据优先级与全技术性零分定性撤回,试用记录第 2 条现状补充在场。"""
    section = _section(
        AGENT_SETUP.read_text(encoding="utf-8"),
        "## 比较基座与本轮微调效果",
        "## 让 Agent 解读结果与下一步",
    )
    assert "src/workbench/report_summary.py" in section, "单一来源模块路径必须写明"
    assert "`summarize_comparison` 零分分支" in section, "零分对号单一来源必须点名函数"
    assert "失败原因对号处理" in section, "对号行措辞必须写明"
    assert "补数据治不了回声" in section, "复述→改模板不补数据的映射必须写明"
    assert "逐一排除后的选项" in section, "补数据优先级边界必须写明"
    assert "学不出这个任务" in section and "撤回" in section, "全技术性零分定性撤回必须写明"
    assert "不作统计结论" in section, "统计边界必须写明"
    # 试用记录介入点清单第 2 条:现状补充在场,历史原文未改写。
    trial = USER_TRIAL_LOG.read_text(encoding="utf-8")
    item2 = _section(trial, "2. - [ ]", "3. - [ ]")
    assert "2026-09-29 现状补充" in item2, "介入点清单第 2 条缺现状补充"
    assert "summarize_comparison" in item2, "第 2 条补充必须点名单一来源函数"
    assert "对号处理" in item2, "第 2 条补充必须点名对号口径"
    assert "撤回「学不出这个任务」定性" in item2, "第 2 条补充必须点名定性撤回"
    assert "微调后仍是零分" in item2, "第 2 条历史原文必须保留"
    assert "补数据、改任务定义还是停止" in item2, "第 2 条历史原文必须保留"


def test_mismatch_triage_docs_pinned():
    """盲标核验三因分辨钉死(R70):mismatch_triage_lines 单一来源、三种分辨方向、
    对号修正与防背题、通过态如实缺席,试用记录第 5 条现状补充在场。"""
    section = _section(
        AGENT_SETUP.read_text(encoding="utf-8"),
        "### 盲标核验的完整 CLI 用法",
        "## 生成独立数据分区与版本",
    )
    assert "src/workbench/intake_service.py" in section, "单一来源模块路径必须写明"
    assert "`mismatch_triage_lines` 单一来源" in section, "三因分辨单一来源必须点名函数"
    assert "`mismatch_triage` 键" in section, "记录键名必须写明"
    assert "同一套类别词汇" in section, "词汇分辨方向必须写明"
    assert "口径或边界没对齐" in section, "同向错位分辨方向必须写明"
    assert "不像随机记错" in section, "同向错位的统计性质必须写明"
    assert "先展开这条记录核对输入信息" in section, "仅 1 处的核查起点必须写明"
    assert "逐条展开核对" in section, "分散错位方向必须写明"
    assert "修正数据标签（改数据）" in section and "改方案" in section, "对号修正映射必须写明"
    assert "换一组题" in section, "防背题事实必须写明"
    assert "不认定原因" in section, "分辨边界（只给方向）必须写明"
    assert "如实缺席" in section, "通过态/早期存档无键的边界必须写明"
    # 试用记录介入点清单第 5 条:现状补充在场,历史原文未改写。
    trial = USER_TRIAL_LOG.read_text(encoding="utf-8")
    item5 = _section(trial, "5. - [ ] 盲标核验未通过", "6. - [ ]")
    assert "2026-09-29 现状补充" in item5, "介入点清单第 5 条缺现状补充"
    assert "三因分辨" in item5, "第 5 条补充必须点名三因分辨"
    assert "mismatch_triage_lines" in item5, "第 5 条补充必须点名单一来源函数"
    assert "对号修正" in item5, "第 5 条补充必须点名对号修正"
    assert "不认定原因" in item5, "第 5 条补充必须写明分辨边界"
    assert "三种原因的分辨与修正动作" in item5, "第 5 条历史原文必须保留"
    assert "改数据 or 改方案" in item5, "第 5 条历史原文必须保留"


def test_low_baseline_triage_docs_pinned():
    """探针低于基线方向分辨钉死(R71):low_baseline_triage_lines 单一来源、四路
    分流、对号处理、label_vocabulary 如实降级、弱信号改标签门槛,试用记录第 3、
    4 条现状补充在场。"""
    section = _section(
        AGENT_SETUP.read_text(encoding="utf-8"),
        "- **可学性探针（可选证据，非门禁）**：",
        "这三道关卡都不使用 LLM 判断",
    )
    assert "src/workbench/learnability_probe.py" in section or (
        "`low_baseline_triage_lines` 单一来源" in section
    ), "单一来源必须点名"
    assert "`low_baseline_triage_lines` 单一来源" in section
    assert "先加大 `max_new_tokens` 重测" in section, "截断分流方向必须写明"
    assert "不在这份开发集的标签里出现过" in section, "词汇分流方向必须写明"
    assert "没有用任务的答案词汇作答" in section, "词汇分流的理由必须写明"
    assert "没有按输入区分作答" in section, "同答分流方向必须写明"
    assert "任务定义或标注口径" in section, "任务定义分流方向必须写明"
    assert "不动数据" in section and "改任务定义" in section, "对号处理映射必须写明"
    assert "保留低分证据" in section, "如实保留边界必须写明"
    assert "不认定原因" in section, "分辨边界（只给方向）必须写明"
    assert "`label_vocabulary`" in section, "记录字段必须写明"
    assert "如实降级" in section, "旧记录降级边界必须写明"
    assert "`WEAK_SIGNAL_RULE` 单一来源" in section, "弱信号门槛单一来源必须点名"
    assert "单凭模型不认同不改标签" in section, "改标签门槛必须写明"
    assert "人工核对后仍不认同才修正数据" in section
    # 试用记录介入点清单第 3、4 条:现状补充在场,历史原文未改写。
    trial = USER_TRIAL_LOG.read_text(encoding="utf-8")
    item3 = _section(trial, "3. - [ ] 探针判定", "4. - [ ]")
    assert "2026-09-29 现状补充" in item3, "介入点清单第 3 条缺现状补充"
    assert "low_baseline_triage_lines" in item3, "第 3 条补充必须点名单一来源函数"
    assert "方向分辨" in item3, "第 3 条补充必须点名方向分辨"
    assert "label_vocabulary" in item3, "第 3 条补充必须点名标签全数字段"
    assert "不认定原因" in item3, "第 3 条补充必须写明分辨边界"
    assert "能否独立分辨是提示模板问题还是任务定义问题" in item3, "第 3 条历史原文必须保留"
    item4 = _section(trial, "4. - [ ] 探针「标签问题候选", "5. - [ ]")
    assert "2026-09-29 现状补充" in item4, "介入点清单第 4 条缺现状补充"
    assert "WEAK_SIGNAL_RULE" in item4, "第 4 条补充必须点名单一来源常量"
    assert "单凭模型不认同不改标签" in item4, "第 4 条补充必须写明门槛"
    assert "原始来源行" in item4, "第 4 条补充必须写明弱信号不冒充溯源"
    assert "弱信号供参考" in item4, "第 4 条历史原文必须保留"


def test_answer_sheet_docs_pinned():
    """待补答案清单钉死(R72):answer_sheet 单一来源、填写表结构、三条规则、
    诚实降级路径、交接物不是回传文件,试用记录第 10 条现状补充在场。"""
    section = _section(
        AGENT_SETUP.read_text(encoding="utf-8"),
        "缺答案的行不再只是一段页面提示",
        "### 连续数值答案的如实边界",
    )
    assert "src/workbench/answer_sheet.py" in section, "单一来源模块路径必须写明"
    assert "单一来源" in section, "单一来源必须点名"
    assert "`answer-sheet SESSION_ID" in section, "CLI 命令名必须写明"
    assert "导出待补答案清单（交给填写人）" in section, "页面按钮文案必须写明"
    assert "待填答案" in section, "填写表列结构必须写明"
    assert "按行ID 对号回填" in section, "回传对号机制必须写明"
    assert "不要凭猜测填" in section, "防编造门槛必须写明"
    assert "监督信号" in section, "监督信号后果必须写明"
    assert "留空" in section and "任务定义问题" in section, "输入不足处置必须写明"
    assert "比编一个答案更有价值" in section, "留空优于编造必须写明"
    assert "结构上不可能泄露监督标签" in section, "防泄露边界必须写明"
    assert "当前没有缺答案的行，无需导出清单" in section, "空态诚实降级必须写明"
    assert "先运行 analyze" in section, "无预览降级必须写明"
    assert "或零密钥的 baseline-analyze" in section, "零密钥路径必须一并点名"
    assert "交接物不是回传文件" in section, "回传通道边界必须写明"
    assert "`missing_answer_rows`" in section, "行筛选与判定共用必须写明"
    assert "既定保留" in section, "未成熟标签不混入必须写明"
    # 试用记录介入点清单第 10 条:现状补充在场,历史原文未改写。
    trial = USER_TRIAL_LOG.read_text(encoding="utf-8")
    item10 = _section(trial, "10. - [ ] 「需补齐", "11. - [ ]")
    assert "2026-09-29 现状补充" in item10, "介入点清单第 10 条缺现状补充"
    assert "answer_sheet" in item10, "第 10 条补充必须点名单一来源模块"
    assert "answer-sheet" in item10, "第 10 条补充必须点名 CLI 命令"
    assert "行ID 对号回填" in item10, "第 10 条补充必须点名回传对号机制"
    assert "不要凭猜测" in item10, "第 10 条补充必须写明防编造门槛"
    assert "未成熟标签行不混入" in item10, "第 10 条补充必须写明筛选边界"
    assert "能否找到了解业务的人并按指引回传文件" in item10, "第 10 条历史原文必须保留"


def test_demo_task_docs_pinned():
    """内置演示任务文档锚点:入口边界/摘要门控/诚实降级必须与产品原文一致。"""
    text = AGENT_SETUP.read_text(encoding="utf-8")
    section = _section(text, "### 内置演示任务", "### 多份资料一起分析")
    assert "第一次使用？用内置演示任务开始" in section, "入口文案必须与页面一致"
    assert "src/workbench/demo_task.py" in section, "单一来源模块必须点名"
    assert "虚构" in section, "数据虚构属性必须写明"
    assert "与真实任务完全相同" in section, "关卡一致性必须写明"
    assert "没有预设结论" in section, "无预设结论必须写明"
    assert "不跳过任何门禁" in section, "不跳门禁必须写明"
    assert "按文件内容摘要比对" in section, "摘要门控机制必须写明"
    assert "同名不同内容不算" in section, "摘要与文件名区分必须写明"
    assert "不会混进任何真实任务" in section, "隔离边界必须写明"
    assert "如实隐藏" in section, "诚实降级必须写明"
    assert "demo_sample" in section, "降级机制必须点名单一来源函数"
    assert "不编造演示数据" in section, "不编造门槛必须写明"
    # 试用记录冷启动补充在场,历史结构未改写。
    trial = USER_TRIAL_LOG.read_text(encoding="utf-8")
    assert "第一次使用？用内置演示任务开始" in trial, "试用记录缺冷启动入口补充"
    assert "src/workbench/demo_task.py" in trial, "试用记录补充必须点名单一来源模块"


def test_demo_cli_parity_docs_pinned():
    """演示任务 CLI 平权钉死(R80):create --demo 与 baseline-analyze 的同源、
    互斥、诚实拒绝边界在 agent-setup 写明——零密钥旅程不再只属于页面。"""
    section = _section(
        AGENT_SETUP.read_text(encoding="utf-8"), "### 内置演示任务", "### 多份资料一起分析"
    )
    assert "CLI 有同源的两个零密钥入口" in section, "CLI 平权主题句必须在场"
    assert "create --demo" in section, "演示冷启动 CLI 命令必须写明"
    assert "不收 `--input`/`--goal`" in section, "互斥边界必须写明"
    assert "`--scope full`" in section, "互斥清单必须覆盖 scope=full 矛盾"
    assert "逐一点名并直接报错，不静默忽略" in section, "不静默忽略边界必须写明"
    assert "baseline-analyze SESSION_ID --target 答案列" in section, "基础分析 CLI 必须写明"
    assert "[--instruction 补充指令]" in section, "CLI 独有的补充指令参数必须写明"
    assert "没有 Agent 服务？用基础分析开始" in section, "必须与页面入口词汇一致"
    assert "propose_baseline_analysis" in section, "单一来源函数必须点名"
    assert "baseline-deterministic" in section, "确定性方案的 model 记名必须写明"
    assert "CLI 同口径，不悄悄覆盖" in section, "已有分析时的同口径边界必须写明"
    assert "可调整字段重新生成" in section, "基础→基础替换(R82)必须写明"
    assert "预览与确认状态随之失效" in section, "替换的失效语义必须与 Agent 重分析同口径"
    assert "needs_data_revision 的零密钥出口" in section, "R82 死胡同根治定位句必须在场"
    assert "已替换此前的基础分析" in section, "替换时 stderr 先说明必须写明"
    assert "逐一点名缺哪些" in section, "--temporal 缺参点名必须写明"
    assert "不静默退回随机切分" in section, "时间分区不静默降级必须写明"
    assert "summarize_analysis" in section, "人话摘要同源必须点名"
    assert "--target 类别 --group 编号 --exclude 处理结果" in section, (
        "演示冷启动组合命令必须逐字可照抄"
    )


def test_quickstart_onboarding_docs_pinned():
    """快速开始与本地模型准备钉死(R74):演示入口文案与页面一致、零密钥承诺、
    hf download 可照抄命令、自动发现位点与镜像指引;agent-setup 同源分界钉死。"""
    quickstart = _section(README.read_text(encoding="utf-8"), "### 快速开始", "### 显存参考")
    assert "第一次使用？用内置演示任务开始" in quickstart, "演示入口文案必须与页面一致"
    assert "不需要任何 API 密钥" in quickstart, "零密钥承诺必须写明"
    assert "hf download Qwen/Qwen3-0.6B --local-dir models/Qwen3-0.6B" in quickstart, (
        "下载命令必须可照抄"
    )
    assert "HF_ENDPOINT=https://hf-mirror.com" in quickstart, "镜像指引必须在场"
    assert "本机已准备的候选模型" in quickstart, "自动发现位点必须与页面词汇一致"
    assert "docs/agent-setup.md" in quickstart, "完整说明链接必须在场"
    section = _section(
        AGENT_SETUP.read_text(encoding="utf-8"),
        "## 准备本地基础模型",
        "## 让 Agent 推荐训练方案",
    )
    assert "只读取 tokenizer" in section, "阶段分界(tokenizer)必须写明"
    assert "可学性探针与训练需要完整权重" in section, "阶段分界(完整权重)必须写明"
    assert "不自动下载模型" in section, "不自动下载边界必须写明"
    assert "hf download Qwen/Qwen3-0.6B --local-dir models/Qwen3-0.6B" in section, (
        "命令必须与 README 快速开始同源"
    )
    assert "discover_local_models" in section, "发现机制必须点名实现"
    assert "只读文件、不加载权重" in section, "发现边界必须写明"
    assert "文件完整只表示可以进一步检查" in section, "文件完整边界必须与 model-list 同口径"
    assert "HF_ENDPOINT=https://hf-mirror.com" in section, "镜像指引必须在场"


def test_quickstart_model_and_install_guidance_consistent():
    """快速开始一致性钉死(R75):点名的 0.6B 在中文显存表可查且数字与英文表同源,
    国内 pip 镜像指引与 HF 镜像指引同场。"""
    text = README.read_text(encoding="utf-8")
    quickstart = _section(text, "### 快速开始", "### 显存参考")
    assert "mirrors.aliyun.com/pypi/simple/" in quickstart, "国内 pip 镜像指引必须在场"
    vram = _section(text, "### 显存参考")
    assert "| Qwen3-0.6B | ~1.2 GB | ~1 GB |" in vram, "快速开始点名的 0.6B 必须在显存表可查"
    assert "| Qwen3-1.7B | ~2.0 GB | ~2 GB |" in vram, "次小档 1.7B 同步在场"
    # 数字与英文 Model Compatibility 表同源:两表不各说各话。
    assert "| Qwen3 0.6B | ~1.2 GB | ~1 GB |" in text, "英文表 0.6B 行必须在场"
    assert "| Qwen3 1.7B | ~2.0 GB | ~2 GB |" in text, "英文表 1.7B 行必须在场"


def test_english_quickstart_onboarding_parity_pinned():
    """英文 Quick Start 对齐钉死(R76):演示入口、零密钥承诺、本地模型准备与
    自动发现词汇与中文快速开始同一承诺;命令逐字同源。"""
    text = README.read_text(encoding="utf-8")
    quickstart = _section(text, "## 🚀 Quick Start", "## 🧙")
    assert "「第一次使用？用内置演示任务开始」" in quickstart, "英文版演示入口文案必须与页面一致"
    assert "no API key at all" in quickstart, "零密钥承诺必须有英文对照"
    assert "hf download Qwen/Qwen3-0.6B --local-dir models/Qwen3-0.6B" in quickstart, (
        "下载命令必须与中文快速开始逐字同源"
    )
    assert "「本机已准备的候选模型」" in quickstart, "自动发现词汇必须与页面一致"
    assert "auto-discovered" in quickstart, "自动发现行为必须有英文说明"
    # 强锚:镜像句与链接句钉新段独有措辞——裸命令/裸链接在既有 Option 3 块中已出现,弱 pin 挡不住删段漂移。
    assert "In China, run `export HF_ENDPOINT=https://hf-mirror.com` first" in quickstart, (
        "英文侧镜像指引必须与中文同场"
    )
    assert "for the full local-model guide" in quickstart, "完整说明链接必须绑住新段"


def test_live_dashboard_link_honestly_scoped():
    """在线仪表盘链接如实定界(R78):托管实例不含目标与数据主线与零密钥演示任务,
    Option 2 必须在链接旁如实说明——零安装通道不得过度承诺(诚实红线)。"""
    option2 = _section(
        README.read_text(encoding="utf-8"),
        "### Option 2: Try the Live Dashboard",
        "### Option 3: CLI training",
    )
    assert "https://benluo.art/qlora-dashboard/" in option2, "在线链接必须在场"
    assert "Goal & Data" in option2, "定界必须点名缺失的主线工作流"
    assert "zero-key built-in demo task" in option2, "定界必须点名零密钥演示任务不在托管实例"
    assert "run it locally with Option 1" in option2, "主线必须如实指向本地安装路径"


def test_hero_gif_is_real_recorded_asset():
    """hero GIF 实物钉死(R79):真实 UI 录制的零密钥演示旅程替换裂图占位。

    README hero 原引用不存在的 dashboard.gif(裂图),此前以 DOCUMENTED_PENDING_ASSETS
    显式豁免;R79 用本地 Streamlit + Playwright 逐门禁截图组装成实物。本测试钉死:
    文件真实存在、体积守住录制指南的 5MB 约束、占位说明句与 TODO 注释随之撤下、
    hero 说明如实标注零密钥与虚构数据(诚实红线)。
    """
    gif = ROOT / "docs" / "assets" / "dashboard.gif"
    assert gif.exists(), "hero GIF 必须真实存在(待录制豁免已随实物落地撤销)"
    assert gif.stat().st_size < 5 * 1024 * 1024, "GIF 体积必须小于 5MB(录制指南约束)"

    text = README.read_text(encoding="utf-8")
    assert "Replace this with a 30s GIF" not in text, "占位说明句必须随实物落地撤下"
    assert "RECORDING_TODO" not in text, "录制 TODO 注释必须随实物落地清理"
    hero = _section(text, "# TuneSmith", "## 📌")
    assert "docs/assets/dashboard.gif" in hero, "hero 必须引用实物 GIF"
    assert "no API key" in hero, "hero 说明必须如实标注零密钥旅程"
    assert "fictional" in hero, "hero 说明必须如实标注演示数据为虚构"
    # 锚点必须真实解析:GitHub 只给标题(h1–h6)生成锚 id,<summary> 不生成——
    # hero 链接必须指向 details 内真标题的锚(em dash 删除后双空格→双连字符,
    # 与目录 #-distributed-training-fsdp--deepspeed 同规则;summary 无锚,勿再钉死锚)。
    assert "[Recording Guide](#how-the-hero-gif-was-captured--and-how-to-regenerate-it)" in hero, (
        "hero 的指南锚点必须指向 details 内真标题锚"
    )
    assert "### How the hero GIF was captured — and how to regenerate it" in text, (
        "锚点目标标题必须在场"
    )
