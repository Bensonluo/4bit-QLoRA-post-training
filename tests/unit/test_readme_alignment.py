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


# README 录制指南明确声明 dashboard.gif 为待录制的占位资源(保存路径已写死)。
# 除该显式占位外,任何仓库内引用都必须真实存在。
DOCUMENTED_PENDING_ASSETS = {"docs/assets/dashboard.gif"}


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


def _cli_help_text(monkeypatch, capsys, tmp_path, *command):
    """进程内跑真实 CLI 的 --help(参考 test_label_verify_cli 的 argv 注入模式)。"""
    from scripts import data_intake

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
