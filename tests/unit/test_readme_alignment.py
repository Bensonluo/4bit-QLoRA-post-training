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
    assert "此前的确认可能是盲点头" in section, "配错提示必须写明"
    assert "尚未核验" in section and "建议先运行 `contrast-check`" in section
    assert "软门禁" in section and "不阻断确认" in section, "软门禁边界必须写明"
    assert "提交不收 `--revision`" in section, "提交参数边界必须写明"
    assert "自动失效" in section, "失效规则必须写明"


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
    assert "scoring-list` 只列清单" in section, "list 例外必须写明"
    assert "软件不会自动确认评分规则" in section, "草稿态不自动确认边界必须写明"
    assert "绑定当前业务目标与输入/答案语义" in section, "已确认规则的绑定语义必须写明"
    assert "数据修订后兼容规则可继续用" in section, "兼容复用与重新确认的分界必须写明"
    assert "当前没有可确认的评分方案" in section, "澄清态口径必须写明"
    assert "不等于严格准确率" in section, "均值/通过率不作严格准确率必须写明"
    assert "不构成业务达标的判断" in section, "固定边界句必须写明"
    assert "summarize_scoring" in section, "页面渲染位点必须点名摘要函数"
    assert "同源同词汇" in section, "页面与 CLI 同口径承诺必须写明"


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
    assert "plan-list` 只列清单" in section, "list 例外必须写明"
    assert "方案可供确认" in section, "状态三态(ready)必须写明"
    assert "需要先完善数据" in section, "状态三态(needs_data)必须写明"
    assert "当前条件不支持" in section, "状态三态(unsupported)必须写明"
    assert "推荐理由与尚未验证的限制原文" in section, "理由/限制如实复述必须写明"
    assert "需要你先回答的业务问题" in section, "待答业务问题必须写明"
    assert "LoRA rank" in section, "关键参数点名必须写明"
    assert "probe" in section, "预检证据的真实存放位点必须写明"
    assert "不会自动启动" in section, "确认准备的不启动边界必须写明"
    assert "不构成训练效果或业务达标的判断" in section, "固定边界句必须写明"
