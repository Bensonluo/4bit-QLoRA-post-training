"""02 评测结果页空态闭环指引（R121）+ 通用文案残留收口。

R120 把「用我的 test 集评测」接到用户自己的 test.json，结果落
domains/entity_matching/data/results/ → 本页 load_eval_data 自动取最新。
但用户在评测前先到本页（或评测失败后回来看）撞上的是死胡同空态
「1. 运行领域评测脚本」——无路径无指引，通用旅程最后一环缺「怎么让
这里出现结果」。R121 空态对 entity_matching 域给出页内一键路径
（训练实验室 → 训练动态 → 下一步 → ⚡ 用我的 test 集评测）+ 终端等价命令。

同轮收口两处通用化文案残留（grep 一手核实为 ui/ 最后两处医疗/主数据示例）：
- 02 页「暂无评测领域」空态的示例命令 domains/medical_entity/evaluate.py
  → scripts/eval_entity_match.py（任意领域 test 集通用）；
- 06_Chat 系统提示词 help「垂类任务（如主数据匹配）」→「实体匹配/名称
  归一化任务（如供应商名归一化）」。

保护性钉：本页 3×「扫描并导入 MLflow」按钮与 3× 导入 spinner 是
test_register_unification_ui / test_loading_feedback_ui 的既有钉，
空态改造不得动它们（本文件镜像钉一次，防自伤）。
"""

from pathlib import Path

import pytest

pytest.importorskip("streamlit")

ROOT = Path(__file__).resolve().parents[2]
PAGE_EVAL = ROOT / "ui" / "pages" / "02_Evaluation.py"
PAGE_CHAT = ROOT / "ui" / "pages" / "06_Chat.py"


def _source(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_entity_matching_empty_state_guides_back_to_lab():
    """通用域空态闭环钉：entity_matching 无结果时必须给出「怎么让这里出现
    结果」——页内一键路径（训练实验室 → 下一步 → ⚡ 用我的 test 集评测）
    + 终端等价命令，而非死胡同「运行领域评测脚本」。"""
    source = _source(PAGE_EVAL)
    guard_pos = source.find('if selected_domain == "entity_matching":')
    assert guard_pos != -1, "空态必须对 entity_matching 域分支（通用域主路）"
    # 区间钉：分支体 = guard → 该分支的导入 expander 之前（含 CLI 命令）
    block_end = source.find('st.markdown("2. 在下方导入历史评测结果")', guard_pos)
    assert block_end != -1, "空态分支必须保留导入历史结果的出路"
    block = source[guard_pos:block_end]
    assert "怎么让这里出现结果" in block
    assert "训练实验室" in block and "🧭 下一步" in block, (
        "必须指路训练实验室的下一步面板（非专家不知道评测入口在哪）"
    )
    assert "⚡ 用我的 test 集评测" in block, "必须点名与 00 页一致的按钮文案"
    assert "Data Wizard" in block, "必须说明 test.json 的来源（Data Wizard 导出）"
    assert "数据向导" in block, "R135：来源指引必须带中文名（首页/页面标题均以数据向导为主名）"
    assert "scripts/eval_entity_match.py" in block and "--test-file" in block, (
        "终端等价命令必须在场（与 00 页下一步面板的双路惯例一致）"
    )


def test_fallback_empty_state_kept_for_case_domains():
    """非通用域（如医疗案例域）空态维持原可选操作文案——案例域的用户
    仍走领域评测脚本路径，不误导他们去找不存在的 wizard test 门。"""
    source = _source(PAGE_EVAL)
    assert "**可选操作：**" in source and "1. 运行领域评测脚本" in source, (
        "else 分支（案例域）原文案必须保留"
    )
    # 极性钉：通用分支在前、案例回退在后（两分支文案分别被上/本钉锚住）
    guard_pos = source.find('if selected_domain == "entity_matching":')
    fallback_pos = source.find('"**可选操作：**"')
    assert -1 < guard_pos < fallback_pos, "通用分支必须在前，案例域回退在后"


def test_import_entry_pins_survive_empty_state_rewrite():
    """镜像钉（防自伤）：既有 3× 导入按钮与 3× spinner 钉在空态改造后
    必须原样在场——test_register_unification_ui / test_loading_feedback_ui
    的钉不因 R121 重写而失锚。"""
    source = _source(PAGE_EVAL)
    assert source.count('st.button("扫描并导入 MLflow"') == 3
    assert source.count('with st.spinner("正在扫描并导入历史评测结果到 MLflow…"):') == 3


def test_no_domains_example_is_generic():
    """「暂无评测领域」空态示例命令通用化钉：示例必须是
    scripts/eval_entity_match.py（任意领域），医疗领域脚本不再作唯一示例。"""
    source = _source(PAGE_EVAL)
    assert "domains/medical_entity/evaluate.py" not in source, (
        "02 页不得再以医疗领域脚本为示例（generic-first 残留收口）"
    )
    # 通用 CLI 至少出现两次：无领域空态示例 + entity_matching 空态等价命令
    assert source.count("scripts/eval_entity_match.py") >= 2


def test_chat_system_prompt_help_generic():
    """06_Chat 系统提示词 help 通用化钉：示例从「垂类任务（如主数据匹配）」
    换成通用实体匹配示例（供应商名归一化）——sidebar 帮助文案不再以
    仓库内案例领域为默认心智。"""
    source = _source(PAGE_CHAT)
    assert "垂类任务" not in source and "主数据匹配" not in source
    assert "供应商名归一化" in source
