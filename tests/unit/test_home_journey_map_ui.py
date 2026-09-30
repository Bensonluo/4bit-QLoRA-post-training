"""app.py 首页快捷操作 → 全流程一览（R123）：4 钮 → 8 页旅程网格。

选点依据：旧版快捷操作只有 00/01/02/03 四钮，05 数据向导 / 04 模型注册 /
06 对话验证 / 07 目标与数据 全靠侧栏摸索；README 已宣称「8 页覆盖全生命
周期」（README:388），首页却只露出 4 页。非专家新用户问「从哪开始、一共
几步」时首页自身无法回答（用户在 S1497 真实问过）——旅程网格按序排列 +
每步一句说明，首页一屏直达全部 8 页。

钉的设计（R121 规则：动用户可见文案/按钮必须 grep 全测试目录查受影响钉
——已查：test_register_unification_ui.py:153-158 对旧四钮文案只做「在场」
断言，扩展不删锚即全绿；test_home_empty_state_ui.py 的空态钉不涉本节）：
① 8 页全覆盖钉：8 个旅程按钮 + 8 个 switch_page 目标全在场；
② 旅程排序钉：05→00→01→02→03→04→06→07 源码序递增 + 引导 caption；
③ 接线实证钉：点「🧙 数据向导」元素树切到 05（app.py 入口直跑机制，
   test_home_empty_state_ui.py 场景 D 同款）——新钮不是幽灵按钮；
④ 保活钉：旧四钮字面不动（test_register_unification 的锚继续有意义）+
   hero（先说业务目标 + 🧩 主按钮）不动。
"""

from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("streamlit")

ROOT = Path(__file__).resolve().parents[2]
PAGE_HOME = ROOT / "ui" / "app.py"

# (按钮文案, switch_page 目标) —— 旅程序
_JOURNEY = [
    ("🧙 数据向导", "pages/05_Data_Wizard.py"),
    ("🏋️ 发起训练", "pages/00_Training_Lab.py"),
    ("📊 查看实验", "pages/01_Experiments.py"),
    ("🎯 评测结果", "pages/02_Evaluation.py"),
    ("⚖️ 对比模型", "pages/03_Model_Comparison.py"),
    ("🏛️ 模型注册", "pages/04_Registry.py"),
    ("💬 对话验证", "pages/06_Chat.py"),
    ("🧩 目标与数据", "pages/07_Data_Intake.py"),
]


def _source() -> str:
    return PAGE_HOME.read_text(encoding="utf-8")


def _boot_home(monkeypatch):
    """首页 AppTest 启动（fetch_runs 打桩空 DF：旅程网格与运行记录无关，
    空 DF 让最近动态走空态分支，与本节断言互不干扰——同
    test_home_empty_state_ui.py 的跨机稳定打桩法）。"""
    from streamlit.testing.v1 import AppTest

    import ui.queries

    monkeypatch.setattr(ui.queries, "fetch_runs", lambda *a, **k: pd.DataFrame())
    page = AppTest.from_file(str(PAGE_HOME), default_timeout=60)
    page.run()
    return page


def test_journey_map_covers_all_eight_pages(monkeypatch):
    """8 页全覆盖钉：旅程按钮与 switch_page 目标 8/8 在场——首页一屏
    直达全部 8 页（README 宣称的完整生命周期在首页有完整入口）。"""
    source = _source()
    page = _boot_home(monkeypatch)
    assert not page.exception, [e.message for e in page.exception]
    labels = [b.label for b in page.button]
    for want_label, target in _JOURNEY:
        assert want_label in labels, f"旅程按钮缺失:{want_label}"
        # 按钮经 _JOURNEY_STEPS 表驱动 st.switch_page(target)，目标串落在表内
        # （字面 switch_page 调用不存在是循环结构使然，接线由钉③实证）
        assert f'"{target}"' in source, f"跳转目标缺失:{target}"


def test_journey_order_and_lead_caption_pinned():
    """旅程排序钉：8 步按「数据 → 训练 → 台账 → 评测 → 对比 → 注册 →
    对话」源码序递增 + 引导 caption（「按完整旅程排序」+ 首次使用指路
    07）在场——排序即产品语义，乱序重排（如按页号 00→07 排）当场可抓。"""
    source = _source()
    positions = [source.find(f'"{label}"') for label, _ in _JOURNEY]
    assert all(pos >= 0 for pos in positions), f"旅程按钮字面缺失:{positions}"
    assert positions == sorted(positions), f"旅程按钮必须按旅程序排列:{positions}"
    assert "按完整旅程排序" in source, "引导 caption（旅程序说明）必须在场"
    assert "首次使用建议从「🧩 目标与数据」开始" in source, "首次使用指路必须在场"


def test_journey_button_navigates_to_wizard(monkeypatch):
    """接线实证钉：点「🧙 数据向导」元素树切到 05（标题「🧙 数据向导 ·
    Data Wizard」，R134 统一命名）——新增四钮真接线，不是只指不发的幽灵按钮。"""
    page = _boot_home(monkeypatch)
    btn = next(b for b in page.button if b.label == "🧙 数据向导")
    btn.click().run()
    assert not page.exception, [e.message for e in page.exception]
    assert any(t.value == "🧙 数据向导 · Data Wizard" for t in page.title), "必须切到 05 数据向导"


def test_legacy_labels_and_hero_kept_alive():
    """保活钉：旧四钮字面原样保留（test_register_unification_ui.py 的
    中文化锚继续有效）+ hero（先说业务目标 + 🧩 分析我的目标与数据主
    按钮）不动——旅程网格是扩展不是替换。"""
    source = _source()
    for legacy in ("🏋️ 发起训练", "📊 查看实验", "🎯 评测结果", "⚖️ 对比模型"):
        assert legacy in source, f"旧快捷操作字面不得删:{legacy}"
    assert "先说业务目标，再看数据" in source, "hero 文案不得动"
    assert 'st.button("🧩 分析我的目标与数据", type="primary")' in source, "hero 主按钮不得动"


def test_wizard_name_unified_across_beginner_path():
    """命名统一钉（R134）：首页 3 处「数据向导」指针落地的 05 页标题必须自带
    同名（此前首页叫数据向导、页面自称 Data Wizard——空态用户被指到的第一站
    就对不上名字）；「乱写法表格」退场，首页表格词汇与 05 页「原始表格」统一，
    不再有目标页不认识的第二套词汇。"""
    wizard_source = (ROOT / "ui" / "pages" / "05_Data_Wizard.py").read_text(encoding="utf-8")
    assert "🧙 数据向导 · Data Wizard" in wizard_source, "05 标题必须自带首页所用的中文名"
    assert "乱写法" not in _source(), "首页不得再有 05 页不认识的第二套表格词汇"


def test_hero_acceptance_claim_covers_real_uploaders():
    """hero 覆盖面钉（R134）：首页宣称的样例格式不得窄于 07 上传器实际接受面
    （CSV/Excel/JSONL）——低报覆盖面让有多格式数据的用户误以为进不去。"""
    source = _source()
    hero_intro = source.split("让 Agent")[0]
    for fmt in ("CSV", "Excel", "JSONL"):
        assert fmt in hero_intro, f"hero 样例格式宣称缺 {fmt}（07 上传器实际接受）"
