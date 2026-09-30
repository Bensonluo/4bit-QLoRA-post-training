"""双数据门互指路牌（R133）：00_Training_Lab / 05_Data_Wizard → 07_Data_Intake。

R130 轮审计证实：07 是首页推荐的旅程首步（hero 按钮 + 旅程格），但 00 与 05
两页对 07 **零提及**——习惯直回这两页的用户永远看不到引导式路径的存在。
本轮在两页加 caption 路牌 + 「🚪 去数据入口」接线按钮。落点约束：
00 的 dataset 输入在 st.form 内（表单禁普通按钮），路牌置于表单开始前；
05 置于顶部 caption 后、依赖导入守卫前（无依赖时也可见）。
"""

from pathlib import Path

import pytest

pytest.importorskip("streamlit")

UI = Path(__file__).resolve().parents[2] / "ui"
INTAKE_TITLE = "从业务目标和数据开始"  # 07 页 st.title 原文（07_Data_Intake.py:40）
DOOR_BUTTON = "🚪 去数据入口"


def _nav_and_run(page_path: str):
    """从 app.py 入口导航到目标页并运行：保证页面在多页应用真实上下文中渲染。"""
    from streamlit.testing.v1 import AppTest

    page = AppTest.from_file(str(UI / "app.py"), default_timeout=60)
    page.run()
    assert not page.exception, [e.message for e in page.exception]
    page.switch_page(page_path).run()
    assert not page.exception, [e.message for e in page.exception]
    return page


def test_training_lab_signposts_guided_intake():
    """00 路牌钉：配置表单前 caption 点名两扇数据门 + 按钮真实切到 07。"""
    page = _nav_and_run("pages/00_Training_Lab.py")
    captions = " ".join(c.value for c in page.caption)
    assert "数据还没准备好" in captions, "路牌必须回应「表单里数据集从哪来」的即时困惑"
    assert "数据入口" in captions and "Data Wizard" in captions
    button = next(b for b in page.button if b.label == DOOR_BUTTON)
    assert not button.disabled
    button.click().run()
    assert not page.exception, [e.message for e in page.exception]
    assert any(t.value == INTAKE_TITLE for t in page.title), "点击必须真实切到 07 数据入口"


def test_wizard_signposts_guided_intake():
    """05 路牌钉：顶部 caption 区分两扇门定位 + 按钮真实切到 07。"""
    page = _nav_and_run("pages/05_Data_Wizard.py")
    captions = " ".join(c.value for c in page.caption)
    assert "你找对了" in captions, "路牌必须先确认用户没走错门（有表直转就是本页）"
    assert "数据入口" in captions
    button = next(b for b in page.button if b.label == DOOR_BUTTON)
    assert not button.disabled
    button.click().run()
    assert not page.exception, [e.message for e in page.exception]
    assert any(t.value == INTAKE_TITLE for t in page.title), "点击必须真实切到 07 数据入口"
