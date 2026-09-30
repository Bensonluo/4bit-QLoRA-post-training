"""05 页样本预览（R126）：生成前用真实数据验证列映射。

选点依据：映射接错列（非专家最常见错误）旧世界要等第④步生成完、甚至训练
完才发现。第③步末「👀 试生成前 2 条样本」用前 3 行真实数据 + 当前映射/
参数即时试跑模板——生成前就能看到样本长什么样。负例池小的局限如实声明
（预览行数 3 < n_candidates 时候选数少于设置值），完整生成以第④步为准。

钉的设计（R121 规则：动 05 页前已 grep 全测试目录——test_loading_feedback_ui
的 spinner ≥20 下限与演示按钮钉不受影响：预览块新增 1 处 spinner 只增不减）：
① 接线实证：点通用演示 → 点试生成 → 真实 Alpaca 样本渲染（instruction
   含模板指令文本、input 含候选列表）——预览不是空壳；
② 缓存钉：预览渲染后 plain rerun 仍在（session 缓存）；改一列映射
   （role_code → 不使用）预览即失效——指纹机制防「旧预览冒充当前配置」；
③ 极性钉：未载入表格前试生成按钮不存在（st.stop 在①步）；按钮 disabled
   门控源码钉（disabled=not ready）；
④ 诚实性钉：预览行数局限 caption + 新数据载入时预览一并失效的源码钉。
"""

from pathlib import Path

import pytest

pytest.importorskip("streamlit")

ROOT = Path(__file__).resolve().parents[2]
PAGE_WIZARD = ROOT / "ui" / "pages" / "05_Data_Wizard.py"


def _source() -> str:
    return PAGE_WIZARD.read_text(encoding="utf-8")


def _boot(monkeypatch, tmp_path: Path):
    from streamlit.testing.v1 import AppTest

    import ui.config

    monkeypatch.setattr(ui.config, "PROJECT_ROOT", tmp_path)
    page = AppTest.from_file(str(PAGE_WIZARD), default_timeout=30)
    page.run()
    return page


def _load_demo(page):
    demo = next(b for b in page.button if "通用演示" in b.label)
    return demo.click().run()


def _md(page) -> list[str]:
    return [m.value for m in page.markdown]


def test_preview_button_absent_until_table_loaded(monkeypatch, tmp_path):
    """极性钉：未载入表格（①步 st.stop）→ 试生成按钮不存在。"""
    page = _boot(monkeypatch, tmp_path)
    assert not page.exception, [e.message for e in page.exception]
    assert not any("试生成" in b.label for b in page.button), "无表格时不得出现试生成"


def test_preview_renders_real_samples_and_survives_rerun(monkeypatch, tmp_path):
    """接线 + 缓存钉：演示数据 → 点试生成 → 真实样本渲染（模板指令文本 +
    候选列表在场）；plain rerun 后仍在——指纹缓存让预览跨交互存活，非专家
    改别处参数时预览不闪没。"""
    page = _boot(monkeypatch, tmp_path)
    _load_demo(page)
    assert not page.exception, [e.message for e in page.exception]
    pv_btn = next(b for b in page.button if "试生成" in b.label)
    assert not pv_btn.disabled, "演示数据映射合法 → 预览可用"
    pv_btn.click().run()
    assert not page.exception, [e.message for e in page.exception]
    md = _md(page)
    assert any("从候选列表中选出" in v for v in md), "instruction 模板指令必须在场"
    assert any("候选:" in v for v in md), "input 候选列表必须在场"
    # plain rerun（无新交互）→ 预览仍在（session 缓存 + 指纹未变）
    page.run()
    assert not page.exception, [e.message for e in page.exception]
    assert any("从候选列表中选出" in v for v in _md(page)), "rerun 后预览必须仍在"


def test_preview_invalidated_when_mapping_changes(monkeypatch, tmp_path):
    """指纹钉：预览渲染后改一列映射（编码 → 不使用）→ 旧预览退场——旧
    预览不得冒充当前配置（映射变了样本就变，缓存必须跟着失效）。"""
    page = _boot(monkeypatch, tmp_path)
    _load_demo(page)
    pv_btn = next(b for b in page.button if "试生成" in b.label)
    pv_btn.click().run()
    assert any("从候选列表中选出" in v for v in _md(page)), "前置：预览已渲染"
    page.selectbox(key="role_code").set_value("（不使用）").run()
    assert not page.exception, [e.message for e in page.exception]
    assert not any("从候选列表中选出" in v for v in _md(page)), "映射已变 → 旧预览必须退场"


def test_preview_honesty_pins_in_source():
    """诚实性钉：①预览行数局限 caption（负例池小说明——预览不是完整生成
    的等价物）；②按钮 disabled 门控（映射/比例不合法时不可点，与生成按钮
    同门）；③新数据载入让旧预览失效（两处载入路径都 pop）。"""
    source = _source()
    assert "负例池小，候选数可能少于设置值" in source, "预览局限 caption 必须在场"
    assert '"👀 试生成前 2 条样本",\n    disabled=not ready,' in source, (
        "试生成按钮必须与生成按钮同门（ready 门控）"
    )
    assert source.count('st.session_state.pop("wizard_preview", None)') >= 3, (
        "上传/路径/试生成失败三处都必须失效旧预览"
    )
    assert "样本 {_i}（试生成）" in source, "试生成样本标签钉"
