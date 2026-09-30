"""app.py 首页「📍 旅程进度」（R125）：R123 答了「从哪开始」，本节答「我走到哪了」。

选点依据：旅程网格之后，返回用户（第二天打开首页）看到的仍是静态导航 +
运行数统计——没有「四步旅程各自产物是否就绪」的回答，得自己挨页摸索。
本节用四个纯文件系统探测（wizard 训练集 / adapter / 通用评测结果 / 可对话
模型）渲染进度，缺什么给一句「怎么补」指路。

钉的设计（R121 规则：动首页文案前已 grep 全测试目录——R123 的旅程钉、
test_register_unification 的旧四钮在场钉、test_home_empty_state 的空态钉
均不受影响，本节是纯新增段）：
① 空态钉：零产物 → 4 步全 ⬜ + 「产物就绪 0/4」——新用户不被假进度误导；
② 部分就绪钉：wizard 集合 + adapter 在场、评测缺席 → 3/4 且「跑出评测」
   仍是 ⬜（未完成步骤不因其他步骤完成而虚亮）；
③ 全就绪钉：四类产物齐 → 🎉 success 分支（全链路跑通的庆祝与续用指路）；
④ 步骤序钉：四步源码序 = 准备数据→完成训练→跑出评测→对话可用
   （与 R123 旅程网格同一叙述序，乱序重排当场可抓）。
"""

import json
from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("streamlit")

ROOT = Path(__file__).resolve().parents[2]
PAGE_HOME = ROOT / "ui" / "app.py"

_STEPS = ["准备数据", "完成训练", "跑出评测", "对话可用"]


def _boot_home(monkeypatch, tmp_path: Path):
    """首页 AppTest 启动：PROJECT_ROOT 换 tmp（产物探测读 tmp），fetch_runs
    打桩空 DF（运行统计与最近动态走空态，与本节断言互不干扰）。"""
    from streamlit.testing.v1 import AppTest

    import ui.config
    import ui.queries

    monkeypatch.setattr(ui.config, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(ui.queries, "fetch_runs", lambda *a, **k: pd.DataFrame())
    page = AppTest.from_file(str(PAGE_HOME), default_timeout=60)
    page.run()
    return page


def _md_values(page) -> list[str]:
    return [m.value for m in page.markdown]


def test_progress_empty_state_all_four_pending(tmp_path, monkeypatch):
    """空态钉：零产物 → 4× ⬜ + 「产物就绪 0/4」——诚实显示未开始，且每步
    带「怎么开始」的指路（不是只报缺）。"""
    page = _boot_home(monkeypatch, tmp_path)
    assert not page.exception, [e.message for e in page.exception]
    md = _md_values(page)
    for step in _STEPS:
        assert f"⬜ **{step}**" in md or any(f"⬜ **{step}**" in v for v in md), (
            f"空态步骤必须 ⬜:{step}"
        )
    assert any("产物就绪 **0/4**" in v for v in md), "空态计数必须在场"
    assert any("数据向导 5 分钟" in v for v in md), "空态必须给开始指路"
    # 首页顶栏系统状态（MLflow/Plotly ✅）本就是 success，极性只能按内容断
    assert not any("全链路已跑通" in s.value for s in page.success), (
        "零产物不得出现全链路跑通 success"
    )


def test_progress_partial_ready_keeps_pending_step_honest(tmp_path, monkeypatch):
    """部分就绪钉：wizard 集合 + adapter 在场、评测缺席 → 3/4，且「跑出
    评测」仍 ⬜——每步独立探测，完成的步骤不连带虚亮未完成步骤。"""
    wiz = tmp_path / "outputs" / "wizard" / "demo_suppliers"
    wiz.mkdir(parents=True)
    (wiz / "train.json").write_text("[]", encoding="utf-8")
    run_dir = tmp_path / "outputs" / "run-a"
    run_dir.mkdir(parents=True)
    (run_dir / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": "Qwen/Qwen2.5-0.5B-Instruct"}),
        encoding="utf-8",
    )
    page = _boot_home(monkeypatch, tmp_path)
    assert not page.exception, [e.message for e in page.exception]
    md = _md_values(page)
    assert any("产物就绪 **3/4**" in v for v in md), "wizard+adapter → 3/4"
    assert any("✅ **准备数据**" in v for v in md), "wizard 在场 → 准备数据 ✅"
    assert any("✅ **完成训练**" in v for v in md), "adapter 在场 → 完成训练 ✅"
    # adapter 可被 discovery 发现 → 对话可用 ✅（对话不需要先合并）
    assert any("✅ **对话可用**" in v for v in md), "adapter 可直接对话"
    assert any("⬜ **跑出评测**" in v for v in md), "评测缺席必须保持 ⬜"
    assert not any("全链路已跑通" in s.value for s in page.success), (
        "3/4 不得出现全链路跑通 success"
    )


def test_progress_all_ready_shows_celebration(tmp_path, monkeypatch):
    """全就绪钉：四类产物齐 → 🎉 success（全链路从表格到对话的闭环宣告），
    计数行退场（成功态不重复报 4/4 数字）。"""
    for setup in (
        lambda: (tmp_path / "outputs" / "wizard" / "w").mkdir(parents=True),
        lambda: (tmp_path / "outputs" / "run-b").mkdir(parents=True),
        lambda: (tmp_path / "domains" / "entity_matching" / "data" / "results").mkdir(parents=True),
    ):
        setup()
    (tmp_path / "outputs" / "wizard" / "w" / "train.json").write_text("[]", encoding="utf-8")
    (tmp_path / "outputs" / "run-b" / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": "Qwen/Qwen2.5-0.5B-Instruct"}),
        encoding="utf-8",
    )
    (
        tmp_path / "domains" / "entity_matching" / "data" / "results" / "eval_detail_x.json"
    ).write_text("{}", encoding="utf-8")
    page = _boot_home(monkeypatch, tmp_path)
    assert not page.exception, [e.message for e in page.exception]
    assert page.success, "四产物齐必须走 🎉 success 分支"
    assert any("全链路已跑通" in s.value for s in page.success), "庆祝文案必须在场"
    assert any("✅ **跑出评测**" in v for v in _md_values(page)), "评测在场 → ✅"


def test_progress_step_order_pinned():
    """步骤序钉：四步源码序与 R123 旅程网格同叙述序（数据→训练→评测→
    对话）——乱序重排（如按字母序）当场可抓。"""
    source = PAGE_HOME.read_text(encoding="utf-8")
    positions = [source.find(f'"{step}"') for step in _STEPS]
    assert all(pos > 0 for pos in positions), f"步骤字面缺失:{positions}"
    assert positions == sorted(positions), f"步骤必须按旅程序:{positions}"
    assert "📍 旅程进度" in source, "段落标题钉"
