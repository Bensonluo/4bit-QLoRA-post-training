"""app.py 首页空态指路接线(R109 双切片之一):指路句从纯文字升级为按钮。

选点依据(r109-scout 审计裁决首选):app:181 空态 st.info「暂无训练记录。
去训练实验室发起第一个实验吧。」纯文字无按钮——与 R108 修的 05:124
幽灵按钮句同型(指路牌自己不接线)。程度轻于 06 页死端(首页视线内有
hero+四个快捷按钮),但指路句自身就该可点。

switch 机制实证(r109-scout /tmp 四场景):app.py 页内按钮在
AppTest.from_file(ui/app.py) 直跑下点按即切页(场景 A/D:标题切到
「🏋️ 训练实验室」零异常)——app.py 本身就是入口,不需要 R108 的
入口锚定两段式(那是页文件 from_file 的坑:ui/pages/ 下无嵌套 pages/)。

空态钉必须 monkeypatch ui.queries.fetch_runs→空 DF:本机 outputs/mlruns
非空,不 patch 则空态分支不渲染(跨机不稳)。patch 时序简单:app.py 在
run() 期间才 from ui.queries import fetch_runs(:86/:154 两处,均 try 块内
运行时导入),run 前 setattr 即生效。
"""

from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("streamlit")

ROOT = Path(__file__).resolve().parents[2]
PAGE_HOME = ROOT / "ui" / "app.py"


def _boot_home(monkeypatch, runs):
    """首页 AppTest 启动,fetch_runs 打桩成指定 DataFrame。"""
    from streamlit.testing.v1 import AppTest

    import ui.queries

    monkeypatch.setattr(ui.queries, "fetch_runs", lambda *a, **k: runs)
    page = AppTest.from_file(str(PAGE_HOME), default_timeout=60)
    page.run()
    return page


def test_home_empty_state_offers_training_exit(monkeypatch):
    """空态指路钉:无训练记录时 info 在场且紧跟可点的「发起第一个实验」
    主按钮——指路句不再只指不发。"""
    page = _boot_home(monkeypatch, pd.DataFrame())
    assert not page.exception, [e.message for e in page.exception]
    assert any("暂无训练记录" in i.value for i in page.info), "空态 info 必须在场"
    assert any(b.label == "🏋️ 发起第一个实验" for b in page.button), "空态必须有接线的主按钮"


def test_home_empty_state_button_navigates_to_training_lab(monkeypatch):
    """switch 旅程钉:点「发起第一个实验」,元素树切到 00 训练实验室
    (app.py 入口直跑机制,scout 场景 D 实证)。"""
    page = _boot_home(monkeypatch, pd.DataFrame())
    btn = next(b for b in page.button if b.label == "🏋️ 发起第一个实验")
    btn.click().run()
    assert not page.exception, [e.message for e in page.exception]
    assert any(t.value == "🏋️ 训练实验室" for t in page.title), "必须切到 00 训练实验室"


def test_home_nonempty_recent_activity_renders(monkeypatch):
    """非空分支保活(特征化钉,生而绿):有运行记录时最近动态列表照常渲染、
    空态分支退场。最小列集 {run_id, status}——app:159-168 全 row.get 带
    默认值,scout 已逐行核实。"""
    page = _boot_home(
        monkeypatch, pd.DataFrame([{"run_id": "abcdef1234567890", "status": "FINISHED"}])
    )
    assert not page.exception, [e.message for e in page.exception]
    assert not any("暂无训练记录" in i.value for i in page.info), "非空不得渲染空态分支"
    # 正向断言堵 try/except 盲区(r109-reviewer nit-2 采纳):渲染循环若碎,
    # 异常被 app 自己吞掉转「加载最近动态失败」info——不含「暂无训练记录」,
    # 仅负向断言测试仍绿
    assert any("abcdef12" in m.value for m in page.markdown), "最近动态列表必须真渲染"
