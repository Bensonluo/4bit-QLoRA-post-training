"""01 实验页空态死端指路(R109 双切片之二):st.stop 封页无出路修复。

选点依据(r109-scout 次级扫描裁决采纳):01:38-40 空态 st.info「暂无实验
记录。请先在训练实验室(Training Lab)发起训练。」+ st.stop() 封整页——
比 app.py 更死(整页无任何出路,用户只能靠侧栏导航自救)。与 app:181/
06:112 同族,一轮清完「空态死端」族(app 首页 + 01 + 06),04_Registry
空态指路质量登记 R110。

TDD 坑与实证(R108/R109 双实证在案):01 页内按钮的 switch click 旅程钉
必须入口锚定——ui/pages/ 目录下无嵌套 pages/,AppTest.from_file(01 页
文件)直点 switch 按钮会抛 StreamlitAPIException「Could not find page」;
须 from_file(ui/app.py) 后 at.switch_page("pages/01_Experiments.py")
进入(scout 场景 C 同机制)。

非空分支保活双钉并用:源码钉直锁本轮结构 delta(info 文案+按钮先于
st.stop+switch 接线,成本最低);旅程特征化钉锁运行时保活(最小假 DF
{run_id, status}——r109-reviewer 探针实证 01 非空路径全程列守卫,完整
跑通 KPI/筛选/表格,原「会碎在下游列依赖」的论据已证伪并修正)。
"""

from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("streamlit")
pytest.importorskip("mlflow")

ROOT = Path(__file__).resolve().parents[2]
UI = ROOT / "ui"
PAGE_EXP = UI / "pages/01_Experiments.py"


def _source(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _nav_to_experiments_empty(monkeypatch):
    """入口锚定导航到 01 页空态:app.py 真实根先跑(不对其内容断言),
    跑完再打桩 ui.queries(fetch_runs→空 DF/fetch_experiments→[]),
    switch 进 01 时页模块运行时导入读到已替换属性。"""
    from streamlit.testing.v1 import AppTest

    import ui.queries

    at = AppTest.from_file(str(UI / "app.py"), default_timeout=60)
    at.run()
    assert not at.exception, [e.message for e in at.exception]
    monkeypatch.setattr(ui.queries, "fetch_runs", lambda *a, **k: pd.DataFrame())
    monkeypatch.setattr(ui.queries, "fetch_experiments", lambda *a, **k: [])
    return at.switch_page("pages/01_Experiments.py").run()


def test_experiments_empty_state_offers_training_exit(monkeypatch):
    """空态指路钉:无实验记录时 info 在场且紧跟可点的「去训练实验室」
    主按钮——st.stop 封页前给出最后一条出路。"""
    at = _nav_to_experiments_empty(monkeypatch)
    assert not at.exception, [e.message for e in at.exception]
    assert any("暂无实验记录" in i.value for i in at.info), "空态 info 必须在场"
    assert any(b.label == "🏋️ 去训练实验室" for b in at.button), "空态必须有接线的主按钮"


def test_experiments_empty_state_button_navigates_to_training_lab(monkeypatch):
    """switch 旅程钉(入口锚定):点「去训练实验室」元素树切到 00 页。"""
    at = _nav_to_experiments_empty(monkeypatch)
    btn = next(b for b in at.button if b.label == "🏋️ 去训练实验室")
    btn.click().run()
    assert not at.exception, [e.message for e in at.exception]
    assert any(t.value == "🏋️ 训练实验室" for t in at.title), "必须切到 00 训练实验室"


def test_experiments_nonempty_branch_structure_preserved():
    """非空分支结构保活(源码钉):空态 info 文案与 st.stop 顺序钉住——本轮
    只在 info 与 st.stop 之间插入按钮,不碰其他任何结构。直锁结构 delta
    成本最低;非空页面的运行时保活由兄弟旅程钉承担(见下)。"""
    source = _source(PAGE_EXP)
    assert 'st.info("暂无实验记录。请先在训练实验室（Training Lab）发起训练。")' in source
    block = source.split("暂无实验记录", 1)[1]
    stop_pos = block.find("st.stop()")
    btn_pos = block.find('st.button("🏋️ 去训练实验室"')
    switch_pos = block.find('st.switch_page("pages/00_Training_Lab.py")')
    assert stop_pos != -1, "空态分支必须保持 st.stop 封页语义"
    assert btn_pos != -1 and btn_pos < stop_pos, "出路按钮必须在 st.stop 之前"
    assert switch_pos != -1 and btn_pos < switch_pos, "按钮必须接线到 00 页"


def test_experiments_nonempty_page_renders(monkeypatch):
    """非空页面运行时保活(旅程特征化钉,r109-reviewer nit-1 采纳升级:
    原「最小假 DF 会碎在下游列依赖」的论据被探针证伪——01 非空路径全程
    列守卫,最小 DF 完整跑通 KPI/筛选/表格)。有实验记录时空态分支
    退场、整页照常渲染。"""
    from streamlit.testing.v1 import AppTest

    import ui.queries

    at = AppTest.from_file(str(UI / "app.py"), default_timeout=60)
    at.run()
    assert not at.exception, [e.message for e in at.exception]
    monkeypatch.setattr(
        ui.queries,
        "fetch_runs",
        lambda *a, **k: pd.DataFrame([{"run_id": "abcdef1234567890", "status": "FINISHED"}]),
    )
    monkeypatch.setattr(ui.queries, "fetch_experiments", lambda *a, **k: [])
    at.switch_page("pages/01_Experiments.py").run()
    assert not at.exception, [e.message for e in at.exception]
    assert not any("暂无实验记录" in i.value for i in at.info), "非空不得渲染空态分支"
