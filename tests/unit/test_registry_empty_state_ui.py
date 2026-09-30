"""04 注册表页空态死端指路(R110):专家语域双出路 → 00 页按钮接线。

选点依据(r110-scout 审计裁决采纳,R109 轮报登记候选):04:37-42 空态
st.info 两条出路全是专家语域——①「在训练配置 LoggingConfig 中设置
register_model=True」(要求改 Python 配置类)②「运行 python scripts/
registry_cli.py register」(要求跑终端命令)——非专家画像两条都走不通,
且 st.stop() 封页、零按钮零页面内出路。本机 outputs/mlruns/models/
真实为空(scout 实证 A:info 在场+buttons=[]),空态就是当前真实状态
——活死端坐实。

按钮去向裁决 00 页(不是 01,scout 论据):00 页同时覆盖两条真实旅程
——(a) Configure 页展开「高级参数」后的「模型注册表」区有「训练后
自动注册」checkbox(00:321/325 实控件名,R107 起 advanced 门内——
info 指路须写明展开路径,否则又成 R108 05:124 幽灵按钮);(b) 已跑完
训练的用户走 Activity「🧭 下一步」的 📦 合并导出+注册 CLI(00:66-97)。
01 页只有 run 历史无任何注册动作,指过去才是真死端。

打桩实证(scout B):04:19 from ui.queries import fetch_model_versions
是页运行时导入,switch 前 setattr ui.queries.fetch_model_versions→
空 DF 即被读取(R109 时序范式)。本机 registry 恰好为空,不打桩测试
也能过,但钉必须打桩——否则未来本机一注册模型钉就静默漂移。
"""

from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("streamlit")
pytest.importorskip("mlflow")

ROOT = Path(__file__).resolve().parents[2]
UI = ROOT / "ui"
PAGE_REG = UI / "pages/04_Registry.py"


def _source(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _nav_to_registry_empty(monkeypatch):
    """入口锚定导航到 04 页空态:app.py 真实根先跑(不对其内容断言),
    跑完再打桩 ui.queries.fetch_model_versions→空 DF,switch 进 04 时
    页模块运行时导入读到已替换属性。"""
    from streamlit.testing.v1 import AppTest

    import ui.queries

    at = AppTest.from_file(str(UI / "app.py"), default_timeout=60)
    at.run()
    assert not at.exception, [e.message for e in at.exception]
    monkeypatch.setattr(ui.queries, "fetch_model_versions", lambda *a, **k: pd.DataFrame())
    return at.switch_page("pages/04_Registry.py").run()


def test_registry_empty_state_offers_training_exit(monkeypatch):
    """空态指路钉:info 在场且点名 00 页真实控件「训练后自动注册」
    (不再只丢 LoggingConfig/CLI 专家语域),紧跟可点的「去训练实验室」
    主按钮——空态不再是指完就死的路牌。"""
    at = _nav_to_registry_empty(monkeypatch)
    assert not at.exception, [e.message for e in at.exception]
    assert any("暂无已注册的模型" in i.value for i in at.info), "空态 info 必须在场"
    assert any("训练后自动注册" in i.value for i in at.info), (
        "info 必须点名真实控件(R108 幽灵按钮教训)"
    )
    assert any(b.label == "🏋️ 去训练实验室" for b in at.button), "空态必须有接线的主按钮"


def test_registry_empty_state_button_navigates_to_training_lab(monkeypatch):
    """switch 旅程钉(入口锚定):点「去训练实验室」元素树切到 00 页。"""
    at = _nav_to_registry_empty(monkeypatch)
    btn = next(b for b in at.button if b.label == "🏋️ 去训练实验室")
    btn.click().run()
    assert not at.exception, [e.message for e in at.exception]
    assert any(t.value == "🏋️ 训练实验室" for t in at.title), "必须切到 00 训练实验室"


def test_registry_empty_branch_exit_before_stop():
    """空态分支出路结构钉(源码钉,r110-reviewer nit-1 采纳改名:原名
    nonempty_branch 名实不符,钉的是空态分支内部顺序):info 锚点与
    st.stop 顺序钉住——本轮只在 info 与 st.stop 之间插入按钮,info 文案
    重写,不碰其他任何结构。R112 兄弟钉批量同步补 switch<stop 断言
    (生而绿,披露):旧三断言下把 switch 挪到 st.stop 之后仍全绿——
    按钮在 stop 前但接线死了,只有旅程钉能抓;补上后结构钉也抓得住。"""
    source = _source(PAGE_REG)
    assert "暂无已注册的模型" in source
    block = source.split("暂无已注册的模型", 1)[1]
    stop_pos = block.find("st.stop()")
    btn_pos = block.find('st.button("🏋️ 去训练实验室"')
    switch_pos = block.find('st.switch_page("pages/00_Training_Lab.py")')
    assert stop_pos != -1, "空态分支必须保持 st.stop 封页语义"
    assert btn_pos != -1 and btn_pos < stop_pos, "出路按钮必须在 st.stop 之前"
    assert switch_pos != -1 and btn_pos < switch_pos, "按钮必须接线到 00 页"
    assert switch_pos < stop_pos, "switch 接线必须在 st.stop 之前(R112)"


def test_registry_nonempty_page_renders(monkeypatch):
    """非空页面运行时保活(旅程特征化钉,生而绿):r110-scout C 场景实证
    最小 7 列全字段单行 DF 完整跑通 04 非空路径(选择器/KPI/表格/别名
    动作区),空态分支退场。R109 nit-1 教训:论据用实证不用臆测。"""
    from streamlit.testing.v1 import AppTest

    import ui.queries

    at = AppTest.from_file(str(UI / "app.py"), default_timeout=60)
    at.run()
    assert not at.exception, [e.message for e in at.exception]
    monkeypatch.setattr(
        ui.queries,
        "fetch_model_versions",
        lambda *a, **k: pd.DataFrame(
            [
                {
                    "name": "Qwen3-1.7B-QLoRA",
                    "version": 3,
                    "aliases": "champion",
                    "current_stage": "None",
                    "status": "READY",
                    "created": 1760000000000,
                    "run_id": "abcdef1234567890",
                }
            ]
        ),
    )
    at.switch_page("pages/04_Registry.py").run()
    assert not at.exception, [e.message for e in at.exception]
    assert not any("暂无已注册的模型" in i.value for i in at.info), "非空不得渲染空态分支"
    assert any(m.value == "1" for m in at.metric if m.label == "版本数"), "版本数 metric 必须真渲染"


def test_registry_corrupt_store_shows_error_not_crash(monkeypatch):
    """损坏守卫钉(R129):注册表读取异常时如实 st.error + 出口按钮,不裸栈、
    也不落回失实的「暂无已注册的模型」空态——读不到 ≠ 没有(诚实红线)。"""
    from streamlit.testing.v1 import AppTest

    import ui.queries

    at = AppTest.from_file(str(UI / "app.py"), default_timeout=60)
    at.run()
    assert not at.exception, [e.message for e in at.exception]

    def _boom(*a, **k):
        raise RuntimeError("模拟存储损坏")

    monkeypatch.setattr(ui.queries, "fetch_model_versions", _boom)
    at.switch_page("pages/04_Registry.py").run()
    assert not at.exception, [e.message for e in at.exception]
    assert any("读取模型注册表失败" in e.value for e in at.error), "必须如实报错而非裸栈"
    assert any(b.label == "🏋️ 去训练实验室" for b in at.button), "报错态也必须有出路按钮"
    assert not any("暂无已注册的模型" in i.value for i in at.info), (
        "读不到 ≠ 没有,不得渲染失实空态文案"
    )
