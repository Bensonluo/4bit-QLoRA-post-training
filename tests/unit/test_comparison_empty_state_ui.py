"""03 对比页空态死端指路(R111):0/1 双语义分化 + 00 页出口接线。

选点依据(r111-scout 审计裁决采纳,R110 轮报登记候选经重大改写):
03「双空态」实为「一活一幽灵」——state#1(03:34-36「暂无包含评测数据
的领域」)是幽灵:list_domains() 返回 adapter registry keys,而
register_adapter(MedicalEntityAdapter()) 在 domain_adapters.py:222
模块导入时无条件执行 → 生产上永远返回 ["medical_entity"],state#1 仅
双重打桩下可达(scout 探针 A),登记不动。唯一活空态是 state#2
(03:41-43 len(data)<2),且一个分支吞两种语义:len==0(全新克隆零评测)
与 len==1(单模型)共用同一句「请先完成更多评测」——对 len==0 用户失实
(他们需要的是第一个,不是更多)。

出口裁决 00 页不去 02(scout 全读核实):02 页是「看结果+导入已有文件」
页,无任何发起评测的 UI 能力;len==0 用户指去 02 会落 02:106 自己的
空态(建议跑终端脚本+导入返回「未发现可导入的评测结果文件」)= 空→空
死端接力。00 是家族统一出口(R108 06/R109 01/R110 04 三先例)+训练
表单永在场。

诚实红线(scout 核实,文案必须守):00 的「下一步」评测命令
(scripts/evaluate.py)不写 domains/<domain>/data/results/;03 所读
eval_detail 文件的唯一生产者是 domains/medical_entity/eval/report.py
(终端)。指路句不得宣称「跑完命令结果会出现在本页」,只能停在「先训练
出(第二个)模型」这一 00 真实能力上。跨页断点属 IA 级,登记不修。

打桩实证(scout 五场景):monkeypatch.setattr(ui.components.
domain_adapters, "load_eval_data", ...)——03 顶层 from-import 在
switch 时执行,读到源模块已替换属性(R109 ui.queries 同机制)。本机
数据源非空(17 份 eval_detail)→ 空态钉必须打桩。最小 2-dict fixture
完整渲染(fmt_* 全 None-safe)——非空特征化钉已预验证;断言勿依赖
执行摘要分支(读真实本地目录,他机可能走 03:144/146 info 分支)。
"""

from pathlib import Path

import pytest

pytest.importorskip("streamlit")

ROOT = Path(__file__).resolve().parents[2]
UI = ROOT / "ui"
PAGE_CMP = UI / "pages/03_Model_Comparison.py"


def _source(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _nav_to_comparison(monkeypatch, records):
    """入口锚定导航到 03 页:app.py 真实根先跑(不对其内容断言),跑完再
    打桩 ui.components.domain_adapters.load_eval_data→指定记录列表,
    switch 进 03 时页模块顶层 from-import 读到已替换属性。"""
    from streamlit.testing.v1 import AppTest

    import ui.components.domain_adapters as da

    at = AppTest.from_file(str(UI / "app.py"), default_timeout=60)
    at.run()
    assert not at.exception, [e.message for e in at.exception]
    monkeypatch.setattr(da, "load_eval_data", lambda *a, **k: list(records))
    return at.switch_page("pages/03_Model_Comparison.py").run()


def test_comparison_len0_empty_state_offers_training_exit(monkeypatch):
    """len==0 空态指路钉:零评测结果时 warning 如实说「还没有任何模型的
    评测结果」(不再失实的「请先完成更多评测」),紧跟指路 info 与可点的
    「去训练实验室」主按钮。"""
    at = _nav_to_comparison(monkeypatch, [])
    assert not at.exception, [e.message for e in at.exception]
    assert any("还没有任何模型的评测结果" in w.value for w in at.warning), "len0 必须如实说零结果"
    assert any("训练实验室" in i.value for i in at.info), "指路 info 必须在场"
    assert any(b.label == "🏋️ 去训练实验室" for b in at.button), "空态必须有接线的主按钮"


def test_comparison_empty_state_button_navigates_to_training_lab(monkeypatch):
    """switch 旅程钉(入口锚定):点「去训练实验室」元素树切到 00 页。"""
    at = _nav_to_comparison(monkeypatch, [])
    btn = next(b for b in at.button if b.label == "🏋️ 去训练实验室")
    btn.click().run()
    assert not at.exception, [e.message for e in at.exception]
    assert any(t.value == "🏋️ 训练实验室" for t in at.title), "必须切到 00 训练实验室"


def test_comparison_len1_copy_names_the_single_model(monkeypatch):
    """len==1 语义分化钉:单模型时 warning 点名唯一模型名,与 len==0 文案
    分化——两种状态不再共用一句失实指路。"""
    at = _nav_to_comparison(monkeypatch, [{"model": "甲模型"}])
    assert not at.exception, [e.message for e in at.exception]
    assert any("甲模型" in w.value for w in at.warning), "len1 必须点名唯一模型"
    assert not any("还没有任何模型的评测结果" in w.value for w in at.warning), (
        "len1 不得渲染 len0 文案"
    )
    assert any(b.label == "🏋️ 去训练实验室" for b in at.button), "len1 同样需要出路按钮"


def test_comparison_empty_branch_exit_before_stop():
    """空态分支出路结构钉(源码钉):len(data)<2 分支内按钮与接线先于
    st.stop——出路必须在封页前。命名循 R110 nit-1 先例(钉空态分支
    顺序,不叫 nonempty)。R112 兄弟钉批量同步补 switch<stop 断言
    (生而绿,披露):旧三断言下把 switch 挪到 st.stop 之后仍全绿——
    接线死了只有旅程钉能抓,补上后结构钉也抓。"""
    source = _source(PAGE_CMP)
    assert "len(data) < 2" in source
    block = source.split("len(data) < 2", 1)[1]
    stop_pos = block.find("st.stop()")
    btn_pos = block.find('st.button("🏋️ 去训练实验室"')
    switch_pos = block.find('st.switch_page("pages/00_Training_Lab.py")')
    assert stop_pos != -1, "空态分支必须保持 st.stop 封页语义"
    assert btn_pos != -1 and btn_pos < stop_pos, "出路按钮必须在 st.stop 之前"
    assert switch_pos != -1 and btn_pos < switch_pos, "按钮必须接线到 00 页"
    assert switch_pos < stop_pos, "switch 接线必须在 st.stop 之前(R112)"


def test_comparison_nonempty_page_renders(monkeypatch):
    """非空页面运行时保活(旅程特征化钉,生而绿):最小 2-dict fixture
    (scout 探针 E 预验证 fmt_* 全 None-safe)完整渲染,空态分支退场。
    断言不依赖执行摘要分支(读真实本地目录,跨机不稳)。"""
    at = _nav_to_comparison(monkeypatch, [{"model": "甲"}, {"model": "乙"}])
    assert not at.exception, [e.message for e in at.exception]
    assert not any("对比至少需要" in w.value for w in at.warning), "非空不得渲染空态分支"
    assert any(s.value == "指标对比" for s in at.subheader), "指标对比区必须真渲染"
