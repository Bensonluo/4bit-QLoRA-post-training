"""06 Chat 空态死端指路(R108):空选择器 → 训练页指路 + 跳转。

选点依据(r108-scout 审计裁决首选):06 页是「训练完的模型不该只躺在
outputs/ 里」的收尾体验,但新克隆机(outputs/ 为空)上是死端三件套——
①:37-48 选择器只剩「⌨️ 自定义」一个选项 ②:54-58 随即抛两个专家级
裸输入框(HF 名/LoRA adapter 术语零解释) ③:112-114 主区说「在左侧
选择模型并点⚡加载模型」,指向一个没有任何模型的列表。全页零
switch_page 零指路。发现层无辜:test_inference_discovery.py:82-86 已钉
空态返回 [],死的是页面旅程层。

AppTest switch_page 双场景实证(r108-scout /tmp 实验,streamlit 1.57.0,
与 R107 select_slider 专用访问器同型教训——按实证形态写钉,不猜 API):
- 直接 AppTest.from_file(页文件)点页内 switch 按钮 → StreamlitAPIException
  「Could not find page: pages/00_Training_Lab.py」(switch_page 按
  entrypoint 目录解析相对路径,页文件目录下无嵌套 pages/)
- 入口锚定 AppTest.from_file(ui/app.py) → at.switch_page("pages/06_Chat.py")
  → 点页内按钮 → 树切到目标页(titles 含 00 页标题),零异常
⇒ switch 旅程钉必须入口锚定;官方 docstring 同调(app_test.py:133-137
「single page … use AppTest.switch_page()」)。全库首例页内 switch 按钮
旅程钉。

monkeypatch 时序:app.py 先以真实 PROJECT_ROOT 跑(只读扫描,不对其
内容断言 → 跨机稳定),跑完再 patch 成 tmp_path,switch 进 06 时页模块
才 import 并读 ui.config.PROJECT_ROOT → outputs/ 为空 → 空态分支。
app.py 被真实执行,torch/mlflow 导入进程内摊销——整套门禁实测秒级
(r108-reviewer 口径修正,勿按「5-10s/进程」报成本)。
"""

import json
from pathlib import Path

import pytest

pytest.importorskip("streamlit")

ROOT = Path(__file__).resolve().parents[2]
UI = ROOT / "ui"
PAGE_CHAT = UI / "pages/06_Chat.py"


def _source(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _nav_to_chat(tmp_path, monkeypatch):
    """入口锚定导航到 06 页空态:先跑 app.py(真实根,只读),跑完再
    monkeypatch PROJECT_ROOT=tmp_path,switch_page 进入 06(页模块此刻才
    import,读到已替换属性——05 页先例 test_loading_feedback_ui.py:139)。"""
    from streamlit.testing.v1 import AppTest

    import ui.config

    at = AppTest.from_file(str(UI / "app.py"), default_timeout=60)
    at.run()
    assert not at.exception, [e.message for e in at.exception]
    monkeypatch.setattr(ui.config, "PROJECT_ROOT", tmp_path)
    return at.switch_page("pages/06_Chat.py").run()


def test_chat_empty_state_offers_training_path(tmp_path, monkeypatch):
    """空态指路:outputs/ 无产物时主区不得再说「在左侧选择模型」(列表里
    没有任何模型,指路失效),改指 00 页训练;自定义底座逃生门如实披露
    (急着体验可不训练直接聊);侧栏自定义控件不封死(专家路径保活)。"""
    at = _nav_to_chat(tmp_path, monkeypatch)
    assert not at.exception, [e.message for e in at.exception]
    assert any("还没有可对话的训练产物" in i.value for i in at.info), "空态必须出现训练指路说明"
    assert not any("在左侧选择模型并点" in i.value for i in at.info), (
        "空态不得保留「在左侧选择模型」——指向没有模型的列表"
    )
    assert any(b.label == "🏋️ 去训练一个模型" for b in at.button), "空态必须有去训练的主按钮"
    assert any(t.label == "底座模型（HF 名或本地路径）" for t in at.text_input), (
        "侧栏自定义底座逃生门必须保活(空态不封死专家路径)"
    )


def test_chat_empty_state_switch_navigates_to_training_lab(tmp_path, monkeypatch):
    """指路按钮真跳转(全库首例页内 switch 旅程钉):入口锚定下点
    「去训练一个模型」,元素树切到 00 训练实验室——不是只渲染按钮不接线。"""
    at = _nav_to_chat(tmp_path, monkeypatch)
    btn = next(b for b in at.button if b.label == "🏋️ 去训练一个模型")
    btn.click().run()
    assert not at.exception, [e.message for e in at.exception]
    assert any(t.value == "🏋️ 训练实验室" for t in at.title), "必须切到 00 训练实验室"


def test_chat_nonempty_branch_guidance_preserved(tmp_path, monkeypatch):
    """非空分支保活(升级为旅程钉,r108-reviewer nit-1 采纳):outputs/ 有
    adapter 时「在左侧选择模型并点⚡加载模型」指路句照常渲染、新空态分支
    不渲染——锁分支行为而非源码字串,未来同义改文案不碎。switch_page
    目标仍按 UI 调用形态钉(与 app.py:27/05:348 同形)。"""
    adapter_dir = tmp_path / "outputs" / "demo-adapter"
    adapter_dir.mkdir(parents=True)
    (adapter_dir / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": "Qwen/Qwen2.5-0.5B-Instruct"}),
        encoding="utf-8",
    )
    at = _nav_to_chat(tmp_path, monkeypatch)
    assert not at.exception, [e.message for e in at.exception]
    assert any("在左侧选择模型并点" in i.value for i in at.info), (
        "非空分支指路句必须照常渲染(本轮只动空态分支)"
    )
    assert not any("还没有可对话的训练产物" in i.value for i in at.info), "非空分支不得渲染空态指路"
    source = _source(PAGE_CHAT)
    assert 'st.switch_page("pages/00_Training_Lab.py")' in source, "跳转目标必须钉住"
