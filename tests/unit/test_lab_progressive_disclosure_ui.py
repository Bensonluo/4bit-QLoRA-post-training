"""00 配置标签渐进披露(R107):基础/高级分层 + 学习率分级化。

选点依据(r107-scout 审计裁决):00 页是画像(非专家使用者)的主任务页,
当前 ~22 控件同权重铺开——LoRA 秩/Alpha/Dropout/自由文本学习率/梯度累积/
注册表三件套与底座模型/数据集/运行名同级。自由文本学习率是真实失败模式:
解析守卫只在输入后 warn(旧 384-392),提交才拦(旧 505),且要求非专家懂
「2e-4」记法。仓库已有全部惯用法:00:219-222 GRPO 隐藏注册表区块的表单外
兜底变量先例;00:206-215 技术单选置表单外的即时重渲染先例(高级参数 toggle
同理:表单内 toggle 须提交才生效,无法即时展开);07:2972「高级」expander
跨页范式。

AppTest 实证(/tmp 快照,1.57.0):page.toggle[i].set_value(bool).run() 可用;
select_slider 不进 page.slider 桶,专用访问器 page.select_slider——旅程钉
按实证形态写,不猜 API。

钉型:①AppTest 旅程钉锁披露行为本身(基础视图专家控件缺席/展开后到场)
②契约完整性钉——高级隐藏时配置预览 YAML 仍含 lora/training/logging 全键
与预设默认值(兜底变量机制,本轮风险的 bug 族:表单读未定义变量 NameError
或契约键静默消失)③学习率分级钉(select_slider+挡位;自由文本与解析错误
路径全退场)④toggle 表单外钉(即时重渲染语义,镜像技术单选先例)+兜底块
存在性。

标签-逻辑三耦合(R106 实证,本轮不动):技术单选标签 .lower() 进 SCRIPTS
查表;平台控件 "CUDA" 成员测试;量化单选标签即逻辑值(4-bit QLoRA 字串
不动)。DPO/GRPO 技术区块保持仅按技术门控渲染(选 DPO 本身即专家动作,
不叠加 advanced 门,免新增兜底)。
"""

import re
from pathlib import Path

import pytest
import yaml

pytest.importorskip("streamlit")

ROOT = Path(__file__).resolve().parents[2]
PAGE_LAB = ROOT / "ui/pages/00_Training_Lab.py"


def _source(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _boot_lab_page(tmp_path, monkeypatch):
    """真页 AppTest 启动(R104 旅程钉先例):PROJECT_ROOT 经 monkeypatch
    替换——AppTest 内 from-import 读到已替换的 ui.config 属性,Activity
    探测走空 outputs/ 的裸调分支,配置标签页的断言不受真实 outputs/ 干扰。"""
    from streamlit.testing.v1 import AppTest

    import ui.config

    monkeypatch.setattr(ui.config, "PROJECT_ROOT", tmp_path)
    page = AppTest.from_file(str(PAGE_LAB), default_timeout=30)
    page.run()
    return page


def _preview_yaml(page):
    """从 page.code 里找出可解析为 dict 且含 model 键的 YAML 配置预览。"""
    for block in page.code:
        try:
            parsed = yaml.safe_load(block.value)
        except yaml.YAMLError:
            continue
        if isinstance(parsed, dict) and "model" in parsed:
            return parsed
    return None


def _advanced_toggle(page):
    """按标签找高级参数 toggle(r107-reviewer nit-2:位置索引在页面新增
    toggle 时会静默指错对象,具名查找错则响亮失败)。"""
    return next(t for t in page.toggle if t.label == "高级参数")


def test_configure_defaults_to_basic_view(tmp_path, monkeypatch):
    """基础视图(默认):必填控件在场(底座模型/数据集/最大样本数/轮数/
    运行名/开始训练),专家控件缺席(零滑杆零挡位滑杆——LoRA trio/验证集
    比例/Dropout/学习率全在高级)。披露 caption 在场(should-fix 采纳:
    隐藏参数有预设值这一事实必须如实告知,轮报声明的设计决策需要钉)。"""
    page = _boot_lab_page(tmp_path, monkeypatch)
    assert not page.exception, [e.message for e in page.exception]

    toggles = [(t.label, t.value) for t in page.toggle]
    assert ("高级参数", False) in toggles, "高级参数 toggle 必须存在且默认关"

    labels_tb = [t.label for t in page.text_input]
    assert "数据集（HF 名称或本地路径）" in labels_tb, "基础:数据集在场"
    assert "运行名" in labels_tb, "基础:运行名在场"
    sel_labels = [s.label for s in page.selectbox]
    assert "底座模型" in sel_labels, "基础:底座模型在场"
    assert [b.label for b in page.button if b.label == "🚀 开始训练"], "基础:提交按钮在场"

    assert len(page.slider) == 0, (
        f"基础视图不得渲染滑杆(LoRA/Dropout/验证集比例属高级):{[s.label for s in page.slider]}"
    )
    assert len(page.select_slider) == 0, (
        f"基础视图不得渲染挡位滑杆(学习率属高级):{[s.label for s in page.select_slider]}"
    )
    assert any("高级参数未展开" in c.value for c in page.caption), (
        "基础视图必须如实披露:隐藏参数正以预设默认值进入契约"
    )


def test_advanced_toggle_reveals_expert_controls(tmp_path, monkeypatch):
    """展开高级:LoRA trio(秩/Dropout)、验证集比例滑杆与学习率挡位滑杆
    到场——披露行为本身可交互(toggle 表单外,即时重渲染)。披露 caption
    退场(参数已可见,提示不再有意义)。"""
    page = _boot_lab_page(tmp_path, monkeypatch)
    _advanced_toggle(page).set_value(True).run()
    assert not page.exception, [e.message for e in page.exception]

    slider_labels = [s.label for s in page.slider]
    assert "LoRA 秩（r）" in slider_labels, "高级:LoRA 秩滑杆必须到场"
    assert "Dropout 比例" in slider_labels, "高级:Dropout 滑杆必须到场"
    assert "验证集比例" in slider_labels, "高级:验证集比例滑杆必须到场"
    assert [s.label for s in page.select_slider if s.label == "学习率"], (
        "高级:学习率挡位滑杆必须到场"
    )
    assert not any("高级参数未展开" in c.value for c in page.caption), (
        "展开后披露 caption 必须退场(提示不再对应现实)"
    )


def test_dpo_sections_stay_technique_gated_in_basic_view(tmp_path, monkeypatch):
    """DPO/GRPO 区块仅按技术门控,不叠加 advanced 门(r107-reviewer
    should-fix 采纳:轮报声明的设计决策此前无钉——下轮改门者会让全部既有
    测试保持绿而行为静默漂移)。选 DPO 本身即专家动作,Beta 滑杆在基础
    视图必须到场,否则 dpo_beta 未定义 → 提交即 NameError。"""
    page = _boot_lab_page(tmp_path, monkeypatch)
    technique = next(r for r in page.radio if r.label == "训练后技术")
    technique.set_value("DPO").run()
    assert not page.exception, [e.message for e in page.exception]
    slider_labels = [s.label for s in page.slider]
    assert "Beta（β）" in slider_labels, "DPO 区块必须随技术选择到场(不受 advanced 门)"
    assert "参考模型" in [s.label for s in page.selectbox], "DPO 参考模型选择器必须到场"


def test_hidden_advanced_keeps_config_contract_complete(tmp_path, monkeypatch):
    """契约完整性(本轮风险的 bug 族):高级隐藏时配置预览 YAML 必须仍含
    model/lora/training/data/logging 全键与预设默认值——兜底变量机制
    (00:219-222 GRPO 先例)保证隐藏的专家参数以预设值流入契约,而不是
    NameError 或键静默消失。标准预设口径:r=16, lr=2e-4, grad_accum=8。"""
    page = _boot_lab_page(tmp_path, monkeypatch)
    parsed = _preview_yaml(page)
    assert parsed is not None, "配置预览 YAML 必须渲染且可解析"
    assert {"model", "lora", "training", "data", "logging"} <= set(parsed), (
        f"高级隐藏时契约键不得缺失:{sorted(parsed)}"
    )
    assert parsed["lora"]["r"] == 16, "隐藏时 LoRA 秩必须取预设兜底值 16"
    assert parsed["training"]["learning_rate"] == pytest.approx(2e-4), (
        "隐藏时学习率必须取预设兜底值 2e-4"
    )
    assert parsed["training"]["gradient_accumulation_steps"] == 8, (
        "隐藏时梯度累积必须取预设兜底值 8"
    )
    assert "quantization_bits" in parsed["model"], "量化键必须在场(平台相关值)"
    # 展开后契约不缩水(同键集合)
    page.toggle[0].set_value(True).run()
    parsed_adv = _preview_yaml(page)
    assert parsed_adv is not None and set(parsed_adv) >= set(parsed), "展开高级后契约键集合不得缩水"


def test_learning_rate_is_tiered_not_free_text():
    """学习率从自由文本改挡位滑杆:非专家不必懂「2e-4」记法,解析错误
    路径(lr_error/warn)整体退场——错误不可能发生在不存在自由文本的地方。
    挡位值域覆盖两个预设值(1e-4/2e-4)。首参正则兼容 ruff 单行/逐参换行
    两种调用形态(参数多必然逐参换行,单行字面量钉会误伤合法格式)。"""
    source = _source(PAGE_LAB)
    assert re.search(r'st\.select_slider\(\s*"学习率"', source), "学习率必须是挡位滑杆(首参即标签)"
    assert 'st.text_input("学习率"' not in source, "自由文本学习率必须退场"
    assert "lr_error" not in source, "学习率解析错误路径必须整体退场"
    tiers = re.search(r"lr_tiers\s*=\s*\[([^\]]+)\]", source)
    assert tiers, "学习率挡位表必须存在"
    values = [float(v) for v in tiers.group(1).split(",")]
    assert 2e-4 in values and 1e-4 in values, "挡位必须覆盖预设值 1e-4/2e-4"


def test_toggle_outside_form_with_preset_fallbacks():
    """高级参数 toggle 必须在表单外(表单内 toggle 须提交才生效,无法即时
    展开——技术单选置表单外的同一先由,00:206-215 注释);表单前必须有
    预设兜底赋值块(隐藏时 config_dict 读的就是这些变量)。"""
    source = _source(PAGE_LAB)
    toggle = re.search(r'st\.toggle\(\s*"高级参数"', source)
    assert toggle, "高级参数 toggle 必须存在(首参即标签)"
    form_idx = source.index('with st.form("training_config")')
    assert toggle.start() < form_idx, "toggle 必须在表单外(即时重渲染语义)"
    # 兜底赋值在 toggle 与表单之间(隐藏时契约完整性的一手保证)
    between = source[toggle.start() : form_idx]
    for fallback in (
        "validation_split = 0.1",
        "max_length = 512",
        "batch_size = 1",
        "lora_dropout = 0.05",
    ):
        assert fallback in between, f"预设兜底必须先于表单:{fallback}"
