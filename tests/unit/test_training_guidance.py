"""training_guidance 单一来源的口径测试：参数大白话、分区引导与测试集条数提醒。"""

from src.workbench.training_guidance import (
    LR_TIER_DISCLAIMER,
    learning_rate_suggestion,
    manual_training_parameter_lines,
    small_test_set_line,
    split_settings_guidance_lines,
)


def test_manual_training_parameter_lines_cover_every_form_knob() -> None:
    lines = manual_training_parameter_lines()
    assert lines[0].startswith("参数大白话：训练轮数＝")
    for phrase in (
        "每设备 batch size＝",
        "梯度累积步数＝",
        "学习率＝",
        "LoRA rank＝",
        "4-bit 量化＝",
        "最大 token 长度＝",
    ):
        assert any(phrase in line for line in lines)
    assert lines[-1].startswith("推荐起步值（小数据）：")
    assert "与默认值一致；先跑通再调。" in lines[-1]


def test_learning_rate_suggestion_tiers_and_disclaimer() -> None:
    low_lr, low_reason = learning_rate_suggestion(500)
    assert low_lr == 0.0001
    assert "全量 500 条（< 2,000）" in low_reason
    assert "建议学习率起步 5e-5~1e-4" in low_reason
    high_lr, high_reason = learning_rate_suggestion(2000)
    assert high_lr == 0.0002
    assert "全量 2000 条（≥ 2,000）" in high_reason
    assert "非本产品实测" in LR_TIER_DISCLAIMER


def test_split_settings_guidance_explains_when_to_change() -> None:
    lines = split_settings_guidance_lines()
    assert any("验证集比例＝" in line for line in lines)
    assert any("比例可提到 0.15–0.2" in line for line in lines)
    assert any("可复现分区种子＝" in line for line in lines)


def test_small_test_set_line_uses_honest_arithmetic_below_threshold() -> None:
    assert small_test_set_line(3) == (
        "独立测试集共 3 条——每条约占最终通过率 33 个百分点；"
        "少于 30 条时结论偶然性大（粗略经验，不是统计保证）。"
    )
    assert small_test_set_line(1) is not None


def test_small_test_set_line_silent_when_adequate_or_invalid() -> None:
    assert small_test_set_line(30) is None
    assert small_test_set_line(100) is None
    assert small_test_set_line(0) is None
