"""Baseline analysis gives a zero-Agent entry that still tells the truth."""

import pytest

from src.workbench.baseline_analysis import propose_baseline_analysis
from src.workbench.intake_models import IntakeAnalysis
from src.workbench.intake_service import IntakeService
from tests.unit.test_data_intake import CSV


@pytest.fixture()
def store(tmp_path):
    service = IntakeService(tmp_path / "intake")
    session = service.create("根据客户描述判断售后类别", "工单.csv", CSV)
    return service, session


def test_baseline_requires_target_and_covers_all_columns(store):
    _, session = store
    with pytest.raises(ValueError, match="答案列"):
        propose_baseline_analysis(session, target_column="不存在")
    analysis = propose_baseline_analysis(session, target_column="类别")
    assert isinstance(analysis, IntakeAnalysis)
    roles = {role.column: role.role for role in analysis.task.field_roles}
    assert set(roles) == {"编号", "客户描述", "类别", "处理结果"}
    assert roles["类别"] == "target"
    assert "复述题目" in analysis.recipe.instruction
    assert analysis.recipe.group_columns == []
    assert analysis.recipe.targets[0].value_kind == "categorical"
    assert not analysis.questions
    assert any("不判断业务含义" in finding.message for finding in analysis.findings)


def test_baseline_respects_groups_and_exclusions(store):
    _, session = store
    analysis = propose_baseline_analysis(
        session,
        target_column="类别",
        group_columns=["编号"],
        excluded_columns=["处理结果"],
    )
    roles = {role.column: role.role for role in analysis.task.field_roles}
    assert roles["编号"] == "group"
    assert roles["处理结果"] == "metadata"
    assert analysis.recipe.group_columns == ["编号"]
    inputs = {field.column for field in analysis.recipe.inputs}
    assert inputs == {"客户描述"}


def test_baseline_applies_through_the_real_preview_pipeline(store):
    service, session = store
    updated = service.apply_analysis(
        session,
        propose_baseline_analysis(session, target_column="类别", group_columns=["编号"]),
        model="baseline-deterministic",
    )
    assert updated.preview is not None
    rows = updated.preview.rows
    assert rows
    assert "客户描述" in rows[0].input
    assert rows[0].target in {"质量", "物流"}
    assert any(finding.kind == "needs_business_input" for finding in updated.analysis.findings)


def test_forecast_shaped_goal_gets_leakage_warning(store):
    """预测型目标在零密钥路径得到泄漏预警:不做时间分区=假效果,如实告知。"""
    _, original = store
    forecast = original.model_copy(deep=True)
    object.__setattr__(forecast, "goal", "根据披露文本预测未来20个交易日是否上涨")
    analysis = propose_baseline_analysis(forecast, target_column="类别")
    warnings = [f for f in analysis.findings if "泄漏" in f.message]
    assert warnings, "预测型目标必须有泄漏预警"
    assert "泄漏" in warnings[0].message
    plain = propose_baseline_analysis(original, target_column="类别")
    assert not any("泄漏" in f.message for f in plain.findings)


def test_high_cardinality_target_gets_wrong_column_warning(tmp_path):
    """近乎全唯一的答案列得到「可能选错答案列」预警;抽取类任务不阻断。"""
    service = IntakeService(tmp_path / "intake")
    rows = "描述,工单编号\n" + "".join(f"问题{i},GD-{i:03d}\n" for i in range(1, 11))
    session = service.create("根据描述生成工单编号", "tickets.csv", rows.encode())
    analysis = propose_baseline_analysis(session, target_column="工单编号")
    warnings = [f for f in analysis.findings if "选错" in f.message or "唯一" in f.message]
    assert warnings and "编号/ID" in warnings[0].message
    # 正常分类目标不触发
    normal = IntakeService(tmp_path / "intake2")
    normal_rows = "描述,类别\n" + "".join(
        f"问题{i},{'质量' if i % 2 else '物流'}\n" for i in range(1, 11)
    )
    session2 = normal.create("判断类别", "t.csv", normal_rows.encode())
    analysis2 = propose_baseline_analysis(session2, target_column="类别")
    assert not any("唯一" in f.message for f in analysis2.findings)


def test_label_variants_are_flagged_for_cleanup(tmp_path):
    """同一业务含义的多种写法被检出并建议归一;干净标签不触发。"""
    service = IntakeService(tmp_path / "intake")
    rows = "描述,类别\n" + "".join(
        f"问题{i},{'质量。' if i % 3 == 0 else '质量' if i % 3 == 1 else '物流'}\n"
        for i in range(1, 10)
    )
    session = service.create("判断类别", "t.csv", rows.encode())
    analysis = propose_baseline_analysis(session, target_column="类别")
    warnings = [f for f in analysis.findings if "多种写法" in f.message]
    assert warnings and "质量" in warnings[0].message and "map_values" in warnings[0].message

    clean = IntakeService(tmp_path / "intake2")
    rows2 = "描述,类别\n" + "".join(
        f"问题{i},{'质量' if i % 2 else '物流'}\n" for i in range(1, 10)
    )
    session2 = clean.create("判断类别", "t.csv", rows2.encode())
    assert not any(
        "多种写法" in f.message
        for f in propose_baseline_analysis(session2, target_column="类别").findings
    )


def test_open_text_target_gets_honest_expectation_statement(tmp_path):
    """开放文本答案在旅程开始就被告知:无自动评分,输出靠人工核对。"""
    service = IntakeService(tmp_path / "intake")
    long = "根据您的反馈我们已安排专员跟进处理并将持续关注解决进度。" * 3
    rows = "描述,答复\n" + "".join(f"问题{i},{long}（变体{i}）\n" for i in range(1, 10))
    session = service.create("根据客户问题生成标准答复", "replies.csv", rows.encode())
    analysis = propose_baseline_analysis(session, target_column="答复")
    statements = [f for f in analysis.findings if "没有可执行的自动评分规则" in f.message]
    assert statements and "人工核对" in statements[0].message
    assert "业务评分规则" in statements[0].message


def test_duplicate_inputs_are_flagged(tmp_path):
    """输入完全重复的行被检出(样本量虚高+训练重复);干净数据不触发。"""
    service = IntakeService(tmp_path / "intake")
    rows = "描述,类别\n" + "".join(
        f"{'杯子破损' if i % 2 else '屏幕碎裂'},{'质量' if i % 2 else '物流'}\n" for i in range(10)
    )
    session = service.create("判断类别", "t.csv", rows.encode())
    analysis = propose_baseline_analysis(session, target_column="类别")
    warnings = [f for f in analysis.findings if "完全重复" in f.message]
    assert warnings and "样本量虚高" in warnings[0].message

    clean = IntakeService(tmp_path / "intake2")
    rows2 = "描述,类别\n" + "".join(
        f"不同的问题描述第{i}条,{'质量' if i % 2 else '物流'}\n" for i in range(10)
    )
    session2 = clean.create("判断类别", "t.csv", rows2.encode())
    assert not any(
        "完全重复" in f.message
        for f in propose_baseline_analysis(session2, target_column="类别").findings
    )


def test_severe_class_imbalance_is_flagged(tmp_path):
    """多数类占比≥80%时警告「准确率会骗人」;均衡数据不触发。"""
    service = IntakeService(tmp_path / "intake")
    rows = "描述,类别\n" + "".join(
        f"问题{i},{'质量' if i <= 9 else '物流'}\n" for i in range(1, 11)
    )
    session = service.create("判断类别", "t.csv", rows.encode())
    analysis = propose_baseline_analysis(session, target_column="类别")
    warnings = [f for f in analysis.findings if "不均衡" in f.message]
    assert warnings and "准确率会骗人" in warnings[0].message and "90%" in warnings[0].message

    balanced = IntakeService(tmp_path / "intake2")
    rows2 = "描述,类别\n" + "".join(
        f"问题{i},{'质量' if i % 2 else '物流'}\n" for i in range(1, 11)
    )
    session2 = balanced.create("判断类别", "t.csv", rows2.encode())
    assert not any(
        "不均衡" in f.message
        for f in propose_baseline_analysis(session2, target_column="类别").findings
    )
