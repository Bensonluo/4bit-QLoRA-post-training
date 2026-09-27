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


BASELINE_TEMPORAL = {
    "available_at_column": "记录时间",
    "prediction_at_column": "决策时间",
    "label_end_at_column": "窗口结束",
    "validation_start": "2026-02-01T00:00:00Z",
    "test_start": "2026-03-01T00:00:00Z",
    "observation_end": "2026-04-01T00:00:00Z",
}

TEMPORAL_ROWS = [
    (
        "A1",
        "训练样本一的描述",
        "质量",
        "2026-01-01T00:00:00Z",
        "2026-01-01T00:00:00Z",
        "2026-01-02T00:00:00Z",
    ),
    (
        "A2",
        "训练样本二的描述",
        "物流",
        "2026-01-05T00:00:00Z",
        "2026-01-05T00:00:00Z",
        "2026-01-06T00:00:00Z",
    ),
    (
        "A3",
        "验证样本一的描述",
        "质量",
        "2026-02-01T00:00:00Z",
        "2026-02-01T00:00:00Z",
        "2026-02-02T00:00:00Z",
    ),
    (
        "A4",
        "测试样本一的描述",
        "物流",
        "2026-03-01T00:00:00Z",
        "2026-03-01T00:00:00Z",
        "2026-03-02T00:00:00Z",
    ),
]


def temporal_store(tmp_path):
    service = IntakeService(tmp_path / "intake")
    header = "编号,描述,类别,记录时间,决策时间,窗口结束\n"
    body = "".join(
        f"{num},{desc},{label},{available},{prediction},{end}\n"
        for num, desc, label, available, prediction, end in TEMPORAL_ROWS
    )
    session = service.create("根据描述判断售后类别", "工单.csv", (header + body).encode())
    return service, session


def test_baseline_temporal_policy_builds_point_in_time_recipe(tmp_path):
    """零密钥路径支持用户指定时间分区字段：方案走真实时间分区,边界如实声明。"""
    _, session = temporal_store(tmp_path)
    analysis = propose_baseline_analysis(
        session,
        target_column="类别",
        excluded_columns=["编号"],
        temporal_policy=BASELINE_TEMPORAL,
    )
    policy = analysis.recipe.temporal_split
    assert policy is not None
    assert policy.available_at_column == "记录时间"
    assert policy.prediction_at_column == "决策时间"
    assert policy.label_end_at_column == "窗口结束"
    inputs = {field.column for field in analysis.recipe.inputs}
    assert "窗口结束" not in inputs and "描述" in inputs
    roles = {role.column: role.role for role in analysis.task.field_roles}
    assert roles["窗口结束"] == "metadata"
    reasons = {role.column: role.reason for role in analysis.task.field_roles}
    assert "不得作为模型输入" in reasons["窗口结束"]
    assert "时间分区引用" in reasons["决策时间"]
    assert "时间分区" in analysis.recipe.split_rationale
    joined = "\n".join(finding.message for finding in analysis.findings)
    assert "时间分区方案由用户指定" in joined
    assert "时间分区按用户指定的字段与边界执行" in joined
    # 有时间方案时不再给「请配置 Agent」的泄漏预警。
    assert not any("泄漏" in finding.message for finding in analysis.findings)


def test_baseline_forecast_goal_with_temporal_policy_no_longer_demands_agent(tmp_path):
    """预测型目标配上用户指定的时间方案后,原泄漏预警不再出现。"""
    _, session = temporal_store(tmp_path)
    forecast = session.model_copy(deep=True)
    object.__setattr__(forecast, "goal", "根据描述预测未来20个交易日是否上涨")
    analysis = propose_baseline_analysis(
        forecast, target_column="类别", temporal_policy=BASELINE_TEMPORAL
    )
    assert analysis.recipe.temporal_split is not None
    assert not any("泄漏" in f.message for f in analysis.findings)
    without = propose_baseline_analysis(forecast, target_column="类别")
    assert any("泄漏" in f.message for f in without.findings)


def test_baseline_temporal_unknown_columns_rejected(tmp_path):
    _, session = temporal_store(tmp_path)
    with pytest.raises(ValueError, match="时间分区字段不在数据字段中"):
        propose_baseline_analysis(
            session,
            target_column="类别",
            temporal_policy=BASELINE_TEMPORAL | {"label_end_at_column": "不存在"},
        )


def test_baseline_temporal_applies_and_materializes_point_in_time(tmp_path):
    """基础分析 + 用户时间方案走完整真实链路:预览可确认,物化按时间分区并保留排除。"""
    service, session = temporal_store(tmp_path)
    session = service.apply_analysis(
        session,
        propose_baseline_analysis(
            session,
            target_column="类别",
            excluded_columns=["编号"],
            temporal_policy=BASELINE_TEMPORAL,
        ),
        model="baseline-deterministic",
    )
    assert session.preview is not None
    assert not any("尚未成熟" in row.issues[0] for row in session.preview.rows if row.issues)
    session = service.confirm(session.session_id, session.revision)
    full_header = "编号,描述,类别,记录时间,决策时间,窗口结束\n"
    full_body = (
        "".join(
            f"{num},{desc},{label},{available},{prediction},{end}\n"
            for num, desc, label, available, prediction, end in TEMPORAL_ROWS
        )
        + "A5,尚未成熟的描述,质量,2026-03-31T00:00:00Z,2026-03-31T00:00:00Z,2026-04-02T00:00:00Z\n"
    )
    session = service.validate_full_data(
        session.session_id, session.revision, "full.csv", (full_header + full_body).encode()
    )
    session = service.confirm_full_data(session.session_id, session.revision)
    session = service.materialize_dataset(
        session.session_id,
        session.revision,
        registry_root=tmp_path / "registry",
        validation_fraction=0.9,
        test_fraction=0.9,
        seed=999,
        independent_rows_confirmed=True,
    )
    assert session.dataset.statistics["split_method"] == "temporal"
    assert session.dataset.statistics["row_counts"] == {"train": 2, "validation": 1, "test": 1}
    assert session.dataset.statistics["excluded_rows"] == 1
    import json

    with open(session.dataset.paths["manifest"]) as handle:
        manifest = json.load(handle)
    assert manifest["metadata"]["seed"] is None
    assert manifest["metadata"]["excluded_rows"][0]["row_id"] == "r000005"
    assert manifest["metadata"]["temporal_policy"]["prediction_at_column"] == "决策时间"
