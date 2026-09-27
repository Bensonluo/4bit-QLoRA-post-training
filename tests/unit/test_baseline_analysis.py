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
    assert any(
        finding.kind == "needs_business_input" for finding in updated.analysis.findings
    )
