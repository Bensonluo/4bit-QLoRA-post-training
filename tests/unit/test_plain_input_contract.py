"""The declared input renderer must match the data actually sent to the model."""

from pathlib import Path

import pytest

from src.workbench.intake_models import FieldBinding, FieldRole, IntakeAnalysis
from src.workbench.intake_service import IntakeService
from src.workbench.recipes import preview_recipe
from src.workbench.sources import content_digest
from tests.unit.test_data_intake import CSV, analysis, recipe
from tests.unit.test_data_materialize import FULL, SAMPLE, _analysis


def two_input_plan():
    value = analysis().model_dump()
    role = next(item for item in value["task"]["field_roles"] if item["column"] == "处理结果")
    role.update(role="input", available_at_prediction=True, reason="预测时已知")
    value["recipe"]["inputs"].append({"column": "处理结果", "label": "结果"})
    return IntakeAnalysis.model_validate(value)


def test_plain_multiple_inputs_rejected_without_changing_task(tmp_path):
    service = IntakeService(tmp_path / "intake")
    session = service.create("判断类别", "sample.csv", CSV)
    before = service.load(session.session_id).model_dump()
    plan = two_input_plan()
    plan.recipe.input_render = "plain"
    with pytest.raises(ValueError, match="纯文本直出只支持一个输入字段"):
        service.apply_analysis(session, plan)
    assert service.load(session.session_id).model_dump() == before


def test_labeled_multiple_inputs_and_plain_single_input_remain_exact(tmp_path):
    service = IntakeService(tmp_path / "intake")
    session = service.create("判断类别", "sample.csv", CSV)
    plan = two_input_plan()
    session = service.apply_analysis(session, plan)
    assert session.preview.rows[0].input == "客户描述: 杯子破损\n结果: 补发"
    plain = recipe(input_render="plain")
    assert preview_recipe(session.source, plain).rows[0].input == "杯子破损"
    assert recipe().input_render == "labeled"


def test_legacy_invalid_renderer_can_be_read_but_not_rematerialized(tmp_path):
    service = IntakeService(tmp_path / "intake")
    plan = _analysis(["order"])
    plan.task.field_roles[0] = FieldRole(
        column="customer", role="input", reason="预测时已知", available_at_prediction=True
    )
    plan.recipe.inputs.append(FieldBinding(column="customer", label="客户"))
    session = service.create("分类", "sample.csv", SAMPLE)
    session = service.apply_analysis(session, plan)
    session = service.confirm(session.session_id, session.revision)
    session = service.validate_full_data(session.session_id, session.revision, "full.csv", FULL)
    session = service.confirm_full_data(session.session_id, session.revision)
    session = service.materialize_dataset(session.session_id, session.revision)
    artifact = session.dataset
    files = {name: Path(path).read_bytes() for name, path in artifact.paths.items()}

    # Old versions accepted this combination and rendered it with field labels.
    session.analysis.recipe.input_render = "plain"
    digest = content_digest(session.analysis.recipe.model_dump())
    session.preview.recipe_digest = digest
    session.full_data.preview.recipe_digest = digest
    session.full_data.approved_recipe_digest = digest
    session.dataset.recipe_digest = digest
    session = service._save(session, session.revision)
    assert service.load(session.session_id).analysis.recipe.input_render == "plain"
    with pytest.raises(ValueError, match="纯文本直出只支持一个输入字段"):
        service.materialize_dataset(session.session_id, session.revision)
    assert service.load(session.session_id).revision == session.revision
    assert {name: Path(path).read_bytes() for name, path in artifact.paths.items()} == files
