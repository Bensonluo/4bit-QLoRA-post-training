"""Full uploads remain distinct from samples and preserve business review boundaries."""

import pytest

from src.workbench.intake_service import IntakeService, next_action
from tests.unit.test_data_intake import CSV, analysis, recipe

FULL = "编号,客户描述,类别,处理结果\n100,收到破杯,质量,补发\n101,快递太慢,物流,查询\n102,杯把断了,质量,补发\n".encode()


@pytest.fixture()
def service(tmp_path):
    return IntakeService(tmp_path / "intake")


def approved(service, *, scope="sample", target_kind="categorical", group_columns=None):
    session = service.create("根据客户首次描述判断问题类型", "sample.csv", CSV, scope=scope)
    plan = recipe(targets=[{"column": "类别", "label": "类别", "value_kind": target_kind}])
    if group_columns is not None:
        plan.group_columns = group_columns
    proposal = analysis(recipe=plan.model_dump())
    for role in proposal.task.field_roles:
        if role.role == "group" and role.column not in plan.group_columns:
            role.role = "metadata"
    session = service.apply_analysis(session, proposal)
    return service.confirm(session.session_id, session.revision)


def test_full_validation_preserves_sample_identity_and_original_bytes(service):
    original = approved(service)
    session = service.validate_full_data(original.session_id, original.revision, "full.csv", FULL)
    assert next_action(session) == "review_full_data"
    assert session.source == original.source
    assert session.preview == original.preview
    assert session.analysis == original.analysis
    assert session.confirmed_examples == original.confirmed_examples
    report = session.full_data
    assert report.source.digest != original.source.digest
    assert report.preview.source_digest == report.source.digest
    assert report.sample_source_digest == original.source.digest
    assert report.sample_confirmed_revision == original.confirmed_revision
    assert report.preview.rows[0].input != original.preview.rows[0].input
    assert report.preview.rows[0].row_id == original.preview.rows[0].row_id
    assert report.preview.counts["ready"] == 3
    assert (service.root / session.session_id / "source.csv").read_bytes() == CSV
    assert (
        service.root / session.session_id / "full" / f"{report.source.digest}.csv"
    ).read_bytes() == FULL


def test_missing_required_column_keeps_readable_report_without_replacing_recipe(service):
    original = approved(service)
    session = service.validate_full_data(
        original.session_id, original.revision, "full.csv", "编号,客户描述\n1,破了\n".encode()
    )
    assert next_action(session) == "needs_full_data_revision"
    assert session.full_data.preview is None
    assert session.full_data.schema_drift["missing_required"] == ["类别"]
    assert session.full_data.issues[0].severity == "blocking"
    assert session.analysis == original.analysis
    assert len(session.full_data.source.rows) == 1
    with pytest.raises(ValueError, match="未解决"):
        service.confirm_full_data(session.session_id, session.revision)


def test_new_categories_and_columns_require_review_without_silent_mapping(service):
    session = approved(service)
    full = (
        FULL.decode()
        .replace("处理结果\n", "处理结果,新字段\n")
        .replace("补发\n", "补发,x\n")
        .replace("查询\n", "查询,y\n")
    )
    full += "103,可以开发票吗,发票,回复,z\n"
    session = service.validate_full_data(
        session.session_id, session.revision, "full.csv", full.encode()
    )
    assert next_action(session) == "review_full_data"
    assert session.full_data.new_target_values == {"类别": ["发票"]}
    assert session.full_data.schema_drift["extra"] == ["新字段"]
    assert {i.code for i in session.full_data.issues} >= {"new_categories", "extra_columns"}
    assert session.full_data.preview.rows[-1].target == "发票"
    session = service.confirm_full_data(session.session_id, session.revision, ["r000004"])
    assert next_action(session) == "awaiting_dataset_split"
    assert session.full_data.confirmed_examples[0].row_id == "r000004"
    assert session.confirmed_examples[0].expected_target == "质量"


@pytest.mark.parametrize("target_kind", ["open_text", "unspecified"])
def test_novel_open_text_is_not_mislabeled_as_category_error(service, target_kind):
    session = approved(service, target_kind=target_kind)
    full = "编号,客户描述,类别,处理结果\n1,新问题,一段新的完整回答,回复\n".encode()
    session = service.validate_full_data(session.session_id, session.revision, "full.csv", full)
    assert next_action(session) == "review_full_data"
    assert session.full_data.new_target_values == {}
    assert session.full_data.preview.rows[0].target == "一段新的完整回答"


@pytest.mark.parametrize(
    "row,code",
    [
        ("1,问题,,补发\n", "needs_label"),
        ("1,,质量,补发\n", "invalid"),
        ("1,问题,质量,补发\n2,问题,物流,查询\n", "conflict"),
        (",问题,质量,补发\n", "missing_group_values"),
    ],
)
def test_full_quality_problems_block_without_dropping_records(service, row, code):
    session = approved(service)
    full = ("编号,客户描述,类别,处理结果\n" + row).encode()
    session = service.validate_full_data(session.session_id, session.revision, "full.csv", full)
    assert next_action(session) == "needs_full_data_revision"
    issue = next(i for i in session.full_data.issues if i.code == code)
    assert issue.severity == "blocking"
    assert issue.row_ids[0] == "r000001"
    assert len(session.full_data.preview.rows) == row.count("\n")
    with pytest.raises(ValueError):
        service.confirm_full_data(session.session_id, session.revision)


def test_no_group_field_does_not_invent_grouping_requirement(service):
    session = approved(service, group_columns=[])
    session = service.validate_full_data(session.session_id, session.revision, "full.csv", FULL)
    assert next_action(session) == "review_full_data"
    assert not any(i.code == "missing_group_values" for i in session.full_data.issues)
    assert any(i.code == "split_not_validated" for i in session.full_data.issues)


def test_scope_and_revision_are_checked_before_validating_or_confirming(service):
    session = service.create("目标", "s.csv", CSV)
    with pytest.raises(ValueError, match="先确认"):
        service.validate_full_data(session.session_id, session.revision, "f.csv", FULL)
    session = approved(service)
    with pytest.raises(ValueError, match="方案已变化"):
        service.validate_full_data(session.session_id, session.revision - 1, "f.csv", FULL)
    with pytest.raises(ValueError, match="不能自动当作全量"):
        service.validate_full_data(session.session_id, session.revision)
    current = service.validate_full_data(session.session_id, session.revision, "f.csv", FULL)
    with pytest.raises(ValueError, match="已变化"):
        service.confirm_full_data(current.session_id, session.revision)
    with pytest.raises(ValueError, match="有效全量预览行"):
        service.confirm_full_data(current.session_id, current.revision, ["r999999"])


def test_business_changes_invalidate_full_report_but_preserve_its_evidence(service):
    session = approved(service)
    session = service.validate_full_data(session.session_id, session.revision, "f.csv", FULL)
    session = service.confirm_full_data(session.session_id, session.revision)
    digest = session.full_data.source.digest
    session = service.answer(session.session_id, "这里的类别还要区分新的业务范围")
    assert session.full_data.status == "stale"
    assert session.full_data.confirmed_revision is None
    assert session.full_data.source.digest == digest
    assert next_action(session) == "awaiting_analysis"
    session = service.apply_analysis(session, analysis())
    assert session.full_data.status == "stale"
    session = service.confirm(session.session_id, session.revision)
    assert next_action(session) == "awaiting_full_data"
    with pytest.raises(ValueError):
        service.confirm_full_data(session.session_id, session.revision)
    session = service.validate_full_data(session.session_id, session.revision, "f.csv", FULL)
    assert next_action(session) == "review_full_data"


def test_full_initial_upload_can_be_validated_without_uploading_again(service):
    session = approved(service, scope="full")
    session = service.validate_full_data(session.session_id, session.revision)
    assert session.full_data.source.digest == session.source.digest
    assert next_action(session) == "review_full_data"
    assert not any(i.code == "same_as_sample" for i in session.full_data.issues)


def test_reanalysis_without_feedback_also_invalidates_existing_report(service):
    session = approved(service)
    session = service.validate_full_data(session.session_id, session.revision, "f.csv", FULL)
    updated = service.apply_analysis(session, analysis())
    assert updated.full_data.status == "stale"
    assert next_action(updated) == "review_preview"


def test_full_answer_disagreement_with_approved_sample_uses_input_not_row_id(service):
    session = approved(service)
    full = "编号,客户描述,类别,处理结果\n10,全新问题,质量,补发\n11,杯子破损,物流,补发\n".encode()
    session = service.validate_full_data(session.session_id, session.revision, "f.csv", full)
    issue = next(i for i in session.full_data.issues if i.code == "sample_answer_disagreement")
    assert issue.row_ids == ["r000002"]
    assert session.full_data.preview.counts["ready"] == 2
    assert next_action(session) == "needs_full_data_revision"
