"""One malformed saved business report must not hide healthy history."""

from copy import deepcopy
from uuid import uuid4

import pytest

from src.workbench.business_evaluation import (
    BusinessEvaluationService,
    EvaluationProtocol,
    Generation,
    comparison_key,
)
from tests.unit.test_business_evaluation import fixture_factory, model_paths, task  # noqa: F401


@pytest.fixture
def reports(request, tmp_path):
    _, session = request.getfixturevalue("task")
    models = request.getfixturevalue("model_paths")
    service = BusinessEvaluationService(tmp_path / "evaluation")
    valid = service.compare(
        session,
        models,
        EvaluationProtocol("classification_exact"),
        runtime_factory=fixture_factory(session, []),
    )
    return service, session, valid


@pytest.mark.parametrize(
    "field,value",
    [
        ("created_at", None),
        ("created_at", 17),
        ("created_at", " "),
        ("status", None),
        ("status", "done"),
        ("notes", None),
        ("notes", ["说明", 17]),
        ("models", []),
    ],
)
def test_damaged_metadata_is_isolated_from_healthy_sibling(reports, field, value):
    service, session, valid = reports
    bad = deepcopy(valid)
    bad.evaluation_id = uuid4().hex
    setattr(bad, field, value)
    service._save(bad)
    before = {path.name: path.read_bytes() for path in service.root.iterdir()}

    assert [report.evaluation_id for report in service.list_reports(session.dataset.version)] == [
        valid.evaluation_id
    ]
    assert [item["evaluation_id"] for item in service.load_errors] == [bad.evaluation_id]
    assert {path.name: path.read_bytes() for path in service.root.iterdir()} == before
    with pytest.raises(ValueError):
        service.get_report(bad.evaluation_id)


def test_malformed_suite_metadata_cannot_crash_history(reports):
    service, session, valid = reports
    bad = deepcopy(valid)
    bad.evaluation_id = uuid4().hex
    bad.dataset["evaluation_suite"] = ["not-a-reference"]
    service._save(bad)
    assert [report.evaluation_id for report in service.list_reports(session.dataset.version)] == [
        valid.evaluation_id
    ]
    assert [item["evaluation_id"] for item in service.load_errors] == [bad.evaluation_id]


@pytest.mark.parametrize("damage", ["label", "duplicate_label", "errors", "requested", "source"])
def test_damaged_model_details_do_not_hide_healthy_history(reports, damage):
    service, _, valid = reports
    bad = deepcopy(valid)
    bad.evaluation_id = uuid4().hex
    if damage == "label":
        bad.models[0].pop("label")
    elif damage == "duplicate_label":
        bad.models[1]["label"] = bad.models[0]["label"]
    elif damage == "errors":
        bad.models[0]["errors"] = None
    elif damage == "requested":
        bad.models[0]["requested_model"] = ["not a model"]
    else:
        for model in bad.models:
            model["rows"][0]["source"] = None
    service._save(bad)
    assert [report.evaluation_id for report in service.list_reports()] == [valid.evaluation_id]
    assert [item["evaluation_id"] for item in service.load_errors] == [bad.evaluation_id]


def test_null_suite_reference_can_be_filtered_without_crashing(reports):
    service, _, report = reports
    report.dataset["evaluation_suite"] = None
    report.comparison_key = comparison_key(report.dataset, report.protocol)
    service._save(report)
    assert service.get_report(report.evaluation_id).dataset["evaluation_suite"] is None
    assert service.list_reports(suite_id="unrelated") == []
    assert service.load_errors == []


@pytest.mark.parametrize("field", ["scorer", "row_count"])
def test_running_report_with_no_models_still_requires_usable_metadata(reports, field):
    service, _, valid = reports
    bad = deepcopy(valid)
    bad.evaluation_id = uuid4().hex
    bad.status = "running"
    bad.models = []
    (bad.protocol if field == "scorer" else bad.dataset).pop(field)
    bad.comparison_key = comparison_key(bad.dataset, bad.protocol)
    service._save(bad)
    assert [report.evaluation_id for report in service.list_reports()] == [valid.evaluation_id]
    assert [item["evaluation_id"] for item in service.load_errors] == [bad.evaluation_id]


def test_failed_observations_cannot_claim_clean_completion(task, model_paths, tmp_path):  # noqa: F811
    session = task[1]
    service = BusinessEvaluationService(tmp_path / "evaluation")
    report = service.compare(
        session,
        model_paths,
        EvaluationProtocol("classification_exact"),
        runtime_factory=fixture_factory(
            session, [], lambda *_: Generation("truncated", truncated=True)
        ),
    )
    assert report.status == "completed_with_failures"
    report.status = "completed"
    service._save(report)
    with pytest.raises(ValueError):
        service.get_report(report.evaluation_id)


def test_release_failure_cannot_claim_completed_comparison(task, model_paths, tmp_path):  # noqa: F811
    session = task[1]
    service = BusinessEvaluationService(tmp_path / "evaluation")
    report = service.compare(
        session,
        model_paths,
        EvaluationProtocol("classification_exact"),
        runtime_factory=fixture_factory(session, [], close_error=True),
    )
    assert report.status == "release_failed"
    report.status = "completed"
    service._save(report)
    with pytest.raises(ValueError, match="释放失败"):
        service.get_report(report.evaluation_id)
