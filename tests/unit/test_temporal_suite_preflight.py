"""Temporal suites preserve fixed cases and account for every excluded source row."""

import csv
import io
from copy import deepcopy

import pytest

from src.data_flywheel.dataset_registry import LocalDatasetRegistry
from src.workbench.evaluation_suites import (
    EvalSuiteService,
    assert_compatible,
    evaluation_cases,
    verify_training_membership,
)
from src.workbench.intake_models import (
    DataRecipe,
    FieldBinding,
    FieldRole,
    IntakeAnalysis,
    TaskSpec,
    TemporalSplitPolicy,
)
from src.workbench.intake_service import IntakeService
from src.workbench.training_preflight import _verified_partitions, preflight_dataset
from tests.unit.test_intake_training_preflight import _republish
from tests.unit.test_intake_training_preflight import tokenizer as tokenizer

POLICY = dict(
    available_at_column="available_at",
    prediction_at_column="prediction_at",
    label_end_at_column="label_end_at",
    validation_start="2024-03-01T00:00:00Z",
    test_start="2024-05-01T00:00:00Z",
    observation_end="2024-06-30T23:59:59Z",
)


def _row(event, prediction, label_end, label="yes", available=None):
    return {
        "event_id": event,
        "symbol": "SAME",
        "text": "公开财报事件 " + event,
        "available_at": available or prediction + "T00:00:00Z",
        "prediction_at": prediction + "T01:00:00Z",
        "label_end_at": label_end + "T01:00:00Z",
        "label": label,
    }


def _rows():
    return [
        _row("train-1", "2024-01-02", "2024-01-30"),
        _row("train-2", "2024-01-15", "2024-02-12"),
        _row("purged-train", "2024-02-15", "2024-03-14"),
        _row("dev-1", "2024-03-01", "2024-03-29"),
        _row("dev-2", "2024-04-01", "2024-04-29"),
        _row("purged-dev", "2024-04-15", "2024-05-13"),
        _row("test-1", "2024-05-01", "2024-05-29"),
        _row("test-2", "2024-06-01", "2024-06-29"),
        _row("unmatured", "2024-06-15", "2024-07-13", label=""),
    ]


def _csv(rows):
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue().encode()


def _setup(root, *, rows=None, policy=None):
    service = IntakeService(root / "intake")
    sample = _row("sample", "2023-11-01", "2023-11-29")
    session = service.create(
        "根据公开财报事件预测随后一段时间上涨或未上涨", "sample.csv", _csv([sample])
    )
    roles = {"event_id": "group", "text": "input", "prediction_at": "input", "label": "target"}
    analysis = IntakeAnalysis(
        task=TaskSpec(
            goal=session.goal,
            usage_input="预测时已公开的财报",
            desired_output="方向类别",
            row_meaning="一个财报事件",
            supervision_source="随后窗口观测标签",
            success_criteria=["独立后续时间段检验"],
            field_roles=[
                FieldRole(
                    column=key,
                    role=roles.get(key, "metadata"),
                    reason="已确认字段语义",
                    available_at_prediction=True if roles.get(key) == "input" else None,
                )
                for key in sample
            ],
        ),
        findings=[],
        recipe=DataRecipe(
            instruction="判断方向",
            inputs=[
                FieldBinding(column="text", label="财报"),
                FieldBinding(column="prediction_at", label="预测时点"),
            ],
            targets=[FieldBinding(column="label", label="方向", value_kind="categorical")],
            group_columns=["event_id"],
            temporal_split=TemporalSplitPolicy(**(policy or POLICY)),
        ),
        training_approach="SFT",
        next_steps=["检查时间边界"],
    )
    session = service.apply_analysis(session, analysis)
    session = service.confirm(session.session_id, session.revision)
    session = _revise(service, session, rows or _rows())
    session = service.materialize_dataset(session.session_id, session.revision)
    return service, session


def _revise(service, session, rows):
    session = service.validate_full_data(
        session.session_id, session.revision, "full.csv", _csv(rows)
    )
    return service.confirm_full_data(session.session_id, session.revision)


@pytest.fixture
def temporal(tmp_path):
    service, session = _setup(tmp_path)
    suites = EvalSuiteService(tmp_path / "suites")
    return service, session, suites, suites.freeze(session)


def test_temporal_preflight_accounts_for_purged_and_unmatured_rows(temporal, tokenizer):
    _, session, _, ref = temporal
    report = preflight_dataset(session, tokenizer, 128)
    assert report["status"] == "passed"
    assert {key: value["rows"] for key, value in report["splits"].items()} == {
        "train": 2,
        "validation": 2,
        "test": 2,
    }
    registry = LocalDatasetRegistry(session.dataset.registry_root)
    metadata = registry.get_split_manifest(session.dataset.name, session.dataset.version)[
        "metadata"
    ]
    assert metadata["temporal_policy"] == POLICY
    excluded = metadata["excluded_rows"]
    assert len(excluded) == 3
    assert {row["reason"] for row in excluded} == {
        "label_window_crosses_validation_start",
        "label_window_crosses_test_start",
        "label_not_mature",
    }
    assert next(row for row in excluded if row["reason"] == "label_not_mature")["target"] is None
    assert {row["row_id"] for row in excluded}.isdisjoint(row["row_id"] for row in report["rows"])
    assert len(report["rows"]) + len(excluded) == len(session.full_data.source.rows)
    assert verify_training_membership(ref, session)


def test_revised_source_keeps_cases_but_new_events_follow_time_not_default_train(temporal):
    service, original, suites, ref = temporal
    anchor = original.model_copy(deep=True)
    anchor.dataset.evaluation_suite = ref
    previous = evaluation_cases(anchor)
    rows = list(reversed(_rows())) + [
        _row("new-train", "2024-01-10", "2024-02-07"),
        _row("new-dev", "2024-03-10", "2024-04-07"),
        _row("new-test", "2024-05-10", "2024-06-07"),
        _row("future", "2024-08-01", "2024-08-29", label=""),
    ]
    session = _revise(service, original, rows)
    session = service.materialize_dataset(
        session.session_id, session.revision, evaluation_suite=ref
    )
    report = {"issues": []}
    parts = _verified_partitions(session, report)
    assert not report["issues"]
    assert {key: len(value) for key, value in parts.items()} == {
        "train": 3,
        "validation": 3,
        "test": 3,
    }
    assert any(row["metadata"]["group"]["event_id"] == "new-train" for row in parts["train"])
    assert not any(
        row["metadata"]["group"]["event_id"] in {"future", "new-dev", "new-test"}
        for row in parts["train"]
    )
    current = evaluation_cases(session)
    assert current["evaluation_key"] == previous["evaluation_key"]
    assert len(current["records"]) == ref["case_counts"]["validation"] == 2
    assert len(session.dataset.statistics["reserved_rows"]["validation"]) == 1
    assert all(
        row["metadata"]["source_digest"] == session.dataset.source_digest
        for row in current["records"]
    )
    assert verify_training_membership(ref, session)
    assert suites.load(ref)["temporal_policy"] == POLICY


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_exclusion",
        "exclusion_reason",
        "exclusion_original",
        "row_time",
        "policy",
        "swap_partitions",
    ],
)
def test_preflight_recomputes_time_and_exclusion_evidence_after_registry_republish(
    temporal, mutation
):
    _, original, _, _ = temporal
    session = original.model_copy(deep=True)

    def mutate(parts, metadata):
        if mutation == "missing_exclusion":
            metadata["excluded_rows"].pop()
        elif mutation == "exclusion_reason":
            metadata["excluded_rows"][0]["reason"] = "pretend-safe"
        elif mutation == "exclusion_original":
            metadata["excluded_rows"][0]["original"]["label"] = "fabricated"
        elif mutation == "row_time":
            parts["train"][0]["metadata"]["temporal"]["label_end_at"] = "2024-06-01T00:00:00Z"
        elif mutation == "policy":
            metadata["temporal_policy"]["observation_end"] = "2025-01-01T00:00:00Z"
        else:
            parts["train"], parts["test"] = parts["test"], parts["train"]

    _republish(session, mutate)
    with pytest.raises(ValueError):
        _verified_partitions(session, {"issues": []})


def test_changing_case_time_without_changing_model_input_cannot_reuse_suite(temporal):
    service, original, _, ref = temporal
    rows = _rows()
    rows[3]["available_at"] = "2024-02-28T00:00:00Z"
    revised = _revise(service, original, rows)
    with pytest.raises(ValueError, match="时间已改变"):
        assert_compatible(revised, ref)


def test_observation_end_and_boundaries_are_frozen_in_suite(temporal, tmp_path):
    _, _, _, ref = temporal
    policy = {**POLICY, "observation_end": "2024-07-01T00:00:00Z"}
    _, session = _setup(tmp_path / "another", policy=policy)
    with pytest.raises(ValueError, match="观察截止"):
        assert_compatible(session, ref)


def test_same_company_can_cross_time_but_same_event_cannot(temporal):
    service, original, _, _ = temporal
    parts = _verified_partitions(original, {"issues": []})
    assert all(parts.values())  # All original rows have symbol SAME; event IDs are distinct.
    rows = _rows()
    rows[3]["event_id"] = rows[0]["event_id"]
    candidate = service.validate_full_data(
        original.session_id, original.revision, "event-cross.csv", _csv(rows)
    )
    assert any(issue.severity == "blocking" for issue in candidate.full_data.issues)
    with pytest.raises(ValueError):
        service.confirm_full_data(candidate.session_id, candidate.revision)


def test_temporal_case_semantics_and_legacy_case_hash_remain_distinct(temporal):
    from src.workbench.evaluation_suites import _semantic

    _, original, suites, ref = temporal
    case = suites.load(ref)["cases"]["validation"][0]
    assert "temporal" in _semantic(case)
    changed = deepcopy(case)
    changed["metadata"]["temporal"]["available_at"] = "2024-02-28T00:00:00Z"
    assert _semantic(case) != _semantic(changed)
    legacy = deepcopy(case)
    legacy["metadata"].pop("temporal")
    assert set(_semantic(legacy)) == {"instruction", "input", "output", "group"}
    bound = original.model_copy(deep=True)
    bound.dataset.evaluation_suite = ref
    assert verify_training_membership(ref, bound)
