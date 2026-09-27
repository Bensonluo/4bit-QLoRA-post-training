"""Fixed evaluation cases survive revised full sources without leaking into training."""

import csv
import io
import json
from pathlib import Path

import pytest

from src.workbench.evaluation_suites import (
    EvalSuiteService,
    assert_compatible,
    evaluation_cases,
    verify_suite,
    verify_training_membership,
)
from src.workbench.intake_service import IntakeService
from src.workbench.training_preflight import _verified_partitions
from tests.unit.test_data_materialize import FULL, _full, _load
from tests.unit.test_intake_training_preflight import _republish


@pytest.fixture
def frozen(tmp_path):
    service = IntakeService(tmp_path / "intake")
    session = _full(service)
    session = service.materialize_dataset(session.session_id, session.revision)
    suites = EvalSuiteService(tmp_path / "suites")
    return service, session, suites, suites.freeze(session)


def _rows():
    return list(csv.DictReader(io.StringIO(FULL.decode())))


def _revise(service, session, rows):
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=["customer", "order", "text", "label"])
    writer.writeheader()
    writer.writerows(rows)
    session = service.validate_full_data(
        session.session_id, session.revision, "revised.csv", output.getvalue().encode()
    )
    return service.confirm_full_data(session.session_id, session.revision)


def _export(service, session, ref, **options):
    return service.materialize_dataset(
        session.session_id, session.revision, evaluation_suite=ref, **options
    )


def _anchor(session, ref):
    bound = session.model_copy(deep=True)
    bound.dataset.evaluation_suite = ref
    return bound


def test_freeze_is_immutable_reusable_and_validates_anchor(frozen):
    _, session, suites, ref = frozen
    assert suites.freeze(session) == ref == suites.get(ref["suite_id"])
    assert suites.list_suites() == [ref]
    suite = suites.load(ref)
    assert suite["anchor_dataset"]["version"] == session.dataset.version
    assert suite["case_counts"] == ref["case_counts"]
    assert verify_training_membership(ref, session)
    bound = _anchor(session, ref)
    assert verify_training_membership(ref, bound)
    report = {"issues": []}
    assert _verified_partitions(bound, report)
    assert not report["issues"]
    assert suites.freeze(bound) == ref


def test_reordered_upload_and_new_independent_rows_keep_fixed_case_ids_and_keys(frozen):
    service, original, _, ref = frozen
    before = evaluation_cases(_anchor(original, ref))
    rows = list(reversed(_rows())) + [
        {"customer": "c99", "order": "o99", "text": "new independent", "label": "yes"}
    ]
    current = _export(service, _revise(service, original, rows), ref)
    after = evaluation_cases(current)
    assert current.dataset.source_digest != original.dataset.source_digest
    assert current.dataset.version != original.dataset.version
    assert after["evaluation_key"] == before["evaluation_key"]
    assert after["cases_digest"] == before["cases_digest"]
    assert [r["metadata"]["evaluation_case_id"] for r in after["records"]] == [
        r["metadata"]["evaluation_case_id"] for r in before["records"]
    ]
    assert all(
        r["metadata"]["source_digest"] == current.dataset.source_digest for r in after["records"]
    )
    parts = _load(current.dataset)
    assert sum(map(len, parts.values())) == len(rows)
    assert any("new independent" in row["input"] for row in parts["train"])
    assert verify_training_membership(ref, current)
    assert evaluation_cases(current, "test")["evaluation_key"] != after["evaluation_key"]
    repeated = _export(service, current, ref, seed=187, validation_fraction=0.2, test_fraction=0.2)
    assert repeated.dataset.version == current.dataset.version


def test_new_heldout_object_rows_and_duplicate_rows_are_reserved_not_scored(frozen):
    service, original, suites, ref = frozen
    case = suites.load(ref)["cases"]["validation"][0]
    group = case["metadata"]["group"]
    source_row = next(r for r in _rows() if r["customer"] == group["customer"])
    rows = _rows() + [
        {
            "customer": group["customer"],
            "order": "new-order",
            "text": "new heldout",
            "label": "yes",
        },
        dict(source_row),
    ]
    current = _export(service, _revise(service, original, rows), ref)
    parts = _load(current.dataset)
    assert sum(map(len, parts.values())) == len(rows)
    assert len(evaluation_cases(current)["records"]) == ref["case_counts"]["validation"]
    heldout = [r for r in parts["validation"] if not r["metadata"]["evaluation_scored"]]
    assert len(heldout) == 2
    assert len(current.dataset.statistics["reserved_rows"]["validation"]) == 2
    assert not any("new heldout" in r["input"] for r in parts["train"])


@pytest.mark.parametrize("change", ["label", "text", "customer", "remove"])
def test_changed_or_removed_frozen_case_requires_new_suite(frozen, change):
    service, original, suites, ref = frozen
    case = suites.load(ref)["cases"]["validation"][0]
    row_id = case["metadata"]["source_row_id"]
    rows = _rows()
    index = next(i for i, r in enumerate(original.full_data.source.rows) if r.row_id == row_id)
    if change == "remove":
        rows.pop(index)
    else:
        rows[index][change] = "changed"
    current = _revise(service, original, rows)
    with pytest.raises(ValueError, match="缺失|已改变"):
        _export(service, current, ref)


@pytest.mark.parametrize("other", ["test", "train"])
def test_new_cross_partition_entity_bridge_is_blocked(frozen, other):
    service, original, _, ref = frozen
    parts = _load(original.dataset)
    dev = parts["validation"][0]["metadata"]["group"]
    second = parts[other][0]["metadata"]["group"]
    rows = _rows() + [
        {"customer": dev["customer"], "order": second["order"], "text": "bridge", "label": "yes"}
    ]
    current = _revise(service, original, rows)
    with pytest.raises(ValueError, match="连接"):
        assert_compatible(current, ref)


def test_tampered_suite_and_registry_are_rejected(frozen):
    _, original, suites, ref = frozen
    path = Path(ref["manifest_path"])
    manifest = suites.load(ref)
    manifest["cases"]["validation"][0]["output"] = "tampered"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="哈希"):
        verify_suite(ref)
    with pytest.raises(ValueError):
        evaluation_cases(_anchor(original, ref))


def test_virtual_suite_binding_is_only_allowed_for_exact_anchor_version(frozen):
    _, original, _, ref = frozen
    unrelated = _anchor(original, ref)
    _republish(unrelated, lambda splits, metadata: metadata.update(extra="different-version"))
    with pytest.raises(ValueError, match="套件不一致"):
        verify_training_membership(ref, unrelated)
    with pytest.raises(ValueError, match="套件不一致"):
        evaluation_cases(unrelated)


def test_suite_metadata_cannot_hide_actual_partition_reassignment(frozen):
    service, original, _, ref = frozen
    current = _export(service, original, ref)

    def swap(splits, metadata):
        splits["validation"], splits["test"] = splits["test"], splits["validation"]

    _republish(current, swap)
    with pytest.raises(ValueError, match="保留规则"):
        verify_training_membership(ref, current)
    with pytest.raises(ValueError, match="保留规则"):
        evaluation_cases(current)


def test_incorrect_per_row_scoring_flags_are_rejected(frozen):
    service, original, _, ref = frozen
    current = _export(service, original, ref)
    _republish(
        current,
        lambda splits, metadata: splits["validation"][0]["metadata"].update(
            evaluation_scored=False
        ),
    )
    with pytest.raises(ValueError, match="行标记"):
        verify_training_membership(ref, current)
