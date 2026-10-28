"""Stored decisions must agree with the verified final report."""

import json
import sqlite3

import pytest

from src.workbench.acceptance import AcceptanceService
from src.workbench.business_evaluation import EvaluationProtocol, Generation
from tests.unit.test_acceptance import criteria, runtime
from tests.unit.test_acceptance import model_paths as model_paths
from tests.unit.test_acceptance import task as task


def update(service, identity, edit):
    with sqlite3.connect(service.database) as db:
        payload = db.execute("SELECT snapshot FROM acceptances WHERE id=?", (identity,)).fetchone()[
            0
        ]
        record = json.loads(payload)
        edit(record)
        db.execute("UPDATE acceptances SET snapshot=? WHERE id=?", (json.dumps(record), identity))


def create_attempts(task, model_paths, tmp_path, *, manual=False):
    session = task[1]
    service = AcceptanceService(tmp_path / "acceptance", tmp_path / "eval")
    protocol = EvaluationProtocol("open_review" if manual else "classification_exact")
    standard = criteria(metric="manual_acceptance_rate" if manual else "exact_match")
    first = service.prepare(session, model_paths[0], protocol, standard)
    second = service.prepare(session, model_paths[1], protocol, standard)
    result = service.run(
        first["acceptance_id"],
        session,
        runtime_factory=runtime(service, first, [], lambda _i, _answer: Generation("不符合")),
    )
    return service, session, result, second


def test_automatic_result_corruption_is_rejected_while_healthy_attempt_remains_readable(
    task, model_paths, tmp_path
):
    service, session, result, healthy = create_attempts(task, model_paths, tmp_path)
    assert result["result"]["decision"] == "failed"
    update(
        service, result["acceptance_id"], lambda record: record["result"].update(decision="passed")
    )
    with pytest.raises(ValueError, match="结论或人工判断"):
        service.get(result["acceptance_id"])
    with pytest.raises(ValueError):
        service.list_acceptances(session.session_id)
    records, unreadable = service.list_acceptances_tolerant(session.session_id)
    assert records == [healthy]
    assert unreadable == 1
    assert service.get(healthy["acceptance_id"]) == healthy


def test_legacy_result_without_digest_is_recomputed_from_report(task, model_paths, tmp_path):
    service, _, result, _ = create_attempts(task, model_paths, tmp_path)
    identity = result["acceptance_id"]
    update(service, identity, lambda record: record.pop("outcome_digest"))
    legacy = service.get(identity)
    assert legacy["result"] == result["result"]
    update(service, identity, lambda record: record["result"].update(decision="passed"))
    with pytest.raises(ValueError, match="结论与已保存"):
        service.get(identity)


def test_manual_review_changes_are_recorded_and_detected(task, model_paths, tmp_path):
    service, _, result, _ = create_attempts(task, model_paths, tmp_path, manual=True)
    identity = result["acceptance_id"]
    assert result["status"] == "needs_business_review"
    assert service.get(identity) == result
    result = service.review(identity, [{"index": 0, "decision": "rejected", "reason": "不符合"}])
    assert service.get(identity) == result
    update(service, identity, lambda record: record["decisions"][0].update(reason="被改写"))
    with pytest.raises(ValueError, match="结论或人工判断"):
        service.get(identity)


@pytest.mark.parametrize("damage", ["unknown_index", "duplicate", "invalid_decision"])
def test_legacy_manual_decisions_are_checked_against_actual_cases(
    task, model_paths, tmp_path, damage
):
    service, _, result, _ = create_attempts(task, model_paths, tmp_path, manual=True)
    identity = result["acceptance_id"]
    assert result["decisions"] == []
    assert service.get(identity)["result"]["decision"] == "pending_review"
    item = {"index": 0, "decision": "accepted", "reason": "已核对", "locked": False}

    def corrupt(record):
        record.pop("outcome_digest")
        if damage == "unknown_index":
            item["index"] = 999
        elif damage == "invalid_decision":
            item["decision"] = "maybe"
        record["decisions"] = [item, item.copy()] if damage == "duplicate" else [item]

    update(service, identity, corrupt)
    with pytest.raises(ValueError, match="人工验收判断"):
        service.get(identity)


def test_legacy_manual_result_cannot_accept_a_truncated_case(task, model_paths, tmp_path):
    session = task[1]
    service = AcceptanceService(tmp_path / "acceptance", tmp_path / "eval")
    record = service.prepare(
        session,
        model_paths[0],
        EvaluationProtocol("open_review"),
        criteria(metric="manual_acceptance_rate"),
    )
    result = service.run(
        record["acceptance_id"],
        session,
        runtime_factory=runtime(
            service,
            record,
            [],
            lambda index, _: (
                Generation("截断", truncated=True) if index == 0 else Generation("完整")
            ),
        ),
    )
    identity = result["acceptance_id"]

    def corrupt(saved):
        saved.pop("outcome_digest")
        saved["decisions"][0].update(decision="accepted", locked=False)

    update(service, identity, corrupt)
    with pytest.raises(ValueError, match="失败或截断"):
        service.get(identity)


@pytest.mark.parametrize("status", ["prepared", "running", "blocked", "failed"])
def test_unfinished_or_failed_legacy_records_cannot_display_passed(
    task, model_paths, tmp_path, status
):
    service, _, _, healthy = create_attempts(task, model_paths, tmp_path)
    identity = healthy["acceptance_id"]
    expected = "pending_run" if status in {"prepared", "running"} else "insufficient_evidence"

    def legacy(record):
        record.pop("outcome_digest")
        record["status"] = status
        record["result"] = {"decision": expected}

    update(service, identity, legacy)
    assert service.get(identity)["result"]["decision"] == expected
    update(service, identity, lambda record: record["result"].update(decision="passed"))
    with pytest.raises(ValueError, match="状态与结论"):
        service.get(identity)


def test_readonly_tolerant_listing_skips_corruption_without_writes(task, model_paths, tmp_path):
    service, session, result, healthy = create_attempts(task, model_paths, tmp_path)
    update(service, result["acceptance_id"], lambda record: record["result"].update(score=1.0))
    before = {p.relative_to(tmp_path): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    readonly = AcceptanceService(service.root, service.evaluations.root, read_only=True)
    records, count = readonly.list_acceptances_tolerant(session.session_id)
    assert records == [healthy]
    assert count == 1
    after = {p.relative_to(tmp_path): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    assert before == after
