"""A damaged acceptance row must never identify a different blind-test attempt."""

import json
import sqlite3

import pytest

from src.workbench.acceptance import AcceptanceService
from src.workbench.business_evaluation import EvaluationProtocol
from tests.unit.test_acceptance import criteria, model_paths, runtime, task  # noqa: F401


@pytest.fixture
def attempts(task, model_paths, tmp_path):  # noqa: F811
    service = AcceptanceService(tmp_path / "acceptance", tmp_path / "evaluation")
    session = task[1]
    first = service.prepare(
        session, model_paths[0], EvaluationProtocol("classification_exact"), criteria()
    )
    second = service.prepare(
        session, model_paths[1], EvaluationProtocol("classification_exact"), criteria()
    )
    return service, session, first, second


def replace_row(service, first, second):
    with sqlite3.connect(service.database) as db:
        snapshot = db.execute(
            "SELECT snapshot FROM acceptances WHERE id=?", (second["acceptance_id"],)
        ).fetchone()[0]
        db.execute(
            "UPDATE acceptances SET snapshot=? WHERE id=?", (snapshot, first["acceptance_id"])
        )


def test_snapshot_for_another_acceptance_cannot_be_read_or_run(attempts):
    service, session, first, second = attempts
    replace_row(service, first, second)
    with pytest.raises(ValueError, match="身份或版本"):
        service.get(first["acceptance_id"])
    assert service.get(second["acceptance_id"]) == second
    with pytest.raises(ValueError, match="身份或版本"):
        service.list_acceptances()
    events = []
    with pytest.raises(ValueError, match="身份或版本"):
        service.run(
            first["acceptance_id"], session, runtime_factory=runtime(service, second, events)
        )
    assert events == []
    with sqlite3.connect(service.database) as db:
        assert db.execute("SELECT COUNT(*) FROM revealed_cases").fetchone()[0] == 0


def test_row_revision_must_match_snapshot(attempts):
    service, _, first, _ = attempts
    with sqlite3.connect(service.database) as db:
        db.execute(
            "UPDATE acceptances SET revision=revision+1 WHERE id=?", (first["acceptance_id"],)
        )
    with pytest.raises(ValueError, match="身份或版本"):
        service.get(first["acceptance_id"])


def test_readonly_rejects_wrong_identity_without_modifying_store(attempts):
    service, _, first, second = attempts
    replace_row(service, first, second)
    before = {
        p.relative_to(service.root): p.read_bytes() for p in service.root.rglob("*") if p.is_file()
    }
    readonly = AcceptanceService(service.root, service.evaluations.root, read_only=True)
    with pytest.raises(ValueError, match="身份或版本"):
        readonly.get(first["acceptance_id"])
    assert readonly.get(second["acceptance_id"]) == second
    after = {
        p.relative_to(service.root): p.read_bytes() for p in service.root.rglob("*") if p.is_file()
    }
    assert after == before


def test_corruption_between_preflight_and_claim_cannot_reveal_another_attempt(
    attempts, monkeypatch
):
    service, session, first, second = attempts
    original = service._training_provenance

    def replace_after_check(*args):
        result = original(*args)
        replace_row(service, first, second)
        return result

    monkeypatch.setattr(service, "_training_provenance", replace_after_check)
    events = []
    with pytest.raises(ValueError, match="身份或版本"):
        service.run(
            first["acceptance_id"], session, runtime_factory=runtime(service, first, events)
        )
    assert events == []
    with sqlite3.connect(service.database) as db:
        assert db.execute("SELECT COUNT(*) FROM revealed_cases").fetchone()[0] == 0
    assert service.get(second["acceptance_id"]) == second


def test_rebinding_other_valid_frozen_payload_during_check_is_rejected(attempts, monkeypatch):
    service, session, first, second = attempts
    original = service._training_provenance

    def replace_after_check(*args):
        result = original(*args)
        replacement = {**second, "acceptance_id": first["acceptance_id"]}
        with sqlite3.connect(service.database) as db:
            db.execute(
                "UPDATE acceptances SET snapshot=? WHERE id=?",
                (json.dumps(replacement), first["acceptance_id"]),
            )
        return result

    monkeypatch.setattr(service, "_training_provenance", replace_after_check)
    events = []
    with pytest.raises(ValueError, match="检查期间"):
        service.run(
            first["acceptance_id"], session, runtime_factory=runtime(service, first, events)
        )
    assert not events
    with sqlite3.connect(service.database) as db:
        assert db.execute("SELECT COUNT(*) FROM revealed_cases").fetchone()[0] == 0


def test_legitimate_concurrent_transition_returns_current_record(attempts, monkeypatch):
    service, session, first, _ = attempts
    original = service._training_provenance

    def other_claim_after_check(*args):
        result = original(*args)
        current = service.get(first["acceptance_id"])
        current.update(
            status="blocked",
            blind_test=False,
            result={"decision": "insufficient_evidence", "reason": "heldout_already_revealed"},
        )
        with service._connect() as db:
            service._write(db, current)
        return result

    monkeypatch.setattr(service, "_training_provenance", other_claim_after_check)
    events = []
    result = service.run(
        first["acceptance_id"], session, runtime_factory=runtime(service, first, events)
    )
    assert result == service.get(first["acceptance_id"])
    assert result["status"] == "blocked"
    assert not events
