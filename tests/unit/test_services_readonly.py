"""Browsing real SQLite stores preserves files and their existing integrity checks."""

import json
import sqlite3
from pathlib import Path

import pytest

from src.workbench.acceptance import AcceptanceService
from src.workbench.business_evaluation import EvaluationProtocol
from src.workbench.business_scoring import ScoringService, _binding, load_confirmed
from src.workbench.iterations import IterationService
from src.workbench.sources import canonical, content_digest
from src.workbench.training_plans import TrainingPlanService
from tests.unit.test_acceptance import criteria
from tests.unit.test_business_evaluation import model_paths, task  # noqa: F401
from tests.unit.test_business_scoring import recipe_for

KINDS = ("iterations", "acceptance", "scoring", "plans")
LISTS = {
    "iterations": "list_iterations",
    "acceptance": "list_acceptances",
    "scoring": "list_specs",
    "plans": "list_plans",
}
TABLES = {
    "iterations": "iterations",
    "acceptance": "acceptances",
    "scoring": "scoring_specs",
    "plans": "plans",
}
PREFIXES = {"iterations": "it-", "acceptance": "ac-", "scoring": "sc-", "plans": "tp-"}


def service(kind, root, *, read_only):
    own = root / kind
    if kind == "iterations":
        return IterationService(own, root / "training", root / "evaluations", read_only=read_only)
    if kind == "acceptance":
        return AcceptanceService(own, root / "evaluations", read_only=read_only)
    if kind == "scoring":
        return ScoringService(own, read_only=read_only)
    return TrainingPlanService(own, root / "training", read_only=read_only)


def snapshot(root):
    return {
        str(path.relative_to(root)): path.read_bytes() if path.is_file() else None
        for path in root.rglob("*")
    }


@pytest.mark.parametrize("kind", KINDS)
def test_readonly_missing_store_never_creates_any_parent_or_nested_store(tmp_path, kind):
    reader = service(kind, tmp_path / "new workspace #1", read_only=True)
    assert isinstance(reader.database, Path)
    assert getattr(reader, LISTS[kind])() == []
    with pytest.raises(ValueError, match="找不到"):
        reader.get(PREFIXES[kind] + "1" * 32)
    with pytest.raises(sqlite3.OperationalError):
        reader._connect()
    assert snapshot(tmp_path) == {}


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("corruption", ["invalid_bytes", "missing_schema"])
def test_existing_broken_store_is_reported_not_treated_as_empty(tmp_path, kind, corruption):
    reader = service(kind, tmp_path, read_only=True)
    reader.database.parent.mkdir()
    if corruption == "invalid_bytes":
        reader.database.write_bytes(b"not a SQLite database")
    else:
        with sqlite3.connect(reader.database) as connection:
            connection.execute("CREATE TABLE unrelated (id INTEGER)")
    before = snapshot(tmp_path)
    with pytest.raises(sqlite3.DatabaseError):
        getattr(reader, LISTS[kind])()
    with pytest.raises(sqlite3.DatabaseError):
        reader.get(PREFIXES[kind] + "1" * 32)
    assert snapshot(tmp_path) == before


@pytest.fixture(params=KINDS)
def persisted(request, tmp_path, task, model_paths):  # noqa: F811
    kind = request.param
    session = task[1]
    root = tmp_path / "existing workspace #1"
    writer = service(kind, root, read_only=False)
    if kind == "iterations":
        record = writer._save(
            {
                "iteration_id": "it-" + "1" * 32,
                "session_id": session.session_id,
                "status": "proposed",
            }
        )
        identity = record["iteration_id"]
    elif kind == "acceptance":
        record = writer.prepare(
            session, model_paths[0], EvaluationProtocol("classification_exact"), criteria()
        )
        identity = record["acceptance_id"]
    elif kind == "scoring":
        # A persisted, digest-bound recipe fixture; reading it must not execute
        # scoring code or rerun the OS sandbox validation from its draft phase.
        recipe = recipe_for({"input": "fixture input", "expected": "yes"}).model_dump()
        record = {
            "scoring_id": "sc-" + "1" * 32,
            "session_id": session.session_id,
            "recipe": recipe,
            "binding": _binding(session),
            "status": "confirmed",
        }
        record["spec_digest"] = content_digest({"recipe": recipe, "binding": record["binding"]})
        record["record_digest"] = content_digest(record)
        with writer._connect() as connection:
            connection.execute(
                "INSERT INTO scoring_specs VALUES (?, ?)", (record["scoring_id"], canonical(record))
            )
        identity = record["scoring_id"]
    else:
        context = writer.context(session, [])
        record = writer.save(
            session,
            {
                "status": "needs_data",
                "model_path": None,
                "max_length": None,
                "training_options": {},
                "lora_options": {},
                "model_options": {},
                "rationale": ["先检查业务数据"],
                "limitations": ["尚未检查本地模型"],
                "business_questions": ["仍需确认模型选择"],
            },
            context,
            [],
        )
        identity = record["plan_id"]
    return kind, root, writer, identity, record, session


def test_existing_record_reads_match_writable_service_without_schema_or_file_changes(persisted):
    kind, root, writer, identity, record, session = persisted
    before = snapshot(root)
    reader = service(kind, root, read_only=True)
    assert reader.get(identity) == writer.get(identity) == record
    assert getattr(reader, LISTS[kind])() == [record]
    assert getattr(reader, LISTS[kind])(session.session_id) == [record]
    assert getattr(reader, LISTS[kind])("unrelated-session") == []
    # Mode=ro is an SQLite boundary, independent of UI buttons and permissions.
    with (
        reader._connect(timeout=1) as connection,
        pytest.raises(sqlite3.OperationalError, match="readonly"),
    ):
        connection.execute("CREATE TABLE forbidden_write (id INTEGER)")
    assert snapshot(root) == before


@pytest.mark.parametrize("persisted", ["scoring", "acceptance"], indirect=True)
def test_readonly_does_not_bypass_existing_scoring_or_acceptance_integrity_checks(persisted):
    kind, root, writer, identity, record, _ = persisted
    changed = json.loads(canonical(record))
    if kind == "scoring":
        changed["recipe"]["business_standard"] = "changed without a new digest"
    else:
        changed["criteria"]["minimum_score"] = 0.1
    with writer._connect() as connection:
        connection.execute(
            f"UPDATE {TABLES[kind]} SET snapshot=? WHERE id=?", (canonical(changed), identity)
        )
    reader = service(kind, root, read_only=True)
    before = snapshot(root)
    with pytest.raises(ValueError):
        reader.get(identity)
    assert snapshot(root) == before


@pytest.mark.parametrize("kind", KINDS)
def test_readonly_mutation_is_rejected_before_validation_or_external_work(tmp_path, kind):
    reader = service(kind, tmp_path, read_only=True)
    with pytest.raises(PermissionError, match="只允许浏览"):
        if kind == "iterations":
            reader.propose(
                None,
                parent_run_id=None,
                evaluation_id=None,
                hypothesis=None,
                expected_outcome=None,
                changes=None,
            )
        elif kind == "acceptance":
            reader.prepare(None, None, None, None)
        elif kind == "scoring":
            reader.draft(None, None)
        else:
            reader.save(None, None, None, None)
    assert snapshot(tmp_path) == {}


def test_missing_confirmed_scoring_reference_does_not_initialize_store(tmp_path):
    reference = {
        "root": str(tmp_path / "missing"),
        "scoring_id": "sc-" + "1" * 32,
        "spec_digest": "digest",
    }
    with pytest.raises(ValueError, match="找不到"):
        load_confirmed(reference, None, read_only=True)
    assert snapshot(tmp_path) == {}
