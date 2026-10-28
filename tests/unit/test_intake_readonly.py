"""Display readers must neither create stores nor bypass SQLite's read-only mode."""

from __future__ import annotations

import inspect
import sqlite3
from pathlib import Path

import pytest

from src.workbench.funnel_report import collect_funnel
from src.workbench.intake_service import IntakeService


def snapshot(root: Path) -> dict[str, bytes]:
    return {
        str(path.relative_to(root)): path.read_bytes() for path in root.rglob("*") if path.is_file()
    }


def test_missing_readonly_store_is_empty_without_creating_directories(tmp_path):
    root = tmp_path / "missing" / "intake"
    service = IntakeService(root, read_only=True)
    assert service.database == root / "intake.sqlite"
    assert service.list_sessions() == []
    assert service.list_sessions_tolerant() == ([], 0)
    with pytest.raises(ValueError, match="找不到"):
        service.load("a" * 32)
    assert not root.parent.exists()


def test_existing_readonly_store_preserves_model_and_files(tmp_path):
    writable = IntakeService(tmp_path / "intake")
    session = writable.create("判断类别", "sample.csv", "输入,类别\n杯子破损,质量\n".encode())
    before = snapshot(tmp_path)
    service = IntakeService(writable.root, read_only=True)
    assert service.load(session.session_id) == session
    assert service.list_sessions() == [session]
    assert service.list_sessions_tolerant() == ([session], 0)
    assert snapshot(tmp_path) == before
    with (
        service._connect() as connection,
        pytest.raises(sqlite3.OperationalError, match="readonly"),
    ):
        connection.execute("DELETE FROM sessions")
    assert snapshot(tmp_path) == before
    assert writable.load(session.session_id) == session


@pytest.mark.parametrize("method", ["list_sessions", "list_sessions_tolerant"])
def test_corrupt_database_is_not_an_empty_store(tmp_path, method):
    database = tmp_path / "intake.sqlite"
    database.write_bytes(b"not a sqlite database")
    before = snapshot(tmp_path)
    with pytest.raises(sqlite3.DatabaseError):
        getattr(IntakeService(tmp_path, read_only=True), method)()
    assert snapshot(tmp_path) == before


def test_readonly_constructor_never_repairs_existing_schema(tmp_path):
    database = tmp_path / "intake.sqlite"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE unrelated (value TEXT)")
    before = snapshot(tmp_path)
    service = IntakeService(tmp_path, read_only=True)
    with pytest.raises(sqlite3.OperationalError, match="no such table"):
        service.list_sessions()
    assert snapshot(tmp_path) == before


def test_readonly_lists_still_validate_each_session(tmp_path):
    writable = IntakeService(tmp_path)
    session = writable.create("判断类别", "sample.csv", b"input,target\na,b\n")
    with sqlite3.connect(writable.database) as connection:
        connection.execute("INSERT INTO sessions VALUES (?, ?, ?)", ("invalid", 0, "{}"))
    before = snapshot(tmp_path)
    service = IntakeService(tmp_path, read_only=True)
    with pytest.raises(ValueError):
        service.list_sessions()
    assert service.list_sessions_tolerant() == ([session], 1)
    assert snapshot(tmp_path) == before


@pytest.mark.parametrize(
    "method",
    [
        "_save",
        "create",
        "answer",
        "apply_analysis",
        "analyze",
        "start_contrast_check",
        "submit_contrast_check",
        "confirm",
        "validate_full_data",
        "confirm_full_data",
        "materialize_dataset",
        "start_label_verification",
        "submit_label_verification",
        "add_source",
        "validate_full_sources",
        "preflight_training",
    ],
)
def test_readonly_mutations_fail_before_input_processing_or_external_actions(tmp_path, method):
    root = tmp_path / "missing"
    operation = getattr(IntakeService(root, read_only=True), method)
    # Invalid placeholders deliberately prove the guard precedes parsing, API calls,
    # tokenizer use, source-file writes, and materialization.
    arguments = {
        name: None
        for name, parameter in inspect.signature(operation).parameters.items()
        if parameter.default is inspect.Parameter.empty
    }
    with pytest.raises(PermissionError, match="展示模式"):
        operation(**arguments)
    assert not root.exists()


def test_readonly_funnel_missing_stores_has_no_filesystem_side_effect(tmp_path):
    roots = [
        tmp_path / name
        for name in ("intake", "training", "evaluations", "acceptance", "iterations")
    ]
    report = collect_funnel(*roots, read_only=True)
    assert report["errors"] == []
    assert report["sessions"]["total"] == 0
    assert list(tmp_path.iterdir()) == []


def test_readonly_funnel_isolates_corrupt_intake_database(tmp_path):
    roots = [
        tmp_path / name
        for name in ("intake", "training", "evaluations", "acceptance", "iterations")
    ]
    roots[0].mkdir()
    (roots[0] / "intake.sqlite").write_bytes(b"broken")
    before = snapshot(tmp_path)
    report = collect_funnel(*roots, read_only=True)
    assert report["errors"] == ["数据任务记录"]
    assert snapshot(tmp_path) == before


def test_readonly_task_projection_does_not_create_optional_stores(tmp_path):
    from src.workbench.task_spec_projection import collect_task_spec

    intake = IntakeService(tmp_path / "intake")
    session = intake.create("判断类别", "sample.csv", b"input,target\na,b\n")
    before = snapshot(tmp_path)
    projection = collect_task_spec(
        session.session_id,
        intake.root,
        tmp_path / "scoring",
        tmp_path / "acceptance",
        tmp_path / "evaluations",
        tmp_path / "iterations",
        tmp_path / "training",
        read_only=True,
    )
    assert projection["goal"]["goal"] == session.goal
    assert projection["scoring"] == {"confirmed": [], "draft_count": 0}
    assert projection["acceptance"] == {"state": "未冻结", "records": []}
    assert projection["latest_iteration"] is None
    assert snapshot(tmp_path) == before
    assert [path.name for path in tmp_path.iterdir()] == ["intake"]


def test_readonly_task_projection_does_not_mask_corrupt_optional_store(tmp_path):
    from src.workbench.task_spec_projection import collect_task_spec

    intake = IntakeService(tmp_path / "intake")
    session = intake.create("判断类别", "sample.csv", b"input,target\na,b\n")
    scoring = tmp_path / "scoring"
    scoring.mkdir()
    (scoring / "scoring.sqlite").write_bytes(b"broken")
    before = snapshot(tmp_path)
    with pytest.raises(sqlite3.DatabaseError):
        collect_task_spec(
            session.session_id,
            intake.root,
            scoring,
            tmp_path / "acceptance",
            tmp_path / "evaluations",
            tmp_path / "iterations",
            tmp_path / "training",
            read_only=True,
        )
    assert snapshot(tmp_path) == before
