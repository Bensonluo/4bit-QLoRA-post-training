"""SQLite row identity and revision must bind to the exact task snapshot."""

import json
import sqlite3

import pytest
from streamlit.testing.v1 import AppTest

from src.workbench.intake_service import IntakeService
from tests.unit.test_business_evaluation import task
from tests.unit.test_intake_display_readonly_ui import (
    CSV,
    INTAKE,
    ROOT,
    clean,
    configure,
    observe_storage,
    snapshot,
    unchanged,
)


@pytest.mark.parametrize("damage", ["other-task", "revision", "invalid-id"])
def test_current_snapshot_identity_fails_closed_and_preserves_healthy_task(tmp_path, damage):
    writer = IntakeService(tmp_path / "intake")
    first = writer.create("任务甲", "a.csv", CSV)
    second = writer.create("任务乙", "b.csv", CSV)
    broken_id = first.session_id
    with sqlite3.connect(writer.database) as connection:
        if damage == "other-task":
            other = connection.execute(
                "SELECT snapshot FROM sessions WHERE id=?", (second.session_id,)
            ).fetchone()[0]
            connection.execute("UPDATE sessions SET snapshot=? WHERE id=?", (other, broken_id))
        elif damage == "revision":
            connection.execute("UPDATE sessions SET revision=1 WHERE id=?", (broken_id,))
        else:
            payload = json.loads(
                connection.execute(
                    "SELECT snapshot FROM sessions WHERE id=?", (broken_id,)
                ).fetchone()[0]
            )
            payload["session_id"] = "invalid-task-id"
            connection.execute(
                "UPDATE sessions SET id=?, snapshot=? WHERE id=?",
                ("invalid-task-id", json.dumps(payload, ensure_ascii=False), broken_id),
            )
    before = snapshot(tmp_path)
    service = IntakeService(writer.root, read_only=True)
    sessions, corrupt = service.list_sessions_tolerant()
    assert [item.session_id for item in sessions] == [second.session_id]
    assert corrupt == 1
    with pytest.raises(ValueError):
        service.list_sessions()
    if damage != "invalid-id":
        with pytest.raises(ValueError, match="身份或版本"):
            service.load(broken_id)
    assert service.load(second.session_id).goal == "任务乙"
    unchanged(tmp_path, before)


def test_legacy_zero_revision_defaults_to_matching_stored_version(tmp_path):
    writer = IntakeService(tmp_path / "intake")
    session = writer.create("旧版任务", "source.csv", CSV)
    with sqlite3.connect(writer.database) as connection:
        payload = json.loads(
            connection.execute(
                "SELECT snapshot FROM sessions WHERE id=?", (session.session_id,)
            ).fetchone()[0]
        )
        del payload["revision"]
        connection.execute(
            "UPDATE sessions SET snapshot=? WHERE id=?", (json.dumps(payload), session.session_id)
        )
    reader = IntakeService(writer.root, read_only=True)
    assert reader.load(session.session_id).revision == 0
    assert reader.list_sessions_tolerant()[1] == 0


@pytest.mark.parametrize("damage", ["identity", "revision"])
def test_historical_dataset_snapshot_cannot_be_relabelled(tmp_path, damage):
    writer, session = task.__wrapped__(tmp_path)
    assert (
        writer.dataset_snapshot(session.session_id, session.dataset.version).session_id
        == session.session_id
    )
    with sqlite3.connect(writer.database) as connection:
        payload = json.loads(
            connection.execute(
                "SELECT snapshot FROM revisions WHERE id=? AND revision=?",
                (session.session_id, session.revision),
            ).fetchone()[0]
        )
        if damage == "identity":
            payload["session_id"] = "f" * 32
        else:
            payload["revision"] += 1
        connection.execute(
            "UPDATE revisions SET snapshot=? WHERE id=? AND revision=?",
            (json.dumps(payload, ensure_ascii=False), session.session_id, session.revision),
        )
    before = snapshot(tmp_path)
    reader = IntakeService(writer.root, read_only=True)
    with pytest.raises(ValueError, match="身份或版本"):
        reader.dataset_snapshot(session.session_id, session.dataset.version)
    unchanged(tmp_path, before)


def test_home_display_isolates_wrong_identity_and_opens_healthy_task_without_writes(
    monkeypatch, tmp_path
):
    root = tmp_path / "workspace"
    writer = IntakeService(root / INTAKE)
    first = writer.create("任务甲", "a.csv", CSV)
    second = writer.create("任务乙", "b.csv", CSV)
    with sqlite3.connect(writer.database) as connection:
        other = connection.execute(
            "SELECT snapshot FROM sessions WHERE id=?", (second.session_id,)
        ).fetchone()[0]
        connection.execute("UPDATE sessions SET snapshot=? WHERE id=?", (other, first.session_id))
    before = snapshot(root)
    configure(monkeypatch, root)
    mkdirs, _, statements = observe_storage(monkeypatch, root)
    page = AppTest.from_file(str(ROOT / "ui/app.py"), default_timeout=45).run()
    clean(page)
    assert any("1 个任务快照" in item.value for item in page.warning)
    assert page.button(key=f"open-{second.session_id}")
    assert not any(item.key == f"open-{first.session_id}" for item in page.button)
    page.button(key=f"open-{second.session_id}").click().run()
    clean(page)
    assert page.session_state["intake_id"] == second.session_id
    assert any(item.value == "任务乙" for item in page.title)
    unchanged(root, before)
    assert not mkdirs
    assert not any(
        sql.lstrip().split()[0].upper()
        in {"CREATE", "ALTER", "DROP", "INSERT", "UPDATE", "DELETE", "REPLACE"}
        for sql in statements
    )
