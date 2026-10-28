"""The display homepage names the actual task that its primary action opens."""

from streamlit.testing.v1 import AppTest

from src.workbench.intake_service import IntakeService
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


def test_display_home_names_and_opens_latest_real_task_without_writes(monkeypatch, tmp_path):
    root = tmp_path / "workspace"
    service = IntakeService(root / INTAKE)
    first = service.create("之前的业务任务", "first.csv", CSV)
    latest = service.create("最近创建的真实售后任务", "second.csv", CSV)
    assert first.session_id != latest.session_id
    before = snapshot(root)
    configure(monkeypatch, root)
    mkdirs, _, statements = observe_storage(monkeypatch, root)
    page = AppTest.from_file(str(ROOT / "ui/app.py"), default_timeout=45).run()
    clean(page)
    assert any(
        text.value == f"最近创建：{latest.goal} · {latest.session_id[:8]}" for text in page.text
    )
    entry = next(item for item in page.button if item.label == "浏览最近创建的任务")
    entry.click().run()
    clean(page)
    assert page.session_state["intake_id"] == latest.session_id
    assert any(item.value == latest.goal for item in page.title)
    unchanged(root, before)
    assert not mkdirs
    assert not any(
        sql.lstrip().split()[0].upper()
        in {"CREATE", "ALTER", "DROP", "INSERT", "UPDATE", "DELETE", "REPLACE"}
        for sql in statements
    )


def test_display_home_without_tasks_opens_goal_and_data_without_writes(monkeypatch, tmp_path):
    root = tmp_path / "empty-workspace"
    before = snapshot(root)
    configure(monkeypatch, root)
    mkdirs, connections, _ = observe_storage(monkeypatch, root)
    page = AppTest.from_file(str(ROOT / "ui/app.py"), default_timeout=45).run()
    clean(page)
    assert not any(text.value.startswith("最近创建：") for text in page.text)
    next(item for item in page.button if item.label == "查看目标与数据").click().run()
    clean(page)
    assert any("业务目标" in item.value for item in page.title)
    unchanged(root, before)
    assert not root.exists()
    assert not mkdirs and not connections
