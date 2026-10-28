"""Display navigation reads real task stores without initializing or mutating them."""

import sqlite3
from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from src.workbench.intake_service import IntakeService

ROOT = Path(__file__).resolve().parents[2]
INTAKE = "outputs/workbench/intake"
CSV = "编号,描述,类别\n1,杯子破损,质量\n2,物流延误,物流\n".encode()


def snapshot(root):
    return {
        str(path.relative_to(root)): path.read_bytes() if path.is_file() else None
        for path in root.rglob("*")
    }


def unchanged(root, before):
    after = snapshot(root)
    changes = [
        name
        for name in sorted(set(before) | set(after))
        if name not in before or name not in after or before[name] != after[name]
    ]
    assert not changes, f"Display browsing changed local paths: {changes}"


def configure(monkeypatch, root, *, display=True):
    from ui import config, queries, runtime
    from ui.components import agent_settings, workspace

    monkeypatch.setattr(config, "PROJECT_ROOT", root)
    monkeypatch.setattr(config, "MLFLOW_TRACKING_URI", f"file://{root / 'mlruns'}")
    monkeypatch.setattr(workspace, "SHOWCASE_ROOT", ROOT)
    monkeypatch.setattr(runtime, "DISPLAY_ONLY", display)
    # Imported once by other AppTests; keep the real settings widgets in this mode.
    monkeypatch.setattr(agent_settings, "DISPLAY_ONLY", display)
    monkeypatch.setattr(queries, "fetch_runs", lambda *args, **kwargs: pd.DataFrame())
    monkeypatch.setattr(queries, "fetch_experiments", lambda *args, **kwargs: [])
    for name in ("PROVIDER", "BASE_URL", "MODEL", "API_KEY"):
        monkeypatch.delenv(f"TUNESMITH_AGENT_{name}", raising=False)


def clean(page):
    assert not page.exception, [item.message for item in page.exception]


def direct():
    return AppTest.from_file(str(ROOT / "ui/views/07_Data_Intake.py"), default_timeout=30).run()


def button(page, label):
    return next(item for item in page.button if item.label == label)


def observe_storage(monkeypatch, root):
    """Observe actual calls, allowing reads of pre-existing SQLite stores."""
    mkdirs, connections, statements = [], [], []
    real_mkdir, real_connect = Path.mkdir, sqlite3.connect

    def mkdir(path, *args, **kwargs):
        if path.is_relative_to(root):
            mkdirs.append(str(path.relative_to(root)))
        return real_mkdir(path, *args, **kwargs)

    def connect(database, *args, **kwargs):
        identity = str(database)
        if str(root) in identity:
            connections.append(identity)
        connection = real_connect(database, *args, **kwargs)
        if str(root) in identity:
            connection.set_trace_callback(statements.append)
        return connection

    monkeypatch.setattr(Path, "mkdir", mkdir)
    monkeypatch.setattr(sqlite3, "connect", connect)
    return mkdirs, connections, statements


@pytest.mark.parametrize("entry", ["direct", "history"])
def test_empty_display_entry_does_not_initialize_storage(monkeypatch, tmp_path, entry):
    root = tmp_path / "empty-workspace"
    configure(monkeypatch, root)
    before = snapshot(root)
    mkdirs, connections, _ = observe_storage(monkeypatch, root)
    if entry == "direct":
        page = direct()
    else:
        from ui.components import workspace

        # Exercise the real history-page empty-source navigation exit.
        monkeypatch.setattr(workspace, "SHOWCASE_ROOT", tmp_path / "missing-assets")
        page = AppTest.from_file(str(ROOT / "ui/app.py"), default_timeout=45).run()
        clean(page)
        page.switch_page("views/08_Showcase.py").run()
        clean(page)
        page.button(key="archive-open-tasks").click().run()
    clean(page)
    assert any("业务目标" in item.value for item in page.title)
    unchanged(root, before)
    assert not root.exists()
    assert not mkdirs, f"Display navigation attempted mkdir: {mkdirs}"
    assert not connections, f"Empty display workspace opened SQLite: {connections}"


def test_existing_tasks_can_be_browsed_and_switched_without_store_changes(monkeypatch, tmp_path):
    root = tmp_path / "workspace"
    service = IntakeService(root / INTAKE)
    first = service.create("按首次描述分类", "first.csv", CSV)
    second = service.create("浏览另一个真实任务", "second.csv", CSV)
    before = snapshot(root)
    existing_databases = {str(path) for path in root.rglob("*.sqlite")}
    configure(monkeypatch, root)
    mkdirs, connections, statements = observe_storage(monkeypatch, root)
    page = direct()
    clean(page)
    for session in (first, second, first):
        page.selectbox(key="intake_select").select(session.session_id).run()
        clean(page)
        assert any(item.value == session.goal for item in page.title)
        assert any(item.label == "本地数据事实与原始行" for item in page.expander)
    unchanged(root, before)
    assert not mkdirs, f"Browsing attempted mkdir: {mkdirs}"
    assert connections and all(
        any(database in identity for database in existing_databases) for identity in connections
    ), connections
    writes = [
        sql
        for sql in statements
        if sql.lstrip().split()[0].upper()
        in {"CREATE", "ALTER", "DROP", "INSERT", "UPDATE", "DELETE", "REPLACE"}
    ]
    assert not writes, f"Read-only browsing executed write SQL: {writes}"


def test_display_home_lists_existing_task_without_ddl_or_filesystem_changes(monkeypatch, tmp_path):
    root = tmp_path / "workspace"
    service = IntakeService(root / INTAKE)
    session = service.create("首页已有任务", "source.csv", CSV)
    before = snapshot(root)
    configure(monkeypatch, root)
    mkdirs, connections, statements = observe_storage(monkeypatch, root)
    page = AppTest.from_file(str(ROOT / "ui/app.py"), default_timeout=45).run()
    clean(page)
    assert page.button(key=f"open-{session.session_id}").label == "浏览任务"
    unchanged(root, before)
    assert not mkdirs
    assert connections
    assert not any(
        sql.lstrip().split()[0].upper()
        in {"CREATE", "ALTER", "DROP", "INSERT", "UPDATE", "DELETE", "REPLACE"}
        for sql in statements
    )


def test_analyzed_task_keeps_its_real_preview_without_initializing_auxiliary_stores(
    monkeypatch, tmp_path
):
    from src.workbench.baseline_analysis import propose_baseline_analysis

    root = tmp_path / "workspace"
    service = IntakeService(root / INTAKE)
    session = service.create("按首次描述分类", "first.csv", CSV)
    session = service.apply_analysis(
        session, propose_baseline_analysis(session, target_column="类别"), model="baseline"
    )
    before = snapshot(root)
    configure(monkeypatch, root)
    page = direct()
    clean(page)
    assert any(item.value == session.goal for item in page.title)
    assert any(item.label == "真实转换预览" for item in page.expander)
    assert any(item.value == "模型实际输入" for item in page.caption)
    assert any(item.value == "监督答案" for item in page.caption)
    unchanged(root, before)


def test_materialized_task_renders_training_guidance_without_creating_stores(monkeypatch, tmp_path):
    from src.workbench import local_models, training_preflight
    from tests.unit.test_full_data import FULL, approved

    root = tmp_path / "workspace"
    service = IntakeService(root / INTAKE)
    session = approved(service)
    session = service.validate_full_data(session.session_id, session.revision, "full.csv", FULL)
    session = service.confirm_full_data(session.session_id, session.revision)
    session = service.materialize_dataset(session.session_id, session.revision)
    before = snapshot(root)
    configure(monkeypatch, root)
    monkeypatch.setattr(local_models, "discover_local_models", lambda: [])

    def no_model(*args, **kwargs):
        pytest.fail("Browsing the materialized task must not load a tokenizer.")

    monkeypatch.setattr(training_preflight, "load_local_tokenizer", no_model)
    page = direct()
    clean(page)
    assert any(item.value == session.goal for item in page.title)
    assert any("任务规约投影" in item.label for item in page.expander)
    assert any("训练" in item.value for item in page.subheader)
    unchanged(root, before)


def test_active_wal_task_store_shows_reason_without_touching_database(monkeypatch, tmp_path):
    root = tmp_path / "workspace"
    service = IntakeService(root / INTAKE)
    session = service.create("尚未检查点的任务", "source.csv", CSV)
    writer = sqlite3.connect(service.database)
    try:
        writer.execute("PRAGMA journal_mode=WAL")
        writer.execute(
            "UPDATE sessions SET revision = revision + 1 WHERE id = ?", (session.session_id,)
        )
        writer.commit()
        before = snapshot(root)
        assert any(name.endswith("intake.sqlite-wal") for name in before)
        configure(monkeypatch, root)
        page = direct()
        clean(page)
        assert any("WAL" in item.value and "本地工作台" in item.value for item in page.error)
        assert any(item.label == "浏览公开历史档案" for item in page.button)
        assert not any(item.value == session.goal for item in page.title)
        unchanged(root, before)
    finally:
        writer.close()


def test_forged_disabled_actions_cannot_write_or_execute_callbacks(monkeypatch, tmp_path):
    root = tmp_path / "workspace"
    service = IntakeService(root / INTAKE)
    session = service.create("只读目标", "source.csv", CSV)
    configure(monkeypatch, root)
    page = direct()
    clean(page)
    baseline = snapshot(root)
    calls = []

    def forbidden(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("A disabled action reached the intake mutation API.")

    monkeypatch.setattr(IntakeService, "create", forbidden)
    monkeypatch.setattr(IntakeService, "analyze", forbidden)
    new = button(page, "新建数据任务")
    assert new.disabled
    new.click().run()  # AppTest can forge a client click despite disabled=True.
    clean(page)
    assert page.session_state["intake_id"] == session.session_id
    analyze = button(page, "联合分析目标与数据")
    assert analyze.disabled
    analyze.click().run()
    clean(page)
    assert not calls
    unchanged(root, baseline)


@pytest.mark.parametrize("damage", ["snapshot", "database"])
def test_corrupt_display_store_is_reported_without_repair_or_false_empty_state(
    monkeypatch, tmp_path, damage
):
    root = tmp_path / "workspace"
    service = IntakeService(root / INTAKE)
    session = service.create("损坏前已有的任务", "source.csv", CSV)
    if damage == "snapshot":
        with sqlite3.connect(service.database) as connection:
            connection.execute(
                "UPDATE sessions SET snapshot = ? WHERE id = ?", ("{broken", session.session_id)
            )
    else:
        service.database.write_bytes(b"not a SQLite database")
    before = snapshot(root)
    configure(monkeypatch, root)
    page = direct()
    clean(page)
    messages = "\n".join(item.value for item in [*page.warning, *page.error])
    assert any(word in messages for word in ("损坏", "读取失败", "不可读", "无法读取")), messages
    assert "暂无" not in messages and "没有任务" not in messages
    unchanged(root, before)


def test_local_mode_retains_real_example_creation(monkeypatch, tmp_path):
    root = tmp_path / "local-workspace"
    configure(monkeypatch, root, display=False)
    page = AppTest.from_file(str(ROOT / "ui/app.py"), default_timeout=45).run()
    clean(page)
    create = page.button(key="showcase-create-classification")
    assert not create.disabled
    create.click().run()
    clean(page)
    identity = page.session_state["intake_id"]
    sessions = IntakeService(root / INTAKE).list_sessions()
    assert [session.session_id for session in sessions] == [identity]
    assert sessions[0].analysis is None and sessions[0].source.rows


@pytest.mark.parametrize("damage", ["database", "wal"])
def test_unreadable_scoring_store_is_local_error_while_ready_task_remains_browsable(
    monkeypatch, tmp_path, damage
):
    from tests.unit.test_business_evaluation import task

    root = tmp_path / "workspace"
    # Create and confirm the real task/dataset before entering display mode.
    # This fixture materializes small CSV partitions; it never loads a model.
    _, session = task.__wrapped__(root / "outputs/workbench")
    path = root / "outputs/workbench/business-scoring/scoring.sqlite"
    path.parent.mkdir()
    writer = None
    if damage == "database":
        path.write_bytes(b"not a SQLite database")
    else:
        writer = sqlite3.connect(path)
        writer.execute("PRAGMA journal_mode=WAL")
        writer.execute("CREATE TABLE scoring_specs (id TEXT PRIMARY KEY, snapshot TEXT)")
        writer.commit()
        writer.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        writer.execute("INSERT INTO scoring_specs VALUES ('uncheckpointed', '{}')")
        writer.commit()
        assert Path(str(path) + "-wal").stat().st_size > 0
    before = snapshot(root)
    configure(monkeypatch, root)
    try:
        page = direct()
        clean(page)
        assert any(item.value == session.goal for item in page.title)
        assert any(item.value == "训练前检查" for item in page.subheader)
        assert any(item.value == "自定义业务评分（可选）" for item in page.subheader)
        messages = "\n".join(item.value for item in [*page.warning, *page.error])
        assert "评分" in messages and any(
            phrase in messages for phrase in ("读取失败", "不可读", "无法读取", "暂不可用")
        ), messages
        assert "暂无评分" not in messages
        unchanged(root, before)
    finally:
        if writer is not None:
            writer.close()
