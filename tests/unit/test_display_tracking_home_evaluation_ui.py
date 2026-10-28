"""Display browsing must not let a real MLflow client initialize local stores."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

pytest.importorskip("mlflow")

ROOT = Path(__file__).resolve().parents[2]


def snapshot(root):
    return {
        str(p.relative_to(root)): p.read_bytes() if p.is_file() else None for p in root.rglob("*")
    }


def configure(monkeypatch, root, uri):
    from ui import config, runtime
    from ui.components import workspace

    monkeypatch.setattr(config, "PROJECT_ROOT", root)
    monkeypatch.setattr(config, "MLFLOW_TRACKING_URI", uri)
    monkeypatch.setattr(runtime, "DISPLAY_ONLY", True)
    monkeypatch.setattr(workspace, "SHOWCASE_ROOT", ROOT)


def test_display_home_missing_real_file_store_does_not_create_anything(monkeypatch, tmp_path):
    root = tmp_path / "workspace"
    configure(monkeypatch, root, (root / "mlruns").as_uri())
    page = AppTest.from_file(str(ROOT / "ui/app.py"), default_timeout=45).run()
    assert not page.exception
    assert any("暂无实验记录" in item.value for item in page.caption)
    assert not root.exists()


def test_display_home_preserves_existing_real_run_and_store(monkeypatch, tmp_path):
    import mlflow

    root = tmp_path / "workspace"
    uri = (root / "mlruns").as_uri()
    previous_uri = mlflow.get_tracking_uri()
    try:
        mlflow.set_tracking_uri(uri)
        with mlflow.start_run(run_name="真实本地实验") as run:
            mlflow.log_metric("loss", 0.25)
            run_id = run.info.run_id
        before = snapshot(root)
        configure(monkeypatch, root, uri)
        page = AppTest.from_file(str(ROOT / "ui/app.py"), default_timeout=45).run()
        assert not page.exception, [item.message for item in page.exception]
        table = next(item.value for item in page.dataframe if "运行 ID" in item.value.columns)
        assert table.iloc[0]["记录"] == "真实本地实验"
        assert run_id.startswith(table.iloc[0]["运行 ID"])
        assert table.iloc[0]["训练损失"] == "0.25"
        assert snapshot(root) == before
    finally:
        mlflow.set_tracking_uri(previous_uri)


def test_display_evaluation_rejects_database_without_claiming_empty(monkeypatch, tmp_path):
    from ui.components import domain_adapters

    root = tmp_path / "workspace"
    uri = f"sqlite:///{root / 'mlflow.db'}"
    configure(monkeypatch, root, uri)
    monkeypatch.setattr(domain_adapters, "list_domains", lambda: ["demo"])
    monkeypatch.setattr(domain_adapters, "load_eval_data", lambda _: [])
    page = AppTest.from_file(str(ROOT / "ui/views/02_Evaluation.py"), default_timeout=45).run()
    assert not page.exception
    page.segmented_control(key="evaluation-history-source").set_value("本地领域评测").run()
    assert not page.exception, [item.message for item in page.exception]
    assert any("暂无法" in item.value and "安全读取" in item.value for item in page.warning)
    assert any("无法确认" in item.value for item in page.info)
    assert not root.exists()


def test_display_home_leaves_active_sqlite_wal_untouched_and_reports_unknown(monkeypatch, tmp_path):
    import sqlite3

    root = tmp_path / "workspace"
    root.mkdir()
    database = root / "tracking.sqlite"
    writer = sqlite3.connect(database)
    try:
        writer.execute("PRAGMA journal_mode=WAL")
        writer.execute("CREATE TABLE evidence (value TEXT)")
        writer.execute("INSERT INTO evidence VALUES ('uncheckpointed')")
        writer.commit()
        before = snapshot(root)
        assert any(name.endswith("-wal") for name in before)
        configure(monkeypatch, root, f"sqlite:///{database}")
        page = AppTest.from_file(str(ROOT / "ui/app.py"), default_timeout=45).run()
        assert not page.exception
        assert any("实验记录暂不可读" in item.value for item in page.warning)
        assert not any("暂无实验记录" in item.value for item in page.caption)
        assert snapshot(root) == before
    finally:
        writer.close()
