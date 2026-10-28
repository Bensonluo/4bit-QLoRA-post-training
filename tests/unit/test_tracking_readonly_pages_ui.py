"""Display-only pages must not initialize missing or unsafe MLflow stores."""

import sqlite3
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[2]
PAGES = (
    ("01_Experiments.py", "experiments-history-source", "本地实验"),
    ("04_Registry.py", "registry-source", "本地注册版本"),
)


@pytest.fixture(autouse=True)
def preserve_mlflow_process_state(monkeypatch):
    pytest.importorskip("mlflow")
    from mlflow.tracking._model_registry import utils as registry_utils
    from mlflow.tracking._tracking_service import utils as tracking_utils

    # Both URI setters modify module globals and the process environment.
    monkeypatch.setattr(registry_utils, "_registry_uri", None)
    monkeypatch.setattr(tracking_utils, "_tracking_uri", None)
    monkeypatch.setenv("MLFLOW_REGISTRY_URI", "")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "")


def snapshot(root):
    return {
        str(path.relative_to(root)): path.read_bytes() if path.is_file() else None
        for path in root.rglob("*")
    }


def open_local_page(monkeypatch, uri, filename, key, local):
    from ui import config, runtime

    monkeypatch.setattr(runtime, "DISPLAY_ONLY", True)
    monkeypatch.setattr(config, "MLFLOW_TRACKING_URI", uri)
    page = AppTest.from_file(str(ROOT / "ui/views" / filename), default_timeout=45).run()
    assert not page.exception, [item.message for item in page.exception]
    page.segmented_control(key=key).set_value(local).run()
    assert not page.exception, [item.message for item in page.exception]
    return page


@pytest.mark.parametrize("filename,key,local", PAGES)
@pytest.mark.parametrize("kind", ["missing_file", "sqlite", "missing_registry"])
def test_display_local_view_does_not_create_or_open_unsafe_store(
    monkeypatch, tmp_path, filename, key, local, kind
):
    from ui import queries

    def forbidden(*args, **kwargs):
        pytest.fail("Empty or unsafe stores must not be passed to the MLflow client")

    for name in ("fetch_runs", "fetch_experiments", "fetch_model_versions"):
        monkeypatch.setattr(queries, name, forbidden)
    if kind == "sqlite":
        database = tmp_path / "tracking.sqlite"
        with sqlite3.connect(database) as connection:
            connection.execute("CREATE TABLE evidence (value TEXT)")
            connection.execute("INSERT INTO evidence VALUES ('existing')")
        uri = "sqlite:///" + str(database)
    else:
        root = tmp_path / "mlruns"
        if kind == "missing_registry":
            (root / ".trash").mkdir(parents=True)
        uri = root.as_uri()
    # Existing .trash and no models means an empty registry, not an empty
    # experiments store; don't treat the latter as missing.
    if kind == "missing_registry" and filename == "01_Experiments.py":
        pytest.skip("Existing experiment store is separately covered below")
    before = snapshot(tmp_path)
    page = open_local_page(monkeypatch, uri, filename, key, local)
    text = "\n".join(item.value for item in [*page.info, *page.warning, *page.error])
    if kind == "sqlite":
        assert "无法安全浏览" in text
        assert "暂无实验记录" not in text and "尚未登记模型权重" not in text
    elif filename == "01_Experiments.py":
        assert "暂无实验记录" in text
    else:
        assert "尚未登记模型权重" in text
    assert page.segmented_control(key=key).value == local
    assert snapshot(tmp_path) == before


@pytest.mark.parametrize("filename,key,local", PAGES)
def test_display_can_read_existing_real_file_store_without_mutating_it(
    monkeypatch, tmp_path, filename, key, local
):
    pytest.importorskip("mlflow")
    from mlflow.tracking import MlflowClient

    root = tmp_path / "mlruns"
    uri = root.as_uri()
    client = MlflowClient(tracking_uri=uri)
    experiment_id = client.create_experiment("真实只读实验")
    run = client.create_run(experiment_id)
    client.create_registered_model("real-file-model")
    client.create_model_version("real-file-model", "file:///tmp/example", run.info.run_id)
    before = snapshot(tmp_path)
    page = open_local_page(monkeypatch, uri, filename, key, local)
    if filename == "01_Experiments.py":
        assert any(
            "运行 ID" in table.value.columns and run.info.run_id in table.value["运行 ID"].tolist()
            for table in page.dataframe
        )
    else:
        assert any(item.label == "版本数" and item.value == "1" for item in page.metric)
    assert snapshot(tmp_path) == before


@pytest.mark.parametrize("filename,key,local", PAGES)
def test_safe_preflight_does_not_change_local_mode(monkeypatch, tmp_path, filename, key, local):
    import pandas as pd

    from ui import config, queries, runtime

    calls = []
    monkeypatch.setattr(runtime, "DISPLAY_ONLY", False)
    monkeypatch.setattr(config, "MLFLOW_TRACKING_URI", (tmp_path / "missing").as_uri())

    def empty_runs(*args, **kwargs):
        calls.append("runs")
        return pd.DataFrame()

    def empty_experiments(*args, **kwargs):
        calls.append("experiments")
        return []

    def empty_versions(*args, **kwargs):
        calls.append("versions")
        return pd.DataFrame()

    monkeypatch.setattr(queries, "fetch_runs", empty_runs)
    monkeypatch.setattr(queries, "fetch_experiments", empty_experiments)
    monkeypatch.setattr(queries, "fetch_model_versions", empty_versions)
    page = AppTest.from_file(str(ROOT / "ui/views" / filename), default_timeout=45).run()
    assert not page.exception, [item.message for item in page.exception]
    assert page.segmented_control(key=key).value == local
    assert calls == (["experiments", "runs"] if filename == "01_Experiments.py" else ["versions"])


@pytest.mark.parametrize("filename,key,local", PAGES)
def test_display_query_failure_offers_public_archive_without_false_empty_state(
    monkeypatch, tmp_path, filename, key, local
):
    from ui import queries

    root = tmp_path / "mlruns"
    (root / ".trash").mkdir(parents=True)
    (root / "models").mkdir()

    def broken(*args, **kwargs):
        raise RuntimeError("private-path/invalid-record")

    monkeypatch.setattr(queries, "fetch_runs", broken)
    monkeypatch.setattr(queries, "fetch_experiments", broken)
    monkeypatch.setattr(queries, "fetch_model_versions", broken)
    before = snapshot(tmp_path)
    page = open_local_page(monkeypatch, root.as_uri(), filename, key, local)
    messages = "\n".join(item.value for item in [*page.warning, *page.error, *page.info])
    assert "暂不可读" in messages
    assert "private-path" not in messages
    assert "暂无实验记录" not in messages and "尚未登记模型权重" not in messages
    public = "公开历史验证" if filename == "01_Experiments.py" else "公开交付档案"
    page.segmented_control(key=key).set_value(public).run()
    assert not page.exception
    assert any("公开" in item.value or "交付" in item.value for item in page.caption)
    assert snapshot(tmp_path) == before


@pytest.mark.parametrize("source", ["environment", "global"])
def test_independent_sqlite_registry_is_rejected_without_opening_it(monkeypatch, tmp_path, source):
    import mlflow

    from ui import queries

    tracking_root = tmp_path / "tracking"
    (tracking_root / ".trash").mkdir(parents=True)
    database = tmp_path / "registry.sqlite"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE evidence (value TEXT)")
        connection.execute("INSERT INTO evidence VALUES ('existing')")
    registry_uri = "sqlite:///" + str(database)
    if source == "global":
        mlflow.set_registry_uri(registry_uri)
    else:
        monkeypatch.setenv("MLFLOW_REGISTRY_URI", registry_uri)

    def forbidden(*args, **kwargs):
        pytest.fail("Unsafe independent SQLite registry must not be opened")

    monkeypatch.setattr(queries, "fetch_model_versions", forbidden)
    before = snapshot(tmp_path)
    page = open_local_page(
        monkeypatch, tracking_root.as_uri(), "04_Registry.py", "registry-source", "本地注册版本"
    )
    assert any("无法安全浏览" in item.value for item in page.warning)
    assert not any("尚未登记模型权重" in item.value for item in page.info)
    assert snapshot(tmp_path) == before


def test_missing_tracking_does_not_hide_existing_independent_registry(monkeypatch, tmp_path):
    import mlflow
    from mlflow.tracking import MlflowClient

    from ui import queries

    setup_root = tmp_path / "setup-tracking"
    registry_root = tmp_path / "registry"
    client = MlflowClient(tracking_uri=setup_root.as_uri(), registry_uri=registry_root.as_uri())
    client.create_registered_model("existing-independent-model")
    client.create_model_version("existing-independent-model", "file:///tmp/example")
    mlflow.set_registry_uri(registry_root.as_uri())

    def forbidden(*args, **kwargs):
        pytest.fail("Absent tracking store must not be initialized by the client")

    monkeypatch.setattr(queries, "fetch_model_versions", forbidden)
    tracking_root = tmp_path / "missing-tracking"
    before = snapshot(tmp_path)
    page = open_local_page(
        monkeypatch, tracking_root.as_uri(), "04_Registry.py", "registry-source", "本地注册版本"
    )
    assert any("实验存储不存在" in item.value for item in page.warning)
    assert not any("尚未登记模型权重" in item.value for item in page.info)
    assert snapshot(tmp_path) == before


def test_existing_separate_file_registry_is_readable_without_writes(monkeypatch, tmp_path):
    import mlflow
    from mlflow.tracking import MlflowClient

    from ui.queries import fetch_model_versions

    tracking_root = tmp_path / "tracking"
    registry_root = tmp_path / "registry"
    client = MlflowClient(tracking_uri=tracking_root.as_uri(), registry_uri=registry_root.as_uri())
    client.create_registered_model("independent-model")
    client.create_model_version("independent-model", "file:///tmp/example")
    mlflow.set_registry_uri(registry_root.as_uri())
    fetch_model_versions.clear()
    before = snapshot(tmp_path)
    page = open_local_page(
        monkeypatch, tracking_root.as_uri(), "04_Registry.py", "registry-source", "本地注册版本"
    )
    assert any(item.label == "版本数" and item.value == "1" for item in page.metric)
    assert any(item.value == "independent-model" for item in page.selectbox)
    assert snapshot(tmp_path) == before
