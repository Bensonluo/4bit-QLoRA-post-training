"""Display preflight never initializes a tracking store or reads a database."""

import hashlib
import os
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

from ui.tracking_readonly import TrackingStoreUnavailable, inspect_tracking_uri


def snapshot(root: Path) -> dict[str, str | None]:
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        if path.is_file()
        else None
        for path in root.rglob("*")
    }


def mlflow_process(uri: str, script: str, cwd: Path) -> str:
    env = {**os.environ, "PYTHONWARNINGS": "ignore"}
    result = subprocess.run(
        [sys.executable, "-c", script, uri],
        cwd=cwd,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    return result.stdout


@pytest.mark.parametrize("encoded", [True, False])
def test_missing_local_store_stays_missing(tmp_path, encoded):
    root = tmp_path / "not here #100%" / "mlruns"
    uri = root.as_uri() if encoded else str(root)
    before = snapshot(tmp_path)
    assert inspect_tracking_uri(uri) == "missing"
    assert inspect_tracking_uri(uri, registry=True) == "missing"
    assert snapshot(tmp_path) == before


def test_real_initialized_file_store_and_registry_can_be_read_without_file_changes(tmp_path):
    root = tmp_path / "track #100%" / "mlruns"
    uri = root.as_uri()
    setup = (
        "import mlflow,sys; mlflow.set_tracking_uri(sys.argv[1]); "
        "eid=mlflow.create_experiment('proof'); "
        "mlflow.start_run(experiment_id=eid); mlflow.log_metric('loss',0.1); "
        "mlflow.end_run(); mlflow.search_registered_models()"
    )
    mlflow_process(uri, setup, tmp_path)
    before = snapshot(tmp_path)
    assert inspect_tracking_uri(uri) == "ready"
    assert inspect_tracking_uri(uri, registry=True) == "ready"
    query = (
        "import mlflow,sys; mlflow.set_tracking_uri(sys.argv[1]); "
        "experiments=mlflow.search_experiments(); "
        "runs=mlflow.search_runs([item.experiment_id for item in experiments]); "
        "models=mlflow.search_registered_models(); print(len(runs), len(models))"
    )
    assert mlflow_process(uri, query, tmp_path).strip() == "1 0"
    assert snapshot(tmp_path) == before


def test_file_store_without_registry_is_reported_without_initializing_it(tmp_path):
    root = tmp_path / "mlruns"
    uri = root.as_uri()
    setup = "import mlflow,sys; mlflow.set_tracking_uri(sys.argv[1]); mlflow.search_experiments()"
    mlflow_process(uri, setup, tmp_path)
    assert not (root / "models").exists()
    before = snapshot(tmp_path)
    assert inspect_tracking_uri(uri) == "ready"
    assert inspect_tracking_uri(uri, registry=True) == "missing_registry"
    assert snapshot(tmp_path) == before


def test_independent_real_file_registry_does_not_require_a_tracking_trash_folder(tmp_path):
    root = tmp_path / "registry #100%"
    uri = root.as_uri()
    query = (
        "import sys; from mlflow.store.model_registry.file_store import FileStore; "
        "store=FileStore(sys.argv[1]); print(len(store.search_registered_models(max_results=100)))"
    )
    assert mlflow_process(uri, query, tmp_path).strip() == "0"
    assert (root / "models").is_dir()
    assert not (root / ".trash").exists()
    before = snapshot(tmp_path)
    assert inspect_tracking_uri(uri, registry=True) == "ready"
    with pytest.raises(TrackingStoreUnavailable):
        inspect_tracking_uri(uri)
    assert mlflow_process(uri, query, tmp_path).strip() == "0"
    assert snapshot(tmp_path) == before


def test_independent_registry_without_models_is_missing_without_initialization(tmp_path):
    root = tmp_path / "registry"
    root.mkdir()
    before = snapshot(tmp_path)
    assert inspect_tracking_uri(root.as_uri(), registry=True) == "missing_registry"
    assert snapshot(tmp_path) == before


@pytest.mark.parametrize("kind", ["file", "broken_symlink"])
def test_invalid_models_path_is_rejected_without_initialization(tmp_path, kind):
    root = tmp_path / "registry"
    root.mkdir()
    if kind == "file":
        (root / "models").write_text("not a directory")
    else:
        (root / "models").symlink_to(tmp_path / "absent", target_is_directory=True)
    before = snapshot(tmp_path)
    with pytest.raises(TrackingStoreUnavailable):
        inspect_tracking_uri(root.as_uri(), registry=True)
    assert snapshot(tmp_path) == before


@pytest.mark.parametrize("kind", ["empty_root", "trash_file", "root_file", "broken_symlink"])
def test_incomplete_or_invalid_existing_file_store_does_not_initialize(tmp_path, kind):
    root = tmp_path / "mlruns"
    if kind == "root_file":
        root.write_bytes(b"not a directory")
    elif kind == "broken_symlink":
        root.symlink_to(tmp_path / "absent", target_is_directory=True)
    else:
        root.mkdir()
        if kind == "trash_file":
            (root / ".trash").write_text("not a directory")
    before = snapshot(tmp_path)
    with pytest.raises(TrackingStoreUnavailable) as error:
        inspect_tracking_uri(root.as_uri())
    assert str(tmp_path) not in str(error.value)
    assert snapshot(tmp_path) == before


@pytest.mark.parametrize(
    "uri",
    [
        "sqlite:////tmp/user-data.db",
        "postgresql://user:password@example.com/database",
        "https://example.com/",
        "file://elsewhere/private/mlruns",
        "file:///tmp/mlruns?secret=1",
        "relative/mlruns",
        "",
    ],
)
def test_unsupported_uri_fails_closed_without_exposing_sensitive_values(uri):
    with pytest.raises(TrackingStoreUnavailable) as error:
        inspect_tracking_uri(uri)
    assert uri not in str(error.value) if uri else True
    assert "password" not in str(error.value) and "secret" not in str(error.value)


def test_encoded_null_in_local_uri_is_reported_without_disclosing_path_or_writing(tmp_path):
    uri = tmp_path.as_uri() + "/sensitive%00value"
    before = snapshot(tmp_path)
    with pytest.raises(TrackingStoreUnavailable) as error:
        inspect_tracking_uri(uri)
    assert str(tmp_path) not in str(error.value)
    assert "sensitive" not in str(error.value)
    assert error.value.__suppress_context__
    assert snapshot(tmp_path) == before


def test_sqlite_wal_is_rejected_without_changing_any_sidecars(tmp_path):
    database = tmp_path / "tracking.db"
    writer = sqlite3.connect(database)
    try:
        writer.execute("PRAGMA journal_mode=WAL")
        writer.execute("CREATE TABLE experiment (name TEXT)")
        writer.commit()
        writer.execute("INSERT INTO experiment VALUES ('uncheckpointed')")
        writer.commit()
        before = snapshot(tmp_path)
        assert (tmp_path / "tracking.db-wal").stat().st_size > 0
        with pytest.raises(TrackingStoreUnavailable):
            inspect_tracking_uri(f"sqlite:///{database}")
        with pytest.raises(TrackingStoreUnavailable):
            inspect_tracking_uri(f"sqlite:///{database}", registry=True)
        assert snapshot(tmp_path) == before
    finally:
        writer.close()
