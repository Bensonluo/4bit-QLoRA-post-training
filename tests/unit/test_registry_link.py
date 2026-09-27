"""run→Registry 反查:注册状态、未注册指引、查询失败如实呈现。"""

from types import SimpleNamespace

import pytest

from src.workbench import registry_link


def _client(monkeypatch, versions=None, runs_found=1):
    client = SimpleNamespace()
    experiment = SimpleNamespace(experiment_id="exp-1")
    client.get_experiment_by_name = lambda name: experiment
    run = SimpleNamespace(info=SimpleNamespace(run_id="mfr-123"))
    client.search_runs = lambda ids, filter_string=None, max_results=None: [run] * runs_found
    client.search_model_versions = lambda query: versions or []
    module = pytest.importorskip("mlflow.tracking")
    monkeypatch.setattr(module, "MlflowClient", lambda tracking_uri=None: client)


def test_registered_version_is_reported(monkeypatch):
    _client(
        monkeypatch,
        versions=[
            SimpleNamespace(name="Qwen-工单", version=3, aliases=["champion"], current_stage="Production")
        ],
    )
    result = registry_link.run_registration_status("wb-1", "sqlite:///x.db")
    assert result["status"] == "registered"
    assert result["versions"][0]["name"] == "Qwen-工单"
    assert result["versions"][0]["aliases"] == ["champion"]


def test_not_registered_gives_exact_command(monkeypatch):
    _client(monkeypatch, versions=[])
    result = registry_link.run_registration_status("wb-2", "sqlite:///x.db")
    assert result["status"] == "not_registered"
    assert "merge_adapter.py" in result["how_to_register"]
    assert "registry_cli.py register" in result["how_to_register"]


def test_tracking_miss_and_failure_are_honest(monkeypatch):
    _client(monkeypatch, runs_found=0)
    assert (
        registry_link.run_registration_status("wb-3", "sqlite:///x.db")["status"]
        == "tracking_not_found"
    )
    module = pytest.importorskip("mlflow.tracking")

    def boom(tracking_uri=None):
        raise RuntimeError("连不上")

    monkeypatch.setattr(module, "MlflowClient", boom)
    assert registry_link.run_registration_status("wb-4", "sqlite:///x.db")["status"] == "lookup_failed"


def _version_client(monkeypatch, run_id="mfr-9", tags=None, params=None, metrics=None):
    client = SimpleNamespace()
    client.get_model_version = lambda name, version: SimpleNamespace(run_id=run_id)
    client.get_run = lambda rid: SimpleNamespace(
        data=SimpleNamespace(
            tags=tags or {}, params=params or {}, metrics=metrics or {}
        )
    )
    module = pytest.importorskip("mlflow.tracking")
    monkeypatch.setattr(module, "MlflowClient", lambda tracking_uri=None: client)


def test_version_lineage_workbench_triangle(monkeypatch):
    _version_client(
        monkeypatch,
        tags={
            "workbench.run_id": "wb-abc",
            "workbench.dataset_version": "data-v7",
            "workbench.config_digest": "cfg-1",
        },
        params={"data.dataset_name": "/path/train.json"},
        metrics={"train_loss": 0.5},
    )
    result = registry_link.version_lineage("模型A", 2, "sqlite:///x.db")
    assert result["status"] == "workbench"
    assert result["workbench_run_id"] == "wb-abc"
    assert result["dataset_version"] == "data-v7"
    assert result["training_dataset"] == "/path/train.json"


def test_version_lineage_external_and_missing(monkeypatch):
    _version_client(monkeypatch, tags={}, params={"model.name": "Qwen/Qwen3-4B"})
    result = registry_link.version_lineage("旧模型", 1, "sqlite:///x.db")
    assert result["status"] == "external"
    assert result["base_model"] == "Qwen/Qwen3-4B"

    client = SimpleNamespace()
    client.get_model_version = lambda name, version: SimpleNamespace(run_id="")
    module = pytest.importorskip("mlflow.tracking")
    monkeypatch.setattr(module, "MlflowClient", lambda tracking_uri=None: client)
    assert registry_link.version_lineage("M", 9, "sqlite:///x.db")["status"] == "no_source_run"
