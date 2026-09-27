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
