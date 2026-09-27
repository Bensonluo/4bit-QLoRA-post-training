"""run→Registry 反查:注册状态、未注册指引、查询失败如实呈现。"""

from types import SimpleNamespace

import pytest

from src.workbench import registry_link


def _client(monkeypatch, versions=None):
    client = SimpleNamespace()
    model = SimpleNamespace(name="Qwen-工单")
    client.search_registered_models = lambda max_results=None: [model]
    client.search_model_versions = lambda query: versions or []

    def get_run(run_id):
        if not run_id:
            raise RuntimeError("no run")
        return SimpleNamespace(
            data=SimpleNamespace(tags={"workbench.run_id": "wb-1"}, params={}, metrics={})
        )

    client.get_run = get_run
    module = pytest.importorskip("mlflow.tracking")
    monkeypatch.setattr(module, "MlflowClient", lambda tracking_uri=None: client)


def test_registered_version_is_reported(monkeypatch):
    _client(
        monkeypatch,
        versions=[
            SimpleNamespace(
                name="Qwen-工单",
                version=3,
                aliases=["champion"],
                current_stage="Production",
                run_id="mfr-1",
            )
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


def test_tracking_failure_is_honest(monkeypatch):
    module = pytest.importorskip("mlflow.tracking")

    def boom(tracking_uri=None):
        raise RuntimeError("连不上")

    monkeypatch.setattr(module, "MlflowClient", boom)
    assert (
        registry_link.run_registration_status("wb-4", "sqlite:///x.db")["status"] == "lookup_failed"
    )
