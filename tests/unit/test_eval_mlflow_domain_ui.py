"""Domain pages must never attribute an unrelated or unlabelled MLflow run."""

import json
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def file_store(monkeypatch, tmp_path):
    mlflow = pytest.importorskip("mlflow")
    from mlflow.tracking import MlflowClient
    from mlflow.tracking._tracking_service import utils as tracking_utils

    from ui import config, runtime
    from ui.components import archive_navigation, domain_adapters

    uri = (tmp_path / "mlruns").as_uri()
    monkeypatch.setattr(tracking_utils, "_tracking_uri", None)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    monkeypatch.setattr(config, "MLFLOW_TRACKING_URI", uri)
    monkeypatch.setattr(config, "DOMAINS_DIR", tmp_path / "domains")
    monkeypatch.setattr(domain_adapters, "DOMAINS_DIR", tmp_path / "domains")
    monkeypatch.setattr(runtime, "DISPLAY_ONLY", False)
    monkeypatch.setattr(archive_navigation, "render_archive_source", lambda **kwargs: False)
    mlflow.set_tracking_uri(uri)
    client = MlflowClient(tracking_uri=uri)
    experiment_id = client.create_experiment("domain-evaluation")
    return client, experiment_id, tmp_path / "domains"


def add_run(client, experiment_id, name, domain=None):
    tags = {"model_name": name}
    if domain is not None:
        tags["eval_domain"] = domain
    run = client.create_run(experiment_id, tags=tags)
    client.log_metric(run.info.run_id, "overall_accuracy", 0.8)
    return run.info.run_id


def page():
    result = AppTest.from_file(str(ROOT / "ui/views/02_Evaluation.py"), default_timeout=45).run()
    assert not result.exception, [item.message for item in result.exception]
    return result


def test_fallback_filters_domains_and_keeps_unlabelled_unknown(file_store):
    client, experiment_id, _ = file_store
    add_run(client, experiment_id, "通用模型", "entity_matching")
    add_run(client, experiment_id, "医疗模型", "medical_entity")
    add_run(client, experiment_id, "未知领域模型")
    before = {r.info.run_id: r.data.tags for r in client.search_runs([experiment_id])}

    result = page()
    assert [item.label for item in result.metric] == ["通用模型"]
    assert any("未标明领域" in item.value for item in result.warning)
    assert any("MLflow" in item.value for item in result.caption)
    result.selectbox[0].select("medical_entity").run()
    assert not result.exception, [item.message for item in result.exception]
    assert [item.label for item in result.metric] == ["医疗模型"]
    assert any("未标明领域" in item.value for item in result.warning)
    assert {r.info.run_id: r.data.tags for r in client.search_runs([experiment_id])} == before


def test_unlabelled_runs_do_not_claim_empty_domain(file_store):
    client, experiment_id, _ = file_store
    add_run(client, experiment_id, "未知领域模型")
    result = page()
    assert not result.metric
    warnings = "\n".join(item.value for item in result.warning)
    assert "未标明领域" in warnings and "无法确认" in warnings
    assert "暂无评测数据" not in warnings


def test_page_import_tags_the_source_domain(file_store):
    client, experiment_id, domains = file_store
    for domain, name in (("entity_matching", "通用模型"), ("medical_entity", "医疗模型")):
        results = domains / domain / "data/results"
        results.mkdir(parents=True)
        (results / "eval_detail_20261028.json").write_text(
            json.dumps([{"model": name, "overall_accuracy": 0.75}]), encoding="utf-8"
        )
    result = page()
    button = next(item for item in result.button if item.label == "扫描并导入 MLflow")
    button.click().run(timeout=45)
    assert not result.exception, [item.message for item in result.exception]
    runs = client.search_runs([experiment_id])
    assert {(r.data.tags["model_name"], r.data.tags.get("eval_domain")) for r in runs} == {
        ("通用模型", "entity_matching"),
        ("医疗模型", "medical_entity"),
    }
