"""Malformed numeric evidence stays unknown instead of crashing or fabricating metrics."""

import json
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[2]


def render_page(monkeypatch, tmp_path, filename, source_key, local_source, records):
    from ui import config, runtime
    from ui.components import domain_adapters

    domain_root = tmp_path / "domains"
    result_dir = domain_root / "entity_matching/data/results"
    result_dir.mkdir(parents=True)
    (result_dir / "eval_detail_20261028.json").write_text(json.dumps(records))
    monkeypatch.setattr(domain_adapters, "DOMAINS_DIR", domain_root)
    monkeypatch.setattr(config, "MLFLOW_TRACKING_URI", (tmp_path / "missing-mlruns").as_uri())
    monkeypatch.setattr(runtime, "DISPLAY_ONLY", True)
    page = AppTest.from_file(str(ROOT / "ui/views" / filename), default_timeout=45).run()
    assert not page.exception
    page.segmented_control(key=source_key).set_value(local_source).run()
    assert not page.exception, [item.message for item in page.exception]
    assert not (tmp_path / "missing-mlruns").exists()
    return page


@pytest.mark.parametrize(
    "bad_value", [True, 10**400, float("nan"), float("inf"), "not-a-number", None]
)
def test_comparison_renders_invalid_top_level_metrics_as_unknown(monkeypatch, tmp_path, bad_value):
    page = render_page(
        monkeypatch,
        tmp_path,
        "03_Model_Comparison.py",
        "comparison-history-source",
        "本地模型对照",
        [
            {"model": "A", "overall_accuracy": bad_value, "throughput_per_sec": bad_value},
            {"model": "B", "overall_accuracy": 0.75, "throughput_per_sec": 2},
        ],
    )
    frame = next(item.value for item in page.dataframe if "差值（B − A）" in item.value.columns)
    for label in ("总体准确率", "吞吐 (样本/秒)"):
        row = frame.loc[frame["指标"] == label].iloc[0]
        assert row["A · A"] == "—" and row["差值（B − A）"] == "—"
    assert frame.loc[frame["指标"] == "总体准确率"].iloc[0]["B · B"] == "75.0%"


def test_evaluation_retains_valid_model_with_extreme_metrics(monkeypatch, tmp_path):
    page = render_page(
        monkeypatch,
        tmp_path,
        "02_Evaluation.py",
        "evaluation-history-source",
        "本地领域评测",
        [
            {"model": "bad", "overall_accuracy": 10**400, "avg_latency_ms": float("nan")},
            {"model": "good", "overall_accuracy": 0.8, "avg_latency_ms": 12.0},
        ],
    )
    assert [(item.label, item.value) for item in page.metric] == [("bad", "—"), ("good", "80.0%")]


def test_extreme_finite_values_do_not_produce_infinite_deltas_or_cost(monkeypatch, tmp_path):
    page = render_page(
        monkeypatch,
        tmp_path,
        "03_Model_Comparison.py",
        "comparison-history-source",
        "本地模型对照",
        [
            {"model": "A", "overall_accuracy": -1e308, "throughput_per_sec": 5e-324},
            {"model": "B", "overall_accuracy": 1e308, "throughput_per_sec": 2},
        ],
    )
    metrics = next(item.value for item in page.dataframe if "差值（B − A）" in item.value.columns)
    assert metrics.loc[metrics["指标"] == "总体准确率"].iloc[0]["差值（B − A）"] == "—"
    cost = next(
        item.value for item in page.dataframe if "100 万样本耗时" in item.value["指标"].values
    )
    assert cost.loc[cost["指标"] == "100 万样本耗时"].iloc[0]["A · A"] == "—"
