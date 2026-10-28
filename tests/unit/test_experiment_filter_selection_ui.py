"""Clearing explicit experiment filters must not silently show every run."""

from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def page(monkeypatch):
    from ui import queries, runtime
    from ui.components import archive_navigation

    monkeypatch.setattr(runtime, "DISPLAY_ONLY", False)
    monkeypatch.setattr(archive_navigation, "render_archive_source", lambda **kwargs: False)
    monkeypatch.setattr(
        queries,
        "fetch_runs",
        lambda *_: pd.DataFrame(
            [
                {"run_id": "a" * 32, "status": "FINISHED", "experiment_id": "1"},
                {"run_id": "b" * 32, "status": "RUNNING", "experiment_id": "2"},
            ]
        ),
    )
    monkeypatch.setattr(queries, "fetch_experiments", lambda *_: [("1", "售后"), ("2", "物流")])
    result = AppTest.from_file(str(ROOT / "ui/views/01_Experiments.py"), default_timeout=30).run()
    assert not result.exception, [error.message for error in result.exception]
    return result


def test_clearing_status_shows_no_runs_and_restoring_filters_recovers(page):
    assert any("全部运行（2）" in item.value for item in page.subheader)
    next(item for item in page.multiselect if item.label == "状态").set_value([]).run()
    assert not page.exception
    assert any("没有符合筛选条件" in item.value for item in page.info)
    assert not any(item.label == "选择一个运行" for item in page.selectbox)
    next(item for item in page.multiselect if item.label == "状态").set_value(["FINISHED"]).run()
    assert not page.exception
    assert any("全部运行（1）" in item.value for item in page.subheader)
    assert next(item for item in page.selectbox if item.label == "选择一个运行").value == "a" * 32


def test_clearing_experiments_shows_no_runs_and_model_empty_remains_unfiltered(page):
    next(item for item in page.multiselect if item.label == "实验").set_value([]).run()
    assert not page.exception
    assert any("没有符合筛选条件" in item.value for item in page.info)
    next(item for item in page.multiselect if item.label == "实验").set_value(["物流"]).run()
    assert not page.exception
    assert any("全部运行（1）" in item.value for item in page.subheader)
    assert next(item for item in page.selectbox if item.label == "选择一个运行").value == "b" * 32


def test_real_mlflow_history_filter_is_read_only(monkeypatch, tmp_path):
    from mlflow.tracking import MlflowClient
    from mlflow.tracking._tracking_service import utils as tracking_utils

    from ui import config, runtime

    monkeypatch.setattr(tracking_utils, "_tracking_uri", None)
    uri = (tmp_path / "mlruns").as_uri()
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    client = MlflowClient(tracking_uri=uri)
    first_id = client.create_experiment("真实售后")
    second_id = client.create_experiment("真实物流")
    first = client.create_run(first_id)
    second = client.create_run(second_id)
    client.set_terminated(first.info.run_id, status="FINISHED")
    monkeypatch.setattr(runtime, "DISPLAY_ONLY", True)
    monkeypatch.setattr(config, "MLFLOW_TRACKING_URI", uri)

    def snapshot():
        return {p.relative_to(tmp_path): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}

    before = snapshot()
    result = AppTest.from_file(str(ROOT / "ui/views/01_Experiments.py"), default_timeout=45).run()
    result.segmented_control(key="experiments-history-source").set_value("本地实验").run()
    assert not result.exception, [error.message for error in result.exception]
    next(item for item in result.multiselect if item.label == "状态").set_value([]).run()
    assert not result.exception
    assert any("没有符合筛选条件" in item.value for item in result.info)
    next(item for item in result.multiselect if item.label == "状态").set_value(["RUNNING"]).run()
    assert not result.exception
    selected = next(item for item in result.selectbox if item.label == "选择一个运行")
    assert selected.value == second.info.run_id
    assert snapshot() == before
