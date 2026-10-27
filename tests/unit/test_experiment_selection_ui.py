"""Local experiment names are labels; every action targets an actual run ID."""

from copy import deepcopy
from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("streamlit")
pytest.importorskip("mlflow")
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[2]
PAGE = ROOT / "ui/pages/01_Experiments.py"
RUN_A, RUN_B, RUN_C = "11111111-actual-a", "22222222-actual-b", "33333333-actual-c"


def widget(elements, label):
    return next(item for item in elements if item.label == label)


@pytest.fixture
def records():
    return pd.DataFrame(
        [
            {
                "run_id": RUN_A,
                "tags.mlflow.runName": "capable-auk-759",
                "experiment_id": "e1",
                "status": "FINISHED",
                "params.model.name": "model-a",
                "params.training.num_epochs": 1,
                "metrics.loss": 0.9,
            },
            {
                "run_id": RUN_B,
                "tags.mlflow.runName": "capable-auk-759",
                "experiment_id": "e1",
                "status": "FAILED",
                "params.model.name": "model-b",
                "params.training.num_epochs": 2,
                "metrics.loss": 0.4,
            },
            {
                "run_id": RUN_C,
                "tags.mlflow.runName": "other-run",
                "experiment_id": "e2",
                "status": "FINISHED",
                "params.model.name": "model-c",
                "params.training.num_epochs": 3,
                "metrics.loss": 0.2,
            },
        ]
    )


@pytest.fixture
def browser(monkeypatch, records):
    import mlflow

    from ui import queries, runtime
    from ui.components import charts

    calls = {"history": [], "comparisons": [], "deleted": [], "cleared": []}

    class Runs:
        def __call__(self, *args, **kwargs):
            return records.copy(deep=True)

        def clear(self):
            calls["cleared"].append(True)

    def history(uri, run_id, metric):
        calls["history"].append((run_id, metric))
        return [(1, 0.5), (2, 0.2)]

    original = charts.make_bar_comparison

    def comparison(models, metrics, title):
        calls["comparisons"].append((list(models), deepcopy(metrics)))
        return original(models, metrics, title)

    monkeypatch.setattr(runtime, "DISPLAY_ONLY", False)
    monkeypatch.setattr(
        queries, "fetch_experiments", lambda *a, **k: [("e1", "first"), ("e2", "second")]
    )
    monkeypatch.setattr(queries, "fetch_runs", Runs())
    monkeypatch.setattr(queries, "fetch_metric_history_by_run_id", history)
    monkeypatch.setattr(charts, "make_bar_comparison", comparison)
    monkeypatch.setattr(mlflow, "delete_run", lambda run_id: calls["deleted"].append(run_id))
    return AppTest.from_file(str(PAGE), default_timeout=30).run(), calls


def test_same_named_second_run_loads_its_own_metric_history(browser):
    page, calls = browser
    assert not page.exception
    widget(page.selectbox, "选择一个运行").set_value(RUN_B).run()
    assert not page.exception
    assert calls["history"][-1] == (RUN_B, "loss")
    assert widget(page.selectbox, "选择一个运行").value == RUN_B


def test_comparison_uses_only_selected_ids_and_preserves_selected_order(browser):
    page, calls = browser
    selector = widget(page.multiselect, "选择 2–5 个运行进行对比")
    assert selector.options[0] != selector.options[1]
    selector.set_value([RUN_C, RUN_B]).run()
    assert not page.exception
    labels, metrics = calls["comparisons"][-1]
    assert labels == ["other-run", "capable-auk-759（22222222）"]
    assert metrics == {"loss": [0.2, 0.4]}
    difference = next(item.value for item in page.dataframe if "model.name" in item.value.index)
    assert list(difference.columns) == labels
    assert difference.loc["model.name"].tolist() == ["model-c", "model-b"]


def test_second_same_named_run_details_and_delete_target_exact_id(browser):
    page, calls = browser
    widget(page.selectbox, "选择一个运行").set_value(RUN_B).run()
    assert not page.exception
    params = next(
        item.value for item in page.dataframe if list(item.value.columns) == ["参数", "值"]
    )
    assert dict(zip(params["参数"], params["值"]))["model.name"] == "model-b"
    assert any(RUN_B in item.value for item in page.caption)
    widget(page.button, "确认删除").click().run()
    assert not page.exception
    assert calls["deleted"] == [RUN_B]
    assert calls["cleared"] == [True]


@pytest.mark.parametrize("missing", [None, float("nan"), "", "   ", pd.NA])
def test_missing_display_name_falls_back_to_identifiable_id(browser, records, missing):
    page, calls = browser
    records.loc[1, "tags.mlflow.runName"] = missing
    page.run()
    selector = widget(page.selectbox, "选择一个运行")
    assert selector.options[1] == "未命名运行（22222222）"
    selector.set_value(RUN_B).run()
    assert not page.exception
    assert calls["history"][-1][0] == RUN_B


def test_absent_name_column_still_binds_ids(browser, records):
    page, calls = browser
    records.drop(columns=["tags.mlflow.runName"], inplace=True)
    page.run()
    selector = widget(page.selectbox, "选择一个运行")
    assert selector.options == [
        "未命名运行（11111111）",
        "未命名运行（22222222）",
        "未命名运行（33333333）",
    ]
    selector.set_value(RUN_B).run()
    assert not page.exception
    assert calls["history"][-1][0] == RUN_B


def test_duplicate_names_with_colliding_short_ids_extend_visible_prefix(browser, records):
    page, calls = browser
    records.loc[0, "run_id"] = "deadbeef0-actual-a"
    records.loc[1, "run_id"] = "deadbeef1-actual-b"
    page.run()
    selector = widget(page.selectbox, "选择一个运行")
    assert selector.options[:2] == ["capable-auk-759（deadbeef0）", "capable-auk-759（deadbeef1）"]
    selector.set_value("deadbeef1-actual-b").run()
    assert not page.exception
    assert calls["history"][-1][0] == "deadbeef1-actual-b"


def test_filtered_empty_set_stops_before_selectors_and_can_be_restored(browser):
    page, _ = browser
    widget(page.multiselect, "状态").set_value(["FINISHED"]).run()
    widget(page.multiselect, "模型").set_value(["model-b"]).run()
    assert not page.exception
    assert any("没有符合筛选条件" in item.value for item in page.info)
    assert not any(item.label == "选择一个运行" for item in page.selectbox)
    assert not any(item.label == "选择 2–5 个运行进行对比" for item in page.multiselect)
    widget(page.multiselect, "模型").set_value([]).run()
    assert not page.exception
    assert widget(page.selectbox, "选择一个运行").value == RUN_A


def test_display_only_cannot_delete_selected_local_record(browser, monkeypatch):
    from ui import runtime
    from ui.components import archive_navigation

    page, calls = browser
    monkeypatch.setattr(runtime, "DISPLAY_ONLY", True)
    monkeypatch.setattr(archive_navigation, "render_archive_source", lambda **kwargs: False)
    page.run()
    widget(page.selectbox, "选择一个运行").set_value(RUN_B).run()
    button = widget(page.button, "确认删除")
    assert button.disabled
    button.click().run()
    assert not page.exception
    assert calls["deleted"] == []


def test_public_archive_entry_does_not_query_local_mlflow(monkeypatch):
    from ui import queries, runtime

    def forbidden(*args, **kwargs):
        raise AssertionError("Public archives must not query local MLflow")

    monkeypatch.setattr(runtime, "DISPLAY_ONLY", True)
    monkeypatch.setattr(queries, "fetch_runs", forbidden)
    monkeypatch.setattr(queries, "fetch_experiments", forbidden)
    page = AppTest.from_file(str(PAGE), default_timeout=30).run()
    assert not page.exception
    assert not any(item.label == "选择一个运行" for item in page.selectbox)
    assert any("公开" in item.value for item in page.caption)
