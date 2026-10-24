"""Read real historical curves through supported MLflow client methods."""

import pytest

pytest.importorskip("mlflow")
from mlflow.tracking import MlflowClient

from ui.queries import (
    fetch_metric_history_by_name,
    fetch_metric_history_by_run_id,
    fetch_metric_names_by_name,
)


@pytest.fixture()
def recorded_run(tmp_path):
    uri = f"sqlite:///{tmp_path / 'metrics.db'}"
    client = MlflowClient(tracking_uri=uri)
    experiment = client.create_experiment("Historical evidence")
    run = client.create_run(experiment, tags={"mlflow.runName": "recorded-curve"})
    for step, value in enumerate([2.0, 1.25, 0.8]):
        client.log_metric(run.info.run_id, "loss", value, timestamp=1000 + step, step=step)
    client.set_terminated(run.info.run_id)
    return uri, run.info.run_id


def test_metric_curve_by_uuid_reads_logged_steps(recorded_run):
    uri, run_id = recorded_run
    assert fetch_metric_history_by_run_id(uri, run_id, "loss") == [(0, 2.0), (1, 1.25), (2, 0.8)]
    assert fetch_metric_history_by_run_id(uri, run_id, "not-logged") == []


def test_metric_curve_by_name_includes_nondefault_experiments(recorded_run):
    uri, _ = recorded_run
    assert fetch_metric_names_by_name(uri, "recorded-curve") == ["loss"]
    assert fetch_metric_history_by_name(uri, "recorded-curve", "loss") == [
        (0, 2.0),
        (1, 1.25),
        (2, 0.8),
    ]
    assert fetch_metric_history_by_name(uri, "absent-run", "loss") == []
