"""Cached MLflow query layer for the dashboard pages.

Every Streamlit rerun (widget interaction, button click) re-executes page
scripts top to bottom; an uncached ``mlflow.search_runs`` on the file-store
backend costs hundreds of milliseconds each time. These helpers cache the
DataFrame/tuple results for 30s — fresh enough for live loss curves, cheap
enough for interactive pages. ``st.cache_data`` returns a copy of cached
DataFrames on every call, so callers may filter/mutate freely.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

_TTL_SECONDS = 30


@st.cache_data(ttl=_TTL_SECONDS, show_spinner=False)
def fetch_experiments(tracking_uri: str) -> list[tuple[str, str]]:
    """Return [(experiment_id, name), ...] for all experiments."""
    import mlflow

    mlflow.set_tracking_uri(tracking_uri)
    return [(e.experiment_id, e.name) for e in mlflow.search_experiments()]


@st.cache_data(ttl=_TTL_SECONDS, show_spinner=False)
def fetch_runs(tracking_uri: str) -> pd.DataFrame:
    """Return the search_runs DataFrame across all experiments (newest first)."""
    import mlflow

    mlflow.set_tracking_uri(tracking_uri)
    experiments = mlflow.search_experiments()
    if not experiments:
        return pd.DataFrame()
    return mlflow.search_runs(
        experiment_ids=[e.experiment_id for e in experiments],
        order_by=["start_time DESC"],
    )


@st.cache_data(ttl=_TTL_SECONDS, show_spinner=False)
def fetch_metric_history_by_run_id(
    tracking_uri: str, run_id: str, metric: str
) -> list[tuple[int, float]]:
    """Return [(step, value), ...] for one metric of one MLflow run uuid."""
    from mlflow.tracking import MlflowClient

    history = MlflowClient(tracking_uri=tracking_uri).get_metric_history(run_id, metric)
    return [(m.step, m.value) for m in history]


@st.cache_data(ttl=_TTL_SECONDS, show_spinner=False)
def fetch_metric_names_by_name(tracking_uri: str, run_name: str) -> list[str]:
    """Return the sorted metric names logged for a run (resolved by runName tag)."""
    from mlflow.tracking import MlflowClient

    run_id = _unique_run_id_by_name(tracking_uri, run_name)
    if run_id is None:
        return []
    run = MlflowClient(tracking_uri=tracking_uri).get_run(run_id)
    return sorted(run.data.metrics.keys())


def _unique_run_id_by_name(tracking_uri: str, run_name: str) -> str | None:
    runs = fetch_runs(tracking_uri)
    if runs.empty or "tags.mlflow.runName" not in runs.columns:
        return None
    matching = runs[runs["tags.mlflow.runName"] == run_name]
    if len(matching) > 1:
        raise ValueError(
            "有多个同名实验，不能确定这次训练对应哪条曲线；"
            "请到「实验记录」按运行 ID 查看，不会代选第一条记录。"
        )
    return None if matching.empty else str(matching.iloc[0]["run_id"])


@st.cache_data(ttl=_TTL_SECONDS, show_spinner=False)
def fetch_metric_history_by_name(
    tracking_uri: str, run_name: str, metric: str
) -> list[tuple[int, float]]:
    """Return [(step, value), ...] resolving a run by its ``mlflow.runName`` tag.

    Used by the Activity tab where only the TrainingLab run_name is known.
    """
    from mlflow.tracking import MlflowClient

    run_id = _unique_run_id_by_name(tracking_uri, run_name)
    if run_id is None:
        return []
    history = MlflowClient(tracking_uri=tracking_uri).get_metric_history(run_id, metric)
    return [(m.step, m.value) for m in history]


@st.cache_data(ttl=_TTL_SECONDS, show_spinner=False)
def fetch_model_versions(tracking_uri: str, name: str | None = None) -> pd.DataFrame:
    """Return registered model versions as a DataFrame (newest version per name kept).

    Columns: name, version(int), current_stage, aliases (comma-joined string),
    run_id, status, created. ``aliases`` comes from ModelVersion.aliases —
    the registry-native replacement for deprecated stages.
    """
    import mlflow
    from mlflow.tracking import MlflowClient

    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()
    if name:
        # Escape single quotes in the filter string to avoid injection.
        escaped = name.replace("'", "''")
        versions = client.search_model_versions(f"name='{escaped}'")
    else:
        versions = client.search_model_versions()
    rows = [
        {
            "name": v.name,
            "version": int(v.version),
            "current_stage": v.current_stage,
            "aliases": ", ".join(getattr(v, "aliases", None) or []),
            "run_id": v.run_id,
            "status": v.status,
            "created": pd.to_datetime(int(v.creation_timestamp), unit="ms").strftime("%m-%d %H:%M"),
        }
        for v in versions
    ]
    return pd.DataFrame(rows)
