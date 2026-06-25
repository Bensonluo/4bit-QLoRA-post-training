"""Sidebar filter helpers for the dashboard."""

from __future__ import annotations

import streamlit as st

from ui.config import DOMAINS_DIR


def experiment_filter(label: str = "Experiment") -> str | None:
    """Dropdown of MLflow experiments."""
    try:
        import mlflow

        from ui.config import MLFLOW_TRACKING_URI
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        experiments = mlflow.search_experiments()
        exp_names = [e.name for e in experiments]
        if not exp_names:
            return None
        return st.sidebar.selectbox(label, exp_names)
    except Exception:
        return None


def run_multi_select(
    experiment_name: str | None = None,
    label: str = "Select Runs",
) -> list[str]:
    """Multi-select runs from an experiment."""
    try:
        import mlflow

        from ui.config import MLFLOW_TRACKING_URI
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        if experiment_name:
            exp = mlflow.get_experiment_by_name(experiment_name)
            if exp is None:
                return []
            runs = mlflow.search_runs(experiment_ids=[exp.experiment_id])
        else:
            runs = mlflow.search_runs()
        if runs.empty:
            return []
        run_names = runs.get("tags.mlflow.runName", runs["run_id"]).tolist()
        return st.sidebar.multiselect(label, run_names, max_selections=5)
    except Exception:
        return []


def metric_selector(
    available_metrics: list[str],
    label: str = "Metric",
) -> str:
    """Dropdown to pick a metric."""
    return st.sidebar.selectbox(label, available_metrics, index=0)


def domain_filter(label: str = "Domain") -> str | None:
    """Dropdown of available domains (scans domains/ dir)."""
    if not DOMAINS_DIR.exists():
        return None
    domains = [
        d.name for d in DOMAINS_DIR.iterdir()
        if d.is_dir() and not d.name.startswith("_") and d.name != "__pycache__"
    ]
    if not domains:
        return None
    return st.sidebar.selectbox(label, domains)
