"""Experiments — Browse and compare all training/eval runs."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st

st.set_page_config(page_title="Experiments", page_icon="📊", layout="wide")
st.title("📊 Experiments")

try:
    import mlflow  # noqa: F401  (still needed: delete_run + metric history)
    import pandas as pd

    from ui.config import MLFLOW_TRACKING_URI
    from ui.queries import fetch_experiments, fetch_runs
except ImportError:
    st.error("Install mlflow to use this page: `pip install mlflow`")
    st.stop()

# Fetch all runs (cached 30s — see ui/queries.py)
all_experiments = fetch_experiments(MLFLOW_TRACKING_URI)
exp_id_by_name = {name: eid for eid, name in all_experiments}
runs = fetch_runs(MLFLOW_TRACKING_URI)

if runs.empty:
    st.info("No experiments found. Start training from the Training Lab.")
    st.stop()

# ── KPI Cards ───────────────────────────────────────────────────

total = len(runs)
finished = len(runs[runs["status"] == "FINISHED"]) if "status" in runs.columns else 0
running = len(runs[runs["status"] == "RUNNING"]) if "status" in runs.columns else 0
failed = len(runs[runs["status"] == "FAILED"]) if "status" in runs.columns else 0

kpi = st.columns(4)
with kpi[0]:
    st.metric("Total Runs", total)
with kpi[1]:
    st.metric("Completed", finished)
with kpi[2]:
    st.metric("Running", running)
with kpi[3]:
    st.metric("Failed", failed, delta=f"-{failed}" if failed else None)

st.divider()

# ── Filters ─────────────────────────────────────────────────────

with st.expander("🔍 Filters", expanded=False):
    f1, f2, f3 = st.columns(3)
    with f1:
        statuses = runs["status"].unique().tolist() if "status" in runs.columns else []
        selected_status = st.multiselect("Status", statuses, default=statuses)
    with f2:
        model_vals = (
            runs["params.model.name"].dropna().unique().tolist()
            if "params.model.name" in runs.columns
            else []
        )
        selected_models = st.multiselect("Model", model_vals, default=[])
    with f3:
        exp_names = [name for _, name in all_experiments]
        selected_exps = st.multiselect("Experiment", exp_names, default=exp_names)

    if selected_status and "status" in runs.columns:
        runs = runs[runs["status"].isin(selected_status)]
    if selected_models and "params.model.name" in runs.columns:
        runs = runs[runs["params.model.name"].isin(selected_models)]
    if selected_exps and "experiment_id" in runs.columns:
        wanted_ids = [exp_id_by_name[n] for n in selected_exps if n in exp_id_by_name]
        runs = runs[runs["experiment_id"].isin(wanted_ids)]

st.subheader(f"All Runs ({len(runs)})")

# ── Runs Table ──────────────────────────────────────────────────

display_map = {
    "tags.mlflow.runName": "Run Name",
    "status": "Status",
    "start_time": "Start Time",
    "params.model.name": "Model",
    "params.training.num_epochs": "Epochs",
    "params.training.learning_rate": "LR",
    "params.lora.r": "LoRA r",
    "metrics.train_loss": "Train Loss",
    "metrics.eval/eval_loss": "Eval Loss",
    "params.data.dataset_name": "Dataset",
}
display_cols = {k: v for k, v in display_map.items() if k in runs.columns}
if display_cols:
    df_display = runs[list(display_cols.keys())].rename(columns=display_cols)
    st.dataframe(df_display, width="stretch", hide_index=True)
else:
    st.dataframe(runs.head(20), width="stretch", hide_index=True)

st.divider()

# ── Compare Runs ────────────────────────────────────────────────

st.subheader("Compare Runs")

run_name_col = "tags.mlflow.runName" if "tags.mlflow.runName" in runs.columns else "run_id"
run_names = runs[run_name_col].tolist()
selected = st.multiselect("Select 2–5 runs", run_names, max_selections=5)

if len(selected) >= 2:
    selected_runs = runs[runs[run_name_col].isin(selected)]

    # Metric comparison
    metric_cols = [c for c in runs.columns if c.startswith("metrics.")]
    if metric_cols:
        compare_metrics = st.multiselect("Metrics", metric_cols, default=metric_cols[:3])
        if compare_metrics:
            models = selected_runs[run_name_col].tolist()
            metrics_data = {}
            for mc in compare_metrics:
                # None (not 0.0) for missing metrics — never fabricate data points.
                metrics_data[mc.replace("metrics.", "")] = [
                    None if pd.isna(v) else float(v) for v in selected_runs[mc].tolist()
                ]
            from ui.components.charts import make_bar_comparison

            fig = make_bar_comparison(models, metrics_data, "Metric Comparison")
            st.plotly_chart(fig, width="stretch")

    # Param diff
    param_cols = [c for c in runs.columns if c.startswith("params.")]
    if param_cols:
        with st.expander("Parameter Diff"):
            diff_data = {}
            for pc in param_cols:
                vals = selected_runs[pc].tolist()
                if len(set(str(v) for v in vals)) > 1:
                    diff_data[pc.replace("params.", "")] = vals
            if diff_data:
                diff_df = pd.DataFrame(diff_data, index=selected_runs[run_name_col].tolist())
                st.dataframe(diff_df.T, width="stretch")
            else:
                st.info("All selected runs have identical parameters.")

st.divider()

# ── Run Details ─────────────────────────────────────────────────

st.subheader("Run Details")
selected_run = st.selectbox("Select a run", run_names)
if selected_run:
    run_row = runs[runs[run_name_col] == selected_run].iloc[0]
    run_id = run_row["run_id"]

    d1, d2 = st.columns(2)
    with d1, st.expander("Parameters"):
        params = {
            k.replace("params.", ""): v for k, v in run_row.items() if k.startswith("params.")
        }
        if params:
            st.dataframe(
                pd.DataFrame(list(params.items()), columns=["Parameter", "Value"]),
                width="stretch",
                hide_index=True,
            )
    with d2, st.expander("Metrics"):
        metrics = {
            k.replace("metrics.", ""): v for k, v in run_row.items() if k.startswith("metrics.")
        }
        if metrics:
            st.dataframe(
                pd.DataFrame(list(metrics.items()), columns=["Metric", "Value"]),
                width="stretch",
                hide_index=True,
            )

    with st.expander("Metric Curve"):
        try:
            from ui.queries import fetch_metric_history_by_run_id

            metric_names = [
                c.replace("metrics.", "") for c in runs.columns if c.startswith("metrics.")
            ]
            if metric_names:
                default_idx = metric_names.index("loss") if "loss" in metric_names else 0
                chosen = st.selectbox("Metric", metric_names, index=default_idx)
                history = fetch_metric_history_by_run_id(MLFLOW_TRACKING_URI, run_id, chosen)
                if history:
                    from ui.components.charts import make_metric_timeseries

                    fig = make_metric_timeseries({selected_run: history}, chosen)
                    st.plotly_chart(fig, width="stretch")
        except Exception as e:
            st.warning(f"Could not load metric history: {e}")

    # Single run delete
    if st.button("🗑 Delete This Run", type="secondary"):
        mlflow.delete_run(run_id)
        st.success("Run deleted.")
        st.rerun()
