"""QLoRA Post-Training Lab — Dashboard Home."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

from ui.config import MLFLOW_TRACKING_URI, PROJECT_ROOT

st.set_page_config(
    page_title="QLoRA Post-Training Lab",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🧪 QLoRA Post-Training Lab")
st.caption("Configure, train, evaluate, and compare models — all in one place.")

# ── System Status Bar ───────────────────────────────────────────

status_cols = st.columns([1, 1, 1, 1, 2])

with status_cols[0]:
    try:
        import mlflow
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.search_experiments()
        st.success("MLflow", icon="✅")
    except Exception:
        st.error("MLflow", icon="❌")

with status_cols[1]:
    try:
        import plotly  # noqa: F401
        st.success("Plotly", icon="✅")
    except ImportError:
        st.error("Plotly", icon="❌")

with status_cols[2]:
    try:
        import transformers  # noqa: F401
        st.success("Transformers", icon="✅")
    except ImportError:
        st.warning("Transformers", icon="⚠️")

with status_cols[3]:
    try:
        import torch
        device = "CUDA" if torch.cuda.is_available() else "MPS" if torch.backends.mps.is_available() else "CPU"
        st.info(f"PyTorch ({device})", icon="🎮" if device == "CUDA" else "💻")
    except ImportError:
        st.error("PyTorch", icon="❌")

with status_cols[4]:
    st.markdown(f"<div style='text-align:right;color:#94A3B8;font-size:0.85rem'>Project: {PROJECT_ROOT.name}</div>", unsafe_allow_html=True)

st.divider()

# ── Quick Stats ─────────────────────────────────────────────────

try:
    import mlflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    all_exps = mlflow.search_experiments()
    all_runs = mlflow.search_runs(experiment_ids=[e.experiment_id for e in all_exps]) if all_exps else None

    if all_runs is not None and not all_runs.empty:
        total = len(all_runs)
        finished = len(all_runs[all_runs["status"] == "FINISHED"]) if "status" in all_runs.columns else 0
        running = len(all_runs[all_runs["status"] == "RUNNING"]) if "status" in all_runs.columns else 0
        failed = len(all_runs[all_runs["status"] == "FAILED"]) if "status" in all_runs.columns else 0
    else:
        total = finished = running = failed = 0
except Exception:
    total = finished = running = failed = 0

stat_cols = st.columns(5)
with stat_cols[0]:
    st.metric("Total Runs", total)
with stat_cols[1]:
    st.metric("Completed", finished, delta=None)
with stat_cols[2]:
    st.metric("Running", running)
with stat_cols[3]:
    st.metric("Failed", failed, delta=f"-{failed}" if failed else None)
with stat_cols[4]:
    try:
        from src.tracking.runner import TrainingRunner
        runner = TrainingRunner(project_root=str(PROJECT_ROOT))
        active = runner.list_active()
        st.metric("Active Jobs", len(active))
    except Exception:
        st.metric("Active Jobs", 0)

st.divider()

# ── Quick Actions ───────────────────────────────────────────────

st.subheader("Quick Actions")

qa_cols = st.columns(4)
with qa_cols[0]:
    if st.button("🏋️ New Training", width='stretch', type="primary"):
        st.switch_page("pages/00_Training_Lab.py")
with qa_cols[1]:
    if st.button("📊 View Experiments", width='stretch'):
        st.switch_page("pages/01_Experiments.py")
with qa_cols[2]:
    if st.button("🎯 Evaluation", width='stretch'):
        st.switch_page("pages/02_Evaluation.py")
with qa_cols[3]:
    if st.button("⚖️ Compare Models", width='stretch'):
        st.switch_page("pages/03_Model_Comparison.py")

st.divider()

# ── Recent Activity ─────────────────────────────────────────────

st.subheader("Recent Activity")

try:
    import mlflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    all_exps = mlflow.search_experiments()
    if all_exps:
        runs = mlflow.search_runs(
            experiment_ids=[e.experiment_id for e in all_exps],
            order_by=["start_time DESC"],
            max_results=5,
        )
        if not runs.empty:
            for _, row in runs.iterrows():
                name = row.get("tags.mlflow.runName", row["run_id"][:8])
                status = row.get("status", "UNKNOWN")
                model = row.get("params.model.name", "—")
                loss = row.get("metrics.train_loss", None)
                start = row.get("start_time", "—")

                status_emoji = {"FINISHED": "✅", "RUNNING": "🟢", "FAILED": "🔴"}.get(status, "⚪")
                loss_str = f" | Loss: {loss:.3f}" if loss is not None else ""

                st.markdown(
                    f"<div style='padding:0.5rem 0;border-bottom:1px solid #334155;'>"
                    f"<b>{status_emoji} {name}</b> <span style='color:#94A3B8'>— {model}{loss_str}</span>"
                    f"<span style='float:right;color:#64748B;font-size:0.8rem'>{start}</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
        else:
            st.info("No training runs yet. Start your first experiment from the Training Lab.")
    else:
        st.info("No experiments yet. Start your first training run.")
except Exception as e:
    st.info(f"Could not load activity: {e}")
