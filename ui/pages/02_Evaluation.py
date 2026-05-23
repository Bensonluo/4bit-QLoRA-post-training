"""Evaluation — Domain-specific evaluation visualization and comparison."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st

from ui.config import DOMAINS_DIR, MLFLOW_TRACKING_URI
from ui.components.domain_adapters import (
    list_domains,
    get_adapter,
    get_domain_display_name,
    load_eval_data,
)

st.set_page_config(page_title="Evaluation", page_icon="🎯", layout="wide")
st.title("🎯 Evaluation Results")

# ── Domain Selector ─────────────────────────────────────────────

domains = list_domains()
if not domains and DOMAINS_DIR.exists():
    domains = [
        d.name for d in DOMAINS_DIR.iterdir()
        if d.is_dir() and not d.name.startswith("_")
    ]

if not domains:
    st.info(
        "No evaluation domains found.\n\n"
        "**Get started:**\n"
        "1. Run a domain evaluation script (e.g. `domains/medical_entity/evaluate.py`)\n"
        "2. Or import existing results below"
    )
    with st.expander("📥 Import Historical Results"):
        if st.button("Scan & Import to MLflow"):
            from src.tracking.eval_logger import log_eval_to_mlflow
            imported = 0
            for domain_dir in DOMAINS_DIR.iterdir():
                if not domain_dir.is_dir() or domain_dir.name.startswith("_"):
                    continue
                results_dir = domain_dir / "data" / "results"
                if not results_dir.exists():
                    continue
                for json_file in results_dir.glob("eval_detail_*.json"):
                    try:
                        log_eval_to_mlflow(json_file, experiment_name="domain-evaluation")
                        imported += 1
                    except Exception as e:
                        st.warning(f"Failed to import {json_file.name}: {e}")
            if imported:
                st.success(f"Imported {imported} file(s).")
            else:
                st.info("No evaluation files found.")
    st.stop()

selected_domain = st.selectbox(
    "Domain",
    domains,
    format_func=lambda x: get_domain_display_name(x) if get_adapter(x) else x,
)

# ── Load Data ───────────────────────────────────────────────────

data = load_eval_data(selected_domain)

# Also try MLflow
mlflow_data: list[dict] = []
try:
    import mlflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    exp = mlflow.get_experiment_by_name("domain-evaluation")
    if exp:
        runs = mlflow.search_runs(experiment_ids=[exp.experiment_id])
        if not runs.empty:
            for _, row in runs.iterrows():
                mlflow_data.append({
                    "model": row.get("tags.model_name", row.get("tags.mlflow.runName", "unknown")),
                    "source": "mlflow",
                    **{k.replace("metrics.", ""): v for k, v in row.items() if k.startswith("metrics.")},
                })
except Exception:
    pass

if not data and mlflow_data:
    data = mlflow_data

if not data:
    st.warning("No evaluation data for this domain yet.")
    st.markdown("**Options:**")
    st.markdown("1. Run the domain evaluation script")
    st.markdown("2. Import historical results below")
    with st.expander("📥 Import Historical Results"):
        if st.button("Scan & Import"):
            from src.tracking.eval_logger import log_eval_to_mlflow
            imported = 0
            results_dir = DOMAINS_DIR / selected_domain / "data" / "results"
            if results_dir.exists():
                for json_file in results_dir.glob("eval_detail_*.json"):
                    try:
                        log_eval_to_mlflow(json_file, experiment_name="domain-evaluation")
                        imported += 1
                    except Exception as e:
                        st.warning(f"Failed: {e}")
            if imported:
                st.success(f"Imported {imported} file(s).")
                st.rerun()
            else:
                st.info("No files found.")
    st.stop()

adapter = get_adapter(selected_domain)

# ── Overview ────────────────────────────────────────────────────

st.subheader("Overview")

if adapter:
    adapter.render_summary(data)
else:
    cols = st.columns(min(len(data), 4))
    for i, model_data in enumerate(data[:4]):
        with cols[i]:
            acc = model_data.get("overall_accuracy", 0)
            st.metric(
                label=model_data.get("model", f"Model {i+1}"),
                value=f"{acc:.1%}" if acc else "N/A",
            )

st.divider()

# ── Detailed Charts ─────────────────────────────────────────────

st.subheader("Detailed Analysis")

if adapter:
    adapter.render_detail(data)
else:
    st.info("Install domain adapter for detailed charts.")

st.divider()

# ── Error Analysis ──────────────────────────────────────────────

if adapter:
    with st.expander("🔍 Error Analysis", expanded=False):
        adapter.render_error_analysis(data)

st.divider()

# ── Import ──────────────────────────────────────────────────────

with st.expander("📥 Import Historical Results to MLflow"):
    if st.button("Scan & Import"):
        from src.tracking.eval_logger import log_eval_to_mlflow
        imported = 0
        for domain_dir in DOMAINS_DIR.iterdir():
            if not domain_dir.is_dir() or domain_dir.name.startswith("_"):
                continue
            results_dir = domain_dir / "data" / "results"
            if not results_dir.exists():
                continue
            for json_file in results_dir.glob("eval_detail_*.json"):
                try:
                    log_eval_to_mlflow(json_file, experiment_name="domain-evaluation")
                    imported += 1
                except Exception as e:
                    st.warning(f"Failed to import {json_file.name}: {e}")
        if imported:
            st.success(f"Imported {imported} evaluation result file(s) to MLflow.")
        else:
            st.info("No new evaluation files found to import.")
