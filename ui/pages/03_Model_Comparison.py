"""Model Comparison — Side-by-side comparison with deltas and executive summary."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st

from ui.config import DOMAINS_DIR, MLFLOW_TRACKING_URI
from ui.components.domain_adapters import load_eval_data, list_domains, get_domain_display_name

st.set_page_config(page_title="Model Comparison", page_icon="⚖️", layout="wide")
st.title("⚖️ Model Comparison")

# ── Domain Selector ─────────────────────────────────────────────

domains = list_domains()
if not domains and DOMAINS_DIR.exists():
    domains = [d.name for d in DOMAINS_DIR.iterdir() if d.is_dir() and not d.name.startswith("_")]

if not domains:
    st.info("No domains with evaluation data available.")
    st.stop()

domain = st.selectbox("Domain", domains, format_func=get_domain_display_name)
data = load_eval_data(domain)

if len(data) < 2:
    st.warning("Need at least 2 models for comparison. Run more evaluations.")
    st.stop()

# ── Model Selectors ─────────────────────────────────────────────

models = [d.get("model", f"Model {i}") for i, d in enumerate(data)]

sel_cols = st.columns(2)
with sel_cols[0]:
    model_a_name = st.selectbox("Model A", models, index=0)
with sel_cols[1]:
    model_b_name = st.selectbox("Model B", models, index=min(1, len(models) - 1))

model_a = next((d for d in data if d.get("model") == model_a_name), data[0])
model_b = next((d for d in data if d.get("model") == model_b_name), data[min(1, len(data) - 1)])

st.divider()

# ── Side-by-Side Cards ──────────────────────────────────────────

st.subheader("Metric Comparison")

comparison_metrics = [
    ("Overall Accuracy", "overall_accuracy", True),
    ("MRR", "mrr", False),
    ("Avg Confidence", "avg_confidence", False),
    ("Avg Latency (ms)", "avg_latency_ms", False),
    ("Throughput (samples/s)", "throughput_per_sec", False),
]

# Model A card
with st.container(border=True):
    st.markdown(f"### {model_a_name}")
    a_cols = st.columns(len(comparison_metrics))
    for i, (label, key, is_pct) in enumerate(comparison_metrics):
        with a_cols[i]:
            val = model_a.get(key, 0)
            if is_pct:
                st.metric(label, f"{val:.1%}")
            else:
                fmt = ".3f" if abs(val) < 1 else ".1f"
                st.metric(label, f"{val:{fmt}}")

# Delta row
st.markdown("<div style='text-align:center;font-size:1.5rem;padding:0.5rem 0'>↓ Delta (B − A)</div>", unsafe_allow_html=True)

with st.container(border=True):
    st.markdown(f"### {model_b_name}")
    b_cols = st.columns(len(comparison_metrics))
    for i, (label, key, is_pct) in enumerate(comparison_metrics):
        with b_cols[i]:
            val_b = model_b.get(key, 0)
            val_a = model_a.get(key, 0)
            delta = val_b - val_a
            if is_pct:
                st.metric(label, f"{val_b:.1%}", delta=f"{delta:+.1%}")
            else:
                fmt = ".3f" if abs(val_b) < 1 else ".1f"
                st.metric(label, f"{val_b:{fmt}}", delta=f"{delta:+{fmt}}")

st.divider()

# ── Breakdowns ──────────────────────────────────────────────────

st.subheader("Breakdown")

b1, b2 = st.columns(2)

with b1:
    st.markdown("**Accuracy by Difficulty**")
    difficulties = ["easy", "medium", "hard"]
    diff_cols = st.columns(len(difficulties))
    for i, diff in enumerate(difficulties):
        with diff_cols[i]:
            acc_a = model_a.get("accuracy_by_difficulty", {}).get(diff, 0)
            acc_b = model_b.get("accuracy_by_difficulty", {}).get(diff, 0)
            delta = acc_b - acc_a
            st.metric(f"{diff.title()}", f"{acc_b:.1%}", delta=f"{delta:+.1%}")

with b2:
    st.markdown("**Accuracy by Entity Type**")
    entity_types = list(model_a.get("accuracy_by_type", {}).keys()) or ["drug", "hospital"]
    ent_cols = st.columns(len(entity_types))
    for i, etype in enumerate(entity_types):
        with ent_cols[i]:
            acc_a = model_a.get("accuracy_by_type", {}).get(etype, 0)
            acc_b = model_b.get("accuracy_by_type", {}).get(etype, 0)
            delta = acc_b - acc_a
            st.metric(f"{etype.title()}", f"{acc_b:.1%}", delta=f"{delta:+.1%}")

st.divider()

# ── Executive Summary ───────────────────────────────────────────

st.subheader("Executive Summary")
results_dir = DOMAINS_DIR / domain / "data" / "results"
if results_dir.exists():
    summaries = sorted(results_dir.glob("executive_summary_*.md"), reverse=True)
    if summaries:
        with open(summaries[0]) as f:
            st.markdown(f.read())
    else:
        st.info("No executive summary file found.")
else:
    st.info("No results directory found.")

st.divider()

# ── Cost Estimation ─────────────────────────────────────────────

st.subheader("Cost Estimation")

latency_a = model_a.get("avg_latency_ms", 0)
latency_b = model_b.get("avg_latency_ms", 0)
tp_a = model_a.get("throughput_per_sec", 1)
tp_b = model_b.get("throughput_per_sec", 1)

import pandas as pd
cost_df = pd.DataFrame({
    "Metric": ["Avg Latency", "Throughput", "Time for 1M samples", "Deployment", "Data Security"],
    model_a_name: [
        f"{latency_a:.0f} ms",
        f"{tp_a:.0f} samples/s",
        f"{1_000_000 / max(tp_a, 0.1) / 3600:.1f} hours",
        "Local GPU" if latency_a > 0 else "N/A",
        "On-premise",
    ],
    model_b_name: [
        f"{latency_b:.0f} ms",
        f"{tp_b:.0f} samples/s",
        f"{1_000_000 / max(tp_b, 0.1) / 3600:.1f} hours",
        "Local GPU" if latency_b > 0 else "N/A",
        "On-premise",
    ],
})
st.dataframe(cost_df, width='stretch', hide_index=True)
