"""Model Registry — browse versions, manage aliases and legacy stages."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st

st.set_page_config(page_title="Model Registry", page_icon="🏛️", layout="wide")
st.title("🏛️ Model Registry")

try:
    from mlflow.tracking import MlflowClient  # noqa: F401  (availability probe)

    from ui.config import MLFLOW_TRACKING_URI
    from ui.queries import fetch_model_versions
except ImportError:
    st.error("Install mlflow to use this page: `pip install mlflow`")
    st.stop()


def _tracker():
    """The repo's tracker abstraction — same construction as registry_cli.py."""
    from config.base import LoggingConfig
    from src.tracking import get_tracker

    cfg = LoggingConfig(use_mlflow=True)
    cfg.mlflow_tracking_uri = MLFLOW_TRACKING_URI
    return get_tracker(cfg)


versions_df = fetch_model_versions(MLFLOW_TRACKING_URI)

if versions_df.empty:
    st.info(
        "No registered models yet. Set `register_model=True` in the training "
        "LoggingConfig, or run `python scripts/registry_cli.py register`."
    )
    st.stop()

# ── Model selector + KPI cards ──────────────────────────────────

model_names = sorted(versions_df["name"].unique())
model_name = st.selectbox("Registered Model", model_names)
model_versions = versions_df[versions_df["name"] == model_name].sort_values(
    "version", ascending=False
)

champion_row = model_versions[model_versions["aliases"].str.contains("champion", na=False)]
challenger_row = model_versions[model_versions["aliases"].str.contains("challenger", na=False)]

k1, k2, k3 = st.columns(3)
with k1:
    st.metric("Versions", len(model_versions))
with k2:
    st.metric(
        "🏷 Champion", f"v{champion_row.iloc[0]['version']}" if not champion_row.empty else "—"
    )
with k3:
    st.metric(
        "🥈 Challenger",
        f"v{challenger_row.iloc[0]['version']}" if not challenger_row.empty else "—",
    )

st.caption(
    "Aliases (champion / challenger) are MLflow's replacement for registry "
    "stages, deprecated since 2.9.0. Load a model by alias: "
    f"`models:/{model_name}@champion`"
)

st.dataframe(
    model_versions[["version", "current_stage", "aliases", "status", "created", "run_id"]],
    width="stretch",
    hide_index=True,
)

st.divider()

# ── Alias actions (recommended path) ─────────────────────────────

st.subheader("Aliases")
a1, a2, a3, a4 = st.columns([1, 1, 1, 1])
version_options = [str(v) for v in model_versions["version"].tolist()]
with a1:
    alias_version = st.selectbox("Version", version_options, key="alias_version")
with a2:
    alias_name = st.selectbox("Alias", ["champion", "challenger"], key="alias_name")
with a3:
    if st.button("🏷 Set", width="stretch"):
        _tracker().set_model_alias(model_name, alias_version, alias_name)
        fetch_model_versions.clear()
        st.success(f"`{alias_name}` → v{alias_version}")
        st.rerun()
with a4:
    if st.button("🗑 Remove", width="stretch"):
        _tracker().delete_model_alias(model_name, alias_name)
        fetch_model_versions.clear()
        st.success(f"Removed alias `{alias_name}`")
        st.rerun()

st.divider()

# ── Legacy stage transition (deprecated but still functional) ────

with st.expander("⚙️ Stage Transition (legacy)"):
    st.caption(
        "Registry stages are deprecated since MLflow 2.9.0 and will be removed "
        "in a future release — prefer aliases above."
    )
    s1, s2, s3 = st.columns([1, 1, 1])
    with s1:
        stage_version = st.selectbox("Version", version_options, key="stage_version")
    with s2:
        stage = st.selectbox(
            "Stage", ["None", "Staging", "Production", "Archived"], key="stage_name"
        )
    with s3:
        st.markdown("<div style='padding-top:1.8rem'></div>", unsafe_allow_html=True)
        if st.button("Apply", width="stretch"):
            _tracker().transition_model_stage(model_name, stage_version, stage)
            fetch_model_versions.clear()
            st.success(f"v{stage_version} → {stage}")
            st.rerun()
