"""Model Registry — browse versions, manage aliases and legacy stages."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st

st.set_page_config(page_title="模型注册表", page_icon="🏛️", layout="wide")
st.title("🏛️ 模型注册表")

try:
    from mlflow.tracking import MlflowClient  # noqa: F401  (availability probe)

    from ui.config import MLFLOW_TRACKING_URI
    from ui.queries import fetch_model_versions
except ImportError:
    st.error("本页需要 mlflow：`pip install mlflow`")
    st.stop()


def _tracker():
    """The repo's tracker abstraction — same construction as registry_cli.py."""
    from config.base import LoggingConfig
    from src.tracking import get_tracker

    cfg = LoggingConfig(use_mlflow=True)
    cfg.mlflow_tracking_uri = MLFLOW_TRACKING_URI
    return get_tracker(cfg)


try:
    versions_df = fetch_model_versions(MLFLOW_TRACKING_URI)
except Exception as exc:
    # 注册表读取失败（存储目录损坏/权限拒绝等）不得打崩整页（R129）：如实报错 +
    # 家族范式出口；不落回「暂无已注册的模型」空态——读不到 ≠ 没有，诚实红线
    st.error(
        f"读取模型注册表失败：{exc}。注册表存储（outputs/mlruns）可能已损坏或无权限，"
        "模型可能仍在——修复目录权限后刷新本页即可。"
    )
    if st.button("🏋️ 去训练实验室", type="primary"):
        st.switch_page("pages/00_Training_Lab.py")
    st.stop()

if versions_df.empty:
    st.info(
        "暂无已注册的模型。最省事的路：去训练实验室，在配置页展开「高级参数」→"
        "「模型注册表」区勾选「训练后自动注册」，训练完成后模型会自动登记到这里。"
        "已训练好的模型也可用命令手动登记：`python scripts/registry_cli.py register`。"
    )
    # st.stop 封页前给出最后一条出路（R108/R109 空态死端范式）：否则用户只能靠侧栏自救
    if st.button("🏋️ 去训练实验室", type="primary"):
        st.switch_page("pages/00_Training_Lab.py")
    st.stop()

# ── Model selector + KPI cards ──────────────────────────────────

model_names = sorted(versions_df["name"].unique())
model_name = st.selectbox("已注册模型", model_names)
model_versions = versions_df[versions_df["name"] == model_name].sort_values(
    "version", ascending=False
)

champion_row = model_versions[model_versions["aliases"].str.contains("champion", na=False)]
challenger_row = model_versions[model_versions["aliases"].str.contains("challenger", na=False)]

k1, k2, k3 = st.columns(3)
with k1:
    st.metric("版本数", len(model_versions))
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
    "别名（champion / challenger）是 MLflow 对注册表阶段（stages）的替代"
    "——stages 自 2.9.0 起已弃用。按别名加载模型："
    f"`models:/{model_name}@champion`"
)

st.dataframe(
    model_versions[["version", "current_stage", "aliases", "status", "created", "run_id"]],
    width="stretch",
    hide_index=True,
)

st.divider()

# ── Alias actions (recommended path) ─────────────────────────────

st.subheader("别名（Aliases）")
a1, a2, a3, a4 = st.columns([1, 1, 1, 1])
version_options = [str(v) for v in model_versions["version"].tolist()]
with a1:
    alias_version = st.selectbox("版本", version_options, key="alias_version")
with a2:
    alias_name = st.selectbox("别名", ["champion", "challenger"], key="alias_name")
with a3:
    if st.button("🏷 设置", width="stretch"):
        _tracker().set_model_alias(model_name, alias_version, alias_name)
        fetch_model_versions.clear()
        st.success(f"已设置别名 `{alias_name}` → v{alias_version}")
        st.rerun()
with a4:
    if st.button("🗑 移除", width="stretch"):
        _tracker().delete_model_alias(model_name, alias_name)
        fetch_model_versions.clear()
        st.success(f"已移除别名 `{alias_name}`")
        st.rerun()

st.divider()

# ── Legacy stage transition (deprecated but still functional) ────

with st.expander("⚙️ 阶段迁移（旧版）"):
    st.caption(
        "注册表阶段（stages）自 MLflow 2.9.0 起已弃用，未来版本将移除——建议优先使用上方的别名管理。"
    )
    s1, s2, s3 = st.columns([1, 1, 1])
    with s1:
        stage_version = st.selectbox("版本", version_options, key="stage_version")
    with s2:
        stage = st.selectbox(
            "阶段", ["None", "Staging", "Production", "Archived"], key="stage_name"
        )
    with s3:
        st.markdown("<div style='padding-top:1.8rem'></div>", unsafe_allow_html=True)
        if st.button("应用", width="stretch"):
            _tracker().transition_model_stage(model_name, stage_version, stage)
            fetch_model_versions.clear()
            st.success(f"已迁移 v{stage_version} → {stage}")
            st.rerun()
