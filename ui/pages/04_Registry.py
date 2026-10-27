"""Model Registry — browse versions, manage aliases and legacy stages."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st

from ui.components.shell import display_navigation
from ui.registry_actions import change_alias
from ui.runtime import DISPLAY_ONLY, action_button

if TYPE_CHECKING:
    from src.tracking.mlflow_tracker import MLflowTracker, _NoOpTracker

st.set_page_config(page_title="模型注册表", page_icon="🏛️", layout="wide")
st.title("🏛️ 模型注册表")

source = st.segmented_control(
    "记录来源",
    ["本地注册版本", "公开交付档案"],
    default="公开交付档案" if DISPLAY_ONLY else "本地注册版本",
    selection_mode="single",
    required=True,
    key="registry-source",
)
if source == "公开交付档案":
    from ui.components.delivery_archive import render_delivery_archive

    render_delivery_archive(Path(__file__).resolve().parents[2])
    st.stop()

try:
    from mlflow.tracking import MlflowClient  # noqa: F401  (availability probe)

    from ui.config import MLFLOW_TRACKING_URI
    from ui.queries import fetch_model_versions
except ImportError:
    st.error("本页需要 mlflow：`pip install mlflow`")
    st.stop()


def _tracker() -> MLflowTracker | _NoOpTracker:
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
    if DISPLAY_ONLY:
        display_navigation("展示环境尚未登记模型权重，可浏览已有的业务任务与历史训练记录。")
        st.stop()
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
notice = st.session_state.get("registry-notice")
if notice and notice["model"] == model_name:
    st.caption("最近一次别名操作（提交时核对的结果）")
    getattr(st, notice["level"])(notice["message"])
model_versions = versions_df[versions_df["name"] == model_name].sort_values(
    "version", ascending=False
)


def _has_alias(value: object, name: str) -> bool:
    return isinstance(value, str) and name in {part.strip() for part in value.split(",")}


champion_row = model_versions[
    model_versions["aliases"].map(lambda value: _has_alias(value, "champion"))
]
challenger_row = model_versions[
    model_versions["aliases"].map(lambda value: _has_alias(value, "challenger"))
]

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
version_options = [str(v) for v in model_versions["version"].tolist()]
alias_name = st.selectbox("别名", ["champion", "challenger"], key="alias_name")
bound_rows = model_versions[
    model_versions["aliases"].map(lambda value: _has_alias(value, alias_name))
]
bound_version = str(bound_rows.iloc[0]["version"]) if not bound_rows.empty else None
current_binding = f"v{bound_version}" if bound_version is not None else "未绑定"
st.markdown(f"当前绑定：**{alias_name} → {current_binding}**")
st.code(f"models:/{model_name}@{alias_name}", language=None)


def _change_alias(model: str, alias: str, expected: str | None, target: str | None) -> None:
    # Callback args belong to the page the user actually saw, before its rerun.
    result = change_alias(MLFLOW_TRACKING_URI, model, alias, expected, target)
    st.session_state["registry-notice"] = {
        "model": model,
        "level": result.level,
        "message": result.message,
    }
    fetch_model_versions.clear()


set_column, remove_column = st.columns(2)
with set_column, st.container(border=True):
    st.markdown("#### 设置目标版本")
    alias_version = st.selectbox("目标版本", version_options, key="alias_version")
    st.caption(f"{model_name} · {alias_name}：{current_binding} → v{alias_version}")
    same_binding = alias_version == bound_version
    action_button(
        "🏷 设置",
        width="stretch",
        disabled=same_binding,
        on_click=_change_alias,
        args=(model_name, alias_name, bound_version, alias_version),
    )
    if same_binding:
        st.caption("目标版本已绑定此别名，无需重复设置。")
with remove_column, st.container(border=True):
    st.markdown("#### 移除当前绑定")
    st.caption(f"{model_name} · {alias_name}：{current_binding} → 未绑定")
    st.write("移除的是上方别名的当前绑定；所有模型版本与权重文件都会保留。")
    action_button(
        "🗑 移除",
        width="stretch",
        disabled=bound_version is None,
        on_click=_change_alias,
        args=(model_name, alias_name, bound_version, None),
    )
    if bound_version is None:
        st.caption("此别名尚未绑定版本，无需移除。")

if st.button("刷新注册版本", key="registry-refresh"):
    fetch_model_versions.clear()
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
        if action_button("应用", width="stretch"):
            _tracker().transition_model_stage(model_name, stage_version, stage)
            fetch_model_versions.clear()
            st.success(f"已迁移 v{stage_version} → {stage}")
            st.rerun()
