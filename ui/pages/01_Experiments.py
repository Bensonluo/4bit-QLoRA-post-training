"""Experiments — Browse and compare all training/eval runs."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st

st.set_page_config(page_title="实验记录", page_icon="📊", layout="wide")
st.title("📊 实验记录")

try:
    import mlflow  # noqa: F401  (still needed: delete_run + metric history)
    import pandas as pd

    from ui.config import MLFLOW_TRACKING_URI
    from ui.queries import fetch_experiments, fetch_runs
except ImportError:
    st.error("本页需要 mlflow：`pip install mlflow`")
    st.stop()

# Fetch all runs (cached 30s — see ui/queries.py)
all_experiments = fetch_experiments(MLFLOW_TRACKING_URI)
exp_id_by_name = {name: eid for eid, name in all_experiments}
runs = fetch_runs(MLFLOW_TRACKING_URI)

# MLflow 状态枚举 → 中文显示(选项值仍为原始枚举,筛选逻辑不动)
_STATUS_ZH = {"FINISHED": "已完成", "RUNNING": "运行中", "FAILED": "失败"}


def _status_label(value: str) -> str:
    return _STATUS_ZH.get(str(value), str(value))


if runs.empty:
    st.info("暂无实验记录。请先在训练实验室（Training Lab）发起训练。")
    st.stop()

# ── KPI Cards ───────────────────────────────────────────────────

total = len(runs)
finished = len(runs[runs["status"] == "FINISHED"]) if "status" in runs.columns else 0
running = len(runs[runs["status"] == "RUNNING"]) if "status" in runs.columns else 0
failed = len(runs[runs["status"] == "FAILED"]) if "status" in runs.columns else 0

kpi = st.columns(4)
with kpi[0]:
    st.metric("运行总数", total)
with kpi[1]:
    st.metric("已完成", finished)
with kpi[2]:
    st.metric("运行中", running)
with kpi[3]:
    st.metric("失败", failed, delta=f"-{failed}" if failed else None)

st.divider()

# ── Filters ─────────────────────────────────────────────────────

with st.expander("🔍 筛选", expanded=False):
    f1, f2, f3 = st.columns(3)
    with f1:
        statuses = runs["status"].unique().tolist() if "status" in runs.columns else []
        selected_status = st.multiselect(
            "状态", statuses, default=statuses, format_func=_status_label
        )
    with f2:
        model_vals = (
            runs["params.model.name"].dropna().unique().tolist()
            if "params.model.name" in runs.columns
            else []
        )
        selected_models = st.multiselect("模型", model_vals, default=[])
    with f3:
        exp_names = [name for _, name in all_experiments]
        selected_exps = st.multiselect("实验", exp_names, default=exp_names)

    if selected_status and "status" in runs.columns:
        runs = runs[runs["status"].isin(selected_status)]
    if selected_models and "params.model.name" in runs.columns:
        runs = runs[runs["params.model.name"].isin(selected_models)]
    if selected_exps and "experiment_id" in runs.columns:
        wanted_ids = [exp_id_by_name[n] for n in selected_exps if n in exp_id_by_name]
        runs = runs[runs["experiment_id"].isin(wanted_ids)]

st.subheader(f"全部运行（{len(runs)}）")

# ── Runs Table ──────────────────────────────────────────────────

display_map = {
    "tags.mlflow.runName": "运行名",
    "status": "状态",
    "start_time": "开始时间",
    "params.model.name": "模型",
    "params.training.num_epochs": "轮数",
    "params.training.learning_rate": "学习率",
    "params.lora.r": "LoRA r",
    "metrics.train_loss": "训练损失",
    "metrics.eval/eval_loss": "验证损失",
    "params.data.dataset_name": "数据集",
}
display_cols = {k: v for k, v in display_map.items() if k in runs.columns}
if display_cols:
    df_display = runs[list(display_cols.keys())].rename(columns=display_cols)
    if "状态" in df_display.columns:
        df_display["状态"] = df_display["状态"].map(_STATUS_ZH).fillna(df_display["状态"])
    st.dataframe(df_display, width="stretch", hide_index=True)
else:
    st.dataframe(runs.head(20), width="stretch", hide_index=True)

st.divider()

# ── Compare Runs ────────────────────────────────────────────────

st.subheader("对比运行")

run_name_col = "tags.mlflow.runName" if "tags.mlflow.runName" in runs.columns else "run_id"
run_names = runs[run_name_col].tolist()
selected = st.multiselect("选择 2–5 个运行进行对比", run_names, max_selections=5)

if len(selected) >= 2:
    selected_runs = runs[runs[run_name_col].isin(selected)]

    # Metric comparison
    metric_cols = [c for c in runs.columns if c.startswith("metrics.")]
    if metric_cols:
        compare_metrics = st.multiselect("指标", metric_cols, default=metric_cols[:3])
        if compare_metrics:
            models = selected_runs[run_name_col].tolist()
            metrics_data = {}
            for mc in compare_metrics:
                # None (not 0.0) for missing metrics — never fabricate data points.
                metrics_data[mc.replace("metrics.", "")] = [
                    None if pd.isna(v) else float(v) for v in selected_runs[mc].tolist()
                ]
            from ui.components.charts import make_bar_comparison

            fig = make_bar_comparison(models, metrics_data, "指标对比")
            st.plotly_chart(fig, width="stretch")

    # Param diff
    param_cols = [c for c in runs.columns if c.startswith("params.")]
    if param_cols:
        with st.expander("参数差异"):
            diff_data = {}
            for pc in param_cols:
                vals = selected_runs[pc].tolist()
                if len(set(str(v) for v in vals)) > 1:
                    diff_data[pc.replace("params.", "")] = vals
            if diff_data:
                diff_df = pd.DataFrame(diff_data, index=selected_runs[run_name_col].tolist())
                st.dataframe(diff_df.T, width="stretch")
            else:
                st.info("所选运行的参数完全相同。")

st.divider()

# ── Run Details ─────────────────────────────────────────────────

st.subheader("运行详情")
selected_run = st.selectbox("选择一个运行", run_names)
if selected_run:
    run_row = runs[runs[run_name_col] == selected_run].iloc[0]
    run_id = run_row["run_id"]

    d1, d2 = st.columns(2)
    with d1, st.expander("参数"):
        params = {
            k.replace("params.", ""): v for k, v in run_row.items() if k.startswith("params.")
        }
        if params:
            st.dataframe(
                pd.DataFrame(list(params.items()), columns=["参数", "值"]),
                width="stretch",
                hide_index=True,
            )
    with d2, st.expander("指标"):
        metrics = {
            k.replace("metrics.", ""): v for k, v in run_row.items() if k.startswith("metrics.")
        }
        if metrics:
            st.dataframe(
                pd.DataFrame(list(metrics.items()), columns=["指标", "值"]),
                width="stretch",
                hide_index=True,
            )

    with st.expander("指标曲线"):
        try:
            from ui.queries import fetch_metric_history_by_run_id

            metric_names = [
                c.replace("metrics.", "") for c in runs.columns if c.startswith("metrics.")
            ]
            if metric_names:
                default_idx = metric_names.index("loss") if "loss" in metric_names else 0
                chosen = st.selectbox("指标", metric_names, index=default_idx)
                history = fetch_metric_history_by_run_id(MLFLOW_TRACKING_URI, run_id, chosen)
                if history:
                    from ui.components.charts import make_metric_timeseries

                    fig = make_metric_timeseries({selected_run: history}, chosen)
                    st.plotly_chart(fig, width="stretch")
        except Exception as e:
            st.warning(f"加载指标历史失败：{e}")

    # 单运行删除:软删除(入 .trash,UI 无恢复入口)——必须二次确认,
    # 00 页 Activity 删除的 popover 范式(caption 如实说明影响范围)。
    with st.popover("🗑 删除此运行", use_container_width=True):
        st.caption(
            "软删除：运行记录将从界面移除（底层进入 MLflow 回收站 .trash），"
            "config 与日志文件不受影响。"
        )
        if st.button("确认删除", type="primary"):
            mlflow.delete_run(run_id)
            fetch_runs.clear()  # 30s TTL 缓存不清会让被删运行残留,与 caption 承诺矛盾(04 页 idiom)
            st.success("已删除。")
            st.rerun()
