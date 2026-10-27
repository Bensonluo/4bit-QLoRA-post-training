"""Experiments — Browse and compare all training/eval runs."""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st

from ui.components.archive_navigation import archive_link, render_archive_source
from ui.components.shell import display_navigation
from ui.runtime import DISPLAY_ONLY, action_button

st.set_page_config(page_title="实验记录", page_icon="📊", layout="wide")
st.title("📊 实验记录")

if render_archive_source(key="experiments-history", local_label="本地实验"):
    st.stop()

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
    archive_link()
    if DISPLAY_ONLY:
        display_navigation(
            "展示环境暂无实验记录，可先浏览示例任务的数据与核验过程。", experiments=False
        )
        st.stop()
    st.info("暂无实验记录。请先在训练实验室（Training Lab）发起训练。")
    # st.stop 封页前给出最后一条出路（R108 06 页范式）：否则用户只能靠侧栏自救
    if st.button("🏋️ 去训练实验室", type="primary"):
        st.switch_page("views/00_Training_Lab.py")
    st.stop()

if (
    "run_id" not in runs.columns
    or not runs["run_id"].map(lambda value: isinstance(value, str) and bool(value.strip())).all()
    or runs["run_id"].duplicated().any()
):
    st.error("实验记录缺少唯一的运行 ID，无法可靠选择详情或操作；请检查实验存储。")
    archive_link()
    st.stop()

all_run_ids = runs["run_id"].tolist()
names = {
    row["run_id"]: name.strip() if isinstance(name, str) and name.strip() else "未命名运行"
    for _, row in runs.iterrows()
    for name in [row.get("tags.mlflow.runName")]
}
name_counts = Counter(names.values())
short_ids = {identity: identity[:8] for identity in all_run_ids}
while True:
    prefix_counts = Counter(short_ids.values())
    collisions = {prefix for prefix, count in prefix_counts.items() if count > 1}
    if not collisions:
        break
    short_ids = {
        identity: identity[: len(prefix) + 1] if prefix in collisions else prefix
        for identity, prefix in short_ids.items()
    }
run_labels = {}
for identity in all_run_ids:
    name = names[identity]
    run_labels[identity] = (
        f"{name}（{short_ids[identity]}）"
        if name_counts[name] > 1 or name == "未命名运行"
        else name
    )


def run_label(identity: str) -> str:
    return run_labels[identity]


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
if runs.empty:
    st.info("没有符合筛选条件的实验记录，请调整状态、模型或实验筛选。")
    st.stop()

# ── Runs Table ──────────────────────────────────────────────────

display_map = {
    "tags.mlflow.runName": "运行名",
    "run_id": "运行 ID",
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

run_ids = runs["run_id"].tolist()
run_rows = runs.set_index("run_id", drop=False)
selected = st.multiselect(
    "选择 2–5 个运行进行对比", run_ids, format_func=run_label, max_selections=5
)

if len(selected) >= 2:
    selected_runs = run_rows.loc[selected]

    # Metric comparison
    metric_cols = [c for c in runs.columns if c.startswith("metrics.")]
    if metric_cols:
        compare_metrics = st.multiselect("指标", metric_cols, default=metric_cols[:3])
        if compare_metrics:
            models = [run_label(identity) for identity in selected]
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
                diff_df = pd.DataFrame(
                    diff_data, index=[run_label(identity) for identity in selected]
                )
                st.dataframe(diff_df.T, width="stretch")
            else:
                st.info("所选运行的参数完全相同。")

st.divider()

# ── Run Details ─────────────────────────────────────────────────

st.subheader("运行详情")
selected_run = st.selectbox("选择一个运行", run_ids, format_func=run_label)
if selected_run:
    run_row = run_rows.loc[selected_run]
    run_id = selected_run
    source = run_row.get("tags.evidence.source")
    source_label = "公开历史验证" if isinstance(source, str) and source.strip() else "本地实验"
    st.caption(f"{run_label(run_id)} · {source_label} · 运行 ID：{run_id}")

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

                    fig = make_metric_timeseries({run_label(selected_run): history}, chosen)
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
        if action_button("确认删除", type="primary"):
            mlflow.delete_run(run_id)
            fetch_runs.clear()  # 30s TTL 缓存不清会让被删运行残留,与 caption 承诺矛盾(04 页 idiom)
            st.success("已删除。")
            st.rerun()
