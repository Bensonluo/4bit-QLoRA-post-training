"""TuneSmith — Dashboard Home."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

from ui.config import MLFLOW_TRACKING_URI, PROJECT_ROOT

st.set_page_config(
    page_title="TuneSmith — 训练工作台",
    page_icon="🔨",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🔨 TuneSmith")
st.caption("配置、训练、评测、对比模型——一站式完成。")

st.subheader("先说业务目标，再看数据")
st.write("提供一份 CSV 样例，让 Agent 帮你判断数据是否适合、还缺什么，并预览真实处理结果。")
if st.button("🧩 分析我的目标与数据", type="primary"):
    st.switch_page("pages/07_Data_Intake.py")

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

        mps = getattr(torch.backends, "mps", None)
        device = (
            "CUDA"
            if torch.cuda.is_available()
            else "MPS"
            if mps is not None and mps.is_available()
            else "CPU"
        )
        st.info(f"PyTorch ({device})", icon="🎮" if device == "CUDA" else "💻")
    except ImportError:
        st.error("PyTorch", icon="❌")

with status_cols[4]:
    st.markdown(
        f"<div style='text-align:right;color:#94A3B8;font-size:0.85rem'>项目：{PROJECT_ROOT.name}</div>",
        unsafe_allow_html=True,
    )

st.divider()

# ── Quick Stats ─────────────────────────────────────────────────

try:
    from ui.queries import fetch_runs

    all_runs = fetch_runs(MLFLOW_TRACKING_URI)

    if not all_runs.empty:
        total = len(all_runs)
        finished = (
            len(all_runs[all_runs["status"] == "FINISHED"]) if "status" in all_runs.columns else 0
        )
        running = (
            len(all_runs[all_runs["status"] == "RUNNING"]) if "status" in all_runs.columns else 0
        )
        failed = (
            len(all_runs[all_runs["status"] == "FAILED"]) if "status" in all_runs.columns else 0
        )
    else:
        total = finished = running = failed = 0
except Exception:
    total = finished = running = failed = 0

stat_cols = st.columns(5)
with stat_cols[0]:
    st.metric("运行总数", total)
with stat_cols[1]:
    st.metric("已完成", finished, delta=None)
with stat_cols[2]:
    st.metric("运行中", running)
with stat_cols[3]:
    st.metric("失败", failed, delta=f"-{failed}" if failed else None)
with stat_cols[4]:
    try:
        from src.tracking.runner import TrainingRunner

        runner = TrainingRunner(project_root=str(PROJECT_ROOT))
        active = runner.list_active()
        st.metric("进行中的任务", len(active))
    except Exception:
        st.metric("进行中的任务", 0)

st.divider()

# ── Quick Actions / 全流程一览（R123）─────────────────────────────
# 旧版只有 00/01/02/03 四钮——05/04/06/07 全靠侧栏摸索；README 已宣称
# 「8 页覆盖全生命周期」，首页理应一屏可达全部 8 页。按旅程顺序排列，
# 每步一句非专家说明（新用户问「从哪开始、一共几步」时本节直接回答）。

st.subheader("快捷操作")
st.caption(
    "按完整旅程排序：数据 → 训练 → 台账 → 评测 → 对比 → 注册 → 对话。首次使用建议从「🧩 目标与数据」开始（本页顶部也可进入）。"
)

# (按钮文案, 一步说明, 跳转目标, 是否主按钮)
_JOURNEY_STEPS = [
    ("🧙 数据向导", "乱写法表格 → 带质检的训练集", "pages/05_Data_Wizard.py", False),
    ("🏋️ 发起训练", "配置 → 预检 → 启动 → 监控", "pages/00_Training_Lab.py", True),
    ("📊 查看实验", "全部运行的历史台账", "pages/01_Experiments.py", False),
    ("🎯 评测结果", "test 集判分 · 难度分层", "pages/02_Evaluation.py", False),
    ("⚖️ 对比模型", "两个模型并排比差值", "pages/03_Model_Comparison.py", False),
    ("🏛️ 模型注册", "版本 · champion/challenger", "pages/04_Registry.py", False),
    ("💬 对话验证", "和你训练的模型聊天", "pages/06_Chat.py", False),
    ("🧩 目标与数据", "说目标给样例，Agent 带着走", "pages/07_Data_Intake.py", False),
]

for row_start in (0, 4):
    qa_cols = st.columns(4)
    for offset, (label, desc, target, primary) in enumerate(
        _JOURNEY_STEPS[row_start : row_start + 4]
    ):
        with qa_cols[offset]:
            if st.button(label, width="stretch", type="primary" if primary else "secondary"):
                st.switch_page(target)
            st.caption(desc)

st.divider()

# ── Recent Activity ─────────────────────────────────────────────

st.subheader("最近动态")

try:
    import pandas as pd

    from ui.queries import fetch_runs

    runs = fetch_runs(MLFLOW_TRACKING_URI).head(5)
    if not runs.empty:
        for _, row in runs.iterrows():
            name = row.get("tags.mlflow.runName", row["run_id"][:8])
            status = row.get("status", "UNKNOWN")
            model = row.get("params.model.name", "—")
            loss = row.get("metrics.train_loss", None)
            start_raw = row.get("start_time", None)
            start = (
                pd.to_datetime(start_raw, unit="ms").strftime("%m-%d %H:%M")
                if start_raw is not None
                else "—"
            )

            status_emoji = {"FINISHED": "✅", "RUNNING": "🟢", "FAILED": "🔴"}.get(status, "⚪")
            loss_str = f" | 损失: {loss:.3f}" if loss is not None else ""

            st.markdown(
                f"<div style='padding:0.5rem 0;border-bottom:1px solid #334155;'>"
                f"<b>{status_emoji} {name}</b> <span style='color:#94A3B8'>— {model}{loss_str}</span>"
                f"<span style='float:right;color:#64748B;font-size:0.8rem'>{start}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )
    else:
        st.info("暂无训练记录。去训练实验室发起第一个实验吧。")
        # 指路句自己接线（R108 06 页范式）：空态用户不必再去找别处的按钮
        if st.button("🏋️ 发起第一个实验", type="primary"):
            st.switch_page("pages/00_Training_Lab.py")
except Exception as e:
    st.info(f"加载最近动态失败：{e}")
