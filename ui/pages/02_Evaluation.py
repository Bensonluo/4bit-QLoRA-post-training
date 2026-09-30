"""Evaluation — Domain-specific evaluation visualization and comparison."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st

from ui.components.domain_adapters import (
    get_adapter,
    get_domain_display_name,
    list_domains,
    load_eval_data,
)
from ui.config import DOMAINS_DIR, MLFLOW_TRACKING_URI

logger = logging.getLogger("qlora")

st.set_page_config(page_title="评测结果", page_icon="🎯", layout="wide")
st.title("🎯 评测结果")

# ── Domain Selector ─────────────────────────────────────────────

domains = list_domains()
if not domains and DOMAINS_DIR.exists():
    domains = [d.name for d in DOMAINS_DIR.iterdir() if d.is_dir() and not d.name.startswith("_")]

if not domains:
    st.info(
        "暂无评测领域。\n\n"
        "**开始使用：**\n"
        "1. 运行评测脚本（如 `scripts/eval_entity_match.py`，任意领域 test 集通用）\n"
        "2. 或在下方导入历史评测结果"
    )
    with st.expander("📥 导入历史评测结果"):
        if st.button("扫描并导入 MLflow"):
            from src.tracking.eval_logger import log_eval_to_mlflow

            imported = 0
            with st.spinner("正在扫描并导入历史评测结果到 MLflow…"):
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
                            st.warning(f"导入 {json_file.name} 失败：{e}")
            if imported:
                st.success(f"已导入 {imported} 个评测结果文件到 MLflow。")
            else:
                st.info("未发现可导入的评测结果文件。")
    st.stop()

selected_domain = st.selectbox(
    "领域",
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
                mlflow_data.append(
                    {
                        "model": row.get(
                            "tags.model_name", row.get("tags.mlflow.runName", "unknown")
                        ),
                        "source": "mlflow",
                        **{
                            k.replace("metrics.", ""): v
                            for k, v in row.items()
                            if k.startswith("metrics.")
                        },
                    }
                )
except Exception as exc:
    # MLflow is a secondary source — local eval files win, and a missing or
    # unreachable store must not break the page. Keep it visible at debug.
    logger.debug("MLflow eval-run import skipped: %s", exc)

if not data and mlflow_data:
    data = mlflow_data

if not data:
    st.warning("该领域暂无评测数据。")
    if selected_domain == "entity_matching":
        # 通用域空态闭环（R121）：R120 的「用我的 test 集评测」把结果送到这里——
        # 空态必须回答「怎么让这里出现结果」，而不是死胡同「运行领域评测脚本」
        st.markdown("**怎么让这里出现结果？**")
        st.markdown(
            "1. **页内一键**：**训练实验室** → 📋 训练动态 → 已完成训练的"
            "「🧭 下一步」→ **⚡ 用我的 test 集评测**"
            "（用你数据向导（Data Wizard）导出的 test.json，任意领域通用）"
        )
        st.code(
            "python scripts/eval_entity_match.py "
            "--model-path outputs/sft/<run> --test-file outputs/wizard/<dataset>/test.json",
            language="bash",
        )
        st.markdown("2. 在下方导入历史评测结果")
    else:
        st.markdown("**可选操作：**")
        st.markdown("1. 运行领域评测脚本")
        st.markdown("2. 在下方导入历史评测结果")
    with st.expander("📥 导入历史评测结果"):
        if st.button("扫描并导入 MLflow"):
            from src.tracking.eval_logger import log_eval_to_mlflow

            imported = 0
            results_dir = DOMAINS_DIR / selected_domain / "data" / "results"
            with st.spinner("正在扫描并导入历史评测结果到 MLflow…"):
                if results_dir.exists():
                    for json_file in results_dir.glob("eval_detail_*.json"):
                        try:
                            log_eval_to_mlflow(json_file, experiment_name="domain-evaluation")
                            imported += 1
                        except Exception as e:
                            st.warning(f"导入 {json_file.name} 失败：{e}")
            if imported:
                st.success(f"已导入 {imported} 个评测结果文件到 MLflow。")
                st.rerun()
            else:
                st.info("未发现可导入的评测结果文件。")
    st.stop()

adapter = get_adapter(selected_domain)

# ── Overview ────────────────────────────────────────────────────

st.subheader("总览")

if adapter:
    adapter.render_summary(data)
else:
    cols = st.columns(min(len(data), 4))
    for i, model_data in enumerate(data[:4]):
        with cols[i]:
            acc = model_data.get("overall_accuracy", 0)
            st.metric(
                label=model_data.get("model", f"模型 {i + 1}"),
                value=f"{acc:.1%}" if acc else "N/A",
            )

st.divider()

# ── Detailed Charts ─────────────────────────────────────────────

st.subheader("详细分析")

if adapter:
    adapter.render_detail(data)
else:
    st.info("安装领域适配器后可查看详细图表。")

st.divider()

# ── Error Analysis ──────────────────────────────────────────────

if adapter:
    with st.expander("🔍 错误分析", expanded=False):
        adapter.render_error_analysis(data)

st.divider()

# ── Import ──────────────────────────────────────────────────────

with st.expander("📥 导入历史评测结果到 MLflow"):
    if st.button("扫描并导入 MLflow"):
        from src.tracking.eval_logger import log_eval_to_mlflow

        imported = 0
        # 同一「扫描并导入」操作在本页有三处入口(无域分支/无数据分支/此处),
        # 反馈三处同在——只包一处会让最常到达的入口反而裸跑。
        with st.spinner("正在扫描并导入历史评测结果到 MLflow…"):
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
                        st.warning(f"导入 {json_file.name} 失败：{e}")
        if imported:
            st.success(f"已导入 {imported} 个评测结果文件到 MLflow。")
        else:
            st.info("未发现可导入的评测结果文件。")
