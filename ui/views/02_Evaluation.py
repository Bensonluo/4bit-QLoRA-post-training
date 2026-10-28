"""Evaluation — Domain-specific evaluation visualization and comparison."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st

from ui.components.archive_navigation import archive_link, render_archive_source
from ui.components.domain_adapters import (
    UnreadableEvalData,
    get_adapter,
    get_domain_display_name,
    list_domains,
    load_eval_data,
)
from ui.components.format import fmt_pct
from ui.components.shell import display_navigation
from ui.config import DOMAINS_DIR, MLFLOW_TRACKING_URI
from ui.runtime import DISPLAY_ONLY, action_button

logger = logging.getLogger("qlora")

st.set_page_config(page_title="评测结果", page_icon="🎯", layout="wide")
st.title("🎯 评测结果")
st.caption("查看评测指标、分组差异与错误样本；业务验收在对应任务中完成。")

if render_archive_source(key="evaluation-history", local_label="本地领域评测"):
    st.stop()

# ── Domain Selector ─────────────────────────────────────────────

domains = list_domains()
if not domains and DOMAINS_DIR.exists():
    domains = [d.name for d in DOMAINS_DIR.iterdir() if d.is_dir() and not d.name.startswith("_")]

if not domains:
    archive_link()
    if DISPLAY_ONLY:
        display_navigation("展示环境尚未收录领域评测结果，可先浏览样例任务与历史实验。")
        st.stop()
    st.info(
        "暂无评测领域。\n\n"
        "**开始使用：**\n"
        "1. 运行评测脚本（如 `scripts/eval_entity_match.py`，任意领域 test 集通用）\n"
        "2. 或在下方导入历史评测结果"
    )
    with st.expander("📥 导入历史评测结果"):
        if action_button("扫描并导入 MLflow"):
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
                            log_eval_to_mlflow(
                                json_file,
                                experiment_name="domain-evaluation",
                                domain=domain_dir.name,
                            )
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
local_results_unreadable = isinstance(data, UnreadableEvalData)

# Also try MLflow
mlflow_data: list[dict] = []
mlflow_unavailable = False
mlflow_unattributed = 0
try:
    import mlflow

    if DISPLAY_ONLY:
        from ui.tracking_readonly import inspect_tracking_uri

        tracking_state = inspect_tracking_uri(MLFLOW_TRACKING_URI)
    else:
        tracking_state = "ready"
    if tracking_state == "ready":
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        exp = mlflow.get_experiment_by_name("domain-evaluation")
        if exp:
            runs = mlflow.search_runs(experiment_ids=[exp.experiment_id])
            if not runs.empty:
                for _, row in runs.iterrows():
                    run_domain = row.get("tags.eval_domain")
                    if not isinstance(run_domain, str) or not run_domain.strip():
                        mlflow_unattributed += 1
                        continue
                    if run_domain != selected_domain:
                        continue
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
    if DISPLAY_ONLY:
        mlflow_unavailable = True
        st.warning("本地实验存储暂无法在展示环境安全读取；已有评测文件仍可查看。")

if not data:
    if mlflow_unattributed:
        st.warning(
            f"有 {mlflow_unattributed} 条实验评测记录未标明领域，无法确认是否属于当前领域，"
            "因此未显示；可从原始评测文件重新导入。"
        )
    if mlflow_data:
        data = mlflow_data
        st.caption("当前显示与所选领域匹配的 MLflow 评测指标；逐条预测请查看原始评测文件。")

if not data:
    if DISPLAY_ONLY:
        display_navigation(
            "无法确认该领域是否有本地实验评测记录。样例任务与历史训练记录仍可浏览。"
            if mlflow_unavailable or local_results_unreadable or mlflow_unattributed
            else "该领域尚未收录评测结果。样例任务与历史训练记录仍可浏览。"
        )
        st.stop()
    if local_results_unreadable or mlflow_unattributed:
        st.warning("当前无法确认该领域是否有评测结果。")
    else:
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
        if action_button("扫描并导入 MLflow"):
            from src.tracking.eval_logger import log_eval_to_mlflow

            imported = 0
            results_dir = DOMAINS_DIR / selected_domain / "data" / "results"
            with st.spinner("正在扫描并导入历史评测结果到 MLflow…"):
                if results_dir.exists():
                    for json_file in results_dir.glob("eval_detail_*.json"):
                        try:
                            log_eval_to_mlflow(
                                json_file,
                                experiment_name="domain-evaluation",
                                domain=selected_domain,
                            )
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
            st.metric(
                label=model_data.get("model", f"模型 {i + 1}"),
                value=fmt_pct(model_data.get("overall_accuracy")),
            )

results_tab, errors_tab, records_tab = st.tabs(["指标与分组", "错误样本", "评测记录管理"])
with results_tab:
    st.subheader("详细分析")
    if adapter:
        adapter.render_detail(data)
    else:
        st.info("该领域暂无分组图表，可查看上方指标与原始评测记录。")
with errors_tab:
    if adapter:
        if not any(item.get("per_sample") for item in data):
            st.info("当前评测记录未包含逐条预测，无法展示错误样本。")
        else:
            adapter.render_error_analysis(data)
    else:
        st.info("该领域尚无错误样本视图。")
with records_tab:
    st.caption("已有评测结果可导入实验记录；导入不会重新运行模型评测。")
    with st.expander("📥 导入历史评测结果到 MLflow"):
        if action_button("扫描并导入 MLflow"):
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
                            log_eval_to_mlflow(
                                json_file,
                                experiment_name="domain-evaluation",
                                domain=domain_dir.name,
                            )
                            imported += 1
                        except Exception as e:
                            st.warning(f"导入 {json_file.name} 失败：{e}")
            if imported:
                st.success(f"已导入 {imported} 个评测结果文件到 MLflow。")
            else:
                st.info("未发现可导入的评测结果文件。")
