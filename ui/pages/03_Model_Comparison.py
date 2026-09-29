"""Model Comparison — Side-by-side comparison with deltas and executive summary."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd
import streamlit as st

from ui.components.domain_adapters import get_domain_display_name, list_domains, load_eval_data
from ui.components.format import delta as _delta
from ui.components.format import fmt_num as _fmt_num
from ui.components.format import fmt_pct as _fmt_pct
from ui.config import DOMAINS_DIR

st.set_page_config(page_title="模型对比", page_icon="⚖️", layout="wide")
st.title("⚖️ 模型对比")

# ── Domain Selector ─────────────────────────────────────────────

domains = list_domains()
if not domains and DOMAINS_DIR.exists():
    domains = [d.name for d in DOMAINS_DIR.iterdir() if d.is_dir() and not d.name.startswith("_")]

if not domains:
    st.info("暂无包含评测数据的领域。")
    st.stop()

domain = st.selectbox("领域", domains, format_func=get_domain_display_name)
data = load_eval_data(domain)

if len(data) < 2:
    st.warning("对比至少需要 2 个模型的评测结果，请先完成更多评测。")
    st.stop()

# ── Model Selectors ─────────────────────────────────────────────

models = [d.get("model", f"模型 {i}") for i, d in enumerate(data)]

sel_cols = st.columns(2)
with sel_cols[0]:
    model_a_name = st.selectbox("模型 A", models, index=0)
with sel_cols[1]:
    model_b_name = st.selectbox("模型 B", models, index=min(1, len(models) - 1))

model_a = next((d for d in data if d.get("model") == model_a_name), data[0])
model_b = next((d for d in data if d.get("model") == model_b_name), data[min(1, len(data) - 1)])

st.divider()

# ── Side-by-Side Cards ──────────────────────────────────────────

st.subheader("指标对比")

comparison_metrics = [
    ("总体准确率", "overall_accuracy", True),
    ("MRR", "mrr", False),
    ("平均置信度", "avg_confidence", False),
    ("平均延迟 (ms)", "avg_latency_ms", False),
    ("吞吐 (样本/秒)", "throughput_per_sec", False),
]

# Model A card
with st.container(border=True):
    st.markdown(f"### {model_a_name}")
    a_cols = st.columns(len(comparison_metrics))
    for i, (label, key, is_pct) in enumerate(comparison_metrics):
        with a_cols[i]:
            val = model_a.get(key)
            if is_pct:
                st.metric(label, _fmt_pct(val))
            else:
                st.metric(label, _fmt_num(val))

# Delta row
st.markdown(
    "<div style='text-align:center;font-size:1.5rem;padding:0.5rem 0'>↓ 差值（B − A）</div>",
    unsafe_allow_html=True,
)

with st.container(border=True):
    st.markdown(f"### {model_b_name}")
    b_cols = st.columns(len(comparison_metrics))
    for i, (label, key, is_pct) in enumerate(comparison_metrics):
        with b_cols[i]:
            val_b = model_b.get(key)
            val_a = model_a.get(key)
            if is_pct:
                st.metric(label, _fmt_pct(val_b), delta=_delta(val_a, val_b, pct=True))
            else:
                st.metric(label, _fmt_num(val_b), delta=_delta(val_a, val_b, pct=False))

st.divider()

# ── Breakdowns ──────────────────────────────────────────────────

st.subheader("分组明细")

b1, b2 = st.columns(2)

with b1:
    st.markdown("**按难度分组的准确率**")
    difficulties = ["easy", "medium", "hard"]
    diff_cols = st.columns(len(difficulties))
    for i, diff in enumerate(difficulties):
        with diff_cols[i]:
            acc_a = model_a.get("accuracy_by_difficulty", {}).get(diff)
            acc_b = model_b.get("accuracy_by_difficulty", {}).get(diff)
            st.metric(f"{diff.title()}", _fmt_pct(acc_b), delta=_delta(acc_a, acc_b, pct=True))

with b2:
    st.markdown("**按实体类型分组的准确率**")
    entity_types = list(model_a.get("accuracy_by_type", {}).keys()) or ["drug", "hospital"]
    ent_cols = st.columns(len(entity_types))
    for i, etype in enumerate(entity_types):
        with ent_cols[i]:
            acc_a = model_a.get("accuracy_by_type", {}).get(etype)
            acc_b = model_b.get("accuracy_by_type", {}).get(etype)
            st.metric(f"{etype.title()}", _fmt_pct(acc_b), delta=_delta(acc_a, acc_b, pct=True))

st.divider()

# ── Executive Summary ───────────────────────────────────────────

st.subheader("执行摘要")
results_dir = DOMAINS_DIR / domain / "data" / "results"
if results_dir.exists():
    summaries = sorted(results_dir.glob("executive_summary_*.md"), reverse=True)
    if summaries:
        with open(summaries[0]) as f:
            st.markdown(f.read())
    else:
        st.info("未找到执行摘要文件。")
else:
    st.info("未找到结果目录。")

st.divider()

# ── Cost Estimation ─────────────────────────────────────────────

st.subheader("成本估算")

latency_a = model_a.get("avg_latency_ms", 0)
latency_b = model_b.get("avg_latency_ms", 0)
tp_a = model_a.get("throughput_per_sec", 1)
tp_b = model_b.get("throughput_per_sec", 1)

cost_df = pd.DataFrame(
    {
        "指标": [
            "平均延迟",
            "吞吐",
            "100 万样本耗时",
            "部署方式",
            "数据安全",
        ],
        model_a_name: [
            f"{latency_a:.0f} ms",
            f"{tp_a:.0f} 样本/秒",
            f"{1_000_000 / max(tp_a, 0.1) / 3600:.1f} 小时",
            "本地 GPU" if latency_a > 0 else "不适用",
            "本地部署",
        ],
        model_b_name: [
            f"{latency_b:.0f} ms",
            f"{tp_b:.0f} 样本/秒",
            f"{1_000_000 / max(tp_b, 0.1) / 3600:.1f} 小时",
            "本地 GPU" if latency_b > 0 else "不适用",
            "本地部署",
        ],
    }
)
st.dataframe(cost_df, width="stretch", hide_index=True)
