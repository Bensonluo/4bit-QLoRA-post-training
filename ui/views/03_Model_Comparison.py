"""Model Comparison — Side-by-side comparison with deltas and executive summary."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd
import streamlit as st

from ui.components.archive_navigation import archive_link, render_archive_source
from ui.components.domain_adapters import (
    UnreadableEvalData,
    difficulty_label,
    entity_type_label,
    get_adapter,
    get_domain_display_name,
    list_domains,
    load_eval_data,
)
from ui.components.format import delta as _delta
from ui.components.format import finite_number
from ui.components.format import fmt_num as _fmt_num
from ui.components.format import fmt_pct as _fmt_pct
from ui.components.shell import display_navigation
from ui.config import DOMAINS_DIR
from ui.runtime import DISPLAY_ONLY

st.set_page_config(page_title="模型对比", page_icon="⚖️", layout="wide")
st.title("⚖️ 模型对比")
st.caption("选择两条评测记录，先看指标差值，再检查分组与错误样本。")

if render_archive_source(key="comparison-history", local_label="本地模型对照"):
    st.stop()

# ── Domain Selector ─────────────────────────────────────────────

domains = list_domains()
if not domains and DOMAINS_DIR.exists():
    domains = [d.name for d in DOMAINS_DIR.iterdir() if d.is_dir() and not d.name.startswith("_")]

if not domains:
    archive_link()
    if DISPLAY_ONLY:
        display_navigation("展示环境尚未收录模型对照结果，可先浏览样例任务与历史实验。")
        st.stop()
    # 幽灵分支接线（R128）：adapter 注册表导入时无条件注册两个域，此分支生产不可达
    # （R111 scout 探针 A 双桩可达）；但空态家族范式（R108-R110）要求 st.stop 封页前
    # 必给出路——注册表未来变化时不留死端
    st.info("暂无包含评测数据的领域——先去训练实验室发起一次训练。")
    if st.button("🏋️ 去训练实验室", type="primary"):
        st.switch_page("views/00_Training_Lab.py")
    st.stop()

domain = st.selectbox("领域", domains, format_func=get_domain_display_name)
data = load_eval_data(domain)

if len(data) < 2:
    archive_link()
    if DISPLAY_ONLY:
        display_navigation(
            "当前评测记录不可读，无法判断能否对照。可浏览已有的任务与实验记录。"
            if isinstance(data, UnreadableEvalData)
            else "当前领域不足两条模型评测记录，暂不能对照。可浏览已有的任务与实验记录。"
        )
        st.stop()
    if not data:
        if isinstance(data, UnreadableEvalData):
            st.warning("评测记录无法读取，当前无法判断该领域是否有结果。")
        else:
            st.warning("还没有任何模型的评测结果。")
    else:
        st.warning(
            f"目前只有 1 个模型的评测结果（{data[0].get('model', '模型 0')}），对比至少需要 2 个。"
        )
    # 本页价值先行（R128）：空态先说清「这里能给你什么」——此前这句只存在于首页
    # 旅程网格的格子里（app.py），用户点到本页只看到两行冷文案
    st.markdown(
        "本页做的事：把两个模型的评测结果**并排对比**（总体准确率、MRR、分组明细、成本），"
        "回答「新模型比旧的好多少」。"
    )
    if domain == "entity_matching":
        # 旗舰空态（R128，对齐 02 页 R120/R121 范式）。R111 时代「不宣称结果会出现在
        # 本页」的出口裁决依据已失效：R120 页内评测已核实喂饱本页数据源——
        # 00 页「⚡ 用我的 test 集评测」→ runner.launch_entity_eval →
        # scripts/eval_entity_match.py → 写 domains/entity_matching/data/results/
        # eval_detail_*.json = load_eval_data 所读目录
        st.markdown("**怎么让这里出现对比结果？**")
        st.markdown(
            "1. **训练（下一个）模型**：去训练实验室发起训练（数据可先用数据向导（Data Wizard）从一张表生成）"
        )
        st.markdown(
            "2. **评测它**：训练完成后在 📋 训练动态 → 「🧭 下一步」→"
            "**⚡ 用我的 test 集评测**（用你数据向导（Data Wizard）导出的 test.json，"
            "结果自动进入本页数据源）"
        )
        st.markdown("3. **换一个模型重复 1–2**（比如基座 vs 微调），凑满 2 个评测结果")
        st.code(
            "python scripts/eval_entity_match.py "
            "--model-path outputs/sft/<run> --test-file outputs/wizard/<dataset>/test.json",
            language="bash",
        )
    # 案例域（medical_entity）不加旗舰指路：其评测结果走领域评测脚本路径，指去
    # wizard test 门是失实指路（对齐 02 页 fallback 分支裁决）；共享出口句对两域诚实。
    # 出口裁决 00 不去 02（r111-scout 核实）：02 无发起评测能力，指过去是空→空接力
    st.info("去训练实验室发起（下一个）训练吧。已有的评测结果可在「评测结果」页查看。")
    # st.stop 封页前给出最后一条出路（R108-R110 空态死端范式）：否则用户只能靠侧栏自救
    if st.button("🏋️ 去训练实验室", type="primary"):
        st.switch_page("views/00_Training_Lab.py")
    st.stop()

# ── Select distinct records, including repeated display names ──────────────

models = [d.get("model", f"模型 {i + 1}") for i, d in enumerate(data)]
duplicate_names = len(set(models)) != len(models)


def record_label(index: int) -> str:
    return f"{models[index]}（记录 {index + 1}）" if duplicate_names else models[index]


sel_cols = st.columns(2)
with sel_cols[0]:
    model_a_index = st.selectbox("模型 A", range(len(data)), index=0, format_func=record_label)
with sel_cols[1]:
    model_b_index = st.selectbox(
        "模型 B",
        [i for i in range(len(data)) if i != model_a_index],
        format_func=record_label,
    )
model_a, model_b = data[model_a_index], data[model_b_index]
model_a_name, model_b_name = record_label(model_a_index), record_label(model_b_index)
a_column, b_column = f"A · {model_a_name}", f"B · {model_b_name}"

st.subheader("指标对比")
st.caption(
    "差值为 B − A。准确率、MRR、吞吐越高越好，延迟越低越好；置信度供参考。请核对评测集与运行条件。"
)
comparison_metrics = [
    ("总体准确率", "overall_accuracy", True),
    ("MRR", "mrr", False),
    ("平均置信度", "avg_confidence", False),
    ("平均延迟 (ms)", "avg_latency_ms", False),
    ("吞吐 (样本/秒)", "throughput_per_sec", False),
]
metric_rows = []
for label, key, is_pct in comparison_metrics:
    a, b = model_a.get(key), model_b.get(key)
    formatter = _fmt_pct if is_pct else _fmt_num
    difference = _delta(a, b, pct=is_pct)
    # Accuracy is a difference in percentage points, not relative growth.
    if is_pct and difference is not None:
        difference = difference.replace("%", " 个百分点")
    metric_rows.append(
        {
            "指标": label,
            a_column: formatter(a),
            b_column: formatter(b),
            "差值（B − A）": difference or "—",
        }
    )
st.dataframe(pd.DataFrame(metric_rows), width="stretch", hide_index=True)

breakdown_tab, errors_tab, cost_tab = st.tabs(["分组差异", "错误样本", "成本与记录"])
with breakdown_tab:
    st.subheader("分组明细")
    rows = []
    for key, title, labeler in (
        ("accuracy_by_difficulty", "难度", difficulty_label),
        ("accuracy_by_type", "实体类型", entity_type_label),
    ):
        a_groups, b_groups = model_a.get(key) or {}, model_b.get(key) or {}
        groups = list(dict.fromkeys([*a_groups, *b_groups]))
        for group in groups:
            a, b = a_groups.get(group), b_groups.get(group)
            difference = _delta(a, b, pct=True)
            rows.append(
                {
                    "分组": f"{title} · {labeler(group)}",
                    a_column: _fmt_pct(a),
                    b_column: _fmt_pct(b),
                    "差值（B − A）": difference.replace("%", " 个百分点") if difference else "—",
                }
            )
    if rows:
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    else:
        st.info("所选记录未包含分组指标。")
with errors_tab:
    adapter = get_adapter(domain)
    selected_records = [model_a, model_b]
    if adapter and any(item.get("per_sample") for item in selected_records):
        # Distinguish widget keys when two recorded runs share a model name.
        adapter.render_error_analysis(
            [
                {**model_a, "model": f"A · {model_a_name}"},
                {**model_b, "model": f"B · {model_b_name}"},
            ]
        )
    else:
        st.info("所选记录未包含可查看的逐条预测，无法展示错误样本。")
with cost_tab:
    st.subheader("成本估算")
    st.caption("按记录中的吞吐估算处理时间；未计入模型加载和数据读写。缺少实测值时不作推算。")

    def recorded_time(record: dict) -> str:
        throughput = record.get("throughput_per_sec")
        if finite_number(throughput) and isinstance(throughput, (int, float)) and throughput > 0:
            hours = 1_000_000 / throughput / 3600
            return f"{hours:.1f} 小时" if finite_number(hours) else "—"
        return "—"

    cost_df = pd.DataFrame(
        {
            "指标": ["平均延迟 (ms)", "吞吐 (样本/秒)", "100 万样本耗时"],
            a_column: [
                _fmt_num(model_a.get("avg_latency_ms")),
                _fmt_num(model_a.get("throughput_per_sec")),
                recorded_time(model_a),
            ],
            b_column: [
                _fmt_num(model_b.get("avg_latency_ms")),
                _fmt_num(model_b.get("throughput_per_sec")),
                recorded_time(model_b),
            ],
        }
    )
    st.dataframe(cost_df, width="stretch", hide_index=True)
    with st.expander("领域历史摘要"):
        st.subheader("执行摘要")
        st.caption("领域归档摘要，供补充阅读；可能涉及其他模型与评测批次。")
        results_dir = DOMAINS_DIR / domain / "data" / "results"
        summaries = sorted(results_dir.glob("executive_summary_*.md"), reverse=True)
        if summaries:
            st.markdown(summaries[0].read_text())
        else:
            st.info("未找到执行摘要文件。")
