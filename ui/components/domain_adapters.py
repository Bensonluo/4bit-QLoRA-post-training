"""Domain chart adapter registry for extensible evaluation pages."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod

import streamlit as st

from ui.components.charts import make_grouped_bar, make_scatter_plot
from ui.components.format import fmt_num, fmt_pct
from ui.config import DOMAINS_DIR

# 难度/实体类型枚举 → 中文显示映射(数据值不动,只动显示层;03 页共用)。
# 实体类型是开放分类(drug/hospital/device/…)——未收录的类型原样回退。
_DIFFICULTY_ZH = {"easy": "简单", "medium": "中等", "hard": "困难"}
_ENTITY_TYPE_ZH = {"drug": "药品", "hospital": "医院"}


def difficulty_label(value: str) -> str:
    """难度枚举 → 中文显示;未知值原样返回(数据契约不动)。"""
    return _DIFFICULTY_ZH.get(str(value), str(value))


def entity_type_label(value: str) -> str:
    """实体类型枚举 → 中文显示;开放分类,未收录原样返回。"""
    return _ENTITY_TYPE_ZH.get(str(value), str(value))


def _entity_types_in(data: list[dict]) -> list[str]:
    """Union of entity types across models, in first-seen order.

    Entity types are an open taxonomy (drug, hospital, device, ...) — chart
    axes and filters must follow the eval file instead of a hardcoded list.
    """
    seen: dict[str, None] = {}
    for model_data in data:
        seen.update(dict.fromkeys(model_data.get("accuracy_by_type") or {}))
    return list(seen)


class DomainChartAdapter(ABC):
    """Base class for domain-specific evaluation visualization."""

    domain_name: str = ""
    display_name: str = ""

    @abstractmethod
    def render_summary(self, data: list[dict]) -> None:
        """Render overview metric cards."""

    @abstractmethod
    def render_detail(self, data: list[dict]) -> None:
        """Render detailed charts."""

    @abstractmethod
    def render_error_analysis(self, data: list[dict]) -> None:
        """Render per-sample error table."""


class MedicalEntityAdapter(DomainChartAdapter):
    domain_name = "medical_entity"
    display_name = "医疗实体匹配"

    def render_summary(self, data: list[dict]) -> None:
        if not data:
            st.info("暂无评测数据。")
            return

        cols = st.columns(min(len(data), 4))
        for i, model_data in enumerate(data[:4]):
            with cols[i]:
                st.metric(
                    label=model_data.get("model", f"模型 {i + 1}"),
                    value=fmt_pct(model_data.get("overall_accuracy")),
                    delta=f"MRR: {fmt_num(model_data.get('mrr'))} | "
                    f"{fmt_num(model_data.get('avg_latency_ms'))}ms",
                )

    def render_detail(self, data: list[dict]) -> None:
        if not data:
            return

        models = [d.get("model", f"模型 {i}") for i, d in enumerate(data)]
        difficulties = ["easy", "medium", "hard"]

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("按难度分组的准确率")
            values = {}
            for d in data:
                name = d.get("model", "")
                acc = d.get("accuracy_by_difficulty", {})
                # Missing bucket → None (rendered as a gap), not a 0-height bar.
                values[name] = [acc.get(diff) for diff in difficulties]
            fig = make_grouped_bar(
                [difficulty_label(d) for d in difficulties], models, values, "难度", "准确率"
            )
            st.plotly_chart(fig, width="stretch")

        with col2:
            st.subheader("按实体类型分组的准确率")
            entity_types = _entity_types_in(data)
            if entity_types:
                values2 = {}
                for d in data:
                    name = d.get("model", "")
                    acc = d.get("accuracy_by_type", {})
                    values2[name] = [acc.get(et) for et in entity_types]
                fig2 = make_grouped_bar(
                    [entity_type_label(et) for et in entity_types],
                    models,
                    values2,
                    "实体类型",
                    "准确率",
                )
                st.plotly_chart(fig2, width="stretch")
            else:
                st.info("最新评测文件中没有实体类型分组数据。")

        # Latency vs Accuracy scatter — only models reporting BOTH metrics; a
        # missing value would otherwise plot a fabricated (0, 0) point.
        st.subheader("延迟 vs 准确率")
        points = [
            (d.get("avg_latency_ms"), d.get("overall_accuracy"), models[i])
            for i, d in enumerate(data)
        ]
        points = [
            (lat, acc, name)
            for lat, acc, name in points
            if isinstance(lat, (int, float)) and isinstance(acc, (int, float))
        ]
        if points:
            latencies, accuracies, names = zip(*points)
            fig3 = make_scatter_plot(
                list(latencies), list(accuracies), list(names), "平均延迟 (ms)", "准确率"
            )
            st.plotly_chart(fig3, width="stretch")
        else:
            st.info("暂无模型同时报告延迟与准确率。")

    def render_error_analysis(self, data: list[dict]) -> None:
        # 无独立 subheader:唯一消费方是 02 页「🔍 错误分析」expander,
        # 内层再立标题会连续重复两行「错误分析」。
        for model_data in data:
            model_name = model_data.get("model", "未知")
            per_sample = model_data.get("per_sample", [])
            if not per_sample:
                continue

            errors = [s for s in per_sample if not s.get("correct", True)]
            with st.expander(f"{model_name} — {len(errors)} 个错误样本"):
                difficulty_filter = st.selectbox(
                    "按难度筛选",
                    ["All"] + ["easy", "medium", "hard"],
                    key=f"err_{model_name}_diff",
                    format_func=lambda v: "全部" if v == "All" else difficulty_label(v),
                )
                # Filter options follow the taxonomy present in THIS eval
                # file's errors, not a hardcoded list.
                entity_types_seen = sorted({e.get("entity_type", "") for e in errors} - {""})
                entity_filter = st.selectbox(
                    "按实体类型筛选",
                    ["All"] + entity_types_seen,
                    key=f"err_{model_name}_entity",
                    format_func=lambda v: "全部" if v == "All" else entity_type_label(v),
                )
                filtered = errors
                if difficulty_filter != "All":
                    filtered = [e for e in filtered if e.get("difficulty") == difficulty_filter]
                if entity_filter != "All":
                    filtered = [e for e in filtered if e.get("entity_type") == entity_filter]

                for err in filtered[:20]:
                    color = "green" if err.get("correct") else "red"
                    st.markdown(
                        f"**查询：**`{err.get('query', '')}` | "
                        f"<span style='color:{color}'>"
                        f"预测：`{err.get('predicted_name', '')}` | "
                        f"正确答案：`{err.get('ground_truth', '')}`"
                        f"</span> | "
                        f"置信度：{fmt_num(err.get('confidence'))} | "
                        f"难度：{difficulty_label(err.get('difficulty', ''))}",
                        unsafe_allow_html=True,
                    )


_REGISTRY: dict[str, DomainChartAdapter] = {}


class EntityMatchingAdapter(MedicalEntityAdapter):
    """通用实体匹配域（R120 评测闭环）：任意领域向导 test 集的评测结果。

    图表逻辑与 medical 案例完全同构（子类复用，非复制分叉）——评测页消费
    的 eval_detail 契约对两者一致。registered 在 medical 之前 → 02/03 域
    选择器默认落通用域（generic-first 门面）。
    """

    domain_name = "entity_matching"
    display_name = "实体匹配（通用）"


def register_adapter(adapter: DomainChartAdapter) -> None:
    _REGISTRY[adapter.domain_name] = adapter


def get_adapter(domain: str) -> DomainChartAdapter | None:
    return _REGISTRY.get(domain)


def list_domains() -> list[str]:
    return list(_REGISTRY.keys())


def get_domain_display_name(domain: str) -> str:
    adapter = _REGISTRY.get(domain)
    return adapter.display_name if adapter else domain


def load_eval_data(domain: str) -> list[dict]:
    """Load the latest eval_detail_*.json for a domain."""
    domain_dir = DOMAINS_DIR / domain / "data" / "results"
    if not domain_dir.exists():
        return []
    json_files = sorted(domain_dir.glob("eval_detail_*.json"), reverse=True)
    if not json_files:
        return []
    with open(json_files[0]) as f:
        return json.load(f)


# Register built-in adapters（顺序即 02/03 选择器默认项：通用域在前）
register_adapter(EntityMatchingAdapter())
register_adapter(MedicalEntityAdapter())
