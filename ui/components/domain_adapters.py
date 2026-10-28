"""Domain chart adapter registry for extensible evaluation pages."""

from __future__ import annotations

import json
import math
from abc import ABC, abstractmethod
from collections import Counter
from typing import cast

import streamlit as st

from ui.components.charts import make_grouped_bar, make_scatter_plot
from ui.components.format import finite_number, fmt_num, fmt_pct
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


def _model_labels(data: list[dict]) -> list[str]:
    names = [item.get("model") or f"模型 {index + 1}" for index, item in enumerate(data)]
    counts = Counter(names)
    labels = [
        f"{name}（记录 {index + 1}）" if counts[name] > 1 else name
        for index, name in enumerate(names)
    ]
    if len(set(labels)) != len(labels):
        # A real model name can itself equal a generated disambiguation label.
        # Appending each row's unique index to every label avoids collisions.
        return [f"{name}（记录 {index + 1}）" for index, name in enumerate(names)]
    return labels


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
        labels = _model_labels(data)
        for i, model_data in enumerate(data[:4]):
            with cols[i]:
                st.metric(
                    label=labels[i],
                    value=fmt_pct(model_data.get("overall_accuracy")),
                    delta=f"MRR: {fmt_num(model_data.get('mrr'))} | "
                    f"{fmt_num(model_data.get('avg_latency_ms'))}ms",
                )

    def render_detail(self, data: list[dict]) -> None:
        if not data:
            return

        models = _model_labels(data)
        difficulties = ["easy", "medium", "hard"]

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("按难度分组的准确率")
            values = {}
            for name, d in zip(models, data, strict=True):
                acc = d.get("accuracy_by_difficulty") or {}
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
                for name, d in zip(models, data, strict=True):
                    acc = d.get("accuracy_by_type") or {}
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
            if finite_number(lat) and finite_number(acc)
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
        for index, (model_name, model_data) in enumerate(
            zip(_model_labels(data), data, strict=True)
        ):
            per_sample = model_data.get("per_sample", [])
            if not per_sample:
                continue

            errors = [s for s in per_sample if s.get("correct") is False]
            unknown = [
                s
                for s in per_sample
                if s.get("correct") is not False and s.get("correct") is not True
            ]
            with st.expander(f"{model_name} — {len(errors)} 个错误样本"):
                difficulty_filter = st.selectbox(
                    "按难度筛选",
                    ["All"] + ["easy", "medium", "hard"],
                    key=f"err_{model_name}_{index}_diff",
                    format_func=lambda v: "全部" if v == "All" else difficulty_label(v),
                )
                # Filter options follow the taxonomy present in THIS eval
                # file's errors, not a hardcoded list.
                entity_types_seen = sorted(
                    {
                        value
                        for e in errors
                        if isinstance(value := e.get("entity_type"), str) and value
                    }
                )
                entity_filter = st.selectbox(
                    "按实体类型筛选",
                    ["All"] + entity_types_seen,
                    key=f"err_{model_name}_{index}_entity",
                    format_func=lambda v: "全部" if v == "All" else entity_type_label(v),
                )
                filtered = errors
                if difficulty_filter != "All":
                    filtered = [e for e in filtered if e.get("difficulty") == difficulty_filter]
                if entity_filter != "All":
                    filtered = [e for e in filtered if e.get("entity_type") == entity_filter]

                if len(filtered) > 20:
                    st.caption(f"筛选后共 {len(filtered)} 条，仅展示前 20 条。")
                for err in filtered[:20]:
                    _render_sample(err)
            if unknown:
                with st.expander(f"{model_name} — {len(unknown)} 个状态未知样本"):
                    st.caption("这些记录缺少明确的正确或错误状态，不能计入错误数。")
                    if len(unknown) > 20:
                        st.caption(f"共 {len(unknown)} 条，仅展示前 20 条。")
                    for sample in unknown[:20]:
                        _render_sample(sample, unknown_status=True)


def _render_sample(sample: dict, *, unknown_status: bool = False) -> None:
    # Both queries and model outputs are untrusted data, not markup. Rendering
    # them as text preserves their contents without interpreting HTML or Markdown.
    raw_status = repr(sample["correct"]) if "correct" in sample else "未记录"
    status = f"\n原始状态：{raw_status}" if unknown_status else ""
    difficulty = sample.get("difficulty")
    st.text(
        f"查询：{sample.get('query', '')}\n"
        f"预测：{sample.get('predicted_name', '')}\n"
        f"正确答案：{sample.get('ground_truth', '')}\n"
        f"置信度：{fmt_num(sample.get('confidence'))} | "
        f"难度：{difficulty_label(difficulty) if difficulty is not None else '未记录'}"
        f"{status}"
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


class UnreadableEvalData(list[dict]):
    """No readable result file exists; this is not evidence of zero records."""


def _valid_group_metric(value: object) -> bool:
    if value is None:
        return True
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _valid_eval_data(value: object) -> bool:
    """Only expose records whose containers can be safely displayed.

    Missing metrics and an empty result list are valid; malformed records are
    rejected as a whole so a partial file cannot masquerade as a complete run.
    """
    if not isinstance(value, list):
        return False
    for record in value:
        if not isinstance(record, dict) or not record:
            return False
        if "model" in record and (
            not isinstance(record["model"], str) or not record["model"].strip()
        ):
            return False
        for key in ("accuracy_by_difficulty", "accuracy_by_type"):
            groups = record.get(key)
            if groups is not None:
                if not isinstance(groups, dict):
                    return False
                if any(not _valid_group_metric(value) for value in groups.values()):
                    return False
        samples = record.get("per_sample")
        if samples is not None and (
            not isinstance(samples, list) or not all(isinstance(item, dict) for item in samples)
        ):
            return False
    return True


def load_eval_data(domain: str) -> list[dict]:
    """Load the latest eval_detail_*.json for a domain.

    新→旧遍历，首个可解析者胜出。损坏文件（评测中途被杀的半截 JSON 等）如实
    st.warning（结论→为什么→修法）并跳过，不再让 02/03 整页裸栈（R129）。
    """
    domain_dir = DOMAINS_DIR / domain / "data" / "results"
    if not domain_dir.exists():
        return []
    json_files = sorted(domain_dir.glob("eval_detail_*.json"), reverse=True)
    if not json_files:
        return []
    corrupt: list[str] = []
    for json_file in json_files:
        try:
            with open(json_file, encoding="utf-8") as f:
                data = json.load(f)
            if not _valid_eval_data(data):
                raise ValueError("评测记录结构不完整。")
        except (ValueError, OSError, UnicodeDecodeError):
            corrupt.append(json_file.name)
            continue
        if corrupt:
            st.warning(
                f"较新的评测结果文件损坏，已跳过：{'；'.join(corrupt)}。"
                f"下面显示的是更早一次评测的结果（{json_file.name}）；重新跑一次评测即可恢复最新。"
            )
        return cast(list[dict], data)
    st.warning(
        f"评测结果文件损坏，已全部跳过：{'；'.join(corrupt)}。"
        "多半是评测中途被中断；重新跑一次评测即可恢复。"
    )
    return UnreadableEvalData()


# Register built-in adapters（顺序即 02/03 选择器默认项：通用域在前）
register_adapter(EntityMatchingAdapter())
register_adapter(MedicalEntityAdapter())
