"""Domain chart adapter registry for extensible evaluation pages."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional

import streamlit as st
import plotly.graph_objects as go

from ui.config import DOMAINS_DIR, CHART_COLORS
from ui.components.charts import make_grouped_bar, make_calibration_chart, make_scatter_plot


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
    display_name = "Medical Entity Matching"

    def render_summary(self, data: list[dict]) -> None:
        if not data:
            st.info("No evaluation data available.")
            return

        cols = st.columns(min(len(data), 4))
        for i, model_data in enumerate(data[:4]):
            with cols[i]:
                acc = model_data.get("overall_accuracy", 0)
                mrr = model_data.get("mrr", 0)
                latency = model_data.get("avg_latency_ms", 0)
                st.metric(
                    label=model_data.get("model", f"Model {i+1}"),
                    value=f"{acc:.1%}",
                    delta=f"MRR: {mrr:.3f} | {latency:.0f}ms",
                )

    def render_detail(self, data: list[dict]) -> None:
        if not data:
            return

        models = [d.get("model", f"Model {i}") for i, d in enumerate(data)]
        difficulties = ["easy", "medium", "hard"]

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Accuracy by Difficulty")
            values = {}
            for d in data:
                name = d.get("model", "")
                acc = d.get("accuracy_by_difficulty", {})
                values[name] = [acc.get(diff, 0) for diff in difficulties]
            fig = make_grouped_bar(difficulties, models, values, "Difficulty", "Accuracy")
            st.plotly_chart(fig, width='stretch')

        with col2:
            st.subheader("Accuracy by Entity Type")
            entity_types = ["drug", "hospital"]
            values2 = {}
            for d in data:
                name = d.get("model", "")
                acc = d.get("accuracy_by_type", {})
                values2[name] = [acc.get(et, 0) for et in entity_types]
            fig2 = make_grouped_bar(entity_types, models, values2, "Entity Type", "Accuracy")
            st.plotly_chart(fig2, width='stretch')

        # Latency vs Accuracy scatter
        st.subheader("Latency vs Accuracy")
        latencies = [d.get("avg_latency_ms", 0) for d in data]
        accuracies = [d.get("overall_accuracy", 0) for d in data]
        fig3 = make_scatter_plot(latencies, accuracies, models, "Avg Latency (ms)", "Accuracy")
        st.plotly_chart(fig3, width='stretch')

    def render_error_analysis(self, data: list[dict]) -> None:
        st.subheader("Error Analysis")
        for model_data in data:
            model_name = model_data.get("model", "Unknown")
            per_sample = model_data.get("per_sample", [])
            if not per_sample:
                continue

            errors = [s for s in per_sample if not s.get("correct", True)]
            with st.expander(f"{model_name} — {len(errors)} errors"):
                difficulty_filter = st.selectbox(
                    "Filter by difficulty",
                    ["All"] + ["easy", "medium", "hard"],
                    key=f"err_{model_name}_diff",
                )
                entity_filter = st.selectbox(
                    "Filter by entity type",
                    ["All"] + ["drug", "hospital"],
                    key=f"err_{model_name}_entity",
                )
                filtered = errors
                if difficulty_filter != "All":
                    filtered = [e for e in filtered if e.get("difficulty") == difficulty_filter]
                if entity_filter != "All":
                    filtered = [e for e in filtered if e.get("entity_type") == entity_filter]

                for err in filtered[:20]:
                    color = "green" if err.get("correct") else "red"
                    st.markdown(
                        f"**Query:** `{err.get('query', '')}` | "
                        f"<span style='color:{color}'>"
                        f"Predicted: `{err.get('predicted_name', '')}` | "
                        f"Ground truth: `{err.get('ground_truth', '')}`"
                        f"</span> | "
                        f"Confidence: {err.get('confidence', 0):.2f} | "
                        f"Difficulty: {err.get('difficulty', '')}",
                        unsafe_allow_html=True,
                    )


_REGISTRY: dict[str, DomainChartAdapter] = {}


def register_adapter(adapter: DomainChartAdapter) -> None:
    _REGISTRY[adapter.domain_name] = adapter


def get_adapter(domain: str) -> Optional[DomainChartAdapter]:
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


# Register built-in adapters
register_adapter(MedicalEntityAdapter())
