"""Shared configuration for the Streamlit dashboard."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MLFLOW_TRACKING_URI = f"file://{PROJECT_ROOT / 'outputs' / 'mlruns'}"
DOMAINS_DIR = PROJECT_ROOT / "domains"
CONFIGS_DIR = PROJECT_ROOT / "outputs" / "configs"

CHART_COLORS = [
    "#636EFA",  # Muted blue
    "#EF553B",  # Red
    "#00CC96",  # Green
    "#AB63FA",  # Purple
    "#FFA15A",  # Orange
    "#19D3F3",  # Cyan
    "#FF6692",  # Pink
    "#B6E880",  # Lime
]

METRIC_CARD_STYLE = """
<style>
.metric-card {{
    background-color: #1E293B;
    border-radius: 0.75rem;
    padding: 1.25rem;
    text-align: center;
    border: 1px solid #334155;
}}
.metric-card .value {{
    font-size: 1.75rem;
    font-weight: 700;
    color: #F8FAFC;
}}
.metric-card .label {{
    font-size: 0.85rem;
    color: #94A3B8;
    margin-top: 0.25rem;
}}
.metric-card .delta {{
    font-size: 0.8rem;
    margin-top: 0.25rem;
}}
.metric-card .delta.positive {{ color: #4ADE80; }}
.metric-card .delta.negative {{ color: #F87171; }}
</style>
"""

MODEL_OPTIONS = {
    # Qwen2.5
    "Qwen/Qwen2.5-0.5B-Instruct": "~1.5 GB (4-bit)",
    "Qwen/Qwen2.5-1.5B-Instruct": "~2.3 GB (4-bit)",
    "Qwen/Qwen2.5-3B-Instruct": "~4.5 GB (4-bit)",
    # Qwen3
    "Qwen/Qwen3-0.6B": "~1.2 GB (4-bit)",
    "Qwen/Qwen3-1.7B": "~2.0 GB (4-bit)",
    "Qwen/Qwen3-4B-Instruct-2507": "~3.5 GB (4-bit)",
    "Qwen/Qwen3-8B": "~6.0 GB (4-bit)",
    "Qwen/Qwen3-14B": "~10.0 GB (4-bit)",
    # LLaMA
    "meta-llama/Llama-3.2-1B-Instruct": "~1.8 GB (4-bit)",
    "meta-llama/Llama-3.2-3B-Instruct": "~4.5 GB (4-bit)",
}
