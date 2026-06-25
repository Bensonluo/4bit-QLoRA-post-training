"""Reusable Plotly chart builders for the dashboard."""

from __future__ import annotations

import plotly.graph_objects as go

from ui.config import CHART_COLORS


def make_metric_timeseries(
    runs_data: dict[str, list[tuple[int, float]]],
    metric_name: str = "loss",
) -> go.Figure:
    """Multi-line time series chart for loss/LR curves."""
    fig = go.Figure()
    for i, (run_name, data) in enumerate(runs_data.items()):
        if not data:
            continue
        steps, values = zip(*data)
        fig.add_trace(go.Scatter(
            x=list(steps),
            y=list(values),
            mode="lines",
            name=run_name,
            line=dict(color=CHART_COLORS[i % len(CHART_COLORS)], width=2),
        ))
    fig.update_layout(
        title=f"{metric_name.replace('_', ' ').title()} Over Steps",
        xaxis_title="Step",
        yaxis_title=metric_name.title(),
        template="plotly_dark",
        height=400,
        margin=dict(l=60, r=20, t=50, b=40),
    )
    return fig


def make_bar_comparison(
    models: list[str],
    metrics: dict[str, list[float]],
    metric_name: str = "Accuracy",
) -> go.Figure:
    """Horizontal grouped bar chart for metric comparison."""
    fig = go.Figure()
    for i, (metric_key, values) in enumerate(metrics.items()):
        fig.add_trace(go.Bar(
            y=models,
            x=values,
            orientation="h",
            name=metric_key,
            marker_color=CHART_COLORS[i % len(CHART_COLORS)],
        ))
    fig.update_layout(
        title=f"{metric_name} Comparison",
        template="plotly_dark",
        height=max(300, 60 * len(models)),
        margin=dict(l=150, r=20, t=50, b=40),
        barmode="group",
    )
    return fig


def make_calibration_chart(
    calibration_data: dict[str, dict[str, dict]],
) -> go.Figure:
    """Reliability diagram: confidence bins vs actual accuracy."""
    fig = go.Figure()
    # Ideal diagonal
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode="lines", name="Ideal",
        line=dict(dash="dash", color="gray", width=1),
    ))
    for i, (model_name, bins) in enumerate(calibration_data.items()):
        x_vals, y_vals = [], []
        for bin_name, info in bins.items():
            low, high = bin_name.replace(">=", "0.9-1.0").split("-") if "-" in bin_name else (bin_name.replace(">=", ""), "1.0")
            try:
                mid = (float(low.replace(">=", "")) + float(high)) / 2
            except ValueError:
                mid = 0.5
            x_vals.append(mid)
            y_vals.append(info.get("accuracy", 0))
        fig.add_trace(go.Scatter(
            x=x_vals, y=y_vals,
            mode="lines+markers",
            name=model_name,
            line=dict(color=CHART_COLORS[i % len(CHART_COLORS)], width=2),
        ))
    fig.update_layout(
        title="Confidence Calibration",
        xaxis_title="Predicted Confidence",
        yaxis_title="Actual Accuracy",
        template="plotly_dark",
        height=400,
        margin=dict(l=60, r=20, t=50, b=40),
    )
    return fig


def make_radar_chart(
    models: list[str],
    metrics: dict[str, list[float]],
    metric_labels: list[str] | None = None,
) -> go.Figure:
    """Radar/spider chart for multi-dimensional model comparison."""
    labels = metric_labels or list(metrics.keys())
    fig = go.Figure()
    for i, model in enumerate(models):
        values = [metrics[m][i] if i < len(metrics[m]) else 0 for m in metrics]
        fig.add_trace(go.Scatterpolar(
            r=values + [values[0]],
            theta=labels + [labels[0]],
            fill="toself",
            name=model,
            opacity=0.3,
            line=dict(color=CHART_COLORS[i % len(CHART_COLORS)]),
        ))
    fig.update_layout(
        template="plotly_dark",
        height=450,
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True,
    )
    return fig


def make_grouped_bar(
    categories: list[str],
    models: list[str],
    values: dict[str, list[float]],
    x_label: str = "Category",
    y_label: str = "Accuracy",
) -> go.Figure:
    """Grouped bar chart (e.g., accuracy by difficulty/entity_type)."""
    fig = go.Figure()
    for i, model in enumerate(models):
        fig.add_trace(go.Bar(
            name=model,
            x=categories,
            y=values.get(model, []),
            marker_color=CHART_COLORS[i % len(CHART_COLORS)],
        ))
    fig.update_layout(
        barmode="group",
        template="plotly_dark",
        height=400,
        margin=dict(l=60, r=20, t=50, b=40),
        xaxis_title=x_label,
        yaxis_title=y_label,
    )
    return fig


def make_scatter_plot(
    x: list[float],
    y: list[float],
    labels: list[str],
    x_label: str = "Latency (ms)",
    y_label: str = "Accuracy",
) -> go.Figure:
    """Scatter with hover labels (e.g., latency vs accuracy)."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=y, mode="markers+text",
        text=labels, textposition="top center",
        marker=dict(size=12, color=CHART_COLORS[:len(x)]),
    ))
    fig.update_layout(
        template="plotly_dark",
        height=400,
        margin=dict(l=60, r=20, t=50, b=40),
        xaxis_title=x_label,
        yaxis_title=y_label,
    )
    return fig


def make_progress_gauge(value: float, max_val: float = 1.0, label: str = "Progress") -> go.Figure:
    """Gauge chart for training progress."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value / max_val * 100,
        number={"suffix": "%"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": CHART_COLORS[0]},
            "steps": [
                {"range": [0, 50], "color": "#1E293B"},
                {"range": [50, 100], "color": "#334155"},
            ],
        },
        title={"text": label},
    ))
    fig.update_layout(
        template="plotly_dark",
        height=250,
        margin=dict(l=30, r=30, t=60, b=10),
    )
    return fig
