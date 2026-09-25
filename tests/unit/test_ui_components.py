"""Unit tests for shared UI helpers — format policy, entity-type union, chart color cycling."""

from __future__ import annotations

from ui.components.charts import make_grouped_bar, make_scatter_plot
from ui.components.domain_adapters import _entity_types_in
from ui.components.format import delta, fmt_num, fmt_pct
from ui.config import CHART_COLORS


class TestFormatHelpers:
    def test_fmt_pct_numeric(self) -> None:
        assert fmt_pct(0.6831) == "68.3%"
        assert fmt_pct(1) == "100.0%"
        assert fmt_pct(0) == "0.0%"

    def test_fmt_pct_non_numeric_renders_dash(self) -> None:
        assert fmt_pct(None) == "—"
        assert fmt_pct("n/a") == "—"
        assert fmt_pct(True) == "—"  # bool is an int subclass — excluded on purpose

    def test_fmt_num_precision(self) -> None:
        assert fmt_num(0.683) == "0.683"
        assert fmt_num(62.4) == "62.4"
        assert fmt_num(None) == "—"
        assert fmt_num("err") == "—"

    def test_delta_numeric(self) -> None:
        assert delta(0.5, 0.62, pct=True) == "+12.0%"
        assert delta(0.5, 0.52, pct=False) == "+0.020"
        assert delta(50.0, 62.4, pct=False) == "+12.4"

    def test_delta_non_numeric_is_none(self) -> None:
        assert delta(None, 0.5, pct=True) is None
        assert delta(0.5, "n/a", pct=False) is None


class TestEntityTypesUnion:
    def test_union_preserves_first_seen_order(self) -> None:
        data = [
            {"model": "A", "accuracy_by_type": {"drug": 0.5, "hospital": 0.6}},
            {"model": "B", "accuracy_by_type": {"device": 0.7, "drug": 0.8}},
        ]
        assert _entity_types_in(data) == ["drug", "hospital", "device"]

    def test_missing_or_empty_mappings(self) -> None:
        assert _entity_types_in([{"model": "A"}]) == []
        assert _entity_types_in([{"model": "A", "accuracy_by_type": None}]) == []
        assert _entity_types_in([]) == []


class TestChartBuilders:
    def test_scatter_colors_cycle_past_palette(self) -> None:
        # 12 points vs 8 palette colors — the marker color list must still
        # have 12 entries (a bare slice would silently hand Plotly 8).
        n = len(CHART_COLORS) + 4
        fig = make_scatter_plot(list(range(n)), [0.5] * n, [f"m{i}" for i in range(n)])
        colors = fig.data[0].marker.color
        assert len(colors) == n
        assert set(colors) <= set(CHART_COLORS)

    def test_scatter_none_coordinates_survive(self) -> None:
        fig = make_scatter_plot([1.0, None], [None, 0.5], ["a", "b"])
        assert fig.data[0].x == (1.0, None)
        assert fig.data[0].y == (None, 0.5)

    def test_grouped_bar_accepts_none_gaps(self) -> None:
        fig = make_grouped_bar(
            ["drug", "hospital"], ["A", "B"], {"A": [0.5, None], "B": [None, 0.7]}
        )
        assert fig.data[0].y == (0.5, None)
