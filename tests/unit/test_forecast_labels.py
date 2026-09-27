"""Actual local event/price/calendar transformations; no market performance claims."""

import json
from datetime import datetime, timedelta, timezone

import pytest

from src.workbench.composition import CompositionRecipe, compose_sources, required_sources
from src.workbench.forecast_labels import ForecastLabelsStep, build_forecast_labels
from src.workbench.sources import read_source


def source(name, rows, *, scope="full"):
    return read_source(
        name + ".jsonl", "\n".join(json.dumps(row) for row in rows).encode(), scope=scope
    )


@pytest.fixture
def financial_sources():
    start = datetime(2025, 1, 1, 21, tzinfo=timezone.utc)
    # Deliberately irregular synthetic calendar; never presented as an exchange calendar.
    closes = [start + timedelta(days=index * 2 + (index > 10)) for index in range(48)]
    events = [
        {
            "event_id": "filing-a",
            "ticker": "SYN",
            "published_at": (closes[0] + timedelta(hours=1)).isoformat(),
            "text": "已公开的虚构财报",
        },
    ]
    prices = [
        {"symbol": "SYN", "close_at": close.isoformat(), "adjusted": str(100 + index)}
        for index, close in enumerate(closes)
    ]
    calendar = [{"close_at": close.isoformat()} for close in closes]
    return closes, events, prices, calendar


def step(closes, **updates):
    values = {
        "price_source": "prices",
        "calendar_source": "calendar",
        "event_id_column": "event_id",
        "symbol_column": "ticker",
        "available_at_column": "published_at",
        "price_symbol_column": "symbol",
        "price_close_at_column": "close_at",
        "price_value_column": "adjusted",
        "calendar_close_at_column": "close_at",
        "price_basis": "split_adjusted_close",
        "observation_end": closes[-1].isoformat(),
    }
    return ForecastLabelsStep(**{**values, **updates})


def test_horizon_counts_explicit_sessions_and_composition_keeps_all_sources(financial_sources):
    closes, events, prices, calendar = financial_sources
    sources = {
        "events": source("events", events),
        "prices": source("prices", prices),
        "calendar": source("calendar", calendar),
    }
    recipe = CompositionRecipe(base_source="events", steps=[step(closes)])
    assert required_sources(recipe) == required_sources(recipe.model_dump()) == set(sources)
    result = compose_sources(sources, recipe)
    assert result.can_confirm
    assert result.requires_review
    row = result.source.rows[0].values
    assert row["forecast_prediction_at"] == closes[1].isoformat()
    assert row["forecast_label_end_at"] == closes[21].isoformat()
    assert row["forecast_reference_price"] == "101"
    assert row["forecast_target_price"] == "121"
    assert row["forecast_direction"] == "上涨"
    assert row["text"] == events[0]["text"]
    assert {ref.source_digest for ref in result.origins["r000001"]} == {
        value.digest for value in sources.values()
    }
    assert result.source.scope == "full"
    sources["calendar"].scope = "sample"
    assert compose_sources(sources, recipe).source.scope == "sample"


def test_future_quotes_cannot_mature_an_unobserved_label(financial_sources):
    closes, events, prices, calendar = financial_sources
    outcome = build_forecast_labels(
        events,
        source("prices", prices),
        source("calendar", calendar),
        step(closes, observation_end=closes[10].isoformat()),
    )
    row = outcome["rows"][0]
    assert row["forecast_status"] == "label_not_observed"
    assert row["forecast_direction"] is None
    assert row["forecast_target_price"] is None
    assert row["forecast_realized_return"] is None
    assert row["forecast_label_end_at"] == closes[21].isoformat()
    assert not any(issue["severity"] == "blocking" for issue in outcome["issues"])


def test_missing_intermediate_price_is_not_skipped_to_shorten_window(financial_sources):
    closes, events, prices, calendar = financial_sources
    del prices[10]
    outcome = build_forecast_labels(
        events, source("prices", prices), source("calendar", calendar), step(closes)
    )
    assert len(outcome["rows"]) == len(events)
    assert outcome["rows"][0]["forecast_direction"] is None
    assert any("缺少1条行情" in issue["message"] for issue in outcome["issues"])


@pytest.mark.parametrize(
    "kind",
    ["duplicate_price", "duplicate_calendar", "naive_time", "bad_price", "calendar_coverage"],
)
def test_ambiguous_or_missing_source_evidence_blocks_without_dropping_events(
    financial_sources, kind
):
    closes, events, prices, calendar = financial_sources
    if kind == "duplicate_price":
        prices.append(dict(prices[1]))
    elif kind == "duplicate_calendar":
        calendar.append(dict(calendar[1]))
    elif kind == "naive_time":
        events[0]["published_at"] = "2025-01-01"
    elif kind == "bad_price":
        prices[1]["adjusted"] = "NaN"
    else:
        events[0]["published_at"] = (closes[0] - timedelta(hours=1)).isoformat()
    outcome = build_forecast_labels(
        events, source("prices", prices), source("calendar", calendar), step(closes)
    )
    assert any(issue["severity"] == "blocking" for issue in outcome["issues"])
    assert len(outcome["rows"]) == 1
    assert outcome["rows"][0]["text"] == events[0]["text"]
    assert outcome["rows"][0]["forecast_direction"] is None


def test_close_timestamp_tie_uses_next_session_and_flat_price_is_not_up(financial_sources):
    closes, events, prices, calendar = financial_sources
    events[0]["published_at"] = closes[1].isoformat()
    for row in prices:
        row["adjusted"] = "100.00"
    outcome = build_forecast_labels(
        events, source("prices", prices), source("calendar", calendar), step(closes)
    )
    row = outcome["rows"][0]
    assert row["forecast_prediction_at"] == closes[2].isoformat()
    assert row["forecast_label_end_at"] == closes[22].isoformat()
    assert row["forecast_direction"] == "未上涨"
    assert row["forecast_realized_return"] == "0"


def test_forecast_requires_both_prices_and_calendar(financial_sources):
    closes, events, prices, _ = financial_sources
    sources = {"events": source("events", events), "prices": source("prices", prices)}
    result = compose_sources(sources, CompositionRecipe(base_source="events", steps=[step(closes)]))
    assert not result.can_confirm
    assert len(result.source.rows) == 1
    assert any(issue.code == "missing_source" for issue in result.issues)
