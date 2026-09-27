"""Agent context stays small while complete local lineage remains unchanged."""

from copy import deepcopy

from src.agent.intake import _preview_origins


def test_preview_preserves_every_source_and_boundaries_without_mutating_lineage():
    origins = {
        "r000001": [
            {"source_digest": source, "row_id": f"r{i:06d}"}
            for source, count in (("events", 1), ("calendar", 23), ("prices", 21))
            for i in range(count)
        ],
        "r000002": [{"source_digest": "events", "row_id": "r000002"}],
    }
    original = deepcopy(origins)
    result = _preview_origins(origins)
    shown = result["origins"]["r000001"]
    assert len(shown) == 5
    assert {r["source_digest"] for r in shown} == {"events", "calendar", "prices"}
    assert {r["row_id"] for r in shown if r["source_digest"] == "prices"} == {
        "r000000",
        "r000020",
    }
    assert result["origin_counts"]["r000001"] == {"total": 45, "shown": 5}
    assert result["origins"]["r000002"] == origins["r000002"]
    assert origins == original
