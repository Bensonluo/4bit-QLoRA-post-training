"""The Agent sees actual temporal exclusions, rather than guessing split behavior."""

import json

from src.agent.intake import analyze_intake
from src.workbench.sources import profile_source
from tests.unit.test_data_intake import ScriptedModel
from tests.unit.test_temporal_suite_preflight import _setup


def test_agent_receives_actual_time_counts_and_exclusion_reasons(tmp_path):
    _, session = _setup(tmp_path)
    session.source = session.full_data.source
    session.sources = {"main": session.source}
    session.profile = profile_source(session.source)
    model = ScriptedModel(
        [
            ("profile_data", {}),
            ("inspect_rows", {"row_ids": []}),
            ("preview_recipe", session.analysis.recipe.model_dump()),
            ("submit_analysis", session.analysis.model_dump()),
        ]
    )
    _, trace = analyze_intake(session, model)
    previews = [
        json.loads(message["content"])
        for message in model.seen[-1]
        if message["role"] == "tool" and "temporal_preview" in message["content"]
    ]
    actual = previews[0]["temporal_preview"]
    assert actual["counts"] == {"train": 2, "validation": 2, "test": 2}
    assert {row["reason"] for row in actual["excluded"]} == {
        "label_not_mature",
        "label_window_crosses_validation_start",
        "label_window_crosses_test_start",
    }
    assert actual["status"] == "calculated" and all(entry["ok"] for entry in trace)
