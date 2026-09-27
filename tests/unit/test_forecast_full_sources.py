"""Forecast source discovery requires independent event, price and session-calendar files."""

import json

from src.workbench.composition import CompositionRecipe, compose_sources
from src.workbench.intake_models import IntakeAnalysis
from src.workbench.intake_service import IntakeService
from tests.unit.test_full_data_cli import invoke

EVENTS = "event,symbol,available,text\ne1,AAA,2024-01-02T10:00:00Z,公开事件内容\n".encode()
PRICES = b"symbol,close_at,value\nAAA,2024-01-01T16:00:00Z,100\nAAA,2024-01-02T16:00:00Z,101\nAAA,2024-01-03T16:00:00Z,102\nAAA,2024-01-04T16:00:00Z,103\n"
CALENDAR = b"close_at\n2024-01-01T16:00:00Z\n2024-01-02T16:00:00Z\n2024-01-03T16:00:00Z\n2024-01-04T16:00:00Z\n"
COMPOSITION = {
    "base_source": "main",
    "steps": [
        {
            "operation": "forecast_labels",
            "price_source": "prices",
            "calendar_source": "calendar",
            "event_id_column": "event",
            "symbol_column": "symbol",
            "available_at_column": "available",
            "price_symbol_column": "symbol",
            "price_close_at_column": "close_at",
            "price_value_column": "value",
            "calendar_close_at_column": "close_at",
            "price_basis": "split_adjusted_close",
            "observation_end": "2024-01-31T00:00:00Z",
            "horizon_sessions": 2,
        }
    ],
}


def forecast_session(service, *, scope="sample"):
    session = service.create("按公开事件预测之后两个交易日方向", "events.csv", EVENTS, scope=scope)
    for alias, data in (("prices", PRICES), ("calendar", CALENDAR)):
        session = service.add_source(
            session.session_id, session.revision, alias, alias + ".csv", data, scope=scope
        )
    result = compose_sources(session.sources, CompositionRecipe.model_validate(COMPOSITION))
    assert result.can_confirm
    proposal = IntakeAnalysis.model_validate(
        {
            "task": {
                "goal": session.goal,
                "usage_input": "公开事件文本",
                "desired_output": "未来方向",
                "row_meaning": "一个事件",
                "supervision_source": "明确价格口径及交易日历派生",
                "success_criteria": ["方向正确"],
                "field_roles": [
                    {
                        "column": name,
                        "role": "group"
                        if name == "event"
                        else "input"
                        if name == "text"
                        else "target"
                        if name == "forecast_direction"
                        else "metadata",
                        "reason": "明确字段用途",
                        **({"available_at_prediction": True} if name == "text" else {}),
                    }
                    for name in result.source.columns
                ],
            },
            "composition": COMPOSITION,
            "recipe": {
                "instruction": "根据事件文本预测方向",
                "inputs": [{"column": "text", "label": "文本"}],
                "targets": [
                    {"column": "forecast_direction", "label": "方向", "value_kind": "categorical"}
                ],
                "group_columns": ["event"],
                "temporal_split": {
                    "available_at_column": "available",
                    "prediction_at_column": "forecast_prediction_at",
                    "label_end_at_column": "forecast_label_end_at",
                    "validation_start": "2024-01-10T00:00:00Z",
                    "test_start": "2024-01-20T00:00:00Z",
                    "observation_end": "2024-01-31T00:00:00Z",
                },
            },
            "findings": [],
            "training_approach": "SFT",
            "next_steps": ["核对价格口径与完整日历"],
        }
    )
    session = service.apply_analysis(session, proposal)
    return service.confirm(session.session_id, session.revision)


def test_full_sources_cli_requires_calendar_and_keeps_three_originals(tmp_path):
    service = IntakeService(tmp_path / "intake")
    session = forecast_session(service)
    files = {}
    for alias, data in (("main", EVENTS), ("prices", PRICES), ("calendar", CALENDAR)):
        path = tmp_path / (alias + ".csv")
        path.write_bytes(data)
        files[alias] = path
    base = [
        "full-sources",
        session.session_id,
        "--revision",
        session.revision,
        "--source",
        f"main={files['main']}",
        "--source",
        f"prices={files['prices']}",
    ]
    missing = invoke(service, *base)
    assert missing.returncode == 2 and "calendar" in missing.stderr
    assert service.load(session.session_id).revision == session.revision
    complete = invoke(service, *base, "--source", f"calendar={files['calendar']}")
    assert complete.returncode == 0, complete.stderr
    payload = json.loads(complete.stdout)
    assert set(payload["full_data"]["sources"]) == {"main", "prices", "calendar"}
    assert payload["full_data"]["preview"]["rows"][0]["target"] == "上涨"
    assert payload["sources"]["main"]["scope"] == "sample"
    assert any(
        "calendar" in str(step) or "forecast" in str(step)
        for step in payload["full_data"]["composition_report"]["steps"]
    )
