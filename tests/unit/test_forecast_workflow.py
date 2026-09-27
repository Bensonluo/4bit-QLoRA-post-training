"""Synthetic disclosure → real labels → temporal partitions → fixed evaluation cases."""

from datetime import datetime, timedelta, timezone

from src.data_flywheel.dataset_registry import LocalDatasetRegistry
from src.workbench.composition import CompositionRecipe, compose_sources
from src.workbench.evaluation_suites import EvalSuiteService, evaluation_cases
from src.workbench.intake_models import IntakeAnalysis
from src.workbench.intake_service import IntakeService, next_action
from src.workbench.training_preflight import _verified_partitions
from tests.unit.test_composed_intake import rows_bytes
from tests.unit.test_forecast_labels import step


def run_workflow(root):
    # This is an explicit synthetic calendar, not an assertion about US market holidays.
    closes = [
        datetime(2023, 1, 1, 21, tzinfo=timezone.utc) + timedelta(days=i * 2) for i in range(180)
    ]
    events = [
        {
            "event_id": f"event-{i}",
            "ticker": "SYN",
            "published_at": (closes[i] - timedelta(hours=1)).isoformat(),
            "text": f"虚构公司第{i}次公开披露：收入为{1000 + i}，单位美元。",
        }
        for i in [1, 5, 35, 51, 55, 85, 101, 105, 135]
    ]
    prices = [
        {"symbol": "SYN", "close_at": close.isoformat(), "adjusted": str(100 + i % 30)}
        for i, close in enumerate(closes)
    ]
    calendar = [{"close_at": close.isoformat()} for close in closes]
    files = {
        "main": ("events.jsonl", rows_bytes(events)),
        "prices": ("prices.jsonl", rows_bytes(prices)),
        "calendar": ("calendar.jsonl", rows_bytes(calendar)),
    }
    service = IntakeService(root / "intake")
    session = service.create(
        "从已公开财报预测未来20个交易日上涨或未上涨",
        "sample.jsonl",
        rows_bytes([events[0], events[-1]]),
    )
    for alias in ("prices", "calendar"):
        session = service.add_source(session.session_id, session.revision, alias, *files[alias])
    composition = CompositionRecipe(
        base_source="main",
        steps=[step(closes, observation_end=closes[140].isoformat())],
    )
    composed = compose_sources(session.sources, composition)
    roles = {"text": "input", "event_id": "group", "forecast_direction": "target"}
    analysis = IntakeAnalysis.model_validate(
        {
            "task": {
                "goal": session.goal,
                "usage_input": "预测前公开的财报",
                "desired_output": "上涨或未上涨",
                "row_meaning": "一次披露",
                "supervision_source": "完整窗口行情与显式交易日历",
                "success_criteria": ["在独立后续时期对照基座与训练期多数类基准"],
                "field_roles": [
                    {
                        "column": column,
                        "role": roles.get(column, "metadata"),
                        "reason": "核对公开时间与监督用途",
                        "available_at_prediction": column == "text",
                    }
                    for column in composed.source.columns
                ],
            },
            "composition": composition.model_dump(),
            "recipe": {
                "instruction": "根据公开财报预测20个交易日方向，只输出上涨或未上涨",
                "inputs": [{"column": "text", "label": "财报"}],
                "targets": [
                    {"column": "forecast_direction", "label": "方向", "value_kind": "categorical"}
                ],
                "group_columns": ["event_id"],
                "temporal_split": {
                    "available_at_column": "published_at",
                    "prediction_at_column": "forecast_prediction_at",
                    "label_end_at_column": "forecast_label_end_at",
                    "validation_start": closes[50].isoformat(),
                    "test_start": closes[100].isoformat(),
                    "observation_end": closes[140].isoformat(),
                },
            },
            "findings": [],
            "training_approach": "确认数据后使用SFT验证",
            "next_steps": ["核对转换与排除原因"],
        }
    )
    session = service.apply_analysis(session, analysis)
    assert next_action(session) == "review_preview"
    assert session.preview.rows[-1].target is None
    session = service.confirm(session.session_id, session.revision)
    session = service.validate_full_sources(session.session_id, session.revision, files)
    assert next_action(session) == "review_full_data"
    session = service.confirm_full_data(session.session_id, session.revision)
    session = service.materialize_dataset(session.session_id, session.revision)
    report = {"issues": []}
    parts = _verified_partitions(session, report)
    assert not report["issues"]
    assert {key: len(rows) for key, rows in parts.items()} == {
        "train": 2,
        "validation": 2,
        "test": 2,
    }
    registry = LocalDatasetRegistry(session.dataset.registry_root)
    metadata = registry.get_split_manifest(session.dataset.name, session.dataset.version)[
        "metadata"
    ]
    excluded = metadata["excluded_rows"]
    assert {row["reason"] for row in excluded} == {
        "label_window_crosses_validation_start",
        "label_window_crosses_test_start",
        "label_not_mature",
    }
    assert len(excluded) == 3
    assert next(row for row in excluded if row["reason"] == "label_not_mature")["target"] is None
    all_rows = [row for values in parts.values() for row in values]
    assert {row["metadata"]["source_row_id"] for row in all_rows} | {
        row["row_id"] for row in excluded
    } == {row.row_id for row in session.full_data.source.rows}
    for row in all_rows:
        assert {ref["source_digest"] for ref in row["metadata"]["origins"]} == {
            source.digest for source in session.full_data.sources.values()
        }
        assert "forecast_" not in row["input"]
    suites = EvalSuiteService(root / "suites")
    ref = suites.freeze(session)
    session.dataset.evaluation_suite = ref
    before = evaluation_cases(session)
    # Reordering source files keeps the same fixed questions and temporal assignments.
    files["main"] = ("reordered.jsonl", rows_bytes(list(reversed(events))))
    session = service.validate_full_sources(session.session_id, session.revision, files)
    session = service.confirm_full_data(session.session_id, session.revision)
    session = service.materialize_dataset(
        session.session_id, session.revision, evaluation_suite=ref
    )
    assert evaluation_cases(session)["evaluation_key"] == before["evaluation_key"]
    _verified_partitions(session, {"issues": []})
    return {
        "data_kind": "synthetic_events_prices_calendar",
        "session_id": session.session_id,
        "dataset_version": session.dataset.version,
        "suite_id": ref["suite_id"],
        "input_events": len(events),
        "horizon_sessions": 20,
        "included": {key: len(rows) for key, rows in parts.items()},
        "excluded": [
            {"event": row["original"]["event_id"], "reason": row["reason"]} for row in excluded
        ],
        "sample_pending_target": None,
        "all_included_rows_have_three_source_lineage": True,
        "fixed_suite_survives_source_reorder": True,
        "real_market_data": False,
        "live_agent_called": False,
        "model_trained": False,
        "predictive_performance_verified": False,
    }


def test_forecast_intake_to_temporal_materialization_and_fixed_suite(tmp_path):
    run_workflow(tmp_path)
