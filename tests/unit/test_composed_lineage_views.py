"""Source lineage distinguishes separate views of the same workbook."""

from copy import deepcopy

import pytest

from src.agent.intake import _preview_origins
from src.workbench.business_evaluation import (
    BusinessEvaluationService,
    EvaluationProtocol,
    Generation,
)
from src.workbench.composition import CompositionRecipe, SourceRef, compose_sources
from src.workbench.evaluation_diagnostics import EvaluationDiagnostics
from src.workbench.intake_service import IntakeService
from tests.unit.test_business_evaluation import fixture_factory
from tests.unit.test_business_evaluation import model_paths as model_paths
from tests.unit.test_composed_intake import _workbook_bytes, analysis_for
from tests.unit.test_forecast_labels import financial_sources as financial_sources
from tests.unit.test_intake_adapters import real_sandbox as real_sandbox


def composed_task(tmp_path, *, legacy=False, with_adapter=False):
    workbook = _workbook_bytes(
        ("工单", ("ticket", "description"), [(str(i), f"问题{i}") for i in range(12)]),
        ("标签", ("id", "category"), [(str(i), "咨询") for i in range(12)]),
    )
    service = IntakeService(tmp_path / "intake")
    session = service.create("预测工单类型", "same.xlsx", workbook, scope="full", sheet="工单")
    session = service.add_source(
        session.session_id,
        session.revision,
        "labels",
        "same.xlsx",
        workbook,
        scope="full",
        sheet="标签",
    )
    plan = analysis_for(session)
    if with_adapter:
        composed = compose_sources(
            session.sources, CompositionRecipe.model_validate(plan.composition)
        )
        example = {"__row_id": "r000001", **composed.source.rows[0].values}
        counterexample = {**example, "__row_id": "negative", "description": ""}
        plan_value = plan.model_dump()
        plan_value["adapter"] = {
            "source_code": (
                "def transform(rows, config):\n"
                "    result = []\n"
                "    for row in rows:\n"
                "        if not row['description']:\n"
                "            raise ValueError('missing description')\n"
                "        result.append({**row, 'extra': row['description'].strip()})\n"
                "    return result\n"
            ),
            "new_columns": ["extra"],
            "examples": [
                {
                    "name": "business",
                    "kind": "business",
                    "rows": [example],
                    "expected_rows": [{**example, "extra": example["description"]}],
                },
                {
                    "name": "negative",
                    "kind": "counterexample",
                    "rows": [counterexample],
                    "expect_error": True,
                },
            ],
        }
        plan_value["task"]["field_roles"].append(
            {"column": "extra", "role": "unused", "reason": "保留转换后的辅助值"}
        )
        from src.workbench.intake_models import IntakeAnalysis

        plan = IntakeAnalysis.model_validate(plan_value)
    session = service.apply_analysis(session, plan)
    session = service.confirm(session.session_id, session.revision)
    session = service.validate_full_sources(session.session_id, session.revision)
    if legacy:
        # A pre-fix record collapsed the two views into one digest/row reference.
        for key, refs in session.full_data.composition_report["origins"].items():
            session.full_data.composition_report["origins"][key] = [
                {"source_digest": refs[0]["source_digest"], "row_id": refs[0]["row_id"]}
            ]
        session = service._save(session, session.revision)
    session = service.confirm_full_data(session.session_id, session.revision)
    session = service.materialize_dataset(
        session.session_id, session.revision, registry_root=tmp_path / "registry"
    )
    return service, session


def report_for(session, models, tmp_path):
    return BusinessEvaluationService(tmp_path / "evaluation").compare(
        session,
        models,
        EvaluationProtocol("classification_exact"),
        runtime_factory=fixture_factory(
            session, [], lambda _label, _index, _expected: Generation("错误")
        ),
    )


def test_same_workbook_different_sheets_remain_distinct_in_real_diagnostics(tmp_path, model_paths):
    _, session = composed_task(tmp_path)
    origins = session.full_data.composition_report["origins"]["r000001"]
    assert len(origins) == 2
    assert {ref["source_alias"] for ref in origins} == {"main", "labels"}
    assert len({ref["source_digest"] for ref in origins}) == 1
    assert len({ref["source_view_digest"] for ref in origins}) == 2
    report = report_for(session, model_paths, tmp_path)
    cases = EvaluationDiagnostics(report, session).read_cases()["cases"]
    assert cases
    for case in cases:
        original = {row["alias"]: row for row in case["original_rows"]}
        assert set(original) == {"main", "labels"}
        assert original["main"]["values"]["description"] in case["prompt"]
        assert original["labels"]["values"]["category"] == case["expected"]


def test_historical_aliasless_ambiguous_lineage_cannot_invent_original_rows(tmp_path, model_paths):
    _, session = composed_task(tmp_path, legacy=True)
    report = report_for(session, model_paths, tmp_path)
    with pytest.raises(ValueError, match="来源存在歧义"):
        EvaluationDiagnostics(report, session)


def test_current_lineage_cannot_rebind_one_alias_to_another_sheet(tmp_path, model_paths):
    _, session = composed_task(tmp_path)
    report = report_for(session, model_paths, tmp_path)
    damaged = session.model_copy(deep=True)
    damaged.full_data.sources["main"] = damaged.full_data.sources["labels"].model_copy(deep=True)
    with pytest.raises(ValueError, match="无法定位"):
        EvaluationDiagnostics(report, damaged)


def test_composed_lineage_survives_real_adapter(tmp_path, model_paths, request):
    request.getfixturevalue("real_sandbox")
    _, session = composed_task(tmp_path, with_adapter=True)
    assert session.full_data.adapter_report["validation"]["status"] == "passed"
    report = report_for(session, model_paths, tmp_path)
    cases = EvaluationDiagnostics(report, session).read_cases()["cases"]
    assert cases
    for case in cases:
        original = {row["alias"]: row for row in case["original_rows"]}
        assert set(original) == {"main", "labels"}
        assert original["main"]["values"]["description"] in case["prompt"]
        assert original["labels"]["values"]["category"] == case["expected"]


def test_agent_preview_keeps_distinct_alias_boundaries_without_mutation():
    origins = {
        "r1": [
            {"source_alias": alias, "source_digest": "same", "row_id": str(i)}
            for alias in ("main", "labels")
            for i in range(3)
        ]
    }
    before = deepcopy(origins)
    result = _preview_origins(origins)
    assert {(ref["source_alias"], ref["row_id"]) for ref in result["origins"]["r1"]} == {
        ("main", "0"),
        ("main", "2"),
        ("labels", "0"),
        ("labels", "2"),
    }
    assert origins == before


def test_old_source_ref_serialization_stays_compatible():
    reference = {"source_digest": "abc", "row_id": "r1"}
    assert SourceRef.model_validate(reference).model_dump() == reference


def test_forecast_lineage_keeps_same_workbook_sheet_sources_distinct(request):
    from src.workbench.sources import read_source
    from tests.unit.test_forecast_labels import step

    closes, events, prices, calendar = request.getfixturevalue("financial_sources")
    workbook = _workbook_bytes(
        ("事件", tuple(events[0]), [tuple(row.values()) for row in events]),
        ("行情", tuple(prices[0]), [tuple(row.values()) for row in prices]),
        ("日历", tuple(calendar[0]), [tuple(row.values()) for row in calendar]),
    )
    sources = {
        alias: read_source("same.xlsx", workbook, scope="full", sheet=sheet)
        for alias, sheet in (("events", "事件"), ("prices", "行情"), ("calendar", "日历"))
    }
    result = compose_sources(sources, CompositionRecipe(base_source="events", steps=[step(closes)]))
    assert result.can_confirm
    refs = result.origins["r000001"]
    assert {ref.source_alias for ref in refs} == {"events", "prices", "calendar"}
    assert len({ref.source_digest for ref in refs}) == 1
    assert len({ref.source_view_digest for ref in refs}) == 3
    assert sum(ref.source_alias == "prices" for ref in refs) > 1
    assert sum(ref.source_alias == "calendar" for ref in refs) > 1
