"""Goal → real composition → confirmed full sources → immutable train partitions."""

import json

import pytest

from src.data_flywheel.dataset_registry import LocalDatasetRegistry
from src.workbench.composition import CompositionRecipe, compose_sources
from src.workbench.intake_models import IntakeAnalysis
from src.workbench.intake_service import IntakeService, next_action
from tests.unit.test_data_composition import join
from tests.unit.test_data_intake import ScriptedModel


def rows_bytes(rows):
    return "\n".join(json.dumps(row, ensure_ascii=False) for row in rows).encode()


def files(count):
    return {
        "main": (
            "tickets.jsonl",
            rows_bytes([{"ticket": str(i), "description": f"问题{i}"} for i in range(count)]),
        ),
        "labels": (
            "labels.jsonl",
            rows_bytes([{"id": str(i), "category": "咨询"} for i in range(count)]),
        ),
    }


def setup(service, scope="sample"):
    data = files(3)
    session = service.create("预测工单类型", *data["main"], scope=scope)
    return service.add_source(
        session.session_id, session.revision, "labels", *data["labels"], scope=scope
    )


def analysis_for(session):
    composition = CompositionRecipe(base_source="main", steps=[join()])
    result = compose_sources(session.sources, composition)
    return IntakeAnalysis.model_validate(
        {
            "task": {
                "goal": session.goal,
                "usage_input": "客户描述",
                "desired_output": "类别",
                "row_meaning": "一行一个工单",
                "supervision_source": "质检表",
                "success_criteria": ["与人工类别一致"],
                "field_roles": [
                    {
                        "column": c,
                        "role": "input"
                        if c == "description"
                        else "target"
                        if c == "label_category"
                        else "group"
                        if c == "ticket"
                        else "metadata",
                        "reason": "业务字段核实",
                        "available_at_prediction": c == "description",
                    }
                    for c in result.source.columns
                ],
            },
            "findings": [],
            "composition": composition.model_dump(),
            "recipe": {
                "instruction": "判断工单类型",
                "inputs": [{"column": "description", "label": "描述"}],
                "targets": [
                    {"column": "label_category", "label": "类别", "value_kind": "categorical"}
                ],
                "group_columns": ["ticket"],
            },
            "training_approach": "样本验收后SFT",
            "next_steps": ["核对预览，再提供全量资料"],
        }
    )


def test_agent_composition_is_real_and_full_sources_keep_lineage(tmp_path):
    service = IntakeService(tmp_path)
    original = setup(service)
    analysis = analysis_for(original)
    model = ScriptedModel(
        [
            ("list_sources", {}),
            ("inspect_source", {"alias": "labels"}),
            ("preview_composition", analysis.composition),
            ("profile_data", {}),
            ("inspect_rows", {"row_ids": []}),
            ("preview_recipe", analysis.recipe.model_dump()),
            ("submit_analysis", analysis.model_dump()),
        ]
    )
    session = service.analyze(original.session_id, model)
    assert next_action(session) == "review_preview"
    assert session.sources["main"] == original.source
    assert session.source.digest != original.source.digest
    assert session.preview.rows[0].target == "咨询"
    session = service.confirm(session.session_id, session.revision)
    with pytest.raises(ValueError, match="每份全量"):
        service.validate_full_sources(session.session_id, session.revision)
    session = service.validate_full_sources(session.session_id, session.revision, files(8))
    assert next_action(session) == "review_full_data"
    assert len(session.source.rows) == 3
    assert len(session.full_data.source.rows) == 8
    session = service.confirm_full_data(session.session_id, session.revision)
    session = service.materialize_dataset(session.session_id, session.revision)
    registry = LocalDatasetRegistry(session.dataset.registry_root)
    parts = {
        split: registry.load_split(session.dataset.name, session.dataset.version, split)
        for split in ("train", "validation", "test")
    }
    records = sum(parts.values(), [])
    assert len(records) == 8
    for row in records:
        assert {ref["source_digest"] for ref in row["metadata"]["origins"]} == {
            source.digest for source in session.full_data.sources.values()
        }
    updated = service.add_source(
        session.session_id, session.revision, "labels", *files(4)["labels"]
    )
    assert updated.analysis is None and updated.dataset is None
    assert updated.full_data.status == "stale"
    assert registry.get_split_manifest(session.dataset.name, session.dataset.version)


def test_unpreviewed_composition_cannot_reuse_old_recipe_preview(tmp_path):
    service = IntakeService(tmp_path)
    session = setup(service)
    analysis = analysis_for(session)
    model = ScriptedModel(
        [
            ("profile_data", {}),
            ("inspect_rows", {"row_ids": []}),
            ("submit_analysis", analysis.model_dump()),
            ("preview_composition", analysis.composition),
            ("submit_analysis", analysis.model_dump()),
            ("profile_data", {}),
            ("inspect_rows", {"row_ids": []}),
            ("preview_recipe", analysis.recipe.model_dump()),
            ("submit_analysis", analysis.model_dump()),
        ]
    )
    updated = service.analyze(session.session_id, model)
    assert updated.tool_trace[2]["ok"] is False
    assert updated.tool_trace[4]["ok"] is False
    assert next_action(updated) == "review_preview"


def test_full_declared_sources_can_be_reused_without_reupload(tmp_path):
    service = IntakeService(tmp_path)
    session = setup(service, "full")
    session = service.apply_analysis(session, analysis_for(session))
    session = service.confirm(session.session_id, session.revision)
    session = service.validate_full_data(session.session_id, session.revision)
    assert session.full_data.source.scope == "full"
    assert len(session.full_data.sources) == 2
    assert next_action(session) == "review_full_data"
