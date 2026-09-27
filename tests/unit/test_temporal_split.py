"""Point-in-time splitting preserves exclusions and forbids future leakage."""

import json

import pytest

from src.workbench.intake_models import DataRecipe, FieldRole, PreviewRow, TemporalSplitPolicy
from src.workbench.intake_service import IntakeService
from src.workbench.temporal_split import is_pending_label, temporal_assignment
from tests.unit.test_data_intake import analysis, recipe


def policy(**updates):
    value = dict(
        available_at_column="available",
        prediction_at_column="prediction",
        label_end_at_column="end",
        validation_start="2026-02-01T00:00:00Z",
        test_start="2026-03-01T00:00:00Z",
        observation_end="2026-04-01T00:00:00Z",
    )
    return TemporalSplitPolicy(**(value | updates))


def row(identity, prediction, end, *, available=None, label="up", group=None, text=None):
    return PreviewRow(
        row_id=identity,
        original={"available": available or prediction, "prediction": prediction, "end": end},
        input=text or identity,
        target=label,
        group={"event": group or identity},
        status="ready" if label else "needs_label",
    )


def ready_rows():
    return [
        row("train", "2026-01-01T00:00:00Z", "2026-01-02T00:00:00Z"),
        row("validation", "2026-02-01T00:00:00Z", "2026-02-02T00:00:00Z"),
        row("test", "2026-03-01T00:00:00Z", "2026-04-01T00:00:00Z"),
    ]


def test_boundaries_exclusions_and_utc():
    rows = ready_rows() + [
        row("cross1", "2026-01-31T00:00:00Z", "2026-02-01T00:00:00Z"),
        row("cross2", "2026-02-28T00:00:00Z", "2026-03-01T00:00:00Z"),
        row("pending", "2026-03-31T00:00:00Z", "2026-04-02T00:00:00Z", label=None),
    ]
    rows[0].original["available"] = "2026-01-01T08:00:00+08:00"
    result = temporal_assignment(rows, policy(), ["event"])
    assert result["assignments"] == {"train": [0], "validation": [1], "test": [2]}
    assert result["times_by_row"]["train"]["available_at"] == "2026-01-01T00:00:00Z"
    assert [r["reason"] for r in result["excluded_rows"]] == [
        "label_window_crosses_validation_start",
        "label_window_crosses_test_start",
        "label_not_mature",
    ]
    assert result["excluded_rows"][-1]["original"] == rows[-1].original
    assert is_pending_label(rows[-1], policy())


@pytest.mark.parametrize(
    "prediction,end,available",
    [
        ("2026-01-01", "2026-01-02T00:00:00Z", None),
        ("2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z", None),
        ("2026-01-01T00:00:00Z", "2026-01-03T00:00:00Z", "2026-01-02T00:00:00Z"),
        ("bad", "2026-05-01T00:00:00Z", None),
    ],
)
def test_invalid_or_future_information_blocks_even_missing_target(prediction, end, available):
    invalid = row("bad", prediction, end, available=available, label=None)
    assert not is_pending_label(invalid, policy())
    with pytest.raises(ValueError):
        temporal_assignment([invalid], policy(), ["event"])


def test_mature_missing_labels_not_hidden_as_window_exclusions():
    value = row("missing", "2026-01-31T00:00:00Z", "2026-02-01T00:00:00Z", label=None)
    assert not is_pending_label(value, policy())
    with pytest.raises(ValueError, match="成熟样本缺标签"):
        temporal_assignment([value], policy(), ["event"])


@pytest.mark.parametrize("duplicate_input", [True, False])
def test_connected_entities_cannot_cross_time(duplicate_input):
    rows = ready_rows()
    if duplicate_input:
        rows[1].input = rows[0].input
    else:
        rows[1].group = rows[0].group
    with pytest.raises(ValueError, match="跨时间分区"):
        temporal_assignment(rows, policy(), ["event"])


def test_group_connected_to_exclusion_fully_retained():
    rows = ready_rows() + [
        row("late", "2026-01-31T00:00:00Z", "2026-02-01T00:00:00Z", group="train")
    ]
    result = temporal_assignment(rows, policy(), ["event"])
    assert result["assignments"]["train"] == []
    assert result["excluded_rows"][0]["reason"] == "connected_to_excluded_row"


def test_policy_legacy_digest_and_future_column_guard():
    legacy = recipe()
    assert "temporal_split" not in legacy.model_dump()
    assert DataRecipe.model_validate(legacy.model_dump()).model_dump() == legacy.model_dump()
    with pytest.raises(ValueError, match="明确时区"):
        policy(validation_start="2026-02-01")
    with pytest.raises(ValueError, match="不同"):
        policy(label_end_at_column="prediction")
    with pytest.raises(ValueError, match="不得作为模型输入"):
        recipe(inputs=[{"column": "end", "label": "end"}], temporal_split=policy())


def temporal_session(tmp_path):
    service = IntakeService(tmp_path / "intake")
    header = "编号,客户描述,类别,处理结果,available,prediction,end\n"
    body = ""
    for i, (start, end) in enumerate(
        (("2026-01-01", "2026-01-02"), ("2026-02-01", "2026-02-02"), ("2026-03-01", "2026-03-02")),
        1,
    ):
        body += f"{i},独立事件{i},质量,结果,{start}T00:00:00Z,{start}T00:00:00Z,{end}T00:00:00Z\n"
    session = service.create("测试已知时间窗口", "sample.csv", (header + body).encode())
    proposal = analysis(recipe=recipe(temporal_split=policy()).model_dump())
    proposal.task.field_roles += [
        FieldRole(column=column, role="metadata", reason="时间隔离事实")
        for column in ("available", "prediction", "end")
    ]
    session = service.apply_analysis(session, proposal)
    session = service.confirm(session.session_id, session.revision)
    full = (
        header
        + body
        + "4,尚未成熟事件,,结果,2026-03-31T00:00:00Z,2026-03-31T00:00:00Z,2026-04-02T00:00:00Z\n"
    )
    session = service.validate_full_data(
        session.session_id, session.revision, "full.csv", full.encode()
    )
    return service, session


def test_full_review_materialization_preserves_all_rows_and_ignores_random(tmp_path):
    service, session = temporal_session(tmp_path)
    assert not any(issue.severity == "blocking" for issue in session.full_data.issues)
    assert session.full_data.preview.rows[-1].target is None
    session = service.confirm_full_data(session.session_id, session.revision)
    session = service.materialize_dataset(
        session.session_id,
        session.revision,
        registry_root=tmp_path / "registry",
        validation_fraction=0.9,
        test_fraction=0.9,
        seed=999,
    )
    assert session.dataset.statistics["row_counts"] == {"train": 1, "validation": 1, "test": 1}
    assert session.dataset.statistics["excluded_rows"] == 1
    with open(session.dataset.paths["manifest"]) as handle:
        manifest = json.load(handle)
    assert manifest["metadata"]["excluded_rows"][0]["row_id"] == "r000004"
    assert manifest["metadata"]["seed"] is None
    assert manifest["metadata"]["temporal_policy"] == policy().model_dump()
    for split in ("train", "validation", "test"):
        with open(session.dataset.paths[split]) as handle:
            records = [json.loads(line) for line in handle]
        assert records[0]["metadata"]["temporal"]["available_at"].endswith("Z")


def test_forecast_recipe_requires_matching_temporal_policy_and_rejects_future_inputs(tmp_path):
    from src.workbench.recipes import validate_analysis

    _, session = temporal_session(tmp_path)
    proposal = session.analysis.model_copy(deep=True)
    proposal.composition = {
        "steps": [
            {
                "operation": "forecast_labels",
                "available_at_column": "available",
                "observation_end": policy().observation_end,
            }
        ]
    }
    proposal.recipe.temporal_split = None
    with pytest.raises(ValueError, match="不能随机切分"):
        validate_analysis(session.source, proposal)
    proposal.recipe.temporal_split = policy()
    with pytest.raises(ValueError, match="必须与预测标签组合一致"):
        validate_analysis(session.source, proposal)
    from src.workbench.intake_models import FieldBinding

    proposal.recipe.inputs.append(FieldBinding(column="forecast_target_price", label="未来价格"))
    with pytest.raises(ValueError, match="不得作为模型输入"):
        validate_analysis(session.source, proposal)


def test_sample_preview_rejects_future_information_without_waiting_for_full(tmp_path):
    from src.workbench.recipes import preview_recipe

    _, session = temporal_session(tmp_path)
    source = session.source.model_copy(deep=True)
    source.rows[0].values["available"] = "2026-01-02T00:00:00Z"
    preview = preview_recipe(source, session.analysis.recipe)
    assert preview.rows[0].status == "invalid"
    assert "不能使用预测之后" in preview.rows[0].issues[-1]
