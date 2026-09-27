"""Confirmed directions reach real data previews without treating model outputs as labels."""

from dataclasses import asdict

import pytest

from src.agent.revisions import revise_data_for_iteration
from src.workbench.business_evaluation import EvaluationProtocol, Generation
from src.workbench.intake_models import Transform
from src.workbench.iterations import IterationService
from src.workbench.sources import content_digest
from tests.unit.test_business_evaluation import fixture_factory, model_paths, task  # noqa: F401
from tests.unit.test_data_intake import ScriptedModel


@pytest.fixture
def revision_environment(task, model_paths, tmp_path):  # noqa: F811
    intake, session = task
    iterations = IterationService(tmp_path / "iterations", tmp_path / "runs", tmp_path / "eval")
    report = iterations.evaluations.compare(
        session,
        model_paths,
        EvaluationProtocol("classification_exact"),
        runtime_factory=fixture_factory(session, [], lambda *_: Generation("模型错误回答")),
    )
    record = iterations._save(
        {
            "iteration_id": "it-" + "a" * 32,
            "session_id": session.session_id,
            "goal": session.goal,
            "status": "confirmed",
            "data_change": True,
            "hypothesis": "描述边缘空白可能影响转换",
            "expected_outcome": "输入规范化且不改变标签",
            "changes": "只对客户描述strip，保持答案与分组",
            "parent_evaluation_id": report.evaluation_id,
            "parent_report_digest": content_digest(asdict(report)),
            "evaluation_suite": iterations.suites.freeze(session),
        }
    )
    return intake, session, iterations, record


def revision_client(session):
    result = session.analysis.model_copy(deep=True)
    result.recipe.inputs[0].transforms = [Transform(operation="strip")]
    return ScriptedModel(
        [
            ("submit_analysis", result.model_dump()),
            ("inspect_revision_summary", {}),
            ("inspect_revision_cases", {"limit": 1}),
            ("profile_data", {}),
            ("inspect_rows", {"row_ids": []}),
            ("preview_recipe", result.recipe.model_dump()),
            ("submit_analysis", result.model_dump()),
        ]
    )


def test_confirmed_direction_executes_real_preview_and_records_actual_diff(revision_environment):
    intake, session, iterations, record = revision_environment
    client = revision_client(session)
    updated = revise_data_for_iteration(
        intake, iterations, record["iteration_id"], client, expected_revision=session.revision
    )
    assert updated.preview is not None
    assert updated.confirmed_revision is None
    assert updated.dataset is None
    assert updated.full_data.status == "stale"
    assert updated.source == session.source
    assert [row.target for row in updated.preview.rows] == [
        row.target for row in session.preview.rows
    ]
    assert all(row.target != "模型错误回答" for row in updated.preview.rows)
    saved = iterations.get(record["iteration_id"])["data_revision"]
    assert saved["changed_components"] == ["recipe"]
    assert saved["next_action"] == "review_preview"
    assert saved["to_revision"] == updated.revision
    assert saved["tool_trace"][0]["ok"] is False
    assert any(
        item["tool"] == "inspect_revision_cases" and item["ok"] for item in saved["tool_trace"]
    )
    assert (
        intake.dataset_snapshot(session.session_id, session.dataset.version).dataset
        == session.dataset
    )


def test_revision_uses_matching_historical_evidence_after_user_feedback(revision_environment):
    intake, session, iterations, record = revision_environment
    client = revision_client(session)
    current = intake.answer(session.session_id, "请保持业务标签，只规范化描述空白。")
    assert current.dataset is None
    updated = revise_data_for_iteration(
        intake, iterations, record["iteration_id"], client, expected_revision=current.revision
    )
    assert updated.revision > current.revision
    assert updated.analysis.recipe.targets == session.analysis.recipe.targets


def test_revision_rejects_stale_or_unconfirmed_scope_before_agent(revision_environment):
    intake, session, iterations, record = revision_environment
    client = ScriptedModel([])
    with pytest.raises(ValueError, match="任务已更新"):
        revise_data_for_iteration(
            intake,
            iterations,
            record["iteration_id"],
            client,
            expected_revision=session.revision - 1,
        )
    iterations._save({**record, "status": "proposed"}, record["revision"])
    with pytest.raises(ValueError, match="已确认"):
        revise_data_for_iteration(
            intake, iterations, record["iteration_id"], client, expected_revision=session.revision
        )
    assert not client.seen
    assert intake.load(session.session_id).revision == session.revision
