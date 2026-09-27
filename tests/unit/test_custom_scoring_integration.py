"""Real OS scoring across development diagnostics, frozen acceptance and iteration."""

from copy import deepcopy
from dataclasses import asdict
from unittest.mock import MagicMock
from uuid import uuid4

import pytest

from src.workbench.acceptance import AcceptanceService
from src.workbench.business_evaluation import (
    BusinessEvaluationService,
    EvaluationModel,
    EvaluationProtocol,
    Generation,
    assert_comparable,
    comparison_key,
)
from src.workbench.business_scoring import ScoringRecipe, ScoringService
from src.workbench.evaluation_diagnostics import EvaluationDiagnostics
from src.workbench.evaluation_suites import EvalSuiteService, evaluation_cases
from src.workbench.iterations import IterationService
from src.workbench.sources import content_digest
from tests.unit.test_acceptance import criteria, runtime
from tests.unit.test_business_evaluation import fixture_factory
from tests.unit.test_business_evaluation import model_paths as model_paths
from tests.unit.test_business_evaluation import task as task

CODE = """def transform(rows, config):
    result = []
    for row in rows:
        if row["output"] == "explode":
            raise ValueError("cannot score this output")
        correct = row["output"].rstrip("。") == row["expected"]
        score = 1.0 if correct else config["partial_score"] if row["output"] == "partial" else 0.0
        reason = "accepted" if correct else "partial" if row["output"] == "partial" else "rejected"
        result.append({**row, "score": score, "reason": reason})
    return result
"""


def _confirmed(root, session, partial=0.5):
    scoring = ScoringService(root)
    example = evaluation_cases(session)["records"][0]
    recipe = ScoringRecipe(
        business_standard="允许末尾句号，部分正确按约定比例给分，但低于逐题通过线。",
        source_code=CODE,
        config={"partial_score": partial},
        pass_threshold=0.75,
        examples=[
            {
                "name": "business",
                "input": example["input"],
                "expected": example["output"],
                "output": example["output"] + "。",
                "score": 1.0,
                "reason": "accepted",
                "kind": "business",
            },
            {
                "name": "counterexample",
                "input": example["input"],
                "expected": example["output"],
                "output": "partial",
                "score": partial,
                "reason": "partial",
                "kind": "counterexample",
            },
        ],
    )
    draft = scoring.draft(session, recipe)
    assert draft["validation"]["status"] == "passed"
    reference = scoring.confirm(draft["scoring_id"], session)
    return EvaluationProtocol("custom_rules", custom_scoring=reference), recipe


@pytest.fixture
def custom(task, tmp_path):
    session = task[1].model_copy(deep=True)
    session.dataset.evaluation_suite = EvalSuiteService(tmp_path / "suites").freeze(task[1])
    protocol, recipe = _confirmed(tmp_path / "scoring", session)
    return session, protocol, recipe


def test_real_custom_scoring_preserves_denominator_and_diagnostics_recompute_in_sandbox(
    custom, model_paths, tmp_path
):
    session, protocol, recipe = custom

    def generate(label, index, expected):
        if label == "base":
            if index == 0:
                raise RuntimeError("fixture generation failure")
            if index == 1:
                return Generation(expected, truncated=True)
            return Generation("partial")
        return Generation(expected + "。")

    service = BusinessEvaluationService(tmp_path / "eval")
    report = service.compare(
        session, model_paths, protocol, runtime_factory=fixture_factory(session, [], generate)
    )
    assert report.status == "completed_with_failures"
    assert report.protocol["custom_scoring_recipe"] == recipe.model_dump()
    assert report.protocol["custom_scoring_digest"] == content_digest(recipe.model_dump())
    base, tuned = report.models
    count = report.dataset["row_count"]
    assert base["metrics"]["business_score"] == 0.5 * (count - 2) / count
    assert base["metrics"]["pass_rate"] == 0
    assert tuned["metrics"]["pass_rate"] == tuned["metrics"]["business_score"] == 1
    assert all(model["metrics"]["exact_match"] is None for model in report.models)
    assert [row["business_score"] for row in base["rows"][:2]] == [0.0, 0.0]
    diagnostics = EvaluationDiagnostics(report, session)
    cases = diagnostics.read_cases()["cases"]
    assert any(
        case["scoring_reason"] == "partial" and case["business_score"] == 0.5 for case in cases
    )
    assert diagnostics.summary()["protocol"]["custom_scoring_recipe"]["source_code"] == CODE
    altered = deepcopy(report)
    altered.models[1]["rows"][0]["business_score"] = 0.8
    with pytest.raises(ValueError, match="隔离重算"):
        EvaluationDiagnostics(altered, session)
    altered = deepcopy(report)
    altered.protocol["custom_scoring_recipe"]["config"]["partial_score"] = 0.7
    altered.comparison_key = comparison_key(altered.dataset, altered.protocol)
    with pytest.raises(ValueError, match="完整规则"):
        EvaluationDiagnostics(altered, session)


def test_custom_execution_failure_never_falls_back_to_exact_match(custom, model_paths, tmp_path):
    session, protocol, _ = custom
    report = BusinessEvaluationService(tmp_path / "eval").compare(
        session,
        model_paths,
        protocol,
        runtime_factory=fixture_factory(
            session, [], lambda label, index, expected: Generation("explode")
        ),
    )
    assert report.status == "completed_with_failures"
    for model in report.models:
        assert model["metrics"]["pass_rate"] == model["metrics"]["business_score"] == 0
        assert model["metrics"]["exact_match"] is None
        assert all(
            row["status"] == "failed" and row["business_score"] == 0 for row in model["rows"]
        )
        assert all("自定义评分失败" in row["error"] for row in model["rows"])
    diagnostics = EvaluationDiagnostics(report, session)
    assert all(case["error_type"] == "scoring_error" for case in diagnostics.read_cases()["cases"])


def test_unconfirmed_or_noncustom_rules_rejected_before_model_load(custom, model_paths, tmp_path):
    session, protocol, _ = custom
    for scorer, reference in [
        ("custom_rules", None),
        ("classification_exact", protocol.custom_scoring),
        ("open_review", {}),
    ]:
        with pytest.raises(ValueError):
            EvaluationProtocol(scorer, custom_scoring=reference).validate()
    wrong = {**protocol.custom_scoring, "spec_digest": "changed"}
    factory = MagicMock()
    with pytest.raises(ValueError, match="指纹"):
        BusinessEvaluationService(tmp_path / "eval").compare(
            session,
            model_paths,
            EvaluationProtocol("custom_rules", custom_scoring=wrong),
            runtime_factory=factory,
        )
    factory.assert_not_called()


def test_acceptance_freezes_full_recipe_and_uses_pass_rate_not_average(
    custom, model_paths, tmp_path
):
    session, protocol, recipe = custom
    service = AcceptanceService(tmp_path / "acceptance", tmp_path / "eval")
    with pytest.raises(ValueError, match="pass_rate"):
        service.prepare(session, model_paths[0], protocol, criteria())
    record = service.prepare(
        session, model_paths[0], protocol, criteria(metric="pass_rate", minimum_score=0.4)
    )
    assert record["scoring_identity"]["custom_scoring_recipe"] == recipe.model_dump()
    outcome = service.run(
        record["acceptance_id"],
        session,
        runtime_factory=runtime(service, record, [], lambda index, expected: Generation("partial")),
    )
    assert outcome["status"] == "completed", outcome
    assert outcome["result"]["decision"] == "failed"
    assert outcome["result"]["metric"] == "pass_rate"
    assert outcome["result"]["score"] == 0
    assert outcome["result"]["business_score"] == 0.5
    assert outcome["report"]["protocol"]["custom_scoring_digest"] == content_digest(
        recipe.model_dump()
    )
    final = service.evaluations.get_report(outcome["evaluation_id"])
    assert final.dataset["purpose"] == "final_acceptance"
    with pytest.raises(ValueError, match="独立测试集"):
        EvaluationDiagnostics(final, session)
    assert not service.evaluations.list_reports()


def test_different_rule_content_blocks_comparison_and_iteration_binding(
    custom, model_paths, tmp_path, monkeypatch
):
    session, protocol, _ = custom
    revised, _ = _confirmed(tmp_path / "another-rule", session, partial=0.25)
    evaluations = BusinessEvaluationService(tmp_path / "eval")
    models = [
        model_paths[0],
        model_paths[1],
        EvaluationModel("new", model_paths[1].base_model, model_paths[1].adapter_path),
    ]
    reports = [
        evaluations.compare(
            session,
            models,
            rule,
            runtime_factory=fixture_factory(
                session, [], lambda label, index, expected: Generation("partial")
            ),
        )
        for rule in (protocol, revised)
    ]
    with pytest.raises(ValueError, match="协议不同"):
        assert_comparable(*reports)
    iterations = IterationService(tmp_path / "iterations", tmp_path / "training", evaluations.root)
    parent = {"run_id": "parent", "status": "succeeded", "model_path": model_paths[0].base_model}
    child = {"run_id": "child", "status": "succeeded"}
    monkeypatch.setattr(
        iterations.training,
        "get_status",
        lambda identity: parent if identity == "parent" else child,
    )
    monkeypatch.setattr(iterations, "_has_model", lambda *args: True)
    record = iterations._save(
        {
            "iteration_id": "it-" + uuid4().hex,
            "session_id": session.session_id,
            "goal": session.goal,
            "status": "running",
            "new_run_id": "child",
            "parent_run_id": "parent",
            "model_path": model_paths[0].base_model,
            "parent_evaluation_id": reports[0].evaluation_id,
            "parent_report_digest": content_digest(asdict(reports[0])),
            "evaluation_suite": session.dataset.evaluation_suite,
        }
    )
    with pytest.raises(ValueError, match="换规则"):
        iterations.bind_evaluation(record["iteration_id"], session, reports[1].evaluation_id)
    same = iterations.bind_evaluation(record["iteration_id"], session, reports[0].evaluation_id)
    assert same["status"] == "evaluated"
    assert same["decision_metric"] == "pass_rate"
    assert all(item["metrics"]["exact_match"] is None for item in same["results"])
