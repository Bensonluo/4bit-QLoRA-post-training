"""Final acceptance executes one chosen model on fixed test cases, never development."""

import csv
import io
from unittest.mock import MagicMock

import pytest

from src.agent.evaluation import assess_evaluation
from src.data.loaders import render_alpaca_prompt
from src.workbench.business_evaluation import (
    BusinessEvaluationService,
    EvaluationProtocol,
    Generation,
    assert_comparable,
)
from src.workbench.evaluation_diagnostics import EvaluationDiagnostics
from src.workbench.evaluation_suites import EvalSuiteService, evaluation_cases
from src.workbench.materialize import materialize_dataset
from tests.unit.test_business_evaluation import fixture_factory
from tests.unit.test_business_evaluation import model_paths as model_paths
from tests.unit.test_business_evaluation import task as task


@pytest.fixture
def final_task(task, tmp_path):
    _, session = task
    reference = EvalSuiteService(tmp_path / "suites").freeze(session)
    session = session.model_copy(deep=True)
    session.dataset.evaluation_suite = reference
    return session


def _runtime(session, events, generate=None, fail_close=False):
    records = evaluation_cases(session, split="test")["records"]
    answers = {render_alpaca_prompt({**row, "output": ""}): row["output"] for row in records}

    class Runtime:
        def __init__(self, model):
            events.append(("load", model.label))
            self.index = 0

        def generate(self, prompt, protocol):
            # Key lookup fails if implementation accidentally consumes validation or train.
            expected = answers[prompt]
            events.append(("prompt", prompt))
            index = self.index
            self.index += 1
            return generate(index, expected) if generate else Generation(expected)

        def close(self):
            events.append(("close",))
            if fail_close:
                raise RuntimeError("fixture release error")

    return Runtime


def test_final_only_consumes_fixed_test_and_does_not_enter_default_report_list(
    final_task, model_paths, tmp_path
):
    session = final_task
    service = BusinessEvaluationService(tmp_path / "eval")
    events = []
    final = service.evaluate_final(
        session,
        model_paths[1],
        EvaluationProtocol("classification_exact"),
        runtime_factory=_runtime(session, events),
    )
    assert final.status == "completed"
    assert final.dataset["purpose"] == "final_acceptance"
    assert final.dataset["split"] == "test"
    assert len(final.models) == 1
    assert final.models[0]["label"] == "tuned"
    assert final.models[0]["metrics"]["exact_match"] == 1
    assert final.models[0]["identity"]["adapter"]["content_digest"]
    selected = evaluation_cases(session, split="test")
    assert {row["source"]["evaluation_case_id"] for row in final.models[0]["rows"]} == {
        row["metadata"]["evaluation_case_id"] for row in selected["records"]
    }
    assert events[0] == ("load", "tuned")
    assert events[-1] == ("close",)
    assert service.get_report(final.evaluation_id).comparison_key == final.comparison_key
    assert service.list_reports() == []
    assert [r.evaluation_id for r in service.list_reports(purpose="final_acceptance")] == [
        final.evaluation_id
    ]
    dev = service.compare(
        session,
        model_paths,
        EvaluationProtocol("classification_exact"),
        runtime_factory=fixture_factory(session, []),
    )
    assert [r.evaluation_id for r in service.list_reports()] == [dev.evaluation_id]
    assert {r.evaluation_id for r in service.list_reports(purpose=None)} == {
        dev.evaluation_id,
        final.evaluation_id,
    }
    assert dev.dataset["split"] == "validation"
    assert dev.comparison_key != final.comparison_key


def test_final_denominator_preserves_generation_failure_and_truncation(
    final_task, model_paths, tmp_path
):
    def generate(index, expected):
        if index == 0:
            raise RuntimeError("fixture generation failure")
        return Generation(expected, truncated=True, generated_tokens=8)

    report = BusinessEvaluationService(tmp_path / "eval").evaluate_final(
        final_task,
        model_paths[0],
        EvaluationProtocol("classification_exact", max_new_tokens=8),
        runtime_factory=_runtime(final_task, [], generate),
    )
    outcome = report.models[0]
    total = final_task.dataset.evaluation_suite["case_counts"]["test"]
    assert report.status == "completed_with_failures"
    assert len(outcome["rows"]) == outcome["metrics"]["total"] == total
    assert outcome["metrics"]["exact_match"] == outcome["metrics"]["scorable_coverage"] == 0
    assert outcome["rows"][0]["status"] == "failed"
    assert all(row["status"] == "truncated" for row in outcome["rows"][1:])


def test_final_requires_suite_and_exactly_one_chosen_model(task, final_task, model_paths, tmp_path):
    service = BusinessEvaluationService(tmp_path / "eval")
    with pytest.raises(ValueError, match="冻结"):
        service.evaluate_final(task[1], model_paths[0], EvaluationProtocol("classification_exact"))
    with pytest.raises(ValueError, match="一个"):
        service.evaluate_final(final_task, model_paths, EvaluationProtocol("classification_exact"))
    assert not service.list_reports(purpose=None)


def test_final_cannot_feed_development_comparison_diagnostics_or_agent(
    final_task, model_paths, tmp_path
):
    report = BusinessEvaluationService(tmp_path / "eval").evaluate_final(
        final_task,
        model_paths[0],
        EvaluationProtocol("classification_exact"),
        runtime_factory=_runtime(final_task, []),
    )
    with pytest.raises(ValueError, match="最终验收"):
        assert_comparable(report, report)
    with pytest.raises(ValueError, match="独立测试集"):
        EvaluationDiagnostics(report, final_task)
    client = MagicMock()
    with pytest.raises(ValueError, match="独立测试集"):
        assess_evaluation(report, final_task, client)
    assert not client.mock_calls


def test_open_final_outputs_require_business_review_even_with_failures(
    final_task, model_paths, tmp_path
):
    def generate(index, expected):
        if index == 0:
            raise RuntimeError("fixture error")
        return Generation("开放任务回答")

    report = BusinessEvaluationService(tmp_path / "eval").evaluate_final(
        final_task,
        model_paths[0],
        EvaluationProtocol("open_review"),
        runtime_factory=_runtime(final_task, [], generate),
    )
    assert report.status == "needs_business_review"
    assert report.models[0]["metrics"]["exact_match"] is None
    assert report.models[0]["rows"][0]["status"] == "failed"
    assert all(row["status"] == "needs_business_review" for row in report.models[0]["rows"][1:])
    assert all(row["correct"] is None for row in report.models[0]["rows"])


def test_final_reserved_rows_do_not_expand_frozen_test_denominator(
    task, final_task, model_paths, tmp_path
):
    intake, original = task
    case = evaluation_cases(final_task, split="test")["records"][0]
    source_id = case["metadata"]["source_row_id"]
    related = next(row.values for row in original.full_data.source.rows if row.row_id == source_id)
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(related))
    writer.writeheader()
    writer.writerows(row.values for row in original.full_data.source.rows)
    writer.writerow({**related, "客户描述": "同一测试对象的新保留资料"})
    revised = intake.validate_full_data(
        original.session_id, original.revision, "revised.csv", buffer.getvalue().encode()
    )
    revised = intake.confirm_full_data(revised.session_id, revised.revision)
    revised.dataset = materialize_dataset(
        revised,
        registry_root=original.dataset.registry_root,
        name=original.dataset.name,
        evaluation_suite=final_task.dataset.evaluation_suite,
    )
    report = BusinessEvaluationService(tmp_path / "eval").evaluate_final(
        revised,
        model_paths[0],
        EvaluationProtocol("classification_exact"),
        runtime_factory=_runtime(revised, []),
    )
    assert report.dataset["row_count"] == final_task.dataset.evaluation_suite["case_counts"]["test"]
    assert revised.dataset.statistics["row_counts"]["test"] == report.dataset["row_count"] + 1
    assert not any("新保留资料" in row["prompt"] for row in report.models[0]["rows"])


def test_final_model_load_failure_and_release_failure_are_not_success(
    final_task, model_paths, tmp_path
):
    service = BusinessEvaluationService(tmp_path / "eval")

    def broken(_model):
        raise RuntimeError("fixture load failure")

    report = service.evaluate_final(
        final_task,
        model_paths[0],
        EvaluationProtocol("classification_exact"),
        runtime_factory=broken,
    )
    assert report.status == "completed_with_failures"
    assert len(report.models[0]["rows"]) == report.dataset["row_count"]
    assert report.models[0]["metrics"]["exact_match"] == 0
    events = []
    release = service.evaluate_final(
        final_task,
        model_paths[0],
        EvaluationProtocol("classification_exact"),
        runtime_factory=_runtime(final_task, events, fail_close=True),
    )
    assert release.status == "release_failed"
    assert len([e for e in events if e[0] == "load"]) == 1
