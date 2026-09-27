"""Comparison protocol tests; injected runtimes do not establish model quality."""

import json
from pathlib import Path

import pytest

pytest.importorskip("datasets")
pytest.importorskip("transformers")

from src.workbench.business_evaluation import (
    BusinessEvaluationService,
    EvaluationModel,
    EvaluationProtocol,
    Generation,
    _score,
    assert_comparable,
)
from src.workbench.intake_service import IntakeService
from tests.unit.test_full_data import approved


@pytest.fixture()
def task(tmp_path):
    service = IntakeService(tmp_path / "intake")
    session = approved(service)
    rows = "编号,客户描述,类别,处理结果\n" + "".join(
        f"{i},第{i}个独立问题,{'质量' if i % 2 else '物流'},补发\n" for i in range(10, 19)
    )
    session = service.validate_full_data(
        session.session_id, session.revision, "full.csv", rows.encode()
    )
    session = service.confirm_full_data(session.session_id, session.revision)
    session = service.materialize_dataset(
        session.session_id,
        session.revision,
        registry_root=tmp_path / "registry",
        validation_fraction=0.34,
        test_fraction=0.22,
    )
    return service, session


@pytest.fixture()
def model_paths(tmp_path):
    paths = []
    for name in ("base", "adapter"):
        directory = tmp_path / name
        directory.mkdir()
        (directory / "config.json").write_text("{}")
        (directory / "pytorch_model.bin").write_bytes(f"protocol-test-{name}".encode())
        paths.append(str(directory))
    return [
        EvaluationModel("base", paths[0]),
        EvaluationModel("tuned", paths[0], adapter_path=paths[1]),
    ]


def fixture_factory(session, events, generate=None, close_error=False):
    from src.data.loaders import render_alpaca_prompt

    records = [
        json.loads(line)
        for line in Path(session.dataset.paths["validation"]).read_text().splitlines()
    ]
    answers = {render_alpaca_prompt({**row, "output": ""}): row["output"] for row in records}

    class Runtime:
        def __init__(self, model):
            self.model = model
            self.index = 0
            events.append(("load", model.label))

        def generate(self, prompt, protocol):
            events.append(("generate", self.model.label, prompt, protocol.max_new_tokens))
            index = self.index
            self.index += 1
            return (
                generate(self.model.label, index, answers[prompt])
                if generate
                else Generation(answers[prompt])
            )

        def close(self):
            events.append(("close", self.model.label))
            if close_error:
                raise RuntimeError("fixture release failure")

    return Runtime


def test_identical_prompts_sequential_release_and_failure_denominator(task, model_paths, tmp_path):
    _, session = task
    events = []

    def generation(label, index, expected):
        if label == "base":
            if index == 0:
                raise RuntimeError("fixture generation failure")
            if index == 1:
                return Generation(expected, truncated=True, generated_tokens=8)
            return Generation(expected + "\n### Input:\nextra")
        return Generation(expected)

    service = BusinessEvaluationService(tmp_path / "evaluations")
    report = service.compare(
        session,
        model_paths,
        EvaluationProtocol("classification_exact", max_new_tokens=8),
        runtime_factory=fixture_factory(session, events, generation),
    )
    assert report.status == "completed_with_failures"
    assert report.dataset["split"] == "validation"
    assert events.index(("close", "base")) < events.index(("load", "tuned"))
    base_prompts = [event[2] for event in events if event[:2] == ("generate", "base")]
    tuned_prompts = [event[2] for event in events if event[:2] == ("generate", "tuned")]
    assert base_prompts == tuned_prompts
    assert all(prompt.endswith("### Response:\n") for prompt in base_prompts)
    total = report.dataset["row_count"]
    base, tuned = report.models
    assert len(base["rows"]) == len(tuned["rows"]) == total
    assert base["metrics"]["exact_match"] == 0
    assert base["metrics"]["scorable_coverage"] == (total - 2) / total
    assert tuned["metrics"]["exact_match"] == 1
    assert base["rows"][1]["truncated"] is True
    assert base["rows"][-1]["output"].endswith("extra")
    assert base["rows"][-1]["correct"] is False
    assert base["identity"]["base"]["content_digest"] == tuned["identity"]["base"]["content_digest"]
    assert tuned["identity"]["adapter"]["content_digest"]
    saved = json.loads((service.root / f"{report.evaluation_id}.json").read_text())
    assert saved["models"][0]["rows"] == base["rows"]
    assert service.get_report(report.evaluation_id).comparison_key == report.comparison_key
    assert len(service.list_reports(session.dataset.version)) == 1
    assert service.list_reports("another-version") == []
    with pytest.raises(ValueError, match="无效"):
        service.get_report("../elsewhere")


def test_different_generation_protocols_cannot_be_compared(task, model_paths, tmp_path):
    _, session = task
    service = BusinessEvaluationService(tmp_path / "evaluations")
    first = service.compare(
        session,
        model_paths,
        EvaluationProtocol("classification_exact", max_new_tokens=8),
        runtime_factory=fixture_factory(session, []),
    )
    second = service.compare(
        session,
        model_paths,
        EvaluationProtocol("classification_exact", max_new_tokens=16),
        runtime_factory=fixture_factory(session, []),
    )
    with pytest.raises(ValueError, match="协议不同"):
        assert_comparable(first, second)
    assert_comparable(first, first)


def test_failed_model_load_keeps_all_development_rows(task, model_paths, tmp_path):
    _, session = task
    empty = tmp_path / "empty_model"
    empty.mkdir()
    events = []
    models = [EvaluationModel("broken", str(empty)), model_paths[1]]
    report = BusinessEvaluationService(tmp_path / "eval").compare(
        session,
        models,
        EvaluationProtocol("classification_exact"),
        runtime_factory=fixture_factory(session, events),
    )
    assert report.status == "completed_with_failures"
    broken = report.models[0]
    assert broken["metrics"]["total"] == session.dataset.statistics["row_counts"]["validation"]
    assert broken["metrics"]["exact_match"] == broken["metrics"]["generation_coverage"] == 0
    assert all(row["status"] == "failed" for row in broken["rows"])
    assert events[0] == ("load", "tuned")


def test_uncertain_release_prevents_loading_next_model(task, model_paths, tmp_path):
    _, session = task
    events = []
    report = BusinessEvaluationService(tmp_path / "eval").compare(
        session,
        model_paths,
        EvaluationProtocol("classification_exact"),
        runtime_factory=fixture_factory(session, events, close_error=True),
    )
    assert report.status == "release_failed"
    assert ("load", "tuned") not in events
    with pytest.raises(ValueError, match="尚未完整"):
        assert_comparable(report, report)


def test_model_content_change_between_loads_is_not_hidden_by_identity_cache(
    task, model_paths, tmp_path
):
    _, session = task
    events = []
    base_runtime = fixture_factory(session, events)

    class ChangingRuntime(base_runtime):
        def close(self):
            super().close()
            if self.model.label == "base":
                (Path(self.model.base_model) / "pytorch_model.bin").write_bytes(
                    b"changed-after-first-model"
                )

    report = BusinessEvaluationService(tmp_path / "eval").compare(
        session,
        model_paths,
        EvaluationProtocol("classification_exact"),
        runtime_factory=ChangingRuntime,
    )
    assert ("load", "tuned") not in events
    assert report.status == "completed_with_failures"
    assert "模型文件发生变化" in report.models[1]["errors"][0]
    assert report.models[1]["metrics"]["exact_match"] == 0


def test_open_tasks_collect_outputs_without_claiming_quality(task, model_paths, tmp_path):
    _, session = task
    report = BusinessEvaluationService(tmp_path / "eval").compare(
        session,
        model_paths,
        EvaluationProtocol("open_review"),
        runtime_factory=fixture_factory(session, []),
    )
    assert report.status == "completed"
    assert all(model["metrics"]["exact_match"] is None for model in report.models)
    assert all(
        row["status"] == "needs_business_review" for model in report.models for row in model["rows"]
    )
    assert any("不宣称质量达标" in note for note in report.notes)


def test_tampered_dataset_is_rejected_before_loading_models(task, model_paths, tmp_path):
    _, session = task
    path = Path(session.dataset.paths["validation"])
    path.write_text(path.read_text() + "\n")
    events = []
    with pytest.raises(ValueError, match="modified"):
        BusinessEvaluationService(tmp_path / "eval").compare(
            session,
            model_paths,
            EvaluationProtocol("classification_exact"),
            runtime_factory=lambda model: events.append(model),
        )
    assert events == []


def test_structured_score_checks_selected_fields_and_json_types():
    protocol = EvaluationProtocol("json_fields_exact", fields=("quantity", "approved"))
    assert _score(
        '{"quantity": 1, "approved": true}', '{"approved":true,"quantity":1}', protocol
    ) == (True, {"quantity": True, "approved": True})
    correct, fields = _score(
        '{"quantity": true, "approved": true}', '{"approved":true,"quantity":1}', protocol
    )
    assert correct is False and fields["quantity"] is False
    with pytest.raises(ValueError, match="重复"):
        _score(
            '{"quantity":1,"quantity":2,"approved":true}',
            '{"approved":true,"quantity":1}',
            protocol,
        )
    with pytest.raises(ValueError):
        _score('{"quantity":NaN,"approved":true}', '{"approved":true,"quantity":1}', protocol)


def test_unsupported_scorer_is_explicit(task, model_paths, tmp_path):
    _, session = task
    with pytest.raises(ValueError, match="不受支持"):
        BusinessEvaluationService(tmp_path / "eval").compare(
            session, model_paths, EvaluationProtocol("invented_business_score")
        )


def test_structured_protocol_runs_end_to_end(task, model_paths, tmp_path):
    service, session = task
    original = session.full_data.sources["main"]
    payload = (
        service.root / session.session_id / "full" / f"{original.digest}.{original.format}"
    ).read_bytes()
    plan = session.analysis.model_copy(deep=True)
    plan.recipe.output_format = "json"
    session = service.apply_analysis(session, plan)
    session = service.confirm(session.session_id, session.revision)
    session = service.validate_full_data(session.session_id, session.revision, "full.csv", payload)
    session = service.confirm_full_data(session.session_id, session.revision)
    session = service.materialize_dataset(
        session.session_id,
        session.revision,
        registry_root=tmp_path / "registry",
        validation_fraction=0.34,
        test_fraction=0.22,
    )

    def generation(label, index, expected):
        if label == "base":
            return Generation("not-json" if index == 0 else "{}")
        return Generation(expected)

    report = BusinessEvaluationService(tmp_path / "eval").compare(
        session,
        model_paths,
        EvaluationProtocol("json_fields_exact", fields=("类别",)),
        runtime_factory=fixture_factory(session, [], generation),
    )
    base, tuned = report.models
    assert base["rows"][0]["status"] == "failed"
    assert base["rows"][0]["output"] == "not-json"
    assert base["metrics"]["exact_match"] == 0
    assert base["metrics"]["field_accuracy"] == {"类别": 0}
    assert tuned["metrics"]["field_accuracy"] == {"类别": 1}


def test_suite_comparison_excludes_training_provenance_but_binds_actual_answers_and_protocol(
    task, model_paths, tmp_path
):
    from copy import deepcopy

    from src.workbench.business_evaluation import comparison_key
    from src.workbench.sources import content_digest

    service = BusinessEvaluationService(tmp_path / "eval")
    report = service.compare(
        task[1],
        model_paths,
        EvaluationProtocol("classification_exact"),
        runtime_factory=fixture_factory(task[1], []),
    )
    legacy_revised = deepcopy(report)
    legacy_revised.dataset["version"] = "other-training-version"
    legacy_revised.comparison_key = comparison_key(legacy_revised.dataset, legacy_revised.protocol)
    with pytest.raises(ValueError):
        assert_comparable(report, legacy_revised)
    report.dataset["evaluation_suite"] = {
        "suite_id": "fixed-suite",
        "cases_digest": "fixed-cases",
    }
    report.protocol["evaluation_key"] = "fixed-dev"
    report.protocol["answers_digest"] = content_digest(
        [row["expected"] for row in report.models[0]["rows"]]
    )
    report.comparison_key = comparison_key(report.dataset, report.protocol)
    revised = deepcopy(report)
    revised.dataset.update(
        version="next-training-version",
        source_digest="updated-source",
        recipe_digest="next-processing-recipe",
        split_sha256="new-metadata",
    )
    revised.comparison_key = comparison_key(revised.dataset, revised.protocol)
    assert_comparable(report, revised)
    service._save(revised)
    assert service.get_report(revised.evaluation_id).dataset["version"] == "next-training-version"
    assert len(service.list_reports(suite_id="fixed-suite")) == 1
    assert service.list_reports(suite_id="another-suite") == []
    for field in ("answers_digest", "prompt_digest", "max_new_tokens"):
        changed = deepcopy(revised)
        changed.protocol[field] = 17 if field == "max_new_tokens" else "changed"
        changed.comparison_key = comparison_key(changed.dataset, changed.protocol)
        with pytest.raises(ValueError, match="协议不同"):
            assert_comparable(report, changed)
    legacy = deepcopy(report)
    legacy.dataset.pop("evaluation_suite")
    legacy.comparison_key = comparison_key(legacy.dataset, legacy.protocol)
    with pytest.raises(ValueError):
        assert_comparable(legacy, report)


def suite_rounds(task, tmp_path):
    """Real registry versions share fixed questions but gain independent/held-out rows."""
    import csv
    import io

    from src.workbench.evaluation_suites import EvalSuiteService
    from src.workbench.materialize import materialize_dataset

    service, original = task
    reference = EvalSuiteService(tmp_path / "suites").freeze(original)
    anchor = original.model_copy(deep=True)
    anchor.dataset.evaluation_suite = reference
    dev = json.loads(Path(original.dataset.paths["validation"]).read_text().splitlines()[0])
    source_id = dev["metadata"]["source_row_id"]
    related = next(row.values for row in original.full_data.source.rows if row.row_id == source_id)
    columns = list(original.full_data.source.rows[0].values)
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=columns)
    writer.writeheader()
    writer.writerows(row.values for row in original.full_data.source.rows)
    writer.writerow({**related, "客户描述": "相同客户另一条保留资料"})
    writer.writerow({**related, "编号": "9999", "客户描述": "新增独立训练问题"})
    revised = service.validate_full_data(
        original.session_id, original.revision, "next.csv", buffer.getvalue().encode()
    )
    revised = service.confirm_full_data(revised.session_id, revised.revision)
    revised.dataset = materialize_dataset(
        revised,
        registry_root=original.dataset.registry_root,
        name=original.dataset.name,
        evaluation_suite=reference,
    )
    return anchor, revised


def test_fixed_suite_compares_actual_different_train_versions_without_expanding_cases(
    task, model_paths, tmp_path
):
    from src.workbench.evaluation_diagnostics import EvaluationDiagnostics

    anchor, revised = suite_rounds(task, tmp_path)
    service = BusinessEvaluationService(tmp_path / "eval")
    reports = [
        service.compare(
            session,
            model_paths,
            EvaluationProtocol("classification_exact"),
            runtime_factory=fixture_factory(
                session,
                [],
                lambda label, index, expected: Generation(
                    "错误答案" if label == "base" else expected
                ),
            ),
        )
        for session in (anchor, revised)
    ]
    assert anchor.dataset.version != revised.dataset.version
    assert revised.dataset.statistics["row_counts"]["validation"] > reports[1].dataset["row_count"]
    assert reports[0].dataset["row_count"] == reports[1].dataset["row_count"]
    assert reports[0].dataset["source_digest"] != reports[1].dataset["source_digest"]
    assert_comparable(*reports)
    assert [row["source"]["evaluation_case_id"] for row in reports[0].models[0]["rows"]] == [
        row["source"]["evaluation_case_id"] for row in reports[1].models[0]["rows"]
    ]
    for report, session in zip(reports, (anchor, revised)):
        diagnostics = EvaluationDiagnostics(report, session)
        assert diagnostics.summary()["bad_case_count"] == report.dataset["row_count"]
        case = diagnostics.read_cases()["cases"][0]
        assert case["original_rows"][0]["source_digest"] == session.full_data.sources["main"].digest
    with pytest.raises(ValueError, match="数据版本"):
        EvaluationDiagnostics(reports[0], revised)
