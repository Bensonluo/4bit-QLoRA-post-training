"""Diagnoses use complete development evidence and preserve uncertainty."""

import json
from copy import deepcopy

import pytest

pytest.importorskip("datasets")
pytest.importorskip("transformers")

from src.workbench.business_evaluation import (
    BusinessEvaluationService,
    EvaluationProtocol,
    Generation,
)
from src.workbench.evaluation_diagnostics import EvaluationDiagnostics
from src.workbench.sources import content_digest
from tests.unit.test_business_evaluation import fixture_factory
from tests.unit.test_business_evaluation import model_paths as model_paths
from tests.unit.test_business_evaluation import task as task


def make_report(task, models, root, generate=None, scorer="classification_exact"):
    _, session = task

    def output(label, index, expected):
        if label == "base":
            if index == 0:
                return Generation(
                    expected + "\n### Input:\nextra", truncated=True, generated_tokens=32
                )
            if index == 1:
                raise RuntimeError("observed generation failure")
            return Generation("wrong answer")
        return Generation(expected)

    return BusinessEvaluationService(root).compare(
        session,
        models,
        EvaluationProtocol(scorer),
        runtime_factory=fixture_factory(session, [], generate or output),
    )


def test_complete_counts_and_locatable_cases_never_strip_model_output(task, model_paths, tmp_path):
    report = make_report(task, model_paths, tmp_path / "eval")
    diagnostics = EvaluationDiagnostics(report, task[1])
    summary = diagnostics.summary()
    assert summary["bad_case_count"] == summary["error_count"] == 3
    assert summary["models"]["base"]["counts"]["truncated"] == 1
    assert summary["models"]["base"]["counts"]["generation_error"] == 1
    assert summary["models"]["base"]["counts"]["mismatch"] == 1
    assert summary["models"]["tuned"]["counts"]["correct"] == 3
    assert all(fact["kind"] == "observed" for fact in summary["facts"])
    assert all(item["kind"] == "needs_verification" for item in summary["hypotheses"])
    truncation = next(h for h in summary["hypotheses"] if "不等于只需增加长度" in h["statement"])
    assert truncation["based_on"] == "truncated"
    # 输出复现了提示的「### Input:」结构标记——也是回声行为，应有专门假设。
    assert summary["models"]["base"]["counts"].get("instruction_echo") == 1
    echo = next(h for h in summary["hypotheses"] if h["based_on"] == "instruction_echo")
    assert echo["models"] == ["base"]
    case = diagnostics.read_cases(error_type="truncated")["cases"][0]
    assert case["evidence_id"] == "base:0"
    assert case["output"].endswith("### Input:\nextra")
    assert case["original_rows"][0]["values"]["客户描述"] in case["prompt"]
    assert case["original_rows"][0]["source_digest"] == task[1].full_data.sources["main"].digest
    assert case["task"] == task[1].analysis.task.model_dump()
    assert case["processing_plan"]["recipe"] == task[1].analysis.recipe.model_dump()


def test_pagination_has_complete_stable_coverage(task, model_paths, tmp_path):
    diagnostics = EvaluationDiagnostics(make_report(task, model_paths, tmp_path / "eval"), task[1])
    offset, seen = 0, []
    while True:
        page = diagnostics.read_cases(offset=offset, limit=1)
        assert page["total"] == 3
        seen.extend(case["evidence_id"] for case in page["cases"])
        if not page["has_more"]:
            assert page["next_offset"] is None
            break
        offset = page["next_offset"]
    assert seen == ["base:0", "base:1", "base:2"]
    assert diagnostics.read_cases(model="tuned")["total"] == 0
    with pytest.raises(ValueError):
        diagnostics.read_cases(limit=51)
    with pytest.raises(ValueError):
        diagnostics.read_cases(model="unknown")


def test_large_output_uses_explicit_lossless_chunks(task, model_paths, tmp_path):
    output = "完整输出" * 10000
    report = make_report(
        task,
        model_paths,
        tmp_path / "eval",
        generate=lambda label, index, expected: Generation(output),
    )
    diagnostics = EvaluationDiagnostics(report, task[1])
    page = diagnostics.read_cases(limit=1, max_bytes=4096)
    stub = page["cases"][0]
    assert stub["content_status"] == "requires_chunks"
    assert stub["content_bytes"] > 4096
    offset, pieces = 0, []
    while True:
        chunk = diagnostics.read_case_content(stub["evidence_id"], offset=offset, limit=1024)
        assert len(chunk["content"]) <= 1024
        pieces.append(chunk["content"])
        if not chunk["has_more"]:
            break
        offset = chunk["next_offset"]
    original = json.loads("".join(pieces))
    assert original["output"] == output
    assert original["evidence_id"] == stub["evidence_id"]


@pytest.mark.parametrize(
    "mutation", ["version", "protocol", "test_boundary", "prompt", "row_missing", "source"]
)
def test_wrong_snapshot_protocol_or_row_evidence_is_rejected(task, model_paths, tmp_path, mutation):
    report = make_report(task, model_paths, tmp_path / "eval")
    if mutation == "version":
        report.dataset["version"] = "different"
    elif mutation == "protocol":
        report.protocol["max_new_tokens"] += 1
    elif mutation == "test_boundary":
        report.dataset["split"] = "test"
        report.dataset["purpose"] = "final_test"
        report.comparison_key = content_digest(
            {"dataset": report.dataset, "protocol": report.protocol}
        )
    elif mutation == "prompt":
        report.models[0]["rows"][0]["prompt"] = "another prompt"
    elif mutation == "row_missing":
        report.models[0]["rows"].pop()
    else:
        report.models[0]["rows"][0]["source"]["source_row_id"] = "r999999"
    with pytest.raises(ValueError):
        EvaluationDiagnostics(report, task[1])


def test_open_review_is_not_mislabeled_as_incorrect_quality(task, model_paths, tmp_path):
    report = make_report(
        task,
        model_paths,
        tmp_path / "eval",
        scorer="open_review",
        generate=lambda label, index, expected: Generation("开放回答"),
    )
    diagnostics = EvaluationDiagnostics(report, task[1])
    summary = diagnostics.summary()
    assert summary["bad_case_count"] == 6
    assert summary["error_count"] == 0
    assert all(
        case["error_type"] == "needs_business_review" for case in diagnostics.read_cases()["cases"]
    )
    assert all(case["correct"] is None for case in diagnostics.read_cases()["cases"])


@pytest.mark.parametrize(
    "metric,value",
    [
        ("exact_match", 1.0),
        ("total", 1),
        ("scored", 3),
        ("generated", 3),
        ("scorable_coverage", 1.0),
        ("generation_coverage", 1.0),
        ("field_accuracy", {"undeclared": 1.0}),
        ("scored", True),
    ],
)
def test_saved_aggregate_cannot_override_complete_row_evidence(
    task, model_paths, tmp_path, metric, value
):
    service = BusinessEvaluationService(tmp_path / "eval")
    report = make_report(task, model_paths, service.root)
    # The protocol fingerprint does not bind model metrics: validate the actual
    # persisted report against all rows, including generation failure/truncation.
    report.models[0]["metrics"][metric] = value
    service._save(report)
    with pytest.raises(ValueError, match="业务指标与全量评测行不一致"):
        service.get_report(report.evaluation_id)
    with pytest.raises(ValueError, match="业务指标与全量评测行不一致"):
        EvaluationDiagnostics(report, task[1])


def test_open_review_cannot_acquire_a_counterfeit_quality_score(task, model_paths, tmp_path):
    report = make_report(
        task,
        model_paths,
        tmp_path / "eval",
        scorer="open_review",
        generate=lambda label, index, expected: Generation("开放回答"),
    )
    report.models[0]["metrics"]["exact_match"] = 1.0
    with pytest.raises(ValueError, match="业务指标与全量评测行不一致"):
        EvaluationDiagnostics(report, task[1])


def test_eval_show_refuses_counterfeit_aggregate_without_printing_evidence(
    task, model_paths, tmp_path, monkeypatch, capsys
):
    import sys

    from scripts import data_intake

    service = BusinessEvaluationService(tmp_path / "eval")
    report = make_report(task, model_paths, service.root)
    report.models[0]["metrics"]["exact_match"] = 1.0
    service._save(report)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "data_intake.py",
            "--store",
            str(task[0].root),
            "--evaluation-root",
            str(service.root),
            "eval-show",
            report.evaluation_id,
        ],
    )
    assert data_intake.main() == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "业务指标与全量评测行不一致" in captured.err


def test_json_field_aggregate_is_verified_against_all_rows(task, model_paths, tmp_path):
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
    report = BusinessEvaluationService(tmp_path / "eval").compare(
        session,
        model_paths,
        EvaluationProtocol("json_fields_exact", fields=("类别",)),
        runtime_factory=fixture_factory(
            session,
            [],
            lambda label, index, expected: Generation("not-json" if index == 0 else expected),
        ),
    )
    EvaluationDiagnostics(report, session)
    report.models[0]["metrics"]["field_accuracy"]["类别"] = 1.0
    with pytest.raises(ValueError, match="业务指标与全量评测行不一致"):
        EvaluationDiagnostics(report, session)


def test_custom_scoring_coverage_cannot_drop_failed_rows(task, model_paths, tmp_path):
    from tests.unit.test_custom_scoring_integration import _confirmed

    session = task[1]
    protocol, _ = _confirmed(tmp_path / "scoring", session)

    # Generate real sandbox scores using the same failed/truncated output pattern.
    def generation(label, index, expected):
        if index == 0:
            raise RuntimeError("generation failure")
        return Generation(expected, truncated=index == 1)

    report = BusinessEvaluationService(tmp_path / "custom-eval").compare(
        session,
        model_paths,
        protocol,
        runtime_factory=fixture_factory(session, [], generation),
    )
    EvaluationDiagnostics(report, session)
    report.models[0]["metrics"]["scorable_coverage"] = 1.0
    with pytest.raises(ValueError, match="业务指标与全量评测行不一致"):
        EvaluationDiagnostics(report, session)


def test_iteration_cannot_bind_or_adopt_counterfeit_aggregate(
    task, model_paths, tmp_path, monkeypatch
):
    from dataclasses import asdict, replace
    from uuid import uuid4

    from src.workbench.evaluation_suites import EvalSuiteService
    from src.workbench.iterations import IterationService

    session = task[1].model_copy(deep=True)
    session.dataset.evaluation_suite = EvalSuiteService(tmp_path / "suites").freeze(task[1])
    service = IterationService(tmp_path / "iterations", tmp_path / "runs", tmp_path / "eval")
    parent_models = [model_paths[0], replace(model_paths[1], label="parent")]
    parent_report = make_report((task[0], session), parent_models, service.evaluations.root)
    adapter = tmp_path / "new-adapter"
    adapter.mkdir()
    (adapter / "adapter_model.safetensors").write_bytes(b"new-model")
    models = [*parent_models, replace(model_paths[1], label="new", adapter_path=str(adapter))]
    report = make_report(
        (task[0], session),
        models,
        service.evaluations.root,
        generate=lambda label, index, expected: Generation(
            "wrong answer" if label == "new" else expected
        ),
    )
    parent = {
        "run_id": "parent",
        "status": "succeeded",
        "model_path": model_paths[0].base_model,
        "output_dir": model_paths[1].adapter_path,
    }
    child = {**parent, "run_id": "child", "output_dir": str(adapter)}
    monkeypatch.setattr(
        service.training, "get_status", lambda identity: parent if identity == "parent" else child
    )
    record = service._save(
        {
            "iteration_id": "it-" + uuid4().hex,
            "session_id": session.session_id,
            "goal": session.goal,
            "status": "running",
            "new_run_id": "child",
            "parent_run_id": "parent",
            "parent_evaluation_id": parent_report.evaluation_id,
            "parent_report_digest": content_digest(asdict(parent_report)),
            "evaluation_suite": session.dataset.evaluation_suite,
            "model_path": model_paths[0].base_model,
        }
    )
    # Counterfeit a successful new-model score while preserving its genuine wrong outputs.
    report.models[-1]["metrics"]["exact_match"] = 1.0
    service.evaluations._save(report)
    with pytest.raises(ValueError, match="业务指标与全量评测行不一致"):
        service.bind_evaluation(record["iteration_id"], session, report.evaluation_id)
    assert service.get(record["iteration_id"])["status"] == "running"
    with pytest.raises(ValueError, match="完成同题对照"):
        service.decide(record["iteration_id"], "adopt", "不能采用虚假的完整正确率")


def test_read_api_does_not_mutate_original_report_or_evidence(task, model_paths, tmp_path):
    report = make_report(task, model_paths, tmp_path / "eval")
    before = deepcopy(report)
    diagnostics = EvaluationDiagnostics(report, task[1])
    case = diagnostics.read_cases()["cases"][0]
    case["original_rows"][0]["values"]["客户描述"] = "caller edit"
    diagnostics.summary()["models"]["base"]["counts"]["mismatch"] = 999
    assert diagnostics.summary()["models"]["base"]["counts"]["mismatch"] == 1
    assert (
        diagnostics.read_cases()["cases"][0]["original_rows"][0]["values"]["客户描述"]
        != "caller edit"
    )
    assert report == before


def training_report(task, model_paths, tmp_path):
    import hashlib
    from dataclasses import replace
    from pathlib import Path

    from src.workbench.business_evaluation import _model_identity

    session = task[1]
    run_id = "wb-" + "a" * 32
    directory = tmp_path / run_id
    adapter = directory / "model"
    adapter.mkdir(parents=True)
    metrics = {"train_loss": 1.4, "eval_loss": 1.7}
    artifacts = {
        "adapter_config": ("adapter_config.json", b"{}"),
        "adapter_weights": ("adapter_model.safetensors", b"fixture-adapter"),
        "tokenizer_config": ("tokenizer_config.json", b"{}"),
        "metrics": ("workbench_metrics.json", json.dumps(metrics).encode()),
    }
    files = {}
    for key, (name, data) in artifacts.items():
        path = adapter / name
        path.write_bytes(data)
        files[key] = {"path": str(path), "sha256": hashlib.sha256(data).hexdigest()}
    base = _model_identity(Path(model_paths[0].base_model))
    base_identity = {
        "path": base["directory"],
        "config_and_tokenizer_hashes": {"config.json": base["files"]["config.json"]},
        "weights": [{"name": "pytorch_model.bin", "sha256": base["files"]["pytorch_model.bin"]}],
    }
    config = {
        "model": {"name": base["directory"], "max_length": 128},
        "training": {"output_dir": str(adapter), "num_epochs": 1},
        "data": {
            "train_file": session.dataset.paths["train"],
            "validation_file": session.dataset.paths["validation"],
            "validation_split": 0,
        },
        "logging": {"mlflow_tracking_uri": str(tmp_path / "mlruns")},
    }
    manifest = {
        "run_id": run_id,
        "dataset_version": session.dataset.version,
        "config_digest": content_digest(config),
        "model_identity": base_identity,
        "files": files,
    }
    run = {
        **{
            key: manifest[key]
            for key in ("run_id", "dataset_version", "config_digest", "model_identity")
        },
        "session_id": session.session_id,
        "dataset": session.dataset.model_dump(),
        "config": config,
        "output_dir": str(adapter),
        "model_path": base["directory"],
        "status": "running",
        "preflight": {
            "status": "passed",
            "version": session.dataset.version,
            "source_digest": session.dataset.source_digest,
            "recipe_digest": session.dataset.recipe_digest,
            "full_confirmed_revision": session.dataset.full_confirmed_revision,
            "max_length": 128,
            "supervision_strategy": "full_prompt_padding_masked",
            "splits": {"train": {"answer_lost_rows": 0}},
            "rows": [{"secret_raw_test": "must not be sent"}],
        },
    }
    result = {
        "run_id": run_id,
        "status": "succeeded",
        "metrics": metrics,
        "artifacts": {
            **{key: entry["path"] for key, entry in files.items()},
            "manifest": str(adapter / "workbench_manifest.json"),
        },
    }
    for path, value in [
        (adapter / "workbench_manifest.json", manifest),
        (directory / "run.json", run),
        (directory / "result.json", result),
        (directory / "config.json", config),
        (directory / "session.json", session.model_dump()),
    ]:
        path.write_text(json.dumps(value), encoding="utf-8")
    models = [model_paths[0], replace(model_paths[1], adapter_path=str(adapter))]
    return make_report(task, models, tmp_path / "eval"), directory


def test_training_evidence_reads_verified_receipts_not_raw_partitions(task, model_paths, tmp_path):
    report, _ = training_report(task, model_paths, tmp_path)
    evidence = EvaluationDiagnostics(report, task[1]).training_evidence()
    assert evidence["models"][0]["status"] == "not_applicable"
    tuned = evidence["models"][1]
    assert tuned["status"] == "available"
    assert tuned["config"]["training"]["num_epochs"] == 1
    assert tuned["metrics"]["train_loss"] == 1.4
    assert tuned["preflight"]["supervision_strategy"] == "full_prompt_padding_masked"
    assert tuned["mlflow"]["status"] == "not_recorded"
    assert tuned["mlflow"]["run_id"] is None
    assert "secret_raw_test" not in json.dumps(evidence)
    assert "rows" not in tuned["preflight"]


def gguf_training_report(task, model_paths, tmp_path, monkeypatch):
    """Stub the native/export boundary; keep real receipts and token-proof checks."""
    from dataclasses import asdict

    import transformers

    import src.workbench.gguf_evaluation as gguf_module
    from src.workbench.business_evaluation import (
        comparison_key,
        evaluation_metrics,
        generation_token_proof,
    )

    report, directory = training_report(task, model_paths, tmp_path)
    run = json.loads((directory / "run.json").read_text())
    source = {key: deepcopy(report.models[1][key]) for key in ("requested_model", "identity")}
    source.update(
        {key: run[key] for key in ("run_id", "session_id", "dataset_version", "config_digest")}
    )
    quantized = deepcopy(report.models[1])
    quantized.update(
        label="quantized",
        requested_model={
            "label": "quantized",
            "base_model": str(tmp_path / "export" / "hf_tokenizer"),
            "adapter_path": None,
            "revision": None,
            "gguf_path": str(tmp_path / "export" / "model-q4_k_m.gguf"),
            "gguf_runtime_path": str(tmp_path / "native-runtime"),
        },
        identity={
            "format": "gguf",
            "gguf": {
                "file": str(tmp_path / "export" / "model-q4_k_m.gguf"),
                "sha256": "a" * 64,
                "export_evidence_sha256": "b" * 64,
            },
            "runtime": {"binary_path": str(tmp_path / "native-runtime"), "sha256": "c" * 64},
            "tokenizer": {
                "directory": str(tmp_path / "export" / "hf_tokenizer"),
                "sha256": "d" * 64,
                "files": {"tokenizer_config.json": {"size": 2, "sha256": "d" * 64}},
            },
            "source_training": source,
        },
    )
    # The quantized result has its own observed quality, independent of the HF score.
    quantized["rows"][0].update(output="wrong", correct=False)
    quantized["metrics"] = evaluation_metrics(
        quantized["rows"], report.protocol, report.dataset["row_count"]
    )
    report.models.append(quantized)
    report.protocol.update(
        generation="greedy_token_contract_v2",
        max_new_tokens=32,
        input_tokenization="hf_default_special_tokens_native_exact_v1",
        decoding="hf_skip_special_tokens_v1",
        stopping="tokenizer_eos_or_max_new_tokens_v1",
        decoder_version=transformers.__version__,
    )
    for model in report.models:
        for row in model["rows"]:
            if row["output"] is None:
                continue
            generated = [8] * 32 if row["truncated"] else [8, 9]
            row["generated_tokens"] = len(generated)
            row["token_proof"] = generation_token_proof(
                row["prompt"], row["output"], [10 + row["index"], 6], generated, 9
            )
    report.comparison_key = comparison_key(report.dataset, report.protocol)
    current_identity = deepcopy(quantized["identity"])
    calls = []

    def verify(model):
        calls.append(asdict(model))
        return deepcopy(current_identity)

    monkeypatch.setattr(gguf_module, "verified_gguf_model", verify)
    return report, directory, current_identity, calls


def test_gguf_training_evidence_binds_real_hf_source_without_inheriting_score(
    task, model_paths, tmp_path, monkeypatch
):
    report, _, _, calls = gguf_training_report(task, model_paths, tmp_path, monkeypatch)
    diagnostics = EvaluationDiagnostics(report, task[1])
    entry = diagnostics.training_evidence()["models"][2]
    assert entry["status"] == "available", entry
    assert entry["training_run_id"] == report.models[2]["identity"]["source_training"]["run_id"]
    assert entry["training_session_id"] == task[1].session_id
    assert calls == [report.models[2]["requested_model"]]
    assert entry["derived_artifact"]["identity"] == report.models[2]["identity"]
    assert "不继承 HF 分数" in entry["derived_artifact"]["scope"]
    assert entry["identity"]["evaluation"] == report.models[1]["identity"]
    assert report.models[1]["metrics"]["exact_match"] == 1
    assert report.models[2]["metrics"]["exact_match"] == pytest.approx(2 / 3)
    assert diagnostics.summary()["models"]["quantized"]["counts"]["mismatch"] == 1
    assert "secret_raw_test" not in json.dumps(entry)


@pytest.mark.parametrize("changed", ["gguf", "runtime", "tokenizer", "export_evidence"])
def test_gguf_training_evidence_rejects_changed_evaluated_artifact(
    task, model_paths, tmp_path, monkeypatch, changed
):
    report, _, current, _ = gguf_training_report(task, model_paths, tmp_path, monkeypatch)
    if changed == "export_evidence":
        current["gguf"]["export_evidence_sha256"] = "f" * 64
    else:
        current[changed]["sha256"] = "f" * 64
    entry = EvaluationDiagnostics(report, task[1]).training_evidence()["models"][2]
    assert entry["status"] == "invalid"
    assert "评测之后已改变" in entry["reason"]
    assert "config" not in entry


@pytest.mark.parametrize("field", ["run_id", "session_id", "dataset_version", "config_digest"])
def test_gguf_declared_lineage_must_match_verified_source_receipts(
    task, model_paths, tmp_path, monkeypatch, field
):
    report, _, current, _ = gguf_training_report(task, model_paths, tmp_path, monkeypatch)
    forged = "f" * 64 if field == "config_digest" else "forged"
    report.models[2]["identity"]["source_training"][field] = forged
    current["source_training"][field] = forged
    entry = EvaluationDiagnostics(report, task[1]).training_evidence()["models"][2]
    assert entry["status"] == "invalid"
    assert "导出血缘与已核验的训练回执不一致" in entry["reason"]
    assert "config" not in entry


def test_gguf_training_evidence_rechecks_source_adapter_contents(
    task, model_paths, tmp_path, monkeypatch
):
    report, directory, _, _ = gguf_training_report(task, model_paths, tmp_path, monkeypatch)
    (directory / "model" / "adapter_model.safetensors").write_bytes(b"changed after evaluation")
    entry = EvaluationDiagnostics(report, task[1]).training_evidence()["models"][2]
    assert entry["status"] == "invalid"
    assert "适配器或其工作台清单在评测之后已改变" in entry["reason"]
    assert "derived_artifact" not in entry


@pytest.mark.parametrize("identity_key", ["base", "adapter"])
def test_gguf_cannot_masquerade_as_hf_identity(
    task, model_paths, tmp_path, monkeypatch, identity_key
):
    report, _, current, _ = gguf_training_report(task, model_paths, tmp_path, monkeypatch)
    report.models[2]["identity"][identity_key] = report.models[1]["identity"][identity_key]
    current.update(deepcopy(report.models[2]["identity"]))
    with pytest.raises(ValueError, match="GGUF"):
        EvaluationDiagnostics(report, task[1])


@pytest.mark.parametrize("mutation", ["missing_request", "adapter_request", "recursive_source"])
def test_gguf_cannot_use_base_fallback_or_another_gguf_as_training_source(
    task, model_paths, tmp_path, monkeypatch, mutation
):
    report, _, current, _ = gguf_training_report(task, model_paths, tmp_path, monkeypatch)
    model = report.models[2]
    if mutation == "missing_request":
        model["requested_model"]["gguf_path"] = None
    elif mutation == "adapter_request":
        model["requested_model"]["adapter_path"] = report.models[1]["requested_model"][
            "adapter_path"
        ]
    else:
        model["identity"]["source_training"]["identity"]["format"] = "gguf"
        current.update(deepcopy(model["identity"]))
    with pytest.raises(ValueError, match="GGUF"):
        EvaluationDiagnostics(report, task[1])


def test_gguf_diagnostics_requires_v2_and_calls_shared_token_proof_validator(
    task, model_paths, tmp_path, monkeypatch
):
    import src.workbench.evaluation_diagnostics as diagnostics_module
    from src.workbench.business_evaluation import comparison_key

    report, _, _, _ = gguf_training_report(task, model_paths, tmp_path, monkeypatch)
    calls = []

    def reject(report):
        calls.append(report.evaluation_id)
        raise ValueError("invalid complete token proof")

    monkeypatch.setattr(diagnostics_module, "verify_evaluation_report", reject)
    with pytest.raises(ValueError, match="invalid complete token proof"):
        EvaluationDiagnostics(report, task[1])
    assert calls == [report.evaluation_id]
    report.protocol["generation"] = "greedy_v1"
    report.comparison_key = comparison_key(report.dataset, report.protocol)
    with pytest.raises(ValueError, match="GGUF 同题诊断必须提供完整"):
        EvaluationDiagnostics(report, task[1])


@pytest.mark.parametrize(
    "template_name",
    [
        "chat_template.jinja",
        "additional_chat_templates/tool_use.jinja",
        "chat_templates/tool_use.jinja",
    ],
)
def test_training_receipt_cannot_omit_current_chat_template_identity(
    task, model_paths, tmp_path, template_name
):
    from pathlib import Path

    from src.workbench.business_evaluation import _model_identity

    template = Path(model_paths[0].base_model) / template_name
    template.parent.mkdir(parents=True, exist_ok=True)
    template.write_text("{{ messages }}")
    report, directory = training_report(task, model_paths, tmp_path)
    run_path = directory / "run.json"
    manifest_path = directory / "model" / "workbench_manifest.json"
    run, manifest = json.loads(run_path.read_text()), json.loads(manifest_path.read_text())
    # A complete modern receipt can bind the evaluated tokenizer/template.
    run["model_identity"]["config_and_tokenizer_hashes"][template_name] = report.models[1][
        "identity"
    ]["base"]["files"][template_name]
    manifest["model_identity"] = deepcopy(run["model_identity"])
    run_path.write_text(json.dumps(run))
    manifest_path.write_text(json.dumps(manifest))
    report.models[1]["identity"]["adapter"] = _model_identity(directory / "model")
    assert (
        EvaluationDiagnostics(report, task[1]).training_evidence()["models"][1]["status"]
        == "available"
    )
    # An older receipt omitting a template cannot establish the newer model's
    # training provenance even when all hashes it happens to list still match.
    del run["model_identity"]["config_and_tokenizer_hashes"][template_name]
    manifest["model_identity"] = deepcopy(run["model_identity"])
    run_path.write_text(json.dumps(run))
    manifest_path.write_text(json.dumps(manifest))
    report.models[1]["identity"]["adapter"] = _model_identity(directory / "model")
    entry = EvaluationDiagnostics(report, task[1]).training_evidence()["models"][1]
    assert entry["status"] == "invalid"
    assert "训练基座身份与评测记录不一致" in entry["reason"]
    assert "config" not in entry


@pytest.mark.parametrize(
    "target", ["manifest", "adapter", "config", "run_identity", "dataset", "result", "snapshot"]
)
def test_training_evidence_rejects_changed_or_misbound_artifacts(
    task, model_paths, tmp_path, target
):
    report, directory = training_report(task, model_paths, tmp_path)
    if target == "adapter":
        (directory / "model" / "adapter_model.safetensors").write_bytes(b"changed")
    else:
        paths = {
            "manifest": directory / "model" / "workbench_manifest.json",
            "config": directory / "config.json",
            "run_identity": directory / "run.json",
            "dataset": directory / "run.json",
            "result": directory / "result.json",
            "snapshot": directory / "session.json",
        }
        path = paths[target]
        value = json.loads(path.read_text())
        if target == "config":
            value["training"]["num_epochs"] = 100
        elif target == "dataset":
            value["dataset"]["version"] = "another-dataset"
        elif target == "snapshot":
            value["session_id"] = "b" * 32
        else:
            value["run_id"] = "wb-" + "b" * 32
        path.write_text(json.dumps(value))
    entry = EvaluationDiagnostics(report, task[1]).training_evidence()["models"][1]
    assert entry["status"] == "invalid"
    assert "config" not in entry
    assert entry["reason"]


def test_external_adapter_without_workbench_run_is_explicitly_unavailable(
    task, model_paths, tmp_path
):
    report = make_report(task, model_paths, tmp_path / "eval")
    entry = EvaluationDiagnostics(report, task[1]).training_evidence()["models"][1]
    assert entry["status"] == "not_available"
    assert "config" not in entry


def test_training_warnings_and_legacy_identity_are_explicit(task, model_paths, tmp_path):
    from src.workbench.business_evaluation import _model_identity

    report, directory = training_report(task, model_paths, tmp_path)
    run_path = directory / "run.json"
    run = json.loads(run_path.read_text())
    run["preflight"]["status"] = "warnings"
    run["preflight"]["issues"] = [
        {"code": "truncation", "severity": "warning", "message": "用户已核查"}
    ]
    del run["model_identity"]["weights"][0]["sha256"]
    run_path.write_text(json.dumps(run))
    manifest_path = directory / "model" / "workbench_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["model_identity"] = run["model_identity"]
    manifest_path.write_text(json.dumps(manifest))
    # This legacy receipt existed before evaluation; later manifest edits are rejected above.
    report.models[1]["identity"]["adapter"] = _model_identity(directory / "model")
    entry = EvaluationDiagnostics(report, task[1]).training_evidence()["models"][1]
    assert entry["status"] == "available"
    assert entry["preflight"]["status"] == "warnings"
    assert entry["preflight"]["issues"][0]["code"] == "truncation"
    assert entry["identity"]["base_weight_verification"] == "legacy_identity_unverified"


def test_historical_adapter_uses_own_snapshot_on_same_suite(task, model_paths, tmp_path):
    from src.workbench.business_evaluation import EvaluationModel
    from tests.unit.test_business_evaluation import suite_rounds

    original_report, directory = training_report(task, model_paths, tmp_path)
    _, revised = suite_rounds(task, tmp_path)
    models = [EvaluationModel(**model["requested_model"]) for model in original_report.models]
    report = make_report((task[0], revised), models, tmp_path / "later-eval")
    evidence = EvaluationDiagnostics(report, revised).training_evidence()["models"][1]
    assert evidence["status"] == "available", evidence
    assert evidence["training_dataset"]["version"] == task[1].dataset.version
    assert evidence["training_dataset"]["version"] != revised.dataset.version
    assert evidence["config"]["data"]["train_file"] == task[1].dataset.paths["train"]
    assert evidence["config"]["data"]["train_file"] != revised.dataset.paths["train"]
    assert evidence["training_session_digest"] == content_digest(task[1].model_dump())
    (directory / "session.json").unlink()
    unavailable = EvaluationDiagnostics(report, revised).training_evidence()["models"][1]
    assert unavailable["status"] == "not_available"
    assert "config" not in unavailable


def test_instruction_echo_detected_as_systematic_pattern(task, model_paths, tmp_path):
    """复述指令的截断输出被识别为「指令回声」并给出可核查方向，不冒充结论。"""
    _, session = task
    instruction = session.analysis.recipe.instruction

    def generate(label, index, expected):
        if label == "base":
            return Generation(
                instruction[:16] + "……继续复述的内容", truncated=True, generated_tokens=32
            )
        return Generation(expected)

    report = make_report(task, model_paths, tmp_path / "eval", generate=generate)
    diagnostics = EvaluationDiagnostics(report, task[1])
    summary = diagnostics.summary()
    assert summary["models"]["base"]["counts"]["instruction_echo"] == 3
    echo = next(h for h in summary["hypotheses"] if h["based_on"] == "instruction_echo")
    assert echo["models"] == ["base"]
    assert "复述指令" in echo["statement"]
    assert "messages" in echo["statement"]
    case = diagnostics.read_cases(error_type="truncated")["cases"][0]
    assert case["echoes_prompt"] is True
    # 短标签答案天然出现在指令中，不得误判为回声。
    assert summary["models"]["tuned"]["counts"].get("instruction_echo", 0) == 0


def test_paraphrase_echo_from_real_financial_bad_case_is_detected():
    """真实财报坏例是改述式复述（非逐字前缀），长片段共享判定必须命中。"""
    from src.workbench.evaluation_diagnostics import output_echoes_prompt

    instruction = (
        "虚构试点：根据公司披露文本（发布时公开可得），预测该披露后未来20个模拟交易日"
        "（按显式模拟收盘日历）价格相对基准是否上涨；目标为二分类forecast_direction（上涨/未上涨）。"
    )
    paraphrase_echo = "预测未来20个模拟交易日（按显式模拟收盘日历）价格相对基准是否上涨；目标为二分类forecast_direction（上涨/"
    assert output_echoes_prompt(paraphrase_echo, instruction)
    # 短标签与答案+短语说明不误判。
    assert not output_echoes_prompt("上涨", instruction)
    assert not output_echoes_prompt(None, instruction)
    assert not output_echoes_prompt(
        "硬件\n### Explanation: 客户说的屏幕碎了，属于硬件问题。", "客户描述：屏幕碎了"
    )


def test_high_truncation_models_uses_shared_ratio_threshold():
    """截断提示按占比判定：达到 20% 阈值的模型进入提示清单，与对照区口径一致。"""
    from src.workbench.evaluation_diagnostics import high_truncation_models

    def model(label, total, truncated):
        rows = [{"status": "truncated" if i < truncated else "scored"} for i in range(total)]
        return {"label": label, "rows": rows}

    # 1/5 恰好达到阈值 → 提示；1/6 低于阈值 → 不提示。
    assert high_truncation_models([model("甲", 5, 1), model("乙", 6, 1)]) == [("甲", 1)]
    # 无截断、空行列表都不产生提示。
    assert high_truncation_models([model("甲", 4, 0)]) == []
    assert high_truncation_models([{"label": "空", "rows": []}]) == []


def test_dominant_output_models_flags_repeat_dominant_outputs():
    """输出坍缩提示按占比判定：非空输出同一内容 ≥80% 且至少 4 条才点名，None 不计入分母。"""
    from src.workbench.evaluation_diagnostics import dominant_output_models

    def model(label, outputs):
        rows = [{"output": output, "status": "scored"} for output in outputs]
        return {"label": label, "rows": rows}

    repeated = ["无法分类：缺少信息"] * 8 + ["答案甲", "答案乙"]
    assert dominant_output_models([model("本轮微调", repeated)]) == [
        ("本轮微调", 8, 10, "无法分类：缺少信息")
    ]
    # 70% 占比低于阈值 → 不点名。
    assert dominant_output_models([model("甲", ["同一答案"] * 7 + ["其他答案"] * 3)]) == []
    # None（生成失败）不计入占比分母。
    assert dominant_output_models([model("甲", ["同一答案"] * 5 + ["不同答案", None])]) == [
        ("甲", 5, 6, "同一答案")
    ]
    # 少于 4 条非空输出、输出各不相同、空行列表都不点名。
    assert dominant_output_models([model("甲", ["同一答案"] * 3)]) == []
    assert dominant_output_models([model("甲", [f"答案{i}" for i in range(10)])]) == []
    assert dominant_output_models([{"label": "空", "rows": []}]) == []


def _echo_row(index: int, *, prompt_len: int, truncated: bool, echoes: bool) -> dict:
    prompt = "指" * prompt_len
    # 回声输出必须真正共享提示的连续片段（≥12 字）才命中判定；非回声输出不含提示字符。
    output = (prompt[:24] + "然后才是尝试作答的内容") if echoes else "一个完全不同的正常答案"
    return {
        "index": index,
        "prompt": prompt,
        "output": output,
        "status": "truncated" if truncated else "scored",
        "truncated": truncated,
    }


def test_echo_triage_prioritizes_max_new_tokens_when_truncation_overlaps() -> None:
    from src.workbench.evaluation_diagnostics import echo_triage_lines

    rows = [
        _echo_row(0, prompt_len=100, truncated=True, echoes=True),
        _echo_row(1, prompt_len=100, truncated=True, echoes=True),
        _echo_row(2, prompt_len=100, truncated=False, echoes=False),
    ]
    lines = echo_triage_lines("基座", rows, protocol={"max_new_tokens": 32})
    assert lines[0].startswith("检测到指令回声——基座 有 2 题")
    assert "其中 2 题同时触及生成长度上限（当前 max_new_tokens=32）" in lines[0]
    assert "优先核查 max_new_tokens" in lines[0]
    assert any("输入长度相近" in line for line in lines)  # 100 vs 100：不过长方向


def test_echo_triage_rules_out_length_limit_without_overlap() -> None:
    from src.workbench.evaluation_diagnostics import echo_triage_lines

    rows = [_echo_row(i, prompt_len=100, truncated=False, echoes=True) for i in range(3)]
    lines = echo_triage_lines("本轮微调", rows, protocol={"max_new_tokens": 256})
    assert "没有一题触及生成长度上限" in lines[0]
    assert "长度上限不是第一嫌疑" in lines[0]
    assert not any("max_new_tokens 是否小于最短合法答案" in line for line in lines)
    assert not any("字符" in line for line in lines)  # 全部回声：无比对集，不编造长度结论


def test_echo_triage_supports_long_input_direction_by_observed_average() -> None:
    from src.workbench.evaluation_diagnostics import echo_triage_lines

    rows = [
        _echo_row(0, prompt_len=600, truncated=False, echoes=True),
        _echo_row(1, prompt_len=100, truncated=False, echoes=False),
    ]
    lines = echo_triage_lines("基座", rows, protocol={"prompt_renderer": "unknown"})
    assert any("平均 600 字符" in line for line in lines)
    assert any("支持「过长内容淹没答案信号」方向" in line for line in lines)
    assert any("报告未记录本次模板类型" in line for line in lines)


def test_echo_triage_states_alpaca_template_fact_when_recorded() -> None:
    from src.workbench.evaluation_diagnostics import echo_triage_lines

    rows = [_echo_row(0, prompt_len=100, truncated=False, echoes=True)]
    lines = echo_triage_lines(
        "基座", rows, protocol={"prompt_renderer": "render_alpaca_prompt_without_answer"}
    )
    assert any("补全式（Alpaca）提示模板" in line for line in lines)
    assert any("messages 对话格式" in line for line in lines)
    assert lines[-1].startswith("以上是按报告内事实排出的核查顺序")


def test_echo_triage_silent_without_echo_rows() -> None:
    from src.workbench.evaluation_diagnostics import echo_triage_lines

    assert (
        echo_triage_lines("基座", [_echo_row(0, prompt_len=100, truncated=False, echoes=False)])
        == []
    )
    assert echo_triage_lines("基座", []) == []


def test_warning_sentences_single_source_shared_by_page_and_summary() -> None:
    """R102 C+登记-①:截断/坍缩两句警告从页面手抄收编为单一来源 builder。

    页面 st.warning 与 summarize_comparison 此前各写一套——页面句还带着已退场的
    「逐题查看完整输出」内联指令与无检测器支撑的「缺少停止标记」提醒(R101 收敛
    语境句后两处尾句已事实漂移)。收编后页面与摘要渲染同一函数返回值,词汇漂移
    在结构上不可能。
    """
    from src.workbench.evaluation_diagnostics import (
        dominant_warning_sentence,
        truncation_warning_sentence,
    )

    truncation = truncation_warning_sentence([("基座", 2), ("本轮微调", 1)], 32)
    assert truncation == (
        "多个输出因触及生成长度上限被截断（基座 2 题、本轮微调 1 题，当前 max_new_tokens 为 32）；"
        "先核查 max_new_tokens 是否小于最短合法答案、输出是否在重复生成，"
        "再决定是否加长——触及上限不等于只需增加长度。"
    )
    # 协议未记录上限:不拼「当前 max_new_tokens」段,不编造数值
    assert "当前 max_new_tokens" not in truncation_warning_sentence([("基座", 2)], None)

    dominant = dominant_warning_sentence([("基座", 4, 4, "同一回答")])
    assert dominant == (
        "观察到输出高度重复（基座 有 4/4 条输出完全相同（同一回答））：该模型在反复输出同一答案。"
        "请对照开发集答案分布——分布本身集中时,模型可能只是复述多数类,不一定是学坏了;"
        "分布不集中时,需确认每题是否本该有不同答案。这是观察事实,不认定原因。"
    )
    # 超长重复片段截到 24 字 + 省略号,同页面此前手抄口径
    assert ("长" * 24 + "…") in dominant_warning_sentence([("基座", 4, 4, "长" * 30)])
