"""Saved identity contracts use synthetic tokens and do not establish model quality."""

from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

import pytest

from src.workbench.business_evaluation import (
    TOKEN_CONTRACT,
    TOKEN_GENERATION,
    BusinessEvaluationService,
    EvaluationModel,
    EvaluationProtocol,
    EvaluationReport,
    assert_comparable,
    comparison_key,
    evaluation_metrics,
    generation_token_proof,
    verify_evaluation_report,
)
from src.workbench.gguf_evaluation import verify_gguf_report_identity
from src.workbench.sources import content_digest


@pytest.fixture()
def report():
    def hf_identity(directory):
        files = {"model.safetensors": "f" * 64}
        return {"directory": directory, "files": files, "content_digest": content_digest(files)}

    requested = asdict(
        EvaluationModel(
            "quantized",
            "/bundle/hf_tokenizer",
            gguf_path="/bundle/model-q4_k_m.gguf",
            gguf_runtime_path="/tools/native",
        )
    )
    identity = {
        "format": "gguf",
        "gguf": {
            "file": requested["gguf_path"],
            "sha256": "a" * 64,
            "export_evidence_sha256": "b" * 64,
        },
        "runtime": {"binary_path": requested["gguf_runtime_path"], "sha256": "c" * 64},
        "tokenizer": {
            "directory": requested["base_model"],
            "files": {"tokenizer_config.json": {"size": 2, "sha256": "d" * 64}},
        },
        "source_training": {
            "run_id": "wb-" + "a" * 32,
            "session_id": "b" * 32,
            "dataset_version": "v1",
            "config_digest": "d" * 64,
            "requested_model": {"base_model": "/hf/base", "adapter_path": "/hf/adapter"},
            "identity": {"base": hf_identity("/hf/base"), "adapter": hf_identity("/hf/adapter")},
        },
    }
    row = {
        "index": 0,
        "source": {"row_id": "synthetic"},
        "prompt": "question",
        "expected": "yes",
        "output": "yes",
        "status": "scored",
        "correct": True,
        "field_scores": {},
        "truncated": False,
        "generated_tokens": 2,
        "error": "",
        "token_proof": generation_token_proof("question", "yes", [1], [3, 2], 2),
    }
    protocol = {
        **asdict(EvaluationProtocol("classification_exact", max_new_tokens=2)),
        **TOKEN_CONTRACT,
        "generation": TOKEN_GENERATION,
        "decoder_version": "synthetic-contract-test",
        "prompt_renderer": "render_alpaca_prompt_without_answer",
        "prompt_digest": content_digest([row["prompt"]]),
        "answers_digest": content_digest([row["expected"]]),
    }
    dataset = {"purpose": "development_only", "split": "validation", "row_count": 1}
    outcome = {
        "label": "quantized",
        "requested_model": requested,
        "identity": identity,
        "rows": [row],
        "errors": [],
        "metrics": evaluation_metrics([row], protocol, 1),
    }
    return EvaluationReport(
        "a" * 32,
        "2026-10-16T00:00:00Z",
        dataset,
        protocol,
        comparison_key(dataset, protocol),
        status="completed",
        models=[outcome],
    )


def test_identity_validation_is_pure_and_accepts_lexical_path_equivalence(report, monkeypatch):
    def no_filesystem(*args, **kwargs):
        raise AssertionError("saved report validation must not inspect local artifacts")

    monkeypatch.setattr(Path, "resolve", no_filesystem)
    monkeypatch.setattr(Path, "open", no_filesystem)
    monkeypatch.setattr(Path, "stat", no_filesystem)
    report.models[0]["requested_model"]["gguf_path"] = "/bundle/./model-q4_k_m.gguf"
    verify_gguf_report_identity(report.models[0])
    verify_gguf_report_identity(
        {
            "requested_model": {"base_model": "/hf"},
            "identity": {"base": {}, "adapter": None},
        }
    )


TAMPERING = [
    (("requested_model", "gguf_path"), "/other/model.gguf"),
    (("requested_model", "gguf_runtime_path"), "/other/native"),
    (("requested_model", "base_model"), "/other/tokenizer"),
    (("identity", "gguf", "file"), "/other/model.gguf"),
    (("identity", "runtime", "binary_path"), "/other/native"),
    (("identity", "tokenizer", "directory"), "/other/tokenizer"),
    (("identity", "gguf", "sha256"), "g" * 64),
    (("identity", "gguf", "export_evidence_sha256"), "a" * 63),
    (("identity", "runtime", "sha256"), None),
    (("identity", "format"), "hf"),
    (("identity", "base"), None),
    (("identity", "adapter"), {}),
    (("requested_model", "adapter_path"), "/hf/adapter"),
    (("requested_model", "revision"), "main"),
    (("identity", "source_training", "run_id"), ""),
    (("identity", "source_training", "session_id"), None),
    (("identity", "source_training", "dataset_version"), ""),
    (("identity", "source_training", "config_digest"), "not-a-sha256"),
    (("identity", "source_training", "identity", "base"), {}),
    (("identity", "source_training", "identity", "adapter"), None),
    (("identity", "source_training", "identity", "base", "content_digest"), "0" * 64),
    (
        ("identity", "source_training", "identity", "adapter", "files", "model.safetensors"),
        "0" * 64,
    ),
    (("identity", "tokenizer", "files"), {}),
    (("identity", "tokenizer", "files", "tokenizer_config.json", "size"), True),
    (("identity", "tokenizer", "files", "tokenizer_config.json", "size"), -1),
    (("identity", "tokenizer", "files", "tokenizer_config.json", "sha256"), "bad-hash"),
    (("identity", "source_training", "requested_model", "adapter_path"), "/other/adapter"),
]


@pytest.mark.parametrize("keys,value", TAMPERING, ids=[".".join(keys) for keys, _ in TAMPERING])
def test_tampered_gguf_identity_cannot_load_or_be_compared(report, tmp_path, keys, value):
    pristine = deepcopy(report)
    target = report.models[0]
    for key in keys[:-1]:
        target = target[key]
    target[keys[-1]] = value
    with pytest.raises(ValueError, match="GGUF"):
        verify_gguf_report_identity(report.models[0])
    with pytest.raises(ValueError, match="GGUF"):
        verify_evaluation_report(report)
    with pytest.raises(ValueError, match="GGUF"):
        assert_comparable(pristine, report)
    service = BusinessEvaluationService(tmp_path / "reports")
    service._save(report)
    with pytest.raises(ValueError, match="GGUF"):
        service.get_report(report.evaluation_id)
    assert service.list_reports() == []
    assert len(service.load_errors) == 1


def test_empty_load_identity_is_only_allowed_when_no_answer_was_generated(report, tmp_path):
    model = report.models[0]
    model["identity"] = {}
    with pytest.raises(ValueError, match="GGUF"):
        verify_gguf_report_identity(model)
    row = model["rows"][0]
    row.update(output=None, status="failed", correct=False, generated_tokens=None)
    row.pop("token_proof")
    model["metrics"] = evaluation_metrics(model["rows"], report.protocol, 1)
    report.status = "completed_with_failures"
    verify_gguf_report_identity(model)
    verify_evaluation_report(report)
    service = BusinessEvaluationService(tmp_path / "reports")
    service._save(report)
    assert service.get_report(report.evaluation_id).models[0]["metrics"]["exact_match"] == 0


@pytest.mark.parametrize("trigger", ["gguf_path", "gguf_runtime_path", "identity_gguf", "format"])
def test_each_gguf_claim_triggers_validation(report, trigger):
    model = deepcopy(report.models[0])
    model["requested_model"].update(gguf_path=None, gguf_runtime_path=None)
    model["identity"] = {}
    if trigger.startswith("identity"):
        model["identity"]["gguf"] = {}
    elif trigger == "format":
        model["identity"]["format"] = "gguf"
    else:
        model["requested_model"][trigger] = "/unbound/claim"
    with pytest.raises(ValueError, match="GGUF"):
        verify_gguf_report_identity(model)
