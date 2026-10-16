"""Token and process contracts; fake logits never establish business quality."""

import copy
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path

import pytest

from src.workbench.business_evaluation import (
    TOKEN_CONTRACT,
    TOKEN_GENERATION,
    BusinessEvaluationService,
    EvaluationModel,
    EvaluationProtocol,
    Generation,
    LocalEvaluationRuntime,
    _model_identity,
    assert_comparable,
    generation_token_proof,
    verify_evaluation_report,
)
from src.workbench.gguf_evaluation import (
    GGUFLocalEvaluationRuntime,
    gguf_evaluation_model,
    verified_gguf_model,
)
from tests.unit import test_business_evaluation as fixtures

model_paths = fixtures.model_paths
task = fixtures.task


@pytest.fixture()
def tokenizer_dir(tmp_path):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import WhitespaceSplit
    from transformers import PreTrainedTokenizerFast

    backend = Tokenizer(WordLevel({"[UNK]": 0, "OK": 1, "[EOS]": 2}, unk_token="[UNK]"))
    backend.pre_tokenizer = WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="[UNK]", eos_token="[EOS]"
    )
    directory = tmp_path / "tokenizer"
    tokenizer.save_pretrained(directory)
    return directory


def _runtime(tmp_path, tokenizer_dir, mode="ok"):
    binary = tmp_path / "native.py"
    binary.write_text(
        f"#!{sys.executable}\n"
        "import json,sys,time\n"
        "print(json.dumps({'status':'ready','model_context_length':128}),flush=True)\n"
        f"if {mode!r}=='write_timeout': time.sleep(10)\n"
        "for line in sys.stdin:\n"
        " r=json.loads(line)\n"
        f" mode={mode!r}\n"
        " if mode=='timeout': time.sleep(10)\n"
        " if mode=='exit': sys.exit(1)\n"
        " if mode=='error':\n"
        "  print(json.dumps({'status':'error','error':'fixture failure'}),flush=True); continue\n"
        " tokens={'length':[1,1],'early':[1],'bad_eos':[2,1], 'bool':[True,2]}.get(mode,[1,2])\n"
        " inputs=r['expected_prompt_ids'] if mode!='mismatch' else [999]\n"
        " finish='length' if mode in ('length','early') else 'eos'\n"
        " print(json.dumps({'status':'completed','text':'untrusted native text',"
        "'generated_token_ids':tokens,'prompt_token_ids':inputs,'finish_reason':finish}),flush=True)\n"
    )
    binary.chmod(0o755)
    model = EvaluationModel(
        "GGUF",
        str(tokenizer_dir),
        gguf_path=str(tmp_path / "model.gguf"),
        gguf_runtime_path=str(binary),
    )
    identity = {"runtime": {"binary_path": str(binary)}, "gguf": {"metadata": {"token_count": 3}}}
    return GGUFLocalEvaluationRuntime(model, identity=identity)


@pytest.mark.parametrize("mode,text,truncated", [("ok", "OK", False), ("length", "OK OK", True)])
def test_native_ids_use_source_hf_decoder_and_full_denominator(
    tmp_path, tokenizer_dir, mode, text, truncated
):
    runtime = _runtime(tmp_path, tokenizer_dir, mode)
    process = runtime.process
    try:
        generation = runtime.generate(
            "question", EvaluationProtocol("classification_exact", max_new_tokens=2)
        )
        assert generation.text == text and generation.truncated is truncated
        assert generation.generated_tokens == 2
        assert generation.token_proof["prompt_token_ids"] == [0]
    finally:
        runtime.close()
    assert process.poll() is not None
    runtime.close()  # A second close must not affect another process.


@pytest.mark.parametrize("mode", ["early", "bad_eos", "bool", "mismatch", "error", "exit"])
def test_invalid_native_generation_is_not_a_business_answer(tmp_path, tokenizer_dir, mode):
    runtime = _runtime(tmp_path, tokenizer_dir, mode)
    try:
        with pytest.raises(ValueError):
            runtime.generate(
                "question", EvaluationProtocol("classification_exact", max_new_tokens=2)
            )
    finally:
        runtime.close()


def test_native_timeout_releases_own_child(tmp_path, tokenizer_dir):
    runtime = _runtime(tmp_path, tokenizer_dir, "timeout")
    process = runtime.process
    runtime.timeout_seconds = 0.05
    with pytest.raises(ValueError, match="超时"):
        runtime.generate("question", EvaluationProtocol("classification_exact", max_new_tokens=2))
    assert process.poll() is not None
    runtime.close()


def test_native_not_reading_stdin_cannot_block_large_prompt_forever(tmp_path, tokenizer_dir):
    runtime = _runtime(tmp_path, tokenizer_dir, "write_timeout")
    process = runtime.process
    runtime.timeout_seconds = 0.05
    with pytest.raises(ValueError, match="输入超时"):
        runtime.generate("a" * 200000, EvaluationProtocol("classification_exact", max_new_tokens=2))
    assert process.poll() is not None
    runtime.close()


def test_context_overflow_is_refused_before_generation(tmp_path, tokenizer_dir):
    runtime = _runtime(tmp_path, tokenizer_dir)
    try:
        with pytest.raises(ValueError, match="上下文"):
            runtime.generate(
                "question", EvaluationProtocol("classification_exact", max_new_tokens=128)
            )
    finally:
        runtime.close()


def test_hf_token_contract_clears_inherited_logits_modifiers(monkeypatch, tokenizer_dir):
    import torch
    from transformers import AutoTokenizer, GenerationConfig, LlamaConfig, LlamaForCausalLM

    model = LlamaForCausalLM(
        LlamaConfig(
            vocab_size=16,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
        )
    )
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    model.generation_config = GenerationConfig(
        suppress_tokens=list(range(16)), repetition_penalty=1.8, forced_eos_token_id=7
    )
    monkeypatch.setattr(
        "src.models.loader.load_model_and_tokenizer", lambda config: (model, tokenizer)
    )
    runtime = LocalEvaluationRuntime(
        EvaluationModel("HF", str(tokenizer_dir)), enforce_token_contract=True
    )
    try:
        effective, _ = model._prepare_generation_config(
            GenerationConfig(do_sample=False, max_new_tokens=2, eos_token_id=2)
        )
        assert effective.suppress_tokens is None and effective.repetition_penalty == 1
        assert effective.forced_eos_token_id is None
        generated = runtime.generate(
            "question", EvaluationProtocol("classification_exact", max_new_tokens=2)
        )
        inputs = tokenizer("question", return_tensors="pt")["input_ids"]
        expected = []
        with torch.inference_mode():
            for _ in range(2):
                token = int(model(inputs).logits[0, -1].argmax())
                expected.append(token)
                if token == 2:
                    break
                inputs = torch.cat((inputs, torch.tensor([[token]])), dim=1)
        assert generated.token_proof["generated_token_ids"] == expected
    finally:
        runtime.close()


def _mixed_report(task, model_paths, tmp_path, monkeypatch, mode="ok"):
    import src.workbench.gguf_evaluation as gguf

    _, session = task
    models = [
        *model_paths,
        EvaluationModel("GGUF", "bundle", gguf_path="model.gguf", gguf_runtime_path="native"),
    ]
    artifact_identity = {
        "format": "gguf",
        "gguf": {
            "file": str(Path("model.gguf").absolute()),
            "sha256": "a" * 64,
            "export_evidence_sha256": "b" * 64,
        },
        "runtime": {"binary_path": str(Path("native").absolute()), "sha256": "c" * 64},
        "tokenizer": {
            "directory": str(Path("bundle").absolute()),
            "files": {"tokenizer_config.json": {"sha256": "d" * 64, "size": 2}},
        },
        "source_training": {
            "run_id": "wb-" + "1" * 32,
            "session_id": session.session_id,
            "dataset_version": session.dataset.version,
            "config_digest": "e" * 64,
            "requested_model": {
                "base_model": model_paths[1].base_model,
                "adapter_path": model_paths[1].adapter_path,
            },
            "identity": {
                "base": _model_identity(Path(model_paths[1].base_model)),
                "adapter": _model_identity(Path(model_paths[1].adapter_path)),
            },
        },
    }
    monkeypatch.setattr(gguf, "verified_gguf_model", lambda model: artifact_identity)
    records = [
        json.loads(line)
        for line in Path(session.dataset.paths["validation"]).read_text().splitlines()
    ]

    class Runtime:
        def __init__(self, model):
            self.model = model
            self.index = 0

        def generate(self, prompt, protocol):
            text = records[self.index]["output"]
            self.index += 1
            if self.model.gguf_path:
                text = "wrong"
            ids, output_ids, eos = [5, 6], [8, 9], 9
            if self.model.gguf_path and mode == "truncated":
                output_ids = [8, 8]
            if self.model.gguf_path and mode == "mismatch":
                ids = [5, 7]
            proof = generation_token_proof(prompt, text, ids, output_ids, eos)
            if mode == "missing" and self.model.gguf_path:
                proof = None
            return Generation(text, output_ids[-1] != eos, 2, proof)

        def close(self):
            pass

    service = BusinessEvaluationService(tmp_path / "evaluations")
    report = service.compare(
        session,
        models,
        EvaluationProtocol("classification_exact", max_new_tokens=2),
        runtime_factory=Runtime,
    )
    return service, report


@pytest.mark.parametrize(
    "mode,status",
    [("ok", "scored"), ("truncated", "truncated"), ("mismatch", "failed"), ("missing", "failed")],
)
def test_mixed_comparison_scores_gguf_independently_and_rejects_bad_tokens(
    task, model_paths, tmp_path, monkeypatch, mode, status
):
    service, report = _mixed_report(task, model_paths, tmp_path, monkeypatch, mode)
    assert report.protocol["generation"] == TOKEN_GENERATION
    assert all(report.protocol[key] == value for key, value in TOKEN_CONTRACT.items())
    hf, gguf = report.models[1:]
    assert hf["metrics"]["exact_match"] == 1
    assert gguf["metrics"]["exact_match"] == 0
    assert gguf["metrics"]["total"] == hf["metrics"]["total"]
    assert all(row["status"] == status for row in gguf["rows"])
    assert asdict(service.get_report(report.evaluation_id)) == json.loads(
        json.dumps(asdict(report))
    )
    assert_comparable(report, report)


@pytest.mark.parametrize(
    "field,value",
    [
        ("generated_token_ids", [9, 8]),
        ("prompt_token_ids", [999]),
        ("eos_token_ids", [8]),
        ("finish_reason", "length"),
        ("output_digest", "forged"),
    ],
)
def test_report_load_revalidates_native_proof(
    task, model_paths, tmp_path, monkeypatch, field, value
):
    service, report = _mixed_report(task, model_paths, tmp_path, monkeypatch)
    report.models[2]["rows"][0]["token_proof"][field] = value
    service._save(report)
    with pytest.raises(ValueError):
        service.get_report(report.evaluation_id)


def test_cross_report_eos_contract_cannot_be_hidden_by_same_comparison_key(
    task, model_paths, tmp_path, monkeypatch
):
    _, report = _mixed_report(task, model_paths, tmp_path, monkeypatch)
    other = copy.deepcopy(report)
    for model in other.models:
        for row in model["rows"]:
            proof = row["token_proof"]
            proof["generated_token_ids"][-1] = 10
            proof["eos_token_ids"] = [10]
    verify_evaluation_report(other)
    with pytest.raises(ValueError, match="EOS"):
        assert_comparable(report, other)


def test_gguf_cannot_use_legacy_generation_protocol(task, model_paths, tmp_path, monkeypatch):
    from src.workbench.business_evaluation import comparison_key

    _, report = _mixed_report(task, model_paths, tmp_path, monkeypatch)
    report.protocol["generation"] = "greedy_v1"
    report.comparison_key = comparison_key(report.dataset, report.protocol)
    with pytest.raises(ValueError, match="GGUF"):
        verify_evaluation_report(report)


def _export(tmp_path, monkeypatch):
    import src.models.merger as merger
    from src.workbench.model_export import export_model, plan_model_export
    from tests.unit.test_model_export import _fake_gguf, _make_run, _tools

    run, tools = _make_run(tmp_path), _tools(tmp_path)
    Path(run["output_dir"]).parent.joinpath("run.json").write_text(json.dumps(run))

    def merge(adapter, target, **kwargs):
        directory = Path(target)
        directory.mkdir()
        for name, text in {
            "config.json": "{}",
            "model.safetensors": "weights",
            "tokenizer_config.json": "{}",
            "tokenizer.json": "{}",
            "chat_template.jinja": "saved template",
        }.items():
            (directory / name).write_text(text)
        return str(directory)

    monkeypatch.setattr(merger, "merge_adapter_to_dir", merge)
    monkeypatch.setattr("src.utils.gguf_export.convert_to_gguf", _fake_gguf([]))
    monkeypatch.setattr(
        "src.workbench.gguf_evaluation.TrainingRunService.get_status", lambda self, run_id: run
    )
    monkeypatch.setattr(
        "src.utils.gguf_runtime_tools.evaluation_runtime_identity",
        lambda path: {"binary_path": str(path), "sha256": "b" * 64},
    )
    target = tmp_path / "export"
    export_model(plan_model_export(run, target, export_format="gguf", llama_cpp_dir=tools))
    return run, target


def test_portable_tokenizer_and_source_are_bound_to_existing_export(tmp_path, monkeypatch):
    run, target = _export(tmp_path, monkeypatch)
    evidence_before = (target / "export_evidence.json").read_bytes()
    model = gguf_evaluation_model(run, target, tmp_path / "native")
    identity = verified_gguf_model(model)
    assert model.adapter_path is None and not identity.get("adapter")
    assert identity["source_training"]["run_id"] == run["run_id"]
    assert identity["source_training"]["identity"]["adapter"]["directory"] == run["output_dir"]
    assert (target / "hf_tokenizer/chat_template.jinja").read_text() == "saved template"
    assert not list((target / "hf_tokenizer").glob("*.safetensors"))
    assert (target / "export_evidence.json").read_bytes() == evidence_before
    with pytest.raises(ValueError, match="当前训练"):
        gguf_evaluation_model({**run, "run_id": "another"}, target, tmp_path / "native")
    with pytest.raises(ValueError, match="叠加"):
        verified_gguf_model(replace(model, adapter_path=run["output_dir"]))


@pytest.mark.parametrize("mutation", ["gguf", "tokenizer", "receipt", "source", "old_export"])
def test_changed_export_or_training_source_is_refused(tmp_path, monkeypatch, mutation):
    run, target = _export(tmp_path, monkeypatch)
    model = gguf_evaluation_model(run, target, tmp_path / "native")
    if mutation == "gguf":
        with (target / "model-q4_k_m.gguf").open("ab") as handle:
            handle.write(b"changed")
    elif mutation == "tokenizer":
        (target / "hf_tokenizer/tokenizer_config.json").write_text('{"eos_token":"other"}')
    elif mutation == "source":
        Path(run["output_dir"]).joinpath("adapter_model.safetensors").write_text("changed")
    else:
        evidence = json.loads((target / "export_evidence.json").read_text())
        if mutation == "receipt":
            evidence["run_id"] = "forged"
        else:
            evidence.pop("tokenizer_directory")
        (target / "export_evidence.json").write_text(json.dumps(evidence))
    with pytest.raises(ValueError):
        verified_gguf_model(model)
