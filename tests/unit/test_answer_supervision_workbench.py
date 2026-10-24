"""Answer supervision is one immutable contract across workbench handoffs.

Tiny local weights exercise the real worker; fixture generation below verifies
iteration evidence linkage only, and makes no claim about business improvement.
"""

import json
import sqlite3
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import pytest
from datasets import Dataset
from tokenizers import Tokenizer, models, processors
from transformers import PreTrainedTokenizerFast

from src.data.loaders import AlpacaDataset, render_alpaca_prompt
from src.data.sft_collator import AttentionMaskCausalCollator
from src.data.sft_protocol import ANSWER_PROTOCOL, LEGACY_PROTOCOL
from src.data_flywheel.dataset_registry import LocalDatasetRegistry
from src.workbench.business_evaluation import EvaluationModel, EvaluationProtocol, Generation
from src.workbench.iterations import IterationService
from src.workbench.sources import canonical, content_digest
from src.workbench.training_plans import TrainingPlanService
from src.workbench.training_preflight import preflight_dataset
from src.workbench.training_recovery import TrainingRecoveryService, failure_evidence
from src.workbench.training_runs import write_json
from tests.unit.test_intake_training_preflight import NoOffsets
from tests.unit.test_training_plans import _proposal
from tests.unit.test_training_recovery import _exception
from tests.unit.test_workbench_training_runs import _wait, environment  # noqa: F401


@pytest.fixture
def answer_environment(environment):  # noqa: F811
    _, _, _, model = environment
    tokenizer = PreTrainedTokenizerFast.from_pretrained(model, local_files_only=True)
    tokenizer.backend_tokenizer.post_processor = processors.TemplateProcessing(
        single="[BOS] $A", special_tokens=[("[BOS]", tokenizer.bos_token_id)]
    )
    tokenizer.save_pretrained(model)
    return environment


def _tokenizer(env):
    return PreTrainedTokenizerFast.from_pretrained(env[3], local_files_only=True)


def _splits(session):
    artifact = session.dataset
    registry = LocalDatasetRegistry(artifact.registry_root)
    return {
        split: registry.load_split(artifact.name, artifact.version, split)
        for split in ("train", "validation", "test")
    }


def _prepare(env, **changes):
    _, session, training, model = env
    return training.prepare(
        session,
        model,
        max_length=32,
        sft_protocol=changes.pop("sft_protocol", ANSWER_PROTOCOL),
        training_options={
            "num_epochs": 1,
            "batch_size": 4,
            "gradient_accumulation_steps": 1,
            "gradient_checkpointing": False,
            "logging_steps": 1,
            "warmup_ratio": 0,
        },
        **changes,
    )


@pytest.mark.parametrize("slow", [False, True])
@pytest.mark.parametrize("side", ["left", "right"])
@pytest.mark.parametrize("shared_pad_eos", [False, True])
def test_preflight_matches_real_formatter_and_collator_without_prompt_loss(
    answer_environment, slow, side, shared_pad_eos
):
    _, session, _, _ = answer_environment
    tok = _tokenizer(answer_environment)
    tok.padding_side = side
    if shared_pad_eos:
        tok.pad_token = tok.eos_token
    if slow:
        tok = NoOffsets(tok)
        tok.is_fast = False
    report = preflight_dataset(session, tok, 31, sft_protocol=ANSWER_PROTOCOL)
    assert report["status"] == "passed", report["issues"]
    assert report["sft_protocol"] == ANSWER_PROTOCOL
    assert report["supervision_strategy"] == "answer_and_eos_only"
    assert len(report["rows"]) == 9
    assert all(part["unverified_answer_rows"] == 0 for part in report["splits"].values())
    for split, rows in _splits(session).items():
        loader = AlpacaDataset("unused")
        loader.dataset = Dataset.from_list(rows)
        formatted = loader.format_for_training(tok, 31, sft_protocol=ANSWER_PROTOCOL)
        batch = AttentionMaskCausalCollator(tok, sft_protocol=ANSWER_PROTOCOL)(list(formatted))
        for index, raw in enumerate(rows):
            observed = next(
                item
                for item in report["rows"]
                if item["split"] == split and item["row_id"] == raw["metadata"]["source_row_id"]
            )
            mask, labels = batch["attention_mask"][index], batch["labels"][index]
            active_labels = labels[mask == 1].tolist()
            assert active_labels[:-2] == [-100] * (int(mask.sum()) - 2)
            assert active_labels[-2:] == [tok.convert_tokens_to_ids("yes"), tok.eos_token_id]
            assert labels[mask == 0].eq(-100).all()
            assert observed["supervised_tokens"] == int(labels[1:].ne(-100).sum()) == 2
            assert observed["answer_tokens"] == observed["answer_supervised_tokens"] == 1
            assert observed["prompt_supervised_tokens"] == 0
            assert observed["terminal_eos_supervised"] is True
            assert observed["answer_boundary_verified"] is True
            assert observed["answer_supervision_complete"] is True


@pytest.mark.parametrize("truncation_side", ["left", "right"])
def test_answer_preflight_rejects_any_truncation_including_only_eos(
    answer_environment, truncation_side
):
    _, session, _, _ = answer_environment
    tok = _tokenizer(answer_environment)
    tok.truncation_side = truncation_side
    complete = preflight_dataset(session, tok, 32, sft_protocol=ANSWER_PROTOCOL)
    longest = max(row["full_tokens"] for row in complete["rows"])
    report = preflight_dataset(session, tok, longest - 1, sft_protocol=ANSWER_PROTOCOL)
    assert report["status"] == "blocked"
    assert any(
        issue["code"] == "answer_supervision_contract" and issue["severity"] == "blocking"
        for issue in report["issues"]
    )


def test_real_bpe_merge_across_answer_boundary_is_blocked_instead_of_guessed(answer_environment):
    _, session, _, _ = answer_environment
    rows = [row for part in _splits(session).values() for row in part]
    characters = sorted(set("".join(render_alpaca_prompt(row) for row in rows)))
    vocab = {
        word: index
        for index, word in enumerate(["[UNK]", "[PAD]", "[BOS]", "[EOS]", *characters, "\ny"])
    }
    backend = Tokenizer(models.BPE(vocab, [("\n", "y")], unk_token="[UNK]"))
    backend.post_processor = processors.TemplateProcessing(
        single="[BOS] $A", special_tokens=[("[BOS]", vocab["[BOS]"])]
    )
    tok = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="[UNK]",
        pad_token="[PAD]",
        bos_token="[BOS]",
        eos_token="[EOS]",
    )
    prefix = tok(render_alpaca_prompt(rows[0] | {"output": ""}))["input_ids"]
    full = tok(render_alpaca_prompt(rows[0]))["input_ids"]
    assert full[: len(prefix)] != prefix
    report = preflight_dataset(session, NoOffsets(tok), 256, sft_protocol=ANSWER_PROTOCOL)
    assert report["status"] == "blocked"
    issues = [issue for issue in report["issues"] if issue["severity"] == "blocking"]
    assert len(issues) == len(rows)
    assert all(issue["code"] == "answer_supervision_contract" for issue in issues)


def test_plan_roundtrip_binds_selected_protocol_and_rejects_protocol_tampering(
    answer_environment, tmp_path
):
    _, session, training, model = answer_environment
    plans = TrainingPlanService(tmp_path / "plans", training.root)
    plans.training = training
    probe = plans.probe(session, model, 32, sft_protocol=ANSWER_PROTOCOL)
    assert probe["status"] == "passed"
    assert probe["preflight"]["sft_protocol"] == ANSWER_PROTOCOL
    consumption = probe["preflight"]["token_consumption"]
    assert consumption["prompt_supervised_tokens"] == 0
    assert consumption["supervised_tokens"] == 2 * consumption["answer_supervised_tokens"]
    proposal = _proposal(model, sft_protocol=ANSWER_PROTOCOL)
    context = plans.context(session, [str(model)])
    plan = plans.save(session, proposal, context, [])
    reopened = TrainingPlanService(plans.root, training.root)
    reopened.training = training
    assert reopened.get(plan["plan_id"])["proposal"]["sft_protocol"] == ANSWER_PROTOCOL
    prepared = reopened.prepare(plan["plan_id"], session)
    run = prepared["training_run"]
    assert run["status"] == "prepared", run["issues"]
    assert run["config"]["data"]["sft_protocol"] == ANSWER_PROTOCOL
    assert run["preflight"]["sft_protocol"] == ANSWER_PROTOCOL
    assert run["config_digest"] == content_digest(run["config"])
    assert reopened.prepare(plan["plan_id"], session)["run_id"] == run["run_id"]
    another = plans.save(session, proposal, context, [])
    tampered = deepcopy(another)
    tampered["proposal"]["sft_protocol"] = LEGACY_PROTOCOL
    with sqlite3.connect(plans.database) as connection:
        connection.execute(
            "UPDATE plans SET snapshot=? WHERE id=?", (canonical(tampered), another["plan_id"])
        )
    with pytest.raises(ValueError, match="方案内容不一致"):
        reopened.prepare(another["plan_id"], session)
    assert len(training.list_runs()) == 1


def test_start_rechecks_selected_answer_contract_before_launch(answer_environment):
    _, session, training, _ = answer_environment
    run = _prepare(answer_environment)
    assert run["status"] == "prepared"
    incompatible = _tokenizer(answer_environment)
    incompatible.backend_tokenizer.post_processor = processors.TemplateProcessing(
        single="[BOS] $A [EOS]", special_tokens=[("[BOS]", 2), ("[EOS]", 3)]
    )
    assert preflight_dataset(session, incompatible, 32)["status"] == "passed"
    with (
        patch("src.workbench.training_runs.training_tokenizer", return_value=incompatible),
        patch.object(training.runner, "launch_training") as launch,
        pytest.raises(ValueError, match="启动前数据校验未通过"),
    ):
        training.start(run["run_id"], session, acknowledge_warnings=True)
    launch.assert_not_called()
    assert not (training._directory(run["run_id"]) / "started").exists()


def _cpu_worker(env):
    # The actual worker runs unchanged through runpy. Disable hardware discovery
    # only in this temporary test launcher to avoid competing for MPS memory.
    worker = env[2].project_root / "scripts" / "workbench_train.py"
    product = worker.with_name("workbench_train_product.py")
    product.write_text(worker.read_text())
    worker.write_text(
        "import runpy\nimport torch\n"
        "torch.cuda.is_available = lambda: False\n"
        "torch.backends.mps.is_available = lambda: False\n"
        f"runpy.run_path({str(product)!r}, run_name='__main__')\n"
    )


class LinkageRuntime:
    def __init__(self, model):
        pass

    def generate(self, prompt, protocol):
        return Generation("yes")

    def close(self):
        pass


def test_real_cpu_answer_worker_and_iteration_inheritance_and_explicit_switch(
    answer_environment, tmp_path
):
    intake, session, training, model = answer_environment
    _cpu_worker(answer_environment)
    parent = _prepare(answer_environment, model_options={"device_map": "cpu"})
    assert parent["status"] == "prepared", parent["issues"]
    training.start(parent["run_id"], session)
    parent = _wait(training, parent["run_id"])
    assert parent["status"] == "succeeded", training.read_logs(parent["run_id"], tail=100)
    manifest = json.loads(Path(parent["artifacts"]["manifest"]).read_text())
    assert manifest["config_digest"] == parent["config_digest"]
    assert parent["config"]["data"]["sft_protocol"] == ANSWER_PROTOCOL
    service = IterationService(tmp_path / "iterations", training.root, tmp_path / "evaluations")
    service.training = training
    report = service.evaluations.compare(
        session,
        [
            EvaluationModel("base", str(model)),
            EvaluationModel("parent", str(model), parent["output_dir"]),
        ],
        EvaluationProtocol("classification_exact"),
        runtime_factory=LinkageRuntime,
    )
    proposals = []
    for override in ({}, {"sft_protocol": LEGACY_PROTOCOL}):
        proposal = service.propose(
            session,
            parent_run_id=parent["run_id"],
            evaluation_id=report.evaluation_id,
            hypothesis="仅验证监督协议继承",
            expected_outcome="保持训练与评测证据绑定",
            changes="fixture outputs only verify linkage, not business benefit",
            **override,
        )
        assert proposal["options"]["sft_protocol"] == override.get("sft_protocol", ANSWER_PROTOCOL)
        service.confirm(proposal["iteration_id"], session)
        proposals.append(proposal)
    session = intake.materialize_dataset(
        session.session_id, session.revision, evaluation_suite=proposals[0]["evaluation_suite"]
    )
    for proposal in proposals:
        prepared = service.prepare(proposal["iteration_id"], session)
        run = prepared["training_run"]
        assert run["status"] == "prepared", run["issues"]
        assert run["config"]["data"]["sft_protocol"] == proposal["options"]["sft_protocol"]
        assert run["preflight"]["sft_protocol"] == proposal["options"]["sft_protocol"]


def test_real_worker_independently_rejects_answer_boundary_before_training(environment):  # noqa: F811
    _, session, training, _ = environment
    _cpu_worker(environment)
    # The original fixture automatically appends prompt EOS. Legacy consumption
    # succeeds, but answer-only must fail its exact-prefix proof in the worker.
    passed = preflight_dataset(session, _tokenizer(environment), 32)
    assert passed["status"] == "passed"
    with patch("src.workbench.training_runs.preflight_dataset", return_value=passed):
        run = _prepare(environment, model_options={"device_map": "cpu"})
        training.start(run["run_id"], session)
    result = _wait(training, run["run_id"])
    assert result["status"] == "failed"
    assert result["failure"]["stage"] == "preflight", training.read_logs(run["run_id"], tail=100)
    assert "answer_supervision_contract" in result["failure"]["message"]
    assert result["artifacts"] == {}


@pytest.mark.parametrize("historical", [False, True])
@pytest.mark.parametrize("unexpected_data_change", [False, True])
def test_technical_recovery_preserves_selected_or_historical_default_protocol(
    answer_environment, historical, unexpected_data_change
):
    _, session, training, _ = answer_environment
    parent = _prepare(
        answer_environment, sft_protocol=LEGACY_PROTOCOL if historical else ANSWER_PROTOCOL
    )
    assert parent["status"] == "prepared"
    if historical:
        del parent["config"]["data"]["sft_protocol"]
        parent["config_digest"] = content_digest(parent["config"])
    parent.update(status="failed", recover_technical_failures=True)
    directory = training._directory(parent["run_id"])
    write_json(directory / "config.json", parent["config"])
    write_json(directory / "run.json", parent)
    exc = _exception()
    write_json(
        directory / "result.json",
        {
            "run_id": parent["run_id"],
            "status": "failed",
            "failure": {
                "stage": "training",
                "type": "RuntimeError",
                "message": str(exc),
                "technical": failure_evidence(exc),
            },
        },
    )
    recovery = TrainingRecoveryService(
        training.root, training.project_root, training.python_executable
    )
    prepare = recovery.training.prepare

    def prepare_retry(*args, **kwargs):
        child = prepare(*args, **kwargs)
        if unexpected_data_change:
            child["config"]["data"]["max_samples"] = 123
        return child

    with (
        patch.object(recovery.training, "prepare", side_effect=prepare_retry),
        patch.object(
            recovery.training.runner, "launch_training", return_value="test-only"
        ) as launch,
    ):
        result = recovery.recover(parent["run_id"], session, acknowledge_warnings=True)
    if unexpected_data_change:
        assert result["status"] == "retry_blocked"
        launch.assert_not_called()
    else:
        assert result["status"] == "retry_started", result
        launch.assert_called_once()
        child = training._load(result["child_run_id"])
        expected = {"sft_protocol": LEGACY_PROTOCOL, **parent["config"]["data"]}
        assert child["config"]["data"] == expected
        assert child["preflight"]["sft_protocol"] == expected["sft_protocol"]


def test_iteration_accepts_historical_retry_default_but_rejects_other_data_changes(
    tmp_path, monkeypatch
):
    service = IterationService(tmp_path / "iterations", tmp_path / "runs", tmp_path / "eval")
    parent = {
        "run_id": "parent",
        "status": "failed",
        "recover_technical_failures": True,
        "session_id": "task",
        "dataset": {"version": "same"},
        "model_identity": {},
        "recovery": {"child_run_id": "child"},
        "config": {
            "model": {},
            "lora": {},
            "data": {"dataset_name": "immutable"},
            "training": {
                "batch_size": 4,
                "gradient_accumulation_steps": 1,
                "gradient_checkpointing": False,
                "output_dir": "parent",
            },
        },
    }
    child = deepcopy(parent)
    child.update(run_id="child", status="succeeded", recovery_parent_run_id="parent", recovery={})
    child["config"]["data"]["sft_protocol"] = LEGACY_PROTOCOL
    child["config"]["training"].update(
        batch_size=2, gradient_accumulation_steps=2, output_dir="child"
    )
    monkeypatch.setattr(
        service.training, "get_status", lambda key: parent if key == "parent" else child
    )
    assert service._effective_training_run("parent")["run_id"] == "child"
    child["config"]["data"]["max_samples"] = 1
    with pytest.raises(ValueError, match="数据"):
        service._effective_training_run("parent")
    del child["config"]["data"]["max_samples"]
    child["config"]["data"]["sft_protocol"] = ANSWER_PROTOCOL
    with pytest.raises(ValueError, match="数据"):
        service._effective_training_run("parent")
