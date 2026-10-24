"""Planning tool contracts; scripted advice is not evidence of model quality."""

import json
import sys
from copy import deepcopy
from types import SimpleNamespace

import pytest

from src.agent.training import SYSTEM, TrainingProposal, recommend_training


class ScriptedPlanner:
    model = "scripted-training-planner"

    def __init__(self, steps):
        self.steps = iter(steps)
        self.seen = []

    def complete(self, messages, tools):
        self.seen.append(deepcopy(messages))
        name, arguments = next(self.steps)
        if callable(arguments):
            arguments = arguments(messages)
        return {
            "tool_calls": [
                {
                    "id": f"call-{len(self.seen)}",
                    "type": "function",
                    "function": {"name": name, "arguments": json.dumps(arguments)},
                }
            ],
            "reasoning_content": "根据工具证据选择下一步",
        }


def evidence(messages, prefix):
    observed = [json.loads(message["content"]) for message in messages if message["role"] == "tool"]
    return [
        entry["evidence_id"]
        for entry in observed
        if entry.get("evidence_id", "").startswith(prefix)
    ][-1]


def proposal(path, length=128, status="ready"):
    return {
        "status": status,
        "model_path": str(path) if path else None,
        "max_length": length if path else None,
        "training_options": {"num_epochs": 1, "batch_size": 1, "learning_rate": 0.0001},
        "lora_options": {"r": 4, "target_modules": ["q_proj", "v_proj"]},
        "model_options": {"quantization_bits": None},
        "rationale": ["先验证本地候选、数据监督与单轮技术基线。"],
        "limitations": ["tokenizer预检不能证明实际显存充足或业务效果。"],
        "business_questions": [],
    }


def submission(value, with_probe=True):
    def build(messages):
        identities = [evidence(messages, "context:")]
        if with_probe:
            identities.append(evidence(messages, "probe:"))
        return {"proposal": value, "evidence_ids": identities}

    return build


@pytest.fixture
def backend(monkeypatch):
    calls, records = [], []

    class Service:
        def __init__(self, root, training_root):
            self.root = root

        def context(self, session, paths):
            calls.append(("context", paths))
            return {
                "session_id": session.session_id,
                "session_revision": 1,
                "business": {"goal": session.goal},
                "platform": {"device": "cpu"},
                "statistics": {"row_counts": {"train": 10, "validation": 2, "test": 2}},
                "models": [{"model_path": path, "status": "available"} for path in paths],
            }

        def probe(self, session, path, length):
            calls.append(("probe", path, length))
            blocked = length < 32
            return {
                "model_path": path,
                "status": "blocked" if blocked else "passed",
                "preflight": {
                    "status": "blocked" if blocked else "passed",
                    "supervision_strategy": "full_prompt_padding_masked",
                    "issues": [{"severity": "blocking", "code": "answer_lost"}] if blocked else [],
                },
                "issues": [],
            }

        def save(self, session, proposed, context, trace):
            record = deepcopy(
                {
                    "plan_id": "fixture-plan",
                    "proposal": proposed,
                    "context": context,
                    "tool_trace": trace,
                }
            )
            records.append(record)
            return record

    monkeypatch.setitem(
        sys.modules, "src.workbench.training_plans", SimpleNamespace(TrainingPlanService=Service)
    )
    return calls, records


def test_ready_requires_observed_context_exact_successful_probe_and_local_candidate(
    backend, tmp_path
):
    calls, records = backend
    path = (tmp_path / "model").resolve()
    ready = proposal(path)
    client = ScriptedPlanner(
        [
            ("submit_training_plan", {"proposal": ready, "evidence_ids": ["invented"]}),
            ("inspect_training_context", {}),
            ("submit_training_plan", submission(ready, with_probe=False)),
            ("probe_training", {"model_path": str(tmp_path / "unknown"), "max_length": 128}),
            ("probe_training", {"model_path": str(path), "max_length": True}),
            ("probe_training", {"model_path": str(path), "max_length": 8}),
            ("submit_training_plan", submission(proposal(path, 8))),
            ("probe_training", {"model_path": str(path), "max_length": 128}),
            ("submit_training_plan", submission(proposal(path, 256))),
            ("submit_training_plan", submission(ready)),
        ]
    )
    result = recommend_training(
        SimpleNamespace(session_id="fixture", goal="业务分类"),
        [str(path)],
        client,
        tmp_path / "plans",
        tmp_path / "runs",
    )
    assert [item["ok"] for item in result["tool_trace"]] == [
        False,
        True,
        False,
        False,
        False,
        True,
        False,
        True,
        False,
        True,
    ]
    assert [call for call in calls if call[0] == "probe"] == [
        ("probe", str(path), 8),
        ("probe", str(path), 128),
    ]
    assert len(records) == 1
    assert result["proposal"]["max_length"] == 128
    assert result["tool_trace"][-1]["agent_model"] == client.model
    assert not (tmp_path / "runs").exists()
    assert any(
        message.get("reasoning_content")
        for message in client.seen[-1]
        if message["role"] == "assistant"
    )
    assert result["proposal"]["sft_protocol"] == "alpaca_full_prompt_v1"


@pytest.fixture
def protocol_backend(backend, monkeypatch):
    calls, records = backend
    service = sys.modules["src.workbench.training_plans"].TrainingPlanService

    def probe(self, session, path, length, *, sft_protocol="alpaca_full_prompt_v1"):
        calls.append(("probe", path, length, sft_protocol))
        return {
            "model_path": path,
            "max_length": length,
            "sft_protocol": sft_protocol,
            "status": "passed",
            "issues": [],
            "preflight": {
                "status": "passed",
                "issues": [],
                "sft_protocol": sft_protocol,
                "supervision_strategy": (
                    "full_prompt_padding_masked"
                    if sft_protocol == "alpaca_full_prompt_v1"
                    else "answer_and_eos_only"
                ),
            },
        }

    monkeypatch.setattr(service, "probe", probe)
    return calls, records


def test_answer_only_proposal_requires_matching_protocol_probe(protocol_backend, tmp_path):
    path = tmp_path / "model"
    value = proposal(path) | {"sft_protocol": "alpaca_answer_only_v1"}
    client = ScriptedPlanner(
        [
            ("inspect_training_context", {}),
            (
                "probe_training",
                {
                    "model_path": str(path),
                    "max_length": 128,
                    "sft_protocol": "alpaca_answer_only_v1",
                },
            ),
            ("submit_training_plan", submission(value)),
        ]
    )
    result = recommend_training(
        SimpleNamespace(session_id="fixture", goal="目标"),
        [str(path)],
        client,
        tmp_path / "plans",
        tmp_path / "runs",
    )
    assert all(item["ok"] for item in result["tool_trace"])
    assert result["proposal"]["sft_protocol"] == "alpaca_answer_only_v1"
    assert result["context"]["probes"][0]["sft_protocol"] == "alpaca_answer_only_v1"
    assert ("probe", str(path), 128, "alpaca_answer_only_v1") in protocol_backend[0]
    assert len(protocol_backend[1]) == 1


def test_full_and_answer_probes_cannot_be_borrowed_even_at_same_path_and_length(
    protocol_backend, tmp_path
):
    path = tmp_path / "model"
    full = proposal(path)
    answer = full | {"sft_protocol": "alpaca_answer_only_v1"}

    def cite_full(messages):
        entries = [
            json.loads(message["content"]) for message in messages if message["role"] == "tool"
        ]
        full_id = next(
            entry["evidence_id"]
            for entry in entries
            if entry.get("sft_protocol") == "alpaca_full_prompt_v1"
        )
        return {"proposal": answer, "evidence_ids": [evidence(messages, "context:"), full_id]}

    client = ScriptedPlanner(
        [
            ("inspect_training_context", {}),
            ("probe_training", {"model_path": str(path), "max_length": 128}),
            ("submit_training_plan", submission(answer)),
            (
                "probe_training",
                {
                    "model_path": str(path),
                    "max_length": 128,
                    "sft_protocol": "alpaca_answer_only_v1",
                },
            ),
            ("submit_training_plan", submission(full)),
            ("submit_training_plan", cite_full),
            ("submit_training_plan", submission(answer)),
        ]
    )
    result = recommend_training(
        SimpleNamespace(session_id="fixture", goal="目标"),
        [str(path)],
        client,
        tmp_path / "plans",
        tmp_path / "runs",
    )
    assert [entry["ok"] for entry in result["tool_trace"]] == [
        True,
        True,
        False,
        True,
        False,
        False,
        True,
    ]
    probes = result["context"]["probes"]
    assert {probe["sft_protocol"] for probe in probes} == {
        "alpaca_full_prompt_v1",
        "alpaca_answer_only_v1",
    }
    assert probes[0]["evidence_id"] != probes[1]["evidence_id"]
    assert len(protocol_backend[1]) == 1


@pytest.mark.parametrize("protocol", ["native_chat_answer_only_v1", "answer_only", None, 1, []])
def test_forged_or_unexposed_protocol_is_rejected_before_probe_or_save(backend, tmp_path, protocol):
    path = tmp_path / "model"
    invalid = proposal(path) | {"sft_protocol": protocol}
    client = ScriptedPlanner(
        [
            ("inspect_training_context", {}),
            (
                "probe_training",
                {"model_path": str(path), "max_length": 128, "sft_protocol": protocol},
            ),
            ("submit_training_plan", submission(invalid, with_probe=False)),
            (
                "submit_training_plan",
                submission(proposal(None, status="unsupported"), with_probe=False),
            ),
        ]
    )
    result = recommend_training(
        SimpleNamespace(session_id="fixture", goal="目标"),
        [str(path)],
        client,
        tmp_path / "plans",
        tmp_path / "runs",
    )
    assert [entry["ok"] for entry in result["tool_trace"]] == [True, False, False, True]
    assert [call[0] for call in backend[0]] == ["context"]
    assert len(backend[1]) == 1


@pytest.mark.parametrize("mutation", ["top_protocol", "preflight_protocol", "preflight_strategy"])
def test_probe_result_cannot_mislabel_actual_supervision(
    protocol_backend, tmp_path, monkeypatch, mutation
):
    service = sys.modules["src.workbench.training_plans"].TrainingPlanService
    original = service.probe

    def misbound(self, *args, **kwargs):
        observed = original(self, *args, **kwargs)
        if mutation == "top_protocol":
            observed["sft_protocol"] = "alpaca_full_prompt_v1"
        elif mutation == "preflight_protocol":
            observed["preflight"]["sft_protocol"] = "alpaca_full_prompt_v1"
        else:
            observed["preflight"]["supervision_strategy"] = "full_prompt_padding_masked"
        return observed

    monkeypatch.setattr(service, "probe", misbound)
    path = tmp_path / "model"
    client = ScriptedPlanner(
        [
            ("inspect_training_context", {}),
            (
                "probe_training",
                {
                    "model_path": str(path),
                    "max_length": 128,
                    "sft_protocol": "alpaca_answer_only_v1",
                },
            ),
            (
                "submit_training_plan",
                submission(proposal(None, status="unsupported"), with_probe=False),
            ),
        ]
    )
    result = recommend_training(
        SimpleNamespace(session_id="fixture", goal="目标"),
        [str(path)],
        client,
        tmp_path / "plans",
        tmp_path / "runs",
    )
    assert [entry["ok"] for entry in result["tool_trace"]] == [True, False, True]
    assert result["context"]["probes"] == []
    assert "监督协议与请求不一致" in result["tool_trace"][1]["error"]


def test_agent_schema_and_guidance_expose_two_alpaca_objectives_only():
    schema = TrainingProposal.model_json_schema()["properties"]["sft_protocol"]
    assert schema["default"] == "alpaca_full_prompt_v1"
    assert schema["enum"] == ["alpaca_full_prompt_v1", "alpaca_answer_only_v1"]
    assert "精确 token 前缀边界" in SYSTEM
    assert "完整答案及 EOS" in SYSTEM
    assert "不能宣称只答案监督必然更优" in SYSTEM
    assert "不得拿全提示协议的 probe 证据提交只答案方案" in SYSTEM


def test_probe_tool_protocol_is_optional_and_limited_to_two_public_values(backend, tmp_path):
    class SchemaPlanner(ScriptedPlanner):
        def complete(self, messages, tools):
            probe = next(
                tool["function"] for tool in tools if tool["function"]["name"] == "probe_training"
            )
            parameters = probe["parameters"]
            assert parameters["required"] == ["model_path", "max_length"]
            assert parameters["properties"]["sft_protocol"] == {
                "type": "string",
                "enum": ["alpaca_full_prompt_v1", "alpaca_answer_only_v1"],
                "default": "alpaca_full_prompt_v1",
            }
            return super().complete(messages, tools)

    client = SchemaPlanner(
        [
            ("inspect_training_context", {}),
            (
                "submit_training_plan",
                submission(proposal(None, status="unsupported"), with_probe=False),
            ),
        ]
    )
    recommend_training(
        SimpleNamespace(session_id="fixture", goal="目标"),
        [],
        client,
        tmp_path / "plans",
        tmp_path / "runs",
    )


def test_missing_data_or_no_model_can_stop_with_explanation_without_probe(backend, tmp_path):
    calls, records = backend
    value = proposal(None, status="needs_data")
    value["business_questions"] = ["请确认监督答案的业务含义。"]
    client = ScriptedPlanner(
        [
            ("inspect_training_context", {}),
            ("submit_training_plan", submission(value, with_probe=False)),
        ]
    )
    result = recommend_training(
        SimpleNamespace(session_id="fixture", goal="目标未确认"),
        [],
        client,
        tmp_path / "plans",
        tmp_path / "runs",
    )
    assert result["proposal"]["status"] == "needs_data"
    assert result["proposal"]["model_path"] is None
    assert [call[0] for call in calls] == ["context"]
    assert len(records) == 1


def test_unresolved_business_question_or_fabricated_evidence_cannot_be_ready(backend, tmp_path):
    path = tmp_path / "model"
    ready = proposal(path)
    unresolved = {**ready, "business_questions": ["标签是什么意思？"]}
    client = ScriptedPlanner(
        [
            ("inspect_training_context", {}),
            ("probe_training", {"model_path": str(path), "max_length": 128}),
            (
                "submit_training_plan",
                lambda messages: {
                    **submission(ready)(messages),
                    "evidence_ids": [evidence(messages, "context:"), "probe:invented"],
                },
            ),
            ("submit_training_plan", submission(unresolved)),
            ("submit_training_plan", submission({**unresolved, "status": "needs_data"})),
        ]
    )
    result = recommend_training(
        SimpleNamespace(session_id="fixture", goal="目标"),
        [str(path)],
        client,
        tmp_path / "plans",
        tmp_path / "runs",
    )
    assert [item["ok"] for item in result["tool_trace"]] == [True, True, False, False, True]
    assert result["proposal"]["status"] == "needs_data"


def test_agent_cannot_start_train_or_claim_unknown_model(backend, tmp_path):
    client = ScriptedPlanner(
        [
            ("inspect_training_context", {}),
            ("start_training", {"run_id": "invented"}),
            (
                "submit_training_plan",
                submission(proposal(tmp_path / "invented", status="unsupported"), False),
            ),
            ("submit_training_plan", submission(proposal(None, status="unsupported"), False)),
        ]
    )
    result = recommend_training(
        SimpleNamespace(session_id="fixture", goal="目标"),
        [],
        client,
        tmp_path / "plans",
        tmp_path / "runs",
    )
    assert [item["ok"] for item in result["tool_trace"]] == [True, False, False, True]
    assert result["proposal"]["status"] == "unsupported"


def test_invalid_tool_message_stops_without_persisting_plan(backend, tmp_path):
    class InvalidClient:
        model = "invalid"

        def complete(self, messages, tools):
            return {"tool_calls": {"not": "a list"}}

    with pytest.raises(RuntimeError, match="无效"):
        recommend_training(
            SimpleNamespace(session_id="fixture", goal="目标"),
            [],
            InvalidClient(),
            tmp_path / "plans",
            tmp_path / "runs",
        )
    assert backend[1] == []


def test_real_local_tokenizer_probe_saves_plan_without_preparing_training(tmp_path, monkeypatch):
    pytest.importorskip("datasets")
    pytest.importorskip("peft")
    from src.workbench.training_plans import TrainingPlanService
    from tests.unit.test_workbench_training_runs import environment

    _, session, training, model = environment.__wrapped__(tmp_path, monkeypatch)
    value = proposal(model, 32)
    value["lora_options"] = {"r": 4, "target_modules": "all-linear"}
    value["model_options"] = {"quantization_bits": None, "torch_dtype": "float32"}
    client = ScriptedPlanner(
        [
            ("inspect_training_context", {}),
            ("probe_training", {"model_path": str(model), "max_length": 32}),
            ("submit_training_plan", submission(value)),
        ]
    )
    result = recommend_training(session, [str(model)], client, tmp_path / "plans", training.root)
    assert result["status"] == "ready"
    service = TrainingPlanService(tmp_path / "plans", training.root)
    assert service.get(result["plan_id"]) == result
    assert training.list_runs(session.session_id) == []
    observed = [
        json.loads(message["content"]) for message in client.seen[-1] if message["role"] == "tool"
    ]
    probe = next(item for item in observed if item.get("evidence_id", "").startswith("probe:"))
    assert probe["preflight"]["status"] == "passed"
    assert probe["preflight"]["inspected_rows"] == session.dataset.statistics["total_rows"]
    assert probe["preflight"]["supervision_strategy"] == "full_prompt_padding_masked"
    assert "rows" not in probe["preflight"]
