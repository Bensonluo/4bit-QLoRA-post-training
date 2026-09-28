"""Final acceptance freezes business standards and cannot recycle exposed questions."""

import json
import sqlite3

import pytest

pytest.importorskip("datasets")
pytest.importorskip("transformers")

from src.workbench.acceptance import AcceptanceService
from src.workbench.business_evaluation import EvaluationProtocol, Generation
from src.workbench.sources import canonical
from tests.unit.test_business_evaluation import model_paths as model_paths
from tests.unit.test_business_evaluation import task as task


def criteria(**changes):
    return {
        "metric": "exact_match",
        "minimum_score": 1.0,
        "minimum_cases": 1,
        "business_standard": "全部留出业务题完整输出与标准答案严格一致。",
        **changes,
    }


def runtime(service, record, events, response=None):
    from src.data.loaders import render_alpaca_prompt
    from src.workbench.evaluation_suites import evaluation_cases

    snapshot = service._snapshot(record)
    cases = evaluation_cases(snapshot, split="test")["records"]
    answers = {render_alpaca_prompt({**row, "output": ""}): row["output"] for row in cases}

    class Runtime:
        def __init__(self, model):
            events.append(("load", model.label))
            self.index = 0

        def generate(self, prompt, protocol):
            index = self.index
            self.index += 1
            events.append(("generate", index))
            return response(index, answers[prompt]) if response else Generation(answers[prompt])

        def close(self):
            events.append(("close",))

    return Runtime


@pytest.mark.parametrize("outcome", ["passed", "failed", "insufficient_evidence"])
def test_real_holdout_report_is_frozen_and_run_is_idempotent(task, model_paths, tmp_path, outcome):
    session = task[1]
    original = session.model_dump()
    service = AcceptanceService(tmp_path / "acceptance", tmp_path / "eval")
    standard = criteria(minimum_cases=100 if outcome == "insufficient_evidence" else 1)
    record = service.prepare(
        session, model_paths[0], EvaluationProtocol("classification_exact"), standard
    )
    assert session.model_dump() == original
    assert session.dataset.evaluation_suite is None
    assert record["case_count"] == session.dataset.statistics["row_counts"]["test"]
    assert service.get(record["acceptance_id"]) == record
    events = []
    result = service.run(
        record["acceptance_id"],
        session,
        runtime_factory=runtime(
            service,
            record,
            events,
            (lambda index, answer: Generation("wrong")) if outcome == "failed" else None,
        ),
    )
    assert result["status"] == "completed", result
    assert result["result"]["decision"] == outcome
    assert result["report"]["dataset"]["purpose"] == "final_acceptance"
    assert result["report"]["dataset"]["split"] == "test"
    assert result["blind_test"] is True
    assert len(result["report"]["models"]) == 1
    assert len([event for event in events if event[0] == "generate"]) == record["case_count"]
    before = list(events)
    assert (
        service.run(
            record["acceptance_id"], session, runtime_factory=runtime(service, record, events)
        )
        == result
    )
    assert events == before
    assert service.evaluations.list_reports() == []
    assert service.list_acceptances(session.session_id)[0] == result


def test_exposure_claim_blocks_other_model_and_threshold_rewrite(task, model_paths, tmp_path):
    session = task[1]
    service = AcceptanceService(tmp_path / "acceptance", tmp_path / "eval")
    protocol = EvaluationProtocol("classification_exact")
    first = service.prepare(session, model_paths[0], protocol, criteria())
    second = service.prepare(session, model_paths[1], protocol, criteria())
    with pytest.raises(ValueError, match="阈值"):
        service.prepare(session, model_paths[1], protocol, criteria(minimum_score=0.5))
    events = []
    service.run(first["acceptance_id"], session, runtime_factory=runtime(service, first, events))
    events.clear()
    result = service.run(
        second["acceptance_id"], session, runtime_factory=runtime(service, second, events)
    )
    assert result["status"] == "blocked"
    assert result["blind_test"] is False
    assert result["result"]["decision"] == "insufficient_evidence"
    assert events == []


def test_reuploaded_same_questions_with_new_suite_id_cannot_reset_blind_test(
    task, model_paths, tmp_path
):
    from tests.unit.test_business_evaluation import suite_rounds

    service = AcceptanceService(tmp_path / "acceptance", tmp_path / "eval")
    original = task[1]
    first = service.prepare(
        original, model_paths[0], EvaluationProtocol("classification_exact"), criteria()
    )
    service.run(first["acceptance_id"], original, runtime_factory=runtime(service, first, []))
    _, updated = suite_rounds(task, tmp_path)
    # A new anchor version changes suite identity, while actual reserved test questions stay the same.
    from src.workbench.materialize import materialize_dataset

    new_reference = service.suites.freeze(updated, new_suite=True)
    updated.dataset = materialize_dataset(
        updated,
        registry_root=updated.dataset.registry_root,
        name=updated.dataset.name,
        evaluation_suite=new_reference,
    )
    second = service.prepare(
        updated, model_paths[1], EvaluationProtocol("classification_exact"), criteria()
    )
    assert second["evaluation_suite"]["suite_id"] != first["evaluation_suite"]["suite_id"]
    result = service.run(second["acceptance_id"], updated)
    assert result["status"] == "blocked"
    assert result["evaluation_id"] is None


def test_group_keys_are_scoped_to_business_task_but_repeated_inputs_stay_revealed():
    from types import SimpleNamespace

    first = [
        {
            "instruction": "分类",
            "input": "A业务事实",
            "output": "标签",
            "metadata": {"group": {"ticket": "001"}},
        }
    ]
    second = [
        {
            "instruction": "分类",
            "input": "B不同业务事实",
            "output": "标签",
            "metadata": {"group": {"ticket": "001"}},
        }
    ]
    left = set(AcceptanceService._exposure_keys(SimpleNamespace(session_id="first"), first))
    other = set(AcceptanceService._exposure_keys(SimpleNamespace(session_id="second"), second))
    same_task = set(AcceptanceService._exposure_keys(SimpleNamespace(session_id="first"), second))
    assert not left & other
    assert left & same_task
    second[0]["input"] = first[0]["input"]
    second[0]["output"] = "changed label"
    assert left & set(
        AcceptanceService._exposure_keys(SimpleNamespace(session_id="second"), second)
    )


def test_open_review_cannot_override_truncation_and_requires_all_rows(task, model_paths, tmp_path):
    service = AcceptanceService(tmp_path / "acceptance", tmp_path / "eval")
    session = task[1]
    record = service.prepare(
        session,
        model_paths[0],
        EvaluationProtocol("open_review"),
        criteria(metric="manual_acceptance_rate"),
    )
    assert record["case_count"] >= 2

    def response(index, answer):
        return (
            Generation("partial answer", truncated=True)
            if index == 0
            else Generation("完整业务回答")
        )

    result = service.run(
        record["acceptance_id"], session, runtime_factory=runtime(service, record, [], response)
    )
    assert result["status"] == "needs_business_review"
    with pytest.raises(ValueError, match="不能被人工"):
        service.review(
            record["acceptance_id"],
            [{"index": 0, "decision": "accepted", "reason": "不能只看前半段"}],
        )
    for index in range(1, record["case_count"]):
        result = service.review(
            record["acceptance_id"],
            [{"index": index, "decision": "accepted", "reason": "完整输出符合预先声明的业务标准"}],
        )
    assert result["status"] == "completed"
    assert result["result"]["decision"] == "failed"
    assert result["result"]["score"] == (record["case_count"] - 1) / record["case_count"]
    assert result["decisions"][0]["locked"] is True


def test_loading_failure_consumes_attempt_and_never_claims_pass(task, model_paths, tmp_path):
    service = AcceptanceService(tmp_path / "acceptance", tmp_path / "eval")
    session = task[1]
    record = service.prepare(
        session,
        model_paths[0],
        EvaluationProtocol("classification_exact"),
        criteria(minimum_score=0),
    )
    attempts = []

    def broken(model):
        attempts.append(model)
        raise RuntimeError("fixture load failure")

    result = service.run(record["acceptance_id"], session, runtime_factory=broken)
    assert result["result"]["decision"] == "insufficient_evidence"
    assert result["evaluation_id"]
    service.run(record["acceptance_id"], session, runtime_factory=broken)
    assert len(attempts) == 1
    again = service.prepare(
        session,
        model_paths[1],
        EvaluationProtocol("classification_exact"),
        criteria(minimum_score=0),
    )
    assert service.run(again["acceptance_id"], session)["status"] == "blocked"


@pytest.mark.parametrize("change", ["model", "snapshot", "criteria", "report"])
def test_changed_frozen_evidence_is_rejected(task, model_paths, tmp_path, change):
    from pathlib import Path

    service = AcceptanceService(tmp_path / "acceptance", tmp_path / "eval")
    session = task[1]
    record = service.prepare(
        session, model_paths[0], EvaluationProtocol("classification_exact"), criteria()
    )
    identity = record["acceptance_id"]
    if change == "model":
        (Path(model_paths[0].base_model) / "pytorch_model.bin").write_bytes(b"changed")
        with pytest.raises(ValueError, match="模型内容"):
            service.run(identity, session)
    elif change == "snapshot":
        updated = session.model_copy(deep=True)
        updated.revision += 1
        with pytest.raises(ValueError, match="快照已变化"):
            service.run(identity, updated)
    elif change == "criteria":
        record["criteria"]["minimum_score"] = 0
        with sqlite3.connect(service.database) as connection:
            connection.execute(
                "UPDATE acceptances SET snapshot=? WHERE id=?", (canonical(record), identity)
            )
        with pytest.raises(ValueError, match="被修改"):
            service.get(identity)
    else:
        record = service.run(identity, session, runtime_factory=runtime(service, record, []))
        path = service.evaluations.root / f"{record['evaluation_id']}.json"
        report = json.loads(path.read_text())
        report["models"][0]["rows"][0]["output"] = "forged output"
        path.write_text(json.dumps(report))
        with pytest.raises(ValueError, match="报告被修改"):
            service.get(identity)


@pytest.mark.parametrize(
    "override",
    [
        {"minimum_score": True},
        {"minimum_score": float("nan")},
        {"minimum_cases": 0},
        {"minimum_cases": True},
        {"business_standard": " "},
        {"metric": "manual_acceptance_rate"},
    ],
)
def test_acceptance_has_no_implicit_business_threshold(task, model_paths, tmp_path, override):
    service = AcceptanceService(tmp_path / "acceptance", tmp_path / "eval")
    with pytest.raises(ValueError):
        service.prepare(
            task[1],
            model_paths[0],
            EvaluationProtocol("classification_exact"),
            criteria(**override),
        )
    assert service.list_acceptances() == []


def test_unknown_adapter_cannot_turn_perfect_outputs_into_independent_pass(
    task, model_paths, tmp_path
):
    service = AcceptanceService(tmp_path / "acceptance", tmp_path / "eval")
    record = service.prepare(
        task[1], model_paths[1], EvaluationProtocol("classification_exact"), criteria()
    )
    assert record["training_provenance"]["status"] == "not_available"
    result = service.run(
        record["acceptance_id"], task[1], runtime_factory=runtime(service, record, [])
    )
    assert result["result"]["score"] == 1
    assert result["result"]["observed_decision"] == "passed"
    assert result["result"]["decision"] == "insufficient_evidence"


def test_workbench_training_receipt_proves_fixed_holdout_partition(task, model_paths, tmp_path):
    from src.workbench.business_evaluation import EvaluationModel
    from tests.unit.test_evaluation_diagnostics import training_report

    training_evaluation, _ = training_report(task, model_paths, tmp_path)
    selected = EvaluationModel(**training_evaluation.models[1]["requested_model"])
    service = AcceptanceService(tmp_path / "acceptance", tmp_path / "final-eval")
    record = service.prepare(
        task[1], selected, EvaluationProtocol("classification_exact"), criteria()
    )
    assert record["training_provenance"]["status"] == "available"
    assert record["training_provenance"]["training_dataset"]["version"] == task[1].dataset.version
    result = service.run(
        record["acceptance_id"], task[1], runtime_factory=runtime(service, record, [])
    )
    assert result["result"]["decision"] == "passed", result


def test_model_changed_during_generation_keeps_report_but_cannot_pass(task, model_paths, tmp_path):
    from pathlib import Path

    service = AcceptanceService(tmp_path / "acceptance", tmp_path / "eval")
    record = service.prepare(
        task[1], model_paths[0], EvaluationProtocol("classification_exact"), criteria()
    )

    def replace_weights(index, answer):
        if index == 0:
            (Path(model_paths[0].base_model) / "pytorch_model.bin").write_bytes(
                b"changed-during-generation"
            )
        return Generation(answer)

    result = service.run(
        record["acceptance_id"],
        task[1],
        runtime_factory=runtime(service, record, [], replace_weights),
    )
    assert result["status"] == "failed"
    assert result["result"]["decision"] == "insufficient_evidence"
    assert result["evaluation_id"]
    assert result["report"]["models"][0]["metrics"]["exact_match"] == 1


def test_prepare_snapshots_only_the_four_task_spec_elements(task, model_paths, tmp_path):
    service = AcceptanceService(tmp_path / "acceptance", tmp_path / "eval")
    session = task[1]
    projection = {
        "session_id": session.session_id,
        "revision": session.revision,
        "goal": {"goal": "根据客户首次描述判断问题类型", "training_approach": None},
        "answer_semantics": {"instruction": "判断问题类别"},
        "scoring": {"confirmed": [], "draft_count": 0},
        "temporal_split": None,
        "acceptance": {"state": "未冻结", "records": []},
        "latest_iteration": None,
    }
    record = service.prepare(
        session,
        model_paths[0],
        EvaluationProtocol("classification_exact"),
        criteria(),
        task_spec=projection,
    )
    # 只快照四要素:session_id/acceptance/latest_iteration 等投影外壳不进记录。
    assert record["task_spec"] == {
        "goal": projection["goal"],
        "answer_semantics": projection["answer_semantics"],
        "scoring": projection["scoring"],
        "temporal_split": None,
    }
    # 快照信息性存储,不进 frozen_digest:读回核验照常通过且内容一致。
    assert service.get(record["acceptance_id"])["task_spec"] == record["task_spec"]
    # 不传 task_spec(旧调用方式):快照为空 dict,摘要如实不渲染该段。
    plain = service.prepare(
        session, model_paths[1], EvaluationProtocol("classification_exact"), criteria()
    )
    assert plain["task_spec"] == {}
