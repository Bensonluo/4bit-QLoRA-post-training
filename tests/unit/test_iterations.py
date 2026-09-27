"""Real local tiny-model rounds; fixture outputs test linkage, not business improvement."""

from copy import deepcopy
from pathlib import Path

import pytest

from src.workbench.business_evaluation import EvaluationModel, EvaluationProtocol, Generation
from src.workbench.iterations import IterationService
from tests.unit.test_workbench_training_runs import _prepare, _wait, environment  # noqa: F401


def test_technical_retry_keeps_one_business_round_and_validates_both_directions(
    tmp_path, monkeypatch
):
    service = IterationService(tmp_path / "iterations", tmp_path / "runs", tmp_path / "eval")
    parent = {
        "run_id": "parent",
        "status": "failed",
        "recover_technical_failures": True,
        "session_id": "task",
        "dataset": {"version": "same"},
        "model_identity": {"weights": "same"},
        "recovery": {"child_run_id": "child"},
        "config": {
            "model": {},
            "lora": {},
            "data": {},
            "training": {
                "batch_size": 4,
                "gradient_accumulation_steps": 2,
                "gradient_checkpointing": False,
                "learning_rate": 0.001,
                "output_dir": "parent-model",
            },
        },
    }
    child = deepcopy(parent)
    child.update(run_id="child", status="succeeded", recovery_parent_run_id="parent", recovery={})
    child["config"]["training"].update(
        batch_size=2, gradient_accumulation_steps=4, output_dir="child-model"
    )
    monkeypatch.setattr(
        service.training, "get_status", lambda identity: parent if identity == "parent" else child
    )
    assert service._effective_training_run("parent")["run_id"] == "child"
    child["recovery_parent_run_id"] = "unrelated"
    with pytest.raises(ValueError, match="身份"):
        service._effective_training_run("parent")
    child["recovery_parent_run_id"] = "parent"
    child["config"]["training"]["learning_rate"] = 0.002
    with pytest.raises(ValueError, match="范围"):
        service._effective_training_run("parent")
    child["config"]["training"]["learning_rate"] = 0.001
    child["config"]["training"]["gradient_accumulation_steps"] = 1
    with pytest.raises(ValueError, match="范围"):
        service._effective_training_run("parent")


class FixtureRuntime:
    def __init__(self, model):
        self.model = model

    def generate(self, prompt, protocol):
        return Generation("fixture output")

    def close(self):
        pass


def test_confirmed_round_executes_and_requires_actual_three_model_evidence(environment, tmp_path):  # noqa: F811
    intake, session, training, model = environment
    parent = _prepare(environment)
    training.start(parent["run_id"], session)
    parent = _wait(training, parent["run_id"])
    assert parent["status"] == "succeeded", parent
    service = IterationService(tmp_path / "iterations", training.root, tmp_path / "eval")
    service.training = training
    models = [
        EvaluationModel("base", str(model)),
        EvaluationModel("parent", str(model), parent["output_dir"]),
    ]
    report = service.evaluations.compare(
        session, models, EvaluationProtocol("classification_exact"), runtime_factory=FixtureRuntime
    )
    record = service.propose(
        session,
        parent_run_id=parent["run_id"],
        evaluation_id=report.evaluation_id,
        hypothesis="增加训练轮数可能减少格式错误",
        expected_outcome="固定开发题完整输出严格正确率提高",
        changes="从相同基座训练两轮，其余参数与数据不变",
        training_options={"num_epochs": 2},
    )
    identifier = record["iteration_id"]
    with pytest.raises(ValueError, match="确认"):
        service.prepare(identifier, session)
    service.confirm(identifier, session)
    with pytest.raises(ValueError, match="套件"):
        service.prepare(identifier, session)
    session = intake.materialize_dataset(
        session.session_id, session.revision, evaluation_suite=record["evaluation_suite"]
    )
    record = service.prepare(identifier, session)
    assert record["status"] == "prepared", record
    assert record["training_start"] == "base"
    assert record["training_run"]["config"]["training"]["num_epochs"] == 2
    with pytest.raises(ValueError, match="一次"):
        service.prepare(identifier, session)
    service.start(identifier, session)
    run = _wait(training, record["new_run_id"])
    assert run["status"] == "succeeded", run
    with pytest.raises(ValueError, match="同题"):
        service.decide(identifier, "adopt", "尚未对照不可采用")
    incomplete = service.evaluations.compare(
        session, models, EvaluationProtocol("classification_exact"), runtime_factory=FixtureRuntime
    )
    with pytest.raises(ValueError, match="本轮实际"):
        service.bind_evaluation(identifier, session, incomplete.evaluation_id)
    models.append(EvaluationModel("new", str(model), run["output_dir"]))
    report = service.evaluations.compare(
        session, models, EvaluationProtocol("classification_exact"), runtime_factory=FixtureRuntime
    )
    result = service.bind_evaluation(identifier, session, report.evaluation_id)
    assert result["status"] == "evaluated"
    assert len(result["results"]) == 3
    result = service.decide(
        identifier, "insufficient_evidence", "虚构输出只验证软件衔接，不能证明业务收益"
    )
    assert result["status"] == "decided"
    reopened = IterationService(service.root, training.root, service.evaluations.root)
    assert reopened.get(identifier)["decision"] == "insufficient_evidence"
    assert reopened.list_iterations(session.session_id)[0]["iteration_id"] == identifier
    assert Path(parent["output_dir"]).is_dir()


def test_iteration_records_cannot_overwrite_concurrent_confirmation(tmp_path):
    service = IterationService(tmp_path / "iterations", tmp_path / "runs", tmp_path / "eval")
    record = service._save(
        {"iteration_id": "it-" + "a" * 32, "session_id": "b" * 32, "status": "proposed"}
    )
    service._save({**record, "status": "confirmed"}, record["revision"])
    with pytest.raises(ValueError, match="已更新"):
        service._save({**record, "status": "prepared"}, record["revision"])


def test_model_binding_requires_correct_base_and_adapter(tmp_path):
    from types import SimpleNamespace

    from src.workbench.business_evaluation import _model_identity

    for name in ("base", "wrong-base", "adapter"):
        path = tmp_path / name
        path.mkdir()
        (path / "pytorch_model.bin").write_bytes(name.encode())
    run = {"model_path": str(tmp_path / "base"), "output_dir": str(tmp_path / "adapter")}
    wrong = _model_identity(tmp_path / "wrong-base")
    report = SimpleNamespace(
        models=[
            {
                "identity": {
                    "base": wrong,
                    "adapter": _model_identity(tmp_path / "adapter"),
                }
            }
        ]
    )
    assert not IterationService._has_model(report, run)
    assert not IterationService._has_model(report, run, wrong)
    report.models[0]["identity"]["base"] = _model_identity(tmp_path / "base")
    assert IterationService._has_model(report, run)
    run["model_identity"] = {
        "path": run["model_path"],
        "config_and_tokenizer_hashes": {},
        "weights": [{"name": "pytorch_model.bin", "sha256": "different-trained-base"}],
    }
    assert not IterationService._has_model(report, run)


def test_parent_and_bound_evaluation_content_cannot_change_before_decision(tmp_path):
    from dataclasses import asdict

    from src.workbench.business_evaluation import EvaluationReport, comparison_key
    from src.workbench.sources import content_digest

    service = IterationService(tmp_path / "iterations", tmp_path / "runs", tmp_path / "eval")
    dataset, protocol = {"version": "fixture"}, {"scorer": "open_review"}
    report = EvaluationReport(
        "a" * 32,
        "2026-01-01",
        dataset,
        protocol,
        comparison_key(dataset, protocol),
        status="completed",
    )
    service.evaluations._save(report)
    record = service._save(
        {
            "iteration_id": "it-" + "b" * 32,
            "session_id": "c" * 32,
            "status": "evaluated",
            "parent_evaluation_id": report.evaluation_id,
            "evaluation_id": report.evaluation_id,
            "parent_report_digest": content_digest(asdict(report)),
            "evaluation_report_digest": content_digest(asdict(report)),
        }
    )
    assert service._parent_report(record) == report
    report.notes.append("the evidence file has changed")
    service.evaluations._save(report)
    with pytest.raises(ValueError, match="父轮评测证据"):
        service._parent_report(record)
    with pytest.raises(ValueError, match="对照证据"):
        service.decide(record["iteration_id"], "adopt", "不能接受已被替换的报告")


def test_data_scope_and_length_are_explicit_before_proposal(tmp_path):
    from types import SimpleNamespace

    service = IterationService(tmp_path / "iterations", tmp_path / "runs", tmp_path / "eval")
    arguments = dict(
        parent_run_id="unused",
        evaluation_id="unused",
        hypothesis="h",
        expected_outcome="e",
        changes="c",
    )
    with pytest.raises(ValueError, match="布尔"):
        service.propose(SimpleNamespace(dataset=object()), **arguments, data_change="false")
    with pytest.raises(ValueError, match="正整数"):
        service.propose(SimpleNamespace(dataset=object()), **arguments, max_length=True)


def test_incomplete_three_model_comparison_cannot_become_decision_evidence(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from src.workbench.business_evaluation import EvaluationReport, comparison_key

    service = IterationService(tmp_path / "iterations", tmp_path / "runs", tmp_path / "eval")
    dataset, protocol = {"version": "fixture"}, {"scorer": "open_review"}
    report = EvaluationReport(
        "a" * 32,
        "2026-01-01",
        dataset,
        protocol,
        comparison_key(dataset, protocol),
        status="release_failed",
    )
    service.evaluations._save(report)
    record = service._save(
        {
            "iteration_id": "it-" + "b" * 32,
            "session_id": "c" * 32,
            "goal": "目标",
            "status": "running",
            "new_run_id": "new",
        }
    )
    monkeypatch.setattr(service.training, "get_status", lambda run_id: {"status": "succeeded"})
    session = SimpleNamespace(session_id=record["session_id"], goal=record["goal"])
    with pytest.raises(ValueError, match="尚未完整"):
        service.bind_evaluation(record["iteration_id"], session, report.evaluation_id)
    assert service.get(record["iteration_id"])["status"] == "running"


def test_changed_data_must_be_confirmed_after_iteration_confirmation(environment, tmp_path):  # noqa: F811
    intake, original, _, _ = environment
    from src.workbench.evaluation_suites import EvalSuiteService
    from tests.unit.test_data_materialize import FULL

    reference = EvalSuiteService(tmp_path / "suites").freeze(original)
    service = IterationService(tmp_path / "iterations", tmp_path / "runs", tmp_path / "eval")
    session = intake.validate_full_data(
        original.session_id, original.revision, "revised.csv", FULL + b"new-c,new-o,new-input,yes\n"
    )
    session = intake.confirm_full_data(session.session_id, session.revision)
    session = intake.materialize_dataset(
        session.session_id, session.revision, evaluation_suite=reference
    )
    record = service._save(
        {
            "iteration_id": "it-" + "d" * 32,
            "session_id": session.session_id,
            "goal": session.goal,
            "status": "confirmed",
            "evaluation_suite": reference,
            "parent_dataset": original.dataset.model_dump(),
            "data_change": True,
            "confirmed_session_revision": session.revision,
        }
    )
    with pytest.raises(ValueError, match="本轮改进后"):
        service.prepare(record["iteration_id"], session)
    assert service.get(record["iteration_id"])["status"] == "confirmed"
