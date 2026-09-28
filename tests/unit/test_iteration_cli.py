"""CLI keeps confirmed iteration identity through data, training and three-model evaluation."""

import json
import sys

import pytest

from scripts import data_intake
from src.workbench.business_evaluation import EvaluationReport
from src.workbench.evaluation_suites import EvalSuiteService
from src.workbench.intake_service import IntakeService
from tests.unit.test_data_materialize import _full


@pytest.fixture()
def iteration_cli(tmp_path, monkeypatch):
    import src.workbench.business_evaluation as evaluation_module
    import src.workbench.iterations as iteration_module
    import src.workbench.training_runs as training_module

    service = IntakeService(tmp_path / "intake")
    session = _full(service)
    session = service.materialize_dataset(session.session_id, session.revision)
    suite = EvalSuiteService(tmp_path / "suites").freeze(session)
    record = {
        "iteration_id": "it-" + "a" * 32,
        "session_id": session.session_id,
        "status": "confirmed",
        "evaluation_suite": suite,
        "parent_run_id": "parent",
        "new_run_id": "new",
    }
    calls = []

    class Iterations:
        def __init__(self, *args):
            pass

        def get(self, identity):
            assert identity == record["iteration_id"]
            return record

        def propose(self, current, **kwargs):
            calls.append(("propose", kwargs))
            return record

        def confirm(self, identity, current):
            calls.append(("confirm", identity))
            return record

        def prepare(self, identity, current):
            calls.append(("prepare", identity))
            return record

        def start(self, identity, current, **kwargs):
            calls.append(("start", kwargs))
            return record

        def bind_evaluation(self, identity, current, evaluation_id):
            calls.append(("bind", identity, evaluation_id))
            return record

        def decide(self, identity, decision, reason):
            calls.append(("decide", decision, reason))
            return record

        def list_iterations(self, session_id=None):
            return [record]

    class Training:
        def __init__(self, *args, **kwargs):
            pass

        def get_status(self, identity):
            current = service.load(session.session_id)
            return {
                "run_id": identity,
                "session_id": session.session_id,
                "status": "succeeded",
                "dataset_version": current.dataset.version,
                "model_path": f"/tmp/{identity}-base",
                "output_dir": f"/tmp/{identity}-adapter",
            }

    class Evaluation:
        def __init__(self, *args):
            pass

        def compare(self, current, models, protocol):
            calls.append(("compare", models))
            return EvaluationReport(
                "b" * 32, "now", {"version": current.dataset.version}, {}, "key"
            )

    monkeypatch.setattr(iteration_module, "IterationService", Iterations)
    monkeypatch.setattr(training_module, "TrainingRunService", Training)
    monkeypatch.setattr(evaluation_module, "BusinessEvaluationService", Evaluation)

    def invoke(*args):
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "data_intake.py",
                "--store",
                str(service.root),
                "--iteration-root",
                str(tmp_path / "iterations"),
                *map(str, args),
            ],
        )
        return data_intake.main()

    return invoke, service, session, record, calls


def test_proposal_and_confirmed_execution_dispatch_explicit_scope(iteration_cli, capsys):
    invoke, _, session, record, calls = iteration_cli
    assert (
        invoke(
            "iteration-propose",
            session.session_id,
            "--revision",
            session.revision,
            "--parent-run-id",
            "parent",
            "--evaluation-id",
            "b" * 32,
            "--hypothesis",
            "覆盖不足",
            "--expected-outcome",
            "错误减少",
            "--change",
            "补独立样例",
            "--change",
            "保持原题",
            "--data-change",
            "--epochs",
            2,
        )
        == 0
    )
    assert calls[-1][1]["changes"] == "补独立样例\n保持原题"
    assert calls[-1][1]["data_change"] is True
    assert calls[-1][1]["training_options"] == {"num_epochs": 2.0}
    assert json.loads(capsys.readouterr().out)["iteration_id"] == record["iteration_id"]
    for action in ("confirm", "prepare", "start"):
        options = ["--acknowledge-warnings"] if action == "start" else []
        assert (
            invoke(
                "iteration-" + action,
                session.session_id,
                record["iteration_id"],
                "--revision",
                session.revision,
                *options,
            )
            == 0
        )
        assert calls[-1][0] == action
        capsys.readouterr()
    assert calls[-1][1]["acknowledge_warnings"] is True
    assert (
        invoke(
            "iteration-decide",
            record["iteration_id"],
            "--decision",
            "insufficient_evidence",
            "--reason",
            "题数不足",
        )
        == 0
    )
    assert calls[-1] == ("decide", "insufficient_evidence", "题数不足")


def test_iteration_materialize_and_compare_lock_suite_parent_and_bind(iteration_cli, capsys):
    invoke, service, session, record, calls = iteration_cli
    assert (
        invoke(
            "materialize",
            session.session_id,
            "--revision",
            session.revision,
            "--iteration-id",
            record["iteration_id"],
        )
        == 0
    )
    capsys.readouterr()
    current = service.load(session.session_id)
    assert current.dataset.evaluation_suite == record["evaluation_suite"]
    assert (
        invoke(
            "eval-compare",
            session.session_id,
            "new",
            "--revision",
            current.revision,
            "--iteration-id",
            record["iteration_id"],
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["evaluation_id"] == "b" * 32
    models = calls[-2][1]
    assert len(models) == 3
    assert models[0].adapter_path is None
    assert models[1].base_model == "/tmp/parent-base"
    assert models[1].adapter_path == "/tmp/parent-adapter"
    assert models[2].adapter_path == "/tmp/new-adapter"
    assert calls[-1] == ("bind", record["iteration_id"], "b" * 32)


def test_iteration_rejects_stale_revision_and_unconfirmed_materialization(iteration_cli, capsys):
    invoke, _, session, record, calls = iteration_cli
    assert (
        invoke(
            "iteration-confirm",
            session.session_id,
            record["iteration_id"],
            "--revision",
            session.revision + 1,
        )
        == 2
    )
    assert "任务已更新" in capsys.readouterr().err
    assert calls == []
    record["status"] = "proposed"
    assert (
        invoke(
            "materialize",
            session.session_id,
            "--revision",
            session.revision,
            "--iteration-id",
            record["iteration_id"],
        )
        == 2
    )
    assert "请先确认" in capsys.readouterr().err


def test_iteration_revise_uses_confirmed_task_and_explicit_remote_consent(
    iteration_cli, monkeypatch, capsys
):
    import src.agent.revisions as revisions

    invoke, service, session, record, calls = iteration_cli
    for name, value in {
        "PROVIDER": "compatible",
        "BASE_URL": "https://fixture.example/v1",
        "MODEL": "tool-fixture",
        "API_KEY": "fixture-secret",
    }.items():
        monkeypatch.setenv("TUNESMITH_AGENT_" + name, value)

    def revise(intake, iterations, identity, client, *, expected_revision):
        calls.append(("revise", identity, expected_revision, client.model))
        return intake.load(session.session_id)

    monkeypatch.setattr(revisions, "revise_data_for_iteration", revise)
    args = (
        "iteration-revise",
        session.session_id,
        record["iteration_id"],
        "--revision",
        session.revision,
    )
    assert invoke(*args) == 2
    assert "需先允许" in capsys.readouterr().err
    assert calls == []
    assert invoke(*args, "--allow-remote-data") == 0
    assert calls == [("revise", record["iteration_id"], session.revision, "tool-fixture")]
    assert json.loads(capsys.readouterr().out)["revision"] == session.revision
    assert service.load(session.session_id).dataset.version == session.dataset.version
    record["session_id"] = "different-task"
    assert invoke(*args, "--allow-remote-data") == 2
    assert "任务不匹配" in capsys.readouterr().err
    assert len(calls) == 1


def test_iteration_compare_accepts_only_bidirectionally_linked_recovery_child(
    iteration_cli, monkeypatch, capsys
):
    import src.workbench.training_runs as training_module

    invoke, service, session, record, calls = iteration_cli
    assert (
        invoke(
            "materialize",
            session.session_id,
            "--revision",
            session.revision,
            "--iteration-id",
            record["iteration_id"],
        )
        == 0
    )
    capsys.readouterr()
    session = service.load(session.session_id)
    record["new_run_id"] = "original-new"
    original_status = training_module.TrainingRunService.get_status
    link = {"child_run_id": "retry-new"}

    def get_status(self, identity):
        run = original_status(self, identity)
        if identity == "original-new":
            run.update(status="failed", recover_technical_failures=True, recovery=link)
        if identity == "retry-new":
            run["recovery_parent_run_id"] = "original-new"
        return run

    monkeypatch.setattr(training_module.TrainingRunService, "get_status", get_status)
    args = (
        "eval-compare",
        session.session_id,
        "retry-new",
        "--revision",
        session.revision,
        "--iteration-id",
        record["iteration_id"],
    )
    assert invoke(*args) == 0
    capsys.readouterr()
    assert calls[-2][1][2].adapter_path == "/tmp/retry-new-adapter"
    assert calls[-1][0] == "bind"
    before = len(calls)
    link["child_run_id"] = "other-child"
    assert invoke(*args) == 2
    assert "不是本轮获准" in capsys.readouterr().err
    assert len(calls) == before


def test_iteration_confirm_evaluated_prints_delta_summary_to_stderr_only(iteration_cli, capsys):
    invoke, _, session, record, _ = iteration_cli
    # 按该文件既有构造方式，把 record 就地改成 evaluated + 三模型结果形态。
    record.update(
        status="evaluated",
        results=[
            {"label": "基座", "metrics": {"total": 2, "exact_match": 0.0}},
            {"label": "父轮模型", "metrics": {"total": 2, "exact_match": 0.0}},
            {"label": "本轮微调", "metrics": {"total": 2, "exact_match": 0.5}},
        ],
    )
    assert (
        invoke(
            "iteration-confirm",
            session.session_id,
            record["iteration_id"],
            "--revision",
            session.revision,
        )
        == 0
    )
    captured = capsys.readouterr()
    # 三模型题数差人话只进 stderr，与页面决策卡同源同措辞。
    assert "父轮对照：本轮微调比父轮多答对 1 题（1/2 vs 0/2）。" in captured.err
    assert "每题约占 50 个百分点" in captured.err
    # stdout 保持纯 JSON 供脚本消费，不夹带中文人话行。
    assert json.loads(captured.out)["iteration_id"] == record["iteration_id"]
    assert "父轮对照" not in captured.out
    assert "每题约占" not in captured.out
