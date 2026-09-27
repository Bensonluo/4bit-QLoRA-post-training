"""CLI compares the chosen successful run on the bound dataset through the shared service."""

import json
import sys

import pytest

from scripts import data_intake
from src.workbench.business_evaluation import EvaluationReport
from src.workbench.intake_service import IntakeService
from tests.unit.test_full_data import FULL, approved


@pytest.fixture()
def evaluation_cli(tmp_path, monkeypatch):
    import src.workbench.business_evaluation
    import src.workbench.training_runs

    service = IntakeService(tmp_path / "intake")
    session = approved(service)
    session = service.validate_full_data(session.session_id, session.revision, "full.csv", FULL)
    session = service.confirm_full_data(session.session_id, session.revision)
    session = service.materialize_dataset(session.session_id, session.revision)
    run = {
        "run_id": "fixture-run",
        "session_id": session.session_id,
        "status": "succeeded",
        "dataset_version": session.dataset.version,
        "model_path": "/tmp/base",
        "output_dir": "/tmp/adapter",
    }
    calls = []
    report = EvaluationReport(
        "a" * 32, "now", {"version": session.dataset.version}, {}, "key", status="completed"
    )

    class Training:
        def __init__(self, *args, **kwargs):
            pass

        def get_status(self, run_id):
            return run

    class Evaluation:
        def __init__(self, *args, **kwargs):
            pass

        def compare(self, current, models, protocol):
            calls.append((current.session_id, models, protocol))
            return report

        def get_report(self, evaluation_id):
            assert evaluation_id == report.evaluation_id
            return report

    monkeypatch.setattr(src.workbench.training_runs, "TrainingRunService", Training)
    monkeypatch.setattr(src.workbench.business_evaluation, "BusinessEvaluationService", Evaluation)

    def invoke(*args):
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "data_intake.py",
                "--store",
                str(service.root),
                "--evaluation-root",
                str(tmp_path / "eval"),
                *map(str, args),
            ],
        )
        return data_intake.main()

    return invoke, session, run, calls, report


def test_eval_compare_derives_base_adapter_and_categorical_protocol(evaluation_cli, capsys):
    invoke, session, _, calls, report = evaluation_cli
    assert (
        invoke(
            "eval-compare",
            session.session_id,
            "fixture-run",
            "--revision",
            session.revision,
            "--max-new-tokens",
            32,
            "--keep-whitespace",
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["evaluation_id"] == report.evaluation_id
    _, models, protocol = calls[0]
    assert models[0].base_model == models[1].base_model == "/tmp/base"
    assert models[0].adapter_path is None and models[1].adapter_path == "/tmp/adapter"
    assert protocol.scorer == "classification_exact"
    assert protocol.max_new_tokens == 32 and not protocol.strip_whitespace
    assert invoke("eval-show", report.evaluation_id) == 0
    output = capsys.readouterr()
    assert json.loads(output.out)["status"] == "completed"
    # 对照报告附带大白话解读(空报告如实说明没有模型结果)
    assert "该报告没有模型结果。" in output.err


@pytest.mark.parametrize(
    "change",
    [{"dataset_version": "different"}, {"status": "running"}, {"session_id": "other-session"}],
)
def test_eval_compare_rejects_unrelated_or_unfinished_run(evaluation_cli, change, capsys):
    invoke, session, run, calls, _ = evaluation_cli
    run.update(change)
    assert (
        invoke("eval-compare", session.session_id, "fixture-run", "--revision", session.revision)
        == 2
    )
    assert calls == []
    assert "成功训练" in capsys.readouterr().err


def test_eval_analyze_uses_byok_and_requires_remote_data_authorization(
    evaluation_cli, monkeypatch, capsys
):
    import src.agent.evaluation

    invoke, session, _, _, report = evaluation_cli
    monkeypatch.setenv("TUNESMITH_AGENT_PROVIDER", "compatible")
    monkeypatch.setenv("TUNESMITH_AGENT_BASE_URL", "https://fixture.example/v1")
    monkeypatch.setenv("TUNESMITH_AGENT_MODEL", "tool-fixture")
    monkeypatch.setenv("TUNESMITH_AGENT_API_KEY", "fixture-only-secret")
    requests = []

    def assess(current_report, current_session, client, *, output_root):
        requests.append(
            (current_report.evaluation_id, current_session.session_id, client.model, output_root)
        )
        return {
            "evaluation_id": current_report.evaluation_id,
            "model": client.model,
            "assessment": {
                "summary": "核查真实坏例后的建议",
                "observations": [{"statement": "坏例集中在日期字段", "evidence_ids": ["case:1"]}],
                "hypotheses": [],
                "next_steps": ["核对日期字段监督覆盖"],
                "decision": "inspect_data",
                "limitations": ["样本量小"],
                "business_questions": [],
            },
            "tool_trace": [{"tool": "inspect_evaluation_summary", "ok": True}],
        }

    monkeypatch.setattr(src.agent.evaluation, "assess_evaluation", assess)
    args = (
        "eval-analyze",
        session.session_id,
        report.evaluation_id,
        "--revision",
        session.revision,
    )
    assert invoke(*args) == 2
    assert requests == []
    assert "需先允许" in capsys.readouterr().err
    assert invoke(*args, "--allow-remote-data") == 0
    output = capsys.readouterr()
    assert json.loads(output.out)["evaluation_id"] == report.evaluation_id
    assert requests[0][:3] == (report.evaluation_id, session.session_id, "tool-fixture")
    # 与其他业务子命令同口径：stdout 纯 JSON，stderr 追加解读人话摘要。
    assert "这份解读由 Agent 在核查真实工具证据后给出（模型 tool-fixture）。" in output.err
    assert "建议优先处理：先核查数据。" in output.err
    assert "软件不会据此自动改标签" in output.err
    assert "不代表业务效果达标" in output.err
    assert "fixture-only-secret" not in output.out + output.err


def test_final_test_report_cannot_be_sent_to_agent(evaluation_cli, monkeypatch, capsys):
    import src.agent.evaluation

    invoke, session, _, _, report = evaluation_cli
    report.dataset.update(purpose="final_acceptance", split="test")
    monkeypatch.setattr(
        src.agent.evaluation,
        "assess_evaluation",
        lambda *args, **kwargs: pytest.fail("final test evidence must not reach Agent"),
    )
    assert (
        invoke(
            "eval-analyze",
            session.session_id,
            report.evaluation_id,
            "--revision",
            session.revision,
            "--allow-remote-data",
        )
        == 2
    )
    assert "最终独立测试题与坏例不能发送" in capsys.readouterr().err
