"""Final acceptance CLI freezes explicit criteria and never invokes development Agent analysis."""

import json
import sys
from dataclasses import asdict

import pytest

from scripts import data_intake
from src.workbench.intake_service import IntakeService
from tests.unit.test_data_materialize import _full


@pytest.fixture()
def acceptance_cli(tmp_path, monkeypatch):
    import src.workbench.acceptance as acceptance_module
    import src.workbench.training_runs as training_module

    intake = IntakeService(tmp_path / "intake")
    session = _full(intake)
    session = intake.materialize_dataset(session.session_id, session.revision)
    calls = []
    record = {
        "acceptance_id": "acceptance-fixture",
        "session_id": session.session_id,
        "status": "prepared",
        "protocol": {},
        "criteria": {
            "metric": "exact_match",
            "minimum_score": 0.9,
            "minimum_cases": 20,
            "business_standard": "分类必须严格正确",
        },
        "result": {"decision": "pending_run"},
    }

    class Acceptance:
        def __init__(self, *args):
            pass

        def prepare(self, current, model, protocol, criteria, task_spec=None):
            calls.append(("prepare", current.revision, model, protocol, criteria, task_spec))
            record.update(
                model=asdict(model),
                protocol=asdict(protocol),
                criteria=criteria,
                task_spec=task_spec,
            )
            return record

        def run(self, identity, current):
            calls.append(("run", identity, current.revision))
            record.update(status="completed", result={"decision": "insufficient_evidence"})
            return record

        def get(self, identity):
            return record

        def list_acceptances(self, session_id=None):
            return [record]

        def review(self, identity, decisions):
            calls.append(("review", identity, decisions))
            return record

    class Training:
        def __init__(self, *args, **kwargs):
            pass

        def get_status(self, identity):
            return {
                "run_id": identity,
                "session_id": session.session_id,
                "status": "succeeded",
                "dataset_version": session.dataset.version,
                "model_path": "/tmp/base",
                "output_dir": "/tmp/adapter",
            }

    monkeypatch.setattr(acceptance_module, "AcceptanceService", Acceptance)
    monkeypatch.setattr(training_module, "TrainingRunService", Training)

    def invoke(*args):
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "data_intake.py",
                "--store",
                str(intake.root),
                "--scoring-root",
                str(tmp_path / "scoring"),
                "--acceptance-root",
                str(tmp_path / "acceptance"),
                "--evaluation-root",
                str(tmp_path / "evaluations"),
                "--iteration-root",
                str(tmp_path / "iterations"),
                "--training-root",
                str(tmp_path / "training"),
                *map(str, args),
            ],
        )
        return data_intake.main()

    return invoke, session, record, calls


def test_prepare_freezes_user_criteria_before_explicit_single_model_run(acceptance_cli, capsys):
    invoke, session, record, calls = acceptance_cli
    assert (
        invoke(
            "acceptance-prepare",
            session.session_id,
            "run-fixture",
            "--revision",
            session.revision,
            "--business-standard",
            "分类必须严格正确",
            "--minimum-score",
            0.9,
            "--minimum-cases",
            20,
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["status"] == "prepared"
    assert len(calls) == 1
    assert calls[0][2].adapter_path == "/tmp/adapter"
    assert calls[0][4] == {
        "metric": "exact_match",
        "minimum_score": 0.9,
        "minimum_cases": 20,
        "business_standard": "分类必须严格正确",
    }
    assert invoke("acceptance-show", record["acceptance_id"]) == 0
    capsys.readouterr()
    assert len(calls) == 1
    assert (
        invoke(
            "acceptance-run",
            session.session_id,
            record["acceptance_id"],
            "--revision",
            session.revision,
        )
        == 0
    )
    assert calls[-1][0] == "run"
    assert json.loads(capsys.readouterr().out)["result"]["decision"] == "insufficient_evidence"


def test_review_checks_task_revision_and_records_one_explicit_business_judgment(
    acceptance_cli, capsys
):
    invoke, session, record, calls = acceptance_cli
    args = (
        "acceptance-review",
        session.session_id,
        record["acceptance_id"],
        "--revision",
        session.revision,
        "--row-index",
        0,
        "--decision",
        "rejected",
        "--reason",
        "缺少关键处理步骤",
    )
    assert invoke(*args) == 0
    assert calls[-1][2] == [{"index": 0, "decision": "rejected", "reason": "缺少关键处理步骤"}]
    capsys.readouterr()
    assert (
        invoke(
            "acceptance-run",
            session.session_id,
            record["acceptance_id"],
            "--revision",
            session.revision + 1,
        )
        == 2
    )
    assert "任务已更新" in capsys.readouterr().err
    record["session_id"] = "another-task"
    assert invoke(*args) == 2
    assert "业务任务不匹配" in capsys.readouterr().err
    assert len(calls) == 1


def test_acceptance_commands_print_plain_language_summary_to_stderr(acceptance_cli, capsys):
    """验收子命令 stdout 仍是纯 JSON,stderr 追加人话:冻结条款、结论态与「不自动部署」边界。"""
    invoke, session, record, calls = acceptance_cli
    assert (
        invoke(
            "acceptance-prepare",
            session.session_id,
            "run-fixture",
            "--revision",
            session.revision,
            "--business-standard",
            "分类必须严格正确",
            "--minimum-score",
            0.9,
            "--minimum-cases",
            20,
        )
        == 0
    )
    out = capsys.readouterr()
    assert json.loads(out.out)["status"] == "prepared"
    assert "这次最终验收针对模型「待验收模型」" in out.err
    assert "条款已冻结、验收尚未执行" in out.err
    assert "也不能再伪装成首次盲测" in out.err

    assert invoke("acceptance-show", record["acceptance_id"]) == 0
    shown = capsys.readouterr()
    assert json.loads(shown.out)["acceptance_id"] == "acceptance-fixture"
    assert "条款已冻结、验收尚未执行" in shown.err

    assert (
        invoke(
            "acceptance-run",
            session.session_id,
            record["acceptance_id"],
            "--revision",
            session.revision,
        )
        == 0
    )
    run_out = capsys.readouterr()
    assert json.loads(run_out.out)["result"]["decision"] == "insufficient_evidence"
    assert "当前结论：证据不足，不能确认可交付。" in run_out.err
    assert "不会自动部署模型" in run_out.err


def test_prepare_prints_task_spec_before_freeze_and_show_cites_frozen_spec(acceptance_cli, capsys):
    """acceptance-prepare 规约人话先于冻结动作进 stderr;冻结记录携带四要素快照,
    acceptance-show 摘要引用「冻结时引用的任务规约口径」与规约卡同源同词汇。"""
    invoke, session, record, calls = acceptance_cli
    assert (
        invoke(
            "acceptance-prepare",
            session.session_id,
            "run-fixture",
            "--revision",
            session.revision,
            "--business-standard",
            "分类必须严格正确",
            "--minimum-score",
            0.9,
            "--minimum-cases",
            20,
        )
        == 0
    )
    captured = capsys.readouterr()
    assert json.loads(captured.out)["status"] == "prepared"
    assert "的任务规约：由既有确认记录只读汇编" in captured.err
    assert "业务目标：根据业务文本分类" in captured.err
    # 冻结调用带上四要素快照(goal/answer_semantics/scoring/temporal_split 子集语义
    # 由真实服务保证;这里核对 CLI 传的是投影原样、口径与 stderr 同一来源)。
    assert calls[0][5]["goal"]["goal"] == "根据业务文本分类"
    assert invoke("acceptance-show", record["acceptance_id"]) == 0
    shown = capsys.readouterr()
    assert "冻结时引用的任务规约口径：" in shown.err
    assert "业务目标：根据业务文本分类" in shown.err


def test_acceptance_show_prints_gate_arithmetic_to_stderr_and_keeps_stdout_json(
    acceptance_cli, capsys
):
    """acceptance-show 的 stderr 追加门槛分辨率算术(容错 0 道分支),stdout 仍是纯 JSON。"""
    invoke, session, record, calls = acceptance_cli
    record["criteria"]["minimum_cases"] = 5
    record["evaluation_suite"] = {
        "suite_id": "fixed-suite",
        "case_counts": {"validation": 4, "test": 5},
    }
    assert invoke("acceptance-show", record["acceptance_id"]) == 0
    shown = capsys.readouterr()
    assert json.loads(shown.out)["acceptance_id"] == "acceptance-fixture"
    assert "按 90% 通过率门槛与 5 道最终测试题算：需通过 5 道、最多容错 0 道未通过" in shown.err
    assert "容错为 0 道" in shown.err
    assert "按 90% 通过率门槛" not in shown.out
    assert "容错为 0 道" not in shown.out
    assert calls == []
