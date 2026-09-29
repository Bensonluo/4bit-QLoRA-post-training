"""CLI authorization, stale-session rejection, and read-only execution status."""

import json
import sys

import pytest

from scripts import data_intake
from src.workbench.intake_service import IntakeService
from tests.unit.test_data_intake import CSV


@pytest.fixture
def execution_cli(tmp_path, monkeypatch):
    import src.workbench.iteration_execution as execution_module

    intake = IntakeService(tmp_path / "intake")
    session = intake.create("判断类别", "sample.csv", CSV)
    calls = []
    # R99:带上 session_id/session_revision——执行摘要据此插值可照抄的恢复命令,
    # paused 态 stderr 钉真 ID 端到端(与真实服务落盘记录同形状)。
    result = {
        "iteration_id": "it-" + "a" * 32,
        "status": "queued",
        "session_id": session.session_id,
        "session_revision": session.revision,
    }

    class Execution:
        def __init__(self, root, intake_root, iteration_root, training_root, evaluation_root):
            assert root == tmp_path / "iterations" / "executions"
            assert str(intake_root) == str(intake.root)

        def start(self, identity, current, **options):
            calls.append(("start", identity, current.revision, options))
            return result

        def get(self, identity):
            calls.append(("get", identity))
            return result

        def stop(self, identity):
            calls.append(("stop", identity))
            return {**result, "status": "stopped"}

    monkeypatch.setattr(execution_module, "IterationExecutionService", Execution)

    def invoke(*arguments):
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "data_intake.py",
                "--store",
                str(intake.root),
                "--iteration-root",
                str(tmp_path / "iterations"),
                *map(str, arguments),
            ],
        )
        return data_intake.main()

    return invoke, intake, session, result, calls


def test_execute_requires_current_revision_and_explicit_warning_ack(execution_cli, capsys):
    invoke, intake, session, result, calls = execution_cli
    args = ("iteration-execute", session.session_id, result["iteration_id"], "--revision")
    assert invoke(*args, session.revision) == 0
    assert calls[-1][3] == {
        "acknowledge_warnings": False,
        "independent_rows_confirmed": False,
    }
    capsys.readouterr()
    result["status"] = "awaiting_warning_ack"
    assert invoke(*args, session.revision, "--acknowledge-warnings") == 0
    assert calls[-1][3]["acknowledge_warnings"] is True
    capsys.readouterr()
    before = len(calls)
    intake.answer(session.session_id, "更正业务说明")
    assert invoke(*args, session.revision) == 2
    assert len(calls) == before
    assert "任务已更新" in capsys.readouterr().err


def test_status_and_stop_do_not_dispatch_start(execution_cli, capsys):
    invoke, _, _, result, calls = execution_cli
    assert invoke("iteration-execution-status", result["iteration_id"]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "queued"
    assert invoke("iteration-execution-stop", result["iteration_id"]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "stopped"
    assert [call[0] for call in calls] == ["get", "stop"]


def test_execution_commands_print_plain_language_summary_to_stderr(execution_cli, capsys):
    """自动执行子命令 stdout 仍是纯 JSON,stderr 追加人话:进行中/等确认/已停止各如实。"""
    invoke, _, session, result, _ = execution_cli

    assert invoke("iteration-execution-status", result["iteration_id"]) == 0
    queued = capsys.readouterr()
    assert json.loads(queued.out)["status"] == "queued"
    assert "等待后台执行开始" in queued.err
    assert "关闭页面不影响执行" in queued.err

    result["status"] = "awaiting_warning_ack"
    assert invoke("iteration-execution-status", result["iteration_id"]) == 0
    paused = capsys.readouterr()
    assert json.loads(paused.out)["status"] == "awaiting_warning_ack"
    assert "自动执行已暂停" in paused.err
    assert "勾选确认继续才会恢复" in paused.err
    # R99:恢复命令真 ID 插值端到端钉——「勾选」是页面词汇,CLI 用户拿到可照抄的命令
    assert (
        f"iteration-execute {session.session_id} {result['iteration_id']}"
        f" --revision {session.revision} --acknowledge-warnings" in paused.err
    )
    assert "SESSION_ID" not in paused.err

    assert invoke("iteration-execution-stop", result["iteration_id"]) == 0
    stopped = capsys.readouterr()
    assert json.loads(stopped.out)["status"] == "stopped"
    assert "已按请求停止自动执行" in stopped.err
    assert "不代表业务效果达标" in stopped.err
