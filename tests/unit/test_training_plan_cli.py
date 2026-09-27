"""Training recommendation CLI requires explicit remote consent and preparation action."""

import json
import sys

import pytest

from scripts import data_intake
from src.workbench.intake_service import IntakeService
from tests.unit.test_data_materialize import _full


@pytest.fixture()
def plan_cli(tmp_path, monkeypatch):
    import src.agent.training as agent
    import src.workbench.training_plans as planning

    service = IntakeService(tmp_path / "intake")
    session = _full(service)
    session = service.materialize_dataset(session.session_id, session.revision)
    calls = []
    record = {
        "plan_id": "plan-fixture",
        "session_id": session.session_id,
        "status": "ready",
        "proposal": {"model_path": "/tmp/local-a"},
        "preflight": {"status": "passed", "splits": {"train": {"rows": 3}}},
    }

    class Plans:
        def __init__(self, root, training_root):
            pass

        def list_plans(self, session_id=None):
            assert session_id == session.session_id
            return [record]

        def get(self, plan_id):
            assert plan_id == record["plan_id"]
            return record

        def prepare(self, plan_id, current):
            calls.append(("prepare", plan_id, current.revision))
            return {"plan_id": plan_id, "run_id": "prepared-only"}

    def recommend(current, candidates, client, *, output_root, training_root):
        calls.append(("recommend", current.session_id, candidates, client.model))
        return record

    monkeypatch.setattr(planning, "TrainingPlanService", Plans)
    monkeypatch.setattr(agent, "recommend_training", recommend)
    monkeypatch.setenv("TUNESMITH_AGENT_PROVIDER", "compatible")
    monkeypatch.setenv("TUNESMITH_AGENT_BASE_URL", "https://fixture.example/v1")
    monkeypatch.setenv("TUNESMITH_AGENT_MODEL", "tool-fixture")
    monkeypatch.setenv("TUNESMITH_AGENT_API_KEY", "fixture-secret-do-not-print")

    def invoke(*args):
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "data_intake.py",
                "--store",
                str(service.root),
                "--plan-root",
                str(tmp_path / "plans"),
                *map(str, args),
            ],
        )
        return data_intake.main()

    return invoke, session, record, calls


def test_remote_recommendation_sends_candidates_only_after_explicit_consent(plan_cli, capsys):
    invoke, session, record, calls = plan_cli
    args = (
        "plan-recommend",
        session.session_id,
        "--revision",
        session.revision,
        "--model-path",
        "/tmp/local-a",
        "--model-path",
        "/tmp/local-b",
    )
    assert invoke(*args) == 2
    assert calls == []
    assert "需先允许" in capsys.readouterr().err
    assert invoke(*args, "--allow-remote-data") == 0
    output = capsys.readouterr()
    assert json.loads(output.out)["plan_id"] == record["plan_id"]
    assert calls == [
        ("recommend", session.session_id, ["/tmp/local-a", "/tmp/local-b"], "tool-fixture")
    ]
    assert "fixture-secret-do-not-print" not in output.out + output.err


def test_saved_plan_prepare_checks_revision_and_requires_separate_action(plan_cli, capsys):
    invoke, session, record, calls = plan_cli
    assert invoke("plan-list", session.session_id) == 0
    assert json.loads(capsys.readouterr().out)[0]["plan_id"] == record["plan_id"]
    assert invoke("plan-show", record["plan_id"]) == 0
    shown = capsys.readouterr()
    assert calls == []
    # 方案附带的预检证据翻译成大白话(stderr),与 JSON 原始记录并存
    assert "训练前检查通过" in shown.err
    assert (
        invoke(
            "plan-prepare",
            session.session_id,
            record["plan_id"],
            "--revision",
            session.revision + 1,
        )
        == 2
    )
    assert "任务已更新" in capsys.readouterr().err
    assert calls == []
    assert (
        invoke(
            "plan-prepare", session.session_id, record["plan_id"], "--revision", session.revision
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["run_id"] == "prepared-only"
    assert calls == [("prepare", record["plan_id"], session.revision)]


def test_model_list_and_implicit_recommendation_use_complete_discovered_candidates(
    plan_cli, monkeypatch, capsys
):
    import src.workbench.local_models as local_models

    invoke, session, _, calls = plan_cli
    candidates = [
        {"name": "Ready", "model_path": "/tmp/ready", "status": "available"},
        {"name": "Partial", "model_path": "/tmp/partial", "status": "incomplete"},
    ]
    monkeypatch.setattr(local_models, "discover_local_models", lambda roots=None: candidates)
    assert invoke("model-list") == 0
    assert json.loads(capsys.readouterr().out) == candidates
    assert calls == []
    assert (
        invoke(
            "plan-recommend",
            session.session_id,
            "--revision",
            session.revision,
            "--allow-remote-data",
        )
        == 0
    )
    assert calls[0][2] == ["/tmp/ready"]


def test_implicit_recommendation_explains_missing_local_models(plan_cli, monkeypatch, capsys):
    import src.workbench.local_models as local_models

    invoke, session, _, calls = plan_cli
    monkeypatch.setattr(local_models, "discover_local_models", lambda: [])
    assert (
        invoke(
            "plan-recommend",
            session.session_id,
            "--revision",
            session.revision,
            "--allow-remote-data",
        )
        == 2
    )
    assert "未发现完整本地模型" in capsys.readouterr().err
    assert calls == []
