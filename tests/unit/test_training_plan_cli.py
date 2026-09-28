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
            return {
                "plan_id": plan_id,
                "run_id": "prepared-only",
                "training_run": {
                    "run_id": "prepared-only",
                    "status": "prepared",
                    "model_path": "/tmp/local-a",
                    "config": {"training": {"num_epochs": 2}},
                },
            }

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


def test_plan_show_and_prepare_stderr_carry_plain_language_summaries(plan_cli, capsys):
    """plan 子命令 stderr 分层人话:方案本身一层,预检证据一层,准备结果复用
    训练记录摘要;plan-list 只追加一行清单尾行(计数)。stdout 均保持纯 JSON。"""
    invoke, session, record, _ = plan_cli
    record["proposal"] = {
        "model_path": "/tmp/local-a",
        "max_length": 1024,
        "rationale": ["样例结构稳定"],
        "limitations": ["未在业务留出题上验证"],
        "business_questions": [],
        "training_options": {"num_epochs": 2, "batch_size": 1, "learning_rate": 0.0002},
        "lora_options": {"r": 16},
        "model_options": {"quantization_bits": 4},
    }
    assert invoke("plan-show", record["plan_id"]) == 0
    shown = capsys.readouterr()
    assert json.loads(shown.out)["plan_id"] == record["plan_id"]
    err = shown.err
    # 第一层:方案本身(模型+参数+状态+理由+限制+确认边界)
    assert "这份方案建议用 local-a" in err
    assert "最大长度 1024" in err and "训练 2 轮" in err and "batch size 1" in err
    assert "学习率 0.0002" in err and "LoRA rank 16" in err and "4-bit 量化" in err
    assert "当前状态：方案可供确认" in err
    assert "推荐理由：样例结构稳定" in err
    assert "尚未验证的限制：未在业务留出题上验证" in err
    assert "不会自动启动" in err
    # 第二层:方案附带的预检证据(顶层 preflight 兼容形状)
    assert "训练前检查通过" in err
    # plan-list 只追加一行清单尾行(计数),不把逐条方案人话灌进 stderr
    assert invoke("plan-list", session.session_id) == 0
    listed = capsys.readouterr()
    assert "这份方案建议用" not in listed.err
    assert "共 1 条已保存的训练方案。" in listed.err
    # plan-prepare 结果复用训练记录摘要:准备好但未启动
    assert (
        invoke(
            "plan-prepare", session.session_id, record["plan_id"], "--revision", session.revision
        )
        == 0
    )
    prepared = capsys.readouterr()
    assert json.loads(prepared.out)["run_id"] == "prepared-only"
    assert "这次训练基于 local-a" in prepared.err
    assert "方案已准备好并通过检查，还没有开始训练" in prepared.err


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
