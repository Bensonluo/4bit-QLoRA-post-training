"""CLI training controls use saved service records without reconstructing YAML."""

import json
import sys

import pytest

from scripts import data_intake
from src.workbench.intake_service import IntakeService
from tests.unit.test_data_intake import CSV


@pytest.fixture()
def training_cli(tmp_path, monkeypatch):
    import src.workbench.training_runs

    service = IntakeService(tmp_path / "intake")
    session = service.create("分类", "sample.csv", CSV)
    calls = []

    class TrainingFixture:
        def __init__(self, root, project_root=None):
            self.root = root

        def prepare(self, current, model_path, **kwargs):
            calls.append(("prepare", current.session_id, model_path, kwargs))
            return {"run_id": "run-1", "status": "prepared", "config": kwargs}

        def start(self, run_id, current, **kwargs):
            calls.append(("start", run_id, current.session_id, kwargs))
            return {"run_id": run_id, "status": "running"}

        def get_status(self, run_id):
            return {
                "run_id": run_id,
                "status": "succeeded",
                "artifacts": {"adapter_weights": "/tmp/adapter.safetensors"},
            }

        def read_logs(self, run_id, tail=100):
            calls.append(("logs", run_id, tail))
            return "last training log"

        def stop(self, run_id):
            calls.append(("stop", run_id))
            return {"run_id": run_id, "status": "stopped"}

        def list_runs(self, session_id=None):
            return [{"run_id": "run-1", "session_id": session_id}]

    monkeypatch.setattr(src.workbench.training_runs, "TrainingRunService", TrainingFixture)

    def run(*args):
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "data_intake.py",
                "--store",
                str(service.root),
                "--training-root",
                str(tmp_path / "training"),
                *map(str, args),
            ],
        )
        return data_intake.main()

    return run, session, calls


def test_training_prepare_and_start_preserve_bound_session_and_options(training_cli, capsys):
    run, session, calls = training_cli
    assert (
        run(
            "train-prepare",
            session.session_id,
            "--revision",
            session.revision,
            "--model-path",
            "/tmp/base",
            "--epochs",
            2,
            "--batch-size",
            3,
            "--lora-rank",
            16,
            "--load-in-4bit",
        )
        == 0
    )
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "prepared"
    options = calls[-1][3]
    assert options["training_options"]["num_epochs"] == 2
    assert options["training_options"]["batch_size"] == 3
    assert options["lora_options"] == {"r": 16, "lora_alpha": 32}
    assert options["model_options"] == {"quantization_bits": 4}
    assert (
        run(
            "train-start",
            session.session_id,
            "run-1",
            "--revision",
            session.revision,
            "--acknowledge-warnings",
        )
        == 0
    )
    assert calls[-1] == ("start", "run-1", session.session_id, {"acknowledge_warnings": True})


def test_stale_revision_does_not_launch_or_prepare(training_cli, capsys):
    run, session, calls = training_cli
    assert run("train-start", session.session_id, "run-1", "--revision", session.revision + 1) == 2
    assert calls == []
    assert "任务已更新" in capsys.readouterr().err


def test_status_logs_stop_and_list_share_record_interface(training_cli, capsys):
    run, session, calls = training_cli
    assert run("train-status", "run-1") == 0
    assert (
        json.loads(capsys.readouterr().out)["artifacts"]["adapter_weights"]
        == "/tmp/adapter.safetensors"
    )
    assert run("train-logs", "run-1", "--tail", 5) == 0
    assert "last training log" in capsys.readouterr().out
    assert calls[-1] == ("logs", "run-1", 5)
    assert run("train-stop", "run-1") == 0
    assert json.loads(capsys.readouterr().out)["status"] == "stopped"
    assert run("train-list", session.session_id) == 0
    listed = capsys.readouterr()
    assert json.loads(listed.out)[0]["session_id"] == session.session_id
    # 清单尾行:计数一行(summarize_listing 单一来源),不逐条灌训练人话。
    assert "共 1 条已保存的训练版本。" in listed.err


def test_train_status_prints_plain_language_summary(training_cli, capsys):
    """train-status 在 JSON 之外输出大白话状态(观察事实,不是业务结论)。"""
    run, _, _ = training_cli
    assert run("train-status", "run-1") == 0
    err = capsys.readouterr().err
    assert "训练完成" in err
    assert "对照" in err


def test_train_start_forwards_explicit_technical_recovery_authorization(training_cli, capsys):
    run, session, calls = training_cli
    assert (
        run(
            "train-start",
            session.session_id,
            "run-1",
            "--revision",
            session.revision,
            "--recover-technical-failures",
        )
        == 0
    )
    assert calls[-1][3]["recover_technical_failures"] is True
    assert json.loads(capsys.readouterr().out)["status"] == "running"
