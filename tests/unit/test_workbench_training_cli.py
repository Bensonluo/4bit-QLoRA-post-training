"""CLI training controls use saved service records without reconstructing YAML."""

import json
import sys

import pytest

from scripts import data_intake
from src.workbench.intake_service import IntakeService
from tests.unit.test_data_intake import CSV, analysis

_FULL_CSV = (
    "编号,客户描述,类别,处理结果\n" + "".join(f"{i:03d},描述{i},质量,补发\n" for i in range(1, 11))
).encode()


@pytest.fixture()
def training_cli(tmp_path, monkeypatch):
    import src.workbench.training_runs

    service = IntakeService(tmp_path / "intake")
    session = service.create("分类", "sample.csv", CSV)
    # 生产形状的已确认数据集：train-prepare 的学习率分档建议行按 dataset.statistics
    # .row_counts 如实读取（真实服务流产出，不造桩字段）。
    session = service.apply_analysis(session, analysis())
    session = service.confirm(session.session_id, session.revision)
    session = service.validate_full_data(
        session.session_id, session.revision, "full.csv", _FULL_CSV
    )
    session = service.confirm_full_data(session.session_id, session.revision)
    session = service.materialize_dataset(session.session_id, session.revision)
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
    captured = capsys.readouterr()
    result = json.loads(captured.out)
    # 手工参数大白话与学习率分档建议先进 stderr（training_guidance 单一来源，
    # 与页面高级配置同词汇）；stdout 仍是纯 JSON。
    assert "参数大白话：训练轮数＝" in captured.err
    assert "推荐起步值（小数据）：" in captured.err
    assert "学习率分档建议：" in captured.err
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


def test_train_start_prints_task_spec_before_run_summary(training_cli, capsys):
    """train-start 先把任务规约人话打到 stderr(启动前对齐),再走运行摘要;stdout 仍是纯 JSON。"""
    run, session, _ = training_cli
    assert (
        run(
            "train-prepare",
            session.session_id,
            "--revision",
            session.revision,
            "--model-path",
            "/tmp/base",
        )
        == 0
    )
    capsys.readouterr()
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
    captured = capsys.readouterr()
    assert json.loads(captured.out)["status"] == "running"
    assert "的任务规约：由既有确认记录只读汇编" in captured.err
    assert "不代表模型效果达标" in captured.err
    # 规约在前、运行摘要在后:同一条 stderr 流内的顺序契约。
    assert captured.err.index("的任务规约") < captured.err.index("正在训练中")


def test_train_start_run_summary_includes_plan_trace_line(training_cli, capsys, monkeypatch):
    """train-start 运行摘要尾行:run 记录带 plan_trace 时渲染统一轨迹行(单一来源)。"""
    run, session, _ = training_cli
    import src.workbench.training_runs

    # training_cli 夹具已把 TrainingRunService 换成桩类;这里在其上派生,仅让 start
    # 的返回记录多带 plan_trace(1 成功 1 失败),其余夹具行为不动。
    stub_base = src.workbench.training_runs.TrainingRunService

    class TracedStartStub(stub_base):
        def start(self, run_id, current, **kwargs):
            record = super().start(run_id, current, **kwargs)
            return {
                **record,
                "plan_trace": [
                    {"tool": "discover_local_models", "ok": True},
                    {"tool": "probe_model", "ok": False, "error": "缺 tokenizer 文件"},
                ],
            }

    monkeypatch.setattr(src.workbench.training_runs, "TrainingRunService", TracedStartStub)
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
    err = capsys.readouterr().err
    assert "工具核查轨迹：2 次调用，成功 1 次、失败 1 次" in err
    assert "训练方案只依赖成功的调用" in err
