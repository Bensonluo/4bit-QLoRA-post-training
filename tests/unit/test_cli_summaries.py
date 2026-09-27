"""CLI 分派路径的语言化摘要:JSON 记录走 stdout,人话句子走 stderr。

e1291ef 已把 summarize_* 接进 eval-compare / train-* 子命令;这里用 mocked
runtime 走真实分派路径,钉住「用户在终端实际读到什么」:真实对照报告不再
只有空报告占位句,附带预检记录时第二层摘要也必须出现、没带就不编造。
"""

import json
import sys

import pytest

from scripts import data_intake
from src.workbench.business_evaluation import EvaluationReport
from src.workbench.intake_service import IntakeService
from tests.unit.test_full_data import FULL, approved


def _model_result(label, total, accuracy):
    """一份模型对照结果:metrics 决定答对数,rows 全部正常作答(无截断/失败/复述)。"""
    rows = [
        {"status": "correct", "output": "补发", "prompt": "题目:客户首次描述"} for _ in range(total)
    ]
    return {"label": label, "metrics": {"total": total, "exact_match": accuracy}, "rows": rows}


@pytest.fixture()
def eval_compare_cli(tmp_path, monkeypatch):
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
    report = EvaluationReport(
        "b" * 32,
        "now",
        {"version": session.dataset.version},
        {"max_new_tokens": 256, "strip_whitespace": True},
        "key",
        status="completed",
        models=[_model_result("基座", 10, 0.3), _model_result("本轮微调", 10, 0.8)],
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
                "--training-root",
                str(tmp_path / "training"),
                "--evaluation-root",
                str(tmp_path / "eval"),
                *map(str, args),
            ],
        )
        return data_intake.main()

    return invoke, session, report


def test_eval_compare_stderr_carries_comparison_sentences(eval_compare_cli, capsys):
    """eval-compare 在 JSON 之外用 stderr 输出对照大白话;stdout 保持纯 JSON。"""
    invoke, session, report = eval_compare_cli
    assert (
        invoke("eval-compare", session.session_id, "fixture-run", "--revision", session.revision)
        == 0
    )
    captured = capsys.readouterr()
    assert json.loads(captured.out)["evaluation_id"] == report.evaluation_id
    err = captured.err
    assert "这次对照在固定开发集的 10 道题上进行" in err
    assert "基座答对 3/10" in err
    assert "本轮微调答对 8/10" in err
    assert "答对最多的是本轮微调(8/10)" in err
    assert "本轮微调比基座答对更多(8/10 vs 3/10)" in err
    assert "但要注意样本量" in err
    # 10 道题仍触发小样本提示:百分比受单题影响,只当方向参考
    assert "任何百分比都受单题影响很大" in err
    assert "以上是观察事实,不是业务达标结论" in err


@pytest.fixture()
def train_status_cli(tmp_path, monkeypatch):
    import src.workbench.training_runs

    IntakeService(tmp_path / "intake")  # train-status 不读任务,--store 只需可用
    record = {
        "run_id": "run-9",
        "status": "running",
        "model_path": "/models/Qwen3-1.7B",
    }

    class Training:
        def __init__(self, root, project_root=None):
            pass

        def get_status(self, run_id):
            return record

    monkeypatch.setattr(src.workbench.training_runs, "TrainingRunService", Training)

    def invoke(*args):
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "data_intake.py",
                "--store",
                str(tmp_path / "intake"),
                "--training-root",
                str(tmp_path / "training"),
                *map(str, args),
            ],
        )
        return data_intake.main()

    return invoke, record


def test_train_status_appends_preflight_summary_when_record_carries_one(train_status_cli, capsys):
    """记录带预检时,stderr 在训练摘要之外追加预检大白话(第二层,各说各的边界)。"""
    invoke, record = train_status_cli
    record.update(
        {
            "status": "succeeded",
            "config": {"training": {"num_epochs": 2}},
            "metrics": {"train_loss": 0.42},
            "preflight": {
                "status": "passed",
                "max_length": 512,
                "splits": {"train": {"rows": 6, "truncated_rows": 0, "answer_lost_rows": 0}},
                "issues": [],
            },
        }
    )
    assert invoke("train-status", "run-9") == 0
    captured = capsys.readouterr()
    assert json.loads(captured.out)["run_id"] == "run-9"
    err = captured.err
    assert "这次训练基于 Qwen3-1.7B" in err
    assert "训练完成，产出了微调适配器。" in err
    assert "0.4200" in err and "不代表业务效果" in err
    assert "要用同一套开发题与基座对照" in err
    assert "训练前检查通过：6 行数据" in err
    assert "没有内容因长度超限被截断" in err
    assert "不代表训练效果或业务达标" in err


def test_train_status_without_preflight_does_not_invent_one(train_status_cli, capsys):
    """记录没带预检就只给训练状态句,不编造预检结论。"""
    invoke, record = train_status_cli
    assert invoke("train-status", "run-9") == 0
    captured = capsys.readouterr()
    assert json.loads(captured.out)["status"] == "running"
    err = captured.err
    assert "这次训练基于 Qwen3-1.7B" in err
    assert "正在训练中" in err and "关闭页面不影响后台训练" in err
    assert "训练前检查" not in err


def test_materialize_stderr_carries_dataset_summary(tmp_path, monkeypatch, capsys):
    """materialize stderr 追加分区人话摘要:分组路径此前零人话,现在与页面同口径;
    stdout 仍是纯 JSON 任务记录。"""
    service = IntakeService(tmp_path / "intake")
    session = approved(service)
    session = service.validate_full_data(session.session_id, session.revision, "full.csv", FULL)
    session = service.confirm_full_data(session.session_id, session.revision)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "data_intake.py",
            "--store",
            str(service.root),
            "materialize",
            session.session_id,
            "--revision",
            str(session.revision),
            "--validation-fraction",
            "0.2",
            "--test-fraction",
            "0.2",
            "--seed",
            "17",
        ],
    )
    assert data_intake.main() == 0
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["dataset"]["statistics"]["row_counts"] == {
        "train": 1,
        "validation": 1,
        "test": 1,
    }
    err = captured.err
    # 分法句 + 计数 + 分组数 + 比例受分组大小影响
    assert "按业务对象隔离划分" in err
    assert "训练 1 条、验证 1 条、独立测试 1 条" in err
    assert "个独立分组" in err
    assert "实际比例受分组大小影响" in err
    # FULL 3 行 2 类、1/1/1 切分:训练集只见 1 类,覆盖披露必触发
    assert "从未出现在训练集" in err
    # 固定边界句收尾
    assert "分区就绪只说明" in err
