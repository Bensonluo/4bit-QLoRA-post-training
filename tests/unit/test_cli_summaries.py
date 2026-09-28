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


def test_train_status_appends_loss_trend_when_metrics_present(
    train_status_cli, tmp_path, capsys
):
    """记录带指标时,stderr 在训练摘要之后追加逐条 loss 趋势人话(与页面曲线
    同一来源);旧产物目录没有序列文件时如实说没有,不崩、不编造。"""
    invoke, record = train_status_cli
    output = tmp_path / "run-9-output"
    output.mkdir()
    (output / "workbench_loss_history.json").write_text(
        json.dumps([{"step": s, "loss": 2.0 - 0.1 * s} for s in range(6)]),
        encoding="utf-8",
    )
    record.update(
        {
            "status": "succeeded",
            "config": {"training": {"num_epochs": 1}},
            "metrics": {"train_loss": 0.5},
            "output_dir": str(output),
        }
    )
    assert invoke("train-status", "run-9") == 0
    err = capsys.readouterr().err
    assert (
        "loss 从第 0 步的 2.0000 走到第 5 步的 1.5000（共 6 个记录点），整体在下降。" in err
    )
    assert "loss 下降只说明模型在逐步记住训练题" in err
    # 旧产物目录没有序列文件:趋势区如实说没有逐条记录,不让命令崩掉。
    record["output_dir"] = str(tmp_path / "missing-dir")
    assert invoke("train-status", "run-9") == 0
    assert "这次训练没有留下逐条 loss 记录" in capsys.readouterr().err


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


def test_train_lineage_cli_prints_json_and_registration_summary(tmp_path, monkeypatch, capsys):
    """train-lineage 双流契约:stdout 纯 JSON,stderr 给注册状态人话(summarize_registration 单一来源)。"""
    from types import SimpleNamespace

    import mlflow.tracking

    client = SimpleNamespace()
    client.search_registered_models = lambda max_results=None: [
        SimpleNamespace(name="工单分类")
    ]
    client.search_model_versions = lambda query: [
        SimpleNamespace(
            name="工单分类",
            version=3,
            aliases=["champion"],
            current_stage="Production",
            run_id="mfr-1",
        )
    ]
    client.get_run = lambda run_id: SimpleNamespace(
        data=SimpleNamespace(tags={"workbench.run_id": "wb-1"}, params={}, metrics={})
    )
    # registry_link 在函数内 from mlflow.tracking import MlflowClient,
    # 打模块属性即可拦截;lambda 吃掉任意签名,不真连 sqlite。
    monkeypatch.setattr(mlflow.tracking, "MlflowClient", lambda *args, **kwargs: client)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "data_intake.py",
            "--store",
            str(tmp_path / "intake"),
            "--training-root",
            str(tmp_path / "training"),
            "train-lineage",
            "wb-1",
        ],
    )
    assert data_intake.main() == 0
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["status"] == "registered"
    assert payload["versions"][0]["aliases"] == ["champion"]
    # stderr 人话与页面训练记录区同一份摘要:点名版本与别名 + 注册边界句
    assert "这次训练已注册到模型库：工单分类 v3（champion）。" in captured.err
    assert "注册只说明模型库记录了这次训练的产物与血缘，不代表业务效果达标。" in captured.err


def test_train_export_cli_prints_json_and_export_summary(tmp_path, monkeypatch, capsys):
    """train-export 双流契约:stdout 纯 JSON,stderr 给导出人话;合并层打桩,证据链真写。"""
    from pathlib import Path

    import src.models.merger
    import src.workbench.training_runs

    IntakeService(tmp_path / "intake")  # train-export 不读任务,--store 只需可用
    adapter_dir = tmp_path / "training" / "wb-x" / "model"
    adapter_dir.mkdir(parents=True)
    (adapter_dir / "adapter_config.json").write_text("{}", encoding="utf-8")
    (adapter_dir / "adapter_model.safetensors").write_text("w", encoding="utf-8")
    base_dir = tmp_path / "base-model"
    base_dir.mkdir()
    record = {
        "run_id": "wb-x",
        "session_id": "sess-1",
        "dataset_version": "ds-v3",
        "status": "succeeded",
        "model_path": str(base_dir),
        "output_dir": str(adapter_dir),
    }

    class Training:
        def __init__(self, root, project_root=None):
            pass

        def get_status(self, run_id):
            return record

    monkeypatch.setattr(src.workbench.training_runs, "TrainingRunService", Training)

    def fake_merge(adapter, output_dir, base_model_name=None, dtype="bfloat16"):
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "config.json").write_text("{}", encoding="utf-8")
        (out / "model.safetensors").write_text("w", encoding="utf-8")
        return str(out)

    # export_model 在函数内 from src.models.merger import ...,打模块属性即可拦截。
    monkeypatch.setattr(src.models.merger, "merge_adapter_to_dir", fake_merge)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "data_intake.py",
            "--store",
            str(tmp_path / "intake"),
            "--training-root",
            str(tmp_path / "training"),
            "train-export",
            "wb-x",
        ],
    )
    assert data_intake.main() == 0
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["status"] == "exported"
    assert payload["dataset_version"] == "ds-v3"
    # 默认导出目录:训练根目录的 merged 兄弟目录下按 run_id 命名。
    assert payload["output_dir"] == str(tmp_path / "merged" / "wb-x")
    evidence = json.loads(
        (tmp_path / "merged" / "wb-x" / "export_evidence.json").read_text(encoding="utf-8")
    )
    assert evidence["run_id"] == "wb-x"
    assert evidence["dataset_version"] == "ds-v3"
    err = captured.err
    assert "合并导出完成：这次训练的适配器已并入基础模型" in err
    assert str(tmp_path / "merged" / "wb-x") in err
    assert "可被 vLLM、Ollama、LM Studio 直接加载" in err
    assert "export_evidence.json" in err
    assert "导出只产出模型文件与证据记录，不代表业务效果达标，也不会自动部署。" in err


def test_train_export_cli_blocked_run_exits_two(tmp_path, monkeypatch, capsys):
    """盘点不过关是写操作失败:exit 2,stderr 逐条点名原因,不产出任何 JSON。"""
    import src.workbench.training_runs

    IntakeService(tmp_path / "intake")

    class Training:
        def __init__(self, root, project_root=None):
            pass

        def get_status(self, run_id):
            return {"run_id": "wb-y", "status": "running", "model_path": "", "output_dir": ""}

    monkeypatch.setattr(src.workbench.training_runs, "TrainingRunService", Training)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "data_intake.py",
            "--store",
            str(tmp_path / "intake"),
            "--training-root",
            str(tmp_path / "training"),
            "train-export",
            "wb-y",
        ],
    )
    assert data_intake.main() == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "只有成功完成的训练才能合并导出；这次运行当前状态是 running。" in captured.err
