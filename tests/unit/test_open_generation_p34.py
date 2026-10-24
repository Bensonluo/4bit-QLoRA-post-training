"""开放生成任务的 P3.4 自动执行链验证——泛化方向的贯穿闭环。

分类原型已有同款贯穿测试;本测试用开放生成原型走同一条链:审阅式盲标
→ 真实训练 → open_review 对照 → 迭代绑定 decision_metric=manual_review。
不训练真实业务效果,只证明自动执行链对生成原型无协议漂移。
"""

import shutil
import sys
import time
from pathlib import Path

import pytest

pytest.importorskip("datasets")
pytest.importorskip("peft")

from src.workbench.business_evaluation import EvaluationModel, EvaluationProtocol
from src.workbench.iteration_execution import IterationExecutionService
from src.workbench.iterations import IterationService
from tests.unit.test_workbench_training_runs import (  # noqa: F401
    _prepare,
    _wait,
    environment,
)


def test_open_generation_auto_execution_to_manual_review_iteration(
    environment,
    tmp_path,  # noqa: F811
):

    repository = Path(__file__).resolve().parents[2]
    intake, session, training, model = environment
    shutil.copy(
        repository / "scripts/workbench_iterate.py",
        training.project_root / "scripts/workbench_iterate.py",
    )
    parent = _prepare(environment)
    training.start(parent["run_id"], session)
    parent = _wait(training, parent["run_id"])
    assert parent["status"] == "succeeded", parent

    iterations = IterationService(tmp_path / "iterations", training.root, tmp_path / "eval")
    iterations.training = training
    protocol = EvaluationProtocol("open_review", max_new_tokens=8)
    comparison = iterations.evaluations.compare(
        session,
        [
            EvaluationModel("base", str(model)),
            EvaluationModel("parent", str(model), parent["output_dir"]),
        ],
        protocol,
    )
    assert comparison.status in {"completed", "completed_with_failures"}

    proposal = iterations.propose(
        session,
        parent_run_id=parent["run_id"],
        evaluation_id=comparison.evaluation_id,
        hypothesis="开放生成原型验证自动执行链",
        expected_outcome="open_review 协议经自动执行无漂移",
        changes="保持资料和训练配置",
    )
    identity = proposal["iteration_id"]
    iterations.confirm(identity, session)

    executions = IterationExecutionService(
        iterations.root / "executions",
        tmp_path / "intake",
        iterations.root,
        training.root,
        tmp_path / "eval",
        project_root=training.project_root,
        python_executable=sys.executable,
    )
    started = executions.start(identity, session)
    assert started["status"] not in {"blocked", "failed"}, started
    deadline = time.monotonic() + 120
    result = started
    while time.monotonic() < deadline:
        result = executions.get(identity)
        if result["status"] in {"completed", "blocked", "failed", "stopped"}:
            break
        time.sleep(0.2)
    assert result["status"] == "completed", result

    record = iterations.get(identity)
    assert record["status"] == "evaluated"
    assert "decision" not in record
    assert record["decision_metric"] == "manual_review"
    report = iterations.evaluations.get_report(record["evaluation_id"])
    assert report.protocol["scorer"] == "open_review"
    assert len(report.models) == 3
