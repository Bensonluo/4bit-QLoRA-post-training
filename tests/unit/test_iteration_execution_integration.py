"""Actual local worker, tiny random model training and sequential development inference.

The tiny model checks execution handoffs, never business quality or financial prediction.
"""

import shutil
import sys
import time
from pathlib import Path

from src.workbench.business_evaluation import EvaluationModel, EvaluationProtocol
from src.workbench.iterations import IterationService
from tests.unit.test_workbench_training_runs import _prepare, _wait, environment  # noqa: F401


def test_one_authorization_reaches_real_three_model_development_comparison(
    environment,  # noqa: F811
    tmp_path,
):
    from src.workbench.iteration_execution import IterationExecutionService

    intake, session, training, model = environment
    repository = Path(__file__).resolve().parents[2]
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
    protocol = EvaluationProtocol("classification_exact", max_new_tokens=4)
    comparison = iterations.evaluations.compare(
        session,
        [
            EvaluationModel("base", str(model)),
            EvaluationModel("parent", str(model), parent["output_dir"]),
        ],
        protocol,
    )
    assert comparison.status in {"completed", "completed_with_failures"}
    assert all(not entry["errors"] for entry in comparison.models), comparison.models
    proposal = iterations.propose(
        session,
        parent_run_id=parent["run_id"],
        evaluation_id=comparison.evaluation_id,
        hypothesis="虚构样例用于检查已确认配置的自动执行交接",
        expected_outcome="获得同协议开发集证据，效果不作预设",
        changes="保持资料和训练配置，验证后台完成第二轮训练与对照",
    )
    identity = proposal["iteration_id"]
    iterations.confirm(identity, session)
    executions = IterationExecutionService(
        iterations.root / "executions",
        intake.root,
        iterations.root,
        training.root,
        iterations.evaluations.root,
        project_root=training.project_root,
        python_executable=sys.executable,
    )
    started = executions.start(identity, session)
    assert started["status"] not in {"blocked", "failed"}, started
    # There is deliberately no refresh/advance operation: get is a pure read.
    deadline = time.monotonic() + 120
    result = started
    try:
        while time.monotonic() < deadline:
            result = executions.get(identity)
            if result["status"] in {
                "completed",
                "blocked",
                "failed",
                "stopped",
                "awaiting_warning_ack",
            }:
                break
            time.sleep(0.2)
        assert result["status"] == "completed", result
        record = iterations.get(identity)
        assert record["status"] == "evaluated"
        assert "decision" not in record
        report = iterations.evaluations.get_report(record["evaluation_id"])
        assert report.dataset["split"] == "validation"
        assert report.dataset["purpose"] == "development_only"
        assert report.dataset["evaluation_suite"] == proposal["evaluation_suite"]
        assert report.protocol["max_new_tokens"] == 4
        assert len(report.models) == 3
        assert all(not entry["errors"] for entry in report.models), report.models
        assert training.get_status(record["new_run_id"])["status"] == "succeeded"
        run_count = len(training.list_runs(session.session_id))
        duplicate = executions.start(identity, intake.load(session.session_id))
        assert duplicate["evaluation_id"] == result["evaluation_id"]
        assert len(training.list_runs(session.session_id)) == run_count == 2
    finally:
        if result["status"] not in {"completed", "blocked", "failed", "stopped"}:
            executions.stop(identity)
