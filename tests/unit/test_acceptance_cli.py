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
    }

    class Acceptance:
        def __init__(self, *args):
            pass

        def prepare(self, current, model, protocol, criteria):
            calls.append(("prepare", current.revision, model, protocol, criteria))
            record.update(model=asdict(model), protocol=asdict(protocol), criteria=criteria)
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
                "--acceptance-root",
                str(tmp_path / "acceptance"),
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
