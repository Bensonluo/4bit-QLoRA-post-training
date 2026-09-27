"""Business scoring is drafted, explicitly confirmed, and scoped to its task."""

import json
import sys

import pytest

from scripts import data_intake
from src.workbench.business_evaluation import EvaluationReport
from src.workbench.intake_service import IntakeService
from tests.unit.test_data_materialize import _full


@pytest.fixture()
def scoring_cli(tmp_path, monkeypatch):
    import src.agent.scoring as agent
    import src.workbench.acceptance as acceptance
    import src.workbench.business_evaluation as evaluation
    import src.workbench.business_scoring as scoring
    import src.workbench.training_runs as training

    service = IntakeService(tmp_path / "intake")
    session = _full(service)
    session = service.materialize_dataset(session.session_id, session.revision)
    record = {
        "scoring_id": "score-fixture",
        "session_id": session.session_id,
        "status": "draft",
        "spec_digest": "c" * 64,
    }
    calls = []
    root = tmp_path / "scoring"

    class Scoring:
        def __init__(self, *args):
            pass

        def get(self, identity):
            return record

        def list_specs(self, session_id=None):
            return [record]

        def confirm(self, identity, current):
            calls.append(("confirm", identity, current.revision))
            record["status"] = "confirmed"
            return {"root": str(root), "scoring_id": identity, "spec_digest": record["spec_digest"]}

    def recommend(current, standard, client, *, output_root):
        calls.append(("draft", standard, client.model))
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

    class Evaluation:
        def __init__(self, *args):
            pass

        def compare(self, current, models, protocol):
            calls.append(("compare", protocol))
            return EvaluationReport(
                "a" * 32, "now", {"version": current.dataset.version}, {}, "key"
            )

    class Acceptance:
        def __init__(self, *args):
            pass

        def prepare(self, current, model, protocol, criteria):
            calls.append(("acceptance", protocol, criteria))
            return {"status": "prepared"}

    monkeypatch.setattr(scoring, "ScoringService", Scoring)
    monkeypatch.setattr(agent, "recommend_scoring", recommend)
    monkeypatch.setattr(training, "TrainingRunService", Training)
    monkeypatch.setattr(evaluation, "BusinessEvaluationService", Evaluation)
    monkeypatch.setattr(acceptance, "AcceptanceService", Acceptance)
    for name, value in {
        "PROVIDER": "compatible",
        "BASE_URL": "https://fixture.example/v1",
        "MODEL": "tool-fixture",
        "API_KEY": "fixture-secret",
    }.items():
        monkeypatch.setenv("TUNESMITH_AGENT_" + name, value)

    def invoke(*args):
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "data_intake.py",
                "--store",
                str(service.root),
                "--scoring-root",
                str(root),
                *map(str, args),
            ],
        )
        return data_intake.main()

    return invoke, session, record, calls


def test_scoring_proposal_requires_remote_consent_and_separate_confirmation(scoring_cli, capsys):
    invoke, session, record, calls = scoring_cli
    args = (
        "scoring-propose",
        session.session_id,
        "--revision",
        session.revision,
        "--business-standard",
        "须含全部处理步骤",
    )
    assert invoke(*args) == 2
    assert calls == []
    assert "需先允许" in capsys.readouterr().err
    assert invoke(*args, "--allow-remote-data") == 0
    assert json.loads(capsys.readouterr().out)["status"] == "draft"
    assert [call[0] for call in calls] == ["draft"]
    assert (
        invoke(
            "scoring-confirm",
            session.session_id,
            record["scoring_id"],
            "--revision",
            session.revision,
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["spec_digest"] == record["spec_digest"]
    assert record["status"] == "confirmed"


def test_confirmed_rules_drive_business_metrics_and_acceptance_pass_rate(scoring_cli, capsys):
    invoke, session, record, calls = scoring_cli
    compare = (
        "eval-compare",
        session.session_id,
        "run",
        "--revision",
        session.revision,
        "--scoring-id",
        record["scoring_id"],
    )
    assert invoke(*compare) == 2
    assert "明确确认" in capsys.readouterr().err
    assert calls == []
    record["status"] = "confirmed"
    assert invoke(*compare) == 0
    capsys.readouterr()
    assert calls[-1][1].scorer == "custom_rules"
    assert calls[-1][1].custom_scoring["spec_digest"] == record["spec_digest"]
    assert (
        invoke(
            "acceptance-prepare",
            session.session_id,
            "run",
            "--revision",
            session.revision,
            "--scoring-id",
            record["scoring_id"],
            "--business-standard",
            "业务确认标准",
            "--minimum-score",
            0.8,
            "--minimum-cases",
            10,
        )
        == 0
    )
    assert calls[-1][1].scorer == "custom_rules"
    assert calls[-1][2]["metric"] == "pass_rate"
    capsys.readouterr()
    record["session_id"] = "another-task"
    count = len(calls)
    assert invoke(*compare) == 2
    assert "当前业务任务" in capsys.readouterr().err
    assert len(calls) == count


def test_ambiguous_standard_returns_questions_without_a_confirmable_spec(
    scoring_cli, monkeypatch, capsys
):
    import src.agent.scoring as agent

    invoke, session, _, calls = scoring_cli
    monkeypatch.setattr(
        agent,
        "recommend_scoring",
        lambda *args, **kwargs: {
            "status": "needs_business_input",
            "reason": "专业需具体标准",
            "questions": ["哪些要素必须正确？"],
            "trace": [],
        },
    )
    assert (
        invoke(
            "scoring-propose",
            session.session_id,
            "--revision",
            session.revision,
            "--business-standard",
            "专业一点",
            "--allow-remote-data",
        )
        == 0
    )
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "needs_business_input" and "scoring_id" not in result
    assert calls == []
