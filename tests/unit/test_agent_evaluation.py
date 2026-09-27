"""Evidence/tool contract tests; scripted models do not prove diagnosis quality."""

import json
from copy import deepcopy

import pytest

pytest.importorskip("datasets")

from src.agent.evaluation import assess_evaluation, load_assessments
from src.workbench.business_evaluation import (
    BusinessEvaluationService,
    EvaluationModel,
    EvaluationProtocol,
    Generation,
)
from src.workbench.evaluation_diagnostics import EvaluationDiagnostics
from src.workbench.intake_service import IntakeService
from tests.unit.test_data_intake import ScriptedModel
from tests.unit.test_data_materialize import _full


@pytest.fixture
def evaluated(tmp_path):
    intake = IntakeService(tmp_path / "intake")
    session = _full(intake)
    session = intake.materialize_dataset(session.session_id, session.revision)
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}")
    (model / "model.safetensors").write_bytes(b"fixture-only")

    class Runtime:
        def __init__(self, _model):
            pass

        def generate(self, _prompt, _protocol):
            return Generation("yes\n### Input: unwanted continuation", truncated=True)

        def close(self):
            pass

    report = BusinessEvaluationService(tmp_path / "eval").compare(
        session,
        [EvaluationModel("base", str(model)), EvaluationModel("tuned", str(model))],
        EvaluationProtocol(scorer="classification_exact", max_new_tokens=8),
        runtime_factory=Runtime,
    )
    return intake, session, report


def assessment(citations=None):
    return {
        "summary": "生成没有完整结束，尚不能认定分类达标。",
        "observations": [
            {
                "statement": "输出包含额外内容且达到长度限制。",
                "evidence_ids": citations or ["base:0"],
            }
        ],
        "hypotheses": [
            {
                "statement": "停止监督可能不足，但尚待核查。",
                "evidence_ids": ["base:0"],
                "verification": "检查真实训练 token 与结束标记，再用固定开发集验证。",
            }
        ],
        "decision": "inspect_training",
        "next_steps": ["核查训练格式和停止条件。"],
        "limitations": ["这是虚构小样本开发对照，不能推断客户业务收益。"],
    }


def test_assessment_requires_summary_real_cases_and_resolvable_citations(evaluated, tmp_path):
    intake, session, report = evaluated
    valid = assessment()
    client = ScriptedModel(
        [
            ("submit_evaluation_assessment", valid),
            ("inspect_evaluation_summary", {}),
            ("submit_evaluation_assessment", valid),
            ("inspect_bad_cases", {"limit": 1}),
            ("submit_evaluation_assessment", assessment(["nonexistent:9"])),
            ("submit_evaluation_assessment", valid),
        ]
    )
    original = session.model_dump()
    result = assess_evaluation(report, session, client, output_root=tmp_path)
    assert [entry["ok"] for entry in result["tool_trace"]] == [
        False,
        True,
        False,
        True,
        False,
        True,
    ]
    assert result["assessment"]["hypotheses"][0]["verification"]
    assert result["comparison_key"] == report.comparison_key
    assert load_assessments(tmp_path, report.evaluation_id) == [result]
    assert load_assessments(tmp_path, "another") == []
    assert intake.load(session.session_id).model_dump() == original
    stored = next((tmp_path / "assessments").glob("*.json"))
    modified = json.loads(stored.read_text())
    modified["assessment"]["summary"] = "forged"
    stored.write_text(json.dumps(modified))
    with pytest.raises(ValueError, match="身份不匹配"):
        load_assessments(tmp_path, report.evaluation_id)


def test_invalid_report_is_rejected_before_sending_data_to_agent(evaluated):
    _, session, report = evaluated
    report.models[0]["rows"][0]["expected"] = "invented label"
    client = ScriptedModel([])
    with pytest.raises(ValueError, match="不匹配"):
        assess_evaluation(report, session, client)
    assert client.seen == []


def test_adapter_diagnosis_must_read_available_training_evidence(evaluated, tmp_path):
    _, session, report = evaluated
    # A third-party adapter can lack workbench training receipts; the tool must
    # report that absence before the Agent treats training causes as hypotheses.
    report.models[1]["requested_model"]["adapter_path"] = str(tmp_path / "external-adapter")
    client = ScriptedModel(
        [
            ("inspect_evaluation_summary", {}),
            ("inspect_bad_cases", {}),
            ("submit_evaluation_assessment", assessment()),
            ("inspect_training_evidence", {}),
            ("submit_evaluation_assessment", assessment()),
        ]
    )
    result = assess_evaluation(report, session, client)
    assert result["tool_trace"][2]["ok"] is False
    assert "inspect_training_evidence" in result["tool_trace"][2]["error"]
    assert result["tool_trace"][3]["ok"] is True
    assert result["tool_trace"][4]["ok"] is True


def test_partial_or_skipped_large_case_cannot_be_cited(evaluated):
    _, session, report = evaluated
    report = deepcopy(report)
    report.models[0]["rows"][0]["output"] = "长" * 45000
    diagnostics = EvaluationDiagnostics(report, session)
    page = diagnostics.read_cases(limit=1)
    assert page["cases"][0]["content_status"] == "requires_chunks"
    steps = [
        ("inspect_evaluation_summary", {}),
        ("inspect_bad_cases", {"limit": 1}),
        ("inspect_case_content", {"evidence_id": "base:0", "offset": 10}),
        ("inspect_case_content", {"evidence_id": "base:0", "limit": 16384}),
        ("submit_evaluation_assessment", assessment()),
    ]
    offset = 16384
    while True:
        chunk = diagnostics.read_case_content("base:0", offset=offset)
        steps.append(("inspect_case_content", {"evidence_id": "base:0", "offset": offset}))
        if not chunk["has_more"]:
            break
        offset = chunk["next_offset"]
    steps.append(("submit_evaluation_assessment", assessment()))
    result = assess_evaluation(report, session, ScriptedModel(steps))
    assert result["tool_trace"][2]["ok"] is False
    assert result["tool_trace"][4]["ok"] is False
    assert result["tool_trace"][-1]["ok"] is True
