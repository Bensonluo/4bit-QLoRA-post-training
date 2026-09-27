"""Provider-facing schemas and full-source evidence survive real intake transitions."""

import json

import pytest
from pydantic import ValidationError

from src.agent.intake import TOOLS
from src.workbench.intake_models import IntakeAnalysis
from src.workbench.intake_service import IntakeService
from src.workbench.recipes import validate_analysis
from tests.unit.test_data_intake import ScriptedModel, analysis
from tests.unit.test_full_data import FULL, approved


def test_tool_schemas_inline_nested_objects_without_weakening_validation():
    serialized = json.dumps(TOOLS)
    assert '"$defs"' not in serialized
    assert '"$ref"' not in serialized
    submit = next(
        tool["function"] for tool in TOOLS if tool["function"]["name"] == "submit_analysis"
    )
    task_schema = submit["parameters"]["properties"]["task"]
    assert task_schema["type"] == "object"
    assert task_schema["properties"]["goal"]["type"] == "string"
    roles = task_schema["properties"]["field_roles"]["items"]
    assert roles["type"] == "object"
    assert roles["properties"]["column"]["type"] == "string"
    assert task_schema["additionalProperties"] is False
    assert roles["additionalProperties"] is False

    payload = analysis().model_dump()
    payload["task"]["field_roles"][0]["invented_business_approval"] = True
    with pytest.raises(ValidationError, match="extra_forbidden"):
        IntakeAnalysis.model_validate(payload)
    payload = analysis().model_dump()
    payload["task"] = json.dumps(payload["task"])
    with pytest.raises(ValidationError):
        IntakeAnalysis.model_validate(payload)


@pytest.fixture()
def full_issue_session(tmp_path):
    service = IntakeService(tmp_path / "intake")
    session = approved(service)
    full = FULL.decode().replace("质量,补发", ",补发", 1).encode()
    session = service.validate_full_data(session.session_id, session.revision, "full.csv", full)
    return service, session


def test_agent_reanalysis_reads_full_issues_without_confusing_sample_row_ids(full_issue_session):
    service, session = full_issue_session
    full_digest = session.full_data.source.digest
    full_ref = f"full:{full_digest}:r000001"
    analysis_payload = session.analysis.model_dump()
    analysis_payload["findings"].append(
        {
            "kind": "observed",
            "message": "全量第一条工单缺少类别，需补充可信答案。",
            "evidence_row_ids": [full_ref],
        }
    )
    result = IntakeAnalysis.model_validate(analysis_payload)
    session = service.answer(session.session_id, "请检查全量文件缺少答案的行。")
    model = ScriptedModel(
        [
            ("inspect_full_data", {}),
            ("profile_data", {}),
            ("inspect_rows", {"row_ids": ["r000001"]}),
            ("preview_recipe", result.recipe.model_dump()),
            ("submit_analysis", result.model_dump()),
        ]
    )
    updated = service.analyze(session.session_id, model)
    full_tool = next(message for message in model.seen[1] if message["role"] == "tool")
    payload = json.loads(full_tool["content"])
    assert payload["source_digest"] == full_digest
    assert payload["record_count"] == 3
    assert payload["status"] == "stale"
    assert any(full_ref in issue["row_ids"] for issue in payload["issues"])
    full_row = next(row for row in payload["rows"] if row["row_id"] == full_ref)
    assert full_row["values"]["客户描述"] == "收到破杯"
    assert all(row["row_id"].startswith(f"full:{full_digest}:") for row in payload["rows"])
    assert all(
        row_id.startswith(f"full:{full_digest}:")
        for issue in payload["issues"]
        for row_id in issue["row_ids"]
    )
    sample_tool = next(
        message
        for message in model.seen[3]
        if message["role"] == "tool" and message["tool_call_id"] == "call-3"
    )
    sample_row = json.loads(sample_tool["content"])["rows"][0]
    assert sample_row["row_id"] == "r000001"
    assert sample_row["values"]["客户描述"] == "杯子破损"
    assert updated.analysis.findings[-1].evidence_row_ids == [full_ref]
    assert {"tool": "inspect_full_data", "ok": True} in updated.tool_trace
    assert updated.source.digest != full_digest


@pytest.mark.parametrize("invalid_reference", ["wrong_digest", "missing_row", "field_role"])
def test_full_evidence_references_require_exact_source_and_scope(
    full_issue_session, invalid_reference
):
    _, session = full_issue_session
    full_source = session.full_data.source
    valid_reference = f"full:{full_source.digest}:r000001"
    result = session.analysis.model_copy(deep=True)
    result.findings[0].evidence_row_ids = [valid_reference]
    validate_analysis(session.source, result, full_source=full_source)
    if invalid_reference == "wrong_digest":
        result.findings[0].evidence_row_ids = [f"full:{'0' * 64}:r000001"]
    elif invalid_reference == "missing_row":
        result.findings[0].evidence_row_ids = [f"full:{full_source.digest}:r999999"]
    else:
        result.task.field_roles[0].evidence_row_ids = [valid_reference]
    with pytest.raises(ValueError, match="证据行"):
        validate_analysis(session.source, result, full_source=full_source)
