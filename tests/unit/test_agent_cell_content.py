"""Long-cell evidence reaches the actual intake tool protocol without mutation."""

import csv
import io
import json

import pytest

from src.agent.intake import _read_cell_content, analyze_intake
from src.workbench.intake_service import IntakeService
from src.workbench.sources import canonical, read_source
from tests.unit.test_data_intake import CSV, ScriptedModel, analysis
from tests.unit.test_full_data import FULL, approved


def test_large_csv_cell_survives_upload_reopen_and_agent_chunk_reads(tmp_path):
    value = ('虚构财报，收入说明；包含 "引用"\n' * 8000) + "正文结束，未作截断"
    assert len(value) > 131072
    content = io.StringIO(newline="")
    writer = csv.writer(content)
    writer.writerow(["编号", "客户描述", "类别", "处理结果"])
    writer.writerow(["001", value, "质量", "补发"])
    writer.writerow(["002", "物流未更新", "物流", "补发"])
    previous_limit = csv.field_size_limit()
    service = IntakeService(tmp_path)
    session = service.create("根据描述判断类别", "long.csv", content.getvalue().encode())
    assert csv.field_size_limit() == previous_limit
    session = service.load(session.session_id)
    assert session.source.rows[0].values["客户描述"] == value
    assert session.source.rows[0].values["编号"] == "001"
    assert session.source.rows[1].line == 8003
    chunks = [
        ("read_cell_content", request(offset=offset, limit=16384))
        for offset in range(0, len(value), 16384)
    ]
    client = ScriptedModel([*chunks, *complete_steps(analysis())])
    _, trace = analyze_intake(session, client)
    cells = results(client)[: len(chunks)]
    assert "".join(cell["content"] for cell in cells) == value
    assert cells[-1]["has_more"] is False
    assert all(item["ok"] for item in trace)


def test_large_malformed_csv_still_fails_and_restores_parser_limit():
    previous_limit = csv.field_size_limit()
    data = ('field,answer\n"' + "x" * 150000).encode()
    with pytest.raises(ValueError, match="CSV.*解析失败"):
        read_source("broken.csv", data, delimiter=",")
    assert csv.field_size_limit() == previous_limit


def complete_steps(proposal):
    return [
        ("profile_data", {}),
        ("inspect_rows", {"row_ids": []}),
        ("preview_recipe", proposal.recipe.model_dump()),
        ("submit_analysis", proposal.model_dump()),
    ]


def results(client):
    return [
        json.loads(message["content"]) for message in client.seen[-1] if message["role"] == "tool"
    ]


def request(**updates):
    return {"source_kind": "current", "row_id": "r000001", "column": "客户描述", **updates}


def test_tail_after_2000_reaches_actual_analyze_intake_model(tmp_path):
    value = "公开文字" * 1500 + "仅末尾出现的已审计尾注"
    data = CSV.decode().replace("杯子破损", value).encode()
    session = IntakeService(tmp_path).create("分析合同尾注", "sample.csv", data)
    before = session.model_dump()
    proposal = analysis()
    client = ScriptedModel(
        [
            ("inspect_rows", {"row_ids": ["r000001"]}),
            ("read_cell_content", request(offset=0, limit=4096)),
            ("read_cell_content", request(offset=4096, limit=16384)),
            *complete_steps(proposal),
        ]
    )
    accepted, trace = analyze_intake(session, client)
    observed = results(client)
    assert "展示截断" in observed[0]["rows"][0]["values"]["客户描述"]
    first, second = observed[1:3]
    assert len(first["content"]) == 4096 and first["next_offset"] == 4096 and first["has_more"]
    assert first["content"] + second["content"] == value
    assert second["content"].endswith("仅末尾出现的已审计尾注")
    assert second["total_characters"] == len(value) and not second["has_more"]
    assert second["source_digest"] == session.source.digest
    assert second["evidence_ref"] == f"source:{session.source.digest}:r000001"
    assert accepted == proposal and all(item["ok"] for item in trace)
    assert session.model_dump() == before


@pytest.mark.parametrize(
    "invalid",
    [
        request(offset=True),
        request(offset=-1),
        request(limit=0),
        request(limit=16385),
        request(limit="12"),
        request(offset=99999),
        request(row_id="source:wrong:r000001"),
        request(row_id="r999999"),
        request(column="/etc/passwd"),
        request(source_kind="source"),
        request(source_kind="current", alias="main"),
        request(source_kind="full"),
        request(source_kind="source", alias="../../outside"),
        request(source_kind=[]),
        request(path="https://example.com"),
        {"source_kind": "current", "column": "客户描述"},
    ],
)
def test_invalid_cell_reference_or_pagination_is_rejected_in_tool_protocol(tmp_path, invalid):
    session = IntakeService(tmp_path).create("判断问题类别", "sample.csv", CSV)
    before = session.model_dump()
    client = ScriptedModel([("read_cell_content", invalid), *complete_steps(analysis())])
    _, trace = analyze_intake(session, client)
    assert trace[0]["ok"] is False
    assert "error" in results(client)[0]
    assert session.model_dump() == before


def test_current_composition_and_named_source_are_distinct(tmp_path):
    from tests.unit.test_composed_intake import analysis_for, setup

    session = setup(IntakeService(tmp_path))
    proposal = analysis_for(session)
    before = session.model_dump()
    client = ScriptedModel(
        [
            ("preview_composition", proposal.composition),
            ("read_cell_content", request(column="label_category")),
            ("read_cell_content", request(source_kind="source", alias="labels", column="category")),
            *complete_steps(proposal),
        ]
    )
    _, trace = analyze_intake(session, client)
    observed = results(client)
    current, original = observed[1:3]
    assert current["content"] == original["content"] == "咨询"
    assert current["source_digest"] != original["source_digest"]
    assert current["finding_row_id"] == "r000001"
    assert original["finding_row_id"] == f"source:{session.sources['labels'].digest}:r000001"
    assert all(item["ok"] for item in trace)
    assert session.model_dump() == before


def test_full_report_source_keeps_qualified_identity_and_stale_status(tmp_path):
    service = IntakeService(tmp_path)
    session = approved(service)
    long_value = "全量长记录" * 800 + "全量尾部"
    session = service.validate_full_data(
        session.session_id,
        session.revision,
        "full.csv",
        FULL.decode().replace("收到破杯", long_value).encode(),
    )
    session.full_data.status = "stale"
    before = session.model_dump()
    client = ScriptedModel(
        [
            ("read_cell_content", request(source_kind="full", offset=2000, limit=16384)),
            *complete_steps(session.analysis),
        ]
    )
    _, trace = analyze_intake(session, client)
    cell = results(client)[0]
    assert cell["content"] == long_value[2000:]
    assert cell["full_report_status"] == "stale"
    assert cell["evidence_ref"] == f"full:{session.full_data.source.digest}:r000001"
    assert trace[0]["ok"] and session.model_dump() == before


def test_structured_values_use_deterministic_json_and_exact_empty_end():
    value = {"z": [True, None, "中文"], "a": {"nested": 3}}
    source = read_source(
        "structured.jsonl", json.dumps({"contract": value}, ensure_ascii=False).encode()
    )
    args = request(column="contract", limit=8)
    first = _read_cell_content(source, {"main": source}, None, args)
    second = _read_cell_content(
        source, {"main": source}, None, {**args, "offset": 8, "limit": 16384}
    )
    assert first["field_type"] == "object" and first["encoding"] == "canonical_json"
    assert first["content"] + second["content"] == canonical(value)
    end = _read_cell_content(
        source, {"main": source}, None, {**args, "offset": first["total_characters"]}
    )
    assert end["content"] == "" and end["next_offset"] is None and not end["has_more"]


@pytest.mark.parametrize(
    "inspection",
    [
        ("inspect_source", {"alias": "main"}),
        ("read_cell_content", request(limit=1)),
        ("read_cell_content", request(source_kind="source", alias="main", limit=1)),
    ],
)
def test_current_source_evidence_satisfies_inspection_without_redundant_inspect_rows(
    tmp_path, inspection
):
    session = IntakeService(tmp_path).create("核实样例", "sample.csv", CSV)
    proposal = analysis()
    client = ScriptedModel(
        [
            ("profile_data", {}),
            inspection,
            ("preview_recipe", proposal.recipe.model_dump()),
            ("submit_analysis", proposal.model_dump()),
        ]
    )
    accepted, trace = analyze_intake(session, client)
    assert accepted == proposal
    assert all(item["ok"] for item in trace)
    assert not any(item["tool"] == "inspect_rows" for item in trace)


def test_empty_cell_fragment_does_not_satisfy_current_inspection(tmp_path):
    session = IntakeService(tmp_path).create("核实样例", "sample.csv", CSV)
    proposal = analysis()
    length = len(session.source.rows[0].values["客户描述"])
    client = ScriptedModel(
        [
            ("profile_data", {}),
            ("read_cell_content", request(offset=length)),
            ("preview_recipe", proposal.recipe.model_dump()),
            ("submit_analysis", proposal.model_dump()),
            ("read_cell_content", request(limit=1)),
            ("submit_analysis", proposal.model_dump()),
        ]
    )
    _, trace = analyze_intake(session, client)
    assert not trace[3]["ok"] and "真实证据行" in trace[3]["error"]
    assert trace[-1]["ok"]


@pytest.mark.parametrize(
    "inspection",
    [
        ("inspect_source", {"alias": "main"}),
        (
            "read_cell_content",
            {
                "source_kind": "source",
                "alias": "main",
                "row_id": "r000001",
                "column": "description",
            },
        ),
    ],
)
def test_original_source_cannot_satisfy_transformed_source_inspection(tmp_path, inspection):
    from tests.unit.test_composed_intake import analysis_for, setup

    session = setup(IntakeService(tmp_path))
    proposal = analysis_for(session)
    client = ScriptedModel(
        [
            inspection,
            ("preview_composition", proposal.composition),
            ("profile_data", {}),
            inspection,
            ("preview_recipe", proposal.recipe.model_dump()),
            ("submit_analysis", proposal.model_dump()),
            (
                "read_cell_content",
                {"source_kind": "current", "row_id": "r000001", "column": "label_category"},
            ),
            ("submit_analysis", proposal.model_dump()),
        ]
    )
    _, trace = analyze_intake(session, client)
    assert not trace[5]["ok"] and "真实证据行" in trace[5]["error"]
    assert trace[-1]["ok"]


def test_cell_read_does_not_bypass_profile_guard(tmp_path):
    session = IntakeService(tmp_path).create("核实样例", "sample.csv", CSV)
    proposal = analysis()
    client = ScriptedModel(
        [
            ("read_cell_content", request()),
            ("preview_recipe", proposal.recipe.model_dump()),
            ("submit_analysis", proposal.model_dump()),
            ("profile_data", {}),
            ("submit_analysis", proposal.model_dump()),
        ]
    )
    _, trace = analyze_intake(session, client)
    assert not trace[2]["ok"] and "读取数据画像" in trace[2]["error"]
    assert trace[-1]["ok"]
