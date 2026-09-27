"""The sample-to-plan contract: factual evidence, real previews, and revisions."""

from __future__ import annotations

import json
import sqlite3

import pytest

from src.agent.intake import CompatibleChatClient, analyze_intake
from src.workbench.intake_models import DataRecipe, IntakeAnalysis
from src.workbench.intake_service import IntakeService, next_action
from src.workbench.recipes import preview_recipe, validate_analysis
from src.workbench.sources import profile_source, read_source

CSV = "编号,客户描述,类别,处理结果\n001,杯子破损,质量,补发\n002,物流未更新,物流,补发\n".encode()


def test_structured_cell_does_not_silently_overwrite_duplicate_keys():
    source = read_source(
        "records.jsonl",
        json.dumps(
            {"prompt": "抽取条款", "answer": '{"金额": 10, "金额": 20}'}, ensure_ascii=False
        ).encode(),
    )
    parsed = DataRecipe(
        instruction="抽取金额",
        inputs=[{"column": "prompt", "label": "条款"}],
        targets=[
            {"column": "answer", "label": "答案", "transforms": [{"operation": "parse_json"}]}
        ],
        output_format="json",
    )
    preview = preview_recipe(source, parsed)
    assert preview.counts["invalid"] == 1
    assert "重复字段" in preview.rows[0].issues[0]
    assert preview.rows[0].original["answer"] == '{"金额": 10, "金额": 20}'


def recipe(**updates):
    values = {
        "instruction": "按客户首次描述判断问题类别。",
        "inputs": [{"column": "客户描述", "label": "客户描述"}],
        "targets": [{"column": "类别", "label": "类别"}],
        "group_columns": ["编号"],
    }
    values.update(updates)
    return DataRecipe.model_validate(values)


def analysis(**updates):
    values = {
        "task": {
            "goal": "判断售后问题类型",
            "usage_input": "客户首次描述",
            "desired_output": "问题类别",
            "row_meaning": "一行一个工单",
            "supervision_source": "用户确认类别为人工审核标签",
            "success_criteria": ["问题类别与人工审核一致"],
            "field_roles": [
                {"column": "编号", "role": "group", "reason": "工单标识"},
                {
                    "column": "客户描述",
                    "role": "input",
                    "reason": "首次咨询时可得",
                    "available_at_prediction": True,
                },
                {"column": "类别", "role": "target", "reason": "人工标签"},
                {
                    "column": "处理结果",
                    "role": "unused",
                    "reason": "事后结果，不是问题类别",
                    "available_at_prediction": False,
                },
            ],
        },
        "findings": [
            {
                "kind": "observed",
                "message": "不同问题可能都通过补发处理，处理结果不是问题类别。",
                "evidence_row_ids": ["r000001", "r000002"],
            }
        ],
        "recipe": recipe().model_dump(),
        "training_approach": "确认标签后可考虑 SFT，规模等待全量检查。",
        "next_steps": ["核对样例转换，随后提供全量数据。"],
    }
    values.update(updates)
    return IntakeAnalysis.model_validate(values)


@pytest.fixture()
def service(tmp_path):
    return IntakeService(tmp_path / "intake")


@pytest.fixture()
def session(service):
    return service.create("根据客户首次描述预测类别", "工单.csv", CSV)


class ScriptedModel:
    """Protocol fixture; not evidence of actual LLM business reasoning quality."""

    model = "protocol-test-fixture"

    def __init__(self, steps):
        self.steps = iter(steps)
        self.seen = []

    def complete(self, messages, tools):
        self.seen.append(list(messages))
        name, args = next(self.steps)
        return {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": f"call-{len(self.seen)}",
                    "type": "function",
                    "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)},
                }
            ],
        }


def model_for(result):
    steps = [("profile_data", {}), ("inspect_rows", {"row_ids": []})]
    if result.recipe:
        steps.append(("preview_recipe", result.recipe.model_dump()))
    steps.append(("submit_analysis", result.model_dump()))
    return ScriptedModel(steps)


def test_csv_preserves_codes_whitespace_null_strings_and_multiline():
    source = read_source("s.csv", '编号;描述;答案\n001;" 首行\n第二行 ";NULL\n002;;0\n'.encode())
    assert source.delimiter == ";"
    assert source.rows[0].values == {"编号": "001", "描述": " 首行\n第二行 ", "答案": "NULL"}
    assert source.rows[1].line == 4
    profile = profile_source(source)
    assert profile["columns"]["描述"]["missing_count"] == 1
    assert profile["columns"]["答案"]["missing_count"] == 0
    assert profile["columns"]["编号"]["leading_zero_examples"] == ["001", "002"]
    assert profile["source_scope"] == "sample"


@pytest.mark.parametrize(
    "data", [b"a,a\n1,2\n", b"a,b\n1,2,3\n", b'a,b\n"unterminated,2\n', b"a,b\n"]
)
def test_bad_csv_is_not_silently_repaired(data):
    with pytest.raises(ValueError):
        read_source("s.csv", data)


def test_encoding_and_json_structures_preserved():
    source = read_source("s.csv", "编号,描述\n001,测试\n".encode("gb18030"))
    assert source.rows[0].values["描述"] == "测试"
    source = read_source(
        "s.jsonl",
        b'{"messages":[{"role":"user","content":"hi"}],"label":null}\n{"messages":[],"extra":false}\n',
    )
    assert isinstance(source.rows[0].values["messages"], list)
    assert source.rows[1].values["extra"] is False
    assert profile_source(source)["columns"]["label"]["missing_count"] == 2


def test_utf16_with_bom_auto_detected_and_bomless_rejected():
    """Excel「Unicode 文本」导出(UTF-16 BOM + Tab 分隔)自动识别;无 BOM 不猜。"""
    text = "编号\t客户描述\t类别\n001\t杯子破损\t质量\n002\t物流未更新\t物流\n"
    source = read_source("工单.csv", text.encode("utf-16"))
    assert source.encoding == "utf-16"
    assert source.delimiter == "\t"
    assert source.rows[0].values == {"编号": "001", "客户描述": "杯子破损", "类别": "质量"}
    # UTF-16 BE 同样按 BOM 识别(手动前置 FE FF,等价于 BE 平台的 BOM)
    assert (
        read_source("工单.csv", b"\xfe\xff" + text.encode("utf-16-be")).rows[1].values["编号"]
        == "002"
    )
    # 无 BOM 的 UTF-16 不猜:明确拒绝而不是产出乱码
    with pytest.raises(ValueError, match="无法解码"):
        read_source("工单.csv", text.encode("utf-16-le"))


@pytest.mark.parametrize("data", [b'{"a":1,"a":2}\n', b'{"a":NaN}\n', b"[]\n"])
def test_ambiguous_or_nonfinite_json_rejected(data):
    with pytest.raises(ValueError):
        read_source("s.jsonl", data)


def test_preview_uses_real_rows_and_keeps_group_out_of_input(session):
    preview = preview_recipe(session.source, recipe())
    assert preview.rows[0].input == "客户描述: 杯子破损"
    assert preview.rows[0].target == "质量"
    assert preview.rows[0].group == {"编号": "001"}
    assert "补发" not in preview.rows[0].input
    assert preview.counts["ready"] == 2


def test_unknown_mapping_and_missing_targets_remain_visible():
    source = read_source("s.csv", "编号,客户描述,类别,处理结果\n1, A ,X,R\n2,B,,R\n".encode())
    config = recipe(
        targets=[
            {
                "column": "类别",
                "label": "类别",
                "transforms": [{"operation": "map_values", "mapping": {"Y": "已知"}}],
            }
        ]
    )
    result = preview_recipe(source, config)
    assert result.rows[0].status == "invalid"
    assert result.rows[1].status == "needs_label"
    assert len(result.rows) == 2
    assert source.rows[0].values["客户描述"] == " A "


def test_same_input_conflicts_unless_open_task_explicitly_allows_alternatives():
    source = read_source("s.csv", "编号,客户描述,类别\n1,A,X\n2,A,Y\n".encode())
    assert preview_recipe(source, recipe()).counts["conflict"] == 2
    with pytest.raises(ValueError):
        recipe(allow_multiple_targets=True)
    open_task = recipe(
        allow_multiple_targets=True,
        multiple_targets_reason="创作任务允许多种有效表达，已由业务确认。",
    )
    assert preview_recipe(source, open_task).counts["ready"] == 2


def test_json_output_renders_actual_nested_values():
    source = read_source("s.jsonl", b'{"text":"hello","fields":{"name":"A"}}\n')
    config = DataRecipe(
        instruction="extract",
        inputs=[{"column": "text", "label": "text"}],
        targets=[{"column": "fields", "label": "fields"}],
        output_format="json",
    )
    assert json.loads(preview_recipe(source, config).rows[0].target) == {"fields": {"name": "A"}}


def test_role_mismatch_future_information_and_fake_evidence_rejected(session):
    bad = analysis()
    bad.task.field_roles[1].available_at_prediction = False
    with pytest.raises(ValueError, match="预测时"):
        validate_analysis(session.source, bad)
    bad = analysis()
    bad.findings[0].evidence_row_ids = ["invented"]
    with pytest.raises(ValueError, match="证据行"):
        validate_analysis(session.source, bad)
    with pytest.raises(ValueError, match="不存在"):
        preview_recipe(session.source, recipe(inputs=[{"column": "made_up", "label": "x"}]))


def test_agent_checks_evidence_and_executes_preview_before_acceptance(service, session):
    client = model_for(analysis())
    result = service.analyze(session.session_id, client)
    assert result.preview.rows[0].target == "质量"
    assert [t["tool"] for t in result.tool_trace] == [
        "profile_data",
        "inspect_rows",
        "preview_recipe",
        "submit_analysis",
    ]
    assert next_action(result) == "review_preview"
    confirmed = service.confirm(result.session_id, result.revision, ["r000001"])
    assert next_action(confirmed) == "awaiting_full_data"
    assert len(confirmed.confirmed_examples) == 1
    reopened = IntakeService(service.root).load(result.session_id)
    assert reopened.model_dump() == confirmed.model_dump()


def test_declared_business_group_cannot_be_omitted_from_executable_recipe(session):
    proposal = analysis()
    proposal.recipe.group_columns = []
    with pytest.raises(ValueError, match="recipe.group_columns"):
        validate_analysis(session.source, proposal)
    proposal.recipe.group_columns = ["编号"]
    validate_analysis(session.source, proposal)


def test_agent_cannot_skip_real_preview(session):
    final = analysis()
    client = ScriptedModel(
        [
            ("profile_data", {}),
            ("inspect_rows", {"row_ids": []}),
            ("submit_analysis", final.model_dump()),
            ("preview_recipe", final.recipe.model_dump()),
            ("submit_analysis", final.model_dump()),
        ]
    )
    result, trace = analyze_intake(session, client)
    assert result.task.desired_output == "问题类别"
    assert trace[2]["ok"] is False
    assert "真实预览" in trace[2]["error"]


def test_agent_recovers_from_qualified_ids_in_current_row_evidence(session):
    final = analysis()
    bad = final.model_copy(deep=True)
    qualified = f"source:{session.source.digest}:r000001"
    bad.task.field_roles[0].evidence_row_ids = [qualified]
    client = ScriptedModel(
        [
            ("inspect_source", {"alias": "main"}),
            ("profile_data", {}),
            ("inspect_rows", {"row_ids": [qualified]}),
            ("inspect_rows", {"row_ids": []}),
            ("preview_recipe", final.recipe.model_dump()),
            ("submit_analysis", bad.model_dump()),
            ("submit_analysis", final.model_dump()),
        ]
    )
    result, trace = analyze_intake(session, client)
    assert result == final
    assert not trace[2]["ok"]
    assert "row_ids=[]" in trace[2]["error"]
    assert not trace[5]["ok"]
    assert "r000001" in trace[5]["error"]
    source_feedback = json.loads(client.seen[1][-1]["content"])
    assert source_feedback["rows"][0]["local_row_id"] == "r000001"
    assert source_feedback["rows"][0]["row_id"] == qualified


def test_missing_labels_and_business_ambiguity_do_not_become_ready(service, session):
    missing = analysis(
        recipe=recipe(targets=[]).model_dump(), next_steps=["补充人工问题类别标签。"]
    )
    missing.task.field_roles[2].role = "unknown"
    result = service.analyze(session.session_id, model_for(missing))
    assert next_action(result) == "needs_labels"
    assert result.preview.counts["needs_label"] == 2
    with pytest.raises(ValueError):
        service.confirm(result.session_id, result.revision)
    question = analysis(
        recipe=None,
        questions=[
            {
                "question_id": "q1",
                "question": "你希望预测问题类别，还是处理结果？",
                "why": "处理结果不能直接代表类别。",
            }
        ],
    )
    result = service.analyze(session.session_id, model_for(question))
    assert next_action(result) == "needs_business_answers"
    answered = service.answer(result.session_id, "类别才是目标，来自人工审核；处理结果不作输入。")
    assert answered.analysis is None
    assert answered.answers[-1]["question"] == question.questions[0].question
    result = service.analyze(session.session_id, model_for(analysis()))
    assert next_action(result) == "review_preview"


def test_revisions_invalidate_confirmation_and_reject_stale_write(service, session):
    initial = service.analyze(session.session_id, model_for(analysis()))
    service.confirm(initial.session_id, initial.revision)
    revised = service.answer(initial.session_id, "请重新核实标签。")
    assert revised.confirmed_revision is None and revised.preview is None
    with pytest.raises(ValueError, match="已更新"):
        service.apply_analysis(initial, analysis())
    with sqlite3.connect(service.database) as connection:
        assert connection.execute("SELECT count(*) FROM revisions").fetchone()[0] == 4


def test_remote_data_requires_authorization_and_url_has_no_credentials():
    with pytest.raises(ValueError, match="发送"):
        CompatibleChatClient("https://example.test/v1", "model")
    with pytest.raises(ValueError, match="凭据"):
        CompatibleChatClient("https://secret@example.test/v1", "model", allow_remote=True)
    assert CompatibleChatClient("http://127.0.0.1:11434/v1", "local").model == "local"


def test_capability_gap_can_be_saved_without_fabricated_preview(service, session):
    result = service.analyze(
        session.session_id,
        model_for(analysis(recipe=None, capability_gaps=["需要多表关联，当前入口尚未实现。"])),
    )
    assert next_action(result) == "needs_capability"
    assert result.preview is None
