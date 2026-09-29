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


def test_analyze_cli_appends_analysis_summary_to_stderr(
    monkeypatch, capsys, tmp_path, service, session
):
    """analyze 双流契约:stdout 纯 JSON,stderr 追加与页面同词汇的发现与待确认问题摘要。"""
    import sys

    from scripts import data_intake

    result = analysis()
    monkeypatch.setattr(data_intake, "_client", lambda args, probe=False: model_for(result))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "data_intake.py",
            "--store",
            str(tmp_path / "intake"),
            "analyze",
            session.session_id,
        ],
    )
    assert data_intake.main() == 0
    captured = capsys.readouterr()
    json.loads(captured.out)  # stdout 仍是纯 JSON,人话只走 stderr
    assert "这份分析给出数据判断与待确认问题：发现 1 条、待确认问题 0 个。" in captured.err
    assert "已观察：不同问题可能都通过补发处理，处理结果不是问题类别。" in captured.err
    assert "暂定微调思路：确认标签后可考虑 SFT，规模等待全量检查。" in captured.err
    # 工具核查轨迹与评测解读同格式渲染(夹具 4 次调用全部成功,无失败子句)
    assert "工具核查轨迹：4 次调用，成功 4 次。" in captured.err
    assert "不代表业务效果达标。" in captured.err


def test_agent_product_state_tails_name_exits_and_answer_reanalyze_completes(
    monkeypatch, capsys, tmp_path, service, session
):
    """Agent 产物三态尾行点名真实出口(R86),且点名的出口给出实际走通的证明
    (R89 收口 R86 C-2):needs_business_answers 点名 analyze --answer 且一次调用
    直达 review_preview;needs_capability 点名资料侧出口(调整目标/add-source)
    而不伪造零密钥兜底,走通验证的是其中 add-source 一条——add-source 后分析
    失效回落 awaiting_analysis(add_source 清空旧分析,不伪造进度)、重新 analyze
    且缺口清除后直达 review_preview(「调整目标」是改 goal 的另一路,不在本测试
    证明面);needs_recipe 点名重新 analyze,且重新分析生成方案后同样直达
    review_preview。Agent 分析在场时 baseline-analyze 会被拒绝,后两态尾行
    不得点名它。"""
    import sys

    from scripts import data_intake

    def run_cli(*argv):
        monkeypatch.setattr(
            sys, "argv", ["data_intake.py", "--store", str(tmp_path / "intake"), *argv]
        )
        assert data_intake.main() == 0
        captured = capsys.readouterr()
        return next(line for line in captured.err.splitlines() if line.startswith("下一步状态"))

    holder = {}
    monkeypatch.setattr(data_intake, "_client", lambda args, probe=False: holder["model"])

    # leg A: 业务问题态点名 --answer 出口,且该出口一次调用实际走通。
    questioning = analysis(
        recipe=None,
        questions=[
            {
                "question_id": "q-label-source",
                "question": "类别以哪次审核为准？",
                "why": "同一工单存在两次审核记录。",
            }
        ],
    )
    holder["model"] = model_for(questioning)
    tail = run_cli("analyze", session.session_id)
    assert tail.startswith("下一步状态: needs_business_answers（"), tail
    assert "analyze --answer" in tail and "--answer '你的回答'" in tail
    assert "baseline-analyze" not in tail
    holder["model"] = model_for(analysis())
    tail = run_cli("analyze", session.session_id, "--answer", "以人工审核为准")
    assert tail.startswith("下一步状态: review_preview（"), tail

    # leg B: 能力缺口态点名资料侧出口,不把 baseline-analyze 伪造成兜底。
    holder["model"] = model_for(
        analysis(recipe=None, capability_gaps=["任务需要多源组合，当前入口未配置。"])
    )
    session_b = service.create("合并多表做预测", "工单.csv", CSV)
    tail = run_cli("analyze", session_b.session_id)
    assert tail.startswith("下一步状态: needs_capability（"), tail
    assert "add-source" in tail and "调整目标" in tail
    assert "baseline-analyze" not in tail
    # R89 收口 R86 C-2(腿 B 走通):add-source 补充资料后旧分析失效、状态回落
    # awaiting_analysis——不伪造进度;重新 analyze(缺口清除的产物)直达
    # review_preview,资料侧出口确实能清缺口走出去。不传 --description:CLI 只在
    # 描述非空时走 service.answer(data_intake.py:1472-1475,answer 也会清分析
    # intake_service.py:424-425),空描述使清分析只归因 add_source 自身
    # (:1050-1052),awaiting_analysis 归因单一(r89-reviewer nit-① 采纳)。
    # 夹具只脚本化模型产物,证明的是状态机与 CLI 面的通路,不是 LLM 判断质量
    # (ScriptedModel 同款边界)。
    labels = tmp_path / "labels.csv"
    labels.write_bytes("编号,类别\n001,质量\n002,物流\n".encode())
    session_b = service.load(session_b.session_id)
    tail = run_cli(
        "add-source",
        session_b.session_id,
        "--revision",
        str(session_b.revision),
        "--alias",
        "labels",
        "--input",
        str(labels),
    )
    assert tail.startswith("下一步状态: awaiting_analysis（"), tail
    holder["model"] = model_for(analysis())
    tail = run_cli("analyze", session_b.session_id)
    assert tail.startswith("下一步状态: review_preview（"), tail

    # leg C: 缺方案态点名重新分析;Agent 分析在场,baseline-analyze 不在出口里。
    holder["model"] = model_for(analysis(recipe=None))
    session_c = service.create("判断工单类别", "工单.csv", CSV)
    tail = run_cli("analyze", session_c.session_id)
    assert tail.startswith("下一步状态: needs_recipe（"), tail
    assert "analyze" in tail
    assert "baseline-analyze" not in tail
    # R89 收口 R86 C-2(腿 C 走通):重新 analyze 生成方案,直达 review_preview。
    holder["model"] = model_for(analysis())
    tail = run_cli("analyze", session_c.session_id)
    assert tail.startswith("下一步状态: review_preview（"), tail


def test_materialize_cli_prints_split_guidance_and_small_test_caution(
    monkeypatch, capsys, tmp_path, service, session
):
    """materialize 双流契约:分区设置引导先于落盘,小测试集提醒由 summarize_dataset
    单一来源产出;stdout 仍是纯 JSON。"""
    import sys

    from scripts import data_intake

    prepared = service.apply_analysis(session, analysis())
    prepared = service.confirm(prepared.session_id, prepared.revision)
    full_csv = (
        "编号,客户描述,类别,处理结果\n"
        + "".join(f"{i:03d},描述{i},质量,补发\n" for i in range(1, 11))
    ).encode()
    prepared = service.validate_full_data(
        prepared.session_id, prepared.revision, "full.csv", full_csv
    )
    prepared = service.confirm_full_data(prepared.session_id, prepared.revision)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "data_intake.py",
            "--store",
            str(service.root),
            "materialize",
            prepared.session_id,
            "--revision",
            str(prepared.revision),
            "--registry-root",
            str(tmp_path / "registry"),
        ],
    )
    assert data_intake.main() == 0
    captured = capsys.readouterr()
    json.loads(captured.out)  # stdout 仍是纯 JSON,人话只走 stderr
    # 分区设置「何时该改」引导:与页面分区设置区同源(split_settings_guidance_lines)。
    assert "验证集比例＝" in captured.err
    # 夹具 10 个独立分组按默认比例切出独立测试集 1 条(<30):小测试集提醒行如实出现。
    assert "独立测试集共 1 条" in captured.err
    assert "少于 30 条时结论偶然性大" in captured.err


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


def test_next_action_phrase_translates_every_state_without_fabricating():
    """下一步状态人话对照:13 态全覆盖、与页面词汇同源、未知态不编造。"""
    from src.workbench.intake_service import _NEXT_ACTION_PHRASES, next_action_phrase

    every_state = {
        "awaiting_analysis",
        "needs_business_answers",
        "needs_capability",
        "needs_recipe",
        "needs_data_revision",
        "needs_labels",
        "review_preview",
        "review_full_data",
        "needs_full_data_revision",
        "ready_for_training_preflight",
        "awaiting_dataset_split",
        "awaiting_full_data",
        "awaiting_full_validation",
    }
    # 对照表恰好覆盖 next_action 可能返回的全部状态:多一个是死键,少一个是漏翻译。
    assert set(_NEXT_ACTION_PHRASES) == every_state
    for status in every_state:
        phrase = next_action_phrase(status)
        assert phrase and not phrase.isascii(), f"状态 {status} 必须有人话翻译"
    # 页面词汇同源锚点:这几句与 07_Data_Intake.py 的提示语逐字一致方向。
    assert next_action_phrase("awaiting_analysis") == (
        "尚未分析：配置了 Agent 运行 analyze；没有 Agent 服务用 baseline-analyze 零密钥开始。"
    ), "零密钥用户的尾行必须点名 baseline-analyze,不能只指需密钥的 analyze"
    assert next_action_phrase("needs_business_answers") == (
        "Agent 有业务问题待回答：运行 analyze --answer '你的回答'，一次命令完成回答与重新分析；"
        "阻断确认的问题全部解除后才能确认方案。"
    ), "回答出口必须点名(R86):needs_business_answers 不能只说待回答不给命令"
    assert next_action_phrase("needs_capability") == (
        "Agent 记录了能力缺口，当前资料做不了这个任务：调整目标或用 add-source 补充资料后"
        "运行 analyze 重新分析；重跑同一命令不能消除缺口。"
    ), "能力缺口态必须给资料侧出口且不伪造零密钥兜底(R86)"
    assert next_action_phrase("needs_recipe") == (
        "还没有转换方案：运行 analyze 重新分析生成处理规则；Agent 再提出业务问题时用 --answer 回答。"
    ), "缺方案态必须点名重新分析(R86):Agent 分析在场时 baseline-analyze 会被拒绝,不点名它"
    assert next_action_phrase("needs_data_revision") == (
        "转换存在异常或同输入答案冲突：查看问题行后重新分析——配置了 Agent 运行 analyze；"
        "此前的基础分析可调整字段重跑 baseline-analyze（零密钥）。"
    ), "零密钥死胡同根治(R82):重分析指引必须点名基础分析可重跑的出口"
    assert next_action_phrase("needs_labels") == (
        "样例缺少可学习的答案：先用 answer-sheet 导出待补清单（零密钥）交填写人补齐，"
        "替换原文件后重新分析——配置了 Agent 运行 analyze；基础分析可调整字段重跑 baseline-analyze。"
    ), "零密钥修复工具链必须点名(R83):needs_labels 不只是提醒,还要给出出口"
    assert next_action_phrase("review_preview") == (
        "样例转换含义待确认：核对预览，用 contrast-check 配对、contrast-check-submit 提交"
        "（二连对），之后运行 confirm 确认。"
    ), "配对工具链必须点名(R84):review_preview 不能只说完成对比核验不给命令入口"
    assert next_action_phrase("awaiting_full_data") == (
        "样例转换含义已确认，提供全量文件并运行 full-validate 验证覆盖、冲突与独立分组"
        "（多资料任务用 full-sources）。"
    ), "全量验证入口必须点名(R85):awaiting_full_data 不能只说提供全量数据不给命令"
    assert next_action_phrase("awaiting_full_validation") == (
        "转换含义已确认，运行 full-validate 完成全量业务质量、分区与训练消费检查"
        "（已声明全量可省略 --input）。"
    ), "已声明全量任务的验证入口必须点名(R85),省略 --input 的复用条件一并写明"
    assert next_action_phrase("needs_full_data_revision") == (
        "全量报告仍有阻断问题，修正资料或规则后重跑 full-validate（多资料任务用 full-sources）。"
    ), "阻断态的重验出口必须点名(R85):修正后不是没有下一步的重验"
    assert next_action_phrase("review_full_data") == (
        "全量报告待核对：核对覆盖与问题处理后运行 full-confirm 确认。"
    ), "全量确认命令必须点名(R85):review_full_data 不能只说确认后继续"
    assert next_action_phrase("awaiting_dataset_split") == (
        "可以准备独立训练与评测分区（运行 materialize；尚未认定可以正式训练）。"
    )
    assert next_action_phrase("ready_for_training_preflight") == (
        "数据已就绪，可运行 preflight 做训练前检查（尚未开始训练）。"
    )
    assert next_action_phrase("not_a_real_state") == "", "未知状态不编造人话"
