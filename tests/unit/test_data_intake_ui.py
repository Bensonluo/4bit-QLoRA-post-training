"""UI flows run without GPU/MLflow and use the same production intake service."""

import io
import json
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest

from src.workbench.intake_service import IntakeService, next_action
from tests.unit.test_data_intake import CSV, analysis, model_for

PAGE = Path(__file__).resolve().parents[2] / "ui/pages/07_Data_Intake.py"


@pytest.fixture()
def data_page(tmp_path, monkeypatch):
    import ui.config

    for name in ("PROVIDER", "BASE_URL", "MODEL", "API_KEY"):
        monkeypatch.delenv(f"TUNESMITH_AGENT_{name}", raising=False)
    monkeypatch.setattr(ui.config, "PROJECT_ROOT", tmp_path)
    service = IntakeService(tmp_path / "outputs/workbench/intake")
    session = service.create("根据客户首次描述预测类别", "工单.csv", CSV)
    return service, session, AppTest.from_file(str(PAGE), default_timeout=20)


def button(page, label):
    return next(b for b in page.button if b.label == label)


def test_open_task_analyze_confirm_and_return_to_new(data_page, monkeypatch):
    import src.agent.intake

    service, session, page = data_page
    monkeypatch.setattr(
        src.agent.intake, "CompatibleChatClient", lambda *a, **kw: model_for(analysis())
    )
    page.run()
    assert not page.exception
    assert any(area.label == "希望模型完成什么业务工作？" for area in page.text_area)
    page.selectbox(key="intake_select").select(session.session_id).run()
    assert not page.exception
    button(page, "联合分析目标与数据").click().run()
    assert not page.exception
    assert any(s.value == "真实转换预览" for s in page.subheader)
    assert len(page.metric) == 4
    assert button(page, "确认当前转换含义").disabled
    next(c for c in page.checkbox if c.label.startswith("已核对预览")).check().run()
    button(page, "确认当前转换含义").click().run()
    assert not page.exception
    assert next_action(service.load(session.session_id)) == "awaiting_full_data"
    assert any("尚未认定可以正式训练" in message.value for message in page.success)
    button(page, "新建数据任务").click().run()
    assert not page.exception
    assert any(area.label == "希望模型完成什么业务工作？" for area in page.text_area)


def test_business_question_answer_survives_analysis_and_clears_widget(data_page, monkeypatch):
    import src.agent.intake

    service, session, page = data_page
    pending = analysis(
        recipe=None,
        questions=[
            {"question_id": "q", "question": "类别由谁审核？", "why": "需要可信的监督答案来源。"}
        ],
    )
    service.apply_analysis(session, pending)
    monkeypatch.setattr(
        src.agent.intake, "CompatibleChatClient", lambda *a, **kw: model_for(analysis())
    )
    page.run()
    page.selectbox(key="intake_select").select(session.session_id).run()
    assert any("类别由谁审核" in w.value for w in page.warning)
    next(t for t in page.text_area if t.label == "回答问题或修正理解").input(
        "类别由专门质检人员审核。"
    )
    button(page, "根据补充说明重新分析").click().run()
    assert not page.exception
    assert service.load(session.session_id).answers[-1]["answer"] == "类别由专门质检人员审核。"
    assert next(t for t in page.text_area if t.label == "回答问题或修正理解").value == ""


def test_remote_service_requires_data_consent_in_ui(data_page):
    service, session, page = data_page
    page.run()
    page.selectbox(key="intake_select").select(session.session_id).run()
    next(t for t in page.text_input if t.label == "模型服务 API 地址").input(
        "https://example.test/v1"
    ).run()
    next(t for t in page.text_input if t.label == "支持工具调用的模型名称").input("model")
    button(page, "联合分析目标与数据").click().run()
    assert not page.exception
    assert any("需先允许" in error.value for error in page.error)
    assert service.load(session.session_id).analysis is None


def probe_client(monkeypatch):
    import src.agent.intake

    requests = []

    class Probe:
        def __init__(self, base_url, model, api_key, **kwargs):
            requests.append({"base_url": base_url, "model": model, "api_key": api_key, **kwargs})

        def complete(self, messages, tools):
            requests[-1]["messages"] = messages
            return {
                "tool_calls": [
                    {"function": {"name": "connection_check", "arguments": '{"ok": true}'}}
                ]
            }

    monkeypatch.setattr(src.agent.intake, "CompatibleChatClient", Probe)
    return requests


def test_configure_before_task_and_save_without_credentials(data_page, monkeypatch, tmp_path):
    _, _, page = data_page
    requests = probe_client(monkeypatch)
    page.run()
    page.selectbox(key="agent_provider").select("glm-coding").run()
    assert not page.exception
    assert "coding" in page.text_input(key="agent_base_url").value
    page.text_input(key="agent_api_key").input("test-session-secret").run()
    button(page, "测试模型连接").click().run()
    assert requests[-1]["api_key"] == "test-session-secret"
    assert "客户" not in json.dumps(requests[-1]["messages"], ensure_ascii=False)
    assert not page.checkbox(key="agent_remote_consent").value
    button(page, "保存模型配置").click().run()
    path = tmp_path / "outputs/workbench/agent-settings.json"
    saved = json.loads(path.read_text())
    assert set(saved) == {"provider", "base_url", "model"}
    assert "test-session-secret" not in path.read_text()
    reopened = AppTest.from_file(str(PAGE), default_timeout=20).run()
    assert not reopened.exception
    assert reopened.selectbox(key="agent_provider").value == "glm-coding"
    assert reopened.text_input(key="agent_api_key").value == ""
    assert not reopened.checkbox(key="agent_remote_consent").value


def test_switch_provider_and_endpoint_clear_key_and_consent(data_page, monkeypatch):
    _, _, page = data_page
    requests = probe_client(monkeypatch)
    page.run()
    page.selectbox(key="agent_provider").select("glm-coding").run()
    page.text_input(key="agent_api_key").input("first-provider-secret").run()
    page.checkbox(key="agent_remote_consent").check().run()
    page.selectbox(key="agent_provider").select("glm").run()
    assert page.text_input(key="agent_api_key").value == ""
    assert not page.checkbox(key="agent_remote_consent").value
    button(page, "测试模型连接").click().run()
    assert requests[-1]["api_key"] == ""
    page.text_input(key="agent_api_key").input("second-provider-secret").run()
    page.checkbox(key="agent_remote_consent").check().run()
    page.text_input(key="agent_base_url").input("https://different.example/v1").run()
    assert page.text_input(key="agent_api_key").value == ""
    assert not page.checkbox(key="agent_remote_consent").value
    button(page, "测试模型连接").click().run()
    assert requests[-1]["api_key"] == ""
    assert not page.exception


def test_environment_key_is_bound_to_initial_provider_and_endpoint(data_page, monkeypatch):
    _, _, page = data_page
    monkeypatch.setenv("TUNESMITH_AGENT_PROVIDER", "glm-coding")
    monkeypatch.setenv("TUNESMITH_AGENT_API_KEY", "test-environment-secret")
    requests = probe_client(monkeypatch)
    page.run()
    assert page.text_input(key="agent_api_key").value == ""
    button(page, "测试模型连接").click().run()
    assert requests[-1]["api_key"] == "test-environment-secret"
    page.text_input(key="agent_base_url").input("https://another.example/v1").run()
    button(page, "测试模型连接").click().run()
    assert requests[-1]["api_key"] == ""
    page.selectbox(key="agent_provider").select("glm").run()
    button(page, "测试模型连接").click().run()
    assert requests[-1]["api_key"] == ""
    assert not page.exception


def upload_full_file(monkeypatch, contents):
    import streamlit

    original = streamlit.file_uploader
    uploaded = io.BytesIO(contents)
    uploaded.name = "full.csv"

    def uploader(label, *args, **kwargs):
        return uploaded if label == "提供本次任务的全量文件" else original(label, *args, **kwargs)

    monkeypatch.setattr(streamlit, "file_uploader", uploader)


def test_full_upload_review_and_confirm(data_page, monkeypatch):
    from tests.unit.test_full_data import FULL

    service, session, page = data_page
    session = service.apply_analysis(session, analysis())
    session = service.confirm(session.session_id, session.revision)
    upload_full_file(monkeypatch, FULL)
    page.run()
    page.selectbox(key="intake_select").select(session.session_id).run()
    button(page, "按已确认方案验证全量数据").click().run()
    assert not page.exception
    current = service.load(session.session_id)
    assert next_action(current) == "review_full_data"
    assert current.source == session.source
    assert button(page, "确认全量数据含义").disabled
    next(c for c in page.checkbox if c.label.startswith("已核对全量报告")).check().run()
    button(page, "确认全量数据含义").click().run()
    assert next_action(service.load(session.session_id)) == "awaiting_dataset_split"
    assert any("独立训练与评测分区" in message.value for message in page.success)


def test_full_upload_blockers_cannot_be_confirmed(data_page, monkeypatch):
    service, session, page = data_page
    session = service.apply_analysis(session, analysis())
    session = service.confirm(session.session_id, session.revision)
    upload_full_file(monkeypatch, "编号,客户描述,类别,处理结果\n10,破损,,补发\n".encode())
    page.run()
    page.selectbox(key="intake_select").select(session.session_id).run()
    button(page, "按已确认方案验证全量数据").click().run()
    assert not page.exception
    assert next_action(service.load(session.session_id)) == "needs_full_data_revision"
    assert any("缺少监督答案" in message.value for message in page.error)
    assert not any(b.label == "确认全量数据含义" for b in page.button)


def test_full_initial_source_reuse_and_business_feedback_invalidation(data_page):
    service, _, page = data_page
    session = service.create("分类", "full.csv", CSV, scope="full")
    session = service.apply_analysis(session, analysis())
    session = service.confirm(session.session_id, session.revision)
    page.run()
    page.selectbox(key="intake_select").select(session.session_id).run()
    button(page, "验证首次上传的全量文件").click().run()
    assert next_action(service.load(session.session_id)) == "review_full_data"
    next(t for t in page.text_area if t.label == "回答问题或修正理解").input(
        "类别含义需要修正"
    ).run()
    button(page, "保存业务补充，稍后分析").click().run()
    assert not page.exception
    assert service.load(session.session_id).full_data.status == "stale"
    assert any("全量报告已失效" in message.value for message in page.warning)
    assert not any(b.label == "确认全量数据含义" for b in page.button)


@pytest.mark.parametrize("groups", [["编号"], []])
def test_materialize_actual_partitions_after_full_confirmation(data_page, groups):
    from tests.unit.test_full_data import FULL, approved

    service, _, page = data_page
    session = approved(service, group_columns=groups)
    session = service.validate_full_data(session.session_id, session.revision, "full.csv", FULL)
    session = service.confirm_full_data(session.session_id, session.revision)
    page.run()
    page.selectbox(key="intake_select").select(session.session_id).run()
    if not groups:
        assert button(page, "生成数据集版本").disabled
        next(c for c in page.checkbox if c.label.startswith("已确认每行是独立")).check().run()
    button(page, "生成数据集版本").click().run()
    assert not page.exception
    current = service.load(session.session_id)
    assert next_action(current) == "ready_for_training_preflight"
    assert current.dataset.statistics["row_counts"] == {"train": 1, "validation": 1, "test": 1}
    assert current.dataset.data_config["validation_split"] == 0
    assert any("独立数据分区已生成" in message.value for message in page.success)


def test_add_original_source_preserves_business_description(data_page, monkeypatch):
    import streamlit

    service, session, page = data_page
    original = streamlit.file_uploader
    upload = io.BytesIO("编号,审核类别\n001,质量\n".encode())
    upload.name = "labels.csv"
    monkeypatch.setattr(
        streamlit,
        "file_uploader",
        lambda label, *a, **kw: (
            upload if label == "上传补充原始资料" else original(label, *a, **kw)
        ),
    )
    page.run()
    page.selectbox(key="intake_select").select(session.session_id).run()
    next(t for t in page.text_input if t.label == "补充资料名称").input("labels")
    next(t for t in page.text_area if t.label == "这份资料的用途和关联关系").input(
        "人工审核标签，通过编号关联。"
    )
    button(page, "保存补充资料").click().run()
    assert not page.exception
    saved = service.load(session.session_id)
    assert set(saved.sources) == {"main", "labels"}
    assert saved.sources["main"].digest == session.source.digest
    assert "人工审核标签" in saved.answers[-1]["answer"]


def test_combined_sources_show_lineage_and_accept_separate_full_uploads(data_page, monkeypatch):
    import streamlit

    from tests.unit.test_multisource_cli import FULL_LABELS, FULL_MAIN, composition_session

    service, _, page = data_page
    session = composition_session(service)
    original = streamlit.file_uploader
    uploads = {}
    for alias, contents in (("main", FULL_MAIN), ("labels", FULL_LABELS)):
        value = io.BytesIO(contents)
        value.name = f"{alias}.csv"
        uploads[f"全量原始资料：{alias}"] = value
    monkeypatch.setattr(
        streamlit,
        "file_uploader",
        lambda label, *a, **kw: uploads[label] if label in uploads else original(label, *a, **kw),
    )
    page.run()
    page.selectbox(key="intake_select").select(session.session_id).run()
    assert not page.exception
    assert any(item.value == "资料组合与处理结果" for item in page.subheader)
    assert not any(b.label == "验证首次上传的全量文件" for b in page.button)
    button(page, "按组合方案验证全部全量资料").click().run()
    assert not page.exception
    saved = service.load(session.session_id)
    assert set(saved.full_data.sources) == {"main", "labels"}
    assert saved.full_data.preview.counts["ready"] == 3
    assert any(item.label == "全量资料组合过程与原始来源" for item in page.expander)


def test_preflight_only_loads_tokenizer_on_explicit_button_and_shows_row_failures(
    data_page, monkeypatch
):
    import src.workbench.training_preflight as preflight
    from tests.unit.test_full_data import FULL, approved

    service, _, page = data_page
    session = approved(service)
    session = service.validate_full_data(session.session_id, session.revision, "full.csv", FULL)
    session = service.confirm_full_data(session.session_id, session.revision)
    session = service.materialize_dataset(session.session_id, session.revision)
    calls = []
    token = object()

    def load(path, *, local_files_only):
        calls.append((path, local_files_only))
        return token

    def report(current, tokenizer, max_length):
        assert tokenizer is token
        assert max_length == 6
        return {
            "status": "blocked",
            "scope_note": "UI protocol fixture",
            "issues": [
                {
                    "code": "answer_lost",
                    "severity": "blocking",
                    "message": "答案被截断",
                    "split": "train",
                    "row_ids": ["r000001"],
                }
            ],
            "splits": {"train": {"answer_lost_rows": 1}},
            "rows": [{"row_id": "r000001", "answer_supervised_tokens": 0}],
            "tokenizer": {"name_or_path": "local-fixture"},
        }

    monkeypatch.setattr(preflight, "load_local_tokenizer", load)
    monkeypatch.setattr(preflight, "preflight_dataset", report)
    page.run()
    page.selectbox(key="intake_select").select(session.session_id).run()
    assert calls == []
    next(t for t in page.text_input if t.label == "本地 tokenizer 目录或已缓存标识").input(
        "/tmp/local-fixture"
    )
    next(n for n in page.number_input if n.label == "训练最大 token 长度").set_value(6)
    button(page, "检查实际截断与答案保留").click().run()
    assert not page.exception
    assert calls == [("/tmp/local-fixture", True)]
    assert any("答案被截断" in entry.value for entry in page.error)
    assert any("r000001" in entry.value for entry in page.caption)
    page.run()
    assert len(calls) == 1


def test_adapter_evidence_shows_failed_full_validation_without_executing_source(data_page):
    from tests.unit.test_full_data import FULL, approved

    service, _, page = data_page
    session = approved(service)
    session = service.validate_full_data(session.session_id, session.revision, "full.csv", FULL)
    session.analysis.adapter = {
        "source_code": 'raise AssertionError("UI must never execute adapter source")',
        "new_columns": ["parsed_category"],
        "config": {},
        "examples": [],
    }
    session.adapter_report = {
        "validation": {
            "status": "passed",
            "backend": "macos",
            "source_digest": "code-digest",
            "cases_digest": "cases-digest",
            "limits": {"memory_enforcement": "sampled_rss"},
            "cases": [
                {"name": "真实样例", "kind": "business", "passed": True},
                {"name": "缺失字段反例", "kind": "counterexample", "passed": True},
            ],
        },
        "spec_digest": "spec-digest",
        "origins": {"r000001": [{"source_digest": session.source.digest, "row_id": "r000001"}]},
    }
    session.full_data.adapter_report = {
        "validation": {
            "status": "failed",
            "backend": "macos",
            "cases": [
                {
                    "name": "全量新增格式",
                    "kind": "business",
                    "passed": False,
                    "error": "无法解析新格式",
                }
            ],
        }
    }
    service._save(session, session.revision)
    page.run()
    page.selectbox(key="intake_select").select(session.session_id).run()
    assert not page.exception
    assert any(entry.value == "受限适配验证结果" for entry in page.subheader)
    assert any(entry.value == "全量受限适配验证结果" for entry in page.subheader)
    assert any("隔离测试未通过" in entry.value for entry in page.error)
    assert any("实际隔离后端：macos" in entry.value for entry in page.caption)
    assert any("UI must never execute" in entry.value for entry in page.code)
    rows = [frame.value for frame in page.dataframe]
    assert any("结果" in frame and "未通过" in frame["结果"].values for frame in rows)


def _preview_csv(count, *, missing_from=None):
    return (
        "编号,客户描述,类别,处理结果\n"
        + "".join(
            f"{index:03d},独立问题{index},{'' if missing_from is not None and index >= missing_from else '质量'},补发\n"
            for index in range(1, count + 1)
        )
    ).encode()


def test_sample_problem_after_twenty_rows_is_visible_and_has_concrete_label_next_step(data_page):
    service, _, page = data_page
    session = service.create("判断类别", "sample.csv", _preview_csv(25, missing_from=25))
    session = service.apply_analysis(session, analysis())
    page.run()
    page.selectbox(key="intake_select").select(session.session_id).run()
    assert not page.exception
    assert next_action(service.load(session.session_id)) == "needs_labels"
    assert any(item.label == "r000025 · 缺少答案" for item in page.expander)
    assert any("独立问题25" in item.value for item in page.code)
    assert any(
        "答案（字段：类别）" in item.value and "原始资料与补充文件" in item.value
        for item in page.info
    )
    selector = next(item for item in page.selectbox if item.label == "样例记录筛选")
    selector.select("缺少答案").run()
    row_labels = [item.label for item in page.expander if item.label.startswith("r000")]
    assert row_labels == ["r000025 · 缺少答案"]
    assert not any(item.label == "确认当前转换含义" for item in page.button)


def test_sample_pagination_confirms_only_visible_rows_and_resets_acknowledgment(data_page):
    service, _, page = data_page
    session = service.create("判断类别", "sample.csv", _preview_csv(25))
    session = service.apply_analysis(session, analysis())
    page.run()
    page.selectbox(key="intake_select").select(session.session_id).run()
    next(item for item in page.checkbox if item.label.startswith("已核对预览")).check().run()
    next(item for item in page.number_input if item.label == "样例预览页码").set_value(2).run()
    assert not page.exception
    assert button(page, "确认当前转换含义").disabled
    assert any(item.label.startswith("r000025") for item in page.expander)
    assert not any(item.label.startswith("r000001") for item in page.expander)
    next(item for item in page.checkbox if item.label.startswith("已核对预览")).check().run()
    button(page, "确认当前转换含义").click().run()
    assert not page.exception
    assert {row.row_id for row in service.load(session.session_id).confirmed_examples} == {
        f"r{index:06d}" for index in range(21, 26)
    }


def test_full_problem_pagination_reaches_all_issue_rows_and_points_to_full_reupload(data_page):
    from tests.unit.test_full_data import approved

    service, _, page = data_page
    session = approved(service)
    session = service.validate_full_data(
        session.session_id, session.revision, "full.csv", _preview_csv(45, missing_from=21)
    )
    page.run()
    page.selectbox(key="intake_select").select(session.session_id).run()
    next(item for item in page.selectbox if item.label == "全量记录筛选").select("缺少答案").run()
    next(item for item in page.number_input if item.label == "全量预览页码").set_value(2).run()
    assert not page.exception
    assert any(item.label == "全量 r000045 · 缺少答案" for item in page.expander)
    assert any("独立问题45" in item.value for item in page.code)
    assert any("全量数据验证」重新上传" in item.value for item in page.info)
    assert not any(item.label == "确认全量数据含义" for item in page.button)


SHEET_INPUT_LABEL = "Excel 工作表（留空读第一个）"


def _workbook_bytes(*sheets):
    """sheets: (名称, 表头行, 数据行) 三元组；openpyxl 现场生成多 Sheet 工作簿。"""
    from openpyxl import Workbook

    workbook = Workbook()
    for index, (name, header, rows) in enumerate(sheets):
        sheet = workbook.active if index == 0 else workbook.create_sheet()
        sheet.title = name
        sheet.append(header)
        for row in rows:
            sheet.append(row)
    buffer = io.BytesIO()
    workbook.save(buffer)
    return buffer.getvalue()


def uploaded_bytes(contents, name):
    payload = io.BytesIO(contents)
    payload.name = name
    return payload


def patch_uploader(monkeypatch, label, holder):
    """按标签替换 file_uploader 的返回值；holder["file"] 可在测试中途更换。"""
    import streamlit

    original = streamlit.file_uploader

    def uploader(rendered, *args, **kwargs):
        return holder["file"] if rendered == label else original(rendered, *args, **kwargs)

    monkeypatch.setattr(streamlit, "file_uploader", uploader)


def test_new_intake_form_reads_designated_excel_sheet(data_page, monkeypatch):
    """新建任务表单：仅 Excel 上传显示 sheet 输入，指定后读到对应工作表的数据。"""
    service, existing, page = data_page
    holder = {"file": None}
    patch_uploader(monkeypatch, "提供 CSV、Excel 或 JSONL", holder)
    page.run()
    assert not page.exception
    assert not any(t.label == SHEET_INPUT_LABEL for t in page.text_input)
    holder["file"] = uploaded_bytes(CSV, "工单.csv")
    page.run()
    assert not any(t.label == SHEET_INPUT_LABEL for t in page.text_input)
    holder["file"] = uploaded_bytes(
        _workbook_bytes(
            ("工单表", ("编号", "类别"), [("001", "质量")]),
            ("员工表", ("员工号", "部门"), [("E01", "质检")]),
        ),
        "花名册.xlsx",
    )
    page.run()
    assert not page.exception
    next(t for t in page.text_area if t.label == "希望模型完成什么业务工作？").input("统计员工部门")
    next(t for t in page.text_input if t.label == SHEET_INPUT_LABEL).input("员工表")
    button(page, "读取数据并开始").click().run()
    assert not page.exception
    created = next(s for s in service.list_sessions() if s.session_id != existing.session_id)
    loaded = service.load(created.session_id)
    assert loaded.source.sheet == "员工表"
    assert loaded.source.columns == ["员工号", "部门"]
    assert any(row.values.get("员工号") == "E01" for row in loaded.source.rows)
    assert loaded.source.sheet_note is not None and "工单表" in loaded.source.sheet_note


def test_add_source_form_reads_designated_excel_sheet(data_page, monkeypatch):
    """保存补充资料与新建任务对称：仅 Excel 上传显示 sheet 输入，指定后读到对应工作表。"""
    service, session, page = data_page
    holder = {"file": uploaded_bytes(CSV, "labels.csv")}
    patch_uploader(monkeypatch, "上传补充原始资料", holder)
    page.run()
    page.selectbox(key="intake_select").select(session.session_id).run()
    assert not page.exception
    assert not any(t.label == SHEET_INPUT_LABEL for t in page.text_input)
    holder["file"] = uploaded_bytes(
        _workbook_bytes(
            ("工单表", ("编号", "类别"), [("001", "质量")]),
            ("员工表", ("员工号", "部门"), [("E01", "质检")]),
        ),
        "labels.xlsx",
    )
    page.run()
    next(t for t in page.text_input if t.label == "补充资料名称").input("labels")
    next(t for t in page.text_input if t.label == SHEET_INPUT_LABEL).input("员工表")
    button(page, "保存补充资料").click().run()
    assert not page.exception
    saved = service.load(session.session_id)
    assert saved.sources["labels"].sheet == "员工表"
    assert saved.sources["labels"].columns == ["员工号", "部门"]
    assert any(row.values.get("员工号") == "E01" for row in saved.sources["labels"].rows)
    assert saved.sources["main"].digest == session.source.digest


def test_full_upload_form_reads_designated_excel_sheet(data_page, monkeypatch):
    """全量文件读取设置与新建任务对称：全量 Excel 数据在第二个 sheet 时按指定读取。"""
    from tests.unit.test_full_data import approved

    service, _, page = data_page
    session = approved(service)
    holder = {"file": None}
    patch_uploader(monkeypatch, "提供本次任务的全量文件", holder)
    page.run()
    page.selectbox(key="intake_select").select(session.session_id).run()
    assert not page.exception
    assert not any(t.label == SHEET_INPUT_LABEL for t in page.text_input)
    holder["file"] = uploaded_bytes(
        _workbook_bytes(
            ("说明", ("备注",), [("全量数据在下一个 sheet",)]),
            (
                "全量表",
                ("编号", "客户描述", "类别", "处理结果"),
                [
                    ("100", "收到破杯", "质量", "补发"),
                    ("101", "快递太慢", "物流", "查询"),
                    ("102", "杯把断了", "质量", "补发"),
                ],
            ),
        ),
        "全量.xlsx",
    )
    page.run()
    next(t for t in page.text_input if t.label == SHEET_INPUT_LABEL).input("全量表")
    button(page, "按已确认方案验证全量数据").click().run()
    assert not page.exception
    current = service.load(session.session_id)
    assert next_action(current) == "review_full_data"
    assert current.full_data.sources["main"].sheet == "全量表"
    assert current.full_data.source.columns == ["编号", "客户描述", "类别", "处理结果"]
    assert current.full_data.preview.counts["ready"] == 3


def _fact_note_workbook_bytes() -> bytes:
    """双 sheet + 合并区 + 隐藏行 + 全空行 + 重复表头行:一个工作簿同时触发
    sheet/merged/hidden/blank/dup_header 五条如实标注。"""
    from io import BytesIO

    from openpyxl import Workbook

    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "工单表"
    sheet.append(("编号", "客户描述", "类别"))
    for row in (
        ("001", "杯子破损", "质量"),
        (None, None, None),  # 第 3 行:全空行,照常读入为全空记录
        ("002", "物流未更新", None),  # C4:C5 合并,非首格读空
        ("003", "屏幕碎裂", None),
        ("004", "快递丢失", "物流"),
        ("编号", "客户描述", "类别"),  # 第 7 行:导出拼接产生的重复表头,照常读入为数据行
    ):
        sheet.append(row)
    sheet.merge_cells("C4:C5")
    sheet.row_dimensions[6].hidden = True  # 第 6 行(004,带有效标签)隐藏,照常读入
    staff = workbook.create_sheet("员工表")
    staff.append(("员工号", "部门"))
    staff.append(("E01", "质检"))
    buffer = BytesIO()
    workbook.save(buffer)
    return buffer.getvalue()


def _full_fact_note_workbook_bytes() -> bytes:
    """全量侧夹具:隐藏行带着有效标签(场景 44 形态),全量验证照常通过。"""
    from io import BytesIO

    from openpyxl import Workbook

    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "全量表"
    sheet.append(("编号", "客户描述", "类别", "处理结果"))
    for row in (
        ("100", "收到破杯", "质量", "补发"),
        ("101", "快递太慢", "物流", "查询"),
        ("102", "杯把断了", "质量", "补发"),
    ):
        sheet.append(row)
    sheet.row_dimensions[3].hidden = True  # 101 行隐藏——有效数据,不拦旅程
    buffer = BytesIO()
    workbook.save(buffer)
    return buffer.getvalue()


def test_excel_fact_notes_render_after_create(data_page):
    """如实标注在任务页可见:主来源的 sheet(info)/merged/hidden/blank/dup_header(warning)
    在 scope_note 下按级渲染;CSV 任务没有 Excel 事实,一条都不渲染。"""
    service, existing, page = data_page
    # 两个会话都在首次 run 前建好:AppTest 的 selectbox 选项来自上一次渲染,
    # run 之后才 create 的会话不在旧选项里,select 会静默渲染回旧会话。
    created = service.create(
        "根据客户首次描述判断售后类别", "工单.xlsx", _fact_note_workbook_bytes()
    )
    page.run()
    page.selectbox(key="intake_select").select(existing.session_id).run()
    assert not page.exception
    # CSV 任务没有 Excel 事实:Excel 标注一条都不出现(空行/重复表头属跨格式事实,
    # 由 blank_note/dup_header_note 承担,干净 CSV 不携带)
    assert not any("合并单元格" in w.value for w in page.warning)
    assert not any("隐藏行" in w.value for w in page.warning)
    assert not any("全空行" in w.value for w in page.warning)
    assert not any("与表头完全相同" in w.value for w in page.warning)
    assert not any("sheet" in i.value for i in page.info)

    page.selectbox(key="intake_select").select(created.session_id).run()
    assert not page.exception
    assert any("1 处合并单元格" in w.value for w in page.warning), [w.value for w in page.warning]
    assert any("1 个隐藏行" in w.value for w in page.warning), [w.value for w in page.warning]
    assert any("1 个全空行" in w.value for w in page.warning), [w.value for w in page.warning]
    assert any("1 行与表头完全相同" in w.value for w in page.warning), [
        w.value for w in page.warning
    ]
    assert any("仅读取第一个" in i.value for i in page.info), [i.value for i in page.info]


def test_full_report_renders_excel_fact_notes(data_page, monkeypatch):
    """全量验证报告同样渲染 Excel 事实标注:全量文件的隐藏行在来源 caption 下可见。"""
    from tests.unit.test_full_data import approved

    service, _, page = data_page
    session = approved(service)
    holder = {"file": uploaded_bytes(_full_fact_note_workbook_bytes(), "全量.xlsx")}
    patch_uploader(monkeypatch, "提供本次任务的全量文件", holder)
    page.run()
    page.selectbox(key="intake_select").select(session.session_id).run()
    assert not page.exception
    button(page, "按已确认方案验证全量数据").click().run()
    assert not page.exception
    assert next_action(service.load(session.session_id)) == "review_full_data"
    assert any("1 个隐藏行" in w.value for w in page.warning), [w.value for w in page.warning]


def test_added_source_fact_notes_listed_with_alias(data_page):
    """补充资料的 Excel 标注在「原始资料与补充文件」区按资料名前缀列出。"""
    service, session, page = data_page
    service.add_source(
        session.session_id,
        session.revision,
        "labels",
        "labels.xlsx",
        _fact_note_workbook_bytes(),
    )
    page.run()
    page.selectbox(key="intake_select").select(session.session_id).run()
    assert not page.exception
    assert any(w.value.startswith("labels：") and "1 个隐藏行" in w.value for w in page.warning), [
        w.value for w in page.warning
    ]
    assert any(w.value.startswith("labels：") and "1 个全空行" in w.value for w in page.warning), [
        w.value for w in page.warning
    ]
    assert any(
        w.value.startswith("labels：") and "1 行与表头完全相同" in w.value for w in page.warning
    ), [w.value for w in page.warning]


def test_composed_full_sources_form_reads_designated_excel_sheets(data_page, monkeypatch):
    """组合全量表单与单文件表单对称：每份 Excel 旁的 sheet 输入逐份透传。

    两份资料各指定不同 sheet，任一来源留空或读错 sheet 都无法蒙混成正确全量。
    """
    from tests.unit.test_multisource_cli import LABELS, MAIN, combined_plan

    service, _, page = data_page
    session = service.create("判断工单类别", "tickets.csv", MAIN)
    session = service.add_source(
        session.session_id, session.revision, "labels", "labels.csv", LABELS
    )
    session = service.apply_analysis(session, combined_plan())
    session = service.confirm(session.session_id, session.revision)
    holders = {"main": {"file": None}, "labels": {"file": None}}
    patch_uploader(monkeypatch, "全量原始资料：main", holders["main"])
    patch_uploader(monkeypatch, "全量原始资料：labels", holders["labels"])
    page.run()
    page.selectbox(key="intake_select").select(session.session_id).run()
    assert not page.exception
    assert not any(t.label == SHEET_INPUT_LABEL for t in page.text_input)
    holders["main"]["file"] = uploaded_bytes(
        _workbook_bytes(
            ("说明", ("备注",), [("全量数据在下一个 sheet",)]),
            (
                "工单表",
                ("ticket", "text"),
                [("10", "新杯子损坏"), ("11", "新物流延迟"), ("12", "新配件缺陷")],
            ),
        ),
        "全量工单.xlsx",
    )
    holders["labels"]["file"] = uploaded_bytes(
        _workbook_bytes(
            ("类别表", ("id", "category"), [("10", "质量"), ("11", "物流"), ("12", "质量")]),
            ("说明", ("备注",), [("这份资料的数据在第一个 sheet",)]),
        ),
        "全量类别.xlsx",
    )
    page.run()
    assert not page.exception
    main_sheet = next(
        t for t in page.text_input if t.key == f"full_sheet_{session.session_id}_main"
    )
    labels_sheet = next(
        t for t in page.text_input if t.key == f"full_sheet_{session.session_id}_labels"
    )
    main_sheet.input("工单表")
    labels_sheet.input("类别表")
    button(page, "按组合方案验证全部全量资料").click().run()
    assert not page.exception
    current = service.load(session.session_id)
    assert next_action(current) == "review_full_data"
    assert current.full_data.sources["main"].sheet == "工单表"
    assert current.full_data.sources["labels"].sheet == "类别表"
    assert current.full_data.preview.counts["ready"] == 3
