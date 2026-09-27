"""A new user without any Agent key can start the core path via baseline analysis."""

import pytest

pytest.importorskip("streamlit")

from tests.unit import test_data_intake_ui as intake_ui

data_page = intake_ui.data_page
button = intake_ui.button


TEMPORAL_CSV = (
    "编号,描述,类别,记录时间,决策时间,窗口结束\n"
    "A1,训练样本一的描述,质量,2026-01-01T00:00:00Z,2026-01-01T00:00:00Z,2026-01-02T00:00:00Z\n"
    "A2,训练样本二的描述,物流,2026-01-05T00:00:00Z,2026-01-05T00:00:00Z,2026-01-06T00:00:00Z\n"
    "A3,验证样本一的描述,质量,2026-02-01T00:00:00Z,2026-02-01T00:00:00Z,2026-02-02T00:00:00Z\n"
    "A4,测试样本一的描述,物流,2026-03-01T00:00:00Z,2026-03-01T00:00:00Z,2026-03-02T00:00:00Z\n"
).encode()


@pytest.fixture()
def temporal_page(tmp_path, monkeypatch):
    """带真实时间字段的零密钥任务页面：时间分区路径的端到端测试底座。"""
    from streamlit.testing.v1 import AppTest

    import ui.config
    from src.workbench.intake_service import IntakeService

    for name in ("PROVIDER", "BASE_URL", "MODEL", "API_KEY"):
        monkeypatch.delenv(f"TUNESMITH_AGENT_{name}", raising=False)
    monkeypatch.setattr(ui.config, "PROJECT_ROOT", tmp_path)
    service = IntakeService(tmp_path / "outputs/workbench/intake")
    session = service.create("根据描述判断售后类别", "工单.csv", TEMPORAL_CSV)
    return service, session, AppTest.from_file(str(intake_ui.PAGE), default_timeout=20)


def test_baseline_analysis_entry_needs_no_agent_client(data_page):
    service, session, page = data_page
    page.run()
    page.selectbox(key="intake_select").select(session.session_id).run()
    assert not page.exception
    assert any(e.label.startswith("没有 Agent 服务") for e in page.expander)
    target = next(s for s in page.selectbox if s.label.startswith("答案列"))
    target.select("类别").run()
    groups = next(m for m in page.multiselect if m.label.startswith("业务分组字段"))
    groups.set_value(["编号"]).run()
    apply = next(b for b in page.button if b.label == "生成基础分析并预览")
    assert not apply.disabled
    apply.click().run()
    assert not page.exception
    current = service.load(session.session_id)
    assert current.analysis is not None
    assert current.preview is not None
    assert current.analysis.task.goal == session.goal
    roles = {role.column: role.role for role in current.analysis.task.field_roles}
    assert roles["类别"] == "target" and roles["编号"] == "group"
    assert any("客户描述" in row.input for row in current.preview.rows)
    # 无 Agent 参与的确定性方案也如实声明边界。
    assert any("不判断业务含义" in finding.message for finding in current.analysis.findings)


def test_baseline_analysis_supports_temporal_partition_fields(temporal_page):
    """基础分析可选择时间分区字段与边界；不完整或格式错误就地报错，不悄悄退回随机切分。"""
    service, session, page = temporal_page
    page.run()
    page.selectbox(key="intake_select").select(session.session_id).run()
    assert not page.exception
    target = next(s for s in page.selectbox if s.label.startswith("答案列"))
    target.select("类别").run()
    checkbox = next(c for c in page.checkbox if c.label.startswith("按时间分区切分"))
    checkbox.check().run()
    assert not page.exception
    next(s for s in page.selectbox if s.label.startswith("信息实际可获得时间字段")).select(
        "记录时间"
    ).run()
    next(s for s in page.selectbox if s.label.startswith("作出预测时间字段")).select(
        "决策时间"
    ).run()
    next(s for s in page.selectbox if s.label.startswith("标签窗口结束时间字段")).select(
        "窗口结束"
    ).run()
    apply = next(b for b in page.button if b.label == "生成基础分析并预览")
    # 边界未填：必须报错，不能退回随机切分。
    apply.click().run()
    assert not page.exception
    assert any("时间分区" in error.value for error in page.error)
    # 边界缺时区：不待点击，就地显示在输入框下方。
    next(t for t in page.text_input if t.label.startswith("验证起点")).input("2026-02-01").run()
    assert not page.exception
    assert any("必须明确时区" in error.value for error in page.error)
    next(t for t in page.text_input if t.label.startswith("验证起点")).input(
        "2026-02-01T00:00:00Z"
    ).run()
    next(t for t in page.text_input if t.label.startswith("测试起点")).input(
        "2026-03-01T00:00:00Z"
    ).run()
    next(t for t in page.text_input if t.label.startswith("观察截止")).input(
        "2026-04-01T00:00:00Z"
    ).run()
    apply.click().run()
    assert not page.exception
    current = service.load(session.session_id)
    policy = current.analysis.recipe.temporal_split
    assert policy is not None
    assert policy.available_at_column == "记录时间"
    assert policy.prediction_at_column == "决策时间"
    assert policy.label_end_at_column == "窗口结束"
    assert policy.validation_start == "2026-02-01T00:00:00Z"
    assert any("时间分区方案由用户指定" in f.message for f in current.analysis.findings)


def test_baseline_temporal_wrong_time_column_errors_in_place(temporal_page):
    """把编号当时间字段：服务校验的错误就地显示，方案不落地。"""
    service, session, page = temporal_page
    page.run()
    page.selectbox(key="intake_select").select(session.session_id).run()
    target = next(s for s in page.selectbox if s.label.startswith("答案列"))
    target.select("类别").run()
    next(c for c in page.checkbox if c.label.startswith("按时间分区切分")).check().run()
    next(s for s in page.selectbox if s.label.startswith("信息实际可获得时间字段")).select(
        "编号"
    ).run()
    next(s for s in page.selectbox if s.label.startswith("作出预测时间字段")).select(
        "决策时间"
    ).run()
    next(s for s in page.selectbox if s.label.startswith("标签窗口结束时间字段")).select(
        "窗口结束"
    ).run()
    for label, value in (
        ("验证起点", "2026-02-01T00:00:00Z"),
        ("测试起点", "2026-03-01T00:00:00Z"),
        ("观察截止", "2026-04-01T00:00:00Z"),
    ):
        next(t for t in page.text_input if t.label.startswith(label)).input(value).run()
    next(b for b in page.button if b.label == "生成基础分析并预览").click().run()
    assert not page.exception
    assert any("编号不是有效ISO时间" in error.value for error in page.error)
    assert service.load(session.session_id).analysis is None
