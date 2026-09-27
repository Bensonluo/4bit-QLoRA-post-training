"""A new user without any Agent key can start the core path via baseline analysis."""

import pytest

pytest.importorskip("streamlit")

from tests.unit import test_data_intake_ui as intake_ui

data_page = intake_ui.data_page
button = intake_ui.button


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


def test_baseline_analysis_supports_temporal_partition_fields(data_page):
    """基础分析可选择时间分区字段与边界；不完整时明确报错，不悄悄退回随机切分。"""
    service, session, page = data_page
    page.run()
    page.selectbox(key="intake_select").select(session.session_id).run()
    assert not page.exception
    target = next(s for s in page.selectbox if s.label.startswith("答案列"))
    target.select("类别").run()
    checkbox = next(c for c in page.checkbox if c.label.startswith("按时间分区切分"))
    checkbox.check().run()
    assert not page.exception
    next(s for s in page.selectbox if s.label.startswith("信息实际可获得时间字段")).select(
        "编号"
    ).run()
    next(s for s in page.selectbox if s.label.startswith("作出预测时间字段")).select(
        "客户描述"
    ).run()
    next(s for s in page.selectbox if s.label.startswith("标签窗口结束时间字段")).select(
        "处理结果"
    ).run()
    apply = next(b for b in page.button if b.label == "生成基础分析并预览")
    # 边界未填：必须报错，不能退回随机切分。
    apply.click().run()
    assert not page.exception
    assert any("时间分区" in error.value for error in page.error)
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
    assert policy.available_at_column == "编号"
    assert policy.prediction_at_column == "客户描述"
    assert policy.label_end_at_column == "处理结果"
    assert policy.validation_start == "2026-02-01T00:00:00Z"
    assert any("时间分区方案由用户指定" in f.message for f in current.analysis.findings)
