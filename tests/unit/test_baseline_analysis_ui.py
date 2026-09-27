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
