"""盲标核验页面流:抽取题目隐藏答案 → 逐条作答 → 判定与不一致展示。"""

import pytest

pytest.importorskip("streamlit")

from tests.unit import test_data_intake_ui as intake_ui
from tests.unit.test_full_data import FULL, approved

data_page = intake_ui.data_page


@pytest.fixture()
def verify_page(data_page):
    service, _, page = data_page
    session = approved(service)
    session = service.validate_full_data(session.session_id, session.revision, "full.csv", FULL)
    session = service.confirm_full_data(session.session_id, session.revision)
    return service, session, page


def test_blind_verification_flow_via_page(verify_page):
    service, session, page = verify_page
    page.run()
    page.selectbox(key="intake_select").select(session.session_id).run()
    assert not page.exception
    assert any("盲标核验" in header.value for header in page.subheader)
    # 尚未核验:先抽取题目
    assert any(button.label == "抽取盲标核验题目" for button in page.button)
    next(b for b in page.button if b.label == "抽取盲标核验题目").click().run()
    assert not page.exception

    # 题目展示且不含答案;用业务正确答案作答(来自全量预览的目标值)
    targets = {row.row_id: row.target for row in session.full_data.preview.rows}
    inputs = [t for t in page.text_input if t.key and str(t.key).startswith("lv_")]
    assert inputs
    for field in inputs:
        row_id = str(field.key).rsplit("_", 1)[-1]
        field.input(targets[row_id]).run()
    next(b for b in page.button if b.label == "提交盲标核验答案").click().run()
    assert not page.exception
    assert any("盲标核验已通过" in message.value for message in page.success)
    current = service.load(session.session_id)
    assert current.label_verification["verdict"] == "verified"
    # 通过后不再显示抽取入口
    assert not any(b.label == "抽取盲标核验题目" for b in page.button)


def test_mismatch_shows_per_row_differences_and_blocks_training(verify_page):
    service, session, page = verify_page
    page.run()
    page.selectbox(key="intake_select").select(session.session_id).run()
    next(b for b in page.button if b.label == "抽取盲标核验题目").click().run()
    fields = [t for t in page.text_input if t.key and str(t.key).startswith("lv_")]
    fields[0].input("故意答错").run()
    targets = {row.row_id: row.target for row in session.full_data.preview.rows}
    for field in fields[1:]:
        row_id = str(field.key).rsplit("_", 1)[-1]
        field.input(targets[row_id]).run()
    next(b for b in page.button if b.label == "提交盲标核验答案").click().run()
    assert not page.exception
    assert any("盲标核验未通过" in message.value for message in page.error)
    assert any("故意答错" in block.label for block in page.expander)
    current = service.load(session.session_id)
    assert current.label_verification["verdict"] == "insufficient_agreement"


def test_contrast_check_via_page(data_page, monkeypatch):
    """预览确认前出现配对对比;配对正确后展示通过状态。"""
    import src.agent.intake
    from tests.unit.test_data_intake import analysis, model_for

    service, session, page = data_page
    monkeypatch.setattr(
        src.agent.intake, "CompatibleChatClient", lambda *a, **kw: model_for(analysis())
    )
    page.run()
    page.selectbox(key="intake_select").select(session.session_id).run()
    next(b for b in page.button if b.label == "联合分析目标与数据").click().run()
    assert not page.exception
    assert any("对比核验" in header.value for header in page.subheader)
    next(b for b in page.button if b.label == "开始配对对比").click().run()
    assert not page.exception
    # 用数据中的真实目标完成配对
    targets = {row.row_id: row.target for row in service.load(session.session_id).preview.rows}
    boxes = [
        s for s in page.selectbox if s.key and str(s.key).startswith("cc_")
    ]
    assert len(boxes) == 2
    for box in boxes:
        row_id = str(box.key).rsplit("_", 1)[-1]
        box.select(targets[row_id]).run()
    next(b for b in page.button if b.label == "提交配对").click().run()
    assert not page.exception
    assert any("对比核验已通过" in message.value for message in page.success)
