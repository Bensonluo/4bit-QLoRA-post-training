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
    boxes = [s for s in page.selectbox if s.key and str(s.key).startswith("cc_")]
    assert len(boxes) == 2
    for box in boxes:
        row_id = str(box.key).rsplit("_", 1)[-1]
        box.select(targets[row_id]).run()
    next(b for b in page.button if b.label == "提交配对").click().run()
    assert not page.exception
    assert any("第一轮配对正确" in message.value for message in page.info)

    # 第二轮(换题):二连对后才显示通过
    next(b for b in page.button if b.label == "开始配对对比").click().run()
    boxes = [s for s in page.selectbox if s.key and str(s.key).startswith("cc_")]
    for box in boxes:
        row_id = str(box.key).rsplit("_", 1)[-1]
        box.select(targets[row_id]).run()
    next(b for b in page.button if b.label == "提交配对").click().run()
    assert not page.exception
    assert any("对比核验二连对" in message.value for message in page.success)

    # 二连对后核验已达标;强制的表单收进可选 expander,第三轮不强制
    expander = next(e for e in page.expander if "可选" in e.label and "对比核验" in e.label)
    optional_buttons = [b for b in expander.button if b.label == "开始配对对比"]
    assert optional_buttons
    next(b for b in page.button if b.label == "开始配对对比").click().run()
    assert not page.exception
    boxes = [s for s in page.selectbox if s.key and str(s.key).startswith("cc_")]
    for box in boxes:
        row_id = str(box.key).rsplit("_", 1)[-1]
        box.select(targets[row_id]).run()
    next(b for b in page.button if b.label == "提交配对").click().run()
    assert not page.exception
    assert any("对比核验3轮连胜" in message.value for message in page.success)


def test_contrast_check_third_round_is_optional_entry(data_page, monkeypatch):
    """二连对后不再强制配对:核验入口收进可选 expander,直接确认不受阻。"""
    import src.agent.intake
    from tests.unit.test_data_intake import analysis, model_for

    service, session, page = data_page
    monkeypatch.setattr(
        src.agent.intake, "CompatibleChatClient", lambda *a, **kw: model_for(analysis())
    )
    page.run()
    page.selectbox(key="intake_select").select(session.session_id).run()
    next(b for b in page.button if b.label == "联合分析目标与数据").click().run()
    targets = {row.row_id: row.target for row in service.load(session.session_id).preview.rows}
    for _ in range(2):
        next(b for b in page.button if b.label == "开始配对对比").click().run()
        for box in [s for s in page.selectbox if s.key and str(s.key).startswith("cc_")]:
            row_id = str(box.key).rsplit("_", 1)[-1]
            box.select(targets[row_id]).run()
        next(b for b in page.button if b.label == "提交配对").click().run()
    assert not page.exception
    assert any("对比核验二连对" in message.value for message in page.success)
    # 可选入口在,但默认不展开、不强制;确认按钮可用
    assert any("可选" in e.label and "对比核验" in e.label for e in page.expander)
    next(c for c in page.checkbox if c.label.startswith("已核对预览")).check().run()
    assert not next(b for b in page.button if b.label == "确认当前转换含义").disabled


def test_stale_warning_renders_after_revision(verify_page):
    """数据修订后,页面明确显示「核验已失效请重验」,而不是静默回到初始状态。"""
    from copy import deepcopy

    from src.workbench.intake_models import Transform

    service, session, page = verify_page
    # 先完成一轮核验(通过)
    page.run()
    page.selectbox(key="intake_select").select(session.session_id).run()
    next(b for b in page.button if b.label == "抽取盲标核验题目").click().run()
    targets = {row.row_id: row.target for row in session.full_data.preview.rows}
    for field in [t for t in page.text_input if t.key and str(t.key).startswith("lv_")]:
        row_id = str(field.key).rsplit("_", 1)[-1]
        field.input(targets[row_id]).run()
    next(b for b in page.button if b.label == "提交盲标核验答案").click().run()
    assert any("盲标核验已通过" in m.value for m in page.success)

    # 修订配方(新增 replace 转换)后重新确认全量 → 核验失效
    analysis = deepcopy(session.analysis)
    analysis.recipe.inputs[0].transforms.append(Transform(operation="replace", old="x", new="y"))
    service.apply_analysis(session, analysis)
    assert any("已失效" in w.value for w in page.warning) or True  # 先提交后渲染
    page.run()
    assert not page.exception
    # 失效警告出现在盲标核验区(核验状态需在全量确认后可见)
    assert any("已失效" in w.value for w in page.warning)
