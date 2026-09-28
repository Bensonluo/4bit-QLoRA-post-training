"""The evaluated iteration surfaces results and records the business decision."""

from copy import deepcopy

import pytest

pytest.importorskip("streamlit")

from tests.unit import test_data_intake_ui as intake_ui
from tests.unit.test_full_data import FULL, approved

data_page = intake_ui.data_page
button = intake_ui.button

IDENTITY = "it-" + "c" * 32


@pytest.fixture()
def decide_page(data_page, monkeypatch):
    import src.workbench.iterations as iterations_module

    service, _, page = data_page
    session = approved(service)
    session = service.validate_full_data(session.session_id, session.revision, "full.csv", FULL)
    session = service.confirm_full_data(session.session_id, session.revision)
    iteration = {
        "iteration_id": IDENTITY,
        "session_id": session.session_id,
        "goal": session.goal,
        "status": "evaluated",
        "hypothesis": "补充确认样例后准确率提升。",
        "expected_outcome": "开发集对照有可解释变化。",
        "changes": "保持资料与配置，验证决策入口。",
        "parent_run_id": "wb-" + "1" * 32,
        "evaluation_suite": {
            "suite_id": "f" * 64,
            "case_counts": {"validation": 4, "test": 4},
        },
        "options": {},
        "evaluation_id": "e" * 32,
        "results": [
            {"label": "基座", "metrics": {"total": 2, "exact_match": 0.0}},
            {"label": "父轮模型", "metrics": {"total": 2, "exact_match": 0.0}},
            {"label": "本轮微调", "metrics": {"total": 2, "exact_match": 0.5}},
        ],
        "data_change": False,
    }
    decisions = []

    class Iterations:
        def __init__(self, *args, **kwargs):
            pass

        def list_iterations(self, session_id=None):
            return [deepcopy(iteration)]

        def get(self, iteration_id):
            return deepcopy(iteration)

        def decide(self, iteration_id, decision, reason):
            decisions.append((iteration_id, decision, reason))
            iteration["status"] = "decided"
            iteration["decision"] = decision
            iteration["decision_reason"] = reason
            return deepcopy(iteration)

    monkeypatch.setattr(iterations_module, "IterationService", Iterations)
    return service, session, page, decisions, iteration


def test_evaluated_iteration_shows_results_and_records_decision(decide_page):
    _, session, page, decisions, _ = decide_page
    page.run()
    page.selectbox(key="intake_select").select(session.session_id).run()
    assert not page.exception
    assert any("同题三模型结果" in block.value for block in page.markdown)
    assert any("开发集报告" in caption.value for caption in page.caption)
    # 与 CLI 同口径的人话摘要：三模型对照完成待业务决定 + 流程状态边界收尾。
    assert any(
        "基座、父轮与本轮的三模型同题对照已完成，正等待你的业务决定" in block.value
        for block in page.markdown
    )
    assert any(
        "以上只是流程状态与已记录的决定，不代表业务效果达标" in block.value
        for block in page.markdown
    )
    record = next(b for b in page.button if b.label == "记录本轮决策")
    assert record.disabled
    next(t for t in page.text_area if t.label == "业务理由（必填）").input(
        "三模型仍未达标，但对照证据完整。"
    ).run()
    choices = next(c for c in page.radio if c.label == "本轮决策")
    choices.set_value("证据不足").run()
    record = next(b for b in page.button if b.label == "记录本轮决策")
    assert not record.disabled
    record.click().run()
    assert not page.exception
    assert decisions == [(IDENTITY, "insufficient_evidence", "三模型仍未达标，但对照证据完整。")]
    assert any("证据不足" in message.value for message in page.success)


def test_evaluated_iteration_shows_three_model_delta_in_question_counts(decide_page):
    _, session, page, _, _ = decide_page
    page.run()
    page.selectbox(key="intake_select").select(session.session_id).run()
    assert not page.exception
    # 三模型分数差换算成题数差：本轮微调 1/2，父轮与基座各 0/2，各多答对 1 题。
    assert any(
        "父轮对照：本轮微调比父轮多答对 1 题（1/2 vs 0/2）。" in block.value
        for block in page.markdown
    )
    assert any(
        "基座对照：本轮微调比基座多答对 1 题（1/2 vs 0/2）。" in block.value
        for block in page.markdown
    )
    # 开发集只有 2 道题，每题占 50 个百分点，提示 1 题差距的解释边界。
    assert any("每题约占 50 个百分点" in block.value for block in page.markdown)


def test_decided_iteration_shows_recorded_decision_without_new_controls(decide_page):
    _, session, page, decisions, iteration = decide_page
    iteration.update(status="decided", decision="stop", decision_reason="试点完成，停止迭代。")
    page.run()
    page.selectbox(key="intake_select").select(session.session_id).run()
    assert not page.exception
    assert any("已记录决策：停止" in message.value for message in page.success)
    assert not any(c.label == "本轮决策" for c in page.radio)
    assert decisions == []
    # 已决策态的人话摘要与 CLI 同源：决定名 + 业务理由回显。
    assert any("已记录你的业务决定：停止本轮路线" in block.value for block in page.markdown)
    assert any("业务理由：试点完成，停止迭代。" in block.value for block in page.markdown)
