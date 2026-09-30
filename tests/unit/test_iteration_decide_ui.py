"""The evaluated iteration surfaces results and records the business decision."""

from copy import deepcopy

import pytest

pytest.importorskip("streamlit")

from src.workbench.report_summary import ITERATION_DECISION_NAMES
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
    # 单一决策入口(R93):旧第二套 st.form 决策表单已删,同屏不再两套表单、两种选项词汇并存。
    assert not any(s.label == "这轮结果如何处理？" for s in page.selectbox)
    assert not any(b.label == "保存本轮决策" for b in page.button)
    next(t for t in page.text_area if t.label == "业务理由（必填）").input(
        "三模型仍未达标，但对照证据完整。"
    ).run()
    choices = next(c for c in page.radio if c.label == "本轮决策")
    # 决策单选选项从单一来源派生(R93):就是 ITERATION_DECISION_NAMES 的值,页面不再手写第二套短名。
    assert list(choices.options) == list(ITERATION_DECISION_NAMES.values())
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
    # 已决策回显用单一来源长名(R93):success 行与摘要行同词汇,不再有「停止」短名第二套。
    assert any("已记录决策：停止本轮路线" in message.value for message in page.success)
    assert not any(c.label == "本轮决策" for c in page.radio)
    assert decisions == []
    # 旧的裸枚举回显行(**已记录：** stop · …)已删:决策对非专家只以人话名出现。
    assert not any("**已记录：**" in block.value for block in page.markdown)
    # 已决策态的人话摘要与 CLI 同源：决定名 + 业务理由回显。
    assert any("已记录你的业务决定：停止本轮路线" in block.value for block in page.markdown)
    assert any("业务理由：试点完成，停止迭代。" in block.value for block in page.markdown)


def test_decided_continue_iteration_points_to_next_round_form(decide_page):
    """决策闭环指路钉（R139）：decided+continue 必须点名通往下一轮假设表单的
    真实门牌（用当前数据微调模型 → 「训练 …」折叠器 → 开发集对照 → 将结果转成
    下一轮改进假设）——该表单埋在三层嵌套折叠器里，与决策现场零连接是
    r137-audit 的核心发现（改进环前进段断裂）。指路须如实标注数据版本门槛
    （current_report 语义）；stop 等其它决策不渲染（上一测试守极性）。"""
    _, session, page, _, iteration = decide_page
    iteration.update(
        status="decided", decision="continue", decision_reason="分数接近目标，继续调数据。"
    )
    page.run()
    page.selectbox(key="intake_select").select(session.session_id).run()
    assert not page.exception
    pointers = [i.value for i in page.info if "继续改进的下一步" in i.value]
    assert pointers, "decided+continue 必须渲染闭环指路"
    pointer = pointers[0]
    for door in ("用当前数据微调模型", "开发集对照", "将结果转成下一轮改进假设"):
        assert door in pointer, f"指路必须点名真实门牌：{door}"
    assert "本次评测版本" in pointer, "必须如实标注数据版本门槛"


def test_blocked_iteration_points_to_parent_evidence_with_honest_no_retry(decide_page):
    """阻断恢复指路钉（R142）：blocked 是终态（iterations.prepare 拒绝非 confirmed
    ——每轮只准备一次训练），页面不得指「重新准备本轮」的假门；出路是修因后从
    同一父轮证据起草下一轮。训练已建（new_run_id）时还须指下方训练记录的问题
    清单——训练侧 blocked 的原因不在本卡。"""
    _, session, page, _, iteration = decide_page
    iteration.update(
        status="blocked",
        new_run_id="wb-" + "9" * 32,
        run_id="wb-" + "9" * 32,
        failure="本地模型目录不存在。",
    )
    page.run()
    page.selectbox(key="intake_select").select(session.session_id).run()
    assert not page.exception
    # 阻断事实句（单一来源 summarize_iteration）以红色错误在场，含 failure 原文；
    # 句读由单源归一（不出现「。。」双句号）。
    rendered = [e.value for e in page.error] + [block.value for block in page.markdown]
    assert any("本轮训练准备被阻断：本地模型目录不存在。" in value for value in rendered)
    # 句读归一只约束阻断事实行本身（页面其他既有文案不受此钉管辖）。
    blocked_lines = [value for value in rendered if "本轮训练准备被阻断" in value]
    assert blocked_lines and not any("。。" in value for value in blocked_lines)
    # 诚实边界与 CLI 同源：无原位重试；不得残留指向不存在门的「重新准备」措辞。
    assert any("每轮只准备一次训练" in block.value for block in page.markdown)
    assert not any("重新准备" in block.value for block in page.markdown), (
        "R142：不得指向不存在的重试门"
    )
    # 已建训练：阻断的具体问题在下方训练记录里，不在本卡。
    assert any("该训练记录的问题清单" in c.value for c in page.caption)
    # 门牌指路（R139 四门牌范式）：微调区 → 父轮开发集对照 → 下一轮假设表单 +
    # 数据版本门槛如实标注。
    pointers = [i.value for i in page.info if "从同一份结果起草下一轮" in i.value]
    assert pointers, "blocked 必须渲染恢复指路"
    pointer = pointers[0]
    for door in ("用当前数据微调模型", "开发集对照", "将结果转成下一轮改进假设"):
        assert door in pointer, f"指路必须点名真实门牌：{door}"
    assert "本次评测版本" in pointer, "必须如实标注数据版本门槛"
    assert "不需要重新训练" in pointer, "父轮产物保留是诚实出路的一部分"
