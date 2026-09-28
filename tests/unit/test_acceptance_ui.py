"""Final business acceptance is explicit, single-model, and isolated from Agent optimization."""

from copy import deepcopy
from dataclasses import asdict

import pytest

pytest.importorskip("streamlit")

from tests.unit.test_workbench_training_ui import button, data_page, training_page  # noqa: F401


@pytest.fixture()
def acceptance_page(training_page, monkeypatch):  # noqa: F811
    import src.workbench.acceptance as acceptance_module

    _, session, page, _, training_records = training_page
    training_records.append(
        {
            "run_id": "successful-run",
            "session_id": session.session_id,
            "status": "succeeded",
            "dataset_version": session.dataset.version,
            "model_path": "/tmp/base",
            "output_dir": "/tmp/adapter",
        }
    )
    records, calls = [], []

    class Acceptance:
        def __init__(self, *args):
            pass

        def list_acceptances(self, session_id=None):
            return deepcopy(records)

        def prepare(self, current, model, protocol, criteria, task_spec=None):
            calls.append(("prepare", model, protocol, criteria))
            records.append(
                {
                    "acceptance_id": "acceptance-fixture",
                    "session_id": current.session_id,
                    "status": "prepared",
                    "criteria": criteria,
                    "model": asdict(model),
                    "protocol": asdict(protocol),
                    "evaluation_suite": {"suite_id": "fixed-suite"},
                    "blind_test": True,
                    "report": None,
                    "result": {"decision": "pending_run"},
                    "task_spec": task_spec,
                }
            )
            return records[-1]

        def run(self, identity, current):
            calls.append(("run", identity))
            records[0].update(
                status="completed",
                result={"decision": "insufficient_evidence", "reason": "测试题数低于业务确认门槛"},
                report={
                    "models": [
                        {
                            "rows": [
                                {
                                    "index": 0,
                                    "prompt": "最终独立业务题",
                                    "expected": "参考答案",
                                    "output": "实际模型回答",
                                    "status": "scored",
                                    "truncated": False,
                                }
                            ]
                        }
                    ]
                },
            )
            return records[0]

        def review(self, identity, decisions):
            calls.append(("review", identity, decisions))
            records[0].setdefault("decisions", []).extend(decisions)
            return records[0]

    monkeypatch.setattr(acceptance_module, "AcceptanceService", Acceptance)
    return session, page, records, calls


def test_business_criteria_have_no_default_and_final_execution_is_separate(acceptance_page):
    session, page, records, calls = acceptance_page
    page.run()
    page.selectbox(key="intake_select").select(session.session_id).run()
    assert not page.exception
    assert page.number_input(key="acceptance_min_score_successful-run").value is None
    assert page.number_input(key="acceptance_min_cases_successful-run").value is None
    button(page, "冻结此模型与业务验收标准").click().run()
    assert calls == []
    assert any("软件不替你设定业务门槛" in item.value for item in page.error)
    next(item for item in page.text_area if item.label == "最终业务验收标准").input(
        "至少九成分类严格正确"
    )
    page.number_input(key="acceptance_min_score_successful-run").set_value(90.0)
    page.number_input(key="acceptance_min_cases_successful-run").set_value(20)
    button(page, "冻结此模型与业务验收标准").click().run()
    assert not page.exception
    assert len(calls) == 1 and calls[0][0] == "prepare"
    assert records[0]["criteria"]["minimum_score"] == 0.9
    assert records[0]["criteria"]["minimum_cases"] == 20
    assert records[0]["status"] == "prepared"
    page.run()
    assert len(calls) == 1
    # 冻结后、执行前：与 CLI 同口径的人话摘要说明条款已冻结、留出题尚未占用。
    assert any("条款已冻结、验收尚未执行" in item.value for item in page.markdown)
    button(page, "按冻结标准执行最终验收").click().run()
    assert not page.exception
    assert calls[-1][0] == "run"
    assert any("证据不足" in item.value for item in page.warning)
    # 结论人话与 CLI 同源：证据不足口径 + 原因 + 冻结条款边界收尾。
    assert any("当前结论：证据不足，不能确认可交付" in item.value for item in page.markdown)
    assert any("原因：测试题数低于业务确认门槛" in item.value for item in page.markdown)
    assert any("以上结论只对这次冻结的条款与固定测试题负责" in item.value for item in page.markdown)
    assert any(item.value == "最终独立业务题" for item in page.code)
    assert any(item.value == "实际模型回答" for item in page.code)
    assert not any(item.label == "让 Agent 分析结果与下一步" for item in page.button)


def test_open_final_review_requires_reason_and_forbids_accepting_truncated_output(acceptance_page):
    session, page, records, calls = acceptance_page
    records.append(
        {
            "acceptance_id": "open-final",
            "session_id": session.session_id,
            "status": "needs_business_review",
            "model": {"adapter_path": "/tmp/adapter"},
            "protocol": {"scorer": "open_review"},
            "criteria": {
                "metric": "manual_acceptance_rate",
                "business_standard": "关键步骤齐全",
                "minimum_score": 0.9,
                "minimum_cases": 20,
            },
            "result": {"decision": "pending_review"},
            "report": {
                "models": [
                    {
                        "rows": [
                            {
                                "index": 0,
                                "prompt": "最终问题甲",
                                "expected": "完整步骤",
                                "output": "完整回答",
                                "status": "needs_business_review",
                                "truncated": False,
                            },
                            {
                                "index": 1,
                                "prompt": "最终问题乙",
                                "expected": "完整步骤",
                                "output": "截断片段",
                                "status": "truncated",
                                "truncated": True,
                            },
                        ]
                    }
                ]
            },
        }
    )
    page.run()
    page.selectbox(key="intake_select").select(session.session_id).run()
    assert not page.exception
    selector = next(
        item for item in page.selectbox if item.label == "这条回答是否达到冻结业务标准？"
    )
    selector.select("accepted")
    button(page, "保存这条业务判断").click().run()
    assert calls == []
    assert any("说明业务理由" in item.value for item in page.error)
    next(item for item in page.text_area if item.label == "这条判断的业务理由").input(
        "步骤齐全且符合业务规则"
    )
    button(page, "保存这条业务判断").click().run()
    assert not page.exception
    assert calls[-1] == (
        "review",
        "open-final",
        [{"index": 0, "decision": "accepted", "reason": "步骤齐全且符合业务规则"}],
    )
    page.selectbox(key="acceptance_row_open-final").select(1).run()
    selector = next(
        item for item in page.selectbox if item.label == "这条回答是否达到冻结业务标准？"
    )
    assert "通过" not in selector.options
    assert "不通过" in selector.options
    assert any("不能人工标记通过" in item.value for item in page.warning)
    assert not any(item.label == "让 Agent 分析结果与下一步" for item in page.button)


def test_freeze_area_shows_task_spec_before_freezing_and_records_it(acceptance_page):
    """冻结表单前可见任务规约折叠区(与训练启动前折叠区同源同词汇),
    冻结动作把四要素快照随条款一起写进验收记录。"""
    session, page, records, calls = acceptance_page
    page.run()
    page.selectbox(key="intake_select").select(session.session_id).run()
    assert not page.exception
    labels = [expander.label for expander in page.expander]
    assert any(label == "📋 任务规约（冻结验收条款前的口径）" for label in labels)
    texts = [block.value for block in page.markdown]
    assert any("的任务规约：由既有确认记录只读汇编" in text for text in texts)
    assert any("业务目标：根据客户首次描述判断问题类型" in text for text in texts)
    next(item for item in page.text_area if item.label == "最终业务验收标准").input(
        "至少九成分类严格正确"
    )
    page.number_input(key="acceptance_min_score_successful-run").set_value(90.0)
    page.number_input(key="acceptance_min_cases_successful-run").set_value(20)
    button(page, "冻结此模型与业务验收标准").click().run()
    assert not page.exception
    assert len(calls) == 1 and calls[0][0] == "prepare"
    assert records[0]["task_spec"]["goal"]["goal"] == "根据客户首次描述判断问题类型"


def test_frozen_acceptance_card_renders_gate_arithmetic_line(acceptance_page):
    """冻结验收卡按题集实际题数渲染门槛分辨率算术行(需通过 9 道、容错 1 道),
    与 CLI stderr 同一来源(acceptance_gate_lines 单一来源)。"""
    session, page, records, calls = acceptance_page
    records.append(
        {
            "acceptance_id": "gate-fixture",
            "session_id": session.session_id,
            "status": "prepared",
            "model": {"adapter_path": "/tmp/adapter", "label": "待验收模型"},
            "protocol": {"scorer": "classification_exact"},
            "criteria": {
                "metric": "exact_match",
                "business_standard": "分类必须严格正确",
                "minimum_score": 0.9,
                "minimum_cases": 5,
            },
            "evaluation_suite": {
                "suite_id": "fixed-suite",
                "case_counts": {"validation": 6, "test": 10},
            },
            "blind_test": True,
            "report": None,
            "result": {"decision": "pending_run"},
        }
    )
    page.run()
    page.selectbox(key="intake_select").select(session.session_id).run()
    assert not page.exception
    gate_line = (
        "按 90% 通过率门槛与 10 道最终测试题算：需通过 9 道、最多容错 1 道未通过"
        "（每题占通过率 10 个百分点）。"
    )
    assert any(gate_line in item.value for item in page.markdown)
