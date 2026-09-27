"""Business user reviews an Agent plan before any training is prepared."""

from copy import deepcopy

import pytest

pytest.importorskip("streamlit")

from tests.unit.test_workbench_training_ui import button, data_page, training_page  # noqa: F401


@pytest.fixture()
def planning_page(training_page, monkeypatch):  # noqa: F811
    import src.agent.training as agent
    import src.workbench.local_models as local_models
    import src.workbench.training_plans as planning

    monkeypatch.setattr(local_models, "discover_local_models", lambda: [])

    service, session, page, training_calls, records = training_page
    calls, plans = [], []

    class Plans:
        def __init__(self, *args):
            pass

        def list_plans(self, session_id=None):
            return deepcopy(plans)

        def prepare(self, plan_id, current):
            calls.append(("prepare", plan_id, current.session_id))
            plans[0]["run_id"] = "recommended-run"
            records.append(
                {
                    "run_id": "recommended-run",
                    "status": "prepared",
                    "dataset_version": current.dataset.version,
                    "model_path": "/tmp/local-a",
                    "config": {},
                    "preflight": {"status": "passed"},
                }
            )
            return {"plan_id": plan_id, "run_id": "recommended-run"}

    def recommend(current, candidates, client, *, output_root, training_root):
        calls.append(("recommend", candidates, client.model))
        plans.append(
            {
                "plan_id": "plan-fixture",
                "status": "ready",
                "run_id": None,
                "proposal": {
                    "model_path": candidates[0],
                    "max_length": 512,
                    "training_options": {"num_epochs": 1, "batch_size": 1},
                    "lora_options": {"r": 8},
                    "model_options": {"quantization_bits": None},
                    "rationale": ["按实际 token 检查选择长度"],
                    "limitations": ["本机内存摘要不能替代真实训练结果"],
                    "business_questions": [],
                },
                "context": {"probe": {"status": "passed"}},
                "trace": [{"tool": "probe_candidate", "ok": True}],
            }
        )
        return plans[-1]

    monkeypatch.setattr(planning, "TrainingPlanService", Plans)
    monkeypatch.setattr(agent, "recommend_training", recommend)
    return service, session, page, calls, plans, training_calls


def test_candidate_recommendation_review_prepare_and_start_are_separate(planning_page):
    _, session, page, calls, plans, training_calls = planning_page
    page.run()
    page.selectbox(key="intake_select").select(session.session_id).run()
    assert not page.exception
    assert calls == training_calls == []
    next(item for item in page.text_input if item.label == "支持工具调用的模型名称").input(
        "tool-fixture"
    ).run()
    page.text_area(key=f"plan_models_{session.session_id}").input(
        "/tmp/local-a\n/tmp/local-b\n/tmp/local-a"
    )
    button(page, "让 Agent 推荐训练方案").click().run()
    assert not page.exception
    assert calls[0][0:2] == ("recommend", ["/tmp/local-a", "/tmp/local-b"])
    assert any("按实际 token 检查选择长度" in block.value for block in page.markdown)
    assert training_calls == []
    assert plans[0]["run_id"] is None
    page.run()
    assert len(calls) == 1
    button(page, "确认推荐方案并准备训练").click().run()
    assert not page.exception
    assert calls[-1] == ("prepare", "plan-fixture", session.session_id)
    assert training_calls == []
    button(page, "启动这轮训练").click().run()
    assert not page.exception
    assert training_calls[-1][0] == "start"
    assert len(calls) == 2


@pytest.mark.parametrize("status", ["needs_data", "unsupported"])
def test_unready_plan_is_visible_but_cannot_prepare(planning_page, status):
    _, session, page, calls, plans, training_calls = planning_page
    plans.append(
        {
            "plan_id": "plan-blocked",
            "status": status,
            "proposal": {
                "rationale": ["先补充可信监督数据"],
                "business_questions": ["已有答案由谁审核？"],
            },
        }
    )
    page.run()
    page.selectbox(key="intake_select").select(session.session_id).run()
    assert not page.exception
    assert not any(item.label == "确认推荐方案并准备训练" for item in page.button)
    assert any("已有答案由谁审核" in item.value for item in page.warning)
    assert calls == training_calls == []


def test_remote_training_summary_needs_specific_consent(planning_page):
    _, session, page, calls, _, _ = planning_page
    page.run()
    page.selectbox(key="intake_select").select(session.session_id).run()
    next(item for item in page.text_input if item.label == "模型服务 API 地址").input(
        "https://fixture.example/v1"
    ).run()
    assert not page.exception
    assert button(page, "让 Agent 推荐训练方案").disabled
    next(
        item for item in page.checkbox if (item.key or "").startswith("plan_consent_")
    ).check().run()
    assert not button(page, "让 Agent 推荐训练方案").disabled
    assert calls == []


def test_discovered_complete_models_can_be_selected_without_manual_paths(
    planning_page, monkeypatch
):
    import src.workbench.local_models as local_models

    _, session, page, calls, _, _ = planning_page
    monkeypatch.setattr(
        local_models,
        "discover_local_models",
        lambda: [
            {
                "name": "Prepared",
                "model_path": "/tmp/prepared",
                "status": "available",
                "issues": [],
            },
            {
                "name": "Partial",
                "model_path": "/tmp/partial",
                "status": "incomplete",
                "issues": ["缺少权重"],
            },
        ],
    )
    page.run()
    page.selectbox(key="intake_select").select(session.session_id).run()
    assert not page.exception
    selector = page.multiselect(key=f"plan_discovered_{session.session_id}")
    assert selector.value == []
    assert len(selector.options) == 1
    selector.select("/tmp/prepared").run()
    assert page.text_area(key=f"plan_models_{session.session_id}").value == ""
    assert calls == []
    next(item for item in page.text_input if item.label == "支持工具调用的模型名称").input(
        "tool-fixture"
    ).run()
    button(page, "让 Agent 推荐训练方案").click().run()
    assert not page.exception
    assert calls[0][1] == ["/tmp/prepared"]
