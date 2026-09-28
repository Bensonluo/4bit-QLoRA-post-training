"""Scoring drafts show real validation evidence and require explicit business confirmation."""

from copy import deepcopy
from dataclasses import asdict

import pytest

pytest.importorskip("streamlit")

from tests.unit.test_workbench_training_ui import button, data_page, training_page  # noqa: F401


@pytest.fixture()
def scoring_page(training_page, monkeypatch):  # noqa: F811
    import src.agent.scoring as agent
    import src.workbench.business_evaluation as evaluation
    import src.workbench.business_scoring as scoring

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
    specs, calls, reports = [], [], []

    class Scoring:
        def __init__(self, *args):
            pass

        def list_specs(self, session_id=None):
            return deepcopy(specs)

        def confirm(self, identity, current):
            calls.append(("confirm", identity))
            specs[0]["status"] = "confirmed"
            return {
                "root": "/tmp/scoring",
                "scoring_id": identity,
                "spec_digest": specs[0]["spec_digest"],
            }

    def recommend(current, standard, client, *, output_root):
        calls.append(("draft", standard))
        record = {
            "scoring_id": "score-fixture",
            "status": "draft",
            "session_id": current.session_id,
            "spec_digest": "c" * 64,
            "recipe": {
                "business_standard": standard,
                "pass_threshold": 0.8,
                "source_code": "def transform(rows, config): return rows",
                "examples": [
                    {"name": "完整步骤", "kind": "business", "score": 1.0},
                    {"name": "缺项反例", "kind": "counterexample", "score": 0.5},
                ],
                "config": {},
            },
            "validation": {
                "status": "passed",
                "backend": "fixture-os",
                "cases": [{"name": "完整步骤", "passed": True}],
            },
            "example_results": [
                {"name": "完整步骤", "kind": "business", "score": 1.0, "reason": "全部要求满足"},
                {
                    "name": "缺项反例",
                    "kind": "counterexample",
                    "score": 0.5,
                    "reason": "缺少必要步骤",
                },
            ],
        }
        specs.append(record)
        return record

    class Evaluation:
        def __init__(self, *args):
            pass

        def list_reports(self, dataset_version=None, purpose=None):
            return reports

        def compare(self, current, models, protocol):
            calls.append(("compare", protocol))
            report = evaluation.EvaluationReport(
                "b" * 32,
                "now",
                {"version": current.dataset.version},
                asdict(protocol),
                "comparison-key",
                status="completed",
            )
            for model in models:
                report.models.append(
                    {
                        "label": model.label,
                        "requested_model": asdict(model),
                        "metrics": {
                            "total": 1,
                            "scored": 1,
                            "exact_match": None,
                            "business_score": 0.75,
                            "pass_rate": 0,
                        },
                        "rows": [
                            {
                                "index": 0,
                                "source": {},
                                "prompt": "开发样例",
                                "expected": "完整步骤",
                                "output": "回答缺项",
                                "status": "scored",
                                "correct": False,
                                "truncated": False,
                                "business_score": 0.75,
                                "scoring_reason": "缺少必要步骤",
                            }
                        ],
                    }
                )
            reports.append(report)
            return report

    monkeypatch.setattr(scoring, "ScoringService", Scoring)
    monkeypatch.setattr(agent, "recommend_scoring", recommend)
    monkeypatch.setattr(evaluation, "BusinessEvaluationService", Evaluation)
    return session, page, specs, calls


def test_draft_requires_business_confirmation_then_comparison_uses_business_metrics(scoring_page):
    session, page, specs, calls = scoring_page
    page.run()
    page.selectbox(key="intake_select").select(session.session_id).run()
    next(item for item in page.text_input if item.label == "支持工具调用的模型名称").input(
        "tool-fixture"
    ).run()
    page.text_area(key=f"scoring_standard_{session.session_id}").input("回答须含全部必要步骤")
    button(page, "让 Agent 拟定业务评分规则").click().run()
    assert not page.exception
    assert specs[0]["status"] == "draft"
    assert [item[0] for item in calls] == ["draft"]
    assert button(page, "确认这套业务评分规则").disabled
    # 页面与 CLI 同源的人话摘要：标准句+通过线+不自动确认边界句。
    assert any(
        "这套规则要判断的业务标准：回答须含全部必要步骤。" in item.value for item in page.markdown
    )
    assert any("软件不会自动确认评分规则" in item.value for item in page.markdown)
    assert any("score" in table.value.columns for table in page.dataframe)
    page.checkbox(key="ack_scoring_score-fixture").check().run()
    button(page, "确认这套业务评分规则").click().run()
    assert not page.exception
    assert calls[-1] == ("confirm", "score-fixture")
    page.selectbox(key="eval_scoring_successful-run").select("score-fixture").run()
    button(page, "比较基座与本轮微调效果").click().run()
    assert not page.exception
    assert calls[-1][0] == "compare"
    assert calls[-1][1].scorer == "custom_rules"
    summary = next(table.value for table in page.dataframe if "业务分均值" in table.value.columns)
    assert "业务通过率" in summary.columns
    assert "严格准确率" not in summary.columns
    assert any("缺少必要步骤" in item.value for item in page.markdown)


def test_ambiguous_business_goal_displays_questions_without_confirm_action(
    scoring_page, monkeypatch
):
    import src.agent.scoring as agent

    session, page, _, calls = scoring_page
    monkeypatch.setattr(
        agent,
        "recommend_scoring",
        lambda *args, **kwargs: {
            "status": "needs_business_input",
            "reason": "专业程度需要明确可判定要求",
            "questions": ["哪些关键步骤不可缺少？"],
            "trace": [],
        },
    )
    page.run()
    page.selectbox(key="intake_select").select(session.session_id).run()
    next(item for item in page.text_input if item.label == "支持工具调用的模型名称").input(
        "tool-fixture"
    ).run()
    page.text_area(key=f"scoring_standard_{session.session_id}").input("更专业")
    button(page, "让 Agent 拟定业务评分规则").click().run()
    assert not page.exception
    assert any("哪些关键步骤" in item.value for item in page.markdown)
    assert any("当前没有可确认" in item.value for item in page.info)
    assert not any(item.label == "确认这套业务评分规则" for item in page.button)
    assert calls == []
