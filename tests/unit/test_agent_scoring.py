# ruff: noqa: F811
"""Scoring Agent observes dev evidence, never confirms or reads final questions."""

import json

from src.agent.scoring import recommend_scoring
from tests.unit.test_business_evaluation import task  # noqa: F401
from tests.unit.test_business_scoring import recipe_for
from tests.unit.test_intake_sandbox import sandbox  # noqa: F401


class ScoringClient:
    model = "fixture-tool-loop"

    def __init__(self, clarify=False):
        self.messages = None
        self.step = 0
        self.clarify = clarify

    def complete(self, messages, tools):
        self.messages = messages
        self.step += 1
        if self.step == 1:
            name, args = "inspect_scoring_context", {}
        elif self.clarify:
            name, args = (
                "request_scoring_clarification",
                {"reason": "专业性缺少可判定标准", "questions": ["哪些回答要素是必须满足的？"]},
            )
        else:
            context = json.loads(
                next(message["content"] for message in messages if message["role"] == "tool")
            )
            name, args = (
                "submit_scoring_draft",
                {
                    "recipe": recipe_for(context["development_cases"][0]).model_dump(),
                    "evidence_ids": [context["evidence_id"]],
                },
            )
        return {
            "tool_calls": [
                {
                    "id": f"step-{self.step}",
                    "type": "function",
                    "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)},
                }
            ]
        }


def test_real_agent_drafts_without_confirmation_and_never_sends_test(task, tmp_path, sandbox):
    _, session = task
    client = ScoringClient()
    record = recommend_scoring(
        session, "完整答案必须与标准答案精确相同", client, tmp_path / "rules"
    )
    assert record["status"] == "draft"
    assert record["validation"]["status"] == "passed"
    payload = json.dumps(client.messages, ensure_ascii=False)
    with open(session.dataset.paths["test"]) as handle:
        rows = [json.loads(line) for line in handle]
    assert all(row["input"] not in payload for row in rows)


def test_subjective_standard_returns_specific_questions(task, tmp_path):
    _, session = task
    result = recommend_scoring(
        session, "回答是否专业", ScoringClient(clarify=True), tmp_path / "rules"
    )
    assert result["status"] == "needs_business_input"
    assert result["questions"] and result["reason"]
    assert "scoring_id" not in result
