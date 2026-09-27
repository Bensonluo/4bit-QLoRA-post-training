# ruff: noqa: F811
"""Actual OS execution and business scoring lifecycle contracts."""

import json
from types import SimpleNamespace

import pytest

from src.workbench.business_scoring import (
    ScoringRecipe,
    ScoringService,
    load_confirmed,
    score_outputs,
)
from tests.unit.test_business_evaluation import task  # noqa: F401
from tests.unit.test_intake_sandbox import sandbox  # noqa: F401

SOURCE = """def transform(rows, config):
    return [{**row, "score": 1.0 if row["output"] == row["expected"] else 0.0,
             "reason": "exact" if row["output"] == row["expected"] else "mismatch"} for row in rows]
"""


def recipe_for(case):
    return ScoringRecipe(
        business_standard="完整答案必须与标准答案精确相同",
        source_code=SOURCE,
        pass_threshold=1,
        examples=[
            {
                "name": "正确答案",
                "input": case["input"],
                "expected": case["expected"],
                "output": case["expected"],
                "score": 1,
                "reason": "exact",
                "kind": "business",
            },
            {
                "name": "额外解释",
                "input": case["input"],
                "expected": case["expected"],
                "output": case["expected"] + " 额外解释",
                "score": 0,
                "reason": "mismatch",
                "kind": "counterexample",
            },
        ],
    )


def test_real_draft_confirm_score_and_semantic_binding(task, tmp_path, sandbox):
    _, session = task
    service = ScoringService(tmp_path / "scoring")
    context = service.context(session, "精确相同")
    recipe = recipe_for(context["development_cases"][0])
    record = service.draft(session, recipe)
    assert record["validation"]["status"] == "passed"
    assert [example["score"] for example in record["example_results"]] == [1, 0]
    assert service.list_specs(session.session_id)[0] == record
    with pytest.raises(ValueError, match="尚未确认"):
        load_confirmed(
            {
                "root": str(service.root),
                "scoring_id": record["scoring_id"],
                "spec_digest": record["spec_digest"],
            },
            session,
        )
    stale = session.model_copy(deep=True)
    stale.revision += 1
    with pytest.raises(ValueError, match="已更新"):
        service.confirm(record["scoring_id"], stale)
    reference = service.confirm(record["scoring_id"], session)
    assert load_confirmed(reference, stale) == recipe
    stale.goal += "不同任务"
    with pytest.raises(ValueError, match="语义已变化"):
        load_confirmed(reference, stale)
    example = recipe.examples[0]
    result = score_outputs(
        recipe,
        [
            {
                "__row_id": "r1",
                "input": example.input,
                "expected": example.expected,
                "output": example.output + "extra",
            }
        ],
        sandbox=sandbox,
    )
    assert result[0]["score"] == 0


def test_dev_only_bounded_context_and_real_negative_anchor(task, tmp_path):
    _, session = task
    service = ScoringService(tmp_path / "scoring")
    first = service.context(session, "精确", limit=1)
    assert first["returned"] == 1 and first["has_more"] and first["total"] > 1
    second = service.context(session, "精确", offset=first["next_offset"], limit=1)
    assert second["development_cases"][0] != first["development_cases"][0]
    with open(session.dataset.paths["test"]) as handle:
        test_rows = [json.loads(line) for line in handle]
    payload = json.dumps(first, ensure_ascii=False)
    assert all(row["input"] not in payload for row in test_rows)
    recipe = recipe_for(first["development_cases"][0])
    recipe.examples[1].expected = "假标准"
    with pytest.raises(ValueError, match="不能只改标准答案"):
        service.draft(session, recipe)
    recipe.examples[0].input = "编造资料"
    with pytest.raises(ValueError, match="真实开发题"):
        service.draft(session, recipe)


@pytest.mark.parametrize(
    "change",
    [
        "[]",
        "[{**r, 'input':'changed', 'score':1, 'reason':'x'} for r in rows]",
        "[{**r, 'score':True, 'reason':'x'} for r in rows]",
        "[{**r, 'score':2, 'reason':'x'} for r in rows]",
        "[{**r, 'score':1, 'reason':''} for r in rows]",
    ],
)
def test_actual_sandbox_invalid_output_is_rejected(sandbox, change):
    recipe = recipe_for({"input": "question", "expected": "yes"})
    recipe.source_code = "def transform(rows, config):\n return " + change
    with pytest.raises(ValueError):
        score_outputs(
            recipe,
            [{"__row_id": "r1", "input": "question", "expected": "yes", "output": "yes"}],
            sandbox=sandbox,
        )


def test_unavailable_no_host_execution(tmp_path):
    from src.workbench.sandbox import TransformSandbox

    runner = TransformSandbox()
    runner.detect = lambda: False
    target = tmp_path / "forbidden"
    recipe = recipe_for({"input": "q", "expected": "a"})
    recipe.source_code = (
        f'def transform(rows, config):\n open({str(target)!r}, "w").write("bad")\n return rows'
    )
    with pytest.raises(ValueError, match="未在宿主执行"):
        score_outputs(
            recipe,
            [{"__row_id": "r", "input": "q", "expected": "a", "output": "a"}],
            sandbox=runner,
        )
    assert not target.exists()


def test_nonfinite_and_boolean_score_contract():
    recipe = recipe_for({"input": "q", "expected": "a"})
    for value in (True, float("nan"), float("inf"), "1"):
        altered = recipe.model_dump()
        altered["examples"][0]["score"] = value
        with pytest.raises(ValueError):
            ScoringRecipe.model_validate(altered)
    runner = SimpleNamespace(
        run=lambda *args: SimpleNamespace(
            status="passed",
            rows=[
                {
                    "__row_id": "r",
                    "input": "q",
                    "expected": "a",
                    "output": "a",
                    "score": float("nan"),
                    "reason": "bad",
                }
            ],
        )
    )
    with pytest.raises(ValueError):
        score_outputs(
            recipe,
            [{"__row_id": "r", "input": "q", "expected": "a", "output": "a"}],
            sandbox=runner,
        )


def test_oversized_case_not_truncated(task, tmp_path, monkeypatch):
    _, session = task
    service = ScoringService(tmp_path / "scoring")
    monkeypatch.setattr(
        service,
        "_development_records",
        lambda session: [{"input": "x" * 17000, "output": "a", "metadata": {}}],
    )
    context = service.context(session, "精确")
    assert context["development_cases"][0]["content_status"] == "too_large"
    assert "input" not in context["development_cases"][0]
