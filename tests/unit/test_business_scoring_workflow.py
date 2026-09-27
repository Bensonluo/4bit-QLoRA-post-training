"""Numeric business rules through real isolation; model generation is a fixture."""

import pytest

from src.data.loaders import render_alpaca_prompt
from src.workbench.business_evaluation import (
    BusinessEvaluationService,
    EvaluationProtocol,
    Generation,
)
from src.workbench.evaluation_diagnostics import EvaluationDiagnostics
from src.workbench.evaluation_suites import evaluation_cases
from src.workbench.intake_models import IntakeAnalysis
from src.workbench.intake_service import IntakeService
from src.workbench.sandbox import TransformSandbox
from tests.unit.test_business_evaluation import model_paths as model_paths

SOURCE = r"""import re
def cents(text):
    match = re.fullmatch(r"\s*[¥￥]?\s*(\d+)(?:\.(\d{1,2}))?\s*元?\s*", text)
    if not match:
        return None
    return int(match.group(1)) * 100 + int((match.group(2) or "").ljust(2, "0"))

def transform(rows, config):
    results = []
    for row in rows:
        actual, target = cents(row["output"]), cents(row["expected"])
        if actual is None or target is None:
            score, reason = 0.0, "金额格式无效"
        else:
            distance = abs(actual - target)
            if distance <= config["tolerance_cents"]:
                score, reason = 1.0, "金额误差在允许范围内"
            elif distance <= 100:
                score, reason = 0.5, "金额误差不超过一元但未达标"
            else:
                score, reason = 0.0, "金额误差超过一元"
        results.append({**row, "score": score, "reason": reason})
    return results
"""


def numeric_task(tmp_path):
    service = IntakeService(tmp_path / "intake")
    header = "订单,描述,金额\n"
    rows = [f"o{i},订单约定付款金额为{i}.25元,{i}.25\n" for i in range(10, 30)]
    session = service.create(
        "从订单描述抽取应付金额，保留两位小数", "sample.csv", (header + "".join(rows[:2])).encode()
    )
    proposal = IntakeAnalysis.model_validate(
        {
            "task": {
                "goal": session.goal,
                "usage_input": "订单描述",
                "desired_output": "应付金额",
                "row_meaning": "一行一个独立订单",
                "supervision_source": "本地测试提供已核对的金额",
                "success_criteria": ["允许人民币符号和单位；金额误差最多一分"],
                "field_roles": [
                    {"column": "订单", "role": "group", "reason": "独立订单编号"},
                    {
                        "column": "描述",
                        "role": "input",
                        "reason": "推理时可得的原始描述",
                        "available_at_prediction": True,
                    },
                    {"column": "金额", "role": "target", "reason": "已核对金额"},
                ],
            },
            "findings": [],
            "recipe": {
                "instruction": "提取订单约定的应付金额。",
                "inputs": [{"column": "描述", "label": "描述"}],
                "targets": [{"column": "金额", "label": "金额", "value_kind": "open_text"}],
                "group_columns": ["订单"],
            },
            "training_approach": "金额提取 SFT 测试",
            "next_steps": ["核对金额含义与实际样例"],
        }
    )
    session = service.apply_analysis(session, proposal)
    session = service.confirm(session.session_id, session.revision)
    session = service.validate_full_data(
        session.session_id, session.revision, "full.csv", (header + "".join(rows)).encode()
    )
    session = service.confirm_full_data(session.session_id, session.revision)
    return service.materialize_dataset(
        session.session_id,
        session.revision,
        registry_root=tmp_path / "datasets",
        validation_fraction=0.2,
        test_fraction=0.2,
    )


def test_numeric_rule_scores_partial_credit_without_calling_it_exact_match(tmp_path, model_paths):
    from src.workbench.business_scoring import ScoringRecipe, ScoringService

    runner = TransformSandbox()
    if not runner.detect():
        pytest.skip(runner.unavailable_reason)
    session = numeric_task(tmp_path)
    cases = evaluation_cases(session)["records"]
    real = cases[0]
    examples = [
        {
            "name": "有单位的金额",
            "input": real["input"],
            "expected": real["output"],
            "output": "￥" + real["output"] + "元",
            "score": 1.0,
            "reason": "金额误差在允许范围内",
            "kind": "business",
        },
        {
            "name": "同订单金额差五角",
            "input": real["input"],
            "expected": real["output"],
            "output": f"{float(real['output']) + 0.5:.2f}",
            "score": 0.5,
            "reason": "金额误差不超过一元但未达标",
            "kind": "counterexample",
        },
    ]
    scoring = ScoringService(tmp_path / "scoring")
    draft = scoring.draft(
        session,
        ScoringRecipe(
            business_standard="金额误差最多一分通过；不超过一元计半分但不通过，格式错误零分。",
            source_code=SOURCE,
            config={"tolerance_cents": 1},
            pass_threshold=1.0,
            examples=examples,
        ),
    )
    reference = scoring.confirm(draft["scoring_id"], session)
    answers = {render_alpaca_prompt({**row, "output": ""}): row["output"] for row in cases}

    class Runtime:
        def __init__(self, model):
            self.label = model.label
            self.index = 0

        def generate(self, prompt, protocol):
            answer = answers[prompt]
            self.index += 1
            if self.label == "base":
                return Generation(f"{float(answer) + 0.5:.2f}")
            return Generation("￥" + answer + "元", truncated=self.index == 1)

        def close(self):
            pass

    report = BusinessEvaluationService(tmp_path / "reports").compare(
        session,
        model_paths,
        EvaluationProtocol("custom_rules", custom_scoring=reference),
        runtime_factory=Runtime,
    )
    base, tuned = report.models
    assert base["metrics"]["exact_match"] is None
    assert base["metrics"]["business_score"] == 0.5
    assert base["metrics"]["pass_rate"] == 0
    assert tuned["metrics"]["business_score"] == (len(cases) - 1) / len(cases)
    assert tuned["metrics"]["pass_rate"] == (len(cases) - 1) / len(cases)
    assert tuned["rows"][0]["status"] == "truncated"
    assert report.status == "completed_with_failures"
    evidence = EvaluationDiagnostics(report, session)
    assert evidence.summary()["error_count"] == len(cases) + 1
