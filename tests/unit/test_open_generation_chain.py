"""开放生成任务轻量链验证:open_review 对照真实落盘,协议身份→决策指标口径一致。

不训练模型、不加载权重(runtime 全 mock)。完整迭代链(propose/bind)需要真实
父轮训练产物身份,由 tests/unit/test_iterations 与贯穿测试覆盖;本测试钉住
轻量可跑段的口径不漂移。
"""

import pytest

from src.workbench.baseline_analysis import propose_baseline_analysis
from src.workbench.business_evaluation import (
    BusinessEvaluationService,
    EvaluationModel,
    EvaluationProtocol,
    Generation,
)
from src.workbench.intake_service import IntakeService
from src.workbench.iterations import decision_metric_for_protocol


@pytest.fixture()
def open_session(tmp_path):
    long_q = (
        "客户在收货后发现商品存在质量问题并且多次联系客服没有得到满意答复情绪激动要求立即处理。" * 2
    )
    long_a = "根据您的反馈我们已安排专员跟进处理并将持续关注解决进度直到问题闭环。" * 2
    sample = (
        "编号,客户问题,标准答复\n"
        + "".join(f"{i:03d},{long_q}（{i}）,{long_a}（{i}）\n" for i in range(1, 9))
    ).encode()
    full = (
        "编号,客户问题,标准答复\n"
        + "".join(f"{i:03d},{long_q}（{i}）,{long_a}（{i}）\n" for i in range(1, 11))
    ).encode()
    service = IntakeService(tmp_path / "intake")
    session = service.create("根据客户问题生成标准答复", "qa.csv", sample)
    session = service.apply_analysis(
        session,
        propose_baseline_analysis(session, target_column="标准答复", group_columns=("编号",)),
        model="e2e",
    )
    session = service.confirm(session.session_id, session.revision)
    session = service.validate_full_data(session.session_id, session.revision, "f.csv", full)
    session = service.confirm_full_data(session.session_id, session.revision)
    return service, session


def test_open_generation_review_gate_eval_and_metric_identity(open_session, tmp_path):
    """审阅关卡放行 → open_review 对照落盘 → 决策指标=manual_review 全程一致。"""
    service, session = open_session

    # ① 盲标审阅模式放行(整链前置:开放目标核验是逐条审阅)
    pending = service.start_label_verification(session.session_id, session.revision, sample_size=2)
    assert pending["mode"] == "review"
    service.submit_label_verification(
        session.session_id,
        pending["verification_id"],
        {item["row_id"]: "是" for item in pending["items"]},
    )
    session = service.materialize_dataset(session.session_id, session.revision)
    assert session.label_verification["verdict"] == "verified"

    # ② open_review 对照(runtime 全 mock,真实落盘)
    evaluations = BusinessEvaluationService(tmp_path / "eval")

    def fake_runtime(model):
        class Runtime:
            def generate(self, prompt, protocol):
                return Generation("一段完整的生成答复文本。")

            def close(self):
                pass

        return Runtime()

    report = evaluations.compare(
        service.load(session.session_id),
        [
            EvaluationModel("基座", "/tmp/base"),
            EvaluationModel("本轮微调", "/tmp/base", adapter_path="/tmp/adapter"),
        ],
        EvaluationProtocol("open_review", max_new_tokens=64),
        runtime_factory=fake_runtime,
    )
    assert report.status in {"completed", "completed_with_failures"}
    assert report.protocol["scorer"] == "open_review"
    # open_review 的 exact_match 为 None——不是「全错(0.0)」而是「无此指标」,
    # 与 decision_metric=manual_review 的口径互相印证
    assert all(model["metrics"]["exact_match"] is None for model in report.models)

    # ③ 协议身份→迭代决策指标:同一份报告口径推出 manual_review,不漂移
    assert decision_metric_for_protocol(report.protocol) == "manual_review"
    # 回读落盘报告后再取口径,仍一致(存盘不丢协议身份)
    reloaded = evaluations.get_report(report.evaluation_id)
    assert decision_metric_for_protocol(reloaded.protocol) == "manual_review"
