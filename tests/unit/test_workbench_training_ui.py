"""Workbench interactions dispatch only explicit training actions to the service."""

import json
from copy import deepcopy

import pytest

pytest.importorskip("streamlit")

from tests.unit import test_data_intake_ui as intake_ui
from tests.unit.test_full_data import FULL, approved

data_page = intake_ui.data_page
button = intake_ui.button


@pytest.fixture()
def training_page(data_page, monkeypatch):
    import src.workbench.training_runs

    service, _, page = data_page
    session = approved(service)
    session = service.validate_full_data(session.session_id, session.revision, "full.csv", FULL)
    session = service.confirm_full_data(session.session_id, session.revision)
    session = service.materialize_dataset(session.session_id, session.revision)
    calls = []
    records = []

    class TrainingFixture:
        def __init__(self, root, project_root=None):
            # 真实 TrainingRunService 公开 root(页面注册区与合并导出盘点都读它)。
            self.root = root

        def list_runs(self, session_id=None):
            return deepcopy(records)

        def list_runs_tolerant(self, session_id=None):
            return self.list_runs(session_id), 0

        def prepare(self, current, model_path, **options):
            calls.append(("prepare", current.session_id, model_path, options))
            run = {
                "run_id": "fixture-run",
                "session_id": current.session_id,
                "status": "prepared",
                "model_path": model_path,
                "dataset_version": current.dataset.version,
                "output_dir": "/tmp/fixture-adapter",
                "config": options,
                "preflight": {"status": "passed", "issues": []},
                "issues": [],
            }
            records.append(run)
            return deepcopy(run)

        def get_status(self, run_id):
            return deepcopy(records[0])

        def read_logs(self, run_id, tail=100):
            return (
                "fixture: training progress"
                if records[0]["status"] in {"running", "stopped"}
                else ""
            )

        def start(
            self,
            run_id,
            current,
            *,
            acknowledge_warnings=False,
            recover_technical_failures=False,
            recovery_intake_root=None,
        ):
            if recover_technical_failures:
                calls.append(("recovery_opt_in", run_id, True))
                calls.append(("recovery_store", recovery_intake_root))
            calls.append(("start", run_id, current.session_id, acknowledge_warnings))
            records[0]["status"] = "running"
            return deepcopy(records[0])

        def stop(self, run_id):
            calls.append(("stop", run_id))
            records[0]["status"] = "stopped"
            return deepcopy(records[0])

    monkeypatch.setattr(src.workbench.training_runs, "TrainingRunService", TrainingFixture)
    return service, session, page, calls, records


def test_same_page_prepare_start_logs_stop_are_explicit_actions(training_page):
    _, session, page, calls, records = training_page
    page.run()
    page.selectbox(key="intake_select").select(session.session_id).run()
    assert not page.exception
    assert calls == []
    next(t for t in page.text_input if t.label == "本地基础模型目录").input("/tmp/local-base")
    button(page, "准备本轮训练方案").click().run()
    assert not page.exception
    assert calls[0][:3] == ("prepare", session.session_id, "/tmp/local-base")
    assert calls[0][3]["model_options"] == {"quantization_bits": None}
    assert records[0]["dataset_version"] == session.dataset.version
    assert not any(call[0] == "start" for call in calls)
    button(page, "启动这轮训练").click().run()
    assert not page.exception
    assert calls[-1] == ("start", "fixture-run", session.session_id, False)
    assert any("fixture: training progress" in block.value for block in page.code)
    button(page, "刷新训练状态和日志").click().run()
    assert len([call for call in calls if call[0] == "start"]) == 1
    button(page, "停止这轮训练").click().run()
    assert not page.exception
    assert calls[-1] == ("stop", "fixture-run")
    assert any("已停止" in block.label for block in page.expander)


def test_running_training_shows_live_curve_without_verdict(training_page, tmp_path):
    """环节⑤「实时曲线」:训练中(running)页面读实时序列画曲线并提示刷新,
    不给三态判定——半程数据不足以支持整场结论；即使有临时指标也只画一张曲线。"""
    from types import SimpleNamespace

    from transformers import Trainer
    from transformers.trainer_callback import CallbackHandler, TrainerControl, TrainerState

    from src.workbench.live_loss import LiveLossWriter

    _, session, page, _, records = training_page
    live_output = tmp_path / "live-run"
    live_output.mkdir()
    trainer = Trainer.__new__(Trainer)
    trainer.state = TrainerState()
    trainer.args = SimpleNamespace(include_num_input_tokens_seen="no")
    trainer.control = TrainerControl()
    trainer.callback_handler = CallbackHandler(
        [LiveLossWriter(live_output)], None, None, None, None
    )
    for step in range(6):
        trainer.state.global_step = step
        trainer.log({"loss": 2.0 - 0.1 * step})
    page.run()
    page.selectbox(key="intake_select").select(session.session_id).run()
    next(t for t in page.text_input if t.label == "本地基础模型目录").input("/tmp/local-base").run()
    button(page, "准备本轮训练方案").click().run()
    assert not page.exception
    records[0]["output_dir"] = str(live_output)
    # Some workers publish preliminary metrics while still running; they do
    # not turn a partial curve into a completed-training trend.
    records[0]["metrics"] = {"train_loss": 1.0}
    button(page, "启动这轮训练").click().run()
    assert not page.exception
    assert any("训练中的 loss 曲线" in block.value for block in page.markdown)
    assert any("刷新页面查看最新进度" in block.value for block in page.caption)
    assert not any("整体在下降" in block.value for block in page.caption)
    assert any("本轮训练指标" in block.value for block in page.markdown)
    curves = [json.loads(chart.proto.spec) for chart in page.get("plotly_chart")]
    assert sum(trace.get("name") == "loss" for chart in curves for trace in chart["data"]) == 1


def test_identical_live_curves_from_different_runs_remain_visible(training_page, tmp_path):
    _, session, page, _, records = training_page
    for index in range(2):
        output = tmp_path / f"live-run-{index}"
        output.mkdir()
        (output / "workbench_loss_history.json").write_text(
            json.dumps([{"step": 1, "loss": 2.0}, {"step": 2, "loss": 1.0}]),
            encoding="utf-8",
        )
        records.append(
            {
                "run_id": f"fixture-{index}",
                "session_id": session.session_id,
                "status": "running",
                "model_path": "/tmp/local-base",
                "dataset_version": session.dataset.version,
                "output_dir": str(output),
                "config": {},
                "preflight": {},
            }
        )
    page.run()
    page.selectbox(key="intake_select").select(session.session_id).run()
    assert not page.exception, [item.message for item in page.exception]
    charts = [json.loads(chart.proto.spec) for chart in page.get("plotly_chart")]
    assert sum(trace.get("name") == "loss" for chart in charts for trace in chart["data"]) == 2


def test_freeze_question_suite_does_not_rewrite_parent_training_data(training_page):
    service, session, page, _, _ = training_page
    page.run()
    page.selectbox(key="intake_select").select(session.session_id).run()
    assert not page.exception
    button(page, "固定当前开发与测试题集").click().run()
    assert not page.exception
    assert any("已固定题集" in message.value for message in page.success)
    current = service.load(session.session_id)
    assert current.dataset.version == session.dataset.version
    assert current.revision == session.revision


def test_preflight_warnings_require_review_and_failures_remain_visible(training_page):
    _, session, page, calls, records = training_page
    records.append(
        {
            "run_id": "fixture-run",
            "status": "prepared",
            "dataset_version": session.dataset.version,
            "preflight": {
                "status": "warnings",
                "issues": [{"severity": "warning", "message": "部分上下文被截断"}],
            },
            "config": {},
            "issues": [],
        }
    )
    page.run()
    page.selectbox(key="intake_select").select(session.session_id).run()
    assert button(page, "启动这轮训练").disabled
    next(c for c in page.checkbox if c.label.startswith("已核对任务规约与预检提示")).check().run()
    button(page, "启动这轮训练").click().run()
    assert calls[-1][-1] is True
    records[0].update(status="failed", failure={"stage": "training", "message": "fixture failure"})
    # A fresh session replays the user's refresh after the failure: the warning
    # checkbox is unmounted with the run, so reusing the same AppTest session
    # trips Streamlit's widget-state eviction on this version.
    reloaded = intake_ui.AppTest.from_file(str(intake_ui.PAGE), default_timeout=20)
    reloaded.run()
    reloaded.selectbox(key="intake_select").select(session.session_id).run()
    assert not reloaded.exception
    assert any("fixture failure" in message.value for message in reloaded.error)
    assert not any(b.label == "启动这轮训练" for b in reloaded.button)


def test_prepared_run_shows_task_spec_before_launch_button(training_page):
    """R60: 能启动(prepared 未启动)时,启动按钮前仍可见任务规约折叠区,与规约卡同源同词汇。"""
    _, session, page, _, _ = training_page
    page.run()
    page.selectbox(key="intake_select").select(session.session_id).run()
    assert not page.exception
    next(t for t in page.text_input if t.label == "本地基础模型目录").input("/tmp/local-base")
    button(page, "准备本轮训练方案").click().run()
    assert not page.exception
    labels = [expander.label for expander in page.expander]
    assert any(label == "📋 任务规约（启动本轮训练前的口径）" for label in labels)
    texts = [block.value for block in page.markdown]
    assert any("的任务规约：由既有确认记录只读汇编" in text for text in texts)
    assert any("不代表模型效果达标" in text for text in texts)


@pytest.mark.parametrize("task_kind", ["categorical", "open_text", "iteration"])
def test_successful_training_compares_complete_outputs_and_marks_open_tasks(
    training_page, monkeypatch, tmp_path, task_kind
):
    import src.agent.evaluation as evaluation_agent
    import src.workbench.business_evaluation as evaluation

    service, session, page, _, records = training_page
    if task_kind == "open_text":
        session = approved(service, target_kind="open_text")
        session = service.validate_full_data(session.session_id, session.revision, "full.csv", FULL)
        session = service.confirm_full_data(session.session_id, session.revision)
        session = service.materialize_dataset(session.session_id, session.revision)
    # 逐条 loss 序列落在产物目录里：页面趋势人话与 CLI train-status 读同一份文件。
    adapter_output = tmp_path / "adapter"
    adapter_output.mkdir()
    (adapter_output / "workbench_loss_history.json").write_text(
        json.dumps([{"step": s, "loss": 2.0 - 0.1 * s} for s in range(6)]),
        encoding="utf-8",
    )
    records.append(
        {
            "run_id": "fixture-run",
            "session_id": session.session_id,
            "status": "succeeded",
            "model_path": "/tmp/base",
            "output_dir": str(adapter_output),
            "dataset_version": session.dataset.version,
            "config": {},
            "metrics": {"train_loss": 0.5},
            "issues": [],
        }
    )
    iteration_calls = []
    if task_kind == "iteration":
        import src.workbench.iterations as iteration_module
        import src.workbench.training_runs as training_module
        from src.workbench.evaluation_suites import EvalSuiteService

        suite = EvalSuiteService(service.root / "suites").freeze(session)
        session = service.materialize_dataset(
            session.session_id, session.revision, evaluation_suite=suite
        )
        records[0]["dataset_version"] = session.dataset.version
        iteration = {
            "iteration_id": "it-" + "b" * 32,
            "status": "running",
            "session_id": session.session_id,
            "parent_run_id": "parent-run",
            "new_run_id": "fixture-run",
            "evaluation_suite": suite,
            "hypothesis": "历史错误可能来自监督覆盖不足",
            "expected_outcome": "同题表现改善",
            "changes": "增加训练独立样例",
        }

        class Iterations:
            def __init__(self, *args):
                pass

            def list_iterations(self, session_id=None):
                return [deepcopy(iteration)]

            def bind_evaluation(self, identity, current, evaluation_id):
                iteration_calls.append(("bind", identity, current.session_id, evaluation_id))
                iteration["status"] = "evaluated"

            def decide(self, identity, decision, reason):
                iteration_calls.append(("decide", identity, decision, reason))
                iteration.update(status="decided", decision=decision, decision_reason=reason)

        monkeypatch.setattr(iteration_module, "IterationService", Iterations)
        original_status = training_module.TrainingRunService.get_status

        def get_status(self, run_id):
            if run_id == "parent-run":
                return {
                    "run_id": run_id,
                    "status": "succeeded",
                    "model_path": "/tmp/parent-base",
                    "output_dir": "/tmp/parent-adapter",
                }
            return original_status(self, run_id)

        monkeypatch.setattr(training_module.TrainingRunService, "get_status", get_status)
    reports, calls, assessments, assessment_calls = [], [], [], []

    def assess(report, current, client, *, output_root):
        assessment_calls.append((report.evaluation_id, current.session_id, client.model))
        record = {
            "evaluation_id": report.evaluation_id,
            "session_id": current.session_id,
            "session_revision": current.revision,
            "model": client.model,
            "assessment": {
                "summary": "先核查回答截断",
                "observations": [
                    {"statement": "存在未完整生成的答案", "evidence_ids": ["fixture-evidence-2"]}
                ],
                "hypotheses": [
                    {
                        "statement": "停止条件可能不合适",
                        "evidence_ids": ["fixture-evidence-2"],
                        "verification": "检查实际停止token与完整输出",
                    }
                ],
                "next_steps": ["核查真实格式与停止行为"],
                "decision": "collect_evidence",
                "limitations": ["尚未证明原因，不能直接认定需要多训几轮"],
                "business_questions": [],
            },
            "tool_trace": [{"tool": "inspect_bad_cases", "ok": True}],
        }
        assessments.append(record)
        return record

    monkeypatch.setattr(evaluation_agent, "assess_evaluation", assess)
    monkeypatch.setattr(
        evaluation_agent, "load_assessments", lambda root, evaluation_id: assessments
    )

    class EvaluationFixture:
        def __init__(self, root):
            self.load_errors = [{"evaluation_id": "b" * 32, "reason": "业务指标与全量评测行不一致"}]

        def list_reports(self, dataset_version=None, purpose=None):
            return reports

        def compare(self, current, models, protocol):
            from dataclasses import asdict

            calls.append((models, protocol))
            open_task = protocol.scorer == "open_review"
            result = evaluation.EvaluationReport(
                "a" * 32,
                "now",
                {"version": current.dataset.version, "split": "validation"},
                {"scorer": protocol.scorer},
                "comparison-key",
                status="completed_with_failures",
            )
            for model in models:
                rows = [
                    {
                        "index": 0,
                        "source": {"source_row_id": "r000001"},
                        "prompt": "完整业务输入",
                        "expected": "期望答案",
                        "output": f"{model.label}的完整输出",
                        "status": "needs_business_review" if open_task else "scored",
                        "correct": None if open_task else True,
                        "field_scores": {},
                        "truncated": False,
                        "error": "",
                    },
                    {
                        "index": 1,
                        "source": {"source_row_id": "r000002"},
                        "prompt": "另一输入",
                        "expected": "另一答案",
                        "output": "被截断的片段",
                        "status": "truncated",
                        "correct": None if open_task else False,
                        "field_scores": {},
                        "truncated": True,
                        "error": "达到生成长度限制",
                    },
                ]
                result.models.append(
                    {
                        "label": model.label,
                        "requested_model": asdict(model),
                        "rows": rows,
                        "errors": [],
                        "metrics": {
                            "total": 2,
                            "scored": 0 if open_task else 1,
                            "exact_match": None if open_task else 0.5,
                            "field_accuracy": {},
                        },
                    }
                )
            reports.append(result)
            return result

    monkeypatch.setattr(evaluation, "BusinessEvaluationService", EvaluationFixture)
    page.run()
    page.selectbox(key="intake_select").select(session.session_id).run()
    assert calls == []
    # 环节⑨交接出口在场:成功记录下合并导出折叠区做只读盘点,页面不执行合并。
    assert any(block.label == "📦 合并导出：把这次训练的模型带出工作台" for block in page.expander)
    # 环节⑤可视化:训练指标区给逐条 loss 趋势人话(与 train-status 同源),
    # 原始 flat JSON 收进折叠区,不再裸倾倒。
    assert any("整体在下降" in block.value for block in page.caption)
    assert any("不代表业务效果" in block.value for block in page.caption)
    assert any(block.label == "查看原始指标 JSON" for block in page.expander)
    button(page, "比较基座与本轮微调效果").click().run()
    assert not page.exception
    assert any(
        "评测 bbbbbbbb 的证据无法核验，已隐藏" in message.value
        and "其余有效报告仍可查看" in message.value
        for message in page.warning
    )
    assert calls[0][1].scorer == (
        "open_review" if task_kind == "open_text" else "classification_exact"
    )
    assert any(block.value == "期望答案" for block in page.code)
    assert any(block.value == "基座的完整输出" for block in page.code)
    assert any(block.value == "本轮微调的完整输出" for block in page.code)
    # 一半样本被截断，达到高比例阈值：对照区给出 max_new_tokens 核查提示（观察事实，不认定原因）。
    # R102:警告句与语言化摘要同出 truncation_warning_sentence(单一来源),
    # 页面旧手抄 lead-in「高比例输出截断」退场,钉改指 builder 输出原文。
    assert any(
        "多个输出因触及生成长度上限被截断" in message.value and "max_new_tokens" in message.value
        for message in page.warning
    )
    assert any("触及上限不等于只需增加长度" in message.value for message in page.warning)
    if task_kind == "iteration":
        assert len(calls[0][0]) == 3
        assert calls[0][0][1].base_model == "/tmp/parent-base"
        assert calls[0][0][1].adapter_path == "/tmp/parent-adapter"
        assert any(block.value == "父轮模型的完整输出" for block in page.code)
        assert iteration_calls[0][0] == "bind"
        assert iteration_calls[0][3] == "a" * 32
    if task_kind == "open_text":
        assert any("尚无业务评分规则" in message.value for message in page.warning)
    selector = next(
        select for select in page.selectbox if select.label == "逐样本查看完整输入、期望和输出"
    )
    selector.select(1).run()
    assert not page.exception
    assert any(block.value == "被截断的片段" for block in page.code)
    assert len(calls) == 1
    assert assessment_calls == []
    next(field for field in page.text_input if field.label == "支持工具调用的模型名称").input(
        "tool-fixture"
    ).run()
    original_revision = service.load(session.session_id).revision
    button(page, "让 Agent 分析结果与下一步").click().run()
    assert not page.exception
    assert assessment_calls == [("a" * 32, session.session_id, "tool-fixture")]
    assert any("先核查回答截断" in item.value for item in page.markdown)
    assert any("fixture-evidence-2" in item.value for item in page.caption)
    assert service.load(session.session_id).revision == original_revision
    page.run()
    assert len(assessment_calls) == 1
    assert any("先核查回答截断" in item.value for item in page.markdown)
    next(field for field in page.text_input if field.label == "模型服务 API 地址").input(
        "https://fixture.example/v1"
    ).run()
    assert button(page, "让 Agent 分析结果与下一步").disabled
    next(
        check
        for check in page.checkbox
        if check.label.startswith("允许向已选 Agent 服务发送本次目标")
    ).check().run()
    assert button(page, "让 Agent 分析结果与下一步").disabled
    next(check for check in page.checkbox if check.key == "agent_remote_consent").check().run()
    assert not button(page, "让 Agent 分析结果与下一步").disabled
    assert len(assessment_calls) == 1


def test_confirmed_iteration_rebinds_existing_dataset_before_preparing_and_starting(
    training_page, monkeypatch
):
    import src.workbench.iterations as iteration_module
    from src.workbench.evaluation_suites import EvalSuiteService

    service, session, page, calls, records = training_page
    suite = EvalSuiteService(service.root / "suites").freeze(session)
    iteration = {
        "iteration_id": "it-" + "c" * 32,
        "status": "confirmed",
        "session_id": session.session_id,
        "parent_run_id": "parent-run",
        "new_run_id": None,
        "evaluation_suite": suite,
        "hypothesis": "调整训练轮数可能改善格式",
        "expected_outcome": "严格匹配错误减少",
        "changes": "仅调整训练轮数",
        "data_change": False,
    }

    class Iterations:
        def __init__(self, *args):
            pass

        def list_iterations(self, session_id=None):
            return [deepcopy(iteration)]

        def prepare(self, identity, current):
            assert current.dataset.evaluation_suite == suite
            calls.append(("iteration_prepare", identity, current.dataset.version))
            records.append(
                {
                    "run_id": "next-run",
                    "status": "prepared",
                    "dataset_version": current.dataset.version,
                    "preflight": {"status": "passed"},
                }
            )
            iteration.update(status="prepared", new_run_id="next-run")

        def start(self, identity, current, *, acknowledge_warnings):
            calls.append(("iteration_start", identity, acknowledge_warnings))
            records[0]["status"] = "running"
            iteration["status"] = "running"

    monkeypatch.setattr(iteration_module, "IterationService", Iterations)
    page.run()
    page.selectbox(key="intake_select").select(session.session_id).run()
    assert not page.exception
    assert button(page, "按已确认范围准备下一轮训练").disabled
    button(page, "用本轮固定题集准备数据版本").click().run()
    assert not page.exception
    assert service.load(session.session_id).dataset.evaluation_suite == suite
    assert calls == []
    button(page, "按已确认范围准备下一轮训练").click().run()
    assert not page.exception
    assert calls[-1][0] == "iteration_prepare"
    assert len(calls) == 1
    button(page, "启动这轮训练").click().run()
    assert not page.exception
    assert calls[-1] == ("iteration_start", iteration["iteration_id"], False)
    page.run()
    assert len(calls) == 2


def test_technical_recovery_requires_opt_in_and_keeps_failed_parent_visible(training_page):
    service, session, page, calls, records = training_page
    records.append(
        {
            "run_id": "fixture-run",
            "status": "prepared",
            "dataset_version": session.dataset.version,
            "preflight": {"status": "passed"},
        }
    )
    page.run()
    page.selectbox(key="intake_select").select(session.session_id).run()
    assert not page.exception
    consent = page.checkbox(key="recover_fixture-run")
    assert consent.value is False
    assert calls == []
    consent.check().run()
    button(page, "启动这轮训练").click().run()
    assert not page.exception
    assert ("recovery_opt_in", "fixture-run", True) in calls
    assert ("recovery_store", service.root) in calls
    records[0].update(
        status="failed",
        failure={"stage": "training", "message": "CUDA out of memory"},
        recovery={
            "status": "retry_started",
            "child_run_id": "retry-run",
            "reason": "保持有效 batch 的显存恢复",
            "changes": {"batch_size": {"before": 2, "after": 1}},
        },
    )
    previous_calls = len(calls)
    page.run()
    assert not page.exception
    assert any("CUDA out of memory" in item.value for item in page.error)
    assert any("retry-run" in item.value for item in page.info)
    assert len(calls) == previous_calls
    button(page, "停止此训练及其自动恢复").click().run()
    assert calls[-1] == ("stop", "fixture-run")


def test_preparing_recovery_child_cannot_bypass_recovery_checks_with_manual_start(training_page):
    _, session, page, calls, records = training_page
    records.append(
        {
            "run_id": "retry-run",
            "status": "prepared",
            "recovery_parent_run_id": "failed-parent",
            "dataset_version": session.dataset.version,
            "preflight": {"status": "passed"},
        }
    )
    page.run()
    page.selectbox(key="intake_select").select(session.session_id).run()
    assert not page.exception
    assert not any(item.label == "启动这轮训练" for item in page.button)
    assert any("由获准恢复流程管理" in item.value for item in page.info)
    assert calls == []


def test_stale_prepared_run_cannot_offer_a_dead_start_button(training_page, monkeypatch):
    """旧数据版本的 prepared 方案不再渲染可点击的启动按钮（点击必被服务拒绝）。"""
    import src.workbench.training_runs as training_module

    service, session, page, calls, records = training_page
    records.append(
        {
            "run_id": "stale-run",
            "session_id": session.session_id,
            "status": "prepared",
            "dataset_version": "older-version",
            "model_path": "/tmp/base",
            "output_dir": "/tmp/adapter",
            "preflight": {"status": "passed", "issues": []},
            "config": {},
            "issues": [],
        }
    )
    records.append(
        {
            "run_id": "current-run",
            "session_id": session.session_id,
            "status": "prepared",
            "dataset_version": session.dataset.version,
            "model_path": "/tmp/base",
            "output_dir": "/tmp/adapter2",
            "preflight": {"status": "passed", "issues": []},
            "config": {},
            "issues": [],
        }
    )
    by_id = {record["run_id"]: record for record in records}

    def lookup(self, run_id):
        return deepcopy(by_id[run_id])

    monkeypatch.setattr(training_module.TrainingRunService, "get_status", lookup)
    page.run()
    page.selectbox(key="intake_select").select(session.session_id).run()
    assert not page.exception
    starts = [b for b in page.button if b.label == "启动这轮训练"]
    assert [b.disabled for b in starts] == [False]  # 只有当前数据版本的方案可启动
    assert any("旧数据版本" in caption.value for caption in page.caption)
    assert calls == []


def test_manual_training_parameters_have_plain_language_guidance(training_page):
    """手工训练参数表单带大白话指引与推荐起步值,零密钥用户不必盲填参数。"""
    _, session, page, _, _ = training_page
    page.run()
    page.selectbox(key="intake_select").select(session.session_id).run()
    assert not page.exception
    captions = "\n".join(caption.value for caption in page.caption)
    assert "参数大白话" in captions
    for keyword in (
        "训练轮数",
        "每设备 batch size",
        "梯度累积步数",
        "学习率",
        "LoRA rank",
        "4-bit 量化",
        "最大 token 长度",
    ):
        assert keyword in captions, f"缺少参数「{keyword}」的大白话解释"
    assert "推荐起步值（小数据）" in captions
    assert "0.0002" in captions and "2e-4" in captions
    assert "LoRA rank 8" in captions
    # 单一来源逐行渲染：每条大白话是独立 caption，词形与 src.workbench.training_guidance
    # 同源（参数名无加粗标记），页面不再自带一份内联文案
    assert any(caption.value.startswith("参数大白话：") for caption in page.caption)
    assert any("LoRA rank＝适配器" in caption.value for caption in page.caption)
    assert any("4-bit 量化＝" in caption.value for caption in page.caption)
    assert any(caption.value.startswith("推荐起步值（小数据）：") for caption in page.caption)
    # 指引与表单同时在场:默认值即推荐起步值,用户可以直接准备训练
    next(t for t in page.text_input if t.label == "本地基础模型目录")
    next(b for b in page.button if b.label == "准备本轮训练方案")


def test_learning_rate_tier_suggestion_fills_lower_tier_on_request(training_page):
    """<2,000 条训练集:默认保持 2e-4，点击「采用建议学习率」才填入 1e-4，仍可手改。"""
    _, session, page, calls, _ = training_page
    page.run()
    page.selectbox(key="intake_select").select(session.session_id).run()
    assert not page.exception
    lr_input = next(field for field in page.number_input if field.label == "学习率")
    assert lr_input.value == pytest.approx(0.0002)
    captions = "\n".join(caption.value for caption in page.caption)
    # 夹具各分区 1 行：按训练集降档，并明示依据为外部指南、非本产品实测。
    assert "训练集 1 条（< 2,000）" in captions
    assert "5e-5~1e-4" in captions
    assert "非本产品实测" in captions
    button(page, "采用建议学习率").click().run()
    assert not page.exception
    lr_input = next(field for field in page.number_input if field.label == "学习率")
    assert lr_input.value == pytest.approx(0.0001)
    # 建议只是预填:用户仍可手改,且手改值进入训练准备参数
    next(field for field in page.number_input if field.label == "学习率").set_value(0.0003).run()
    next(field for field in page.text_input if field.label == "本地基础模型目录").input(
        "/tmp/local-base"
    )
    button(page, "准备本轮训练方案").click().run()
    assert not page.exception
    assert calls[0][3]["training_options"]["learning_rate"] == pytest.approx(0.0003)


@pytest.mark.parametrize("bad_count", [None, 0, -1, True, "1"])
def test_learning_rate_suggestion_does_not_invent_missing_training_count(
    training_page, monkeypatch, bad_count
):
    from src.workbench.intake_service import IntakeService

    _, session, page, _, _ = training_page
    original_load = IntakeService.load

    def load_with_damaged_statistics(service, session_id):
        result = original_load(service, session_id)
        if session_id == session.session_id:
            result.dataset.statistics["row_counts"]["train"] = bad_count
        return result

    monkeypatch.setattr(IntakeService, "load", load_with_damaged_statistics)
    page.run()
    page.selectbox(key="intake_select").select(session.session_id).run()
    assert not page.exception
    assert any("训练集条数缺失或无效" in caption.value for caption in page.caption)
    assert not any(item.label == "采用建议学习率" for item in page.button)
    assert next(item for item in page.number_input if item.label == "学习率").value == 0.0002


def test_learning_rate_suggestion_counts_train_only_for_temporal_exclusions(data_page):
    from tests.unit.test_temporal_intake_entrypoints import temporal_session

    service, _, page = data_page
    session = temporal_session(service)
    session = service.materialize_dataset(session.session_id, session.revision)
    assert session.dataset.statistics["total_rows"] == 5
    assert session.dataset.statistics["included_rows"] == 3
    assert session.dataset.statistics["row_counts"]["train"] == 1
    page.run()
    page.selectbox(key="intake_select").select(session.session_id).run()
    assert not page.exception
    assert any("训练集 1 条（< 2,000）" in caption.value for caption in page.caption)
    button(page, "采用建议学习率").click().run()
    assert not page.exception
    assert next(item for item in page.number_input if item.label == "学习率").value == 0.0001


def test_learning_rate_tier_threshold_ignores_validation_and_test(training_page, monkeypatch):
    from src.workbench.intake_service import IntakeService

    _, session, page, _, _ = training_page
    original_load = IntakeService.load

    def load_with_large_evaluation_sets(service, session_id):
        result = original_load(service, session_id)
        if session_id == session.session_id:
            result.dataset.statistics["row_counts"] = {
                "train": 1999,
                "validation": 100,
                "test": 100,
            }
        return result

    monkeypatch.setattr(IntakeService, "load", load_with_large_evaluation_sets)
    page.run()
    page.selectbox(key="intake_select").select(session.session_id).run()
    assert not page.exception
    assert any("训练集 1999 条（< 2,000）" in item.value for item in page.caption)
    button(page, "采用建议学习率").click().run()
    assert next(item for item in page.number_input if item.label == "学习率").value == 0.0001


def test_export_controls_dispatch_only_explicit_gguf_action(training_page, monkeypatch, tmp_path):
    from src.workbench import model_export

    _, session, page, _, records = training_page
    records.append(
        {
            "run_id": "fixture-export",
            "session_id": session.session_id,
            "status": "succeeded",
            "dataset_version": session.dataset.version,
            "model_path": "/tmp/base",
            "output_dir": "/tmp/adapter",
            "metrics": {},
            "artifacts": {},
        }
    )
    exports = []

    def plan(run, output, **options):
        missing = options["export_format"] == "gguf" and not options["llama_cpp_dir"]
        return {
            "status": "toolchain_missing" if missing else "ready",
            "run_id": run["run_id"],
            "output_dir": str(output),
            "reasons": ["请提供本地 llama.cpp 目录。"] if missing else [],
            **options,
        }

    def export(planned):
        exports.append(deepcopy(planned))
        return {
            **planned,
            "status": "exported",
            "gguf_file": str(tmp_path / "custom output" / "model-q5_k_m.gguf"),
            "modelfile": str(tmp_path / "custom output" / "Modelfile"),
        }

    monkeypatch.setattr(model_export, "plan_model_export", plan)
    monkeypatch.setattr(model_export, "export_model", export)
    page.run()
    page.selectbox(key="intake_select").select(session.session_id).run()
    assert not page.exception
    assert exports == []
    page.selectbox(key="export_format_fixture-export").select("gguf").run()
    assert button(page, "导出模型").disabled
    assert any("请提供本地 llama.cpp" in item.value for item in page.caption)
    page.text_input(key="export_llama_cpp_fixture-export").input(
        str(tmp_path / "my llama.cpp")
    ).run()
    page.selectbox(key="export_quantization_fixture-export").select("q5_k_m").run()
    page.text_input(key="export_output_fixture-export_gguf").input(
        str(tmp_path / "custom output")
    ).run()
    assert exports == [], "格式与路径变更、页面刷新都不能触发实际导出"
    assert not button(page, "导出模型").disabled
    button(page, "导出模型").click().run()
    assert not page.exception
    assert len(exports) == 1
    assert exports[0]["export_format"] == "gguf"
    assert exports[0]["quantization"] == "q5_k_m"
    assert exports[0]["llama_cpp_dir"] == str(tmp_path / "my llama.cpp")
    assert exports[0]["output_dir"] == str(tmp_path / "custom output")
    assert any("GGUF 导出完成" in item.value for item in page.success)
    assert any("ollama create" in item.value for item in page.caption)
    page.run()
    assert len(exports) == 1


@pytest.mark.parametrize("valid_export", [True, False])
def test_gguf_same_question_action_runs_existing_task_and_displays_own_results(
    training_page, monkeypatch, tmp_path, valid_export
):
    import sys
    from dataclasses import asdict
    from types import ModuleType

    from scripts import setup_gguf_tools
    from src.workbench import business_evaluation as evaluation

    monkeypatch.setattr(setup_gguf_tools, "default_destination", lambda: tmp_path / "missing-tools")
    _, session, page, _, records = training_page
    records.append(
        {
            "run_id": "fixture-gguf",
            "session_id": session.session_id,
            "status": "succeeded",
            "dataset_version": session.dataset.version,
            "model_path": "/tmp/base",
            "output_dir": "/tmp/adapter",
            "metrics": {},
            "artifacts": {},
        }
    )
    calls, helper_calls, reports = [], [], []
    directory = tmp_path / "exported model"
    runtime = tmp_path / "tools" / "tunesmith-gguf-eval"
    helper = ModuleType("src.workbench.gguf_evaluation")

    def gguf_model(run, export_dir, runtime_path):
        helper_calls.append((run["run_id"], export_dir, runtime_path))
        if not valid_export:
            raise ValueError("GGUF 缺少 hf_tokenizer，请重新导出到新目录。")
        return evaluation.EvaluationModel(
            "导出 GGUF",
            str(export_dir / "hf_tokenizer"),
            gguf_path=str(export_dir / "model-q4_k_m.gguf"),
            gguf_runtime_path=str(runtime_path),
        )

    helper.gguf_evaluation_model = gguf_model
    monkeypatch.setitem(sys.modules, "src.workbench.gguf_evaluation", helper)

    class Evaluation:
        def __init__(self, root):
            self.load_errors = []

        def list_reports(self, dataset_version=None, purpose=None):
            return reports

        def compare(self, current, models, protocol):
            calls.append((current.session_id, models, protocol))
            report = evaluation.EvaluationReport(
                "c" * 32,
                "now",
                {"version": current.dataset.version},
                {
                    "scorer": protocol.scorer,
                    "max_new_tokens": protocol.max_new_tokens,
                    "generation": "greedy_token_contract_v2",
                },
                "comparison-key",
                status="completed_with_failures",
            )
            for model in models:
                gguf = bool(model.gguf_path)
                rows = [
                    {
                        "index": index,
                        "source": {"source_row_id": f"r{index + 1:06d}"},
                        "prompt": f"完整业务问题{index}",
                        "expected": "答案",
                        "output": "答案",
                        "status": "scored",
                        "correct": True,
                        "field_scores": {},
                        "truncated": False,
                        "generated_tokens": 2,
                        "error": "",
                    }
                    for index in range(3)
                ]
                if gguf:
                    rows[1].update(
                        status="failed", output=None, correct=False, error="运行器实际失败"
                    )
                    rows[2].update(status="truncated", correct=False, truncated=True)
                report.models.append(
                    {
                        "label": model.label,
                        "requested_model": asdict(model),
                        "rows": rows,
                        "identity": {"format": "gguf" if gguf else "hf"},
                        "errors": [],
                        "metrics": {
                            "total": 3,
                            "scored": 1 if gguf else 3,
                            "exact_match": 1 / 3 if gguf else 1.0,
                            "field_accuracy": {},
                        },
                    }
                )
            reports.append(report)
            return report

    monkeypatch.setattr(evaluation, "BusinessEvaluationService", Evaluation)
    page.run()
    page.selectbox(key="intake_select").select(session.session_id).run()
    assert not page.exception
    assert button(page, "GGUF 与本轮模型同题对照").disabled
    page.text_input(key="eval_gguf_dir_fixture-gguf").input(str(directory)).run()
    page.text_input(key="eval_gguf_runtime_fixture-gguf").input(str(runtime)).run()
    assert calls == [] and helper_calls == []
    assert not button(page, "GGUF 与本轮模型同题对照").disabled
    page.number_input(key="eval_tokens_fixture-gguf").set_value(19)
    page.checkbox(key="eval_whitespace_fixture-gguf").uncheck()
    button(page, "GGUF 与本轮模型同题对照").click().run()
    assert not page.exception
    assert helper_calls == [("fixture-gguf", directory, runtime)]
    if not valid_export:
        assert calls == []
        assert any("hf_tokenizer" in error.value for error in page.error)
        return
    assert len(calls) == 1
    assert calls[0][0] == session.session_id
    models, protocol = calls[0][1:]
    assert [model.label for model in models] == ["基座", "本轮微调", "导出 GGUF"]
    assert models[1].adapter_path == "/tmp/adapter"
    assert models[2].adapter_path is None and models[2].gguf_path.endswith("model-q4_k_m.gguf")
    assert protocol.max_new_tokens == 19 and protocol.strip_whitespace is False
    assert protocol.scorer == "classification_exact"
    table = next(item.value for item in page.dataframe if "格式" in item.value.columns)
    row = table[table["格式"] == "GGUF"].iloc[0]
    assert row["严格准确率"] == 1 / 3
    assert row["生成失败"] == 1 and row["输出截断"] == 1
    page.selectbox(key=f"eval_row_fixture-gguf_{'c' * 32}").select(1).run()
    assert any("运行器实际失败" in item.value for item in page.warning)
    metadata = next(
        json.loads(item.value) for item in page.json if "comparison_key" in json.loads(item.value)
    )
    assert metadata["models"][2]["format"] == "GGUF"
    assert metadata["models"][2]["file"] == models[2].gguf_path
    assert metadata["models"][2]["runtime"] == str(runtime)
    page.run()
    assert len(calls) == 1, "刷新不能重新运行已记录的 GGUF 对照"


def test_gguf_compare_prefills_existing_tools_and_follows_export_until_overridden(
    training_page, monkeypatch, tmp_path
):
    import sys

    from scripts import setup_gguf_tools
    from src.workbench import gguf_evaluation

    _, session, page, _, records = training_page
    records.append(
        {
            "run_id": "fixture-defaults",
            "session_id": session.session_id,
            "status": "succeeded",
            "dataset_version": session.dataset.version,
            "model_path": "/tmp/base",
            "output_dir": "/tmp/adapter",
            "metrics": {},
            "artifacts": {},
        }
    )
    cache = tmp_path / "default tools"
    selected = tmp_path / "selected tools"
    name = "tunesmith-gguf-eval.exe" if sys.platform == "win32" else "tunesmith-gguf-eval"
    binaries = []
    for directory in (cache, selected):
        binary = directory / "build-tunesmith-eval" / "bin" / name
        binary.parent.mkdir(parents=True)
        binary.write_text("This file must not be executed during a page render.")
        binaries.append(binary)
    monkeypatch.setattr(setup_gguf_tools, "default_destination", lambda: cache)

    def unexpected(*args, **kwargs):
        pytest.fail("默认路径发现不能执行 GGUF helper 或运行器")

    monkeypatch.setattr(gguf_evaluation, "gguf_evaluation_model", unexpected)
    page.run()
    page.selectbox(key="intake_select").select(session.session_id).run()
    assert not page.exception
    assert page.text_input(key="eval_gguf_runtime_fixture-defaults").value == str(binaries[0])
    page.selectbox(key="export_format_fixture-defaults").select("gguf").run()
    page.text_input(key="export_llama_cpp_fixture-defaults").input(str(selected)).run()
    selected_output = tmp_path / "custom gguf output"
    page.text_input(key="export_output_fixture-defaults_gguf").input(str(selected_output)).run()
    assert page.text_input(key="eval_gguf_dir_fixture-defaults").value == str(selected_output)
    assert page.text_input(key="eval_gguf_runtime_fixture-defaults").value == str(binaries[1])
    manual_dir = str(tmp_path / "manual gguf")
    manual_runtime = str(tmp_path / "manual runtime")
    page.text_input(key="eval_gguf_dir_fixture-defaults").input(manual_dir).run()
    page.text_input(key="eval_gguf_runtime_fixture-defaults").input(manual_runtime).run()
    page.text_input(key="export_output_fixture-defaults_gguf").input(
        str(tmp_path / "next output")
    ).run()
    assert page.text_input(key="eval_gguf_dir_fixture-defaults").value == manual_dir
    assert page.text_input(key="eval_gguf_runtime_fixture-defaults").value == manual_runtime
    assert not page.exception
