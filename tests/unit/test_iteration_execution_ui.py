"""The page drives automatic execution only through explicit user actions."""

from copy import deepcopy

import pytest

pytest.importorskip("streamlit")

from tests.unit import test_data_intake_ui as intake_ui
from tests.unit.test_full_data import FULL, approved

data_page = intake_ui.data_page
button = intake_ui.button

IDENTITY = "it-" + "a" * 32


@pytest.fixture()
def execution_page(data_page, monkeypatch):
    import src.workbench.iteration_execution as execution_module
    import src.workbench.iterations as iterations_module

    service, _, page = data_page
    session = approved(service)
    session = service.validate_full_data(session.session_id, session.revision, "full.csv", FULL)
    session = service.confirm_full_data(session.session_id, session.revision)
    iteration = {
        "iteration_id": IDENTITY,
        "session_id": session.session_id,
        "goal": session.goal,
        "status": "confirmed",
        "hypothesis": "补充确认样例后准确率提升。",
        "expected_outcome": "开发集对照有可解释变化。",
        "changes": "保持资料与配置，验证自动交接。",
        "parent_run_id": "wb-" + "1" * 32,
        "evaluation_suite": {
            "suite_id": "f" * 64,
            "case_counts": {"validation": 4, "test": 4},
        },
        "options": {},
        "data_change": False,
    }
    executions: dict[str, dict] = {}
    calls: list[tuple] = []

    class Iterations:
        def __init__(self, *args, **kwargs):
            pass

        def list_iterations(self, session_id=None):
            return [deepcopy(iteration)]

        def get(self, iteration_id):
            return deepcopy(iteration)

    class Execution:
        def __init__(self, *args, **kwargs):
            pass

        def get(self, iteration_id):
            record = executions.get(iteration_id)
            return deepcopy(record) if record else None

        def start(self, iteration_id, current, **options):
            calls.append(("start", iteration_id, options))
            executions[iteration_id] = {
                "iteration_id": iteration_id,
                "status": "training",
                "message": "正在训练。",
                "run_id": "wb-" + "2" * 32,
                "issues": [],
            }
            return deepcopy(executions[iteration_id])

        def stop(self, iteration_id):
            calls.append(("stop", iteration_id))
            executions[iteration_id]["status"] = "stopped"
            return deepcopy(executions[iteration_id])

    monkeypatch.setattr(iterations_module, "IterationService", Iterations)
    monkeypatch.setattr(execution_module, "IterationExecutionService", Execution)
    return service, session, page, calls, executions


def test_confirmed_iteration_executes_once_and_only_through_explicit_click(execution_page):
    _, session, page, calls, _ = execution_page
    page.run()
    page.selectbox(key="intake_select").select(session.session_id).run()
    assert not page.exception
    assert calls == []
    execute = button(page, "按确认方案执行到开发集对照")
    assert not execute.disabled
    execute.click().run()
    assert not page.exception
    assert calls == [("start", IDENTITY, {"independent_rows_confirmed": True})]
    # After submission the page shows progress and the start entry is gone.
    assert any("正在训练" in message.value for message in page.info)
    assert not any(b.label == "按确认方案执行到开发集对照" for b in page.button)
    # 与 CLI 同口径的人话摘要：后台推进 + 关闭页面不影响执行边界。
    assert any(
        "自动执行正在后台推进：训练已按确认方案启动。" in block.value for block in page.markdown
    )
    assert any(
        "后台进程独立于页面与终端运行，关闭页面不影响执行。" in block.value
        for block in page.markdown
    )


def test_running_execution_can_be_stopped_from_the_page(execution_page):
    _, session, page, calls, executions = execution_page
    page.run()
    page.selectbox(key="intake_select").select(session.session_id).run()
    button(page, "按确认方案执行到开发集对照").click().run()
    stop = next(b for b in page.button if b.label == "停止本轮后台执行")
    stop.click().run()
    assert not page.exception
    assert [call[:2] for call in calls] == [("start", IDENTITY), ("stop", IDENTITY)]
    assert executions[IDENTITY]["status"] == "stopped"
    assert any("已停止" in block.value for block in page.markdown) or not any(
        b.label == "停止本轮后台执行" for b in page.button
    )


def test_warning_ack_resume_requires_the_explicit_checkbox(execution_page, monkeypatch):
    _, session, page, calls, executions = execution_page
    page.run()
    page.selectbox(key="intake_select").select(session.session_id).run()

    def start_with_pause(self, iteration_id, current, **options):
        calls.append(("start", iteration_id, options))
        executions[iteration_id] = {
            "iteration_id": iteration_id,
            "status": "awaiting_warning_ack",
            "message": "预检存在需核对的提示。",
            "issues": [{"severity": "warning", "message": "样例较短，可能学不到充分上下文。"}],
            "run_id": "wb-" + "3" * 32,
        }
        return deepcopy(executions[iteration_id])

    import src.workbench.iteration_execution as execution_module

    page_execution = execution_module.IterationExecutionService
    monkeypatch.setattr(page_execution, "start", start_with_pause)
    button(page, "按确认方案执行到开发集对照").click().run()
    assert any("需要您核对" in block.value for block in page.markdown)
    # 暂停态人话摘要与 CLI 同源：需原入口勾选确认才会恢复，不会跳过提示自动训练。
    assert any("自动执行已暂停：预检存在需核对的提示。" in block.value for block in page.markdown)
    assert any(
        "请查看提示内容后，在原入口勾选确认继续才会恢复" in block.value for block in page.markdown
    )
    resume = next(b for b in page.button if b.label == "核对后继续到开发集对照")
    assert resume.disabled
    next(c for c in page.checkbox if c.label.startswith("已核对上述预检提示")).check().run()
    resume = next(b for b in page.button if b.label == "核对后继续到开发集对照")
    assert not resume.disabled
    resume.click().run()
    assert not page.exception
    assert calls[-1] == (
        "start",
        IDENTITY,
        {"acknowledge_warnings": True},
    )
