"""The hosted display shares the local UI but cannot trigger execution or writes."""

from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("streamlit")
from ui import runtime


@pytest.mark.parametrize(
    "helper,widget,value",
    [
        (runtime.action_button, "button", True),
        (runtime.action_submit, "form_submit_button", True),
        (runtime.action_upload, "file_uploader", object()),
        (runtime.action_chat_input, "chat_input", "run this"),
    ],
)
@pytest.mark.parametrize("display", [False, True])
def test_action_boundary_works_for_column_widgets(monkeypatch, helper, widget, value, display):
    monkeypatch.setattr(runtime, "DISPLAY_ONLY", display)
    calls = []

    def callback():
        return None

    def native(*args, **kwargs):
        calls.append((args, kwargs))
        # Even a forged positive widget result must not pass the server boundary.
        return value

    container = SimpleNamespace(**{widget: native})
    result = helper("Action", container=container, disabled=False, on_click=callback)
    assert len(calls) == 1
    assert calls[0][0] == ("Action",)
    if display:
        assert result is (False if widget in {"button", "form_submit_button"} else None)
        assert calls[0][1]["disabled"] is True
        assert "on_click" not in calls[0][1]
    else:
        assert result is value
        assert calls[0][1]["disabled"] is False
        assert calls[0][1]["on_click"] is callback


def test_existing_disabled_state_is_preserved_locally(monkeypatch):
    monkeypatch.setattr(runtime, "DISPLAY_ONLY", False)
    seen = {}

    def button(label, **kwargs):
        seen.update(kwargs)
        return False

    assert not runtime.action_button(
        "Disabled", container=SimpleNamespace(button=button), disabled=True
    )
    assert seen["disabled"] is True


def test_default_streamlit_container_is_used(monkeypatch):
    monkeypatch.setattr(runtime, "DISPLAY_ONLY", True)
    seen = {}

    def button(label, **kwargs):
        seen.update(kwargs)
        return True

    monkeypatch.setattr(runtime.st, "button", button)
    assert not runtime.action_button("Action")
    assert seen["disabled"] is True


def test_data_wizard_blocks_path_reads_and_uploads_in_display_mode(monkeypatch, tmp_path):
    from streamlit.testing.v1 import AppTest

    from src.data.wizard import importers
    from ui import config

    monkeypatch.setattr(runtime, "DISPLAY_ONLY", True)
    monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path)
    reads = []

    def forbidden_read(*args, **kwargs):
        reads.append(args)
        raise AssertionError("A hosted visitor must not read server paths")

    monkeypatch.setattr(importers, "import_table", forbidden_read)
    page = Path(__file__).resolve().parents[2] / "ui/views/05_Data_Wizard.py"
    app = AppTest.from_file(str(page)).run(timeout=30)
    assert not app.exception
    path_input = next(item for item in app.text_input if item.label == "或输入服务器上的文件路径")
    assert path_input.disabled
    assert app.get("file_uploader")[0].proto.disabled
    navigation = next(item for item in app.button if item.label == "查看目标与数据")
    assert not navigation.disabled
    path_input.set_value(str(tmp_path / "not-public.csv"))
    read_button = next(item for item in app.button if item.label == "读取该路径")
    assert read_button.disabled
    # AppTest can force a disabled widget value; the Python guard must still refuse it.
    read_button.click().run(timeout=30)
    assert not app.exception
    assert reads == []
    assert not (tmp_path / "outputs").exists()


def test_registry_keeps_browsing_but_blocks_mutations_in_display_mode(monkeypatch, tmp_path):
    import pandas as pd
    from streamlit.testing.v1 import AppTest

    from src import tracking
    from ui import queries

    monkeypatch.setattr(runtime, "DISPLAY_ONLY", True)
    frame = pd.DataFrame(
        [
            {
                "name": "public-record",
                "version": 1,
                "current_stage": "None",
                "aliases": "champion",
                "status": "READY",
                "created": "2026-10-22",
                "run_id": "recorded-run",
            },
        ]
    )
    monkeypatch.setattr(queries, "fetch_model_versions", lambda *args, **kwargs: frame)
    attempts = []

    def forbidden_tracker(*args, **kwargs):
        attempts.append(args)
        raise AssertionError("Registry writes must not execute in display mode")

    monkeypatch.setattr(tracking, "get_tracker", forbidden_tracker)
    page = Path(__file__).resolve().parents[2] / "ui/views/04_Registry.py"
    app = AppTest.from_file(str(page)).run(timeout=30)
    assert not app.exception
    app.segmented_control(key="registry-source").set_value("本地注册版本").run(timeout=30)
    assert not app.exception
    assert any(item.label == "已注册模型" for item in app.selectbox)
    assert all(item.disabled for item in app.button if item.key != "registry-refresh")
    assert not app.button(key="registry-refresh").disabled
    next(item for item in app.button if item.label == "🏷 设置").click().run(timeout=30)
    assert not app.exception
    assert attempts == []


def test_display_settings_never_load_or_return_environment_credentials(monkeypatch, tmp_path):
    from contextlib import nullcontext

    from ui.components import agent_settings

    monkeypatch.setattr(runtime, "DISPLAY_ONLY", True)
    monkeypatch.setattr(agent_settings, "DISPLAY_ONLY", True)

    def forbidden_environment_read(*args, **kwargs):
        raise AssertionError("The hosted display must not read server credentials")

    monkeypatch.setattr(
        agent_settings,
        "os",
        SimpleNamespace(environ=SimpleNamespace(get=forbidden_environment_read)),
    )
    monkeypatch.setattr(
        agent_settings,
        "load_settings",
        lambda path: SimpleNamespace(
            provider="glm", base_url="https://example.com/v1", model="public-model"
        ),
    )
    state = {}

    class Column:
        def button(self, *args, **kwargs):
            assert kwargs["disabled"]
            return True

    class UI:
        session_state = state

        def expander(self, *args, **kwargs):
            return nullcontext()

        def selectbox(self, *args, **kwargs):
            return state[kwargs["key"]]

        def text_input(self, *args, **kwargs):
            return state.get(kwargs["key"], "")

        def checkbox(self, *args, **kwargs):
            return True

        def caption(self, *args, **kwargs):
            pass

        def columns(self, count):
            return [Column() for _ in range(count)]

    monkeypatch.setattr(agent_settings, "st", UI())
    _, _, key, authorized = agent_settings.render_agent_settings(tmp_path / "settings.json")
    assert key == ""
    assert authorized is False
    assert not (tmp_path / "settings.json").exists()
