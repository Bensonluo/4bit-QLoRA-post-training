"""CLI BYOK behavior without network or real credentials."""

import json
import sys

import pytest

from scripts import data_intake as cli
from src.agent.providers import PROVIDERS, AgentSettings
from src.workbench.intake_service import IntakeService
from tests.unit.test_data_intake import CSV, analysis, model_for


@pytest.fixture()
def run_cli(tmp_path, monkeypatch):
    for name in ("PROVIDER", "BASE_URL", "MODEL", "API_KEY"):
        monkeypatch.delenv(f"TUNESMITH_AGENT_{name}", raising=False)
    config = tmp_path / "agent-settings.json"
    store = tmp_path / "intake"

    def run(*args):
        monkeypatch.setattr(
            sys,
            "argv",
            ["data_intake.py", "--agent-config-path", str(config), "--store", str(store), *args],
        )
        return cli.main()

    return run, config, store


def test_config_switch_uses_new_preset_without_saving_key(run_cli, monkeypatch, capsys):
    run, config, store = run_cli
    monkeypatch.setenv("TUNESMITH_AGENT_API_KEY", "unit-test-secret")
    assert (
        run(
            "agent-config",
            "--provider",
            "compatible",
            "--base-url",
            "https://custom.example/v1",
            "--model",
            "old-model",
        )
        == 0
    )
    assert run("agent-config", "--provider", "glm-coding") == 0
    value = json.loads(config.read_text())
    assert value == {
        "provider": "glm-coding",
        "base_url": PROVIDERS["glm-coding"].base_url,
        "model": PROVIDERS["glm-coding"].model,
    }
    assert "unit-test-secret" not in config.read_text()
    assert "unit-test-secret" not in str(capsys.readouterr())
    assert not store.exists()


def test_config_save_does_not_capture_environment_override(run_cli, monkeypatch):
    run, config, _ = run_cli
    monkeypatch.setenv("TUNESMITH_AGENT_PROVIDER", "glm-coding")
    monkeypatch.setenv("TUNESMITH_AGENT_MODEL", "environment-model")
    assert run("agent-config", "--model", "local-model") == 0
    assert json.loads(config.read_text())["provider"] == "local"
    assert json.loads(config.read_text())["model"] == "local-model"


def test_probe_uses_environment_and_only_synthetic_tool_request(run_cli, monkeypatch, capsys):
    run, config, store = run_cli
    AgentSettings(model="local-model").save(config)
    monkeypatch.setenv("TUNESMITH_AGENT_PROVIDER", "glm-coding")
    monkeypatch.setenv("TUNESMITH_AGENT_MODEL", "account-model")
    monkeypatch.setenv("TUNESMITH_AGENT_API_KEY", "unit-test-secret")
    seen = []

    class Probe:
        def __init__(self, *args, **kwargs):
            seen.append((args, kwargs))

        def complete(self, messages, tools):
            assert len(messages) == 1
            assert tools[0]["function"]["name"] == "connection_check"
            return {
                "tool_calls": [
                    {"function": {"name": "connection_check", "arguments": '{"ok":true}'}}
                ]
            }

    monkeypatch.setattr(cli, "CompatibleChatClient", Probe)
    assert run("agent-check") == 0
    assert seen == [
        (
            (PROVIDERS["glm-coding"].base_url, "account-model", "unit-test-secret"),
            {"allow_remote": True},
        )
    ]
    assert not store.exists()
    assert "unit-test-secret" not in str(capsys.readouterr())


def test_temporary_endpoint_override_does_not_forward_existing_key(run_cli, monkeypatch, capsys):
    run, config, _ = run_cli
    AgentSettings("glm-coding", PROVIDERS["glm-coding"].base_url, "glm-test").save(config)
    monkeypatch.setenv("TUNESMITH_AGENT_API_KEY", "unit-test-secret")
    monkeypatch.setattr(
        cli,
        "CompatibleChatClient",
        lambda *a, **kw: pytest.fail("No request should be constructed"),
    )
    assert run("agent-check", "--base-url", "https://other.example/v1") == 2
    result = capsys.readouterr()
    assert "配套设置" in result.err
    assert "unit-test-secret" not in result.err


def test_analyze_uses_shared_settings_and_retains_model_override(run_cli, monkeypatch):
    run, config, store = run_cli
    service = IntakeService(store)
    session = service.create("根据客户首次描述预测类别", "tickets.csv", CSV)
    AgentSettings(model="saved-model").save(config)
    seen = []

    def create_client(*args, **kwargs):
        seen.append((args, kwargs))
        return model_for(analysis())

    monkeypatch.setattr(cli, "CompatibleChatClient", create_client)
    assert run("analyze", session.session_id, "--model", "override-model") == 0
    assert seen[0] == (("http://localhost:11434/v1", "override-model", ""), {"allow_remote": False})
    assert service.load(session.session_id).preview is not None


def test_remote_analyze_still_requires_business_data_consent(run_cli, capsys):
    run, config, store = run_cli
    service = IntakeService(store)
    session = service.create("根据客户首次描述预测类别", "tickets.csv", CSV)
    AgentSettings("glm-coding", PROVIDERS["glm-coding"].base_url, "glm-test").save(config)
    assert run("analyze", session.session_id) == 2
    assert "需先允许" in capsys.readouterr().err
    assert service.load(session.session_id).analysis is None


def test_invalid_probe_reports_failure_without_creating_task_store(run_cli, monkeypatch, capsys):
    run, config, store = run_cli
    AgentSettings(model="test").save(config)

    class PlainText:
        def complete(self, messages, tools):
            return {"content": "connected"}

    monkeypatch.setattr(cli, "CompatibleChatClient", lambda *a, **kw: PlainText())
    assert run("agent-check") == 2
    assert "未返回有效工具调用" in capsys.readouterr().err
    assert not store.exists()
