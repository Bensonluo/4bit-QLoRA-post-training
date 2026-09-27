"""BYOK configuration and transport boundaries; no real provider requests."""

import io
import json
import socket
import ssl
from urllib.error import HTTPError, URLError

import pytest

from src.agent import intake
from src.agent.providers import PROVIDERS, AgentSettings, check_connection, load_settings
from src.workbench.intake_service import IntakeService
from tests.unit.test_data_intake import CSV, ScriptedModel, analysis


@pytest.mark.parametrize(
    "error, expected",
    [
        (TimeoutError("fixture-secret"), "建立连接或等待响应头阶段超时"),
        (URLError(TimeoutError("fixture-secret")), "建立连接或等待响应头阶段超时"),
        (URLError(socket.gaierror("fixture-secret")), "域名解析失败"),
        (URLError(ssl.SSLError("fixture-secret")), "TLS 连接失败"),
        (URLError(ConnectionRefusedError("fixture-secret")), "拒绝连接"),
        (ConnectionResetError("fixture-secret"), "连接中断"),
        (URLError("fixture-secret"), "尚无法确定具体原因"),
    ],
)
def test_network_error_classification_without_leaking_raw_reason(monkeypatch, error, expected):
    class FailingConnection:
        def open(self, request, timeout):
            raise error

    monkeypatch.setattr(intake, "build_opener", lambda *handlers: FailingConnection())
    client = intake.CompatibleChatClient("http://localhost:11434/v1", "fixture")
    with pytest.raises(RuntimeError, match=expected) as caught:
        client.complete([], [])
    assert "fixture-secret" not in str(caught.value)


def test_timeout_after_headers_reports_body_stage(monkeypatch):
    class SlowBody(io.BytesIO):
        def read(self, *args):
            raise TimeoutError("fixture-secret")

    class Connected:
        def open(self, request, timeout):
            return SlowBody()

    monkeypatch.setattr(intake, "build_opener", lambda *handlers: Connected())
    client = intake.CompatibleChatClient("http://localhost:11434/v1", "fixture")
    with pytest.raises(RuntimeError, match="读取响应正文阶段超时") as caught:
        client.complete([], [])
    assert "fixture-secret" not in str(caught.value)


@pytest.mark.parametrize(
    "value",
    [
        {"provider": 1},
        {"base_url": ["https://service.example/v1"]},
        {"model": None},
        {"api_key": "fixture-secret"},
        ["local", "http://localhost:11434/v1", "model"],
    ],
)
def test_invalid_config_is_rejected_without_rewriting_or_echoing_values(tmp_path, value):
    path = tmp_path / "config.json"
    source = json.dumps(value)
    path.write_text(source)
    with pytest.raises(ValueError, match="配置文件格式无效") as exc:
        load_settings(path, environ={})
    assert "fixture-secret" not in str(exc.value)
    assert path.read_text() == source


def test_unknown_secret_field_cannot_be_created_or_saved(tmp_path):
    path = tmp_path / "settings.json"
    with pytest.raises(TypeError):
        AgentSettings(api_key="fixture-secret").save(path)
    assert not path.exists()


def test_environment_provider_switch_uses_new_endpoint_and_model(tmp_path):
    path = tmp_path / "settings.json"
    AgentSettings("compatible", "https://old.example/v1", "old-model").save(path)
    result = load_settings(path, {"TUNESMITH_AGENT_PROVIDER": "glm-coding"})
    assert result.base_url == PROVIDERS["glm-coding"].base_url
    assert result.model == PROVIDERS["glm-coding"].model
    assert json.loads(path.read_text())["base_url"] == "https://old.example/v1"


class ProbeResponse:
    model = "fixture"

    def __init__(self, response):
        self.response = response

    def complete(self, messages, tools):
        return self.response


def tool_response(arguments):
    return {"tool_calls": [{"function": {"name": "connection_check", "arguments": arguments}}]}


def test_probe_requires_actual_boolean_tool_result():
    result = check_connection(ProbeResponse(tool_response('{"ok": true}')))
    assert "连接成功" in result
    assert "不代表业务分析质量" in result


@pytest.mark.parametrize(
    "response",
    [
        {"content": '{"ok": true}'},
        {"tool_calls": "malformed"},
        {"tool_calls": [None]},
        {"tool_calls": [{"function": {"name": "connection_check"}}]},
        tool_response("invalid JSON"),
        tool_response('{"ok": 1}'),
        tool_response('{"ok": true, "extra": "unexpected"}'),
        tool_response("true"),
    ],
)
def test_probe_rejects_text_malformed_calls_and_numeric_true(response):
    with pytest.raises(RuntimeError, match="未返回有效工具调用"):
        check_connection(ProbeResponse(response))


@pytest.mark.parametrize(
    "endpoint",
    [
        "http://remote.example/v1",
        "https://user:fixture-secret@remote.example/v1",
        "https://remote.example/v1?api_key=fixture-secret",
        "https://remote.example/v1#fixture-secret",
    ],
)
def test_remote_endpoint_rejects_insecure_or_embedded_credentials(endpoint):
    with pytest.raises(ValueError) as exc:
        intake.CompatibleChatClient(endpoint, "model", allow_remote=True)
    assert "fixture-secret" not in str(exc.value)


def test_authorization_is_only_in_header_not_json_body(monkeypatch):
    seen = []

    class Capture:
        def open(self, request, timeout):
            seen.append(request)
            return io.BytesIO(b'{"choices":[{"message":{"content":"hello"}}]}')

    monkeypatch.setattr(intake, "build_opener", lambda *handlers: Capture())
    client = intake.CompatibleChatClient(
        "https://service.example/v1", "test-model", "fixture-secret", allow_remote=True
    )
    response = client.complete([{"role": "user", "content": "test"}], [])
    assert response == {"content": "hello"}
    request = seen[0]
    assert request.full_url == "https://service.example/v1/chat/completions"
    assert request.get_header("Authorization") == "Bearer fixture-secret"
    assert "fixture-secret" not in request.data.decode()
    assert json.loads(request.data)["model"] == "test-model"


def test_http_error_does_not_expose_provider_response_or_credentials(monkeypatch, capsys):
    class Reject:
        def open(self, request, timeout):
            raise HTTPError(
                request.full_url,
                401,
                "provider-message-fixture-secret",
                {},
                io.BytesIO(b'{"error":"provider-body-fixture-secret"}'),
            )

    monkeypatch.setattr(intake, "build_opener", lambda *handlers: Reject())
    client = intake.CompatibleChatClient(
        "https://service.example/v1", "model", "fixture-secret", allow_remote=True
    )
    with pytest.raises(RuntimeError, match="HTTP 401") as exc:
        client.complete([], [])
    assert "fixture-secret" not in str(exc.value)
    assert "fixture-secret" not in str(capsys.readouterr())


def test_redirect_handler_blocks_forwarding_authorization(monkeypatch):
    requests = []

    def opener(*handlers):
        assert len(handlers) == 1
        redirect = handlers[0]

        class RedirectingService:
            def open(self, request, timeout):
                requests.append(request)
                redirected = redirect.redirect_request(
                    request, None, 307, "Temporary Redirect", {}, "https://other.example/v1"
                )
                # This is the forwarding step a redirect handler must prevent.
                requests.append(redirected)
                pytest.fail("The credential-bearing redirect must not be followed")

        return RedirectingService()

    monkeypatch.setattr(intake, "build_opener", opener)
    client = intake.CompatibleChatClient(
        "https://service.example/v1", "model", "fixture-secret", allow_remote=True
    )
    with pytest.raises((RuntimeError, ValueError)):
        client.complete([], [])
    assert len(requests) == 1
    assert requests[0].get_header("Authorization") == "Bearer fixture-secret"


def test_thinking_context_survives_tool_turns_without_persisting(tmp_path):
    service = IntakeService(tmp_path / "intake")
    session = service.create("根据客户首次描述预测类别", "sample.csv", CSV)
    result = analysis()

    class ThinkingFixture(ScriptedModel):
        def complete(self, messages, tools):
            response = super().complete(messages, tools)
            response["reasoning_content"] = "private-reasoning-fixture"
            return response

    model = ThinkingFixture(
        [
            ("profile_data", {}),
            ("inspect_rows", {"row_ids": []}),
            ("preview_recipe", result.recipe.model_dump()),
            ("submit_analysis", result.model_dump()),
        ]
    )
    updated = service.analyze(session.session_id, model)
    assert len(model.seen) == 4
    for context in model.seen[1:]:
        assistant_turns = [message for message in context if message["role"] == "assistant"]
        assert assistant_turns
        assert all(
            message["reasoning_content"] == "private-reasoning-fixture"
            for message in assistant_turns
        )
    assert updated.analysis == result
    assert "private-reasoning-fixture" not in updated.model_dump_json()
    assert "reasoning_content" not in json.dumps(updated.tool_trace)
    assert "private-reasoning-fixture" not in service.load(session.session_id).model_dump_json()
