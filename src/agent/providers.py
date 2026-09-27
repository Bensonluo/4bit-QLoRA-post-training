"""Small BYOK configuration shared by UI and CLI; credentials stay out of storage."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path

from src.agent.intake import ChatClient, validate_endpoint


@dataclass(frozen=True)
class ProviderPreset:
    label: str
    base_url: str
    model: str = ""


# Models are editable because account access differs. Never fall back across endpoints.
PROVIDERS = {
    "local": ProviderPreset("本地模型", "http://localhost:11434/v1"),
    "glm-coding": ProviderPreset(
        "GLM Coding Plan（智谱中国）", "https://open.bigmodel.cn/api/coding/paas/v4", "glm-5.3"
    ),
    "glm": ProviderPreset("智谱 GLM 普通 API", "https://open.bigmodel.cn/api/paas/v4", "glm-5.3"),
    "compatible": ProviderPreset("自定义 OpenAI 兼容服务", ""),
}


@dataclass(frozen=True)
class AgentSettings:
    provider: str = "local"
    base_url: str = "http://localhost:11434/v1"
    model: str = ""

    def __post_init__(self) -> None:
        if not all(isinstance(value, str) for value in (self.provider, self.base_url, self.model)):
            raise ValueError("供应商、API 地址和模型名称必须是文本。")

    def validate(self) -> None:
        if self.provider not in PROVIDERS:
            raise ValueError("未知 Agent 供应商，请选择已有预设或自定义兼容服务。")
        validate_endpoint(self.base_url)

    def save(self, path: Path) -> None:
        self.validate()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(asdict(self), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )


def load_settings(path: Path, environ: Mapping[str, str] | None = None) -> AgentSettings:
    """Explicit environment selection overrides saved public settings, never credentials."""
    env = os.environ if environ is None else environ
    try:
        saved = (
            AgentSettings(**json.loads(path.read_text(encoding="utf-8")))
            if path.exists()
            else AgentSettings()
        )
    except (TypeError, ValueError):
        raise ValueError("Agent 配置文件格式无效，请重新保存供应商配置。") from None
    provider = env.get("TUNESMITH_AGENT_PROVIDER", saved.provider)
    if provider not in PROVIDERS:
        raise ValueError("未知 Agent 供应商，请选择已有预设或自定义兼容服务。")
    defaults = PROVIDERS[provider] if provider != saved.provider else saved
    result = AgentSettings(
        provider,
        env.get("TUNESMITH_AGENT_BASE_URL", defaults.base_url).strip().rstrip("/"),
        env.get("TUNESMITH_AGENT_MODEL", defaults.model).strip(),
    )
    # An empty custom endpoint is an editable draft, not a usable client.
    if result.base_url:
        result.validate()
    return result


def check_connection(client: ChatClient) -> str:
    """Probe actual tool calling with synthetic content, never user task data."""
    messages = [
        {
            "role": "user",
            "content": 'Test this application\'s tool interface: call connection_check with {"ok": true}. Do not answer with plain text.',
        }
    ]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "connection_check",
                "description": "Check the tool interface.",
                "parameters": {
                    "type": "object",
                    "properties": {"ok": {"type": "boolean"}},
                    "required": ["ok"],
                    "additionalProperties": False,
                },
            },
        }
    ]
    message = client.complete(messages, tools)
    try:
        for call in message.get("tool_calls") or []:
            function = call["function"]
            args = json.loads(function["arguments"])
            if (
                function["name"] == "connection_check"
                and isinstance(args, dict)
                and set(args) == {"ok"}
                and args["ok"] is True
            ):
                return "连接成功，模型已返回有效工具调用。此检查不代表业务分析质量已通过验收。"
    except (KeyError, TypeError, ValueError):
        pass  # Malformed responses share the same actionable failure below.
    raise RuntimeError("模型已响应，但未返回有效工具调用；请检查模型是否支持 function calling。")
