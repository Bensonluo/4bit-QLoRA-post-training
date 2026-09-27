"""Session-only credentials with reusable, non-sensitive provider settings."""

from __future__ import annotations

import os
from pathlib import Path

import streamlit as st

from src.agent import intake
from src.agent.providers import PROVIDERS, AgentSettings, check_connection, load_settings


def _endpoint_identity(provider: str, base_url: str) -> tuple[str, str]:
    return provider, base_url.strip().rstrip("/")


def _clear_endpoint_authorization() -> None:
    st.session_state["agent_api_key"] = ""
    st.session_state["agent_remote_consent"] = False


def _change_provider() -> None:
    preset = PROVIDERS[st.session_state["agent_provider"]]
    st.session_state["agent_base_url"] = preset.base_url
    st.session_state["agent_model"] = preset.model
    _clear_endpoint_authorization()


def render_agent_settings(path: Path) -> tuple[str, str, str, bool]:
    """Return current client inputs; save only provider, URL and model on request."""
    if st.session_state.get("agent_settings_path") != str(path):
        environment_key = os.environ.get("TUNESMITH_AGENT_API_KEY", "")
        try:
            settings = load_settings(path)
        except (ValueError, OSError) as exc:
            st.warning(str(exc))
            settings = AgentSettings()
            environment_key = ""
        st.session_state.update(
            agent_settings_path=str(path),
            agent_provider=settings.provider,
            agent_base_url=settings.base_url,
            agent_model=settings.model,
            agent_api_key="",
            agent_remote_consent=False,
            agent_env_identity=_endpoint_identity(settings.provider, settings.base_url),
            agent_env_key=environment_key,
        )
    with st.expander("分析模型设置", expanded=not st.session_state["agent_model"]):
        provider = st.selectbox(
            "Agent 供应商",
            list(PROVIDERS),
            format_func=lambda value: PROVIDERS[value].label,
            key="agent_provider",
            on_change=_change_provider,
        )
        base_url = st.text_input(
            "模型服务 API 地址",
            key="agent_base_url",
            on_change=_clear_endpoint_authorization,
        )
        model = st.text_input("支持工具调用的模型名称", key="agent_model")
        entered_key = st.text_input(
            "API Key（仅当前会话，不写入任务或配置文件）",
            type="password",
            key="agent_api_key",
        )
        identity = _endpoint_identity(provider, base_url)
        environment_key = (
            st.session_state["agent_env_key"]
            if identity == st.session_state["agent_env_identity"]
            else ""
        )
        api_key = entered_key.strip() or environment_key
        if environment_key and not entered_key.strip():
            st.caption("使用本地环境变量中的密钥，仅适用于最初加载的供应商与服务地址。")
        st.caption(
            "保存会记住供应商、地址和模型；密钥仅在当前会话使用，重开后需重新输入或使用环境变量。"
        )
        save, probe = st.columns(2)
        if save.button("保存模型配置"):
            try:
                AgentSettings(provider, base_url.strip().rstrip("/"), model.strip()).save(path)
                st.success("模型配置已保存（不含密钥）。")
            except (ValueError, OSError) as exc:
                st.error(str(exc))
        if probe.button("测试模型连接"):
            try:
                # An explicit probe sends only check_connection's fixed synthetic request.
                client = intake.CompatibleChatClient(base_url, model, api_key, allow_remote=True)
                with st.spinner("检查服务连接及工具调用能力…"):
                    st.success(check_connection(client))
            except (ValueError, RuntimeError, OSError) as exc:
                st.error(str(exc))
        st.caption("连接测试仅发送固定测试内容，不发送业务数据，也不代替下方的数据发送授权。")
        allow_remote = False
        if intake.is_local_endpoint(base_url):
            st.caption("分析请求发送至此本地模型服务。原始文件不会被修改。")
        else:
            allow_remote = st.checkbox(
                "允许向上方服务发送本次业务描述、回答、数据画像和选取的证据行，用于分析与方案预览。",
                key="agent_remote_consent",
            )
    return base_url, model, api_key, allow_remote
