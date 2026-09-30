"""Chat — 与训练产物对话（LlamaBoard Chat 式收尾）。"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st

from src.inference.discovery import ChatModelOption, discover_chat_models
from ui.config import PROJECT_ROOT

st.set_page_config(page_title="Chat", page_icon="💬", layout="wide")

st.title("💬 Chat")
st.caption(
    "选一个训练产物（adapter / 已合并模型）或任意底座，加载后直接对话——"
    "训练完的模型不该只躺在 outputs/ 里。"
)

CUSTOM = "⌨️ 自定义（HF 名 / 本地底座路径）"


@st.cache_resource(show_spinner="加载模型中（首次可能要下轈权重）…")
def _cached_load(base: str, adapter: str | None) -> tuple[Any, Any]:
    from src.inference.chat_engine import load_chat_model

    return load_chat_model(base, adapter)


# ── 侧栏：选模型 + 生成参数 ────────────────────────────────────

options = discover_chat_models(PROJECT_ROOT)
labels = [o.label for o in options] + [CUSTOM]

with st.sidebar:
    choice = st.selectbox(
        "模型",
        labels,
        index=0 if options else len(labels) - 1,
        help="列表来自本机 outputs/ 扫描：🔧 = LoRA adapter（自动读底座名），📦 = 已合并模型",
    )
    opt: ChatModelOption | None = (
        options[labels.index(choice)] if choice != CUSTOM and choice in labels[:-1] else None
    )

    base_model = ""
    adapter_path: str | None = None
    if opt is None:
        base_model = st.text_input(
            "底座模型（HF 名或本地路径）",
            value="Qwen/Qwen2.5-0.5B-Instruct",
        )
        adapter_path = st.text_input("LoRA adapter 路径（可选）", value="").strip() or None
    else:
        st.caption(f"路径：`{opt.path}`")
        if opt.kind == "adapter":
            override = st.text_input(
                "底座覆盖（可选）",
                value="",
                help="默认从 adapter_config.json 读取；底座在本地其他位置时填这里",
            )
            base_model = override.strip() or (opt.base_model or "")
            adapter_path = str(opt.path)
        else:  # merged：自带全部权重
            base_model = str(opt.path)

    system_prompt = st.text_area(
        "System prompt",
        value="",
        help="留空 = 不加。实体匹配/名称归一化任务（如供应商名归一化）把任务提示词贴这里",
    )
    p1, p2 = st.columns(2)
    with p1:
        max_new_tokens = st.slider("最大生成 token 数", 16, 1024, 256, 16)
    with p2:
        temperature = st.slider("采样温度", 0.0, 1.5, 0.7, 0.05)
    thinking = st.checkbox("思考模式（Qwen3）", value=False, help="开启后允许模型先推理再作答")

    if st.button("⚡ 加载模型", type="primary", use_container_width=True):
        if not base_model.strip():
            st.error(
                "底座模型为空：adapter 的 adapter_config.json 里没有底座名，请在「底座覆盖」里补一个。"
            )
        else:
            try:
                _cached_load(base_model.strip(), adapter_path)
                st.session_state["chat_base"] = base_model.strip()
                st.session_state["chat_adapter"] = adapter_path
                st.session_state["chat_messages"] = []
                st.rerun()
            except Exception as exc:
                st.error(f"加载失败：{exc or exc!r}")  # 空消息异常也要给出类型

# ── 主区：会话 ─────────────────────────────────────────────────

loaded_base = st.session_state.get("chat_base")
loaded_adapter = st.session_state.get("chat_adapter")

title_col, clear_col = st.columns([4, 1])
with clear_col:
    if st.button(
        "🧹 清空对话", width="stretch", disabled=not st.session_state.get("chat_messages")
    ):
        st.session_state["chat_messages"] = []
        st.rerun()

if not loaded_base:
    # options 由页首 discover_chat_models(PROJECT_ROOT) 算出：空态时选择器只有「自定义」，不能再说
    # 「在左侧选择模型」——指向没有模型的列表。改指 00 页训练，并如实
    # 披露自定义底座逃生门（专家路径不封死，侧栏控件本就先于 st.stop 渲染）
    if not options:
        st.info(
            "本机还没有可对话的训练产物（🔧 LoRA adapter 或 📦 已合并模型）——"
            "先训练一个模型，回来这里就能和它对话。急着体验的话，"
            "也可以在左侧「⌨️ 自定义」填底座名（如 Qwen/Qwen2.5-0.5B-Instruct）直接聊。"
        )
        if st.button("🏋️ 去训练一个模型", type="primary"):
            st.switch_page("pages/00_Training_Lab.py")
    else:
        st.info("在左侧选择模型并点 **⚡ 加载模型**，然后开始对话。")
    st.stop()

where = f"`{loaded_base}`" + (f" + adapter `{loaded_adapter}`" if loaded_adapter else "")
unload_col, _ = st.columns([1, 3])
with unload_col:
    if st.button("⬅️ 换模型 / 卸载"):
        st.session_state.pop("chat_base", None)
        st.session_state.pop("chat_adapter", None)
        st.session_state["chat_messages"] = []
        st.cache_resource.clear()
        st.rerun()
st.caption(f"已加载：{where}")

model, tokenizer = _cached_load(loaded_base, loaded_adapter)

if "chat_messages" not in st.session_state:
    st.session_state["chat_messages"] = []

for msg in st.session_state["chat_messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("elapsed"):
            st.caption(f"{msg['elapsed']:.1f}s")

prompt = st.chat_input("输入消息，Enter 发送…")
if prompt:
    st.session_state["chat_messages"].append({"role": "user", "content": prompt})

    history = [
        {"role": m["role"], "content": m["content"]} for m in st.session_state["chat_messages"]
    ]
    if system_prompt.strip():
        history = [{"role": "system", "content": system_prompt.strip()}] + history

    from src.inference.chat_engine import stream_reply

    # 流式渲染：st.write_stream 逐块打字机输出，返回值 = 完整回复文本
    # （Streamlit 官方 chat 接法；后台线程 + TextIteratorStreamer 见 chat_engine）
    t0 = time.perf_counter()
    with st.chat_message("user"):
        st.markdown(prompt)
    try:
        with st.chat_message("assistant"):
            reply = st.write_stream(
                stream_reply(
                    model,
                    tokenizer,
                    history,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    enable_thinking=thinking,
                )
            )
    except Exception as exc:
        st.error(f"生成失败：{exc or exc!r}")  # 空消息异常（如 AttributeError）也要给出类型
        st.stop()
    elapsed = time.perf_counter() - t0
    st.session_state["chat_messages"].append(
        {"role": "assistant", "content": reply, "elapsed": elapsed}
    )
    st.rerun()
