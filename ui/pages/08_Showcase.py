"""Public, read-only experiment archive available without local models or MLflow."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st

from ui.components.showcase import render_history
from ui.components.workspace import SHOWCASE_ROOT
from ui.showcase import history_catalog

st.set_page_config(
    page_title="示例与历史 — TuneSmith", page_icon=":material/auto_stories:", layout="wide"
)
st.caption("EXPERIMENT ARCHIVE / 公开验证档案")
st.title("示例与历史")
st.write("数据怎样进入模型，训练时发生了什么，导出的模型答得是否一样。这里保留完整过程。")

try:
    catalog = history_catalog(SHOWCASE_ROOT)
except (OSError, ValueError) as exc:
    st.warning(f"历史目录暂不可读：{exc}")
else:
    kinds = {
        "workbench_handoff": "完整交接",
        "controlled_experiment": "受控实验",
        "product_smoke": "产品集成",
    }
    descriptions = {
        "workbench_handoff": "从数据准备到实际导出，查看四个模型的同题输出。",
        "controlled_experiment": "固定数据与更新次数，比较三种训练监督方式。",
        "product_smoke": "在正式训练链路中，核对基座与微调后的表现。",
    }
    for column, item in zip(st.columns(len(catalog), gap="medium"), catalog, strict=True):
        with column, st.container(border=True, key=f"ts-archive-card-{item['id']}"):
            st.caption(f"{item['date']} · {kinds.get(item['kind'], '历史记录')}")
            st.markdown(f"### {item['title']}")
            st.write(descriptions[item["kind"]])
            st.caption(f"{item['counts']['development_rows']} 道开发题 · 合成任务 · 真实执行")
            if st.button("查看记录", key=f"archive-select-{item['id']}", width="stretch"):
                st.session_state["archive-selection"] = item["id"]
    by_id = {item["id"]: item for item in catalog}
    if by_id:
        selection = st.selectbox(
            "历史记录",
            list(by_id),
            format_func=lambda key: by_id[key]["title"],
            key="archive-selection",
        )
        selected = by_id[selection]
        if selected["kind"] == "workbench_handoff":
            render_history(SHOWCASE_ROOT, key_prefix="archive")
        else:
            import pandas as pd

            labels = {
                "alpaca_full_prompt_v1": "全提示监督",
                "alpaca_answer_only_v1": "仅答案监督 · Alpaca",
                "native_chat_answer_only_v1": "仅答案监督 · 原生聊天（实验）",
                "base": "基础模型",
                "trained": "训练适配器",
            }
            st.subheader(selected["title"])
            st.caption(selected["scope"])
            results = pd.DataFrame(
                {
                    "模型 / 监督方式": labels.get(result["label"], result["label"]),
                    "答对 / 固定题数": f"{result['correct']} / {result['total']}",
                    "截断": result["truncated"],
                }
                for result in selected["results"]
            )
            st.dataframe(results, hide_index=True, width="stretch")
            st.caption(
                "这是同一开发题集上的机制验证。受控三臂属于同次实验，"
                "产品集成使用另一训练与推理路径，不将它们拼成连续提升的迭代曲线。"
            )
            source_name = selected["source"]["name"]
            source_path = SHOWCASE_ROOT / "docs/validation" / source_name
            st.download_button(
                "下载原始历史证据 JSON",
                source_path.read_bytes(),
                file_name=source_name,
                mime="application/json",
            )
            st.caption(f"来源：{source_name} · SHA-256 {selected['source']['sha256']}")
