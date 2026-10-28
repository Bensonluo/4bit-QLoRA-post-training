"""Browse recorded study results without turning archives into a leaderboard."""

from __future__ import annotations

from typing import Any

import streamlit as st

_KINDS = {
    "workbench_handoff": "完整交接",
    "controlled_experiment": "同次受控实验",
    "product_smoke": "独立产品集成",
}
_DESCRIPTIONS = {
    "workbench_handoff": "从数据准备到实际导出，查看四个模型的同题输出。",
    "controlled_experiment": "固定数据与更新次数，比较三种训练监督方式。",
    "product_smoke": "在正式训练链路中，核对基座与微调后的表现。",
}
_LABELS = {
    "alpaca_full_prompt_v1": "全提示监督",
    "alpaca_answer_only_v1": "仅答案监督 · Alpaca",
    "native_chat_answer_only_v1": "仅答案监督 · 原生对话（实验）",
    "base": "基础模型",
    "trained": "产品微调模型",
}


def _select_archive(identity: str) -> None:
    st.session_state["archive-selection"] = identity


def render_archive_cards(catalog: list[dict[str, Any]]) -> None:
    """Show each study's original order, denominator and retained truncations."""
    if not catalog:
        st.caption("当前没有可浏览的公开历史记录。")
        return
    st.caption("已记录的合成开发题结果；各份档案独立展示，不代表业务验收。")
    for start in range(0, len(catalog), 3):
        records = catalog[start : start + 3]
        columns = st.columns(len(records), gap="medium")
        for column, item in zip(columns, records, strict=True):
            with column, st.container(border=True, key=f"ts-archive-card-{item['id']}"):
                st.caption(f"{item['date']} · {_KINDS.get(item['kind'], '历史记录')}")
                st.markdown(f"### {item['title']}")
                st.write(_DESCRIPTIONS.get(item["kind"], "查看这份历史记录的原始数据与结果。"))
                groups = item["counts"].get("development_groups")
                group_scope = f"{groups} 个业务分组" if groups is not None else "分组信息不可核验"
                st.caption(
                    f"{item['counts']['development_rows']} 道合成开发题 · {group_scope} · 真实执行"
                )
                for result in item["results"]:
                    label = _LABELS.get(result["label"], result["label"])
                    st.progress(
                        result["correct"] / result["total"],
                        text=(
                            f"{label} · 答对 {result['correct']} / {result['total']} · "
                            f"截断 {result['truncated']} 题"
                            + (f" · 处理失败 {result['failed']} 题" if result.get("failed") else "")
                        ),
                    )
                selected = st.session_state.get("archive-selection") == item["id"]
                st.button(
                    "当前查看" if selected else "查看记录",
                    key=f"archive-select-{item['id']}",
                    type="primary" if selected else "secondary",
                    width="stretch",
                    on_click=_select_archive,
                    args=(item["id"],),
                )
                if selected:
                    st.markdown("[前往已选详情 ↓](#archive-details)")
