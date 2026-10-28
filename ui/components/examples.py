"""Browsable fictional examples, with an explicit hand-off to a new local task."""

from __future__ import annotations

from html import escape
from pathlib import Path

import pandas as pd
import streamlit as st

from ui import config, runtime


def render_examples(root: Path) -> None:
    from ui.showcase import example_tasks_tolerant

    try:
        examples, unavailable = example_tasks_tolerant(root)
    except (OSError, ValueError):
        st.warning("示例数据暂不可读，请检查示例资料。")
        return
    if unavailable:
        names = "、".join(item["title"] for item in unavailable)
        message = f"示例数据暂不可读：{names}。"
        if examples:
            message += "其余示例仍可浏览。"
        st.warning(message)
    if not examples:
        st.caption("当前没有可用的示例资料，可直接创建自己的业务任务。")
        return

    st.subheader("从一份具体的数据开始")
    st.caption("先看不同任务的输入与输出，再带着自己的业务数据开始。")
    for column, example in zip(st.columns(len(examples), gap="medium"), examples, strict=True):
        with column, st.container(border=True, key=f"ts-example-{example['id']}"):
            st.html(
                f'<div class="ts-example-kind {escape(example["id"])}">'
                f'<span class="ts-example-dot"></span>{escape(example["title"])}</div>'
            )
            st.markdown(f"### {example['goal']}")
            row = example["rows"][0]
            input_text = " · ".join(str(row.get(name, "")) for name in example["input_columns"])
            answer = str(row.get(example["target_column"], ""))
            st.html(
                '<div class="ts-example-pair"><span>输入示例</span>'
                f'<p>{escape(input_text)}</p><div class="ts-example-arrow">↓</div>'
                f'<span>期望答案</span><p class="ts-example-answer">{escape(answer)}</p></div>'
            )
            st.caption(f"虚构示例 · {example['row_count']} 条样例 · CSV")
            if st.button(
                "预览数据",
                key=f"showcase-preview-{example['id']}",
                width="stretch",
                icon=":material/table_view:",
            ):
                st.session_state["showcase-example"] = example["id"]
                st.session_state["showcase-last-example"] = example["id"]
                st.session_state["showcase-preview-open"] = True
            if runtime.action_button(
                "用这份数据开始",
                key=f"showcase-create-{example['id']}",
                width="stretch",
                icon=":material/arrow_forward:",
                help="在本地创建新任务，从业务分析与预览核对开始。",
            ):
                from src.workbench.intake_service import IntakeService
                from ui.components.workspace import open_task

                try:
                    service = IntakeService(config.PROJECT_ROOT / "outputs/workbench/intake")
                    session = service.create(
                        example["goal"],
                        example["name"],
                        example["bytes"],
                        data_description=example["description"],
                    )
                except (OSError, ValueError) as exc:
                    st.error(f"创建示例任务失败：{exc}")
                else:
                    open_task(session.session_id)

    with st.expander(
        "原始示例数据与下载", expanded=st.session_state.get("showcase-preview-open", False)
    ):
        by_id = {example["id"]: example for example in examples}
        if st.session_state.get("showcase-example") not in by_id:
            last_example = st.session_state.get("showcase-last-example")
            chosen = last_example if last_example in by_id else next(iter(by_id))
            st.session_state["showcase-example"] = chosen
            st.session_state["showcase-last-example"] = chosen

        def remember_example() -> None:
            # Widget state is removed when leaving the page; retain the user's
            # selection separately so a navigation back opens the same CSV.
            st.session_state["showcase-last-example"] = st.session_state["showcase-example"]

        selected = st.selectbox(
            "示例数据",
            list(by_id),
            format_func=lambda item: by_id[item]["title"],
            key="showcase-example",
            on_change=remember_example,
        )
        example = by_id[selected]
        st.caption(example["description"])
        st.dataframe(pd.DataFrame(example["rows"]), hide_index=True, width="stretch")
        st.download_button(
            "下载原始 CSV",
            data=example["bytes"],
            file_name=example["name"],
            mime="text/csv",
            key="showcase-example-download",
            icon=":material/download:",
        )
        if runtime.DISPLAY_ONLY:
            st.caption("可在本地工作台使用这份数据开始；此处只读浏览。")
