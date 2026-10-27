"""Select an immutable public archive without mixing it into local run records."""

from pathlib import Path

import streamlit as st

from ui import runtime


def render_archive_source(*, key: str, local_label: str) -> bool:
    public = "公开历史验证"
    selected = st.segmented_control(
        "记录来源",
        [local_label, public],
        default=public if runtime.DISPLAY_ONLY else local_label,
        selection_mode="single",
        required=True,
        key=f"{key}-source",
    )
    if selected != public:
        return False
    from ui.components.showcase import render_history

    if st.button("打开目标与数据", key=f"{key}-browse-task", icon=":material/assignment:"):
        st.switch_page("pages/07_Data_Intake.py")
    render_history(Path(__file__).resolve().parents[2], key_prefix=key)
    return True


def archive_link() -> None:
    if st.button(
        "先看一份完整的历史验证：数据、训练曲线与四模型对照",
        icon=":material/auto_stories:",
        key="empty-open-archive",
    ):
        st.switch_page("pages/08_Showcase.py")
