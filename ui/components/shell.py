"""Shared navigation and visual frame for the single Streamlit frontend."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import streamlit as st

from ui import runtime

UI_ROOT = Path(__file__).resolve().parents[1]


def apply_shell() -> None:
    st.html("<style>" + (UI_ROOT / "styles.css").read_text() + "</style>")
    st.logo(str(UI_ROOT / "assets/tunesmith.svg"), size="large")


def render_environment() -> None:
    with st.sidebar:
        st.html('<div class="ts-sidebar-rule"></div>')
        st.caption("云端展示环境" if runtime.DISPLAY_ONLY else "本地工作台")
        if runtime.DISPLAY_ONLY:
            st.caption("浏览示例与历史验证记录；执行操作在本地工作台完成。")
        with st.expander("运行环境"):
            for package, label in (
                ("streamlit", "界面"),
                ("mlflow-skinny", "实验记录"),
                ("torch", "训练底座"),
            ):
                try:
                    st.caption(f"{label} · {version(package)}")
                except PackageNotFoundError:
                    st.caption(f"{label} · 未安装")


def display_navigation(message: str, *, experiments: bool = True) -> None:
    """Offer available read-only destinations without suggesting execution."""
    st.info(message)
    with st.container(horizontal=True, vertical_alignment="center", gap="medium"):
        if st.button("查看目标与数据", type="primary", key="display-browse-task"):
            st.switch_page("views/07_Data_Intake.py")
        if st.button("浏览示例数据", key="display-browse-examples"):
            from ui.components.workspace import open_examples

            open_examples()
        if experiments and st.button("浏览实验记录", key="display-browse-experiments"):
            st.switch_page("views/01_Experiments.py")
