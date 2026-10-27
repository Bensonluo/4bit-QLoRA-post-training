"""TuneSmith — the shared frame and task-oriented workbench entry."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

from ui.components.shell import apply_shell, render_environment
from ui.components.workspace import render_workspace

st.set_page_config(
    page_title="TuneSmith — 业务模型工作台",
    page_icon=":material/model_training:",
    layout="wide",
    initial_sidebar_state="expanded",
)
apply_shell()
page = st.navigation(
    {
        "业务任务": [
            st.Page(render_workspace, title="任务总览", icon=":material/workspaces:", default=True),
            st.Page("pages/07_Data_Intake.py", title="目标与数据", icon=":material/assignment:"),
            st.Page("pages/08_Showcase.py", title="示例与历史", icon=":material/auto_stories:"),
        ],
        "训练与证据": [
            st.Page(
                "pages/00_Training_Lab.py", title="训练实验室", icon=":material/model_training:"
            ),
            st.Page("pages/01_Experiments.py", title="实验记录", icon=":material/science:"),
            st.Page("pages/02_Evaluation.py", title="评测证据", icon=":material/fact_check:"),
            st.Page(
                "pages/03_Model_Comparison.py", title="模型对照", icon=":material/compare_arrows:"
            ),
        ],
        "模型管理": [
            st.Page("pages/04_Registry.py", title="模型与版本", icon=":material/deployed_code:"),
            st.Page("pages/06_Chat.py", title="模型试用", icon=":material/chat_bubble:"),
        ],
        "数据工具": [
            st.Page("pages/05_Data_Wizard.py", title="数据向导", icon=":material/table_chart:")
        ],
    },
    expanded=True,
)
render_environment()
page.run()
