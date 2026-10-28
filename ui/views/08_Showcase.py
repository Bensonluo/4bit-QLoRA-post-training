"""Public, read-only experiment archive available without local models or MLflow."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st

from ui.components.archive_cards import render_archive_cards
from ui.components.showcase import render_history
from ui.components.workspace import SHOWCASE_ROOT, render_workspace
from ui.showcase import available_history_catalog

st.set_page_config(
    page_title="示例与历史 — TuneSmith", page_icon=":material/auto_stories:", layout="wide"
)
st.caption("EXPERIMENT ARCHIVE / 公开验证档案")
st.title("示例与历史")
st.write("数据怎样进入模型，训练时发生了什么，导出的模型答得是否一样。这里保留完整过程。")
if st.button("浏览示例数据", key="archive-open-examples", icon=":material/table_view:"):
    st.switch_page(
        st.Page(render_workspace, title="任务总览", icon=":material/workspaces:", default=True)
    )

catalog, unavailable = available_history_catalog(SHOWCASE_ROOT)
for item in unavailable:
    st.warning(f"{item['title']}：{item['message']}（{item['sourcefile']}）")

if not catalog:
    st.info("当前没有可读取的历史档案；仍可继续查看目标与数据。")
    if st.button("打开目标与数据", key="archive-open-tasks", type="primary"):
        st.switch_page("views/07_Data_Intake.py")
    st.stop()

render_archive_cards(catalog)
by_id = {item["id"]: item for item in catalog}
if st.session_state.get("archive-selection") not in by_id:
    st.session_state["archive-selection"] = next(iter(by_id))
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
    from ui.components.study_archive import render_study

    render_study(SHOWCASE_ROOT, selection, key_prefix="archive-study")
