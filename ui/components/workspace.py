"""Task overview for the real workbench: goals, next actions and recorded evidence."""

from __future__ import annotations

import sqlite3
from datetime import date, datetime
from html import escape
from math import isfinite
from numbers import Real
from pathlib import Path

import pandas as pd
import streamlit as st

from src.workbench.intake_models import IntakeSession
from src.workbench.intake_service import IntakeService
from ui import config, runtime
from ui.components.task_state import task_state

SHOWCASE_ROOT = Path(__file__).resolve().parents[2]

WORKFLOW = (
    ("业务目标与数据", "明确输入、答案与业务要求"),
    ("监督与数据核验", "真实预览、独立分区、预检"),
    ("本地微调", "按硬件与数据制定训练方案"),
    ("对照与迭代", "同题比较，按坏例改进"),
    ("验收与交付", "独立验收，交接模型与证据"),
)


def open_task(session_id: str | None = None) -> None:
    st.session_state["intake_select"] = session_id or ""
    if session_id:
        st.session_state["intake_id"] = session_id
    else:
        st.session_state.pop("intake_id", None)
    st.switch_page("views/07_Data_Intake.py")


def load_tasks() -> tuple[list[IntakeSession], int]:
    root = config.PROJECT_ROOT / "outputs/workbench/intake"
    if not (root / "intake.sqlite").exists():
        return [], 0
    options = {"read_only": True} if runtime.DISPLAY_ONLY else {}
    return IntakeService(root, **options).list_sessions_tolerant()


def _short_run_ids(runs: pd.DataFrame) -> dict[str, str]:
    """Keep short prefixes distinct among all known runs, not only the visible three."""
    if "run_id" not in runs:
        return {}
    identities = {
        value.strip() for value in runs["run_id"] if isinstance(value, str) and value.strip()
    }
    visible = {
        value.strip()
        for value in runs["run_id"].head(3)
        if isinstance(value, str) and value.strip()
    }
    return {
        identity: next(
            (
                identity[:size]
                for size in range(8, len(identity) + 1)
                if all(
                    other == identity or not other.startswith(identity[:size])
                    for other in identities
                )
            ),
            identity,
        )
        for identity in visible
    }


def _start_time(value: object) -> str:
    if not isinstance(value, (datetime, date)) or pd.isna(value):
        return "未记录"
    timestamp = pd.Timestamp(value)
    zone = timestamp.strftime("%Z") if timestamp.tzinfo else "时区未记录"
    return f"{timestamp:%Y-%m-%d %H:%M:%S}（{zone}）"


def _training_loss(row: pd.Series) -> str:
    for key in ("metrics.train_loss", "metrics.loss"):
        value = row.get(key)
        if isinstance(value, Real) and not isinstance(value, bool) and isfinite(value):
            return f"{value:.4g}"
    return "—"


def render_workspace() -> None:
    failed = False
    try:
        sessions, corrupt = load_tasks()
    except (OSError, ValueError, sqlite3.Error):
        sessions, corrupt, failed = [], 0, True

    with st.container(key="ts-home"):
        st.caption("业务模型后训练")
        st.title("业务模型工作台")
        st.write("围绕一个业务目标，核验监督、训练模型、同题对照，最终交付模型与证据。")
        with st.container(horizontal=True, vertical_alignment="center", gap="medium"):
            if runtime.DISPLAY_ONLY:
                if st.button(
                    "浏览示例任务" if sessions else "查看目标与数据",
                    type="primary",
                    icon=":material/arrow_forward:",
                ):
                    open_task(sessions[0].session_id if sessions else None)
            elif st.button("创建业务任务", type="primary", icon=":material/add:"):
                open_task()
            st.page_link(
                "views/08_Showcase.py",
                label="查看历史成果",
                icon=":material/auto_stories:",
            )

        from ui.components.examples import render_examples
        from ui.components.showcase import render_history_summary

        render_examples(SHOWCASE_ROOT)

        st.subheader("从训练到交接，看一份完整记录")
        st.caption("2026.10.22 · Qwen3-1.7B · Apple Silicon · 已公开的本地合成验证")
        render_history_summary(SHOWCASE_ROOT, key_prefix="home-history")
        if st.button(
            "打开历史档案",
            key="showcase-history",
            icon=":material/arrow_forward:",
        ):
            st.switch_page("views/08_Showcase.py")

        st.html(
            '<ol class="ts-workflow" aria-label="业务任务完整流程">'
            + "".join(
                f'<li><span class="ts-step-number">{i + 1:02d}</span><h3>{escape(title)}</h3><p>{escape(detail)}</p></li>'
                for i, (title, detail) in enumerate(WORKFLOW)
            )
            + "</ol>"
        )

        st.subheader("继续业务任务")
        st.caption("从已有目标与数据接着做，方案、训练与后续判断都保留在任务里。")
        if failed:
            st.error("任务列表暂不可读，请检查本地任务存储；当前无法判断有哪些任务。")
        else:
            if corrupt:
                st.warning(f"有 {corrupt} 个任务快照暂不可读，其余任务仍可继续。")
            if not sessions:
                with st.container(border=True, key="ts-empty-tasks"):
                    st.markdown("**先说业务目标，再看数据**")
                    st.write(
                        "提供一个业务目标和 CSV、Excel 或 JSONL 原始资料，从真实样例预览开始。"
                    )
            else:
                query = (
                    st.text_input(
                        "筛选业务任务",
                        placeholder="按业务目标或文件名查找",
                        label_visibility="collapsed",
                    )
                    if len(sessions) > 4
                    else ""
                )
                filtered = [
                    s
                    for s in sessions
                    if query.casefold() in f"{s.goal} {s.source.name}".casefold()
                ]
                if not filtered:
                    st.info("没有匹配的任务，试试其他业务目标或文件名。")
                else:
                    number = (
                        st.selectbox("任务页码", range(1, (len(filtered) + 3) // 4 + 1))
                        if len(filtered) > 4
                        else 1
                    )
                    for session in filtered[(number - 1) * 4 : number * 4]:
                        state = task_state(session)
                        with st.container(border=True, key=f"ts-task-{session.session_id}"):
                            content, action = st.columns([5, 1.25], vertical_alignment="center")
                            with content:
                                st.html(
                                    f'<div class="ts-task-heading"><h3>{escape(session.goal)}</h3>'
                                    f'<span class="ts-badge {"attention" if state.attention else ""}">{escape(state.label)}</span></div>'
                                )
                                scope = "样例" if session.source.scope == "sample" else "全量资料"
                                st.caption(
                                    f"{session.source.name} · {len(session.source.rows)} 条{scope} · {session.session_id[:8]}"
                                )
                                st.write(state.hint)
                            with action:
                                if st.button(
                                    "浏览任务" if runtime.DISPLAY_ONLY else "继续任务",
                                    key=f"open-{session.session_id}",
                                    width="stretch",
                                ):
                                    open_task(session.session_id)

        st.subheader("最近实验与验证记录")
        st.caption("运行结果与业务达标分别判断；完整参数、曲线和证据在实验记录中查看。")
        try:
            from ui.queries import fetch_runs

            all_runs = fetch_runs(config.MLFLOW_TRACKING_URI)
            runs = all_runs.head(3)
        except Exception:
            st.warning("实验记录暂不可读，请到实验记录页检查数据连接。")
        else:
            if runs.empty:
                st.caption("暂无实验记录。完成数据核验并启动训练后，运行记录会出现在这里。")
            else:
                rows = []
                short_ids = _short_run_ids(all_runs)
                for _, row in runs.iterrows():
                    run_id = row.get("run_id")
                    name = row.get("tags.mlflow.runName")
                    rows.append(
                        {
                            "记录": name.strip()
                            if isinstance(name, str) and name.strip()
                            else "未命名运行",
                            "运行 ID": short_ids.get(run_id.strip(), "未记录")
                            if isinstance(run_id, str) and run_id.strip()
                            else "未记录",
                            "开始时间": _start_time(row.get("start_time")),
                            "状态": {
                                "FINISHED": "已完成",
                                "RUNNING": "运行中",
                                "FAILED": "失败",
                            }.get(row.get("status"), row.get("status", "未知")),
                            "来源": "公开历史验证"
                            if pd.notna(row.get("tags.evidence.source"))
                            else "本地实验",
                            "训练损失": _training_loss(row),
                        }
                    )
                st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
                st.page_link("views/01_Experiments.py", label="查看全部实验记录")
