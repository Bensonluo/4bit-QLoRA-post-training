"""Browse recorded cases, with pair-specific differences and unchanged raw outputs."""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

FILTERS = ("全部题目", "输出或 token 不同", "至少一侧未通过", "存在截断")


def pair_facts(left: dict, right: dict) -> dict[str, Any]:
    """Compare raw text and observed token IDs, keeping absent proof unknown."""
    a = (left.get("token_proof") or {}).get("generated_token_ids")
    b = (right.get("token_proof") or {}).get("generated_token_ids")
    return {
        "same_output": left["output"] == right["output"],
        "same_tokens": a == b if a and b else None,
        "not_passed": left["correct"] is not True or right["correct"] is not True,
        "truncated": bool(left.get("truncated") or right.get("truncated")),
    }


def pair_indices(models: list[dict], left: int, right: int, criterion: str) -> list[int]:
    other = {row["index"]: row for row in models[right]["rows"]}
    selected = []
    for row in models[left]["rows"]:
        facts = pair_facts(row, other[row["index"]])
        matches = {
            "全部题目": True,
            "输出或 token 不同": not facts["same_output"] or facts["same_tokens"] is False,
            "至少一侧未通过": facts["not_passed"],
            "存在截断": facts["truncated"],
        }
        if matches[criterion]:
            selected.append(row["index"])
    return selected


def _status(row: dict) -> str:
    if row.get("truncated"):
        return "… 截断"
    if row.get("output") is None:
        return "未生成"
    if row.get("status") == "failed":
        return "处理失败"
    if row["correct"] is True:
        return "✓ 匹配"
    return "待业务审阅" if row["correct"] is None else "未匹配"


def case_overview(models: list[dict]) -> pd.DataFrame:
    records = []
    maps = [{row["index"]: row for row in model["rows"]} for model in models]
    for row in models[0]["rows"]:
        records.append(
            {
                "题目": f"第 {row['index'] + 1} 题",
                "业务分组": " · ".join(str(v) for v in row["source"].get("group", {}).values()),
                "期望答案": row["expected"],
                **{
                    model["label"]: _status(rows[row["index"]]) for model, rows in zip(models, maps)
                },
            }
        )
    return pd.DataFrame(records, index=[row["index"] for row in models[0]["rows"]])


def _raw_outputs(models: list[dict], selected: int) -> None:
    for start in range(0, len(models), 2):
        for column, model in zip(st.columns(2), models[start : start + 2]):
            row = next(item for item in model["rows"] if item["index"] == selected)
            complete = (row.get("token_proof") or {}).get("finish_reason") == "eos"
            with column, st.container(border=True):
                st.markdown(f"**{model['label']}**")
                st.caption(
                    f"{_status(row)} · {'完整结束' if complete else '未记录完整结束'} · "
                    f"{row['generated_tokens']} token"
                )
                st.code(
                    row["output"] if row["output"] is not None else "未生成输出",
                    language=None,
                    wrap_lines=True,
                )
                if row.get("error"):
                    st.caption(row["error"])


def render_case_comparison(data: dict, *, key_prefix: str) -> None:
    models = data["report"]["models"]
    references = {row["index"]: row for row in models[0]["rows"]}
    mode = st.segmented_control(
        "查看方式",
        ["全部模型", "两模型对照"],
        default="全部模型",
        required=True,
        key=f"{key_prefix}-view",
    )
    visible = list(references)
    displayed = models
    if mode == "两模型对照":
        columns = st.columns(2)
        left = columns[0].selectbox(
            "左侧模型",
            range(len(models)),
            format_func=lambda i: models[i]["label"],
            key=f"{key_prefix}-left",
        )
        options = [i for i in range(len(models)) if i != left]
        right_key = f"{key_prefix}-right"
        if st.session_state.get(right_key) not in options:
            st.session_state[right_key] = options[0]
        right = columns[1].selectbox(
            "右侧模型", options, format_func=lambda i: models[i]["label"], key=right_key
        )
        displayed = [models[left], models[right]]
        all_pairs = zip(models[left]["rows"], models[right]["rows"])
        facts = [pair_facts(a, b) for a, b in all_pairs]
        identical = sum(f["same_output"] and f["same_tokens"] is True for f in facts)
        st.caption(
            f"当前模型对：{identical}/{len(references)} 题的原始文本与生成 token 均一致。"
            "这里比较实际输出，业务匹配仍以期望答案为准。"
        )
        criterion = st.selectbox("筛选当前模型对", FILTERS, key=f"{key_prefix}-filter")
        visible = pair_indices(models, left, right, criterion)

    st.caption(f"题目总览 · 当前显示 {len(visible)}/{len(references)} 题 · 保留业务分组")
    if not visible:
        st.info("当前模型对没有符合筛选条件的题目，可以切回全部题目查看原始输出。")

        def reset_filter() -> None:
            st.session_state[f"{key_prefix}-filter"] = FILTERS[0]

        st.button("显示全部题目", on_click=reset_filter, key=f"{key_prefix}-reset-filter")
        return

    question_key = f"{key_prefix}-question"
    epoch_key = f"{key_prefix}-table-epoch"
    table_key = f"{key_prefix}-case-table-{st.session_state.get(epoch_key, 0)}"
    if st.session_state.get(question_key) not in visible:
        st.session_state[question_key] = visible[0]

    def select_row() -> None:
        rows = st.session_state[table_key]["selection"]["rows"]
        if rows:
            st.session_state[question_key] = visible[rows[0]]

    st.dataframe(
        case_overview(models).loc[visible],
        hide_index=True,
        width="stretch",
        on_select=select_row,
        selection_mode="single-row",
        key=table_key,
    )
    st.caption("点选表格中的一行，或用下方选择器定位原始输出。")

    def clear_row_highlight() -> None:
        st.session_state[epoch_key] = st.session_state.get(epoch_key, 0) + 1

    selected = st.selectbox(
        "选择开发题，查看完整原始输出",
        visible,
        format_func=lambda i: f"开发题 {i + 1} · 期望 {references[i]['expected']}",
        key=question_key,
        on_change=clear_row_highlight,
    )
    reference = references[selected]
    prompt = reference["prompt"]
    _, marker, after_header = prompt.partition("\n\n### Input:\n")
    business_input, end, remainder = after_header.rpartition("\n\n### Response:\n")
    if marker and end and not remainder:
        st.caption("业务输入")
        st.code(business_input, language=None, wrap_lines=True)
    with st.expander("完整模型输入（原始提示）"):
        st.code(prompt, language=None, wrap_lines=True)
    st.caption("期望答案")
    st.code(reference["expected"], language=None, wrap_lines=True)
    _raw_outputs(displayed, selected)
