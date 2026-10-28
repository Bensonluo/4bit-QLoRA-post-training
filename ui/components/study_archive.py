"""Read-only study views that preserve each variant's actual prompt and objective."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

_FORMS = {"all": "全部题型", "classification": "分类", "json": "JSON"}
_PRECISIONS = {"torch.float32": "FP32", "torch.bfloat16": "BF16", "torch.float16": "FP16"}


def _precision(value: str | None) -> str:
    return "未记录" if value is None else _PRECISIONS.get(value, value)


def _result(rows: list[dict[str, Any]]) -> str:
    return f"{sum(row['accepted'] is True for row in rows)} / {len(rows)}"


def _overview(data: dict[str, Any]) -> None:
    variants = data["variants"]
    if data["id"] == "r148-controlled-v2":
        variants = [variant for variant in variants if variant["role"] == "trained"]
    records = []
    for variant in variants:
        rows = variant["rows"]
        records.append(
            {
                "模型 / 监督方式": variant["label"],
                "答对 / 固定题数": _result(rows),
                "分类": _result([row for row in rows if row["form"] == "classification"]),
                "JSON": _result([row for row in rows if row["form"] == "json"]),
                "截断": sum(row["truncated"] for row in rows),
                "训练精度": _precision(variant.get("training_precision"))
                if variant["role"] == "trained"
                else "本次未训练",
                "推理精度": _precision(variant.get("evaluation_precision")),
            }
        )
    st.dataframe(records, hide_index=True, width="stretch")


def _pair(data: dict[str, Any], namespace: str) -> tuple[dict, dict]:
    variants = {variant["id"]: variant for variant in data["variants"]}
    ids = list(variants)
    default = data["default_pair"]
    columns = st.columns(2)
    with columns[0]:
        primary_id = st.selectbox(
            "主要模型 / 监督方式",
            ids,
            index=ids.index(default[0]),
            format_func=lambda key: variants[key]["label"],
            key=f"{namespace}-primary",
        )
        primary = variants[primary_id]
        baseline = primary.get("baseline_id")
        if baseline in variants and st.button(
            "与对应基座对照",
            key=f"{namespace}-use-baseline",
            help="选择本次实验为该提示格式记录的未训练基座。",
        ):
            st.session_state[f"{namespace}-comparison"] = baseline
    others = [identity for identity in ids if identity != primary_id]
    comparison_key = f"{namespace}-comparison"
    if st.session_state.get(comparison_key) not in others:
        st.session_state[comparison_key] = default[1] if default[1] in others else others[0]
    with columns[1]:
        comparison_id = st.selectbox(
            "并排对照",
            others,
            format_func=lambda key: variants[key]["label"],
            key=comparison_key,
        )
    return primary, variants[comparison_id]


def _output(variant: dict, row: dict) -> None:
    with st.container(border=True):
        st.markdown(f"**{variant['label']}**")
        st.caption(f"推理精度 {_precision(variant.get('evaluation_precision'))}")
        status = (
            "输出截断"
            if row["truncated"]
            else "与标准答案一致"
            if row["accepted"]
            else "未通过当前评分"
        )
        eos = row.get("eos_observed")
        ending = "EOS 结束" if eos is True else "未遇 EOS" if eos is False else "EOS 未记录"
        st.caption(f"{status} · {ending} · {row['generated_tokens']} token")
        st.code(row["output"], language=None, wrap_lines=True)
        with st.expander(f"完整提示 · {variant['label']}"):
            st.code(row["prompt"], language=None, wrap_lines=True)
        if row.get("format_valid") is not None:
            st.caption("输出格式符合要求" if row["format_valid"] else "输出格式未符合要求")


def _cases(primary: dict, comparison: dict, cases: list[dict], namespace: str) -> None:
    forms = list(dict.fromkeys(row["form"] for row in primary["rows"]))
    form = st.selectbox(
        "开发题类型",
        ["all", *forms],
        format_func=lambda value: _FORMS.get(value, value),
        key=f"{namespace}-form",
    )
    references = {
        (row["case_id"], row["form"]): row
        for row in primary["rows"]
        if form == "all" or row["form"] == form
    }
    keys = list(references)
    case_key = f"{namespace}-case"
    if st.session_state.get(case_key) not in keys:
        st.session_state[case_key] = keys[0]
    selected = st.selectbox(
        "选择开发题",
        keys,
        format_func=lambda key: f"{_FORMS.get(key[1], key[1])} · {key[0]}",
        key=case_key,
    )
    row = references[selected]
    logical = next(case for case in cases if (case["case_id"], case["form"]) == selected)
    group = logical.get("group_id")
    group_label = group if isinstance(group, str) and group.strip() else "未记录或不可核验"
    st.caption(f"业务输入 · 分组 {group_label}")
    st.code(logical["input"], language=None, wrap_lines=True)
    with st.expander("业务任务要求"):
        st.code(logical["instruction"], language=None, wrap_lines=True)
    st.caption("标准答案")
    st.code(row["expected"], language=None, wrap_lines=True)
    st.caption("每侧保留该模型实际使用的提示；原生聊天与 Alpaca 的完整提示可分别展开。")
    columns = st.columns(2)
    for column, variant in zip(columns, (primary, comparison)):
        actual = next(
            (item for item in variant["rows"] if (item["case_id"], item["form"]) == selected),
            None,
        )
        with column:
            if actual is None:
                st.warning(f"{variant['label']} 没有这道题的原始记录。")
            else:
                _output(variant, actual)


def _training(primary: dict) -> None:
    st.markdown(f"**{primary['label']} · 真实训练记录**")
    if primary["role"] == "baseline":
        st.info("这个基座在本次实验中未训练，没有训练曲线。")
        st.caption(f"本次生成评测使用 {_precision(primary.get('evaluation_precision'))}。")
        return
    history = primary["training_history"]
    st.caption(
        f"训练精度 {_precision(primary.get('training_precision'))} · "
        f"推理精度 {_precision(primary.get('evaluation_precision'))}"
    )
    if not history:
        st.info("这份变体没有可核验的训练日志，不能补画曲线。")
        return
    frame = pd.DataFrame(history)
    columns = st.columns(3)
    columns[0].metric("实际训练更新", str(history[-1]["step"]))
    columns[1].metric("首步 loss", f"{history[0]['loss']:.4g}")
    columns[2].metric("末步 loss", f"{history[-1]['loss']:.4g}")
    st.line_chart(
        frame.set_index("step")[["loss"]],
        x_label="训练更新 step",
        y_label="训练 loss",
        height=280,
        color="#a6afff",
    )
    st.caption(
        f"{len(history)} 个实际训练日志点，只展示所选变体。"
        "不同监督方式的损失目标不同，不能按 loss 高低为它们排名。"
    )
    st.caption("逐题原文来自这次训练结束后的开发评测，中间步骤没有逐步业务评分或生成快照。")
    with st.expander("查看训练日志数值"):
        st.dataframe(frame, hide_index=True, width="stretch")


def _source(data: dict[str, Any], namespace: str) -> None:
    source = data["source"]
    st.caption(f"{data['date']} · {source['path']}")
    with st.expander("来源与内容 SHA256"):
        st.code(source["sha256"], language=None, wrap_lines=True)
        if source.get("archive"):
            st.json(source["archive"])
        if source.get("members"):
            st.dataframe(source["members"], hide_index=True, width="stretch")
    st.download_button(
        "下载本次历史开发记录 JSON",
        json.dumps(data, ensure_ascii=False, indent=2),
        file_name=f"{data['id']}-development-study.json",
        mime="application/json",
        key=f"{namespace}-download",
    )
    with st.expander("记录范围与未验证事项"):
        for limitation in data["limitations"]:
            st.write(limitation)


def render_study(root: Path, study_id: str, *, key_prefix: str = "study") -> None:
    """Render one archived study, never pooling objectives or changing raw outputs."""
    from ui.study_archive import load_study

    try:
        data = load_study(root, study_id)
    except (OSError, ValueError):
        st.warning("这份历史记录暂不可读或核验失败；请检查公开来源与训练日志补件。")
        return
    namespace = f"{key_prefix}-{study_id}"
    st.subheader(data["title"])
    st.caption(f"{data['date']} · {data['scope']}")
    _overview(data)
    primary, comparison = _pair(data, namespace)
    tabs = st.tabs(["同题原文", "训练曲线", "来源与下载"])
    with tabs[0]:
        _cases(primary, comparison, data["cases"], namespace)
    with tabs[1]:
        _training(primary)
    with tabs[2]:
        _source(data, namespace)
