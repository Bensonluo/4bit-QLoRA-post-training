"""Shared read-only views of the public, recorded training and output evidence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st


def _load(root: Path) -> dict[str, Any] | None:
    from ui.showcase import load_showcase

    try:
        return load_showcase(root)
    except (ValueError, OSError) as exc:
        st.warning(f"公开历史记录暂不可读：{exc}。已有本地任务仍可继续。")
        return None


def _scores(data: dict[str, Any]) -> None:
    columns = st.columns(len(data["report"]["models"]))
    for column, model in zip(columns, data["report"]["models"]):
        rows = model["rows"]
        total = len(rows)
        correct = sum(row["correct"] is True for row in rows)
        truncated = sum(row["status"] == "truncated" for row in rows)
        eos = sum(row.get("token_proof", {}).get("finish_reason") == "eos" for row in rows)
        with column, st.container(border=True):
            st.metric(model["label"], f"{correct} / {total}")
            st.progress(correct / total, text=f"严格匹配 · {correct / total:.0%}")
            st.caption(f"完整结束 {eos} 题 · 截断 {truncated} 题")


def _case_scope(data: dict[str, Any]) -> str:
    rows = data["report"]["models"][0]["rows"]
    groups = [row.get("source", {}).get("group") for row in rows]
    if all(isinstance(group, dict) and group for group in groups):
        count = len({json.dumps(group, sort_keys=True, ensure_ascii=False) for group in groups})
        return f"{len(rows)} 道开发题 / {count} 个分组"
    return f"{len(rows)} 道开发题 · 分组未记录"


def render_history_summary(root: Path, *, key_prefix: str = "history-summary") -> None:
    """Compact score cards for the workspace, independent of local model stores."""
    data = _load(root)
    if data is None:
        return
    with st.container(key=key_prefix):
        st.caption(
            f"{data['date']} · 公开历史验证 · {data['config']['model']['name'].replace('___', '.')}"
        )
        _scores(data)
        st.line_chart(
            pd.DataFrame(data["history"]).set_index("step")[["loss"]],
            x_label="更新 step",
            y_label="loss",
            height=130,
            color="#a6afff",
        )
        st.caption(
            f"{data['training']['global_step']} 次实际更新 · {_case_scope(data)} · 合成任务，尚未做业务验收。"
        )


def _outputs(data: dict[str, Any], key_prefix: str) -> None:
    report = data["report"]
    models = report["models"]
    references = {row["index"]: row for row in models[0]["rows"]}

    def label(index: int) -> str:
        row = references[index]
        group = row.get("source", {}).get("group", {})
        context = " · ".join(str(value) for value in group.values())
        return f"开发题 {index + 1} · {context} · 期望 {row['expected']}"

    selected = st.selectbox(
        "选择开发题，查看完整原始输出",
        sorted(references),
        format_func=label,
        key=f"{key_prefix}-question",
    )
    reference = references[selected]
    prompt = reference["prompt"]
    _, marker, after_header = prompt.partition("\n\n### Input:\n")
    business_input, end, remainder = after_header.rpartition("\n\n### Response:\n")
    if marker and end and not remainder:
        st.caption("业务输入")
        st.code(business_input, language=None, wrap_lines=True)
    with st.expander("完整模型输入（原始提示）"):
        st.code(reference["prompt"], language=None, wrap_lines=True)
    st.caption("期望答案")
    st.code(reference["expected"], language=None, wrap_lines=True)
    statuses = {
        "scored": "已评分",
        "truncated": "输出截断",
        "failed": "生成或评分失败",
        "needs_business_review": "待业务评分",
    }
    for start in range(0, len(models), 2):
        columns = st.columns(2)
        for column, model in zip(columns, models[start : start + 2]):
            row = next(item for item in model["rows"] if item["index"] == selected)
            reason = row.get("token_proof", {}).get("finish_reason")
            end = {"eos": "EOS 结束", "length": "未遇 EOS，达到长度上限"}.get(
                reason, "结束方式未记录"
            )
            with column, st.container(border=True):
                st.markdown(f"**{model['label']}**")
                st.caption(
                    f"{statuses.get(row['status'], row['status'])} · "
                    f"截断：{'是' if row['truncated'] else '否'} · {end} · "
                    f"{row['generated_tokens']} token"
                )
                st.code(
                    row["output"] if row["output"] is not None else "未生成输出",
                    language=None,
                    wrap_lines=True,
                )
                if row["correct"] is not None:
                    st.caption("与期望一致" if row["correct"] else "未达到当前严格匹配标准")
                if row.get("error"):
                    st.caption(row["error"])


def _training(data: dict[str, Any]) -> None:
    history = pd.DataFrame(data["history"])
    metrics = st.columns(3)
    metrics[0].metric("实际训练更新", str(data["training"]["global_step"]))
    metrics[1].metric("首步 loss", f"{history.iloc[0]['loss']:.4g}")
    metrics[2].metric("末步 loss", f"{history.iloc[-1]['loss']:.4g}")
    st.line_chart(
        history.set_index("step")[["loss"]],
        x_label="优化更新 step",
        y_label="训练 loss",
        height=280,
        color="#a6afff",
    )
    st.caption(f"{len(history)} 个实际日志点；loss 下降与开发题匹配分别展示。")
    with st.expander("本次训练配置"):
        st.json(data["config"])


def _size(value: int) -> str:
    return (
        f"{value / 1_000_000_000:.2f} GB"
        if value >= 1_000_000_000
        else f"{value / 1_000_000:.1f} MB"
    )


def _artifacts(data: dict[str, Any]) -> None:
    columns = st.columns(len(data["artifacts"]))
    for column, (name, artifact) in zip(columns, data["artifacts"].items()):
        with column, st.container(border=True):
            st.metric(
                "合并 Hugging Face" if name == "hf" else "GGUF Q4_K_M",
                _size(artifact["directory_bytes"]),
            )
            for filename, details in artifact["weight_files"].items():
                st.text(filename)
                st.caption(f"权重文件 · {_size(details['size'])}")
                with st.expander(f"文件内容 SHA256 · {filename}"):
                    st.code(details["sha256"], language=None, wrap_lines=True)
    st.caption("这里展示历史文件信息；权重未包含在公开档案中。")
    costs = data["costs"]
    names = {
        "preparation_seconds": "数据与训练准备",
        "framework_train_runtime_seconds": "训练器运行",
        "training_monitor_seconds": "正式训练总耗时",
        "hf_export_stage_seconds": "HF 合并与导出",
        "gguf_export_stage_seconds": "GGUF 转换与核验",
        "four_model_compare_stage_seconds": "四模型同题对照",
    }
    st.dataframe(
        [
            {
                "实测阶段": label,
                "耗时": f"{costs[key]:.1f} 秒" if costs.get(key) is not None else "未记录",
            }
            for key, label in names.items()
        ],
        hide_index=True,
        width="stretch",
    )
    st.caption("各阶段总耗时与训练器运行时间口径不同，不相加为总成本。电量与电费未测量。")


def _downloads(data: dict[str, Any], key_prefix: str) -> None:
    source = data["source"]
    st.caption(f"公开记录日期：{data['date']} · {source['path']}")
    st.text(f"原始报告状态：{data['report']['status']}")
    with st.expander("来源文件 SHA256"):
        st.code(source["sha256"], language=None, wrap_lines=True)
    columns = st.columns(2)
    columns[0].download_button(
        "下载同题原始报告 JSON",
        json.dumps(data["report"], ensure_ascii=False, indent=2),
        file_name=f"{data['id']}-development-report.json",
        mime="application/json",
        key=f"{key_prefix}-report-download",
        width="stretch",
    )
    columns[1].download_button(
        "下载训练曲线 CSV",
        pd.DataFrame(data["history"]).to_csv(index=False),
        file_name=f"{data['id']}-loss-history.csv",
        mime="text/csv",
        key=f"{key_prefix}-loss-download",
        width="stretch",
    )
    st.download_button(
        "下载训练与交付摘要 JSON",
        json.dumps(
            {
                key: data[key]
                for key in ("config", "training", "costs", "artifacts", "source", "limitations")
            },
            ensure_ascii=False,
            indent=2,
        ),
        file_name=f"{data['id']}-training-handoff.json",
        mime="application/json",
        key=f"{key_prefix}-summary-download",
    )
    with st.expander("记录范围与未验证事项"):
        for limitation in data["limitations"]:
            st.write(limitation)


def render_history(root: Path, *, key_prefix: str = "history") -> None:
    """Render real historical outputs and training evidence without executing work."""
    data = _load(root)
    if data is None:
        return
    st.subheader(data["title"])
    st.caption(f"{data['date']} · 开发者合成验证 · {_case_scope(data)}，尚未做业务验收。")
    _scores(data)
    first = data["report"]["models"][0]
    truncated = sum(row["status"] == "truncated" for row in first["rows"])
    st.caption(
        f"对照已完成，{first['label']} {truncated} 题截断按零分保留；"
        "所有模型使用同题与同一生成协议。原始状态与记录可在「来源与下载」查看。"
    )
    tabs = st.tabs(["同题对照", "训练过程", "交付与成本", "来源与下载"])
    with tabs[0]:
        _outputs(data, key_prefix)
    with tabs[1]:
        _training(data)
    with tabs[2]:
        _artifacts(data)
    with tabs[3]:
        _downloads(data, key_prefix)
