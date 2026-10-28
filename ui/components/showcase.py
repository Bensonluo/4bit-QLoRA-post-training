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
    except (ValueError, OSError):
        st.warning("公开历史记录暂不可读或核验失败；请检查公开来源文件。已有本地任务仍可继续。")
        return None


def _scores(data: dict[str, Any]) -> None:
    columns = st.columns(len(data["report"]["models"]))
    for column, model in zip(columns, data["report"]["models"]):
        rows = model["rows"]
        total = len(rows)
        correct = sum(row["correct"] is True for row in rows)
        truncated = sum(row["status"] == "truncated" for row in rows)
        eos = sum((row.get("token_proof") or {}).get("finish_reason") == "eos" for row in rows)
        failed = sum(row["status"] == "failed" for row in rows)
        with column, st.container(border=True):
            st.metric(model["label"], f"{correct} / {total}")
            st.progress(correct / total, text=f"严格匹配 · {correct / total:.0%}")
            st.caption(
                f"完整结束 {eos} 题 · 截断 {truncated} 题"
                + (f" · 处理失败 {failed} 题" if failed else "")
            )


def _case_scope(data: dict[str, Any]) -> str:
    counts = data["counts"]
    groups = counts.get("development_groups")
    scope = f"{groups} 个分组" if groups is not None else "分组信息不可核验"
    return f"{counts['development_rows']} 道开发题 · {scope}"


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
    st.caption(
        f"{len(history)} 个实际训练日志点；同题对照记录的是第 {data['training']['global_step']} "
        "次更新结束后的输出，中间步骤没有逐步业务评分或生成快照。"
    )
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
        f"对照已结束，{first['label']} {truncated} 题截断按零分保留；"
        "所有模型使用同题与同一生成协议。原始状态与记录可在「来源与下载」查看。"
    )
    if data["report"]["status"] == "completed_with_failures":
        rows = [row for model in data["report"]["models"] for row in model["rows"]]
        truncated_count = sum(row["status"] == "truncated" for row in rows)
        failed_count = sum(row["status"] == "failed" for row in rows)
        st.warning(
            f"对照包含 {truncated_count} 条截断、{failed_count} 条处理失败"
            "（按模型和题目计数），均保留在评分分母中。"
        )
    tabs = st.tabs(["同题对照", "训练过程", "交付与成本", "来源与下载"])
    with tabs[0]:
        from ui.components.case_comparison import render_case_comparison

        render_case_comparison(data, key_prefix=key_prefix)
    with tabs[1]:
        _training(data)
    with tabs[2]:
        _artifacts(data)
    with tabs[3]:
        _downloads(data, key_prefix)
