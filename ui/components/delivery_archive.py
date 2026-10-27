"""Portable file evidence, explicitly separate from locally registered models."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import streamlit as st

_ARTIFACTS = {
    "hf": ("Hugging Face", "合并 HF"),
    "gguf_q4_k_m": ("GGUF · Q4_K_M", "GGUF Q4_K_M"),
}


def render_delivery_archive(root: Path, *, key_prefix: str = "registry-delivery") -> None:
    """Display verified public metadata without opening MLflow or model weights."""
    from ui.showcase import load_showcase

    try:
        data = load_showcase(root)
    except (ValueError, OSError) as exc:
        st.warning(f"公开交付档案暂不可读：{exc}。可切换来源查看本地注册版本。")
        return
    report = data["report"]
    st.subheader("一份训练，两种真实交付格式")
    st.caption(
        f"{data['date']} · 历史文件档案；权重未包含，不是本地注册模型。"
        f"{data['counts']['development_rows']} 道开发题 / "
        f"{data['counts']['development_groups']} 个分组，属于合成验证，尚未做业务验收。"
    )
    models = {model["label"]: model for model in report["models"]}
    comparisons = []
    for column, (kind, (title, label)) in zip(st.columns(2), _ARTIFACTS.items()):
        artifact = data["artifacts"][kind]
        model = models.get(label)
        with column, st.container(border=True):
            st.markdown(f"### {title}")
            weight_bytes = sum(item["size"] for item in artifact["weight_files"].values())
            st.metric("模型权重", f"{weight_bytes / 1_000_000_000:.2f} GB")
            if model is not None:
                rows = model["rows"]
                correct = sum(row["correct"] is True for row in rows)
                eos = sum(row.get("token_proof", {}).get("finish_reason") == "eos" for row in rows)
                st.metric("开发集严格匹配", f"{correct} / {len(rows)}")
                st.caption(f"完整结束 {eos} 题 · 使用实际交付产物独立生成和计分")
                comparisons.append({"format": kind, "label": label, "metrics": model["metrics"]})
            else:
                st.caption("这份文件未找到对应的开发对照记录，分数未知。")
            for name, metadata in artifact["weight_files"].items():
                st.text(name)
                st.caption(f"权重文件精确大小：{metadata['size']:,} 字节")
                with st.expander(f"文件内容 SHA256 · {name}"):
                    st.code(metadata["sha256"], language=None, wrap_lines=True)
            st.caption("来源训练")
            st.code(artifact["source_run_id"], language=None, wrap_lines=True)

    dataset = report["dataset"]
    lineage: dict[str, Any] = {
        "training": {
            "run_id": data["training"]["run_id"],
            "status": data["training"]["status"],
            "optimizer_updates": data["counts"]["optimizer_steps"],
            "finished_at": data["training"]["finished_at"],
        },
        "dataset": {key: dataset[key] for key in ("name", "version", "split", "row_count")},
        "development_evaluation": {
            "evaluation_id": report["evaluation_id"],
            "comparison_key": report["comparison_key"],
            "status": report["status"],
            "results": comparisons,
        },
        "artifacts": data["artifacts"],
        "source": data["source"],
    }
    with st.expander("训练、数据版本与同题对照的关联"):
        st.caption("数据版本")
        st.code(dataset["version"], language=None, wrap_lines=True)
        st.caption("开发对照报告")
        st.code(report["evaluation_id"], language=None, wrap_lines=True)
        st.caption(
            f"实际训练 {data['counts']['optimizer_steps']} 次更新；"
            f"来源文件：{data['source']['path']}"
        )
        st.json(lineage)
    with st.container(horizontal=True, gap="medium"):
        if st.button(
            "查看完整同题对照",
            type="primary",
            icon=":material/compare_arrows:",
            key=f"{key_prefix}-open-comparison",
        ):
            st.session_state["archive-selection"] = data["id"]
            st.switch_page("pages/08_Showcase.py")
        st.download_button(
            "下载交付元数据 JSON",
            json.dumps(lineage, ensure_ascii=False, indent=2),
            file_name=f"{data['id']}-delivery-metadata.json",
            mime="application/json",
            key=f"{key_prefix}-download",
        )
    st.caption(
        "分数来自交付后的独立开发对照；原始报告保留基座截断记录，训练完成和导出成功不代表业务达标。"
    )
