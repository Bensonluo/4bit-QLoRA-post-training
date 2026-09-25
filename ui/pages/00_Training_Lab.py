"""Training Lab — Configure, Launch, and Monitor training runs."""

from __future__ import annotations

import logging
import re
import sys
from contextlib import suppress
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st
import yaml

from src.utils.platform_utils import get_platform
from ui.components.charts import make_metric_timeseries
from ui.config import CONFIGS_DIR, MLFLOW_TRACKING_URI, MODEL_OPTIONS, PROJECT_ROOT

logger = logging.getLogger("qlora")

st.set_page_config(page_title="Training Lab", page_icon="🏋️", layout="wide")


def _validate_run_name(name: str) -> str | None:
    """Return error message if run_name is unsafe for filesystem use."""
    if not name:
        return "Run name cannot be empty."
    if len(name) > 128:
        return "Run name too long (max 128 characters)."
    if name.startswith((".", "-")):
        return "Run name cannot start with '.' or '-'."
    if not re.fullmatch(r"[a-zA-Z0-9_\-]+", name):
        return "Run name may only contain letters, digits, underscores, and hyphens."
    return None


platform = get_platform()
if platform.is_cuda:
    default_platform = "NVIDIA (CUDA)"
elif platform.is_mps:
    default_platform = "Apple Silicon (MPS)"
else:
    default_platform = "CPU"

# ── Header + Presets ────────────────────────────────────────────

col_title, col_presets = st.columns([2, 3])
with col_title:
    st.title("🏋️ Training Lab")
with col_presets:
    st.markdown("<div style='padding-top:0.8rem'></div>", unsafe_allow_html=True)
    pcols = st.columns(3)
    with pcols[0]:
        if st.button("⚡ Quick Test", width="stretch", help="Small model, 100 samples, 1 epoch"):
            st.session_state["preset"] = "quick"
            st.rerun()
    with pcols[1]:
        if st.button("🔥 Standard", width="stretch", help="Full model, 1K samples, 3 epochs"):
            st.session_state["preset"] = "standard"
            st.rerun()
    with pcols[2]:
        if st.button("🚀 Full Run", width="stretch", help="Full model, 10K samples, 5 epochs"):
            st.session_state["preset"] = "full"
            st.rerun()

# Preset defaults
preset = st.session_state.get("preset", None)
if preset == "quick":
    p_model, p_samples, p_epochs, p_r, p_lr, p_grad_accum = (
        "Qwen/Qwen2.5-0.5B-Instruct",
        100,
        1,
        8,
        "2e-4",
        4,
    )
elif preset == "standard":
    p_model, p_samples, p_epochs, p_r, p_lr, p_grad_accum = (
        "Qwen/Qwen2.5-1.5B-Instruct",
        1000,
        3,
        16,
        "2e-4",
        8,
    )
elif preset == "full":
    p_model, p_samples, p_epochs, p_r, p_lr, p_grad_accum = (
        "Qwen/Qwen2.5-1.5B-Instruct",
        10000,
        5,
        32,
        "1e-4",
        8,
    )
else:
    p_model, p_samples, p_epochs, p_r, p_lr, p_grad_accum = (
        "Qwen/Qwen2.5-1.5B-Instruct",
        1000,
        3,
        16,
        "2e-4",
        8,
    )

# ── Tabs ────────────────────────────────────────────────────────

tab_configure, tab_activity = st.tabs(["⚙️ Configure", "📋 Activity"])

# ── Configure Tab ───────────────────────────────────────────────

with tab_configure:
    col_form, col_preview = st.columns([2, 1])

    with col_form:
        platform_choice = (
            st.segmented_control(
                "Platform",
                ["Apple Silicon (MPS)", "NVIDIA (CUDA)", "CPU"],
                default=default_platform,
                help="Select hardware. 4-bit quantization only on NVIDIA CUDA.",
            )
            or default_platform
        )
        is_cuda = "CUDA" in platform_choice

        # Radio lives OUTSIDE the form so switching technique re-renders the
        # form (technique-specific sections) immediately, without a submit.
        st.subheader("Technique")
        technique_label = st.radio(
            "Post-training technique",
            ["SFT", "DPO", "GRPO"],
            index=0,
            horizontal=True,
            help="DPO expects a preference dataset (prompt/chosen/rejected); "
            "GRPO expects prompt+answer data.",
        )
        technique = technique_label.lower()

        with st.form("training_config"):
            st.subheader("Model & Data")
            c1, c2 = st.columns([3, 1])
            with c1:
                model_options_list = list(MODEL_OPTIONS.keys())
                model_name = st.selectbox(
                    "Base Model",
                    model_options_list,
                    index=model_options_list.index(p_model) if p_model in model_options_list else 0,
                    format_func=lambda x: f"{x} ({MODEL_OPTIONS[x]})",
                )
            with c2:
                st.markdown("<div style='padding-top:1.8rem'></div>", unsafe_allow_html=True)
                if not is_cuda:
                    st.badge("Full Precision", color="blue")
                else:
                    st.badge("4-bit QLoRA", color="green")

            dataset = st.text_input("Dataset (HF name or local path)", "yahma/alpaca-cleaned")
            ds1, ds2, ds3 = st.columns(3)
            with ds1:
                max_samples = st.number_input("Max Samples", 10, 100000, p_samples, 100)
            with ds2:
                validation_split = st.slider("Validation Split", 0.05, 0.3, 0.1, 0.05)
            with ds3:
                max_length = st.number_input(
                    "Max Length", 128, 8192, 512, 64, help="Sequence length budget (tokens)"
                )

            st.subheader("Training")
            t1, t2, t3, t4 = st.columns(4)
            with t1:
                epochs = st.number_input("Epochs", 1, 50, p_epochs)
            with t2:
                learning_rate = st.text_input("LR", p_lr)
            with t3:
                batch_size = st.number_input("Batch", 1, 8, 1)
            with t4:
                grad_accum = st.number_input("Grad Accum", 1, 32, p_grad_accum)

            effective_bs = batch_size * grad_accum
            st.caption(f"Effective batch size: **{effective_bs}**")

            st.subheader("LoRA")
            l1, l2, l3 = st.columns(3)
            with l1:
                lora_r = st.slider("Rank (r)", 4, 64, p_r, 4)
            with l2:
                lora_alpha = st.number_input("Alpha", value=lora_r * 2)
            with l3:
                lora_dropout = st.slider("Dropout", 0.0, 0.3, 0.05, 0.01)

            run_name = st.text_input(
                "Run Name", value=f"{model_name.split('/')[-1].lower()}-{epochs}ep"
            )

            if technique == "dpo":
                st.subheader("DPO")
                d1, d2 = st.columns(2)
                with d1:
                    dpo_beta = st.slider(
                        "Beta (β)",
                        0.01,
                        0.5,
                        0.1,
                        0.01,
                        help="Preference strength — lower stays closer to the reference model",
                    )
                with d2:
                    ref_model = st.selectbox(
                        "Reference Model",
                        model_options_list,
                        index=0,
                        format_func=lambda x: f"{x} ({MODEL_OPTIONS[x]})",
                        help="Frozen model for preference anchoring — a smaller one saves VRAM",
                    )
            elif technique == "grpo":
                st.subheader("GRPO")
                g1, g2, g3 = st.columns(3)
                with g1:
                    grpo_beta = st.slider(
                        "Beta (β)",
                        0.0,
                        0.2,
                        0.04,
                        0.01,
                        help="KL penalty strength (0 disables the penalty)",
                    )
                with g2:
                    num_generations = st.number_input("Generations per prompt", 2, 16, 4)
                with g3:
                    reward_funcs = st.multiselect(
                        "Reward Functions",
                        ["format", "accuracy", "length", "cosine", "llm_judge"],
                        default=["format", "accuracy"],
                        help="Registered rewards (src/training/reward_engine.py)",
                    )

            if is_cuda:
                st.subheader("Quantization")
                quant_choice = st.radio(
                    "Mode",
                    ["Full Precision (LoRA)", "4-bit QLoRA"],
                    index=1,
                    horizontal=True,
                )
                quant_bits = 4 if quant_choice == "4-bit QLoRA" else None
            else:
                quant_bits = None
                st.info("Full Precision LoRA — 4-bit requires NVIDIA CUDA", icon="💡")

            submitted = st.form_submit_button("🚀 Start Training", type="primary", width="stretch")

    with col_preview:
        st.subheader("Config Preview")
        # Technique-specific sections consumed by each script's --config loader.
        # Built conditionally: the other techniques' widgets don't exist.
        if technique == "dpo":
            technique_sections: dict = {
                "dpo": {"beta": dpo_beta, "max_length": max_length},
                "reference": {"name": ref_model},
            }
        elif technique == "grpo":
            technique_sections = {
                "grpo": {"beta": grpo_beta, "num_generations": num_generations},
                "reward": {"reward_funcs": reward_funcs},
            }
        else:
            technique_sections = {}

        # LR arrives as free text — parse once, warn instead of crashing the page.
        try:
            lr_value = float(learning_rate)
            lr_error: str | None = None
        except ValueError:
            lr_value = 2e-4
            lr_error = f"Invalid LR {learning_rate!r} — use a number like 2e-4 or 0.0002"
        if lr_error:
            st.warning(lr_error)

        config_dict = {
            "model": {
                "name": model_name,
                "quantization_bits": quant_bits,
                "max_length": max_length,
            },
            "training": {
                "num_epochs": epochs,
                "batch_size": batch_size,
                "gradient_accumulation_steps": grad_accum,
                "learning_rate": lr_value,
                "output_dir": f"./outputs/{run_name}",
            },
            "lora": {"r": lora_r, "lora_alpha": lora_alpha, "lora_dropout": lora_dropout},
            "data": {
                "dataset_name": dataset,
                "max_samples": max_samples,
                "validation_split": validation_split,
            },
            "logging": {"use_mlflow": True, "use_tensorboard": False},
            **technique_sections,
        }
        st.code(yaml.dump(config_dict, default_flow_style=False), language="yaml")

        # VRAM estimate — table-driven from MODEL_OPTIONS (whose values are
        # 4-bit estimates); full-precision scales weights by ~1/0.35 (bf16 vs NF4).
        vram_match = re.search(r"([\d.]+)\s*GB", MODEL_OPTIONS.get(model_name, ""))
        vram_gb = float(vram_match.group(1)) if vram_match else 2.3
        if quant_bits != 4:
            vram_gb /= 0.35
        st.caption(f"Estimated VRAM: **~{vram_gb:.1f} GB**")

    if submitted:
        error = _validate_run_name(run_name)
        if not error and lr_error:
            error = lr_error
        if not error and technique == "grpo" and not reward_funcs:
            error = "GRPO needs at least one reward function."
        if error:
            st.error(error)
            st.stop()

        CONFIGS_DIR.mkdir(parents=True, exist_ok=True)
        config_path = CONFIGS_DIR / f"{run_name}.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False)

        from src.tracking.runner import TrainingRunner

        runner = TrainingRunner(project_root=str(PROJECT_ROOT))
        try:
            rid = runner.launch_training(
                technique=technique,
                config_dict=config_dict,
                run_name=run_name,
            )
            st.success(f"Training started: `{rid}`")
            st.info("Switch to the **Activity** tab to monitor.")
        except Exception as e:
            st.error(f"Failed: {e}")


# ── Activity Tab ────────────────────────────────────────────────

with tab_activity:
    from src.tracking.runner import TrainingRunner

    runner = TrainingRunner(project_root=str(PROJECT_ROOT))
    all_runs = runner.list_all_runs()

    # Header with refresh
    h1, h2 = st.columns([4, 1])
    with h1:
        st.subheader("Training Activity")
    with h2:
        if st.button("🔄 Refresh", width="stretch"):
            st.rerun()

    if not all_runs:
        st.info("No training runs yet. Start one from the Configure tab.")
    else:
        for run_id in reversed(all_runs[-10:]):
            info = runner.get_run_info(run_id)
            status = runner.get_status(run_id)

            with st.container(border=True):
                c1, c2, c3, c4 = st.columns([3, 1, 1, 1])
                with c1:
                    st.markdown(f"**{run_id}**")
                    if info:
                        st.caption(
                            f"Technique: {info.get('technique', '?')} | PID: {info.get('pid', '?')}"
                        )
                with c2:
                    status_color = {"running": "🟢", "finished": "✅", "failed": "🔴"}.get(
                        status, "⚪"
                    )
                    st.metric("Status", f"{status_color} {status.title()}")
                with c3:
                    if status == "running" and st.button("⏹ Stop", key=f"stop_{run_id}"):
                        runner.stop_training(run_id)
                        st.rerun()
                with c4, st.popover("🗑 Delete", key=f"del_{run_id}", use_container_width=True):
                    st.caption(
                        "Removes the local run record. Config, logs, and MLflow data are kept."
                    )
                    if st.button("Confirm delete", key=f"del_confirm_{run_id}", type="primary"):
                        with suppress(KeyError):
                            runner.delete_run(run_id)  # already gone → nothing left to do
                        st.rerun()

                # Live metric chart from MLflow (cached 30s — see ui/queries.py)
                try:
                    from ui.queries import (
                        fetch_metric_history_by_name,
                        fetch_metric_names_by_name,
                    )

                    metric_names = fetch_metric_names_by_name(MLFLOW_TRACKING_URI, run_id)
                    if metric_names:
                        default_idx = metric_names.index("loss") if "loss" in metric_names else 0
                        chosen = st.selectbox(
                            "metric",
                            metric_names,
                            index=default_idx,
                            key=f"metric_{run_id}",
                            label_visibility="collapsed",
                        )
                        history = fetch_metric_history_by_name(MLFLOW_TRACKING_URI, run_id, chosen)
                        if len(history) > 1:
                            fig = make_metric_timeseries({run_id: history}, chosen)
                            st.plotly_chart(fig, width="stretch", height=200)
                except Exception as exc:
                    # Chart fetch is best-effort — a stale run must not break
                    # the dashboard, but keep the failure visible at debug.
                    logger.debug("metric chart fetch failed for %s: %s", run_id, exc)

                logs = runner.read_recent_logs(run_id, tail=15)
                if logs:
                    with st.expander("Recent Logs"):
                        st.code(logs, language="log")
