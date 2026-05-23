"""Training Lab — Configure, Launch, and Monitor training runs."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st
import yaml

from ui.config import MODEL_OPTIONS, CONFIGS_DIR, PROJECT_ROOT
from ui.components.charts import make_metric_timeseries
from src.utils.platform_utils import get_platform

st.set_page_config(page_title="Training Lab", page_icon="🏋️", layout="wide")

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
        if st.button("⚡ Quick Test", width='stretch', help="Small model, 100 samples, 1 epoch"):
            st.session_state["preset"] = "quick"
            st.rerun()
    with pcols[1]:
        if st.button("🔥 Standard", width='stretch', help="Full model, 1K samples, 3 epochs"):
            st.session_state["preset"] = "standard"
            st.rerun()
    with pcols[2]:
        if st.button("🚀 Full Run", width='stretch', help="Full model, 10K samples, 5 epochs"):
            st.session_state["preset"] = "full"
            st.rerun()

# Preset defaults
preset = st.session_state.get("preset", None)
if preset == "quick":
    p_model, p_samples, p_epochs, p_r, p_lr, p_grad_accum = (
        "Qwen/Qwen2.5-0.5B-Instruct", 100, 1, 8, "2e-4", 4
    )
elif preset == "standard":
    p_model, p_samples, p_epochs, p_r, p_lr, p_grad_accum = (
        "Qwen/Qwen2.5-1.5B-Instruct", 1000, 3, 16, "2e-4", 8
    )
elif preset == "full":
    p_model, p_samples, p_epochs, p_r, p_lr, p_grad_accum = (
        "Qwen/Qwen2.5-1.5B-Instruct", 10000, 5, 32, "1e-4", 8
    )
else:
    p_model, p_samples, p_epochs, p_r, p_lr, p_grad_accum = (
        "Qwen/Qwen2.5-1.5B-Instruct", 1000, 3, 16, "2e-4", 8
    )

# ── Tabs ────────────────────────────────────────────────────────

tab_configure, tab_activity = st.tabs(["⚙️ Configure", "📋 Activity"])

# ── Configure Tab ───────────────────────────────────────────────

with tab_configure:
    col_form, col_preview = st.columns([2, 1])

    with col_form:
        platform_choice = st.segmented_control(
            "Platform",
            ["Apple Silicon (MPS)", "NVIDIA (CUDA)", "CPU"],
            default=default_platform,
            help="Select hardware. 4-bit quantization only on NVIDIA CUDA.",
        ) or default_platform
        is_cuda = "CUDA" in platform_choice

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
            ds1, ds2 = st.columns(2)
            with ds1:
                max_samples = st.number_input("Max Samples", 10, 100000, p_samples, 100)
            with ds2:
                validation_split = st.slider("Validation Split", 0.05, 0.3, 0.1, 0.05)

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

            run_name = st.text_input("Run Name", value=f"{model_name.split('/')[-1].lower()}-{epochs}ep")

            if is_cuda:
                st.subheader("Quantization")
                quant_choice = st.radio(
                    "Mode", ["Full Precision (LoRA)", "4-bit QLoRA"],
                    index=1, horizontal=True,
                )
                quant_bits = 4 if quant_choice == "4-bit QLoRA" else None
            else:
                quant_bits = None
                st.info("Full Precision LoRA — 4-bit requires NVIDIA CUDA", icon="💡")

            submitted = st.form_submit_button("🚀 Start Training", type="primary", width='stretch')

    with col_preview:
        st.subheader("Config Preview")
        config_dict = {
            "model": {"name": model_name, "quantization_bits": quant_bits, "max_length": 512},
            "training": {
                "num_epochs": epochs,
                "batch_size": batch_size,
                "gradient_accumulation_steps": grad_accum,
                "learning_rate": float(learning_rate),
                "output_dir": f"./outputs/{run_name}",
            },
            "lora": {"r": lora_r, "lora_alpha": lora_alpha, "lora_dropout": lora_dropout},
            "data": {"dataset_name": dataset, "max_samples": max_samples, "validation_split": validation_split},
            "logging": {"use_mlflow": True, "use_tensorboard": False},
        }
        st.code(yaml.dump(config_dict, default_flow_style=False), language="yaml")

        # VRAM estimate
        vram_gb = 2.3 if "0.5B" in model_name else 4.5 if "3B" in model_name else 2.3
        if quant_bits == 4:
            vram_gb *= 0.35
        st.caption(f"Estimated VRAM: **~{vram_gb:.1f} GB**")

    if submitted:
        CONFIGS_DIR.mkdir(parents=True, exist_ok=True)
        config_path = CONFIGS_DIR / f"{run_name}.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False)

        from src.tracking.runner import TrainingRunner
        runner = TrainingRunner(project_root=str(PROJECT_ROOT))
        try:
            rid = runner.launch_training(
                technique="sft",
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
        if st.button("🔄 Refresh", width='stretch'):
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
                            f"Technique: {info.get('technique', '?')} | "
                            f"PID: {info.get('pid', '?')}"
                        )
                with c2:
                    status_color = {"running": "🟢", "finished": "✅", "failed": "🔴"}.get(status, "⚪")
                    st.metric("Status", f"{status_color} {status.title()}")
                with c3:
                    if status == "running" and st.button("⏹ Stop", key=f"stop_{run_id}"):
                        runner.stop_training(run_id)
                        st.rerun()
                with c4:
                    if st.button("🗑 Delete", key=f"del_{run_id}"):
                        runner.stop_training(run_id)
                        st.rerun()

                # Live loss from MLflow
                try:
                    import mlflow
                    from ui.config import MLFLOW_TRACKING_URI
                    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
                    runs = mlflow.search_runs(
                        filter_string=f"tags.mlflow.runName='{run_id}'",
                        order_by=["start_time DESC"],
                    )
                    if not runs.empty:
                        run_uuid = runs.iloc[0]["run_id"]
                        loss_history = mlflow.get_metric_history(run_uuid, "loss")
                        if loss_history and len(loss_history) > 1:
                            steps = [m.step for m in loss_history]
                            values = [m.value for m in loss_history]
                            fig = make_metric_timeseries({run_id: list(zip(steps, values))}, "loss")
                            st.plotly_chart(fig, width='stretch', height=200)
                except Exception:
                    pass

                logs = runner.read_recent_logs(run_id, tail=15)
                if logs:
                    with st.expander("Recent Logs"):
                        st.code(logs, language="log")
