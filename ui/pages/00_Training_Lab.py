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

st.set_page_config(page_title="训练实验室", page_icon="🏋️", layout="wide")


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


# ── 失败 run 非专家急救包（R124）──────────────────────────────────
# 红徽章之后不能是死胡同：「为什么失败 / 怎么重试」在卡片内就地回答。
# 签名→对策按通用训练失败设计（OOM 恢复口径与 CLAUDE.md 一致），不绑任何域。

_FAILURE_TRIAGE: list[tuple[tuple[str, ...], str, str]] = [
    (
        ("out of memory", "outofmemoryerror", "cuda oom"),
        "显存/内存不足",
        "配置页降一档再启动：max_length 1024→512、LoRA r 16→8，或换更小模型（如 Qwen 0.5B/0.6B）。",
    ),
    (
        ("filenotfounderror", "no such file", "jsondecodeerror"),
        "文件/数据路径问题",
        "确认数据集路径存在且为合法 JSON（Alpaca 格式）；提交前用配置页「🩺 数据集体检与预览」预检。",
    ),
    (
        ("importerror", "modulenotfounderror", "bitsandbytes"),
        "依赖/环境问题",
        '4-bit 量化仅 CUDA 可用（MPS/CPU 会自动切 bf16）；缺依赖用 `pip install -e ".[ui]"` 补齐。',
    ),
    (
        ("connectionerror", "timed out", "huggingface.co", "hf-mirror"),
        "模型下载/网络问题",
        "国内网络先设 `export HF_ENDPOINT=https://hf-mirror.com` 再启动。",
    ),
]


def _diagnose_failure(log_text: str) -> list[tuple[str, str]]:
    """按签名从最近日志匹配失败原因，返回 (诊断, 对策) 列表——无命中返回空。

    签名统一小写匹配（大小写不敏感）；日志为空（无日志文件/未写盘）同样
    返回空，由调用侧走未命中分支的通用三步。
    """
    lowered = log_text.lower()
    return [(diag, fix) for sigs, diag, fix in _FAILURE_TRIAGE if any(s in lowered for s in sigs)]


def _render_next_steps(run_id: str, info: dict) -> None:
    """训练完成 ≠ 终点：定位产物，给出评测/注册/导出的下一步动作。"""
    from src.tracking.next_steps import summarize_run_artifacts

    try:
        arts = summarize_run_artifacts(Path(info.get("config_path", "")), PROJECT_ROOT)
    except Exception as exc:  # 引导面板失败不应影响运行列表本身
        logger.debug("next-steps summary failed for %s: %s", run_id, exc)
        return
    with st.expander("🧭 下一步"):
        if arts.has_adapter and arts.output_dir is not None:
            st.success(f"Adapter 就绪：`{arts.output_dir}`")
            st.markdown(f"想先直观感受效果？到 **💬 Chat** 页选 `{arts.output_dir}` 直接对话。")
            # 页内评测（R120 通用评测闭环）：Data Wizard 备好 test 集时，用
            # 用户自己领域的数据评（scripts/eval_entity_match.py → 落盘
            # domains/entity_matching/data/results/ → 点亮 02/03 两页）；
            # 无 test 集时回退内置实体匹配案例评测（R113 路径，保持不变）
            from src.tracking.runner import TrainingRunner

            _eval_runner = TrainingRunner(project_root=str(PROJECT_ROOT))
            _wizard_test = arts.eval_sets.get("test")
            if _wizard_test:
                _eval_rid = f"enteval-{run_id}"
                _eval_status = _eval_runner.get_status(_eval_rid)
                if _eval_status == "running":
                    st.info(
                        "⏳ 评测运行中（后台，用你自己的 test 集）——本页每 30 秒"
                        "自动刷新；完成后到「评测结果」「模型对比」页查看"
                        "（域=实体匹配（通用））。"
                    )
                    _eval_logs = _eval_runner.read_recent_logs(_eval_rid, tail=15)
                    if _eval_logs:
                        with st.expander("评测日志（最近 15 行）"):
                            st.code(_eval_logs, language="log")
                elif _eval_status == "finished":
                    st.success(
                        "✅ 评测完成——到「评测结果」「模型对比」页查看"
                        "（域=实体匹配（通用），两页自动取最新结果，无需导入）。"
                    )
                else:
                    if _eval_status == "failed":
                        st.error(
                            "评测失败（退出码非 0）——日志见「训练动态」里的 "
                            "enteval 行（最近 15 行，完整文件在 outputs/logs/）。"
                        )
                    else:
                        st.markdown("**页内评测**（用你自己导出的 test 集）")
                        st.caption(
                            "后台加载 adapter，在你的 test.json 上逐条生成并判分"
                            "（与训练同款输入格式，任意领域通用）；完成后"
                            "「评测结果」「模型对比」两页自动可看。"
                            "Apple Silicon 约需数分钟。"
                        )
                    if st.button("⚡ 用我的 test 集评测", key=f"eval_btn_{run_id}", type="primary"):
                        try:
                            _rid = _eval_runner.launch_entity_eval(
                                run_id, str(arts.output_dir), str(_wizard_test)
                            )
                            st.toast(f"评测已启动：{_rid}")
                            st.rerun()
                        except Exception as exc:
                            st.error(f"评测启动失败：{exc or exc!r}")
                st.code(
                    f"python scripts/eval_entity_match.py "
                    f"--model-path {arts.output_dir} --test-file {_wizard_test}",
                    language="bash",
                )
            else:
                # 页内评测（R113）：后台跑领域评测，产物含 baseline + 微调两模型，
                # 直接点亮「评测结果 / 模型对比」两页（load_eval_data 每 rerun
                # glob 取最新，无需导入）
                _eval_rid = f"eval-{run_id}"
                _eval_status = _eval_runner.get_status(_eval_rid)
                if _eval_status == "running":
                    st.info(
                        "⏳ 评测运行中（后台）——本页每 30 秒自动刷新；"
                        "完成后到「评测结果」「模型对比」页查看。"
                    )
                    _eval_logs = _eval_runner.read_recent_logs(_eval_rid, tail=15)
                    if _eval_logs:
                        with st.expander("评测日志（最近 15 行）"):
                            st.code(_eval_logs, language="log")
                elif _eval_status == "finished":
                    st.success(
                        "✅ 评测完成——到「评测结果」「模型对比」页查看"
                        "（两页自动取最新结果，无需导入）。"
                    )
                else:
                    if _eval_status == "failed":
                        st.error(
                            "评测失败（退出码非 0）——日志见「训练动态」里的 eval 行"
                            "（最近 15 行，完整文件在 outputs/logs/）；"
                            "若「评测结果」页已有新结果，以页面为准"
                            "（结果先落盘、收尾后置）。"
                        )
                    else:
                        st.markdown("**页内评测**（内置实体匹配案例）")
                        st.caption(
                            "后台用内置**实体匹配案例**（药品名归一化）的自带测试集评测 adapter："
                            "基线全量 3,136 条 + 你的模型采样 500 条"
                            "（**与你本次训练使用的数据无关**）；"
                            "完成后「评测结果」「模型对比」两页自动可看。"
                            "Apple Silicon 约需数分钟到数十分钟。"
                        )
                    if st.button("⚡ 生成评测文件", key=f"eval_btn_{run_id}", type="primary"):
                        try:
                            _rid = _eval_runner.launch_eval(run_id, str(arts.output_dir))
                            st.toast(f"评测已启动：{_rid}")
                            st.rerun()
                        except Exception as exc:
                            st.error(f"评测启动失败：{exc or exc!r}")
            if arts.registered_name:
                st.info(
                    f"已配置自动注册为 `{arts.registered_name}`（Staging）——"
                    f"到 **Registry** 页查看版本、设置 champion/challenger 别名。"
                )
            else:
                merged_dir = f"outputs/merged/{run_id}"
                merged_path = PROJECT_ROOT / merged_dir
                st.markdown("**合并导出 / 注册**")
                from src.inference.discovery import looks_merged

                if looks_merged(merged_path):
                    st.success(
                        f"已合并：`{merged_dir}` —— 到 **💬 Chat** 页选 📦 `{run_id}` 直接对话，"
                        f"或用下方命令注册进 Registry。"
                    )
                else:
                    base_override = st.text_input(
                        "底座路径覆盖（可选；adapter_config 记录的底座不在本机时填本地目录或 HF 名）",
                        key=f"merge_base_{run_id}",
                    )
                    if st.button("📦 合并导出", key=f"merge_btn_{run_id}", type="primary"):
                        from src.models.merger import merge_adapter_to_dir

                        try:
                            with st.spinner(
                                "合并中：加载底座 → 合并 LoRA → 写盘（1.7B 约 1–3 分钟）…"
                            ):
                                merge_adapter_to_dir(
                                    str(arts.output_dir),
                                    str(merged_path),
                                    base_model_name=base_override.strip() or None,
                                )
                            st.toast("合并完成", icon="📦")
                            st.rerun()  # 重渲染为「已合并 ✅」状态
                        except Exception as exc:
                            st.error(f"合并失败：{exc or exc!r}")
                model_tag = arts.model_name.split("/")[-1] if arts.model_name else "MyModel"
                st.code(
                    f"python scripts/registry_cli.py register --model-dir {merged_dir} "
                    f"--name {model_tag}-QLoRA",
                    language="bash",
                )
        else:
            st.warning(
                f"输出目录中未找到 adapter：`{arts.output_dir}`。"
                f"先看下方日志确认保存路径或失败原因，再决定重训或手动定位。"
            )


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
    st.title("🏋️ 训练实验室")
with col_presets:
    st.markdown("<div style='padding-top:0.8rem'></div>", unsafe_allow_html=True)
    pcols = st.columns(3)
    with pcols[0]:
        if st.button("⚡ 快速测试", width="stretch", help="小模型，100 条样本，1 轮"):
            st.session_state["preset"] = "quick"
            st.rerun()
    with pcols[1]:
        if st.button("🔥 标准", width="stretch", help="完整模型，1,000 条样本，3 轮"):
            st.session_state["preset"] = "standard"
            st.rerun()
    with pcols[2]:
        if st.button("🚀 完整运行", width="stretch", help="完整模型，10,000 条样本，5 轮"):
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

tab_configure, tab_activity = st.tabs(["⚙️ 配置", "📋 训练动态"])

# ── Configure Tab ───────────────────────────────────────────────

with tab_configure:
    # 向导交接横幅：Data Wizard 点「送去训练」跳转而来时，dataset 字段已预填
    handoff = st.session_state.pop("wizard_handoff", None)
    if handoff:
        st.success(
            f"数据集已从 **Data Wizard** 预填：`{handoff['path']}`"
            f"（train {handoff['samples']} 条，已通过 7 项数据体检）。"
            f"确认下方参数后点「开始训练」即可。"
        )

    col_form, col_preview = st.columns([2, 1])

    with col_form:
        platform_choice = (
            st.segmented_control(
                "运行平台",
                ["Apple Silicon (MPS)", "NVIDIA (CUDA)", "CPU"],
                default=default_platform,
                help="选择硬件。4-bit 量化仅在 NVIDIA CUDA 可用。",
            )
            or default_platform
        )
        is_cuda = "CUDA" in platform_choice

        # Radio lives OUTSIDE the form so switching technique re-renders the
        # form (technique-specific sections) immediately, without a submit.
        st.subheader("技术")
        technique_label = st.radio(
            "训练后技术",
            ["SFT", "DPO", "GRPO"],
            index=0,
            horizontal=True,
            help="DPO 需要偏好数据集（prompt/chosen/rejected）；GRPO 需要 prompt+答案数据。",
        )
        technique = technique_label.lower()

        # Registry 开关的兜底初值：GRPO 技术不渲染该区块（grpo_trainer 未实现自动注册），
        # 预览/提交逻辑仍可安全引用这些变量
        register_model = False
        registry_name = ""
        merge_before_register = True

        # 高级参数 toggle 置表单外（与技术单选同理：表单内 toggle 须提交才
        # 生效，无法即时展开专家区块）
        advanced = st.toggle(
            "高级参数",
            value=False,
            help="默认隐藏专家参数（学习率/批大小/LoRA 等），预设值即可开跑；需要精调时展开。",
        )

        # 高级隐藏时的预设兜底（上面临近的 GRPO 机制同一范式）：config_dict
        # 无条件读这些变量，隐藏的专家参数以预设值流入契约，而不是凭空消失
        validation_split = 0.1
        max_length = 512
        batch_size = 1
        grad_accum = p_grad_accum
        lora_r = p_r
        lora_alpha = p_r * 2
        lora_dropout = 0.05
        lr_value = float(p_lr)  # 预设值是代码常量，解析不可能失败
        quant_bits = 4 if is_cuda else None  # 平台默认：CUDA 上 4-bit QLoRA

        with st.form("training_config"):
            st.subheader("模型与数据")
            c1, c2 = st.columns([3, 1])
            with c1:
                model_options_list = list(MODEL_OPTIONS.keys())
                model_name = st.selectbox(
                    "底座模型",
                    model_options_list,
                    index=model_options_list.index(p_model) if p_model in model_options_list else 0,
                    format_func=lambda x: f"{x} ({MODEL_OPTIONS[x]})",
                )
            with c2:
                st.markdown("<div style='padding-top:1.8rem'></div>", unsafe_allow_html=True)
                if not is_cuda:
                    st.badge("全精度", color="blue")
                else:
                    st.badge("4-bit QLoRA", color="green")

            # key 模式：Data Wizard 交接时可从 session_state 预填（见 05_Data_Wizard 收尾）
            dataset = st.text_input(
                "数据集（HF 名称或本地路径）",
                value="yahma/alpaca-cleaned",
                key="dataset_input",
            )
            ds1, ds2, ds3 = st.columns(3)
            with ds1:
                max_samples = st.number_input("最大样本数", 10, 100000, p_samples, 100)
            if advanced:
                with ds2:
                    validation_split = st.slider("验证集比例", 0.05, 0.3, 0.1, 0.05)
                with ds3:
                    max_length = st.number_input(
                        "最大长度", 128, 8192, 512, 64, help="序列长度预算（token）"
                    )

            st.subheader("训练参数")
            t1, t2, t3, t4 = st.columns(4)
            with t1:
                epochs = st.number_input("轮数", 1, 50, p_epochs)
            if advanced:
                with t2:
                    # 学习率挡位化：非专家不必懂「2e-4」记法，解析错误路径
                    # 随自由文本整体退场；挡位覆盖两个预设值（1e-4/2e-4）
                    lr_tiers = [1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3]
                    lr_value = st.select_slider(
                        "学习率",
                        options=lr_tiers,
                        value=float(p_lr) if float(p_lr) in lr_tiers else 2e-4,
                        format_func=lambda v: f"{v:.0e}".replace("e-0", "e-"),
                        help="预设与教程常用挡位；不确定就保持默认",
                    )
                with t3:
                    batch_size = st.number_input("批大小", 1, 8, 1)
                with t4:
                    grad_accum = st.number_input("梯度累积", 1, 32, p_grad_accum)

                effective_bs = batch_size * grad_accum
                st.caption(f"实际批大小：**{effective_bs}**")

            if advanced:
                st.subheader("LoRA")
                l1, l2, l3 = st.columns(3)
                with l1:
                    lora_r = st.slider("LoRA 秩（r）", 4, 64, p_r, 4)
                with l2:
                    lora_alpha = st.number_input("LoRA Alpha", value=lora_r * 2)
                with l3:
                    lora_dropout = st.slider("Dropout 比例", 0.0, 0.3, 0.05, 0.01)

            run_name = st.text_input(
                "运行名",
                # 模型名里的 "." 换成 "-"：默认值必须能通过 _validate_run_name
                # （字母数字下划线连字符），否则用户不改任何参数首跑就报错。
                value=f"{model_name.split('/')[-1].lower().replace('.', '-')}-{epochs}ep",
            )

            if advanced and technique != "grpo":
                st.subheader("模型注册表")
                reg1, reg2 = st.columns([1, 2])
                with reg1:
                    register_model = st.checkbox(
                        "训练后自动注册",
                        value=False,
                        help="训练完成 → 合并 LoRA → 注册为 Registry 新版本（Staging）；"
                        "注册失败不影响训练产物",
                    )
                with reg2:
                    registry_name = st.text_input(
                        "Registry 模型名",
                        value=f"{model_name.split('/')[-1]}-QLoRA",
                        help="同名注册追加新版本，之后可在 Registry 页用 champion/challenger 别名管理",
                    )
                    merge_before_register = st.checkbox(
                        "注册前合并进底座",
                        value=True,
                        help="注册合并后的完整模型（推荐，可直接部署）；关闭则只注册 adapter",
                    )

            if technique == "dpo":
                st.subheader("DPO")
                d1, d2 = st.columns(2)
                with d1:
                    dpo_beta = st.slider(
                        "Beta（β）",
                        0.01,
                        0.5,
                        0.1,
                        0.01,
                        help="偏好强度——越低越贴近参考模型",
                    )
                with d2:
                    ref_model = st.selectbox(
                        "参考模型",
                        model_options_list,
                        index=0,
                        format_func=lambda x: f"{x} ({MODEL_OPTIONS[x]})",
                        help="偏好锚定用的冻结模型——选小的省显存",
                    )
            elif technique == "grpo":
                st.subheader("GRPO")
                g1, g2, g3 = st.columns(3)
                with g1:
                    grpo_beta = st.slider(
                        "Beta（β）",
                        0.0,
                        0.2,
                        0.04,
                        0.01,
                        help="KL 惩罚强度（0 = 关闭惩罚）",
                    )
                with g2:
                    num_generations = st.number_input("每提示词生成数", 2, 16, 4)
                with g3:
                    reward_funcs = st.multiselect(
                        "奖励函数",
                        ["format", "accuracy", "length", "cosine", "llm_judge"],
                        default=["format", "accuracy"],
                        help="已注册奖励（src/training/reward_engine.py）",
                    )

            if advanced and is_cuda:
                st.subheader("量化")
                quant_choice = st.radio(
                    "模式",
                    ["全精度（LoRA）", "4-bit QLoRA"],
                    index=1,
                    horizontal=True,
                )
                quant_bits = 4 if quant_choice == "4-bit QLoRA" else None
            elif not is_cuda:
                st.info("全精度 LoRA——4-bit 量化仅在 NVIDIA CUDA 可用", icon="💡")

            submitted = st.form_submit_button("🚀 开始训练", type="primary", width="stretch")

    with col_preview:
        st.subheader("配置预览")
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
            "logging": (
                {
                    "use_mlflow": True,
                    "use_tensorboard": False,
                    "register_model": True,
                    "registry_model_name": registry_name.strip(),
                    "merge_before_register": merge_before_register,
                }
                if register_model
                else {"use_mlflow": True, "use_tensorboard": False}
            ),
            **technique_sections,
        }
        st.code(yaml.dump(config_dict, default_flow_style=False), language="yaml")
        if not advanced:
            st.caption("高级参数未展开：使用预设默认值（展开「高级参数」可调整）。")

        # VRAM estimate — table-driven from MODEL_OPTIONS (whose values are
        # 4-bit estimates); full-precision scales weights by ~1/0.35 (bf16 vs NF4).
        vram_match = re.search(r"([\d.]+)\s*GB", MODEL_OPTIONS.get(model_name, ""))
        vram_gb = float(vram_match.group(1)) if vram_match else 2.3
        if quant_bits != 4:
            vram_gb /= 0.35
        st.caption(f"显存估算：**~{vram_gb:.1f} GB**")

    # ── 数据集体检与预览（LLaMA-Board「Preview dataset」式，提交前可主动查看）────
    with st.expander("🔍 数据集体检与预览", expanded=False):
        st.caption(
            "不用先提交：填路径点「体检」，看格式识别、样本条数与前 3 条样本。"
            "提交时仍会强制预检（坏路径/坏格式照样拦下）。"
        )
        pv1, pv2 = st.columns([3, 1])
        with pv1:
            pv_dataset = st.text_input(
                "数据集路径（本地 .json/.jsonl）",
                value=st.session_state.get("dataset_input", ""),
                key="preflight_dataset_input",
                placeholder="outputs/wizard/demo_drugs/train.json",
            )
        with pv2:
            st.markdown("<div style='padding-top:1.55rem'></div>", unsafe_allow_html=True)
            pv_check = st.button("🔍 体检", width="stretch")
        if pv_check:
            from src.data.preflight import check_dataset_for_sft, load_preview_records

            pv_path = pv_dataset.strip()
            if not pv_path:
                st.warning("先填一个数据集路径（或直接提交，提交时也会强制预检）。")
            else:
                with st.spinner("正在体检数据集格式与样本…"):
                    pv_errors, pv_insp = check_dataset_for_sft(pv_path)
                if pv_insp is None:
                    st.info(
                        f"`{pv_path}` 按 HF 数据集名处理（无 .json/.jsonl 后缀），"
                        f"本地不做检查，由训练时的加载器解析。"
                    )
                else:
                    for msg in pv_errors:
                        st.error(msg)
                    if not pv_errors:
                        st.success(
                            f"体检通过：**{pv_insp.fmt}** 格式，{pv_insp.n_records} 条样本"
                            f"（SFT 可直接训练）。"
                        )
                        with st.spinner("正在读取样本预览…"):
                            pv_records, pv_err = load_preview_records(Path(pv_path), limit=3)
                        if pv_err:
                            st.error(f"预览读取失败：{pv_err}")
                        else:
                            for i, rec in enumerate(pv_records, 1):
                                with st.expander(f"样本 {i} / {min(3, pv_insp.n_records)}"):
                                    if "messages" in rec:
                                        _ROLE_ICON = {
                                            "system": "🧭 system",
                                            "user": "👤 user",
                                            "assistant": "🤖 assistant",
                                        }
                                        for msg in rec["messages"]:
                                            label = _ROLE_ICON.get(msg["role"], msg["role"])
                                            st.markdown(f"**{label}**")
                                            st.code(str(msg["content"]), language=None)
                                    else:
                                        st.markdown(
                                            f"**instruction**\n\n{rec.get('instruction', '')}"
                                        )
                                        st.markdown(
                                            f"**input**\n\n```\n{rec.get('input', '')}\n```"
                                        )
                                        st.markdown(
                                            f"**output**\n\n```\n{rec.get('output', '')}\n```"
                                        )

    if submitted:
        error = _validate_run_name(run_name)
        if not error and technique == "grpo" and not reward_funcs:
            error = "GRPO 至少需要一个奖励函数。"
        if not error and register_model and not registry_name.strip():
            error = "Registry 模型名不能为空（或取消勾选自动注册）。"
        if error:
            st.error(error)
            st.stop()

        # 数据集预检（SFT）：本地文件在启动前体检——坏路径/坏格式在这里拦下，
        # 而不是让训练子进程下完模型后死在数据加载。HF 数据集名不做本地检查。
        # DPO/GRPO 的数据契约不同（prompt/chosen/rejected 等），不做误导性校验。
        if technique == "sft":
            from src.data.preflight import check_dataset_for_sft

            with st.spinner("正在体检数据集格式与样本…"):
                ds_errors, ds_insp = check_dataset_for_sft(dataset)
            if ds_errors:
                for msg in ds_errors:
                    st.error(msg)
                st.stop()
            if ds_insp is not None:
                st.toast(f"数据集体检通过：{ds_insp.fmt} 格式，{ds_insp.n_records} 条样本")

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
            st.success(f"训练已启动：`{rid}`")
            st.info("切换到**训练动态**标签页查看进度。")
        except Exception as e:
            st.error(f"启动失败：{e}")


# ── Activity Tab ────────────────────────────────────────────────


def _render_activity() -> None:
    """Activity 全量渲染体(R104 自动刷新的 fragment 边界):边界必须包含
    runner 构造与 run 列表读取——每 tick 重读 .run_meta.json,新启动/新删除
    的 run 在 fragment 节拍内也可见;页面容器(container/columns)在函数体内
    创建(fragment 体内建的容器才随节拍更新)。"""
    from src.tracking.runner import TrainingRunner

    runner = TrainingRunner(project_root=str(PROJECT_ROOT))
    all_runs = runner.list_all_runs()

    # Header with refresh
    h1, h2 = st.columns([4, 1])
    with h1:
        st.subheader("训练动态")
    with h2:
        if st.button("🔄 刷新", width="stretch"):
            st.rerun()

    # 运行状态枚举 → 中文显示(与 01 页 _STATUS_ZH 对齐;状态值本身不动;
    # unknown 也不裸显枚举——r106-reviewer nit-1:旧 .title() 显 "Unknown")
    _run_status_zh = {
        "running": "运行中",
        "finished": "已完成",
        "failed": "失败",
        "unknown": "未知",
    }

    if not all_runs:
        st.info("暂无训练运行记录。可在「配置」标签页发起第一次训练。")
    else:
        for run_id in reversed(all_runs[-10:]):
            info = runner.get_run_info(run_id)
            status = runner.get_status(run_id)

            with st.container(border=True):
                c1, c2, c3, c4 = st.columns([3, 1, 1, 1])
                with c1:
                    st.markdown(f"**{run_id}**")
                    if info:
                        # 评测行显示中文名（R113/R120）：领域/通用评测与训练同池
                        _tech = info.get("technique", "?")
                        _tech = "领域评测" if _tech in ("medical_eval", "entity_eval") else _tech
                        st.caption(f"技术：{_tech} | PID：{info.get('pid', '?')}")
                with c2:
                    status_color = {"running": "🟢", "finished": "✅", "failed": "🔴"}.get(
                        status, "⚪"
                    )
                    st.metric("状态", f"{status_color} {_run_status_zh.get(status, status)}")
                with c3:
                    # 停止只面向运行中的 run(r106-reviewer should-fix-1):
                    # stop_training 对已结束进程是静默 no-op,无条件渲染面板
                    # 会让 caption 对已完成/失败的 run 说谎。恢复 HEAD 原有门。
                    if status == "running":
                        with st.popover("⏹ 停止", key=f"stop_{run_id}", use_container_width=True):
                            # 停止=杀掉在途进程,高代价误触面——popover 二次确认;
                            # 文案按行类型分支(R114 obs-6):eval 行无 checkpoint
                            # (R120: entity_eval 通用评测行同属 eval 文案分支)
                            if info.get("technique") == "medical_eval" or (
                                info.get("technique") == "entity_eval"
                            ):
                                st.caption("将终止该运行的评测进程；已写入的日志保留。")
                            else:
                                st.caption(
                                    "将终止该运行的训练进程；已保存的 checkpoint 与日志保留。"
                                )
                            if st.button("确认停止", key=f"stop_confirm_{run_id}", type="primary"):
                                runner.stop_training(run_id)
                                st.rerun()
                with c4, st.popover("🗑 删除", key=f"del_{run_id}", use_container_width=True):
                    st.caption("移除本地运行记录；config、日志与 MLflow 数据保留。")
                    if st.button("确认删除", key=f"del_confirm_{run_id}", type="primary"):
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
                    with st.expander("最近日志"):
                        st.code(logs, language="log")

                # 失败 run 的非专家急救包（R124）：红徽章之后不能是死胡同。
                # 签名命中 → 就地给「疑似原因 → 对策」；未命中 → 通用三步
                # （看日志尾 → 对照清单 → 配置页重试）。评测行（R113/R120
                # 技术值）的失败重试点回下一步面板，不指配置页。
                if status == "failed":
                    hits = _diagnose_failure(logs)
                    with st.expander("🩹 失败诊断与重试"):
                        if hits:
                            st.markdown("**从最近日志匹配到可能原因：**")
                            for diag, fix in hits:
                                st.markdown(f"- 🔴 **{diag}** → {fix}")
                        else:
                            st.markdown(
                                "**未匹配到已知签名**——先展开「最近日志」看最后几行，"
                                "多数失败原因在最后一条 traceback 里。常见对照："
                            )
                            for _, diag, fix in _FAILURE_TRIAGE:
                                st.markdown(f"- **{diag}**：{fix}")
                        if info and info.get("technique") in ("medical_eval", "entity_eval"):
                            st.markdown(
                                "检查 --test-file 路径后，到已完成训练的「🧭 下一步」重新发起评测；"
                                "本条记录可 🗑 删除，日志保留可查。"
                            )
                        else:
                            st.markdown(
                                "改完参数回到「配置」标签页即可重新启动；本条运行记录可 🗑 删除，"
                                "config 与日志保留可查。"
                            )

                # 训练完成后的下一步引导（LlamaBoard Chat/Evaluate/Export 式收尾）。
                # medical_eval 行除外（R113）：评测行没有「训练下一步」可言，
                # 渲染面板会出现 eval-of-eval 嵌套按钮（R120：entity_eval 同门）
                if (
                    status == "finished"
                    and info
                    and info.get("technique") != "medical_eval"
                    and info.get("technique") != "entity_eval"
                ):
                    _render_next_steps(run_id, info)


with tab_activity:
    # 条件应用(R104):仅当存在活跃训练时挂 30s 节拍(与 ui/queries.py 的
    # 30s TTL 缓存对齐,更短只会放大 TTL 过期时的整段重查卡顿);run_every
    # 无数据驱动的停止条件,活跃与否只能在全量重跑时探测。Stop/Delete/刷新
    # 保持全应用 st.rerun() 不变(scope="fragment" 会让它们只重跑局部)。
    from src.tracking.runner import TrainingRunner

    _probe = TrainingRunner(project_root=str(PROJECT_ROOT))
    if _probe.list_active():
        st.fragment(run_every="30s")(_render_activity)()
    else:
        _render_activity()
