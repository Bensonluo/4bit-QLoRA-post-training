"""Ground reviewable SFT/LoRA proposals in local data, hardware and tokenizer facts."""

from __future__ import annotations

import json
import math
import re
import sqlite3
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from src.workbench.materialize import dataset_is_current
from src.workbench.sources import canonical, content_digest
from src.workbench.training_preflight import preflight_dataset
from src.workbench.training_runs import TrainingRunService, model_identity, training_tokenizer

OPTION_FIELDS = {
    "training_options": {
        "num_epochs",
        "batch_size",
        "gradient_accumulation_steps",
        "learning_rate",
        "gradient_checkpointing",
        "warmup_ratio",
        "weight_decay",
        "logging_steps",
        "save_steps",
        "eval_steps",
        "fp16",
        "bf16",
        "max_grad_norm",
    },
    "lora_options": {
        "r",
        "lora_alpha",
        "lora_dropout",
        "target_modules",
        "bias",
        "use_rslora",
        "use_dora",
    },
    "model_options": {
        "quantization_bits",
        "torch_dtype",
        "use_flash_attention",
        "device_map",
        "load_in_8bit",
    },
}
SCOPE = (
    "当前只交接现有 SFT/LoRA。tokenizer 与分区预检不加载训练权重，不能保证模型能装入内存、"
    "所有训练算子兼容或业务效果达标；硬件建议只是起点，最终以真实运行和固定业务评测为准。"
)


def _dataset(session):
    return session.dataset.model_dump() if dataset_is_current(session) else None


def _business(session):
    return {
        "goal": session.goal,
        "task": session.analysis.task.model_dump() if session.analysis else None,
        "recipe": session.analysis.recipe.model_dump()
        if session.analysis and session.analysis.recipe
        else None,
        "questions": [question.model_dump() for question in session.analysis.questions]
        if session.analysis
        else [],
        "training_approach": session.analysis.training_approach if session.analysis else None,
        "capability_gaps": session.analysis.capability_gaps if session.analysis else [],
    }


def _model_facts(model_path):
    from transformers import AutoConfig
    from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES

    path = Path(model_path).expanduser().resolve()
    raw_config = json.loads((path / "config.json").read_text(encoding="utf-8"))
    if not isinstance(raw_config, dict):
        raise ValueError("本地模型 config.json 必须是 JSON 对象。")
    quantization_config = raw_config.get("quantization_config") or {}
    if raw_config.get("quantization") is not None or (
        isinstance(quantization_config, dict)
        and str(quantization_config.get("quant_method", "")).lower() == "mlx"
    ):
        raise ValueError(
            "该候选含 MLX 式预量化配置，不能作为当前 Hugging Face SFT 底座；请选择 HF 原始权重或当前支持的格式。"
        )
    config = AutoConfig.from_pretrained(str(path), local_files_only=True, trust_remote_code=False)
    if config.model_type not in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES or config.is_encoder_decoder:
        raise ValueError("该本地模型不是当前 SFT 路径支持的标准 causal language model。")
    supported = MODEL_FOR_CAUSAL_LM_MAPPING_NAMES[config.model_type]
    supported = (supported,) if isinstance(supported, str) else supported
    if config.architectures and not set(config.architectures) & set(supported):
        raise ValueError(
            "该模型声明的是 embedding/编码器或其他非因果语言模型架构，不能当作当前 SFT 底座。"
        )
    tokenizer = training_tokenizer(path)
    identity = model_identity(path)
    details = config.to_dict()
    capacity = details.get("max_position_embeddings") or details.get("n_positions")
    return {
        "model_path": str(path),
        "status": "available",
        "model_identity": identity,
        "config": {
            key: details[key]
            for key in (
                "model_type",
                "architectures",
                "hidden_size",
                "num_hidden_layers",
                "vocab_size",
                "max_position_embeddings",
                "n_positions",
                "torch_dtype",
                "dtype",
            )
            if key in details
        },
        "context_capacity": capacity,
        "tokenizer": {
            key: getattr(tokenizer, key, None)
            for key in (
                "name_or_path",
                "is_fast",
                "pad_token_id",
                "eos_token_id",
                "padding_side",
                "truncation_side",
                "model_max_length",
            )
        },
        "issues": [],
    }, tokenizer


def _options(proposal, platform):
    from config.base import LoRAConfig, TrainingConfig

    for name, allowed in OPTION_FIELDS.items():
        values = proposal[name]
        if not isinstance(values, dict) or set(values) - allowed:
            raise ValueError(f"{name} 含当前训练方案不支持的字段。")
    training, lora, model = (proposal[key] for key in OPTION_FIELDS)
    TrainingConfig(**training)
    LoRAConfig(**lora)
    for name in (
        "num_epochs",
        "batch_size",
        "gradient_accumulation_steps",
        "learning_rate",
        "r",
        "lora_alpha",
        "max_grad_norm",
    ):
        values = lora if name in ("r", "lora_alpha") else training
        if name in values:
            value = values[name]
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} 必须为有限正数。")
            if (
                name in ("batch_size", "gradient_accumulation_steps", "r", "lora_alpha")
                and type(value) is not int
            ):
                raise ValueError(f"{name} 必须为正整数。")
    for name in ("logging_steps", "save_steps", "eval_steps"):
        if name in training and (type(training[name]) is not int or training[name] <= 0):
            raise ValueError(f"{name} 必须为正整数。")
    for name in ("warmup_ratio", "weight_decay"):
        value = training.get(name, 0)
        if (
            type(value) not in (int, float)
            or not math.isfinite(value)
            or value < 0
            or (name == "warmup_ratio" and value > 1)
        ):
            raise ValueError(f"{name} 数值无效。")
    for values, keys in (
        (training, ("gradient_checkpointing", "fp16", "bf16")),
        (lora, ("use_dora", "use_rslora")),
        (model, ("load_in_8bit", "use_flash_attention")),
    ):
        if any(name in values and type(values[name]) is not bool for name in keys):
            raise ValueError("布尔训练选项必须明确为 true 或 false。")
    if model.get("quantization_bits") not in (None, 4, 8):
        raise ValueError("仅支持 4/8 bit 量化或关闭量化。")
    if (model.get("quantization_bits") or model.get("load_in_8bit")) and not platform["is_cuda"]:
        raise ValueError("当前硬件不支持此训练路径的 bitsandbytes 量化，不能静默改变方案。")
    dtype = model.get("torch_dtype", "bfloat16" if platform["is_cuda"] else "float32")
    if dtype not in ("float32", "float16", "bfloat16"):
        raise ValueError("模型精度无效。")
    if (dtype == "bfloat16" or training.get("bf16")) and not platform["supports_bf16"]:
        raise ValueError("当前硬件没有已确认的 bfloat16 支持。")
    if training.get("fp16") and training.get("bf16"):
        raise ValueError("fp16 和 bf16 不能同时启用。")
    if training.get("fp16") and not platform["is_cuda"]:
        raise ValueError("此路径只在 CUDA 上启用 fp16 mixed precision。")
    if model.get("use_flash_attention"):
        raise ValueError("本轮方案不自动承诺 Flash Attention 内核兼容，请使用标准 attention。")
    if model.get("device_map") not in (None, "auto"):
        raise ValueError("本轮本地交接仅支持默认或 auto device_map。")
    modules = lora.get("target_modules", "all-linear")
    if modules != "all-linear" and not (
        isinstance(modules, list)
        and modules
        and all(isinstance(item, str) and item for item in modules)
    ):
        raise ValueError("LoRA target_modules 必须为 all-linear 或非空模块名列表。")
    # Pin otherwise implicit defaults so saved plans retain their actual precision policy.
    model.setdefault("quantization_bits", 4 if platform["is_cuda"] else None)
    model.setdefault("torch_dtype", dtype)
    model.setdefault("use_flash_attention", False)
    training.setdefault("bf16", platform["is_cuda"] and dtype == "bfloat16")
    training.setdefault("fp16", False)


class TrainingPlanService:
    def __init__(self, root, training_root):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.database = self.root / "plans.sqlite"
        self.training = TrainingRunService(training_root)
        with sqlite3.connect(self.database) as connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS plans (id TEXT PRIMARY KEY, snapshot TEXT NOT NULL)"
            )

    def context(self, session, model_paths: list[str]) -> dict:
        from src.utils.platform_utils import detect_platform, recommend_settings
        from src.workbench.intake_service import next_action

        if not isinstance(model_paths, list) or any(
            not isinstance(path, str) or not path.strip() for path in model_paths
        ):
            raise ValueError("模型候选必须为用户给定的本地目录列表。")
        platform = detect_platform()
        candidates = []
        for path in dict.fromkeys(model_paths):
            try:
                facts, _ = _model_facts(path)
            except (ValueError, OSError, ImportError, TypeError, KeyError) as exc:
                facts = {
                    "model_path": str(Path(path).expanduser().resolve()),
                    "status": "unsupported",
                    "issues": [str(exc)],
                    "model_identity": None,
                    "config": None,
                    "tokenizer": None,
                }
            candidates.append(facts)
        data = _dataset(session)
        action = next_action(session)
        required = {
            "awaiting_analysis": "先用业务目标和样例结构完成数据分析，明确真实输入与监督答案。",
            "needs_business_answers": "先回答 business.questions 中阻止确认的业务问题。",
            "needs_capability": "先解决已列出的数据处理能力缺口，不能跳过实际转换直接训练。",
            "needs_recipe": "先形成可执行的数据处理方案并运行样例预览。",
            "needs_data_revision": "先修正样例中的转换失败或同输入冲突答案，再查看真实预览。",
            "needs_labels": "先补充或核实缺失的监督标签，不能让 Agent 自动编造答案。",
            "review_preview": "先核对真实样例输入与答案，并确认当前业务方案。",
            "awaiting_full_data": "当前只有样例依据，请提交全量资料并沿已确认方案校验。",
            "awaiting_full_validation": "先对已上传全量资料运行当前方案的实际校验。",
            "needs_full_data_revision": "先根据 full_issues 的字段和问题类型修正全量资料，再重新校验。",
            "review_full_data": "先核对全量转换、问题及新增类别，确认实际全量预览。",
            "awaiting_dataset_split": "先按已确认分组生成独立 train/dev/test 分区；无分组时需确认每行是独立对象。",
        }
        full = session.full_data
        if full and action in {"awaiting_full_data", "awaiting_full_validation"}:
            required[action] = "已有全量报告与当前方案不再一致，请用现有全量资料重新校验并确认。"
        readiness = {
            "next_action": action,
            "required_actions": [required[action]] if action in required else [],
            "sample": {
                "scope": session.source.scope,
                "row_count": len(session.source.rows),
                "columns": session.source.columns,
                "preview_counts": session.preview.counts if session.preview else None,
                "confirmed_revision": session.confirmed_revision,
            },
            "full_status": full.status if full else None,
            "full_row_count": len(full.source.rows) if full else None,
            "full_preview_counts": full.preview.counts if full and full.preview else None,
            "full_issues": [
                {
                    "code": issue.code,
                    "severity": issue.severity,
                    "columns": issue.columns,
                    "affected_row_count": len(issue.row_ids),
                }
                for issue in full.issues
            ]
            if full
            else [],
            "scope_note": "仅状态、字段和计数；不包含原始行、模型输入文本或标准答案。",
        }
        return {
            "session_id": session.session_id,
            "session_revision": session.revision,
            "dataset": data,
            "statistics": data["statistics"] if data else None,
            "business": _business(session),
            "data_readiness": readiness,
            "platform": asdict(platform),
            "recommended_settings": recommend_settings(platform),
            "models": candidates,
            "scope_note": SCOPE,
        }

    def probe(self, session, model_path, max_length) -> dict:
        if type(max_length) is not int or max_length <= 0:
            raise ValueError("max_length 必须为正整数。")
        result = {
            "status": "blocked",
            "model_path": str(Path(model_path).expanduser().resolve()),
            "model_identity": None,
            "max_length": max_length,
            "preflight": None,
            "issues": [],
            "scope_note": SCOPE,
        }
        try:
            facts, tokenizer = _model_facts(model_path)
            result["model_identity"] = facts["model_identity"]
            capacity = facts["context_capacity"]
            if isinstance(capacity, int) and max_length > capacity:
                raise ValueError(f"max_length 超过本地模型配置的上下文长度 {capacity}。")
            report = preflight_dataset(session, tokenizer, max_length)
            # Agent sees aggregate facts and concrete risk examples, not every training row.
            rows = report.pop("rows", [])
            report["inspected_rows"] = len(rows)
            report["risk_examples"] = [
                row for row in rows if row.get("was_truncated") or not row.get("supervised_tokens")
            ][:10]
            grouped = {}
            for issue in report["issues"]:
                key = (issue.get("code"), issue["severity"], issue.get("split"))
                if key not in grouped:
                    grouped[key] = {**issue, "row_ids": [], "occurrences": 0}
                grouped[key]["occurrences"] += 1
                grouped[key]["row_ids"] = list(
                    dict.fromkeys(grouped[key]["row_ids"] + issue.get("row_ids", []))
                )[:10]
            report["issues"] = list(grouped.values())
            result.update(status=report["status"], preflight=report, issues=report["issues"])
        except (ValueError, OSError, ImportError, TypeError, KeyError) as exc:
            result["issues"].append({"severity": "blocking", "message": str(exc)})
        return result

    def save(self, session, proposal, context, trace) -> dict:
        proposal = json.loads(canonical(proposal))
        required = {
            "model_path",
            "max_length",
            "rationale",
            "limitations",
            "business_questions",
            "status",
            *OPTION_FIELDS,
        }
        if set(proposal) != required or proposal["status"] not in (
            "ready",
            "needs_data",
            "unsupported",
        ):
            raise ValueError("训练方案字段或状态无效，仅支持当前 SFT/LoRA 方案。")
        for key in ("rationale", "limitations", "business_questions"):
            if not isinstance(proposal[key], list) or any(
                not isinstance(value, str) or not value.strip() for value in proposal[key]
            ):
                raise ValueError(f"{key} 必须是清楚的文本列表。")
        if not proposal["rationale"] or not proposal["limitations"]:
            raise ValueError("请说明方案依据及尚未验证的限制。")
        if (
            context.get("session_id") != session.session_id
            or context.get("session_revision") != session.revision
            or context.get("dataset") != _dataset(session)
            or context.get("business") != _business(session)
        ):
            raise ValueError("方案上下文已过期，请重新读取当前目标和数据。")
        identity, probe = None, None
        if proposal["status"] == "ready":
            if (
                not dataset_is_current(session)
                or proposal["business_questions"]
                or any(question.blocks_confirmation for question in session.analysis.questions)
                or session.analysis.capability_gaps
            ):
                raise ValueError("尚有数据或业务缺口，不能标为 ready。")
            if not isinstance(proposal["model_path"], str) or not proposal["model_path"].strip():
                raise ValueError("ready 方案必须选择已检查的本地模型。")
            path = str(Path(proposal["model_path"]).expanduser().resolve())
            candidate = next(
                (
                    item
                    for item in context.get("models", [])
                    if item["model_path"] == path and item["status"] == "available"
                ),
                None,
            )
            if candidate is None:
                raise ValueError("方案选择的模型不是已经检查的可用候选。")
            from src.utils.platform_utils import detect_platform

            platform = asdict(detect_platform())
            if platform != context["platform"]:
                raise ValueError("硬件条件已变化，请重新检查方案。")
            _options(proposal, platform)
            probe = self.probe(session, path, proposal["max_length"])
            if probe["status"] == "blocked":
                raise ValueError(
                    "实际 tokenizer/数据预检阻断，不能保存 ready 方案："
                    + canonical(probe["issues"])
                )
            identity = probe["model_identity"]
            if identity != candidate["model_identity"]:
                raise ValueError("本地模型内容已变化，请重新检查候选。")
            proposal["model_path"] = path
        record = {
            "plan_id": "tp-" + uuid4().hex,
            "session_id": session.session_id,
            "session_revision": session.revision,
            "dataset": _dataset(session),
            "model_identity": identity,
            "proposal": proposal,
            "status": proposal["status"],
            "context": context,
            "trace": trace,
            "probe": probe,
            "run_id": None,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "scope_note": SCOPE,
        }
        record["proposal_digest"] = content_digest(proposal)
        with sqlite3.connect(self.database) as connection:
            connection.execute(
                "INSERT INTO plans VALUES (?, ?)", (record["plan_id"], canonical(record))
            )
        return record

    def get(self, plan_id):
        if not isinstance(plan_id, str) or not re.fullmatch(r"tp-[0-9a-f]{32}", plan_id):
            raise ValueError("无效训练方案 ID。")
        with sqlite3.connect(self.database) as connection:
            row = connection.execute("SELECT snapshot FROM plans WHERE id=?", (plan_id,)).fetchone()
        if row is None:
            raise ValueError("找不到训练方案。")
        return json.loads(row[0])

    def list_plans(self, session_id=None):
        with sqlite3.connect(self.database) as connection:
            records = [
                json.loads(row[0])
                for row in connection.execute("SELECT snapshot FROM plans ORDER BY rowid DESC")
            ]
        return [
            record for record in records if session_id is None or record["session_id"] == session_id
        ]

    def prepare(self, plan_id, session):
        # Serialize confirmation across UI/CLI processes; a second click reuses the existing run.
        with sqlite3.connect(self.database, timeout=120) as connection:
            connection.execute("BEGIN IMMEDIATE")
            record = self.get(plan_id)
            proposal = record["proposal"]
            if record["status"] != "ready" or proposal["business_questions"]:
                raise ValueError("仅已解决业务问题的 ready 方案可以确认准备。")
            if (
                record["session_id"] != session.session_id
                or record["session_revision"] != session.revision
                or record["dataset"] != _dataset(session)
                or _dataset(session) is None
            ):
                raise ValueError("数据或业务任务已更新，请重新生成方案。")
            if record["proposal_digest"] != content_digest(proposal):
                raise ValueError("已保存方案内容不一致。")
            if model_identity(Path(proposal["model_path"])) != record["model_identity"]:
                raise ValueError("本地模型内容已变化，请重新生成方案。")
            if record["run_id"] is None:
                from src.utils.platform_utils import detect_platform

                _options(proposal, asdict(detect_platform()))
                run = self.training.prepare(
                    session,
                    proposal["model_path"],
                    **{key: proposal[key] for key in ("max_length", *OPTION_FIELDS)},
                    plan_trace=record.get("trace"),
                )
                record["run_id"] = run["run_id"]
                record["prepared_at"] = datetime.now(timezone.utc).isoformat()
                connection.execute(
                    "UPDATE plans SET snapshot=? WHERE id=?", (canonical(record), plan_id)
                )
            else:
                run = self.training.get_status(record["run_id"])
            return {"plan_id": plan_id, "run_id": record["run_id"], "training_run": run}
