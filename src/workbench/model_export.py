"""环节⑨交接出口:把成功训练的适配器合并成可独立加载的模型目录。

北极星九环节的最后一环是「合并导出为可被 vLLM/Ollama/LM Studio 直接加载的
模型,附完整证据链」。此前 merge_adapter_to_dir 只存在于框架层脚本,工作台
旅程(数据 → 训练 → 对照 → 验收 → 决策)走完后没有出口——用户拿到了验收结论,
却没有任何入口把模型带出工作台。本模块补上这一环:只读盘点(plan_model_export)
与实际合并(export_model),合并复用框架层 merge_adapter_to_dir,不另造合并逻辑。
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

# 与 Training Lab「合并导出」及 Chat 页发现共用同一幂等判定:
# config.json + 权重、无 adapter 标志 = 自包含已合并模型。
from src.inference.discovery import looks_merged


def default_export_dir(training_root: Path, run_id: str) -> Path:
    """默认导出目录:训练根目录的 merged 兄弟目录下按 run_id 命名。

    与 outputs/merged 的仓库级约定同构;按 run_id 命名让目录自带血缘指向。
    """
    return Path(training_root).parent / "merged" / run_id


def plan_model_export(run_record: dict, output_dir: str | Path) -> dict:
    """只读盘点一次训练能否合并导出;不加载模型、不写任何文件。

    页面用这份 plan 渲染「📦 合并导出」折叠区(只读盘点,不在页面执行合并);
    CLI 用它决定是否继续执行 export_model。阻塞原因逐条如实点名,不静默降级。
    """
    target = Path(output_dir)
    run_id = str(run_record.get("run_id") or "")
    adapter_dir = Path(run_record.get("output_dir") or "")
    base = run_record.get("model_path")
    has_adapter = (adapter_dir / "adapter_config.json").exists() and (
        (adapter_dir / "adapter_model.safetensors").exists()
        or (adapter_dir / "adapter_model.bin").exists()
    )
    reasons: list[str] = []
    if run_record.get("status") != "succeeded":
        status = "not_succeeded"
        reasons.append(f"只有成功完成的训练才能合并导出；这次运行当前状态是 {run_record.get('status') or '未知'}。")
    elif not has_adapter:
        status = "adapter_missing"
        reasons.append(f"训练记录为成功，但产物目录缺少完整的 adapter 文件：{adapter_dir}")
    elif not base or not Path(str(base)).expanduser().exists():
        status = "base_missing"
        reasons.append(f"训练使用的基础模型目录当前不存在：{base or '记录中缺失'}")
    elif target.exists() and any(target.iterdir()) and not looks_merged(target):
        status = "target_conflict"
        reasons.append(
            f"导出目录已存在且不是完整的已合并模型，请换一个目录或先移走现有内容：{target}"
        )
    else:
        status = "already_exported" if looks_merged(target) else "ready"
    return {
        "status": status,
        "run_id": run_id,
        "session_id": run_record.get("session_id"),
        "dataset_version": run_record.get("dataset_version"),
        "base_model_path": base,
        "adapter_dir": str(adapter_dir),
        "output_dir": str(target),
        "reasons": reasons,
    }


def export_model(plan: dict) -> dict:
    """按 plan 执行合并导出;ready 才动手,already_exported 幂等返回,其余拒绝。

    合并逻辑复用框架层 merge_adapter_to_dir(基座路径来自训练记录,已是本地
    绝对路径,不经过 HF hub 解析);导出完成后在模型目录写入 export_evidence.json
    证据链——拿到模型的人能看到它来自哪次训练、哪个数据版本、哪个基础模型。
    """
    state = plan.get("status")
    if state == "already_exported":
        return {
            "status": "already_exported",
            "run_id": plan.get("run_id"),
            "output_dir": plan.get("output_dir"),
        }
    if state != "ready":
        reasons = plan.get("reasons") or [f"当前状态 {state} 不能合并导出。"]
        raise ValueError("；".join(reasons))
    # 延迟导入:merger 顶层依赖 peft/transformers,工作台 CLI 与页面盘点路径
    # 必须保持轻量(与 train-lineage 延迟导入 mlflow 同一纪律)。
    from src.models.merger import merge_adapter_to_dir

    merged_dir = merge_adapter_to_dir(
        plan["adapter_dir"], plan["output_dir"], base_model_name=plan["base_model_path"]
    )
    evidence = {
        "run_id": plan.get("run_id"),
        "session_id": plan.get("session_id"),
        "dataset_version": plan.get("dataset_version"),
        "base_model_path": plan.get("base_model_path"),
        "adapter_dir": plan.get("adapter_dir"),
        "exported_at": datetime.now(timezone.utc).isoformat(),
    }
    # 证据文件随模型走;一个额外 JSON 不影响 looks_merged 幂等判定。
    (Path(merged_dir) / "export_evidence.json").write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {"status": "exported", **evidence, "output_dir": str(merged_dir)}
