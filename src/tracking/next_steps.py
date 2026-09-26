"""训练完成后的「下一步」引导：从 run 配置定位产物与评测集，给出可执行动作。

行业惯例（LlamaBoard 的 Chat/Evaluate/Export 收尾、H2O 实验完成后的
chat/export）：训练结束不应该是死胡同。本模块做纯文件系统的事实收集——
adapter 在哪、向导是否备好了评测集、是否已配置自动注册——UI 据此渲染
下一步动作（评测命令 / 合并导出 / 注册 / 对比）。不加载任何模型。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from src.data.preflight import looks_like_local_path


@dataclass(frozen=True)
class RunArtifacts:
    """一次已完成训练的产物清单（全部来自文件系统事实，无猜测）。"""

    output_dir: Path | None
    has_adapter: bool
    dataset_name: str
    model_name: str
    eval_sets: dict[str, Path] = field(default_factory=dict)  # {"test": ..., "val": ...}
    registered_name: str | None = None  # config 里 register_model=True 时非空


def _has_adapter(out_dir: Path) -> bool:
    """HF/PEFT adapter 的标志文件：adapter_config.json 或 adapter_model.* 权重。"""
    return (out_dir / "adapter_config.json").exists() or any(out_dir.glob("adapter_model*"))


def summarize_run_artifacts(config_path: Path, project_root: Path) -> RunArtifacts:
    """读取 run 的 config YAML，定位训练产物。配置缺失/损坏时返回安全空值。"""
    try:
        cfg = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except OSError:
        cfg = {}
    except yaml.YAMLError:
        cfg = {}
    if not isinstance(cfg, dict):
        cfg = {}

    training = cfg.get("training", {}) or {}
    data = cfg.get("data", {}) or {}
    logging_cfg = cfg.get("logging", {}) or {}

    out_raw = str(training.get("output_dir", "") or "")
    output_dir: Path | None = None
    if out_raw:
        p = Path(out_raw)
        output_dir = p if p.is_absolute() else (project_root / p)
        output_dir = output_dir.resolve()

    dataset_name = str(data.get("dataset_name", "") or "")

    eval_sets: dict[str, Path] = {}
    if looks_like_local_path(dataset_name):
        ds_path = Path(dataset_name)
        if not ds_path.is_absolute():
            ds_path = (project_root / ds_path).resolve()
        for split in ("test", "val"):
            sibling = ds_path.parent / f"{split}.json"
            if sibling.exists():
                eval_sets[split] = sibling

    registered_name: str | None = None
    if logging_cfg.get("register_model"):
        registered_name = str(logging_cfg.get("registry_model_name", "") or "") or None

    return RunArtifacts(
        output_dir=output_dir,
        has_adapter=bool(output_dir and output_dir.is_dir() and _has_adapter(output_dir)),
        dataset_name=dataset_name,
        model_name=str((cfg.get("model", {}) or {}).get("name", "") or ""),
        eval_sets=eval_sets,
        registered_name=registered_name,
    )
