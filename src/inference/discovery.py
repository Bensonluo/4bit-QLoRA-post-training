"""发现本机可对话的模型：outputs/ 下的 LoRA adapter 与已合并模型。

LlamaBoard 式 Chat 收尾（对齐 LLaMA-Factory WebUI chatter 的 checkpoint
选择器）：训练产物不该只躺在磁盘里。本模块只做文件系统事实收集——
纯标准库、不拉 torch——UI 据此填充「和哪个模型对话」下拉框：

- adapter：``adapter_config.json`` 所在目录，底座名从该配置的
  ``base_model_name_or_path`` 读出（含 run 根目录与 checkpoint-* 子目录）
- merged：``outputs/merged/<name>/`` 含 config.json + 权重文件、
  且无 adapter 标志的独立模型目录
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

_MAX_SCAN = 500  # rglob 上限，防病态目录拖垮页面渲染


@dataclass(frozen=True)
class ChatModelOption:
    """一个可加载对话的本地模型（全部来自文件系统事实，无猜测）。"""

    kind: str  # "adapter" | "merged"
    path: Path
    base_model: str | None  # adapter 的底座名；merged 为 None（自带全部权重）

    @property
    def label(self) -> str:
        icon = "🔧" if self.kind == "adapter" else "📦"
        base = f"（底座 {self.base_model}）" if self.base_model else ""
        return f"{icon} {self.path.name}{base}"


def _read_adapter_base(cfg_path: Path) -> str | None:
    """从 adapter_config.json 读底座名；损坏/缺失字段时返回 None（条目仍列出）。"""
    try:
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(cfg, dict):
        return None
    base = cfg.get("base_model_name_or_path")
    return str(base) if base else None


def looks_merged(d: Path) -> bool:
    """目录是否是一个自包含的已合并模型（config + 权重、无 adapter 标志）。

    Chat 页发现与 Training Lab 的「合并导出」幂等检测共用这一判定。
    """
    if not (d / "config.json").exists():
        return False
    has_weights = any(d.glob("*.safetensors")) or (d / "pytorch_model.bin").exists()
    return has_weights and not (d / "adapter_config.json").exists()


def discover_chat_models(project_root: Path, limit: int = 30) -> list[ChatModelOption]:
    """扫描 outputs/ 找可对话模型，按目录 mtime 新→旧排序，最多 limit 条。"""
    outputs = project_root / "outputs"
    options: list[ChatModelOption] = []
    if not outputs.is_dir():
        return options

    seen: set[Path] = set()
    n_scanned = 0
    for cfg_path in outputs.rglob("adapter_config.json"):
        n_scanned += 1
        if n_scanned > _MAX_SCAN:
            break
        adapter_dir = cfg_path.parent.resolve()
        if adapter_dir in seen:
            continue
        seen.add(adapter_dir)
        options.append(
            ChatModelOption(
                kind="adapter", path=adapter_dir, base_model=_read_adapter_base(cfg_path)
            )
        )

    merged_root = outputs / "merged"
    if merged_root.is_dir():
        for d in sorted(p for p in merged_root.iterdir() if p.is_dir()):
            if looks_merged(d):
                resolved = d.resolve()
                if resolved not in seen:
                    seen.add(resolved)
                    options.append(ChatModelOption(kind="merged", path=resolved, base_model=None))

    options.sort(key=lambda o: o.path.stat().st_mtime, reverse=True)
    return options[:limit]
