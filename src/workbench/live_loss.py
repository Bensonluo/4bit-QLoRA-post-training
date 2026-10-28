"""实时 loss 写入：训练中把逐条 loss 序列增量重写到产物目录。

环节⑤「实时曲线」的写入侧。worker 此前只在训练结束后一次性写
workbench_loss_history.json，训练中页面只能看日志文本；本回调在每次
on_log 追加一个点并整体重写该文件（文件很小，整体重写最简单也最不易
坏）。训练结束时 worker 仍会用完整 log_history 重写同一文件作为权威
序列（extract_loss_history），中途的增量写入只为「实时」。

本模块在顶层导入 transformers（训练侧模块，与 src/training/callbacks.py
同类）；保持 training_progress.py 纯 stdlib，供页面与 CLI 无负担复用。
写失败不抛出——实时曲线是呈现增强，不能让一次磁盘小故障中断训练。
"""

from __future__ import annotations

import json
from contextlib import suppress
from pathlib import Path
from typing import Any
from uuid import uuid4

from transformers.trainer_callback import TrainerCallback

from src.workbench.training_progress import LOSS_HISTORY_FILENAME


class LiveLossWriter(TrainerCallback):
    """逐次日志点把 loss 序列重写到产物目录，供训练中的页面/CLI 读取。"""

    def __init__(self, output_dir: str | Path) -> None:
        self._path = Path(output_dir) / LOSS_HISTORY_FILENAME
        self._points: list[dict[str, float]] = []

    def on_log(
        self, args: Any, state: Any, control: Any, logs: dict[str, Any] | None = None, **kwargs: Any
    ) -> None:
        if not logs or "loss" not in logs:
            return
        # Trainer.log adds step to log_history, but passes the original logs to
        # callbacks. The callback's state is therefore the authoritative step.
        step = getattr(state, "global_step", None)
        if step is None:
            step = logs.get("step")
        if step is None:
            return
        try:
            point: dict[str, float] = {"step": float(step), "loss": float(logs["loss"])}
        except (TypeError, ValueError, OverflowError):
            return
        if logs.get("learning_rate") is not None:
            with suppress(TypeError, ValueError, OverflowError):
                point["learning_rate"] = float(logs["learning_rate"])
        self._points.append(point)
        self._points.sort(key=lambda item: item["step"])
        temporary = self._path.with_name(f".{self._path.name}.{uuid4().hex}.tmp")
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            temporary.write_text(
                json.dumps(self._points, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            temporary.replace(self._path)
        except OSError:
            pass
        finally:
            with suppress(OSError):
                temporary.unlink(missing_ok=True)
