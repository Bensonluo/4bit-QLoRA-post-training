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
from pathlib import Path

from transformers.trainer_callback import TrainerCallback

from src.workbench.training_progress import LOSS_HISTORY_FILENAME


class LiveLossWriter(TrainerCallback):
    """逐次日志点把 loss 序列重写到产物目录，供训练中的页面/CLI 读取。"""

    def __init__(self, output_dir: str | Path) -> None:
        self._path = Path(output_dir) / LOSS_HISTORY_FILENAME
        self._points: list[dict[str, float]] = []

    def on_log(self, args, state, control, logs=None, **kwargs) -> None:
        if not logs or "loss" not in logs or "step" not in logs:
            return
        point: dict[str, float] = {"step": float(logs["step"]), "loss": float(logs["loss"])}
        if logs.get("learning_rate") is not None:
            point["learning_rate"] = float(logs["learning_rate"])
        self._points.append(point)
        self._points.sort(key=lambda item: item["step"])
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.write_text(
                json.dumps(self._points, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except OSError:
            pass
