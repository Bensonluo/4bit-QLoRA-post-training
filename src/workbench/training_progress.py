"""训练过程呈现：把逐条 loss 记录变成非专家看得懂的趋势人话。

北极星环节⑤「可视化」点名「实时曲线、对照表、逐条坏例、诊断提示」。
此前训练脚本把 log_history 压平成单值 dict——曲线在落盘时就被销毁，
页面只剩原始 JSON 倾倒。本模块补两条：
1. extract_loss_history 在训练脚本侧保留逐 step 序列（落盘为
   workbench_loss_history.json，与 flat metrics 并存、互不替代）；
2. loss_trend_lines 把序列翻译成趋势人话（单一来源，页面与 CLI 同源）。

训练中该文件由 worker 的 LiveLossWriter 逐次日志点增量重写（环节⑤
「实时曲线」）；训练结束再用完整 log_history 重写为权威序列。

趋势判定只对比首段/末段均值（各约 1/4 记录点），只陈述观察事实，
不认定原因、不预言业务效果——与截断/坍缩提示同一纪律。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

LOSS_HISTORY_FILENAME = "workbench_loss_history.json"


def extract_loss_history(log_history: list[dict[str, Any]]) -> list[dict[str, float]]:
    """从 HF Trainer 的 log_history 抽出训练 loss 序列（丢弃 eval_* 与汇总行）。

    eval 行的键是 eval_loss，汇总行（train_runtime 等）没有 step 级 loss；
    只保留 step/loss/learning_rate 三个键，按 step 升序返回。
    """
    series: list[dict[str, float]] = []
    for entry in log_history:
        if "loss" not in entry or "step" not in entry:
            continue
        point: dict[str, float] = {"step": float(entry["step"]), "loss": float(entry["loss"])}
        if entry.get("learning_rate") is not None:
            point["learning_rate"] = float(entry["learning_rate"])
        series.append(point)
    series.sort(key=lambda item: item["step"])
    return series


def load_loss_history(output_dir: str | Path | None) -> list[dict[str, float]]:
    """读取产物目录里的逐条 loss 记录；缺失或损坏时返回空（旧产物如实没有）。

    只回读 step/loss 两个键并逐条过滤坏行——损坏的旧文件不让训练信息区
    整体崩掉，曲线与趋势各自把「点数不足」如实说出口。
    """
    if not output_dir:
        return []
    path = Path(output_dir) / LOSS_HISTORY_FILENAME
    if not path.is_file():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    if not isinstance(data, list):
        return []
    return [
        {"step": float(item["step"]), "loss": float(item["loss"])}
        for item in data
        if isinstance(item, dict) and "step" in item and "loss" in item
    ]


def _segment_mean(points: list[dict[str, float]]) -> float:
    return sum(item["loss"] for item in points) / len(points)


def loss_trend_lines(history: list[dict[str, float]], *, in_progress: bool = False) -> list[str]:
    """把逐条 loss 序列翻译成趋势人话：先给首末与点数，再给三态判定。

    三态 = 在下降 / 基本持平 / 末段反而更高，判定依据是首段与末段均值的
    对比（±5% 以内算持平）；每条序列都以「loss 下降不代表业务效果」收尾。
    in_progress=True（训练进行中）只给首末 span，不给三态判定——半程数据
    不足以支持「整体在下降」这类整场结论，等训练完成后再看。
    """
    if not history:
        return ["这次训练没有留下逐条 loss 记录，看不出训练过程的变化。"]
    if len(history) == 1:
        point = history[0]
        return [
            f"逐条 loss 记录只有 1 个点（第 {point['step']:.0f} 步，loss "
            f"{point['loss']:.4f}），看不出趋势。"
        ]
    first, last = history[0], history[-1]
    size = max(1, len(history) // 4)
    head, tail = _segment_mean(history[:size]), _segment_mean(history[-size:])
    span = (
        f"loss 从第 {first['step']:.0f} 步的 {first['loss']:.4f} "
        f"走到第 {last['step']:.0f} 步的 {last['loss']:.4f}（共 {len(history)} 个记录点），"
    )
    if in_progress:
        return [
            span + "训练进行中，趋势判定等训练完成后再看。",
            "loss 下降只说明模型在逐步记住训练题，不代表业务效果；效果要看同一套开发题上的对照报告。",
        ]
    if tail < head * 0.95:
        verdict = "整体在下降。"
    elif tail > head * 1.05:
        verdict = (
            f"末段反而更高（首段均值 {head:.4f} → 末段均值 {tail:.4f}）。中途不降反升"
            "需要核查——常见方向是学习率过大或数据里有异常样本，先看同题对照结果再定。"
        )
    else:
        verdict = (
            f"整体基本持平（首段均值 {head:.4f} → 末段均值 {tail:.4f}）。loss 没降下来"
            "不代表训练失败；先核对配置是否按方案执行，再看同题对照结果。"
        )
    return [
        span + verdict,
        "loss 下降只说明模型在逐步记住训练题，不代表业务效果；效果要看同一套开发题上的对照报告。",
    ]
