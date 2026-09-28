"""training_progress 单元：逐条 loss 序列的抽取、读取与趋势人话。

环节⑤可视化：序列在训练脚本侧保留（extract）、在页面与 train-status 侧
翻译成趋势人话（loss_trend_lines 单一来源）；三态判定只对比首末段均值，
边界句与截断/坍缩提示同一纪律（观察事实，不认定原因）。
"""

import json

from src.workbench.training_progress import (
    LOSS_HISTORY_FILENAME,
    extract_loss_history,
    load_loss_history,
    loss_trend_lines,
)


def test_extract_keeps_step_loss_and_lr_drops_eval_and_summary_rows():
    history = extract_loss_history(
        [
            {"loss": 2.5, "step": 10, "learning_rate": 0.0002, "epoch": 0.1},
            {"eval_loss": 2.0, "step": 10, "eval_runtime": 1.0},
            {"train_runtime": 30.0, "train_loss": 2.1},
            {"loss": 1.8, "step": 20, "learning_rate": 0.0001},
            {"loss": 1.9, "step": 15},
        ]
    )
    assert history == [
        {"step": 10.0, "loss": 2.5, "learning_rate": 0.0002},
        {"step": 15.0, "loss": 1.9},
        {"step": 20.0, "loss": 1.8, "learning_rate": 0.0001},
    ]


def test_load_reads_step_loss_and_survives_missing_or_broken_files(tmp_path):
    out = tmp_path / "adapter"
    out.mkdir()
    assert load_loss_history(out) == []
    assert load_loss_history(None) == []
    (out / LOSS_HISTORY_FILENAME).write_text(
        json.dumps([{"step": 1, "loss": 2.0, "learning_rate": 1e-4}, "junk", {"loss": 3.0}]),
        encoding="utf-8",
    )
    assert load_loss_history(out) == [{"step": 1.0, "loss": 2.0}]
    (out / LOSS_HISTORY_FILENAME).write_text("not json", encoding="utf-8")
    assert load_loss_history(out) == []


def test_trend_declining_series_gives_first_last_and_boundary():
    history = [{"step": float(s), "loss": 2.0 - 0.15 * s} for s in range(10)]
    lines = loss_trend_lines(history)
    assert lines[0] == (
        "loss 从第 0 步的 2.0000 走到第 9 步的 0.6500（共 10 个记录点），整体在下降。"
    )
    assert lines[1] == (
        "loss 下降只说明模型在逐步记住训练题，不代表业务效果；效果要看同一套开发题上的对照报告。"
    )


def test_trend_flat_and_rising_series_name_the_observation():
    flat = loss_trend_lines(
        [{"step": float(s), "loss": 1.0 + (0.01 if s % 2 else 0.0)} for s in range(8)]
    )
    assert "整体基本持平" in flat[0]
    assert "不代表训练失败" in flat[0]
    rising = loss_trend_lines([{"step": float(s), "loss": 1.0 + 0.2 * s} for s in range(8)])
    assert "末段反而更高" in rising[0]
    assert "需要核查" in rising[0]


def test_trend_degenerate_series_says_cannot_read():
    assert loss_trend_lines([]) == ["这次训练没有留下逐条 loss 记录，看不出训练过程的变化。"]
    single = loss_trend_lines([{"step": 7.0, "loss": 1.25}])
    assert single == ["逐条 loss 记录只有 1 个点（第 7 步，loss 1.2500），看不出趋势。"]


def test_trend_in_progress_series_defers_verdict():
    """训练中(in_progress=True)只给跨度与等待句:半程数据不足以支持
    整场三态判定;默认(训练结束)行为不变。"""
    history = [{"step": float(s), "loss": 2.0 - 0.1 * s} for s in range(6)]
    lines = loss_trend_lines(history, in_progress=True)
    assert lines[0] == (
        "loss 从第 0 步的 2.0000 走到第 5 步的 1.5000（共 6 个记录点），训练进行中，"
        "趋势判定等训练完成后再看。"
    )
    assert lines[1] == (
        "loss 下降只说明模型在逐步记住训练题，不代表业务效果；效果要看同一套开发题上的对照报告。"
    )
    assert "整体在下降" not in "".join(lines)
    # 默认 in_progress=False:同一序列走完整三态判定。
    assert "整体在下降" in loss_trend_lines(history)[0]


def test_live_loss_writer_appends_points_and_survives_unwritable_path(tmp_path):
    """实时写入侧:逐次 on_log 追加点并整体重写文件(乱序也按 step 排好);
    eval 行不写;写失败不抛出——呈现增强不能中断训练。"""
    from src.workbench.live_loss import LiveLossWriter

    writer = LiveLossWriter(tmp_path / "run-out")
    writer.on_log(None, None, None, logs={"eval_loss": 1.2, "step": 3})
    assert writer._path.exists() is False
    writer.on_log(None, None, None, logs={"loss": 1.8, "step": 4, "learning_rate": 1e-4})
    writer.on_log(None, None, None, logs={"loss": 2.2, "step": 2})
    points = json.loads(writer._path.read_text(encoding="utf-8"))
    assert points == [
        {"step": 2.0, "loss": 2.2},
        {"step": 4.0, "loss": 1.8, "learning_rate": 1e-4},
    ]
    # 目录不可写(路径父级是个文件):OSError 被吞掉,回调不抛出。
    blocker = tmp_path / "blocker"
    blocker.write_text("a file, not a dir", encoding="utf-8")
    stubborn = LiveLossWriter(blocker / "under")
    stubborn.on_log(None, None, None, logs={"loss": 1.0, "step": 1})
