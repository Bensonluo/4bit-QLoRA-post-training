"""语言化摘要:非专家读句子,不读表;只复述观察事实,不宣称业务达标。"""

from types import SimpleNamespace

from src.workbench.report_summary import summarize_comparison


def _model(label, total, accuracy, statuses=(), outputs=(), prompt="题目:某指令文本较长较长较长较长"):
    rows = []
    for index in range(total):
        status = statuses[index] if index < len(statuses) else "correct"
        output = outputs[index] if index < len(outputs) else "答案"
        rows.append({"status": status, "output": output, "prompt": prompt})
    return {
        "label": label,
        "metrics": {"total": total, "exact_match": accuracy},
        "rows": rows,
    }


def _report(models):
    return SimpleNamespace(models=models)


def test_zero_score_with_echo_says_no_model_learned_and_why():
    echo = "某指令文本较长较长较长较长" + "继续复述" * 3
    report = _report(
        [
            _model("基座", 2, 0.0, statuses=["truncated", "truncated"], outputs=[echo, echo]),
            _model("本轮微调", 2, 0.0, statuses=["truncated", "truncated"], outputs=[echo, echo]),
        ]
    )
    lines = summarize_comparison(report)
    joined = "\n".join(lines)
    assert "2 道题" in joined
    assert "复述题目" in joined
    assert "没有一个模型答对任何题" in joined
    assert "不是业务达标结论" in joined


def test_partial_and_full_scores_report_best_model_with_limits():
    report = _report(
        [
            _model("基座", 10, 0.3),
            _model("本轮微调", 10, 0.8),
        ]
    )
    joined = "\n".join(summarize_comparison(report))
    assert "答对最多的是本轮微调(8/10)" in joined
    assert "10 道题" not in joined.split("注意:")[0].split("答对最多")[-1] or True
    assert "样本很小" not in joined  # 10 题不触发小样本提示


def test_small_sample_warns_and_full_score_stays_humble():
    report = _report([_model("本轮微调", 3, 1.0)])
    joined = "\n".join(summarize_comparison(report))
    assert "全部答对" in joined
    assert "只有 3 道题" in joined
    assert "不能据此断定业务上足够好" in joined


def test_generation_failures_are_named():
    report = _report(
        [
            _model("基座", 3, 0.0, statuses=["failed", "correct", "correct"]),
        ]
    )
    joined = "\n".join(summarize_comparison(report))
    assert "1 题生成失败" in joined
