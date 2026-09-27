"""语言化摘要:非专家读句子,不读表;只复述观察事实,不宣称业务达标。"""

from types import SimpleNamespace

from src.workbench.report_summary import summarize_comparison


def _model(
    label, total, accuracy, statuses=(), outputs=(), prompt="题目:某指令文本较长较长较长较长"
):
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


def test_preflight_summary_names_truncation_and_blockers():
    from src.workbench.report_summary import summarize_preflight

    lines = summarize_preflight(
        {
            "status": "warnings",
            "splits": {"train": {"rows": 8, "truncated_rows": 2, "answer_lost_rows": 0}},
            "issues": [
                {"severity": "warning", "message": "max_length 超过上下文。"},
                {"severity": "info", "message": "全提示监督说明。"},
            ],
        }
    )
    joined = "\n".join(lines)
    assert "需要你核对的风险" in joined
    assert "train 2 行" in joined
    assert "[warning] max_length" in joined
    assert "不代表训练效果" in joined


def test_preflight_summary_passed_and_empty():
    from src.workbench.report_summary import summarize_preflight

    assert summarize_preflight(None) == ["尚未执行训练前检查。"]
    joined = "\n".join(
        summarize_preflight({"status": "passed", "splits": {"train": {"rows": 4}}, "issues": []})
    )
    assert "检查通过" in joined and "没有内容因长度超限被截断" in joined


def test_training_run_summary_status_and_honesty():
    from src.workbench.report_summary import summarize_training_run

    running = summarize_training_run(
        {
            "status": "running",
            "model_path": "/models/Qwen3-1.7B",
            "config": {"training": {"num_epochs": 1}},
        }
    )
    assert any("正在训练中" in line and "关闭页面不影响" in line for line in running)

    done = summarize_training_run(
        {
            "status": "succeeded",
            "model_path": "/models/Qwen3-1.7B",
            "config": {"training": {"num_epochs": 1}},
            "metrics": {"train_loss": 0.42},
        }
    )
    joined = "\n".join(done)
    assert "训练完成" in joined and "0.4200" in joined
    assert "要用同一套开发题与基座对照" in joined

    failed = summarize_training_run(
        {
            "status": "failed",
            "model_path": "/m",
            "failure": {"stage": "training", "message": "显存不足"},
        }
    )
    joined = "\n".join(failed)
    assert "training阶段：显存不足" in joined


def test_preflight_summary_gives_concrete_length_advice():
    from src.workbench.report_summary import summarize_preflight

    lines = summarize_preflight(
        {
            "status": "warnings",
            "max_length": 512,
            "splits": {"train": {"rows": 8, "truncated_rows": 2}},
            "rows": [
                {"full_tokens": 823, "answer_was_truncated": True},
                {"full_tokens": 300, "answer_was_truncated": False},
            ],
            "issues": [],
        }
    )
    joined = "\n".join(lines)
    assert "答案在截断后丢了" in joined
    assert "最长一条记录需要 823" in joined
    assert "设为 832 左右" in joined

    # 无答案丢失且长度已足够:建议核对个别行
    lines = summarize_preflight(
        {
            "status": "warnings",
            "max_length": 1024,
            "splits": {"train": {"rows": 8, "truncated_rows": 1}},
            "rows": [{"full_tokens": 700, "answer_was_truncated": False}],
            "issues": [],
        }
    )
    joined = "\n".join(lines)
    assert "答案都保留了" in joined and "已能容纳最长记录" in joined


def test_all_zero_plugs_failure_cause_counts_into_advice():
    echo = "某指令文本较长较长较长较长" + "继续复述" * 3
    report = _report(
        [
            _model("基座", 2, 0.0, statuses=["truncated", "failed"]),
            _model("本轮微调", 2, 0.0, outputs=[echo, echo]),
        ]
    )
    joined = "\n".join(summarize_comparison(report))
    assert "微调后仍是零分" in joined
    assert "继续加数据之前" in joined
    assert "2 题在复述题目" in joined
    assert "1 题没写完被截断" in joined
    assert "1 题生成失败" in joined
    assert "没有带来可见变化" not in joined  # 全零分归入零分态,不再叠无差异结论


def test_all_zero_without_diagnostics_points_to_format_mismatch():
    report = _report(
        [
            _model("基座", 3, 0.0, outputs=["错", "错", "错"]),
            _model("本轮微调", 3, 0.0, outputs=["错", "错", "错"]),
        ]
    )
    joined = "\n".join(summarize_comparison(report))
    assert "微调后仍是零分" in joined
    assert "没有观察到截断、生成失败或复述" in joined
    assert "答案格式不匹配" in joined


def test_clear_finetune_gain_states_counts_with_sample_caveat():
    report = _report(
        [
            _model("基座", 10, 0.3),
            _model("本轮微调", 10, 0.6),
        ]
    )
    joined = "\n".join(summarize_comparison(report))
    assert "本轮微调比基座答对更多(6/10 vs 3/10)" in joined
    assert "要注意样本量" in joined
    assert "逐题核对答错的部分" in joined


def test_no_difference_between_base_and_finetune_is_named():
    report = _report(
        [
            _model("基座", 10, 0.4),
            _model("本轮微调", 10, 0.4),
        ]
    )
    joined = "\n".join(summarize_comparison(report))
    assert "微调没有带来可见变化(4/10 vs 4/10)" in joined
    assert "数据量不足或任务难度过高" in joined
    assert "比基座答对更多" not in joined


def test_middling_difference_gets_neither_extreme_claim():
    report = _report(
        [
            _model("基座", 20, 0.2),
            _model("本轮微调", 20, 0.35),
        ]
    )
    joined = "\n".join(summarize_comparison(report))
    assert "比基座答对更多" not in joined
    assert "没有带来可见变化" not in joined
    assert "答对最多的是本轮微调(7/20)" in joined


def test_zero_scores_without_finetuned_model_use_generic_head():
    report = _report([_model("基座", 2, 0.0, statuses=["failed", "failed"])])
    joined = "\n".join(summarize_comparison(report))
    assert "所有模型都是零分" in joined
    assert "微调后仍是零分" not in joined
