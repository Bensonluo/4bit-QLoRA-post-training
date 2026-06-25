"""
生成评测对比报告：表格 + 图表 + JSON 详细结果。
"""

import json
from datetime import datetime
from pathlib import Path

from domains.medical_entity.eval.runner import EvalReport

DOMAIN_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = DOMAIN_ROOT / "data" / "results"


def generate_comparison_table(reports: list[EvalReport]) -> str:
    """生成 Markdown 对比表格"""
    lines = []
    lines.append("# 医疗实体匹配模型对比评测报告")
    lines.append(f"\n生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # 总览表
    lines.append("## 1. 总览\n")
    lines.append("| 指标 | " + " | ".join(r.model_name for r in reports) + " |")
    lines.append("|" + "|".join(["---"] * (len(reports) + 1)) + "|")

    metrics = [
        ("总样本数", lambda r: str(r.total)),
        ("正确数", lambda r: str(r.correct_count)),
        ("**Overall Accuracy**", lambda r: f"{r.accuracy():.1%}"),
        ("**MRR**", lambda r: f"{r.mrr():.4f}"),
        ("平均置信度", lambda r: f"{r.avg_confidence():.2f}"),
        ("平均延迟 (ms)", lambda r: f"{r.avg_latency():.0f}"),
        ("吞吐量 (条/秒)", lambda r: f"{r.throughput():.1f}"),
        ("总耗时 (秒)", lambda r: f"{r.total_time_ms/1000:.1f}"),
    ]
    for label, fn in metrics:
        row = f"| {label} | " + " | ".join(fn(r) for r in reports) + " |"
        lines.append(row)

    # 按难度分层
    lines.append("\n## 2. 按难度分层准确率\n")
    lines.append("| 难度 | " + " | ".join(r.model_name for r in reports) + " |")
    lines.append("|" + "|".join(["---"] * (len(reports) + 1)) + "|")
    for diff in ["easy", "medium", "hard"]:
        row = f"| {diff} | " + " | ".join(
            f"{r.accuracy(difficulty=diff):.1%}" for r in reports
        ) + " |"
        lines.append(row)

    # 按实体类型
    lines.append("\n## 3. 按实体类型准确率\n")
    lines.append("| 类型 | " + " | ".join(r.model_name for r in reports) + " |")
    lines.append("|" + "|".join(["---"] * (len(reports) + 1)) + "|")
    for etype in ["drug", "hospital"]:
        row = f"| {etype} | " + " | ".join(
            f"{r.accuracy(entity_type=etype):.1%}" for r in reports
        ) + " |"
        lines.append(row)

    # 置信度校准
    lines.append("\n## 4. 置信度校准\n")
    lines.append("> 理想状态：模型说 90% 置信度时，实际准确率也应约 90%\n")
    for r in reports:
        lines.append(f"\n### {r.model_name}")
        cal = r.confidence_calibration()
        lines.append("| 置信度区间 | 样本数 | 实际准确率 |")
        lines.append("|---|---|---|")
        for bin_name, info in cal.items():
            lines.append(f"| {bin_name} | {info['count']} | {info['accuracy']:.1%} |")

    # 错误分析
    lines.append("\n## 5. 错误样本分析\n")
    for r in reports:
        errors = [res for res in r.results if not res.correct]
        lines.append(f"\n### {r.model_name} ({len(errors)} 个错误)")
        for err in errors[:10]:  # 只展示前10个
            lines.append(
                f"- `{err.query}` → 预测: `{err.predicted_name}` | "
                f"正确: `{err.ground_truth}` | 难度: {err.difficulty} | "
                f"置信度: {err.confidence}"
            )

    return "\n".join(lines)


def generate_executive_summary(reports: list[EvalReport]) -> str:
    """生成给领导看的一页纸摘要"""
    if len(reports) < 2:
        return "至少需要两个模型才能对比"

    # 找到精调模型和最佳 baseline
    ours = reports[-1]  # 最后一个假设是精调模型
    theirs = max(
        [r for r in reports if r != ours], key=lambda r: r.accuracy()
    )

    acc_diff = (ours.accuracy() - theirs.accuracy()) * 100
    hard_diff = (ours.accuracy(difficulty="hard") - theirs.accuracy(difficulty="hard")) * 100
    latency_improvement = theirs.avg_latency() / max(ours.avg_latency(), 0.01)

    lines = [
        "# 执行摘要：精调模型 vs 现有方案\n",
        "## 核心结论",
        f"- 精调模型 **Overall Accuracy: {ours.accuracy():.1%}** vs {theirs.model_name} **{theirs.accuracy():.1%}**，"
        f"提升 **{acc_diff:+.1f}%**",
        f"- 困难样本（错别字/口语化）准确率提升 **{hard_diff:+.1f}%**",
        f"- 推理速度提升 **{latency_improvement:.0f}x**（{ours.avg_latency():.0f}ms vs {theirs.avg_latency():.0f}ms）",
        "",
        "## 分难度对比\n",
        "| 难度 | 现有方案 | 精调模型 | 提升 |",
        "|---|---|---|---|",
    ]
    for diff in ["easy", "medium", "hard"]:
        t = theirs.accuracy(difficulty=diff)
        o = ours.accuracy(difficulty=diff)
        lines.append(f"| {diff} | {t:.1%} | {o:.1%} | {((o-t)*100):+.1f}% |")

    lines.extend([
        "",
        "## 成本估算\n",
        "| 指标 | 现有方案 | 精调模型 |",
        "|---|---|---|",
        f"| 单条延迟 | {theirs.avg_latency():.0f}ms | {ours.avg_latency():.0f}ms |",
        f"| 吞吐量 | {theirs.throughput():.0f}条/秒 | {ours.throughput():.0f}条/秒 |",
        f"| 日处理100万条 | 需要 {1000000/theirs.throughput()/3600:.1f}小时 | 需要 {1000000/ours.throughput()/3600:.1f}小时 |",
        "| 部署方式 | API依赖 | 本地GPU |",
        "| 数据安全 | 出域 | 不出域 |",
    ])

    return "\n".join(lines)


def save_results(reports: list[EvalReport], output_dir: str = None):
    """保存完整评测结果"""
    output_dir = Path(output_dir) if output_dir else RESULTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    # 详细 JSON
    all_summaries = []
    for r in reports:
        summary = r.summary()
        summary["per_sample"] = [
            {
                "query": res.query,
                "ground_truth": res.ground_truth,
                "ground_truth_code": res.ground_truth_code,
                "predicted_name": res.predicted_name,
                "predicted_code": res.predicted_code,
                "confidence": res.confidence,
                "difficulty": res.difficulty,
                "entity_type": res.entity_type,
                "correct": res.correct,
                "latency_ms": round(res.latency_ms, 1),
                "error": res.error,
            }
            for res in r.results
        ]
        all_summaries.append(summary)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    with open(output_dir / f"eval_detail_{timestamp}.json", "w") as f:
        json.dump(all_summaries, f, ensure_ascii=False, indent=2)

    # Markdown 报告
    table_report = generate_comparison_table(reports)
    with open(output_dir / f"comparison_{timestamp}.md", "w") as f:
        f.write(table_report)

    # 执行摘要
    exec_summary = generate_executive_summary(reports)
    with open(output_dir / f"executive_summary_{timestamp}.md", "w") as f:
        f.write(exec_summary)

    print(f"评测结果已保存至 {output_dir}/")
    print(f"  详细数据: eval_detail_{timestamp}.json")
    print(f"  对比报告: comparison_{timestamp}.md")
    print(f"  执行摘要: executive_summary_{timestamp}.md")

    # Log to MLflow if available
    try:
        from src.tracking.eval_logger import log_eval_to_mlflow
        log_eval_to_mlflow(output_dir / f"eval_detail_{timestamp}.json")
    except ImportError:
        pass
