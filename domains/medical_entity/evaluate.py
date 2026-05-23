#!/usr/bin/env python3
"""医疗实体匹配模型评测脚本。

用法:
    cd 4bit-QLoRA-post-training
    python -m domains.medical_entity.evaluate
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from domains.medical_entity.eval.models import (
    CombinedHeuristicBaseline,
    EditDistanceBaseline,
    FinetunedModelSimulator,
    JaccardBaseline,
    LLMAPISimulator,
    RandomBaseline,
)
from domains.medical_entity.eval.report import generate_executive_summary, save_results
from domains.medical_entity.eval.runner import load_test_data, run_evaluation
from src.utils.logging import console


def main():
    test_data = load_test_data()
    console.print(f"[cyan]加载测试数据: {len(test_data)} 条[/cyan]\n")

    models = [
        RandomBaseline(),
        EditDistanceBaseline(),
        JaccardBaseline(),
        CombinedHeuristicBaseline(),
        LLMAPISimulator(),
        FinetunedModelSimulator(),
    ]

    reports = []
    for model in models:
        console.print(f"[yellow]评测: {model.name}[/yellow]")
        report = run_evaluation(model, test_data)
        reports.append(report)
        console.print(
            f"  准确率: {report.accuracy():.1%} | "
            f"easy={report.accuracy(difficulty='easy'):.1%} "
            f"medium={report.accuracy(difficulty='medium'):.1%} "
            f"hard={report.accuracy(difficulty='hard'):.1%}\n"
        )

    save_results(reports)

    console.print("\n[bold green]执行摘要:[/bold green]")
    exec_summary = generate_executive_summary(reports)
    console.print(exec_summary[:500])


if __name__ == "__main__":
    main()
