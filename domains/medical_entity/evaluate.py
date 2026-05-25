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
    RealFinetunedModel,
)
from domains.medical_entity.eval.report import generate_executive_summary, save_results
from domains.medical_entity.eval.runner import load_test_data, run_evaluation
from src.utils.logging import console


def main():
    import argparse

    parser = argparse.ArgumentParser(description="医疗实体匹配评测")
    parser.add_argument("--model-path", type=str, default=None, help="LoRA adapter 路径（真实推理）")
    parser.add_argument("--base-model", type=str, default=None, help="base model 名称")
    parser.add_argument("--skip-real", action="store_true", help="跳过真实模型推理")
    args = parser.parse_args()

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

    if args.model_path and not args.skip_real:
        models.append(RealFinetunedModel(
            model_path=args.model_path,
            base_model=args.base_model,
        ))

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
