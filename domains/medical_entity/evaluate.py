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
    RealFinetunedModel,
    RealLLMAPI,
)
from domains.medical_entity.eval.report import generate_executive_summary, save_results
from domains.medical_entity.eval.runner import load_test_data, run_evaluation
from src.utils.logging import console


def main():
    import argparse

    parser = argparse.ArgumentParser(description="医疗实体匹配评测")
    parser.add_argument("--model-path", type=str, default=None, help="LoRA adapter 路径（真实推理）")
    parser.add_argument("--base-model", type=str, default=None, help="base model 名称")
    parser.add_argument("--api-model", type=str, default=None, help="云端 LLM API 模型名（如 glm-4-flash）")
    parser.add_argument("--api-base-url", type=str, default=None, help="云端 API base URL（默认智谱）")
    parser.add_argument("--local-model", type=str, default=None, help="本地模型名（LM Studio / Ollama）")
    parser.add_argument("--local-base-url", type=str, default="http://127.0.0.1:1234/v1/", help="本地模型 API 地址（默认 LM Studio）")
    parser.add_argument("--test-file", type=str, default=None, help="指定测试集文件（默认 test_raw.json）")
    parser.add_argument("--max-samples", type=int, default=500, help="非 baseline 模型测试条数（默认500）")
    parser.add_argument("--concurrency", type=int, default=3, help="API 并发数（默认3）")
    args = parser.parse_args()

    test_data = load_test_data(path=args.test_file)
    console.print(f"[cyan]加载测试数据: {len(test_data)} 条[/cyan]")

    # 非 baseline 模型限量
    sampled_data = test_data
    if len(test_data) > args.max_samples:
        import random
        random.seed(42)
        sampled_data = random.sample(test_data, args.max_samples)
    console.print(f"[cyan]Baseline 全量: {len(test_data)} 条 | 其他模型: {len(sampled_data)} 条[/cyan]")

    # 加载训练集药品编码，用于 seen/unseen 分析（按编码而非 query 字符串）
    train_codes = set()
    train_file = Path("domains/medical_entity/data/train/train.json")
    if train_file.exists():
        import json as _json
        with open(train_file) as f:
            for item in _json.load(f):
                out = item.get("output", "")
                try:
                    parsed = _json.loads(out)
                    code = parsed.get("code", "")
                    if code:
                        train_codes.add(code)
                except (_json.JSONDecodeError, ValueError):
                    pass
    test_codes = set(s["code"] for s in test_data)
    truly_unseen = test_codes - train_codes
    console.print(f"[cyan]训练药品: {len(train_codes)} 种 | 测试药品: {len(test_codes)} 种 | 真正 unseen: {len(truly_unseen)} 种[/cyan]\n")

    # (model, test_data) 列表
    tasks = []

    tasks.append((CombinedHeuristicBaseline(), test_data))

    if args.api_model:
        tasks.append((RealLLMAPI(
            model_name=args.api_model,
            base_url=args.api_base_url,
        ), sampled_data))

    if args.local_model:
        tasks.append((RealLLMAPI(
            model_name=args.local_model,
            base_url=args.local_base_url,
        ), sampled_data))

    if args.model_path:
        tasks.append((RealFinetunedModel(
            model_path=args.model_path,
            base_model=args.base_model,
        ), sampled_data))

    reports = []

    # Baseline 秒级，先跑
    for model, data in tasks:
        if isinstance(model, CombinedHeuristicBaseline):
            console.print(f"[yellow]评测: {model.name} ({len(data)} 条)[/yellow]")
            report = run_evaluation(model, data, train_codes=train_codes)
            reports.append(report)
            seen_acc = report.accuracy(seen=True)
            unseen_acc = report.accuracy(seen=False)
            console.print(
                f"  准确率: {report.accuracy():.1%} | "
                f"seen={seen_acc:.1%} unseen={unseen_acc:.1%} | "
                f"easy={report.accuracy(difficulty='easy'):.1%} "
                f"medium={report.accuracy(difficulty='medium'):.1%} "
                f"hard={report.accuracy(difficulty='hard'):.1%}\n"
            )

    # 并行跑慢模型
    slow_tasks = [(m, d) for m, d in tasks if not isinstance(m, CombinedHeuristicBaseline)]
    if slow_tasks:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        console.print(f"[bold cyan]并行评测 {len(slow_tasks)} 个模型 (API 并发={args.concurrency})...[/bold cyan]\n")
        with ThreadPoolExecutor(max_workers=len(slow_tasks)) as pool:
            futures = {}
            for model, data in slow_tasks:
                is_local = isinstance(model, RealLLMAPI) and model._base_url and ("localhost" in (model._base_url or "") or "127.0.0.1" in (model._base_url or ""))
                is_cloud = isinstance(model, RealLLMAPI) and not is_local
                if is_cloud:
                    conc = 1
                    data = data[:500]
                elif is_local:
                    conc = args.concurrency
                else:
                    conc = 1
                console.print(f"[yellow]启动: {model.name} ({len(data)} 条, 并发={conc})[/yellow]")
                future = pool.submit(run_evaluation, model, data, concurrency=conc, train_codes=train_codes)
                futures[future] = model

            for future in as_completed(futures):
                model = futures[future]
                report = future.result()
                reports.append(report)
                console.print(
                    f"[green]✓ {model.name} 完成[/green] | "
                    f"准确率: {report.accuracy():.1%} | "
                    f"seen={report.accuracy(seen=True):.1%} unseen={report.accuracy(seen=False):.1%} | "
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
