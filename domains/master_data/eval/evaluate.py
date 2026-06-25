#!/usr/bin/env python3
"""主数据匹配模型评测脚本。

评测多个 base 模型在机构匹配和产品匹配任务上的表现。

用法:
    # 评测 LM Studio 本地模型
    python -m domains.master_data.eval.evaluate --local-model qwen3.5-9b --task institution --max-samples 200

    # 评测云端 API
    python -m domains.master_data.eval.evaluate --api-model glm-5.1 --task product --max-samples 200

    # 评测 MLX 本地模型 + LoRA adapter
    python -m domains.master_data.eval.evaluate --adapter-path outputs/adapters-gemma-26b --mlx-model-id mlx-community/gemma-4-26b-a4b-it-4bit --task institution --max-samples 400
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

import json
import random
import time
from dataclasses import dataclass, field

from rich.console import Console
from rich.table import Table

console = Console()

DOMAIN_ROOT = Path(__file__).resolve().parent.parent


# ════════════════════════════════════════════
# Data Loading
# ════════════════════════════════════════════
def load_eval_data(task: str) -> list[dict]:
    """Load evaluation data. Each sample has messages with ground truth in assistant."""
    path = DOMAIN_ROOT / "data" / "test" / f"eval_{task}.json"
    with open(path) as f:
        return json.load(f)


def extract_ground_truth(sample: dict) -> list[dict]:
    """Extract ground truth from assistant message."""
    for msg in sample["messages"]:
        if msg["role"] == "assistant":
            return json.loads(msg["content"])
    return []


# ════════════════════════════════════════════
# Model Inference
# ════════════════════════════════════════════
def call_model(messages: list[dict], model_name: str, base_url: str, max_tokens: int = 2048) -> tuple[str, float]:
    """Call model API and return (response_text, latency_ms)."""
    import os

    from openai import OpenAI

    is_local = "localhost" in base_url or "127.0.0.1" in base_url
    is_minimax = "minimax" in base_url.lower()
    if is_local:
        api_key = "ollama"
    elif is_minimax:
        api_key = os.environ.get("MINIMAX_API_KEY", "")
    else:
        api_key = os.environ.get("ZHIPUAI_API_KEY") or os.environ.get("OPENAI_API_KEY", "")

    client = OpenAI(api_key=api_key, base_url=base_url)

    kwargs = {
        "model": model_name,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }

    # GLM thinking models: disable thinking
    if "glm" in model_name.lower() and any(v in model_name.lower() for v in ["4.7", "5"]):
        kwargs["extra_body"] = {"thinking": {"type": "disabled"}}

    # Qwen3 thinking models: disable thinking
    if "qwen3" in model_name.lower():
        kwargs["extra_body"] = {"think": False}

    # MiniMax M2.7: thinking cannot be disabled, but can be split into separate field
    if is_minimax:
        kwargs["extra_body"] = {"reasoning_split": True}

    t0 = time.time()
    response = client.chat.completions.create(**kwargs)
    latency = (time.time() - t0) * 1000

    msg = response.choices[0].message
    text = msg.content or getattr(msg, "reasoning", None) or ""
    return text.strip(), latency


def call_model_mlx(messages: list[dict], model, tokenizer, max_tokens: int = 2048) -> tuple[str, float]:
    """Call MLX local model with adapter and return (response_text, latency_ms)."""
    from mlx_lm import generate
    from mlx_lm.sample_utils import make_sampler

    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    sampler = make_sampler(temp=0.1)  # low temp for eval consistency

    t0 = time.time()
    response = generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=1024,
        sampler=sampler,
        verbose=False,
    )
    latency = (time.time() - t0) * 1000
    return response.strip(), latency


def parse_json_array(text: str) -> list[dict]:
    """Parse JSON array from model response, handling reasoning text before/after."""
    import re
    # Try extracting from code block first
    m = re.search(r"```(?:json)?\s*(\[[\s\S]*?\])\s*```", text)
    if m:
        try:
            return json.loads(m.group(1))
        except (json.JSONDecodeError, ValueError):
            pass
    # Try each '[' occurrence from right to left (last valid array wins)
    for start in range(text.rfind("["), -1, -1):
        if text[start] != "[":
            continue
        end = text.rfind("]", start) + 1
        if end <= start:
            continue
        try:
            return json.loads(text[start:end])
        except (json.JSONDecodeError, ValueError):
            continue
    return []


# ════════════════════════════════════════════
# Evaluation
# ════════════════════════════════════════════
@dataclass
class EvalResult:
    model_name: str
    task: str
    total_samples: int = 0
    total_candidates: int = 0
    parse_failures: int = 0
    latencies: list = field(default_factory=list)
    # Institution metrics
    inst_tp: int = 0  # true positive (pred=true, gt=true)
    inst_fp: int = 0  # false positive (pred=true, gt=false)
    inst_tn: int = 0  # true negative (pred=false, gt=false)
    inst_fn: int = 0  # false negative (pred=false, gt=true)
    inst_confidence_correct: list = field(default_factory=list)
    # Product metrics
    prod_grade_correct: int = 0
    prod_grade_details: dict = field(default_factory=dict)  # {grade: {correct, total}}
    prod_core_name_correct: int = 0
    prod_core_name_total: int = 0
    # Top-1 selection metrics (业务核心：是否选出了最佳匹配)
    top1_correct: int = 0  # 选中的最佳匹配与 ground truth 一致
    top1_total: int = 0

    def inst_accuracy(self) -> float:
        return (self.inst_tp + self.inst_tn) / self.total_candidates if self.total_candidates else 0

    def inst_precision(self) -> float:
        return self.inst_tp / (self.inst_tp + self.inst_fp) if (self.inst_tp + self.inst_fp) else 0

    def inst_recall(self) -> float:
        return self.inst_tp / (self.inst_tp + self.inst_fn) if (self.inst_tp + self.inst_fn) else 0

    def inst_f1(self) -> float:
        p, r = self.inst_precision(), self.inst_recall()
        return 2 * p * r / (p + r) if (p + r) else 0

    def prod_accuracy(self) -> float:
        return self.prod_grade_correct / self.total_candidates if self.total_candidates else 0

    def prod_core_name_accuracy(self) -> float:
        return self.prod_core_name_correct / self.prod_core_name_total if self.prod_core_name_total else 0

    def avg_latency(self) -> float:
        return sum(self.latencies) / len(self.latencies) if self.latencies else 0


def _eval_one_sample(
    sample: dict,
    model_name: str,
    base_url: str,
    task: str,
    mlx_model=None,
    mlx_tokenizer=None,
) -> dict:
    """Evaluate a single sample, return per-sample metrics."""
    input_messages = [m for m in sample["messages"] if m["role"] != "assistant"]
    ground_truth = extract_ground_truth(sample)

    result = {
        "latency": 0, "total_samples": 1, "parse_failure": 0,
        "total_candidates": 0, "candidates": [],
    }

    try:
        if mlx_model is not None and mlx_tokenizer is not None:
            response_text, latency = call_model_mlx(input_messages, mlx_model, mlx_tokenizer)
        else:
            response_text, latency = call_model(input_messages, model_name, base_url)
        result["latency"] = latency

        predicted = parse_json_array(response_text)
        if not predicted:
            result["parse_failure"] = 1

        for gt_item in ground_truth:
            result["total_candidates"] += 1
            gt_idx = gt_item.get("index", 0)
            pred_item = next((p for p in predicted if p.get("index") == gt_idx), None) if predicted else None
            result["candidates"].append({"gt": gt_item, "pred": pred_item})

    except Exception as e:
        console.print(f"[red]模型调用失败: {e}[/red]")
        result["parse_failure"] = 1
        result["total_candidates"] = len(ground_truth)
        result["candidates"] = [{"gt": gt, "pred": None} for gt in ground_truth]

    return result


def _merge_sample(result: EvalResult, sample_result: dict, task: str):
    """Merge a single sample result into the aggregate EvalResult."""
    result.total_samples += sample_result["total_samples"]
    result.total_candidates += sample_result["total_candidates"]
    result.latencies.append(sample_result["latency"])
    result.parse_failures += sample_result["parse_failure"]

    # Top-1 selection: find best match in ground truth and prediction
    result.top1_total += 1
    if task == "institution":
        # GT best = matched=true
        gt_best = next((c for c in sample_result["candidates"] if c["gt"].get("matched")), None)
        gt_best_idx = gt_best["gt"].get("index") if gt_best else None
        # Pred best = matched=true with highest confidence
        conf_order = {"High": 3, "Medium": 2, "Low": 1}
        pred_matches = [
            c for c in sample_result["candidates"]
            if c["pred"] and c["pred"].get("matched")
        ]
        if pred_matches:
            pred_best = max(pred_matches, key=lambda c: conf_order.get(c["pred"].get("confidence", "Low"), 0))
            pred_best_idx = pred_best["pred"].get("index")
        else:
            pred_best_idx = None
        if gt_best_idx is not None and pred_best_idx == gt_best_idx:
            result.top1_correct += 1
    elif task == "product":
        grade_score = {"A": 95, "B": 75, "D": 0}
        # GT best = highest grade score
        gt_best = max(sample_result["candidates"], key=lambda c: grade_score.get(c["gt"].get("match_grade", "D"), 0))
        gt_best_idx = gt_best["gt"].get("index")
        # Pred best = highest grade score among valid predictions
        valid_preds = [c for c in sample_result["candidates"] if c["pred"] and c["pred"].get("match_grade") in grade_score]
        if valid_preds:
            pred_best = max(valid_preds, key=lambda c: grade_score.get(c["pred"].get("match_grade", "D"), 0))
            pred_best_idx = pred_best["pred"].get("index")
        else:
            pred_best_idx = None
        if pred_best_idx == gt_best_idx:
            result.top1_correct += 1

    for pair in sample_result["candidates"]:
        gt_item, pred_item = pair["gt"], pair["pred"]
        if task == "institution":
            gt_matched = gt_item.get("matched", False)
            pred_matched = pred_item.get("matched", False) if pred_item else not gt_matched
            if gt_matched and pred_matched:
                result.inst_tp += 1
            elif not gt_matched and not pred_matched:
                result.inst_tn += 1
            elif not gt_matched and pred_matched:
                result.inst_fp += 1
            else:
                result.inst_fn += 1
        elif task == "product":
            gt_grade = gt_item.get("match_grade", "D")
            pred_grade = pred_item.get("match_grade", "X") if pred_item else "X"
            gt_core = gt_item.get("core_name_match", False)
            pred_core = pred_item.get("core_name_match", False) if pred_item else False
            if gt_grade == pred_grade:
                result.prod_grade_correct += 1
            result.prod_grade_details.setdefault(gt_grade, {"correct": 0, "total": 0})
            result.prod_grade_details[gt_grade]["total"] += 1
            if gt_grade == pred_grade:
                result.prod_grade_details[gt_grade]["correct"] += 1
            result.prod_core_name_total += 1
            if gt_core == pred_core:
                result.prod_core_name_correct += 1


def evaluate_model(
    model_name: str,
    base_url: str,
    task: str,
    samples: list[dict],
    concurrency: int = 1,
    log_every: int = 50,
    adapter_path: str | None = None,
    mlx_model_id: str | None = None,
) -> EvalResult:
    """Evaluate a model on a task with optional concurrency."""
    result = EvalResult(model_name=model_name, task=task)
    total = len(samples)

    # Load MLX model + adapter if adapter_path is provided
    mlx_model = None
    mlx_tokenizer = None
    if adapter_path:
        from mlx_lm import load
        console.print(f"[dim]加载 MLX 模型: {mlx_model_id or model_name} + adapter: {adapter_path}[/dim]")
        mlx_model, mlx_tokenizer = load(mlx_model_id or model_name, adapter_path=adapter_path)

    if concurrency > 1 and not adapter_path:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = {pool.submit(_eval_one_sample, s, model_name, base_url, task): i for i, s in enumerate(samples)}
            completed = 0
            for future in as_completed(futures):
                sample_result = future.result()
                _merge_sample(result, sample_result, task)
                completed += 1
                if completed % log_every == 0:
                    _print_progress(result, model_name, completed, total, task)
    else:
        for idx, sample in enumerate(samples):
            sample_result = _eval_one_sample(
                sample, model_name, base_url, task, mlx_model=mlx_model, mlx_tokenizer=mlx_tokenizer
            )
            _merge_sample(result, sample_result, task)
            if (idx + 1) % log_every == 0:
                _print_progress(result, model_name, idx + 1, total, task)

    return result


def _print_progress(result: EvalResult, model_name: str, done: int, total: int, task: str):
    if task == "institution":
        acc = (result.inst_tp + result.inst_tn) / result.total_candidates * 100 if result.total_candidates else 0
        top1 = result.top1_correct / result.top1_total * 100 if result.top1_total else 0
        print(f"  [{model_name}] {done}/{total} acc={acc:.1f}% top1={top1:.1f}% parse_fail={result.parse_failures}")
    else:
        acc = result.prod_grade_correct / result.total_candidates * 100 if result.total_candidates else 0
        top1 = result.top1_correct / result.top1_total * 100 if result.top1_total else 0
        print(f"  [{model_name}] {done}/{total} grade_acc={acc:.1f}% top1={top1:.1f}% parse_fail={result.parse_failures}")


def print_results(results: list[EvalResult]):
    """Print comparison table."""
    console.print("\n[bold]评测结果对比[/bold]\n")

    # Institution results
    inst_results = [r for r in results if r.task == "institution"]
    if inst_results:
        table = Table(title="机构匹配 (Institution Matching)")
        table.add_column("模型", style="cyan")
        table.add_column("Top-1选择", justify="right")
        table.add_column("准确率", justify="right")
        table.add_column("Precision", justify="right")
        table.add_column("Recall", justify="right")
        table.add_column("F1", justify="right")
        table.add_column("解析失败", justify="right")
        table.add_column("平均延迟", justify="right")

        for r in sorted(inst_results, key=lambda x: x.top1_correct / x.top1_total if x.top1_total else 0, reverse=True):
            top1 = r.top1_correct / r.top1_total if r.top1_total else 0
            table.add_row(
                r.model_name,
                f"[bold]{top1:.1%}[/bold]",
                f"{r.inst_accuracy():.1%}",
                f"{r.inst_precision():.1%}",
                f"{r.inst_recall():.1%}",
                f"{r.inst_f1():.1%}",
                f"{r.parse_failures}/{r.total_samples}",
                f"{r.avg_latency():.0f}ms",
            )
        console.print(table)

    # Product results
    prod_results = [r for r in results if r.task == "product"]
    if prod_results:
        table = Table(title="产品匹配 (Product Matching)")
        table.add_column("模型", style="cyan")
        table.add_column("Top-1选择", justify="right")
        table.add_column("等级准确率", justify="right")
        table.add_column("核心名准确率", justify="right")
        table.add_column("A级", justify="right")
        table.add_column("B级", justify="right")
        table.add_column("D级", justify="right")
        table.add_column("解析失败", justify="right")
        table.add_column("平均延迟", justify="right")

        for r in sorted(prod_results, key=lambda x: x.top1_correct / x.top1_total if x.top1_total else 0, reverse=True):
            top1 = r.top1_correct / r.top1_total if r.top1_total else 0
            grades = r.prod_grade_details
            a_acc = f"{grades.get('A', {}).get('correct', 0) / grades.get('A', {}).get('total', 1):.0%}" if grades.get('A') else "-"
            b_acc = f"{grades.get('B', {}).get('correct', 0) / grades.get('B', {}).get('total', 1):.0%}" if grades.get('B') else "-"
            d_acc = f"{grades.get('D', {}).get('correct', 0) / grades.get('D', {}).get('total', 1):.0%}" if grades.get('D') else "-"
            table.add_row(
                r.model_name,
                f"[bold]{top1:.1%}[/bold]",
                f"{r.prod_accuracy():.1%}",
                f"{r.prod_core_name_accuracy():.1%}",
                a_acc, b_acc, d_acc,
                f"{r.parse_failures}/{r.total_samples}",
                f"{r.avg_latency():.0f}ms",
            )
        console.print(table)


# ════════════════════════════════════════════
# Main
# ════════════════════════════════════════════
def main():
    import argparse

    parser = argparse.ArgumentParser(description="主数据匹配评测")
    parser.add_argument("--local-model", type=str, help="LM Studio 本地模型名")
    parser.add_argument("--local-base-url", type=str, default="http://127.0.0.1:1234/v1/", help="LM Studio URL")
    parser.add_argument("--api-model", type=str, help="云端 API 模型名 (如 glm-5.1)")
    parser.add_argument("--api-base-url", type=str, default=None, help="云端 API URL")
    parser.add_argument("--adapter-path", type=str, default=None, help="MLX LoRA adapter 路径（本地直接评测）")
    parser.add_argument("--mlx-model-id", type=str, default=None, help="MLX 模型 ID（用于 --adapter-path 模式）")
    parser.add_argument("--task", choices=["institution", "product", "both"], default="both", help="评测任务")
    parser.add_argument("--max-samples", type=int, default=200, help="每个任务最大评测条数")
    parser.add_argument("--concurrency", type=int, default=3, help="本地模型并发数（默认3）")
    args = parser.parse_args()

    if not args.local_model and not args.api_model and not args.adapter_path:
        console.print("[red]请指定 --local-model、--api-model 或 --adapter-path[/red]")
        return

    results = []

    tasks = ["institution", "product"] if args.task == "both" else [args.task]

    for task in tasks:
        console.print(f"\n[bold cyan]加载 {task} 评测数据...[/bold cyan]")
        data = load_eval_data(task)
        if len(data) > args.max_samples:
            random.seed(42)
            data = random.sample(data, args.max_samples)
        console.print(f"  评测 {len(data)} 条")

        if args.adapter_path:
            model_id = args.mlx_model_id or args.local_model or "mlx-community/gemma-4-26b-a4b-it-4bit"
            console.print(f"[yellow]评测: {model_id} + adapter ({task})[/yellow]")
            r = evaluate_model(
                model_id, "", task, data,
                adapter_path=args.adapter_path,
                mlx_model_id=model_id,
            )
            results.append(r)
            console.print(f"  [green]✓ adapter {task} 完成[/green]")

        if args.local_model:
            console.print(f"[yellow]评测: {args.local_model} ({task}, 并发={args.concurrency})[/yellow]")
            r = evaluate_model(args.local_model, args.local_base_url, task, data, concurrency=args.concurrency)
            results.append(r)
            console.print(f"  [green]✓ {args.local_model} {task} 完成[/green]")

        if args.api_model:
            import os
            if args.api_base_url:
                api_url = args.api_base_url
            elif "minimax" in args.api_model.lower():
                api_url = "https://api.minimax.chat/v1/"
            else:
                api_url = os.environ.get("LLM_API_BASE_URL", "https://open.bigmodel.cn/api/coding/paas/v4/")
            console.print(f"[yellow]评测: {args.api_model} ({task}, 并发={args.concurrency})[/yellow]")
            r = evaluate_model(args.api_model, api_url, task, data, concurrency=args.concurrency)
            results.append(r)
            console.print(f"  [green]✓ {args.api_model} {task} 完成[/green]")

    print_results(results)

    # Save results
    output_dir = DOMAIN_ROOT / "data" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"eval_{timestamp}.json"

    results_json = []
    for r in results:
        d = {
            "model": r.model_name, "task": r.task,
            "total_samples": r.total_samples, "total_candidates": r.total_candidates,
            "parse_failures": r.parse_failures, "avg_latency_ms": r.avg_latency(),
        }
        d["top1_accuracy"] = r.top1_correct / r.top1_total if r.top1_total else 0
        if r.task == "institution":
            d.update({"accuracy": r.inst_accuracy(), "precision": r.inst_precision(),
                       "recall": r.inst_recall(), "f1": r.inst_f1()})
        else:
            d.update({"grade_accuracy": r.prod_accuracy(), "core_name_accuracy": r.prod_core_name_accuracy(),
                       "grade_details": r.prod_grade_details})
        results_json.append(d)

    with open(output_path, "w") as f:
        json.dump(results_json, f, ensure_ascii=False, indent=2)
    console.print(f"\n结果已保存: {output_path}")


if __name__ == "__main__":
    main()
