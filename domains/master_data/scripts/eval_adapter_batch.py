#!/usr/bin/env python3
"""批量评测 MLX LoRA adapter（基于 test_adapter_quick.py 的可靠模式）。"""

import json
import random
import time
from pathlib import Path

from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler

DOMAIN_ROOT = Path(__file__).resolve().parent.parent


def load_eval_data(task: str) -> list[dict]:
    path = DOMAIN_ROOT / "data" / "test" / f"eval_{task}.json"
    with open(path) as f:
        return json.load(f)


def extract_ground_truth(sample: dict) -> list[dict]:
    for msg in sample["messages"]:
        if msg["role"] == "assistant":
            return json.loads(msg["content"])
    return []


def parse_json_array(text: str) -> list[dict]:
    import re
    m = re.search(r"```(?:json)?\s*(\[[\s\S]*?\])\s*```", text)
    if m:
        try:
            return json.loads(m.group(1))
        except (json.JSONDecodeError, ValueError):
            pass
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


def evaluate_adapter(task: str, max_samples: int, model_id: str, adapter_path: str) -> None:
    print(f"[加载模型] {model_id}")
    print(f"[加载adapter] {adapter_path}")
    model, tokenizer = load(model_id, adapter_path=adapter_path)

    data = load_eval_data(task)
    if max_samples and len(data) > max_samples:
        random.seed(42)
        data = random.sample(data, max_samples)

    total = len(data)
    print(f"评测 {total} 条 {task} 数据...")

    parse_failures = 0
    latencies = []

    # Institution metrics
    inst_tp = inst_fp = inst_tn = inst_fn = 0
    top1_correct = 0

    # Product metrics
    prod_grade_correct = 0
    prod_core_name_correct = 0
    prod_core_name_total = 0
    prod_grade_details: dict = {}

    sampler = make_sampler(temp=0.1)

    for idx, sample in enumerate(data):
        input_messages = [m for m in sample["messages"] if m["role"] != "assistant"]
        ground_truth = extract_ground_truth(sample)

        prompt = tokenizer.apply_chat_template(input_messages, tokenize=False, add_generation_prompt=True)

        t0 = time.time()
        try:
            response = generate(model, tokenizer, prompt=prompt, max_tokens=1024, sampler=sampler, verbose=False)
        except Exception as e:
            print(f"  ERROR at {idx + 1}: {e}")
            parse_failures += 1
            continue

        latency = (time.time() - t0) * 1000
        latencies.append(latency)

        predicted = parse_json_array(response)
        if not predicted:
            parse_failures += 1

        # Evaluate this sample
        if task == "institution":
            for gt_item in ground_truth:
                gt_idx = gt_item.get("index", 0)
                pred_item = next((p for p in predicted if p.get("index") == gt_idx), None) if predicted else None

                gt_matched = gt_item.get("matched", False)
                pred_matched = pred_item.get("matched", False) if pred_item else not gt_matched

                if gt_matched and pred_matched:
                    inst_tp += 1
                elif not gt_matched and not pred_matched:
                    inst_tn += 1
                elif not gt_matched and pred_matched:
                    inst_fp += 1
                else:
                    inst_fn += 1

            conf_order = {"High": 3, "Medium": 2, "Low": 1}
            gt_best = next((c for c in ground_truth if c.get("matched")), None)
            gt_best_idx = gt_best.get("index") if gt_best else None

            pred_matches = [p for p in predicted if p.get("matched")]
            if pred_matches:
                pred_best = max(pred_matches, key=lambda p: conf_order.get(p.get("confidence", "Low"), 0))
                pred_best_idx = pred_best.get("index")
            else:
                pred_best_idx = None

            if gt_best_idx is not None and pred_best_idx == gt_best_idx:
                top1_correct += 1

            if (idx + 1) % 50 == 0:
                total_cand = inst_tp + inst_tn + inst_fp + inst_fn
                acc = (inst_tp + inst_tn) / total_cand * 100 if total_cand else 0
                top1 = top1_correct / (idx + 1) * 100
                print(f"  [{idx + 1}/{total}] acc={acc:.1f}% top1={top1:.1f}% parse_fail={parse_failures}")

        elif task == "product":
            grade_score = {"A": 95, "B": 75, "D": 0}
            gt_best = max(ground_truth, key=lambda c: grade_score.get(c.get("match_grade", "D"), 0))
            gt_best_idx = gt_best.get("index")

            valid_preds = [p for p in predicted if p.get("match_grade") in grade_score]
            if valid_preds:
                pred_best = max(valid_preds, key=lambda p: grade_score.get(p.get("match_grade", "D"), 0))
                pred_best_idx = pred_best.get("index")
            else:
                pred_best_idx = None

            if pred_best_idx == gt_best_idx:
                top1_correct += 1

            for gt_item in ground_truth:
                gt_idx = gt_item.get("index", 0)
                pred_item = next((p for p in predicted if p.get("index") == gt_idx), None) if predicted else None

                gt_grade = gt_item.get("match_grade", "D")
                pred_grade = pred_item.get("match_grade", "X") if pred_item else "X"
                gt_core = gt_item.get("core_name_match", False)
                pred_core = pred_item.get("core_name_match", False) if pred_item else False

                if gt_grade == pred_grade:
                    prod_grade_correct += 1

                prod_grade_details.setdefault(gt_grade, {"correct": 0, "total": 0})
                prod_grade_details[gt_grade]["total"] += 1
                if gt_grade == pred_grade:
                    prod_grade_details[gt_grade]["correct"] += 1

                prod_core_name_total += 1
                if gt_core == pred_core:
                    prod_core_name_correct += 1

            if (idx + 1) % 50 == 0:
                total_cand = sum(prod_grade_details.get(g, {}).get("total", 0) for g in ["A", "B", "D"])
                acc = prod_grade_correct / total_cand * 100 if total_cand else 0
                top1 = top1_correct / (idx + 1) * 100
                print(f"  [{idx + 1}/{total}] grade_acc={acc:.1f}% top1={top1:.1f}% parse_fail={parse_failures}")

    # Final results
    print(f"\n{'=' * 60}")
    print(f"评测完成: {task}")
    print(f"{'=' * 60}")

    if task == "institution":
        total_cand = inst_tp + inst_tn + inst_fp + inst_fn
        acc = (inst_tp + inst_tn) / total_cand * 100 if total_cand else 0
        precision = inst_tp / (inst_tp + inst_fp) * 100 if (inst_tp + inst_fp) else 0
        recall = inst_tp / (inst_tp + inst_fn) * 100 if (inst_tp + inst_fn) else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
        top1 = top1_correct / total * 100 if total else 0

        print(f"Top-1: {top1:.1f}%")
        print(f"准确率: {acc:.1f}%")
        print(f"Precision: {precision:.1f}%")
        print(f"Recall: {recall:.1f}%")
        print(f"F1: {f1:.1f}%")
        print(f"解析失败: {parse_failures}/{total}")

        result = {
            "model": model_id, "adapter": adapter_path, "task": task,
            "total_samples": total, "parse_failures": parse_failures,
            "avg_latency_ms": sum(latencies) / len(latencies) if latencies else 0,
            "top1_accuracy": top1_correct / total if total else 0,
            "accuracy": acc / 100, "precision": precision / 100,
            "recall": recall / 100, "f1": f1 / 100,
        }
    else:
        total_cand = sum(prod_grade_details.get(g, {}).get("total", 0) for g in ["A", "B", "D"])
        acc = prod_grade_correct / total_cand * 100 if total_cand else 0
        core_acc = prod_core_name_correct / prod_core_name_total * 100 if prod_core_name_total else 0
        top1 = top1_correct / total * 100 if total else 0

        print(f"Top-1: {top1:.1f}%")
        print(f"等级准确率: {acc:.1f}%")
        print(f"核心名准确率: {core_acc:.1f}%")
        for g in ["A", "B", "D"]:
            d = prod_grade_details.get(g, {"correct": 0, "total": 0})
            if d["total"] > 0:
                print(f"  {g}级: {d['correct']}/{d['total']} = {d['correct'] / d['total'] * 100:.0f}%")
        print(f"解析失败: {parse_failures}/{total}")

        result = {
            "model": model_id, "adapter": adapter_path, "task": task,
            "total_samples": total, "parse_failures": parse_failures,
            "avg_latency_ms": sum(latencies) / len(latencies) if latencies else 0,
            "top1_accuracy": top1_correct / total if total else 0,
            "grade_accuracy": acc / 100, "core_name_accuracy": core_acc / 100,
            "grade_details": prod_grade_details,
        }

    avg_latency = sum(latencies) / len(latencies) if latencies else 0
    print(f"平均延迟: {avg_latency:.0f}ms")

    output_dir = DOMAIN_ROOT / "data" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"eval_adapter_{task}_{timestamp}.json"

    with open(output_path, "w") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存: {output_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="批量评测 MLX LoRA adapter")
    parser.add_argument("--task", choices=["institution", "product", "both"], default="both")
    parser.add_argument("--max-samples", type=int, default=400)
    parser.add_argument("--model-id", type=str, default="mlx-community/gemma-4-26b-a4b-it-4bit")
    parser.add_argument("--adapter-path", type=str, default="domains/master_data/outputs/adapters-gemma-26b")
    args = parser.parse_args()

    tasks = ["institution", "product"] if args.task == "both" else [args.task]
    for task in tasks:
        evaluate_adapter(task, args.max_samples, args.model_id, args.adapter_path)


if __name__ == "__main__":
    main()
