"""在任意领域实体匹配 test 集上评测 adapter / 合并模型（通用评测闭环，R120）。

用法::

    python scripts/eval_entity_match.py \\
        --model-path outputs/sft/my-run \\
        --test-file outputs/wizard/demo_suppliers/test.json

北极星（通用微调台）：用户用 Data Wizard 导出的 test.json（供应商、药品、
机构、零件……任意领域）评自己刚训练的模型；结果写
``domains/entity_matching/data/results/eval_detail_*.json``，直接点亮
「评测结果 / 模型对比」两页（与 domains/medical_entity 评测同构契约——
那是已验证案例，不是评测唯一入口）。

判分/聚合在 src/evaluation/entity_eval.py（纯 stdlib，单测覆盖）；本脚本
只做参数解析、模型懒加载与生成循环。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.entity_eval import build_rows, summarize, write_report  # noqa: E402

DEFAULT_RESULTS_DIR = Path("domains/entity_matching/data/results")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="在任意领域实体匹配 test 集上评测模型（通用评测闭环）"
    )
    parser.add_argument(
        "--model-path",
        required=True,
        help="adapter 目录（含 adapter_config.json）或合并后的完整模型目录",
    )
    parser.add_argument(
        "--test-file", required=True, help="评测集 JSON（Data Wizard 导出的 test.json）"
    )
    parser.add_argument(
        "--base-model",
        default=None,
        help="底座覆盖（默认从 adapter_config.json 读取；merged 目录无需填写）",
    )
    parser.add_argument("--max-samples", type=int, default=200, help="最多评测条数")
    parser.add_argument("--max-new-tokens", type=int, default=64, help="每条生成上限")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help=f"结果目录（默认 {DEFAULT_RESULTS_DIR}，02/03 两页从这里读）",
    )
    parser.add_argument("--model-name", default=None, help="结果中的模型显示名（默认取目录名）")
    return parser.parse_args(argv)


def load_test_records(path: Path) -> list[dict]:
    """读入并校验 test 集：必须是非空 Alpaca 记录列表（有 input/output）。"""
    if not path.exists():
        raise SystemExit(f"✗ 评测集不存在: {path}")
    try:
        records = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"✗ 评测集解析失败: {exc}") from exc
    if not isinstance(records, list) or not records:
        raise SystemExit(f"✗ 评测集为空或不是 JSON 数组: {path}")
    bad = [i for i, r in enumerate(records) if not isinstance(r, dict) or "input" not in r]
    if bad:
        raise SystemExit(f"✗ 第 {bad[0] + 1} 条记录缺少 input 字段（需要 Alpaca 选择题格式）")
    return records


def resolve_model_refs(model_path: str, base_override: str | None) -> tuple[str, str | None]:
    """(底座, adapter) —— adapter 目录自动读底座；merged 目录直接当底座。"""
    p = Path(model_path).expanduser()
    if (p / "adapter_config.json").exists():
        base = base_override
        if base is None:
            with open(p / "adapter_config.json", encoding="utf-8") as f:
                base = str(json.load(f).get("base_model_name_or_path", "") or "")
        if not base:
            raise SystemExit(f"✗ adapter_config.json 未记录底座，且未提供 --base-model: {p}")
        return base, str(p)
    if base_override:
        return base_override, str(p)
    return str(p), None


def generate_predictions(  # noqa: PLR0913
    model_path: str,
    base_override: str | None,
    records: list[dict],
    max_new_tokens: int,
) -> tuple[list[str], list[float]]:
    """逐条贪心生成（prompt 与 SFT 训练同款 Alpaca 模板），返回 (预测, 延迟ms)。"""
    import torch

    from src.data.preprocessors import format_instruction
    from src.inference.chat_engine import load_chat_model

    base, adapter = resolve_model_refs(model_path, base_override)
    print(f"▸ 加载模型: 底座={base}" + (f" adapter={adapter}" if adapter else "（merged）"))
    model, tokenizer = load_chat_model(base, adapter)
    model.eval()

    predictions: list[str] = []
    latencies: list[float] = []
    for i, rec in enumerate(records):
        prompt = format_instruction(
            str(rec.get("instruction", "")), str(rec.get("input", "")), "", "alpaca"
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        latency_ms = (time.perf_counter() - t0) * 1000
        new_tokens = out[0][inputs["input_ids"].shape[1] :]
        predictions.append(tokenizer.decode(new_tokens, skip_special_tokens=True))
        latencies.append(latency_ms)
        if (i + 1) % 20 == 0 or i + 1 == len(records):
            print(f"  [{i + 1}/{len(records)}] 最近延迟 {latency_ms:.0f}ms")
    return predictions, latencies


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    records = load_test_records(Path(args.test_file).expanduser())
    if args.max_samples > 0:
        records = records[: args.max_samples]
    print(f"▸ 评测集: {args.test_file}（{len(records)} 条）")

    predictions, latencies = generate_predictions(
        args.model_path, args.base_model, records, args.max_new_tokens
    )

    rows = build_rows(records, predictions, latencies)
    model_name = args.model_name or Path(args.model_path).name
    summary = summarize(model_name, rows)
    path = write_report([summary], args.results_dir)

    acc = summary["overall_accuracy"]
    acc_text = f"{acc:.1%}" if acc is not None else "—"
    print(f"\n✓ {model_name}: 准确率 {acc_text}（{summary['correct']}/{summary['total']}）")
    for diff, value in summary["accuracy_by_difficulty"].items():
        print(f"    {diff}: {value:.1%}")
    print(f"✓ 结果已写入: {path}")
    print("  → 到「评测结果 / 模型对比」页查看（域=实体匹配（通用））")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
