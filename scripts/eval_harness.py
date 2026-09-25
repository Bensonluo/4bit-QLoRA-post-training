#!/usr/bin/env python3
"""Run EleutherAI lm-evaluation-harness benchmarks on a trained/merged model.

Complements the domain evaluators (``scripts/evaluate.py``) with
industry-standard public benchmarks so fine-tuned models can be anchored
against published baselines. Requires the opt-in evaluation stack:

    pip install -e ".[eval]"

Examples:
    # Smoke test on a merged model (500 samples per task)
    python scripts/eval_harness.py --model-path outputs/merged/run-x \\
        --tasks hellaswag,arc_easy --limit 500

    # Few-shot MMLU subset with the chat template applied
    python scripts/eval_harness.py -m Qwen/Qwen2.5-1.5B-Instruct \\
        --tasks mmlu --num-fewshot 5 --chat-template --output outputs/eval/mmlu.json
"""

import sys
from pathlib import Path

import typer
from rich.table import Table

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.harness import run_harness_eval, summarize_results
from src.utils import console

app = typer.Typer(
    name="eval-harness",
    help="Benchmark models with EleutherAI lm-evaluation-harness",
    add_completion=False,
)


@app.command()
def main(
    model_path: str = typer.Option(
        ...,
        "--model-path",
        "-m",
        help="Merged model directory or HF hub name (adapters must be merged first)",
    ),
    tasks: str = typer.Option(
        "hellaswag,arc_easy",
        "--tasks",
        "-t",
        help="Comma-separated lm-eval task names (list: `lm-eval ls tasks`)",
    ),
    num_fewshot: int | None = typer.Option(
        None,
        "--num-fewshot",
        help="Few-shot examples per task (harness default if omitted)",
    ),
    limit: int = typer.Option(
        0,
        "--limit",
        help="Max samples per task, 0 = full task (limits are for smoke tests only)",
    ),
    batch_size: str = typer.Option("auto", "--batch-size", help="Batch size: 'auto' or an integer"),
    device: str | None = typer.Option(
        None, "--device", help="Device override (cuda, cuda:0, mps, cpu)"
    ),
    dtype: str = typer.Option(
        "auto", "--dtype", help="Model dtype for the hf backend (auto, float16, bfloat16)"
    ),
    chat_template: bool = typer.Option(
        False, "--chat-template", help="Apply the model's chat template to prompts"
    ),
    output: Path | None = typer.Option(
        None, "--output", "-o", help="Write the full harness result JSON to this path"
    ),
) -> None:
    task_list = [t.strip() for t in tasks.split(",") if t.strip()]
    if not task_list:
        console.print("[red]No tasks given — e.g. --tasks hellaswag,arc_easy[/red]")
        raise typer.Exit(code=1)

    results = run_harness_eval(
        model_path,
        task_list,
        num_fewshot=num_fewshot,
        limit=limit or None,
        batch_size=batch_size,
        device=device,
        dtype=dtype,
        apply_chat_template=chat_template,
        output_path=output,
    )

    table = Table(title="lm-evaluation-harness results")
    table.add_column("Task", style="cyan")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    for task, metrics in summarize_results(results).items():
        for metric, value in metrics.items():
            if metric.endswith("_stderr"):
                continue
            if value is None:
                continue
            stderr = metrics.get(f"{metric}_stderr")
            shown = f"{value:.4f}" + (f" ±{stderr:.4f}" if stderr is not None else "")
            table.add_row(task, metric, shown)
    console.print(table)


if __name__ == "__main__":
    app()
