#!/usr/bin/env python3
"""CLI script to evaluate fine-tuned models.

Usage:
    python scripts/evaluate.py --model-path ./outputs/sft --dataset finance-alpaca
"""

import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.base import ModelConfig
from src.models import load_merged_model, load_model_and_tokenizer
from src.data import AlpacaDataset, FinanceDataset
from src.evaluation import compute_perplexity, generate_samples, compare_models
from src.utils import console

app = typer.Typer(
    name="evaluate-model",
    help="Evaluate fine-tuned models",
    add_completion=False,
)


@app.command()
def main(
    model_path: str = typer.Option(
        ...,
        "--model-path",
        "-m",
        help="Path to model (merged or with adapters)",
    ),
    dataset: str = typer.Option(
        "yahma/alpaca-cleaned",
        "--dataset",
        "-d",
        help="Dataset to evaluate on",
    ),
    max_samples: int = typer.Option(
        100,
        "--max-samples",
        "-n",
        help="Number of samples to evaluate",
    ),
    num_generations: int = typer.Option(
        5,
        "--num-generations",
        "-g",
        help="Number of test generations",
    ),
    compare_with: Optional[str] = typer.Option(
        None,
        "--compare-with",
        help="Compare with base model (path or HF name)",
    ),
    output_file: Optional[str] = typer.Option(
        None,
        "--output",
        "-o",
        help="Save results to JSON file",
    ),
):
    """Evaluate model on dataset."""

    console.print("\n[bold cyan]=== Model Evaluation ===[/bold cyan]\n")

    model_dir = Path(model_path)
    if not model_dir.exists():
        console.print(f"[red]✗ Model path not found: {model_path}[/red]")
        raise typer.Exit(1)

    # Load model
    console.print(f"[cyan]Loading model from: {model_path}[/cyan]")
    try:
        # Try loading as merged model first
        model, tokenizer = load_merged_model(model_path)
    except Exception:
        # Fall back to loading with adapters
        config = ModelConfig(name=model_path)
        model, tokenizer = load_model_and_tokenizer(config)

    console.print("[green]✓ Model loaded[/green]\n")

    # Load dataset
    console.print(f"[cyan]Loading dataset: {dataset}[/cyan]")
    if "finance" in dataset.lower():
        eval_dataset = FinanceDataset(dataset, max_samples=max_samples)
    else:
        eval_dataset = AlpacaDataset(dataset, max_samples=max_samples)

    eval_dataset.load()
    console.print(f"[green]✓ Loaded {len(eval_dataset)} samples[/green]\n")

    # Compute perplexity
    console.print("[cyan]Computing perplexity...[/cyan]")
    ppl = compute_perplexity(model, eval_dataset.dataset, tokenizer)
    console.print(f"[green]✓ Perplexity: {ppl:.2f}[/green]\n")

    # Generate samples
    console.print(f"[cyan]Generating {num_generations} samples...[/cyan]\n")
    generations = generate_samples(
        model,
        tokenizer,
        eval_dataset.dataset,
        num_samples=num_generations,
        max_length=256,
    )

    # Print generations
    for i, gen in enumerate(generations, 1):
        console.print(Panel.fit(
            f"[bold]Sample {i}[/bold]\n\n"
            f"[cyan]Prompt:[/cyan] {gen['prompt'][:200]}...\n\n"
            f"[green]Response:[/green] {gen['response']}",
            border_style="cyan",
        ))
        console.print()

    # Compare with base model if requested
    if compare_with:
        console.print(f"[cyan]Comparing with base model: {compare_with}[/cyan]\n")
        comparison = compare_models(
            model_path=model_path,
            base_model_path=compare_with,
            dataset=eval_dataset.dataset,
            tokenizer=tokenizer,
        )

        # Print comparison table
        table = Table(title="Model Comparison")
        table.add_column("Metric", style="cyan")
        table.add_column("Fine-tuned", style="green")
        table.add_column("Base", style="yellow")
        table.add_column("Improvement", style="white")

        for metric, values in comparison.items():
            tuned_val = values.get("tuned", "N/A")
            base_val = values.get("base", "N/A")
            improvement = values.get("improvement", "N/A")

            table.add_row(metric, str(tuned_val), str(base_val), str(improvement))

        console.print(table)
        console.print()

    # Save results if requested
    if output_file:
        import json

        results = {
            "model_path": model_path,
            "dataset": dataset,
            "perplexity": ppl,
            "generations": generations,
        }

        if compare_with:
            results["comparison"] = comparison

        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

        console.print(f"[green]✓ Results saved to: {output_file}[/green]\n")

    console.print("[bold green]✓ Evaluation complete![/bold green]\n")


if __name__ == "__main__":
    app()
