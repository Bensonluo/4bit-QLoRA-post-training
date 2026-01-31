#!/usr/bin/env python3
"""CLI script to download and prepare datasets.

Usage:
    python scripts/download_data.py --dataset finance-alpaca --output-dir ./data/raw
"""

import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from datasets import load_dataset
from src.utils import console

app = typer.Typer(
    name="download-data",
    help="Download datasets from Hugging Face or local sources",
    add_completion=False,
)


@app.command()
def main(
    dataset: str = typer.Option(
        "yahma/alpaca-cleaned",
        "--dataset",
        "-d",
        help="Dataset name on Hugging Face or local path",
    ),
    output_dir: str = typer.Option(
        "./data/raw",
        "--output-dir",
        "-o",
        help="Output directory for downloaded data",
    ),
    split: str = typer.Option(
        "train",
        "--split",
        help="Dataset split to download",
    ),
    num_samples: Optional[int] = typer.Option(
        None,
        "--num-samples",
        "-n",
        help="Number of samples to download (None for all)",
    ),
    format: str = typer.Option(
        "json",
        "--format",
        "-f",
        help="Output format (json, jsonl, parquet)",
    ),
):
    """Download dataset from Hugging Face."""

    console.print("\n[bold cyan]=== Dataset Downloader ===[/bold cyan]\n")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Generate output filename
    dataset_name = dataset.replace("/", "_")
    if num_samples:
        output_file = output_path / f"{dataset_name}_{num_samples}.{format}"
    else:
        output_file = output_path / f"{dataset_name}.{format}"

    console.print(f"[cyan]Dataset: {dataset}[/cyan]")
    console.print(f"[cyan]Split: {split}[/cyan]")
    console.print(f"[cyan]Output: {output_file}[/cyan]")
    console.print(f"[cyan]Samples: {num_samples or 'All'}[/cyan]")
    console.print()

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Loading dataset...", total=None)

            # Load dataset
            try:
                # Try Hugging Face
                ds = load_dataset(dataset, split=split)
            except Exception:
                # Try local file
                console.print("[yellow]Trying to load as local file...[/yellow]")
                ds = load_dataset("json", data_files=dataset, split=split)

            progress.update(task, description=f"Loaded {len(ds)} samples")

            # Limit samples if specified
            if num_samples and len(ds) > num_samples:
                ds = ds.select(range(num_samples))
                progress.update(task, description=f"Limited to {num_samples} samples")

            # Save dataset
            progress.update(task, description=f"Saving to {output_file}...")

            if format == "json":
                ds.to_json(output_file)
            elif format == "jsonl":
                ds.to_json(output_file, orient="records", lines=True)
            elif format == "parquet":
                ds.to_parquet(output_file)
            else:
                console.print(f"[red]Unknown format: {format}[/red]")
                raise typer.Exit(1)

        console.print(f"\n[bold green]✓ Dataset saved to: {output_file}[/bold green]")
        console.print(f"[green]Size: {len(ds)} samples[/green]\n")

    except Exception as e:
        console.print(f"\n[red]✗ Download failed: {e}[/red]\n")
        raise typer.Exit(1)


@app.command()
def list_datasets():
    """List available built-in datasets."""

    console.print("\n[bold cyan]=== Available Datasets ===[/bold cyan]\n")

    datasets = [
        {
            "name": "yahma/alpaca-cleaned",
            "description": "Alpaca instruction dataset (52K instructions)",
            "type": "SFT",
            "size": "52K",
        },
        {
            "name": "tatsu-lab/alpaca",
            "description": "Original Alpaca dataset",
            "type": "SFT",
            "size": "52K",
        },
        {
            "name": "HuggingFaceH4/argilla-dpo-mix-7k",
            "description": "DPO preference pairs",
            "type": "DPO",
            "size": "7K",
        },
        {
            "name": "custom",
            "description": "Your custom data in data/custom/",
            "type": "SFT/DPO",
            "size": "Variable",
        },
    ]

    from rich.table import Table

    table = Table(title="Built-in Datasets")
    table.add_column("Dataset", style="cyan")
    table.add_column("Description", style="white")
    table.add_column("Type", style="yellow")
    table.add_column("Size", style="green")

    for ds in datasets:
        table.add_row(ds["name"], ds["description"], ds["type"], ds["size"])

    console.print(table)
    console.print()


if __name__ == "__main__":
    app()
