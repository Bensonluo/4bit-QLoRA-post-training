#!/usr/bin/env python3
"""Merge a saved LoRA adapter into the base model — standalone, no training needed.

Use this after training (or on a previously-trained adapter checkpoint) to produce
a self-contained merged model directory that can be served or registered.

The base model id is resolved automatically from adapter_config.json unless you
override with --base-model-name.

Examples:
    # Auto-resolve base from adapter config
    python scripts/merge_adapter.py \\
        --adapter-dir outputs/sft/run-xxx \\
        --output-dir outputs/merged/run-xxx

    # Explicit base model + fp16
    python scripts/merge_adapter.py \\
        --adapter-dir outputs/sft/run-xxx \\
        --output-dir outputs/merged/run-xxx \\
        --base-model-name Qwen/Qwen3-1.7B \\
        --dtype float16
"""

import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.merger import merge_adapter_to_dir

app = typer.Typer(
    name="merge-adapter",
    help="Merge a LoRA adapter into the base model (standalone, no training)",
    add_completion=False,
)
console = Console()


@app.command()
def main(
    adapter_dir: str = typer.Option(
        ...,
        "--adapter-dir",
        "-a",
        help="Directory containing the saved adapter (adapter_config.json + weights)",
    ),
    output_dir: str = typer.Option(
        ...,
        "--output-dir",
        "-o",
        help="Where to write the merged model + tokenizer",
    ),
    base_model_name: str = typer.Option(
        None,
        "--base-model-name",
        "-m",
        help="Override base model id (default: read from adapter_config.json)",
    ),
    dtype: str = typer.Option(
        "bfloat16",
        "--dtype",
        "-d",
        help="Precision of merged weights: bfloat16 / float16 / float32",
    ),
):
    """Merge a LoRA adapter into the base model."""

    if not Path(adapter_dir).exists():
        console.print(f"[red]✗ Adapter directory not found: {adapter_dir}[/red]")
        raise typer.Exit(1)

    if not (Path(adapter_dir) / "adapter_config.json").exists():
        console.print(
            f"[red]✗ adapter_config.json not found in {adapter_dir}. "
            "Is this a PEFT adapter directory?[/red]"
        )
        raise typer.Exit(1)

    console.print(Panel.fit(
        "[bold cyan]LoRA Adapter Merge[/bold cyan]\n"
        f"Adapter: {adapter_dir}\n"
        f"Output:  {output_dir}\n"
        f"Base:    {base_model_name or '(auto from adapter_config.json)'}\n"
        f"Dtype:   {dtype}",
        border_style="cyan",
    ))

    merged_path = merge_adapter_to_dir(
        adapter_dir=adapter_dir,
        output_dir=output_dir,
        base_model_name=base_model_name,
        dtype=dtype,
    )

    console.print(f"\n[bold green]✓ Merge complete![/bold green]")
    console.print(f"[cyan]Merged model at: {merged_path}[/cyan]")
    console.print(
        "\n[dim]Next: register with `python scripts/registry_cli.py register "
        f"--model-dir {merged_path} --name <model-name>`[/dim]\n"
    )


if __name__ == "__main__":
    app()
