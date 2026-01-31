#!/usr/bin/env python3
"""CLI script to merge LoRA adapters into base model.

Usage:
    python scripts/merge_lora.py --base-model Qwen/Qwen2.5-1.5B-Instruct --adapter-path ./outputs/sft
"""

import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.base import ModelConfig
from src.models import load_model_and_tokenizer, merge_lora_into_base
from src.utils import console

app = typer.Typer(
    name="merge-lora",
    help="Merge LoRA adapters into base model",
    add_completion=False,
)


@app.command()
def main(
    base_model: str = typer.Option(
        ...,
        "--base-model",
        "-m",
        help="Base model name or path",
    ),
    adapter_path: str = typer.Option(
        ...,
        "--adapter-path",
        "-a",
        help="Path to LoRA adapters",
    ),
    output_path: str = typer.Option(
        "./outputs/merged",
        "--output",
        "-o",
        help="Output path for merged model",
    ),
    quantization_bits: int = typer.Option(
        4,
        "--quantization-bits",
        "-q",
        help="Quantization bits for loading base model",
    ),
    export_gguf: bool = typer.Option(
        False,
        "--export-gguf",
        help="Export to GGUF format (requires llama.cpp)",
    ),
    gguf_quantization: str = typer.Option(
        "q4_k_m",
        "--gguf-quant",
        help="GGUF quantization type",
    ),
):
    """Merge LoRA adapters into base model."""

    console.print("\n[bold cyan]=== LoRA Merger ===[/bold cyan]\n")

    # Verify adapter path exists
    adapter_dir = Path(adapter_path)
    if not adapter_dir.exists():
        console.print(f"[red]✗ Adapter path not found: {adapter_path}[/red]")
        raise typer.Exit(1)

    # Load base model
    model_config = ModelConfig(
        name=base_model,
        quantization_bits=quantization_bits,
    )

    console.print(f"[cyan]Loading base model: {base_model}[/cyan]")
    model, tokenizer = load_model_and_tokenizer(model_config)

    # Merge adapters
    try:
        merge_lora_into_base(
            model=model,
            adapter_path=adapter_path,
            output_path=output_path,
            tokenizer=tokenizer,
        )

        console.print(f"\n[bold green]✓ Merged model saved to: {output_path}[/bold green]\n")

        # Export to GGUF if requested
        if export_gguf:
            from src.models import export_to_gguf

            gguf_path = f"{output_path}/model.gguf"
            export_to_gguf(
                model_path=output_path,
                output_path=gguf_path,
                quantization=gguf_quantization,
            )

    except Exception as e:
        console.print(f"\n[red]✗ Merge failed: {e}[/red]\n")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
