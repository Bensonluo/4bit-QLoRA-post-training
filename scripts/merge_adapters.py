#!/usr/bin/env python3
"""Merge MULTIPLE LoRA adapters into one model (TIES / DARE / SVD family).

Combines two or more saved adapters that share the same base model using peft's
weighted-adapter merging, then merges the combined adapter into the base.

Examples:
    # Two adapters, TIES with 20% density kept
    python scripts/merge_adapters.py \\
        --adapter-dirs outputs/sft/run-a --adapter-dirs outputs/sft/run-b \\
        --output-dir outputs/merged/combined \\
        --combination-type ties --density 0.2

    # DARE with explicit weights and SVD compression
    python scripts/merge_adapters.py \\
        -a outputs/sft/run-a -a outputs/sft/run-b -a outputs/sft/run-c \\
        -o outputs/merged/combined \\
        --weights 0.5,0.3,0.2 \\
        --combination-type dare_ties_svd --density 0.5 --svd-rank 16
"""

import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.merger import WEIGHTED_MERGE_TYPES, merge_adapters_weighted

app = typer.Typer(
    name="merge-adapters",
    help="Merge multiple LoRA adapters into the base model (TIES/DARE/SVD)",
    add_completion=False,
)
console = Console()


@app.command()
def main(
    adapter_dirs: list[str] = typer.Option(
        ...,
        "--adapter-dirs",
        "-a",
        help="Adapter directories (repeat the flag, 2+ required, same base model)",
    ),
    output_dir: str = typer.Option(
        ...,
        "--output-dir",
        "-o",
        help="Where to write the merged model + tokenizer",
    ),
    weights: str = typer.Option(
        None,
        "--weights",
        "-w",
        help="Comma-separated per-adapter weights, e.g. '0.5,0.3,0.2' (default: equal)",
    ),
    combination_type: str = typer.Option(
        "ties",
        "--combination-type",
        "-c",
        help=f"One of: {', '.join(sorted(WEIGHTED_MERGE_TYPES))}",
    ),
    density: float = typer.Option(
        None,
        "--density",
        help="Fraction of delta entries kept, (0,1]. Required for ties/dare/magnitude_prune.",
    ),
    svd_rank: int = typer.Option(
        None, "--svd-rank", help="Output rank for *_svd types (default: max input rank)"
    ),
    dtype: str = typer.Option(
        "bfloat16", "--dtype", help="Merged weights precision (bfloat16/float16/float32)"
    ),
    base_model_name: str = typer.Option(
        None, "--base-model-name", help="Override the base model id"
    ),
) -> None:
    """Merge multiple adapters via peft's weighted merging."""
    weight_list: list[float] | None = None
    if weights:
        weight_list = [float(x) for x in weights.split(",")]

    out = merge_adapters_weighted(
        adapter_dirs=list(adapter_dirs),
        output_dir=output_dir,
        weights=weight_list,
        combination_type=combination_type,
        density=density,
        svd_rank=svd_rank,
        dtype=dtype,
        base_model_name=base_model_name,
    )
    console.print(Panel.fit(f"[green]✓ Merged model saved to:[/green] {out}"))


if __name__ == "__main__":
    app()
