"""Utilities for merging LoRA adapters into base models."""

from pathlib import Path
from typing import Any

from peft import PeftModel
from transformers import PreTrainedModel, PreTrainedTokenizer

from src.utils.logging import console


def merge_adapter_to_dir(
    adapter_dir: str,
    output_dir: str,
    base_model_name: str | None = None,
    dtype: str = "bfloat16",
) -> str:
    """Merge a saved LoRA adapter directory into the base model, writing the result to disk.

    This is the disk-based counterpart to `merge_lora_into_base` (which operates on an
    in-memory model). Use this after training has finished and the adapter has been
    saved — e.g. to prepare a model for MLflow Model Registry.

    The base model name is resolved from the adapter's `adapter_config.json` unless
    `base_model_name` is given explicitly.

    Args:
        adapter_dir: Directory containing the saved adapter (adapter_config.json + weights).
        output_dir: Where to write the merged model + tokenizer.
        base_model_name: Override the base model id (otherwise read from adapter config).
        dtype: Precision of the merged weights ("bfloat16" / "float16" / "float32").

    Returns:
        The absolute output_dir (suitable to pass straight into mlflow.log_artifacts).
    """
    import torch

    console.print("\n[bold cyan]Merging LoRA adapter into base model[/bold cyan]")
    console.print(f"  Adapter: {adapter_dir}")
    console.print(f"  Output:  {output_dir}")
    console.print(f"  Dtype:   {dtype}")

    torch_dtype_map: dict[str, Any] = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    torch_dtype = torch_dtype_map.get(dtype, torch.bfloat16)

    try:
        from peft import AutoPeftModelForCausalLM
    except ImportError as e:
        raise RuntimeError(
            "AutoPeftModelForCausalLM requires `peft>=0.7`. Install with: pip install -U peft"
        ) from e

    # AutoPeftModel reads the base model id from adapter_config.json automatically.
    load_kwargs = {"torch_dtype": torch_dtype}
    if base_model_name:
        # Explicit override — AutoPeftModel still needs the adapter dir as the first arg.
        load_kwargs["adapter_dir"] = adapter_dir
        # Load base separately then attach adapter to honor the override.
        from peft import PeftModel as _PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        console.print(f"[cyan]Loading base model: {base_model_name}[/cyan]")
        base = AutoModelForCausalLM.from_pretrained(base_model_name, torch_dtype=torch_dtype)
        console.print(f"[cyan]Attaching adapter from: {adapter_dir}[/cyan]")
        model = _PeftModel.from_pretrained(base, adapter_dir)
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    else:
        console.print(
            "[cyan]Loading AutoPeftModel (base resolved from adapter_config.json)...[/cyan]"
        )
        model = AutoPeftModelForCausalLM.from_pretrained(adapter_dir, torch_dtype=torch_dtype)
        # Tokenizer lives alongside the adapter (trainer.save_model saves it there).
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(adapter_dir)

    # Merge adapter weights into the base and drop the PEFT wrapper.
    if isinstance(model, PeftModel):
        console.print("[cyan]Merging adapters...[/cyan]")
        model = model.merge_and_unload()
        console.print("[green]✓ Adapters merged[/green]")
    else:
        console.print("[yellow]⚠ Model is not a PeftModel — saving as-is[/yellow]")

    out = Path(output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    console.print(f"[cyan]Saving merged model to: {out}[/cyan]")
    model.save_pretrained(out, safe_serialization=True)
    tokenizer.save_pretrained(out)

    console.print(f"[green]✓ Merged model saved to: {out}[/green]\n")
    return str(out)


def merge_lora_into_base(
    model: PreTrainedModel,
    adapter_path: str,
    output_path: str,
    tokenizer: PreTrainedTokenizer | None = None,
) -> PreTrainedModel:
    """Merge LoRA adapters into base model.

    Args:
        model: Base model with LoRA adapters
        adapter_path: Path to LoRA adapters (if different from model's adapters)
        output_path: Path to save merged model
        tokenizer: Optional tokenizer to save with model

    Returns:
        Merged model
    """
    console.print("\n[bold cyan]Merging LoRA adapters[/bold cyan]")

    # Load adapters if path provided
    if adapter_path and hasattr(model, "load_adapter"):
        console.print(f"[cyan]Loading adapters from: {adapter_path}[/cyan]")
        model.load_adapter(adapter_path)

    # Merge adapters
    console.print("[cyan]Merging adapters...[/cyan]")

    if isinstance(model, PeftModel):
        # Explicit annotation: merge_and_unload's return type depends on the
        # peft version's stubs, and degrades to Any when mypy analyzes a
        # partial import graph (e.g. pre-commit's file-list mode).
        merged_model: PreTrainedModel = model.merge_and_unload()
        console.print("[green]✓ Adapters merged[/green]")
    else:
        console.print("[yellow]⚠ Model is not a PeftModel, skipping merge[/yellow]")
        merged_model = model

    # Save merged model
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    console.print(f"[cyan]Saving merged model to: {output_path}[/cyan]")
    merged_model.save_pretrained(output_dir)

    # Save tokenizer if provided
    if tokenizer:
        tokenizer.save_pretrained(output_dir)

    console.print(f"[green]✓ Merged model saved to: {output_path}[/green]\n")

    return merged_model


def export_to_gguf(
    model_path: str,
    output_path: str,
    quantization: str = "q4_k_m",
) -> None:
    """Export model to GGUF format for llama.cpp.

    This requires llama.cpp to be installed and accessible.

    Args:
        model_path: Path to merged model
        output_path: Path for output GGUF file
        quantization: GGUF quantization type (q4_k_m, q5_k_m, q8_0, etc.)
    """
    import subprocess

    console.print("\n[bold cyan]Exporting to GGUF format[/bold cyan]")
    console.print(f"  Model: {model_path}")
    console.print(f"  Output: {output_path}")
    console.print(f"  Quantization: {quantization}\n")

    # Convert to GGUF
    convert_cmd = [
        "python",
        "llama.cpp/convert.py",
        model_path,
        "--outfile",
        output_path,
        "--outtype",
        quantization,
    ]

    console.print(f"[cyan]Running: {' '.join(convert_cmd)}[/cyan]")

    try:
        subprocess.run(convert_cmd, check=True)
        console.print(f"[green]✓ GGUF model exported to: {output_path}[/green]")
    except subprocess.CalledProcessError as e:
        console.print(f"[red]✗ GGUF export failed: {e}[/red]")
        console.print("[yellow]Note: llama.cpp required for GGUF export[/yellow]")


def load_merged_model(
    model_path: str,
) -> PreTrainedModel:
    """Load a merged model.

    Args:
        model_path: Path to merged model directory

    Returns:
        Loaded merged model
    """
    from transformers import AutoModelForCausalLM

    console.print(f"[cyan]Loading merged model from: {model_path}[/cyan]")

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype="auto",
    )

    console.print("[green]✓ Merged model loaded[/green]")

    return model


def compare_models_before_after(
    base_model: PreTrainedModel,
    tuned_model: PreTrainedModel,
) -> None:
    """Compare base and fine-tuned models.

    Args:
        base_model: Base model before fine-tuning
        tuned_model: Model after fine-tuning
    """
    from rich.table import Table

    console.print("\n[bold cyan]Model Comparison[/bold cyan]\n")

    table = Table(title="Model Parameters")
    table.add_column("Metric", style="cyan")
    table.add_column("Base Model", style="white")
    table.add_column("Tuned Model", style="white")

    # Get parameter counts
    base_params = sum(p.numel() for p in base_model.parameters())
    tuned_params = sum(p.numel() for p in tuned_model.parameters())

    table.add_row("Total Parameters", f"{base_params:,}", f"{tuned_params:,}")

    # Check if parameters are trainable
    base_trainable = sum(p.numel() for p in base_model.parameters() if p.requires_grad)
    tuned_trainable = sum(p.numel() for p in tuned_model.parameters() if p.requires_grad)

    table.add_row("Trainable Parameters", f"{base_trainable:,}", f"{tuned_trainable:,}")

    console.print(table)


if __name__ == "__main__":
    # Test merge functionality
    console.print("[yellow]Merge utilities loaded[/yellow]")
    console.print("[cyan]Use this module after training to merge LoRA adapters[/cyan]")


# combination_type values accepted by peft 0.19's LoraModel.add_weighted_adapter.
WEIGHTED_MERGE_TYPES = frozenset(
    {
        "svd",
        "linear",
        "cat",
        "ties",
        "ties_svd",
        "dare_ties",
        "dare_linear",
        "dare_ties_svd",
        "dare_linear_svd",
        "magnitude_prune",
        "magnitude_prune_svd",
    }
)

# Methods that prune via `density` — required there, ignored elsewhere.
_DENSITY_REQUIRED = frozenset(
    {
        "ties",
        "ties_svd",
        "dare_ties",
        "dare_linear",
        "dare_ties_svd",
        "dare_linear_svd",
        "magnitude_prune",
        "magnitude_prune_svd",
    }
)


def merge_adapters_weighted(
    adapter_dirs: list[str],
    output_dir: str,
    weights: list[float] | None = None,
    combination_type: str = "ties",
    density: float | None = None,
    majority_sign_method: str = "total",
    svd_rank: int | None = None,
    svd_clamp: float | None = None,
    dtype: str = "bfloat16",
    base_model_name: str | None = None,
) -> str:
    """Merge MULTIPLE saved LoRA adapters into the base model with a weighted merge.

    Loads the (shared) base full-precision, attaches every adapter, combines them
    via peft's ``add_weighted_adapter`` (TIES / DARE / magnitude-prune / SVD
    family), then merges the combined adapter and writes a self-contained model
    directory.

    Method guide (peft combination_type):
        - ``ties`` / ``ties_svd`` — TIES (arXiv:2306.01708): magnitude-prune to
          ``density``, resolve sign conflicts by majority, average the rest.
          The reference for combining task adapters without interference.
        - ``dare_ties`` / ``dare_linear`` (+ ``_svd``) — DARE
          (arXiv:2311.03015): randomly drops ``1 - density`` of the delta and
          rescales, then (optionally) TIES-merges. Best when adapters come from
          very different tasks.
        - ``magnitude_prune`` — drop smallest-magnitude entries, no sign
          resolution.
        - ``linear`` — plain weighted sum (no pruning). ``cat`` concatenates
          ranks (output rank = sum — watch VRAM). ``svd`` variants compress the
          combined delta back to ``svd_rank`` so the merged adapter stays small.

    Memory note: merging requires a FULL-PRECISION base (peft cannot merge into
    4-bit weights). Peak RAM ≈ fp16/bf16 model size + adapters.

    Args:
        adapter_dirs: Two or more adapter directories, each with
            adapter_config.json. All must name the SAME base model.
        output_dir: Where to write the merged model + tokenizer.
        weights: Per-adapter weights (may be negative to subtract an effect).
            None = equal weights.
        combination_type: One of WEIGHTED_MERGE_TYPES.
        density: Fraction of delta entries KEPT, in (0, 1]. Required for the
            ties/dare/magnitude_prune family. Papers use 0.05-0.5 (TIES) and
            0.1-0.9 (DARE).
        majority_sign_method: "total" (default) or "frequency" — sign-conflict
            resolution for the ties family.
        svd_rank: Output rank for *_svd types. None = max input rank.
        svd_clamp: Optional SVD output quantile clamp.
        dtype: Precision of merged weights.
        base_model_name: Override the base id (else read from adapter configs).

    Returns:
        The absolute output_dir.
    """
    import json

    import torch

    if len(adapter_dirs) < 2:
        raise ValueError(f"Need at least 2 adapters to merge, got {len(adapter_dirs)}")
    if combination_type not in WEIGHTED_MERGE_TYPES:
        raise ValueError(
            f"combination_type {combination_type!r} not in sorted({sorted(WEIGHTED_MERGE_TYPES)})"
        )
    if weights is not None and len(weights) != len(adapter_dirs):
        raise ValueError(
            f"weights length ({len(weights)}) must match adapter_dirs ({len(adapter_dirs)})"
        )
    if combination_type in _DENSITY_REQUIRED and density is None:
        raise ValueError(
            f"combination_type={combination_type!r} requires density (0 < density <= 1)"
        )
    if density is not None and not 0 < density <= 1:
        raise ValueError(f"density must be in (0, 1], got {density}")
    if majority_sign_method not in ("total", "frequency"):
        raise ValueError("majority_sign_method must be 'total' or 'frequency'")

    # All adapters must target the same base — read each adapter_config.json.
    base_ids: list[str] = []
    for d in adapter_dirs:
        cfg_path = Path(d) / "adapter_config.json"
        if not cfg_path.exists():
            raise FileNotFoundError(f"No adapter_config.json in {d}")
        base_ids.append(json.loads(cfg_path.read_text())["base_model_name_or_path"])
    base_id = base_model_name or base_ids[0]
    mismatches = {d: b for d, b in zip(adapter_dirs, base_ids) if b != base_ids[0]}
    if mismatches and not base_model_name:
        raise ValueError(
            "All adapters must share one base model. "
            f"Conflicts: {mismatches}; expected {base_ids[0]!r}"
        )

    console.print("\n[bold cyan]Weighted-merging LoRA adapters[/bold cyan]")
    console.print(f"  Adapters: {adapter_dirs}")
    console.print(f"  Weights:  {weights or 'equal'}")
    console.print(f"  Method:   {combination_type}" + (f" (density={density})" if density else ""))

    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch_dtype_map: dict[str, Any] = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    torch_dtype = torch_dtype_map.get(dtype, torch.bfloat16)

    console.print(f"[cyan]Loading base model: {base_id}[/cyan]")
    base = AutoModelForCausalLM.from_pretrained(base_id, torch_dtype=torch_dtype)

    adapter_names = [f"adapter_{i}" for i in range(len(adapter_dirs))]
    console.print(f"[cyan]Attaching adapter 0: {adapter_dirs[0]}[/cyan]")
    model = PeftModel.from_pretrained(base, adapter_dirs[0], adapter_name=adapter_names[0])
    for name, d in zip(adapter_names[1:], adapter_dirs[1:]):
        console.print(f"[cyan]Attaching {name}: {d}[/cyan]")
        model.load_adapter(d, adapter_name=name)

    console.print(f"[cyan]Combining via {combination_type}...[/cyan]")
    # peft requires a concrete weights list — None means equal weights here.
    resolved_weights = weights if weights is not None else [1.0] * len(adapter_dirs)
    model.add_weighted_adapter(
        adapters=adapter_names,
        weights=resolved_weights,
        adapter_name="merged",
        combination_type=combination_type,
        density=density,
        majority_sign_method=majority_sign_method,
        svd_rank=svd_rank,
        svd_clamp=svd_clamp,
    )
    model.set_adapter("merged")
    model = model.merge_and_unload(adapter_names=["merged"])
    console.print("[green]✓ Adapters combined and merged[/green]")

    out = Path(output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out, safe_serialization=True)
    try:
        tokenizer = AutoTokenizer.from_pretrained(adapter_dirs[0])
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(base_id)
    tokenizer.save_pretrained(out)

    console.print(f"[green]✓ Merged model saved to {out}[/green]\n")
    return str(out)
