"""Utilities for merging LoRA adapters into base models."""

from pathlib import Path

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

    torch_dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }.get(dtype, torch.bfloat16)

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
        console.print("[cyan]Loading AutoPeftModel (base resolved from adapter_config.json)...[/cyan]")
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
        merged_model = model.merge_and_unload()
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
