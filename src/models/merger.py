"""Utilities for merging LoRA adapters into base models."""

from pathlib import Path
from typing import Optional

from peft import PeftModel
from transformers import PreTrainedModel, PreTrainedTokenizer

from src.utils.logging import console


def merge_lora_into_base(
    model: PreTrainedModel,
    adapter_path: str,
    output_path: str,
    tokenizer: Optional[PreTrainedTokenizer] = None,
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

    console.print(f"\n[bold cyan]Exporting to GGUF format[/bold cyan]")
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
    from transformers import AutoModelForCausalLM, AutoTokenizer

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
