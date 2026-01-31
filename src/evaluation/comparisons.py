"""Model comparison utilities."""

from typing import Dict

from datasets import Dataset
from transformers import PreTrainedModel, PreTrainedTokenizer

from src.evaluation.metrics import compute_perplexity, compute_accuracy
from src.utils import console


def compare_models(
    model_path: str,
    base_model_path: str,
    dataset: Dataset,
    tokenizer: PreTrainedTokenizer,
) -> Dict[str, Dict]:
    """Compare fine-tuned model with base model.

    Args:
        model_path: Path to fine-tuned model
        base_model_path: Path to base model
        dataset: Dataset to compare on
        tokenizer: Tokenizer to use

    Returns:
        Dictionary with comparison metrics
    """
    console.print("\n[bold cyan]=== Model Comparison ===[/bold cyan]\n")

    # Load models
    console.print(f"[cyan]Loading fine-tuned model: {model_path}[/cyan]")
    from src.models import load_merged_model

    try:
        tuned_model, _ = load_merged_model(model_path)
    except Exception:
        # Try loading with adapters
        from src.models import load_model_and_tokenizer
        from config.base import ModelConfig
        config = ModelConfig(name=model_path)
        tuned_model, _ = load_model_and_tokenizer(config)

    console.print(f"[cyan]Loading base model: {base_model_path}[/cyan]")
    base_model, _ = load_merged_model(base_model_path)

    comparison = {}

    # Compare perplexity
    console.print("\n[cyan]Computing perplexity...[/cyan]")
    tuned_ppl = compute_perplexity(tuned_model, dataset, tokenizer)
    base_ppl = compute_perplexity(base_model, dataset, tokenizer)

    ppl_improvement = (base_ppl - tuned_ppl) / base_ppl * 100

    comparison["perplexity"] = {
        "tuned": f"{tuned_ppl:.2f}",
        "base": f"{base_ppl:.2f}",
        "improvement": f"{ppl_improvement:+.1f}%",
    }

    # Compare accuracy (optional, can be slow)
    # tuned_acc = compute_accuracy(tuned_model, dataset, tokenizer)
    # base_acc = compute_accuracy(base_model, dataset, tokenizer)

    console.print(f"\n[green]✓ Comparison complete[/green]")

    return comparison


def side_by_side_generation(
    model_path: str,
    base_model_path: str,
    prompts: list,
    tokenizer: PreTrainedTokenizer,
    max_length: int = 256,
) -> None:
    """Generate side-by-side comparisons.

    Args:
        model_path: Path to fine-tuned model
        base_model_path: Path to base model
        prompts: List of prompts
        tokenizer: Tokenizer
        max_length: Maximum generation length
    """
    from src.models import load_merged_model

    # Load models
    tuned_model, _ = load_merged_model(model_path)
    base_model, _ = load_merged_model(base_model_path)

    tuned_model.eval()
    base_model.eval()

    for i, prompt in enumerate(prompts, 1):
        from rich.table import Table
        from rich.panel import Panel

        console.print(f"\n[bold cyan]Prompt {i}:[/bold cyan] {prompt}\n")

        # Generate with base model
        inputs = tokenizer(prompt, return_tensors="pt").to(base_model.device)

        with torch.no_grad():
            base_outputs = base_model.generate(
                **inputs,
                max_length=max_length,
                temperature=0.7,
                do_sample=True,
            )

        base_response = tokenizer.decode(base_outputs[0], skip_special_tokens=True)

        # Generate with tuned model
        with torch.no_grad():
            tuned_outputs = tuned_model.generate(
                **inputs,
                max_length=max_length,
                temperature=0.7,
                do_sample=True,
            )

        tuned_response = tokenizer.decode(tuned_outputs[0], skip_special_tokens=True)

        # Display side by side
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Base Model", style="yellow")
        table.add_column("Fine-tuned Model", style="green")

        table.add_row(
            base_response[:500] + "..." if len(base_response) > 500 else base_response,
            tuned_response[:500] + "..." if len(tuned_response) > 500 else tuned_response,
        )

        console.print(table)
