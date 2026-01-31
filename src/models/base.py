"""Base model classes and utilities."""

from typing import Optional, Tuple

from transformers import PreTrainedModel, PreTrainedTokenizer


class BaseModelHandler:
    """Base handler for model operations."""

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
    ):
        """Initialize model handler.

        Args:
            model: Pre-trained model
            tokenizer: Pre-trained tokenizer
        """
        self.model = model
        self.tokenizer = tokenizer

    def save_model(self, output_dir: str) -> None:
        """Save model and tokenizer.

        Args:
            output_dir: Directory to save model
        """
        from pathlib import Path

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
    ) -> "BaseModelHandler":
        """Load model from path.

        Args:
            model_path: Path to model directory

        Returns:
            BaseModelHandler instance
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model = AutoModelForCausalLM.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        return cls(model, tokenizer)


def get_model_size(model: PreTrainedModel) -> Tuple[int, str]:
    """Get model size in parameters.

    Args:
        model: Pre-trained model

    Returns:
        Tuple of (parameter_count, human_readable_size)
    """
    param_count = sum(p.numel() for p in model.parameters())

    if param_count >= 1e9:
        size_str = f"{param_count / 1e9:.2f}B"
    elif param_count >= 1e6:
        size_str = f"{param_count / 1e6:.2f}M"
    elif param_count >= 1e3:
        size_str = f"{param_count / 1e3:.2f}K"
    else:
        size_str = str(param_count)

    return param_count, size_str


def print_model_info(model: PreTrainedModel, tokenizer: PreTrainedTokenizer) -> None:
    """Print model information.

    Args:
        model: Pre-trained model
        tokenizer: Pre-trained tokenizer
    """
    from rich.console import Console

    console = Console()

    param_count, size_str = get_model_size(model)
    vocab_size = tokenizer.vocab_size

    console.print("\n[bold cyan]Model Information[/bold cyan]")
    console.print(f"  Parameters: {size_str} ({param_count:,})")
    console.print(f"  Vocabulary: {vocab_size:,}")
    console.print(f"  Model Type: {type(model).__name__}")
    console.print(f"  Device: {next(model.parameters()).device}")

    if hasattr(model.config, "quantization_config"):
        console.print(f"  Quantization: {model.config.quantization_config}")
