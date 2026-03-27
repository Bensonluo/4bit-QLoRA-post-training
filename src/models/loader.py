"""Model loading utilities with platform-aware quantization support.

Supports:
- NVIDIA GPU (CUDA): bitsandbytes 4-bit/8-bit quantization
- Apple Silicon (MPS): bf16/fp16 without quantization
"""

from typing import Optional, Tuple

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)

from config.base import ModelConfig
from src.models.base import print_model_info
from src.utils.logging import console
from src.utils.platform_utils import get_platform, get_torch_dtype

# bitsandbytes is only available on CUDA
_BITSANDBYTES_AVAILABLE = False
try:
    from transformers import BitsAndBytesConfig
    _BITSANDBYTES_AVAILABLE = True
except ImportError:
    pass


def load_tokenizer(
    model_name: str,
    trust_remote_code: bool = True,
) -> PreTrainedTokenizer:
    """Load tokenizer for model.

    Args:
        model_name: Hugging Face model name or path
        trust_remote_code: Whether to trust remote code

    Returns:
        Loaded tokenizer
    """
    console.print(f"[cyan]Loading tokenizer: {model_name}[/cyan]")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=trust_remote_code,
    )

    # Ensure padding token is set
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
            console.print("[yellow]⚠ Pad token set to eos_token[/yellow]")
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            console.print("[yellow]⚠ Added [PAD] token[/yellow]")

    console.print(f"[green]✓ Tokenizer loaded (vocab_size={tokenizer.vocab_size})[/green]")

    return tokenizer


def _get_quantization_config(
    quantization_bits: int = 4,
):
    """Get bitsandbytes quantization config (CUDA only).

    Args:
        quantization_bits: Number of bits (4 or 8)

    Returns:
        BitsAndBytesConfig instance
    """
    if not _BITSANDBYTES_AVAILABLE:
        raise RuntimeError(
            "bitsandbytes is not available. "
            "Quantization requires an NVIDIA GPU with bitsandbytes installed."
        )

    if quantization_bits == 4:
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    elif quantization_bits == 8:
        return BitsAndBytesConfig(
            load_in_8bit=True,
        )
    else:
        raise ValueError(f"Unsupported quantization: {quantization_bits} bits")


def load_model(
    config: ModelConfig,
) -> PreTrainedModel:
    """Load model with platform-aware configuration.

    On CUDA: uses bitsandbytes quantization if configured.
    On MPS (Apple Silicon): loads in bf16/fp16 without quantization.
    On CPU: loads in float32.

    Args:
        config: Model configuration

    Returns:
        Loaded model
    """
    platform_info = get_platform()

    console.print(f"\n[bold cyan]Loading model: {config.name}[/bold cyan]")
    console.print(f"  Platform: {platform_info.description}")
    console.print(f"  Flash Attention: {config.use_flash_attention}")
    console.print(f"  Data Type: {config.torch_dtype}\n")

    # Resolve dtype for platform
    torch_dtype = get_torch_dtype(config.torch_dtype, platform_info)

    # Prepare model loading arguments
    model_kwargs = {
        "trust_remote_code": config.trust_remote_code,
        "torch_dtype": torch_dtype,
    }

    # Platform-specific device and quantization
    if platform_info.is_cuda:
        model_kwargs["device_map"] = config.device_map

        if config.quantization_bits in (4, 8):
            model_kwargs["quantization_config"] = _get_quantization_config(
                config.quantization_bits
            )
            console.print(
                f"[green]✓ {config.quantization_bits}-bit quantization enabled (CUDA/bitsandbytes)[/green]"
            )
        else:
            console.print("[cyan]Loading without quantization (full precision on CUDA)[/cyan]")

    elif platform_info.is_mps:
        # MPS: no device_map, no quantization — load then move to MPS
        console.print("[cyan]Apple Silicon detected — loading in bf16 without quantization[/cyan]")

    else:
        console.print("[yellow]⚠ No GPU detected — loading on CPU (slow)[/yellow]")

    # Flash Attention (CUDA only, not available on MPS)
    if config.use_flash_attention and platform_info.is_cuda:
        model_kwargs["attn_implementation"] = "flash_attention_2"
        console.print("[green]✓ Flash Attention 2 enabled[/green]")
    elif config.use_flash_attention:
        console.print("[yellow]⚠ Flash Attention not available on this platform, using default[/yellow]")

    # Load model
    console.print("[cyan]Loading model (this may take a while)...[/cyan]")

    model = AutoModelForCausalLM.from_pretrained(
        config.name,
        **model_kwargs,
    )

    # Move to MPS after loading (MPS doesn't support device_map)
    if platform_info.is_mps:
        model = model.to("mps")

    console.print(f"[green]✓ Model loaded successfully on {platform_info.device}![/green]\n")

    return model


def load_model_and_tokenizer(
    config: ModelConfig,
) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """Load model and tokenizer with platform-aware configuration.

    This is the main entry point for loading models for training.

    Args:
        config: Model configuration

    Returns:
        Tuple of (model, tokenizer)
    """
    platform_info = get_platform()
    console.print(f"[bold]Platform: {platform_info.description}[/bold]\n")

    # Load tokenizer first
    tokenizer = load_tokenizer(
        model_name=config.name,
        trust_remote_code=config.trust_remote_code,
    )

    # Load model
    model = load_model(config)

    # Print model info
    print_model_info(model, tokenizer)

    return model, tokenizer


def load_base_model_for_dpo(
    config: ModelConfig,
) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """Load base model for DPO training.

    DPO requires a reference model that stays frozen during training.
    This function loads a model that can be used as a reference.

    Args:
        config: Model configuration

    Returns:
        Tuple of (model, tokenizer)
    """
    console.print("[yellow]Loading base model for DPO (reference model)[/yellow]")

    # Load with same config but ensure it's frozen
    model, tokenizer = load_model_and_tokenizer(config)

    # Freeze all parameters
    for param in model.parameters():
        param.requires_grad = False

    console.print("[green]✓ Reference model loaded and frozen[/green]")

    return model, tokenizer


if __name__ == "__main__":
    # Test model loading
    test_config = ModelConfig(
        name="Qwen/Qwen2.5-0.5B-Instruct",  # Smaller model for testing
        quantization_bits=4,
    )

    try:
        model, tokenizer = load_model_and_tokenizer(test_config)
        console.print("\n[green]✓ Model loading test successful![/green]")
    except Exception as e:
        console.print(f"\n[red]✗ Model loading test failed: {e}[/red]")
