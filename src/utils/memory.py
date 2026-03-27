"""GPU/memory monitoring and optimization utilities.

Supports:
- NVIDIA (CUDA): VRAM tracking via torch.cuda
- Apple Silicon (MPS): Unified memory tracking
"""

import gc
import subprocess
import sys
from typing import Dict, Optional

from src.utils.platform_utils import get_platform


def get_vram_usage() -> Dict[str, float]:
    """Get GPU memory usage statistics (works on both CUDA and MPS).

    Returns:
        Dictionary with memory info in GB:
        - allocated: Currently allocated memory
        - reserved: Total reserved memory
        - free: Free memory
        - total: Total memory
    """
    try:
        import torch
        platform_info = get_platform()

        if platform_info.is_cuda:
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            total = torch.cuda.get_device_properties(0).total_memory / 1024**3

            return {
                "allocated": round(allocated, 2),
                "reserved": round(reserved, 2),
                "free": round(total - reserved, 2),
                "total": round(total, 2),
            }

        if platform_info.is_mps:
            # MPS uses unified memory — report system-level usage
            allocated = torch.mps.current_allocated_memory() / 1024**3
            total = platform_info.total_memory_gb

            return {
                "allocated": round(allocated, 2),
                "reserved": round(allocated, 2),  # MPS doesn't separate reserved
                "free": round(total - allocated, 2),
                "total": round(total, 2),
            }

        return {
            "allocated": 0.0,
            "reserved": 0.0,
            "free": 0.0,
            "total": 0.0,
        }
    except Exception as e:
        print(f"Warning: Could not get memory usage: {e}")
        return {
            "allocated": 0.0,
            "reserved": 0.0,
            "free": 0.0,
            "total": 0.0,
        }


def print_vram_usage(prefix: str = "") -> None:
    """Print GPU/memory usage to console.

    Args:
        prefix: Optional prefix string for the output
    """
    vram = get_vram_usage()
    platform_info = get_platform()

    if vram["total"] == 0:
        print(f"{prefix}No GPU detected")
        return

    usage_percent = (vram["allocated"] / vram["total"]) * 100
    mem_type = "VRAM" if platform_info.is_cuda else "Memory"

    print(
        f"{prefix}{mem_type}: {vram['allocated']:.2f}GB / {vram['total']:.2f}GB "
        f"({usage_percent:.1f}%) | Free: {vram['free']:.2f}GB"
    )


def clear_cache() -> None:
    """Clear GPU cache and run garbage collection."""
    try:
        import torch
        platform_info = get_platform()

        if platform_info.is_cuda:
            torch.cuda.empty_cache()
            gc.collect()
            print("CUDA cache cleared")
        elif platform_info.is_mps:
            torch.mps.empty_cache()
            gc.collect()
            print("MPS cache cleared")
        else:
            gc.collect()
            print("No GPU cache to clear (CPU only)")
    except Exception as e:
        print(f"Warning: Could not clear cache: {e}")


def optimize_memory(
    gradient_checkpointing: bool = True,
    use_flash_attention: bool = True,
) -> Dict[str, bool]:
    """Get recommended memory optimization settings.

    Args:
        gradient_checkpointing: Whether to use gradient checkpointing
        use_flash_attention: Whether to use Flash Attention 2

    Returns:
        Dictionary with optimization recommendations
    """
    vram = get_vram_usage()
    platform_info = get_platform()

    if vram["total"] == 0:
        return {
            "gradient_checkpointing": False,
            "use_flash_attention": False,
            "use_8bit": False,
            "use_cpu_offload": False,
            "recommendation": "No GPU detected",
        }

    optimizations = {
        "gradient_checkpointing": gradient_checkpointing,
        "use_flash_attention": use_flash_attention,
        "use_8bit": False,
        "use_cpu_offload": False,
        "recommendation": "",
    }

    if platform_info.is_mps:
        optimizations["use_flash_attention"] = False
        if vram["total"] >= 32:
            optimizations["gradient_checkpointing"] = False
            optimizations["recommendation"] = (
                f"Apple Silicon with {vram['total']:.0f}GB unified memory. "
                "No special optimizations needed."
            )
        else:
            optimizations["gradient_checkpointing"] = True
            optimizations["recommendation"] = (
                "Apple Silicon with limited memory. Gradient checkpointing recommended."
            )
    elif vram["total"] < 8:
        optimizations["gradient_checkpointing"] = True
        optimizations["use_flash_attention"] = True
        optimizations["recommendation"] = (
            "Low VRAM detected. Using gradient checkpointing and Flash Attention."
        )
    elif vram["total"] < 12:
        optimizations["gradient_checkpointing"] = True
        optimizations["recommendation"] = (
            "Medium VRAM. Using gradient checkpointing recommended."
        )
    else:
        optimizations["recommendation"] = "High VRAM. Standard configuration fine."

    return optimizations


def check_remote_gpu(host: str = "windows") -> Dict[str, any]:
    """Check GPU status on remote machine via SSH.

    This is useful when using Mac for development and remote Windows/ Linux for training.

    Args:
        host: Remote hostname (default: "windows")

    Returns:
        Dictionary with remote GPU information
    """
    try:
        result = subprocess.run(
            ["ssh", host, "nvidia-smi", "--query-gpu=name,memory.total,memory.free", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
        )

        if result.returncode != 0:
            return {
                "available": False,
                "error": result.stderr.strip(),
                "host": host,
            }

        output = result.stdout.strip()
        parts = output.split(", ")

        if len(parts) >= 3:
            return {
                "available": True,
                "host": host,
                "gpu_name": parts[0],
                "total_memory_gb": float(parts[1].strip().split()[0]),
                "free_memory_gb": float(parts[2].strip().split()[0]),
            }

    except subprocess.TimeoutExpired:
        return {
            "available": False,
            "error": "SSH connection timed out",
            "host": host,
        }
    except FileNotFoundError:
        return {
            "available": False,
            "error": "SSH command not found",
            "host": host,
        }
    except Exception as e:
        return {
            "available": False,
            "error": str(e),
            "host": host,
        }

    return {
        "available": False,
        "error": "Unknown error",
        "host": host,
    }


def estimate_model_vram(
    model_name: str,
    quantization_bits: int = 4,
    lora_r: int = 16,
    batch_size: int = 1,
    max_length: int = 512,
) -> Dict[str, float]:
    """Estimate VRAM usage for a model configuration.

    Args:
        model_name: Hugging Face model name
        quantization_bits: Quantization (4 or 8)
        lora_r: LoRA rank
        batch_size: Batch size
        max_length: Maximum sequence length

    Returns:
        Dictionary with estimated VRAM usage in GB
    """
    # Extract parameter size from model name if possible
    param_size_map = {
        "0.5b": 0.5,
        "1b": 1.0,
        "1.5b": 1.5,
        "3b": 3.0,
        "7b": 7.0,
        "8b": 8.0,
    }

    params = 1.0  # Default to 1B
    for size_str, param_count in param_size_map.items():
        if size_str in model_name.lower():
            params = param_count
            break

    # Base model VRAM (quantized)
    base_vram = (params * quantization_bits / 16) * 1.2  # 20% overhead

    # LoRA adapter VRAM (very small)
    lora_vram = (lora_r * 8 * 2 * 4) / (1024**3) * params  # Rough estimate

    # Training overhead (optimizer states, gradients)
    if quantization_bits == 4:
        overhead_vram = base_vram * 0.5
    else:
        overhead_vram = base_vram * 0.3

    # Activation memory (sequence length and batch size dependent)
    activation_vram = (max_length / 512) * batch_size * 0.5

    total_vram = base_vram + lora_vram + overhead_vram + activation_vram

    return {
        "base_model_gb": round(base_vram, 2),
        "lora_adapter_gb": round(lora_vram, 3),
        "training_overhead_gb": round(overhead_vram, 2),
        "activation_gb": round(activation_vram, 2),
        "total_gb": round(total_vram, 2),
        "recommended_batch_size": 1 if total_vram > 6 else 2,
    }


if __name__ == "__main__":
    # Test memory utilities
    print("=== Local GPU Memory ===")
    print_vram_usage()
    print()

    print("=== Memory Optimization Recommendations ===")
    optimizations = optimize_memory()
    for key, value in optimizations.items():
        print(f"  {key}: {value}")
    print()

    print("=== Remote GPU Check ===")
    remote = check_remote_gpu("windows")
    print(f"  Available: {remote.get('available', False)}")
    if remote.get("available"):
        print(f"  GPU: {remote.get('gpu_name')}")
        print(f"  Total: {remote.get('total_memory_gb')}GB")
        print(f"  Free: {remote.get('free_memory_gb')}GB")
    else:
        print(f"  Error: {remote.get('error')}")
    print()

    print("=== VRAM Estimation for Qwen 1.5B ===")
    estimate = estimate_model_vram(
        "Qwen/Qwen2.5-1.5B-Instruct",
        quantization_bits=4,
        lora_r=16,
        batch_size=1,
        max_length=1024,
    )
    for key, value in estimate.items():
        print(f"  {key}: {value}")
