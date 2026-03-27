"""Platform detection and device management utilities.

Automatically detects and configures for the available hardware:
- NVIDIA GPU (CUDA) — bitsandbytes quantization, bf16
- Apple Silicon (MPS) — fp16/bf16, no quantization needed (unified memory)
- CPU fallback
"""

import platform
from dataclasses import dataclass
from typing import Optional

import torch


@dataclass(frozen=True)
class PlatformInfo:
    """Detected platform capabilities."""

    device: str  # "cuda", "mps", or "cpu"
    is_cuda: bool
    is_mps: bool
    is_apple_silicon: bool
    supports_quantization: bool  # bitsandbytes available
    supports_bf16: bool
    total_memory_gb: float
    description: str


def detect_platform() -> PlatformInfo:
    """Detect the current hardware platform and capabilities.

    Returns:
        PlatformInfo with detected capabilities.
    """
    is_apple_silicon = platform.processor() == "arm" and platform.system() == "Darwin"

    # Determine device
    if torch.cuda.is_available():
        device = "cuda"
        is_cuda = True
        is_mps = False
        supports_quantization = True
        supports_bf16 = torch.cuda.is_bf16_supported()
        total_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        gpu_name = torch.cuda.get_device_name(0)
        description = f"NVIDIA {gpu_name} ({total_memory_gb:.1f}GB VRAM)"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
        is_cuda = False
        is_mps = True
        supports_quantization = False
        supports_bf16 = True  # Apple Silicon supports bf16 computation
        total_memory_gb = _get_apple_memory_gb()
        description = f"Apple Silicon ({total_memory_gb:.1f}GB unified memory)"
    else:
        device = "cpu"
        is_cuda = False
        is_mps = False
        supports_quantization = False
        supports_bf16 = False
        total_memory_gb = 0.0
        description = "CPU only (no GPU acceleration)"

    return PlatformInfo(
        device=device,
        is_cuda=is_cuda,
        is_mps=is_mps,
        is_apple_silicon=is_apple_silicon,
        supports_quantization=supports_quantization,
        supports_bf16=supports_bf16,
        total_memory_gb=total_memory_gb,
        description=description,
    )


def _get_apple_memory_gb() -> float:
    """Get total memory on Apple Silicon (unified memory)."""
    try:
        result = __import__("subprocess").run(
            ["sysctl", "-n", "hw.memsize"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return int(result.stdout.strip()) / 1024**3
    except Exception:
        pass
    return 0.0


def get_torch_dtype(dtype_str: str, platform_info: Optional[PlatformInfo] = None) -> torch.dtype:
    """Get torch dtype, adjusting for platform capabilities.

    Falls back from bf16 to float16 on platforms that don't support bf16.
    """
    if platform_info is None:
        platform_info = detect_platform()

    if dtype_str == "bfloat16":
        if platform_info.supports_bf16:
            return torch.bfloat16
        return torch.float16  # fallback
    if dtype_str == "float16":
        return torch.float16
    return torch.float32


def recommend_settings(platform_info: Optional[PlatformInfo] = None) -> dict:
    """Recommend training settings based on detected platform.

    Returns:
        Dictionary of recommended settings.
    """
    if platform_info is None:
        platform_info = detect_platform()

    if platform_info.is_cuda:
        if platform_info.total_memory_gb <= 8:
            return {
                "quantization_bits": 4,
                "batch_size": 1,
                "gradient_accumulation_steps": 8,
                "gradient_checkpointing": True,
                "max_length": 512,
                "torch_dtype": "bfloat16",
                "reason": "Low VRAM (<=8GB). Use 4-bit quantization with aggressive memory saving.",
            }
        return {
            "quantization_bits": 4,
            "batch_size": 2,
            "gradient_accumulation_steps": 4,
            "gradient_checkpointing": True,
            "max_length": 1024,
            "torch_dtype": "bfloat16",
            "reason": "Medium+ VRAM. 4-bit quantization with relaxed memory constraints.",
        }

    if platform_info.is_mps:
        return {
            "quantization_bits": None,  # No quantization on MPS
            "batch_size": 4 if platform_info.total_memory_gb >= 32 else 2,
            "gradient_accumulation_steps": 2,
            "gradient_checkpointing": False,
            "max_length": 1024,
            "torch_dtype": "bfloat16",
            "reason": (
                f"Apple Silicon with {platform_info.total_memory_gb:.0f}GB unified memory. "
                "No quantization needed — load model in bf16 directly."
            ),
        }

    return {
        "quantization_bits": None,
        "batch_size": 1,
        "gradient_accumulation_steps": 16,
        "gradient_checkpointing": True,
        "max_length": 256,
        "torch_dtype": "float32",
        "reason": "CPU only. Use small models and short sequences.",
    }


# Singleton — compute once
_platform: Optional[PlatformInfo] = None


def get_platform() -> PlatformInfo:
    """Get cached platform info (computed once)."""
    global _platform
    if _platform is None:
        _platform = detect_platform()
    return _platform


if __name__ == "__main__":
    info = get_platform()
    print(f"Device: {info.device}")
    print(f"Platform: {info.description}")
    print(f"Supports quantization: {info.supports_quantization}")
    print(f"Supports bf16: {info.supports_bf16}")
    print()

    recs = recommend_settings(info)
    print("Recommended settings:")
    for k, v in recs.items():
        print(f"  {k}: {v}")
