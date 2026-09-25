"""Unit tests for platform detection (src/utils/platform_utils.py).

torch availability probes are patched per branch; the singleton cache is
reset around each test so cached state never leaks between suites.
"""

from __future__ import annotations

import subprocess
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import torch

import src.utils.platform_utils as pmod
from src.utils.platform_utils import (
    PlatformInfo,
    detect_platform,
    get_platform,
    get_torch_dtype,
    recommend_settings,
)


@pytest.fixture(autouse=True)
def _reset_singleton() -> Any:
    pmod._platform = None
    yield
    pmod._platform = None


def _cuda_patch(
    *,
    bf16: bool = True,
    device_count: int = 1,
    total_memory_gb: float = 8.0,
    gpu_name: str = "RTX 4060",
) -> dict[str, Any]:
    """Patch bundle for the CUDA branch of detect_platform."""
    props = MagicMock()
    props.total_memory = int(total_memory_gb * 1024**3)
    return {
        "torch.cuda.is_available": patch("torch.cuda.is_available", return_value=True),
        "torch.cuda.is_bf16_supported": patch("torch.cuda.is_bf16_supported", return_value=bf16),
        "torch.cuda.device_count": patch("torch.cuda.device_count", return_value=device_count),
        "torch.cuda.get_device_properties": patch(
            "torch.cuda.get_device_properties", return_value=props
        ),
        "torch.cuda.get_device_name": patch("torch.cuda.get_device_name", return_value=gpu_name),
    }


def _run_with_patches(patches: dict[str, Any], fn: Any) -> Any:
    with (
        patches["torch.cuda.is_available"],
        patches["torch.cuda.is_bf16_supported"],
        patches["torch.cuda.device_count"],
        patches["torch.cuda.get_device_properties"],
        patches["torch.cuda.get_device_name"],
    ):
        return fn()


class TestDetectPlatformCuda:
    def test_single_gpu(self) -> None:
        patches = _cuda_patch(total_memory_gb=8.0, gpu_name="RTX 4060")
        info = _run_with_patches(patches, detect_platform)

        assert info.device == "cuda"
        assert info.is_cuda is True
        assert info.is_mps is False
        assert info.supports_quantization is True
        assert info.supports_bf16 is True
        assert info.gpu_count == 1
        assert info.total_memory_gb == pytest.approx(8.0)
        assert info.description == "NVIDIA RTX 4060 (8.0GB VRAM)"

    def test_multi_gpu_description_includes_count(self) -> None:
        patches = _cuda_patch(device_count=2, total_memory_gb=24.0, gpu_name="RTX 4090")
        info = _run_with_patches(patches, detect_platform)

        assert info.gpu_count == 2
        assert info.description == "2× NVIDIA RTX 4090 (24.0GB VRAM each)"

    def test_bf16_unsupported_flag_propagates(self) -> None:
        patches = _cuda_patch(bf16=False)
        info = _run_with_patches(patches, detect_platform)
        assert info.supports_bf16 is False


class TestDetectPlatformMps:
    def _detect(self) -> PlatformInfo:
        with (
            patch("torch.cuda.is_available", return_value=False),
            patch("torch.backends.mps.is_available", return_value=True),
            patch.object(pmod, "_get_apple_memory_gb", return_value=36.0),
        ):
            return detect_platform()

    def test_mps_capabilities(self) -> None:
        info = self._detect()
        assert info.device == "mps"
        assert info.is_cuda is False
        assert info.is_mps is True
        assert info.supports_quantization is False  # no bitsandbytes on MPS
        assert info.supports_bf16 is True
        assert info.gpu_count == 1
        assert info.total_memory_gb == pytest.approx(36.0)
        assert "unified memory" in info.description


class TestDetectPlatformCpu:
    def test_cpu_fallback(self) -> None:
        with (
            patch("torch.cuda.is_available", return_value=False),
            patch("torch.backends.mps.is_available", return_value=False),
        ):
            info = detect_platform()

        assert info.device == "cpu"
        assert info.is_cuda is False
        assert info.is_mps is False
        assert info.supports_quantization is False
        assert info.supports_bf16 is False
        assert info.gpu_count == 0
        assert info.total_memory_gb == 0.0
        assert info.description == "CPU only (no GPU acceleration)"


class TestGetAppleMemoryGb:
    def test_parses_sysctl_output(self) -> None:
        result = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout="68719476736\n",  # 64 GiB
        )
        with patch("subprocess.run", return_value=result):
            assert pmod._get_apple_memory_gb() == pytest.approx(64.0)

    def test_nonzero_returncode_returns_zero(self) -> None:
        result = subprocess.CompletedProcess(args=[], returncode=1, stdout="")
        with patch("subprocess.run", return_value=result):
            assert pmod._get_apple_memory_gb() == 0.0

    def test_exception_returns_zero(self) -> None:
        with patch("subprocess.run", side_effect=TimeoutError("sysctl hung")):
            assert pmod._get_apple_memory_gb() == 0.0


class TestGetTorchDtype:
    def _info(self, supports_bf16: bool) -> PlatformInfo:
        return PlatformInfo(
            device="cuda",
            is_cuda=True,
            is_mps=False,
            is_apple_silicon=False,
            supports_quantization=True,
            supports_bf16=supports_bf16,
            total_memory_gb=8.0,
            description="test",
        )

    def test_bfloat16_when_supported(self) -> None:
        assert get_torch_dtype("bfloat16", self._info(True)) is torch.bfloat16

    def test_bfloat16_falls_back_to_fp16(self) -> None:
        assert get_torch_dtype("bfloat16", self._info(False)) is torch.float16

    def test_float16_passthrough(self) -> None:
        assert get_torch_dtype("float16", self._info(True)) is torch.float16

    def test_unknown_defaults_to_fp32(self) -> None:
        assert get_torch_dtype("int8", self._info(True)) is torch.float32

    def test_detects_platform_when_not_given(self) -> None:
        fake = self._info(True)
        with patch.object(pmod, "detect_platform", return_value=fake) as mock_detect:
            assert get_torch_dtype("bfloat16") is torch.bfloat16
        mock_detect.assert_called_once()


class TestRecommendSettings:
    def _info(
        self,
        device: str,
        total_memory_gb: float,
    ) -> PlatformInfo:
        return PlatformInfo(
            device=device,
            is_cuda=device == "cuda",
            is_mps=device == "mps",
            is_apple_silicon=False,
            supports_quantization=device == "cuda",
            supports_bf16=device != "cpu",
            total_memory_gb=total_memory_gb,
            description="test",
        )

    def test_low_vram_cuda_recipe(self) -> None:
        recs = recommend_settings(self._info("cuda", 8.0))
        assert recs["quantization_bits"] == 4
        assert recs["batch_size"] == 1
        assert recs["gradient_accumulation_steps"] == 8
        assert recs["gradient_checkpointing"] is True
        assert recs["max_length"] == 512
        assert "Low VRAM" in recs["reason"]

    def test_medium_vram_cuda_recipe(self) -> None:
        recs = recommend_settings(self._info("cuda", 24.0))
        assert recs["batch_size"] == 2
        assert recs["gradient_accumulation_steps"] == 4
        assert recs["max_length"] == 1024

    def test_mps_large_memory_batch4(self) -> None:
        recs = recommend_settings(self._info("mps", 36.0))
        assert recs["quantization_bits"] is None
        assert recs["batch_size"] == 4
        assert recs["gradient_checkpointing"] is False
        assert "unified memory" in recs["reason"]

    def test_mps_small_memory_batch2(self) -> None:
        recs = recommend_settings(self._info("mps", 16.0))
        assert recs["batch_size"] == 2

    def test_cpu_recipe(self) -> None:
        recs = recommend_settings(self._info("cpu", 0.0))
        assert recs["quantization_bits"] is None
        assert recs["batch_size"] == 1
        assert recs["gradient_accumulation_steps"] == 16
        assert recs["max_length"] == 256
        assert recs["torch_dtype"] == "float32"

    def test_detects_platform_when_not_given(self) -> None:
        fake = self._info("cpu", 0.0)
        with patch.object(pmod, "detect_platform", return_value=fake) as mock_detect:
            recommend_settings()
        mock_detect.assert_called_once()


class TestGetPlatformSingleton:
    def test_computed_once_then_cached(self) -> None:
        fake = PlatformInfo(
            device="cpu",
            is_cuda=False,
            is_mps=False,
            is_apple_silicon=False,
            supports_quantization=False,
            supports_bf16=False,
            total_memory_gb=0.0,
            description="fake",
        )
        with patch.object(pmod, "detect_platform", return_value=fake) as mock_detect:
            assert get_platform() is fake
            assert get_platform() is fake  # cached
        mock_detect.assert_called_once()
