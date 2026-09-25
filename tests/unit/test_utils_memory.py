"""Unit tests for GPU/memory monitoring utilities (src/utils/memory.py).

Platform branches are exercised by patching the platform lookup and torch's
device-specific APIs; no real GPU state is required.
"""

import subprocess
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

import src.utils.memory as memory_mod
from src.utils.memory import (
    check_remote_gpu,
    clear_cache,
    estimate_model_vram,
    get_vram_usage,
    optimize_memory,
    print_vram_usage,
)


def _platform(device: str = "mps", total_gb: float = 18.0) -> MagicMock:
    p = MagicMock()
    p.device = device
    p.description = f"test-{device}"
    p.is_cuda = device == "cuda"
    p.is_mps = device == "mps"
    p.total_memory_gb = total_gb
    return p


class TestGetVramUsage:
    def test_cuda_branch(self) -> None:
        with (
            patch("src.utils.memory.get_platform", return_value=_platform("cuda")),
            patch("torch.cuda.memory_allocated", return_value=2 * 1024**3),
            patch("torch.cuda.memory_reserved", return_value=3 * 1024**3),
            patch("torch.cuda.get_device_properties") as mock_props,
        ):
            mock_props.return_value.total_memory = 8 * 1024**3
            vram = get_vram_usage()

        assert vram["allocated"] == 2.0
        assert vram["reserved"] == 3.0
        assert vram["total"] == 8.0
        assert vram["free"] == 5.0

    def test_mps_branch_reports_unified_memory(self) -> None:
        with (
            patch("src.utils.memory.get_platform", return_value=_platform("mps", 18.0)),
            patch("torch.mps.current_allocated_memory", return_value=4 * 1024**3),
        ):
            vram = get_vram_usage()

        assert vram["allocated"] == 4.0
        assert vram["reserved"] == 4.0  # MPS doesn't separate reserved
        assert vram["total"] == 18.0
        assert vram["free"] == 14.0

    def test_cpu_branch_returns_zeros(self) -> None:
        with patch("src.utils.memory.get_platform", return_value=_platform("cpu")):
            vram = get_vram_usage()

        assert vram == {"allocated": 0.0, "reserved": 0.0, "free": 0.0, "total": 0.0}

    def test_exception_returns_zero_dict(self) -> None:
        with patch("src.utils.memory.get_platform", side_effect=RuntimeError("boom")):
            vram = get_vram_usage()

        assert vram == {"allocated": 0.0, "reserved": 0.0, "free": 0.0, "total": 0.0}


class TestPrintVramUsage:
    def test_no_gpu_message(self, capsys: Any) -> None:
        with patch("src.utils.memory.get_vram_usage", return_value={"total": 0.0}):
            print_vram_usage(prefix="[step 10] ")

        assert "No GPU detected" in capsys.readouterr().out

    def test_cuda_labels_vram(self, capsys: Any) -> None:
        vram = {"allocated": 2.0, "reserved": 3.0, "free": 6.0, "total": 8.0}
        with (
            patch("src.utils.memory.get_vram_usage", return_value=vram),
            patch("src.utils.memory.get_platform", return_value=_platform("cuda")),
        ):
            print_vram_usage()

        out = capsys.readouterr().out
        assert "VRAM: 2.00GB / 8.00GB" in out
        assert "Free: 6.00GB" in out

    def test_mps_labels_memory(self, capsys: Any) -> None:
        vram = {"allocated": 4.0, "reserved": 4.0, "free": 14.0, "total": 18.0}
        with (
            patch("src.utils.memory.get_vram_usage", return_value=vram),
            patch("src.utils.memory.get_platform", return_value=_platform("mps")),
        ):
            print_vram_usage()

        assert "Memory: 4.00GB / 18.00GB" in capsys.readouterr().out


class TestClearCache:
    def test_cuda_clears_cuda_cache(self, capsys: Any) -> None:
        with (
            patch("src.utils.memory.get_platform", return_value=_platform("cuda")),
            patch("torch.cuda.empty_cache") as mock_empty,
            patch(f"{memory_mod.__name__}.gc.collect") as mock_gc,
        ):
            clear_cache()

        mock_empty.assert_called_once()
        mock_gc.assert_called_once()
        assert "CUDA cache cleared" in capsys.readouterr().out

    def test_mps_clears_mps_cache(self, capsys: Any) -> None:
        with (
            patch("src.utils.memory.get_platform", return_value=_platform("mps")),
            patch("torch.mps.empty_cache") as mock_empty,
            patch(f"{memory_mod.__name__}.gc.collect") as mock_gc,
        ):
            clear_cache()

        mock_empty.assert_called_once()
        mock_gc.assert_called_once()
        assert "MPS cache cleared" in capsys.readouterr().out

    def test_cpu_only_collects_gc(self, capsys: Any) -> None:
        with (
            patch("src.utils.memory.get_platform", return_value=_platform("cpu")),
            patch(f"{memory_mod.__name__}.gc.collect") as mock_gc,
        ):
            clear_cache()

        mock_gc.assert_called_once()
        assert "CPU only" in capsys.readouterr().out

    def test_exception_swallowed(self, capsys: Any) -> None:
        with patch("src.utils.memory.get_platform", side_effect=RuntimeError("boom")):
            clear_cache()  # must not raise

        assert "Could not clear cache" in capsys.readouterr().out


class TestOptimizeMemory:
    def _run(self, vram_total: float, device: str = "cuda") -> dict[str, Any]:
        vram = {"allocated": 1.0, "reserved": 2.0, "free": vram_total - 2.0, "total": vram_total}
        with (
            patch("src.utils.memory.get_vram_usage", return_value=vram),
            patch(
                "src.utils.memory.get_platform",
                return_value=_platform(device, vram_total),
            ),
        ):
            return optimize_memory()

    def test_no_gpu(self) -> None:
        with patch(
            "src.utils.memory.get_vram_usage",
            return_value={"allocated": 0.0, "reserved": 0.0, "free": 0.0, "total": 0.0},
        ):
            result = optimize_memory()
        assert result["recommendation"] == "No GPU detected"

    def test_mps_large_memory_needs_nothing(self) -> None:
        result = self._run(36.0, "mps")
        assert result["gradient_checkpointing"] is False
        assert result["use_flash_attention"] is False
        assert "No special optimizations" in result["recommendation"]

    def test_mps_limited_memory_keeps_checkpointing(self) -> None:
        result = self._run(16.0, "mps")
        assert result["gradient_checkpointing"] is True
        assert "Gradient checkpointing recommended" in result["recommendation"]

    def test_low_vram_cuda(self) -> None:
        result = self._run(6.0)
        assert result["gradient_checkpointing"] is True
        assert result["use_flash_attention"] is True
        assert "Low VRAM" in result["recommendation"]

    def test_medium_vram_cuda(self) -> None:
        result = self._run(10.0)
        assert result["gradient_checkpointing"] is True
        assert result["use_flash_attention"] is True  # caller default kept

    def test_high_vram_cuda(self) -> None:
        result = self._run(24.0)
        assert result["recommendation"] == "High VRAM. Standard configuration fine."


class TestCheckRemoteGpu:
    def _run_result(self, returncode: int = 0, stdout: str = "", stderr: str = "") -> MagicMock:
        result = MagicMock()
        result.returncode = returncode
        result.stdout = stdout
        result.stderr = stderr
        return result

    def test_parses_gpu_info(self) -> None:
        result = self._run_result(stdout="NVIDIA GeForce RTX 4060, 8192 MiB, 4096 MiB\n")
        with patch("src.utils.memory.subprocess.run", return_value=result):
            info = check_remote_gpu("windows")

        assert info["available"] is True
        assert info["gpu_name"] == "NVIDIA GeForce RTX 4060"
        assert info["total_memory_gb"] == 8192.0
        assert info["free_memory_gb"] == 4096.0

    def test_ssh_failure_reports_error(self) -> None:
        result = self._run_result(returncode=255, stderr="Connection refused")
        with patch("src.utils.memory.subprocess.run", return_value=result):
            info = check_remote_gpu("windows")

        assert info["available"] is False
        assert "Connection refused" in info["error"]

    def test_timeout(self) -> None:
        with patch(
            "src.utils.memory.subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd="ssh", timeout=10),
        ):
            info = check_remote_gpu("windows")

        assert info["available"] is False
        assert "timed out" in info["error"]

    def test_ssh_missing(self) -> None:
        with patch("src.utils.memory.subprocess.run", side_effect=FileNotFoundError("ssh")):
            info = check_remote_gpu("windows")

        assert info["available"] is False
        assert "not found" in info["error"]

    def test_generic_exception_reports_error(self) -> None:
        # Neither TimeoutExpired nor FileNotFoundError — hits the catch-all.
        with patch("src.utils.memory.subprocess.run", side_effect=OSError("DNS failure")):
            info = check_remote_gpu("windows")

        assert info["available"] is False
        assert info["error"] == "DNS failure"

    def test_malformed_output_falls_through(self) -> None:
        result = self._run_result(stdout="garbage")
        with patch("src.utils.memory.subprocess.run", return_value=result):
            info = check_remote_gpu("windows")

        assert info["available"] is False
        assert info["error"] == "Unknown error"


class TestEstimateModelVram:
    def test_known_model_size_parsed(self) -> None:
        est = estimate_model_vram("Qwen/Qwen2.5-1.5B-Instruct", quantization_bits=4)
        assert est["total_gb"] > 0
        # Arithmetic consistency (rounded components sum ≈ rounded total).
        component_sum = (
            est["base_model_gb"]
            + est["lora_adapter_gb"]
            + est["training_overhead_gb"]
            + est["activation_gb"]
        )
        assert est["total_gb"] == pytest.approx(component_sum, abs=0.02)

    def test_unknown_model_defaults_to_1b(self) -> None:
        est = estimate_model_vram("mystery-model")
        # 1B params at 4-bit: base = 1.0 * 4/16 * 1.2 = 0.3 GB
        assert est["base_model_gb"] == 0.3

    def test_8bit_lower_overhead_ratio_than_4bit(self) -> None:
        est4 = estimate_model_vram("Qwen/Qwen2.5-1.5B-Instruct", quantization_bits=4)
        est8 = estimate_model_vram("Qwen/Qwen2.5-1.5B-Instruct", quantization_bits=8)
        # Overhead is 50% of base for 4-bit vs 30% for 8-bit (per-field
        # rounding tolerance — values are round(..., 2)).
        assert est4["training_overhead_gb"] == pytest.approx(est4["base_model_gb"] * 0.5, abs=0.01)
        assert est8["training_overhead_gb"] == pytest.approx(est8["base_model_gb"] * 0.3, abs=0.01)

    def test_recommends_batch_1_when_over_6gb(self) -> None:
        # 8B at 8-bit: base 4.8 + overhead 1.44 + activation 0.5 ≈ 6.74 GB.
        est = estimate_model_vram("Qwen/Qwen2.5-8B-Instruct", quantization_bits=8)
        assert est["total_gb"] > 6
        assert est["recommended_batch_size"] == 1

    def test_recommends_batch_2_when_under_6gb(self) -> None:
        est = estimate_model_vram("Qwen/Qwen2.5-0.5B-Instruct", quantization_bits=4)
        assert est["recommended_batch_size"] == 2
