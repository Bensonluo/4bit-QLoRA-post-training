"""Unit tests for reproducibility utilities (src/utils/seed.py).

cudnn flags are snapshotted/restored around each test so flag mutations
never leak between suites.
"""

from __future__ import annotations

import random
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from src.utils.seed import get_seed, set_seed


@pytest.fixture(autouse=True)
def _restore_cudnn_flags() -> Any:
    deterministic = torch.backends.cudnn.deterministic
    benchmark = torch.backends.cudnn.benchmark
    yield
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = benchmark


class TestSetSeed:
    def test_python_and_numpy_deterministic(self) -> None:
        set_seed(123)
        py_a, np_a = random.random(), np.random.rand()
        set_seed(123)
        py_b, np_b = random.random(), np.random.rand()
        assert py_a == py_b
        assert np_a == np_b

    @patch("torch.cuda.is_available", return_value=False)
    def test_cuda_absent_disables_deterministic_only(self, _mock_avail: MagicMock) -> None:
        set_seed(42)
        assert torch.backends.cudnn.deterministic is False

    @patch("torch.manual_seed")
    @patch("torch.cuda.manual_seed_all")
    @patch("torch.cuda.is_available", return_value=True)
    def test_cuda_present_seeds_all_gpus_and_enables_benchmark(
        self, _mock_avail: MagicMock, mock_seed_all: MagicMock, _mock_manual: MagicMock
    ) -> None:
        # torch.manual_seed is also patched: the real one internally calls
        # cuda.manual_seed_all, which would make the call count ambiguous.
        set_seed(42)
        mock_seed_all.assert_called_once_with(42)
        assert torch.backends.cudnn.benchmark is True

    @patch("torch.cuda.manual_seed_all")
    @patch("torch.cuda.is_available", return_value=True)
    def test_deterministic_mode_disables_benchmark(
        self, _mock_avail: MagicMock, _mock_seed_all: MagicMock
    ) -> None:
        set_seed(42, deterministic=True)
        assert torch.backends.cudnn.deterministic is True
        assert torch.backends.cudnn.benchmark is False

    @patch("torch.manual_seed", side_effect=RuntimeError("torch exploded"))
    def test_torch_failure_warns_instead_of_raising(
        self, _mock_manual: MagicMock, capsys: pytest.CaptureFixture[str]
    ) -> None:
        set_seed(7)  # must not propagate the RuntimeError
        assert "Could not fully set seed" in capsys.readouterr().out


class TestGetSeed:
    def test_returns_int_in_uint32_range(self) -> None:
        seed = get_seed()
        assert isinstance(seed, int)
        assert 0 <= seed < 2**32

    @patch("time.time", return_value=1_700_000_000.9)
    def test_derived_from_current_time(self, _mock_time: MagicMock) -> None:
        assert get_seed() == 1_700_000_000 % 2**32
