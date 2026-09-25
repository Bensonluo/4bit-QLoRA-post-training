"""Unit tests for distributed helpers (src/training/distributed/).

env.py detection is env-var based, so torchrun context is simulated with
monkeypatch. logger.py's rank gating is tested by patching the module-local
`is_rank_zero` import.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.training.distributed.env import (
    get_distributed_info,
    is_rank_zero,
    rank_zero_only,
    setup_distributed,
)
from src.training.distributed.logger import _RankZeroConsole, get_rank_zero_console


class TestGetDistributedInfo:
    def test_single_process_defaults(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for var in ("LOCAL_RANK", "RANK", "WORLD_SIZE"):
            monkeypatch.delenv(var, raising=False)
        info = get_distributed_info()
        assert (info.is_distributed, info.local_rank, info.rank, info.world_size) == (
            False,
            -1,
            0,
            1,
        )
        assert info.is_main_process is True

    def test_torchrun_multi_gpu(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LOCAL_RANK", "1")
        monkeypatch.setenv("RANK", "1")
        monkeypatch.setenv("WORLD_SIZE", "4")
        info = get_distributed_info()
        assert (info.is_distributed, info.local_rank, info.rank, info.world_size) == (
            True,
            1,
            1,
            4,
        )
        assert info.is_main_process is False

    def test_world_size_one_is_not_distributed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # torchrun --nproc_per_node=1 still sets LOCAL_RANK=0 but HF treats
        # world_size==1 as non-distributed — we match that convention.
        monkeypatch.setenv("LOCAL_RANK", "0")
        monkeypatch.setenv("RANK", "0")
        monkeypatch.setenv("WORLD_SIZE", "1")
        assert get_distributed_info().is_distributed is False

    def test_nonzero_rank_is_not_main_even_single_node(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("RANK", "3")
        assert is_rank_zero() is False

    def test_frozen_dataclass(self) -> None:
        info = get_distributed_info()
        with pytest.raises(AttributeError):
            info.rank = 99  # type: ignore[misc]


class TestRankZeroOnly:
    def test_runs_on_rank_zero(self) -> None:
        calls: list[str] = []

        @rank_zero_only
        def work() -> str:
            calls.append("ran")
            return "done"

        assert work() == "done"
        assert calls == ["ran"]

    @patch("src.training.distributed.env.is_rank_zero", return_value=False)
    def test_skipped_on_other_ranks(self, _mock: MagicMock) -> None:
        @rank_zero_only
        def work() -> str:
            raise AssertionError("must not run on non-zero ranks")

        assert work() is None

    def test_preserves_metadata(self) -> None:
        @rank_zero_only
        def documented_fn() -> None: ...

        assert documented_fn.__name__ == "documented_fn"


class TestSetupDistributed:
    def test_single_gpu_noop(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for var in ("LOCAL_RANK", "RANK", "WORLD_SIZE"):
            monkeypatch.delenv(var, raising=False)
        with patch("torch.cuda.set_device") as mock_set:
            info = setup_distributed()
        mock_set.assert_not_called()
        assert info.is_distributed is False

    @patch("torch.cuda.set_device")
    @patch("torch.cuda.is_available", return_value=True)
    def test_pins_cuda_device_when_distributed(
        self, _mock_avail: MagicMock, mock_set: MagicMock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("LOCAL_RANK", "2")
        monkeypatch.setenv("RANK", "2")
        monkeypatch.setenv("WORLD_SIZE", "4")
        info = setup_distributed()
        mock_set.assert_called_once_with(2)
        assert info.world_size == 4


class TestRankZeroConsole:
    def test_print_forwards_on_rank_zero(self) -> None:
        real = MagicMock()
        with patch("src.training.distributed.logger.is_rank_zero", return_value=True):
            _RankZeroConsole(real).print("[green]hi[/green]", highlight=False)
        real.print.assert_called_once_with("[green]hi[/green]", highlight=False)

    def test_print_silent_on_other_ranks(self) -> None:
        real = MagicMock()
        with patch("src.training.distributed.logger.is_rank_zero", return_value=False):
            result = _RankZeroConsole(real).print("noise")
        real.print.assert_not_called()
        assert result is None

    def test_getattr_forwards_on_rank_zero(self) -> None:
        real = MagicMock()
        real.log = "LOG_OBJ"
        with patch("src.training.distributed.logger.is_rank_zero", return_value=True):
            proxy = _RankZeroConsole(real)
            assert proxy.log == "LOG_OBJ"  # non-callable attr passed through

    def test_getattr_wraps_callables_as_noop_off_rank(self) -> None:
        real = MagicMock()
        real.status.return_value = "ctx"
        with patch("src.training.distributed.logger.is_rank_zero", return_value=False):
            proxy = _RankZeroConsole(real)
            assert callable(proxy.status)
            assert proxy.status("training") is None  # no-op, no ctx entered
        real.status.assert_not_called()

    def test_getattr_noncallable_off_rank_returns_value(self) -> None:
        real = MagicMock()
        real.width = 120
        with patch("src.training.distributed.logger.is_rank_zero", return_value=False):
            assert _RankZeroConsole(real).width == 120

    def test_factory_wraps_global_console(self) -> None:
        proxy = get_rank_zero_console()
        assert isinstance(proxy, _RankZeroConsole)
        assert proxy._real is not None
