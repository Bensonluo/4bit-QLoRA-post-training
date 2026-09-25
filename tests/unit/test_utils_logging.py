"""Unit tests for logging utilities (src/utils/logging.py).

Rich console output is captured via capsys (non-terminal rendering is plain
text). The global "qlora" logger is snapshotted/restored around each test to
avoid handler pollution across suites.
"""

from __future__ import annotations

import logging
import sys
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

import src.utils.logging as log_mod
from src.utils.logging import (
    log_gpu_memory,
    log_metrics,
    print_table,
    print_training_summary,
    setup_logging,
    setup_tensorboard,
    setup_wandb,
)


@pytest.fixture(autouse=True)
def _restore_qlora_logger() -> Any:
    logger = logging.getLogger("qlora")
    saved_handlers = list(logger.handlers)
    saved_level = logger.level
    yield
    logger.handlers = saved_handlers
    logger.setLevel(saved_level)


class TestSetupLogging:
    def test_rich_console_handler_installed(self) -> None:
        logger = setup_logging(level="WARNING")
        assert logger.name == "qlora"
        assert logger.level == logging.WARNING
        assert len(logger.handlers) == 1
        assert isinstance(logger.handlers[0], log_mod.RichHandler)
        assert logger.handlers[0].level == logging.WARNING

    def test_plain_stream_handler_when_rich_disabled(self) -> None:
        logger = setup_logging(use_rich=False)
        assert isinstance(logger.handlers[0], logging.StreamHandler)
        assert not isinstance(logger.handlers[0], log_mod.RichHandler)

    def test_file_handler_created_with_debug_level(self, tmp_path: Any) -> None:
        log_file = tmp_path / "logs" / "train.log"
        logger = setup_logging(log_file=str(log_file))

        assert log_file.parent.exists()
        assert len(logger.handlers) == 2
        file_handler = logger.handlers[1]
        assert isinstance(file_handler, logging.FileHandler)
        assert file_handler.level == logging.DEBUG

        logger.info("hello file")
        file_handler.flush()
        assert "hello file" in log_file.read_text()

    def test_existing_handlers_cleared(self) -> None:
        logger = logging.getLogger("qlora")
        logger.addHandler(logging.NullHandler())
        logger.addHandler(logging.NullHandler())

        setup_logging()

        assert len(logger.handlers) == 1  # only the fresh console handler


class TestSetupWandb:
    def test_disabled_returns_none(self, capsys: pytest.CaptureFixture[str]) -> None:
        assert setup_wandb(project="p", config={}, enabled=False) is None
        assert "disabled" in capsys.readouterr().out

    def test_initializes_and_returns_wandb_module(self, capsys: pytest.CaptureFixture[str]) -> None:
        fake_wandb = MagicMock()
        with patch.dict(sys.modules, {"wandb": fake_wandb}):
            result = setup_wandb(project="proj", config={"lr": 1}, entity="team", run_name="r1")

        assert result is fake_wandb
        fake_wandb.init.assert_called_once_with(
            project="proj", entity="team", name="r1", config={"lr": 1}, reinit=True
        )
        assert "W&B initialized" in capsys.readouterr().out

    def test_import_error_returns_none(self, capsys: pytest.CaptureFixture[str]) -> None:
        with patch.dict(sys.modules, {"wandb": None}):
            assert setup_wandb(project="p", config={}) is None
        assert "not installed" in capsys.readouterr().out

    def test_init_failure_returns_none(self, capsys: pytest.CaptureFixture[str]) -> None:
        fake_wandb = MagicMock()
        fake_wandb.init.side_effect = RuntimeError("api key missing")
        with patch.dict(sys.modules, {"wandb": fake_wandb}):
            assert setup_wandb(project="p", config={}) is None
        assert "Failed to initialize" in capsys.readouterr().out


class TestSetupTensorboard:
    def test_disabled_creates_nothing(
        self, tmp_path: Any, capsys: pytest.CaptureFixture[str]
    ) -> None:
        log_dir = tmp_path / "tb"
        setup_tensorboard(str(log_dir), enabled=False)
        assert not log_dir.exists()
        assert capsys.readouterr().out == ""

    def test_enabled_creates_dir(self, tmp_path: Any, capsys: pytest.CaptureFixture[str]) -> None:
        # Widen the rich console so the long tmp path is not wrapped mid-token.
        old_width = log_mod.console.width
        log_mod.console.width = 300
        try:
            log_dir = tmp_path / "tb"
            setup_tensorboard(str(log_dir))
            assert log_dir.exists()
            out = capsys.readouterr().out
            assert "TensorBoard logs" in out
            assert "tensorboard --logdir" in out
        finally:
            log_mod.console.width = old_width


class TestLogMetrics:
    def test_prefix_and_step_forwarded_to_wandb(self, capsys: pytest.CaptureFixture[str]) -> None:
        wandb_run = MagicMock()

        log_metrics({"loss": 0.5, "acc": 0.9}, step=10, prefix="train/", wandb_run=wandb_run)

        wandb_run.log.assert_called_once_with({"train/loss": 0.5, "train/acc": 0.9}, step=10)
        out = capsys.readouterr().out
        assert "Step 10" in out
        assert "loss: 0.5000" in out

    def test_no_prefix_no_step(self, capsys: pytest.CaptureFixture[str]) -> None:
        log_metrics({"loss": 1.0})
        out = capsys.readouterr().out
        assert "Step" not in out
        assert "loss: 1.0000" in out

    def test_without_wandb_run_does_not_raise(self) -> None:
        log_metrics({"loss": 1.0}, step=1)  # wandb_run=None branch


class TestLogGpuMemory:
    @patch("src.utils.logging.log_metrics")
    def test_cuda_branch(self, mock_log: MagicMock) -> None:
        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.memory_allocated", return_value=2 * 1024**3),
            patch("torch.cuda.memory_reserved", return_value=3 * 1024**3),
        ):
            wandb_run = MagicMock()
            log_gpu_memory(5, wandb_run=wandb_run)

        mock_log.assert_called_once_with(
            {"gpu_allocated_gb": 2.0, "gpu_reserved_gb": 3.0},
            step=5,
            prefix="memory/",
            wandb_run=wandb_run,
        )

    @patch("src.utils.logging.log_metrics")
    def test_mps_branch(self, mock_log: MagicMock) -> None:
        with (
            patch("torch.cuda.is_available", return_value=False),
            patch("torch.backends.mps.is_available", return_value=True),
            patch("torch.mps.current_allocated_memory", return_value=1024**3),
        ):
            log_gpu_memory(7)

        mock_log.assert_called_once_with(
            {"mps_allocated_gb": 1.0}, step=7, prefix="memory/", wandb_run=None
        )

    @patch("src.utils.logging.log_metrics")
    def test_no_accelerator_logs_nothing(self, mock_log: MagicMock) -> None:
        with (
            patch("torch.cuda.is_available", return_value=False),
            patch("torch.backends.mps.is_available", return_value=False),
        ):
            log_gpu_memory(1)
        mock_log.assert_not_called()

    @patch("src.utils.logging.log_metrics", side_effect=RuntimeError("boom"))
    def test_exception_is_swallowed_with_warning(
        self, _mock_log: MagicMock, capsys: pytest.CaptureFixture[str]
    ) -> None:
        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.memory_allocated", return_value=0),
            patch("torch.cuda.memory_reserved", return_value=0),
        ):
            log_gpu_memory(1)  # must not raise
        assert "Could not log GPU memory" in capsys.readouterr().out


class TestPrintHelpers:
    def test_print_table_renders_headers_and_rows(self, capsys: pytest.CaptureFixture[str]) -> None:
        print_table(headers=["Metric", "Value"], rows=[["Loss", "2.345"]], title="Metrics")
        out = capsys.readouterr().out
        assert "Metric" in out
        assert "2.345" in out
        assert "Metrics" in out

    def test_print_training_summary_renders_config(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        print_training_summary({"learning_rate": 0.0002, "epochs": 3})
        out = capsys.readouterr().out
        assert "Training Configuration" in out
        assert "learning_rate" in out
        assert "0.0002" in out
