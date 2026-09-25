"""Unit tests for remote-execution utilities (src/utils/execution.py).

All subprocess/SSH calls are patched — nothing leaves the machine.
"""

from __future__ import annotations

import shlex
import subprocess
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from src.utils import execution
from src.utils.execution import (
    RemoteExecutor,
    check_remote_connection,
    execute_on_remote,
    sync_from_remote,
    train_on_remote,
)

SCRIPT = "scripts/train_sft.py"  # real file — train_on_remote checks existence


def _completed(returncode: int = 0, stdout: str = "") -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(args=[], returncode=returncode, stdout=stdout)


class TestExecuteOnRemote:
    @patch("src.utils.execution.subprocess.run")
    def test_capture_branch_forwards_kwargs(self, mock_run: MagicMock) -> None:
        mock_run.return_value = _completed()
        execute_on_remote("windows", "nvidia-smi", capture_output=True, timeout=5)
        kwargs = mock_run.call_args.kwargs
        assert kwargs["capture_output"] is True
        assert kwargs["text"] is True
        assert kwargs["timeout"] == 5
        assert kwargs["shell"] is True

    @patch("src.utils.execution.subprocess.run")
    def test_stream_branch_omits_capture(self, mock_run: MagicMock) -> None:
        execute_on_remote("windows", "python x.py")
        kwargs = mock_run.call_args.kwargs
        assert "capture_output" not in kwargs
        assert kwargs["shell"] is True

    @patch("src.utils.execution.subprocess.run")
    def test_host_and_command_shell_quoted(self, mock_run: MagicMock) -> None:
        # Simple tokens pass through unquoted; anything with whitespace or
        # quotes gets shlex-escaped as one argument.
        execute_on_remote("windows", "nvidia-smi")
        assert mock_run.call_args.args[0] == "ssh windows nvidia-smi"
        execute_on_remote("windows", "echo 'hi there'")
        assert mock_run.call_args.args[0] == "ssh windows " + shlex.quote("echo 'hi there'")


class TestTrainOnRemote:
    def test_missing_script_raises(self) -> None:
        with pytest.raises(FileNotFoundError, match="Script not found"):
            train_on_remote("windows", "scripts/does-not-exist.py", [])

    @patch("src.utils.execution.execute_on_remote")
    def test_missing_data_path_warns_and_skips_sync(
        self, mock_exec: MagicMock, capsys: Any
    ) -> None:
        mock_exec.return_value = _completed()
        train_on_remote(
            "windows", SCRIPT, ["--epochs", "1"], sync_data=True, data_path="no-such-dir-xyz"
        )
        assert "Warning: Data path not found" in capsys.readouterr().out
        # Only the training command ran — no mkdir/rsync.
        assert mock_exec.call_count == 1

    @patch("src.utils.execution.subprocess.run")
    @patch("src.utils.execution.execute_on_remote")
    def test_sync_branch_rsyncs_data_dir(
        self, mock_exec: MagicMock, mock_run: MagicMock, capsys: Any
    ) -> None:
        mock_exec.return_value = _completed()
        train_on_remote(
            "windows",
            SCRIPT,
            [],
            sync_data=True,
            data_path="config",  # real repo dir
        )
        # mkdir on remote, then rsync, then the training command
        assert mock_exec.call_count == 2
        mkdir_cmd = mock_exec.call_args_list[0].args[1]
        assert mkdir_cmd.startswith("mkdir -p")
        rsync_cmd = mock_run.call_args.args[0]
        assert rsync_cmd[0:3] == ["rsync", "-avz", "--progress"]
        assert rsync_cmd[-1] == "windows:config/"
        assert "Data sync complete" in capsys.readouterr().out

    @patch("src.utils.execution.execute_on_remote")
    def test_remote_command_envelopes_script_and_args(
        self, mock_exec: MagicMock, capsys: Any
    ) -> None:
        mock_exec.return_value = _completed()
        train_on_remote("windows", SCRIPT, ["--lr", "1e-4", "--flag with space"])
        remote_cmd = mock_exec.call_args.args[1]
        assert remote_cmd.startswith("cd ")
        assert "source venv/bin/activate" in remote_cmd
        assert "python scripts/train_sft.py" in remote_cmd
        assert "'--flag with space'" in remote_cmd  # shlex-quoted arg

    @patch("src.utils.execution.execute_on_remote")
    def test_failure_exit_code_calls_sys_exit(self, mock_exec: MagicMock) -> None:
        mock_exec.return_value = _completed(returncode=2)
        with patch("src.utils.execution.sys.exit") as mock_exit:
            train_on_remote("windows", SCRIPT, [])
        mock_exit.assert_called_once_with(1)

    @patch("src.utils.execution.execute_on_remote")
    def test_success_prints_completion(self, mock_exec: MagicMock, capsys: Any) -> None:
        mock_exec.return_value = _completed()
        train_on_remote("windows", SCRIPT, [], sync_data=False)
        assert "Training completed successfully" in capsys.readouterr().out


class TestSyncFromRemote:
    @patch("src.utils.execution.subprocess.run")
    def test_builds_rsync_pull_command(self, mock_run: MagicMock, tmp_path: Any) -> None:
        dest = tmp_path / "outputs" / "run-1"
        sync_from_remote("windows", "outputs/run-1", str(dest))
        assert dest.parent.exists()  # parents created
        cmd = mock_run.call_args.args[0]
        assert cmd[0] == "rsync"
        assert cmd[-2] == "windows:outputs/run-1/"
        assert cmd[-1].endswith("/run-1/")
        assert mock_run.call_args.kwargs == {"check": True}


class TestCheckRemoteConnection:
    @patch("src.utils.execution.execute_on_remote")
    def test_zero_returncode_is_true(self, mock_exec: MagicMock) -> None:
        mock_exec.return_value = _completed()
        assert check_remote_connection("windows") is True
        assert mock_exec.call_args.kwargs["timeout"] == 5

    @patch("src.utils.execution.execute_on_remote")
    def test_nonzero_returncode_is_false(self, mock_exec: MagicMock) -> None:
        mock_exec.return_value = _completed(returncode=255)
        assert check_remote_connection("windows") is False

    @patch("src.utils.execution.execute_on_remote", side_effect=subprocess.TimeoutExpired("ssh", 5))
    def test_exception_is_false(self, _mock_exec: MagicMock) -> None:
        assert check_remote_connection("windows") is False


class TestRemoteExecutor:
    @patch("src.utils.execution.check_remote_connection", return_value=False)
    def test_enter_fails_on_dead_host(self, _mock_check: MagicMock) -> None:
        executor = RemoteExecutor("dead-host")
        with pytest.raises(ConnectionError, match="dead-host"), executor:
            pass

    @patch("src.utils.execution.check_remote_connection", return_value=True)
    def test_lifecycle_train_and_optional_sync_back(
        self, _mock_check: MagicMock, capsys: Any
    ) -> None:
        with (
            patch("src.utils.execution.train_on_remote") as mock_train,
            patch("src.utils.execution.sync_from_remote") as mock_sync,
            RemoteExecutor("windows", auto_sync=False) as executor,
        ):
            executor.train(SCRIPT, ["--epochs", "3"], sync_back="outputs/run-1")

        mock_train.assert_called_once_with(
            host="windows",
            script_path=SCRIPT,
            args=["--epochs", "3"],
            sync_data=False,
            data_path=None,
        )
        mock_sync.assert_called_once()
        out = capsys.readouterr().out
        assert "Connected to windows" in out
        assert "finished" in out

    @patch("src.utils.execution.check_remote_connection", return_value=True)
    def test_train_without_sync_back_skips_sync(self, _mock_check: MagicMock) -> None:
        with (
            patch("src.utils.execution.train_on_remote"),
            patch("src.utils.execution.sync_from_remote") as mock_sync,
            RemoteExecutor("windows") as executor,
        ):
            executor.train(SCRIPT, [])
        mock_sync.assert_not_called()

    @patch("src.utils.execution.sync_from_remote")
    def test_sync_outputs_default_dest(self, mock_sync: MagicMock) -> None:
        RemoteExecutor("windows").sync_outputs("outputs/run-1")
        mock_sync.assert_called_once_with("windows", "outputs/run-1", "outputs/run-1")

    @patch("src.utils.execution.sync_from_remote")
    def test_sync_outputs_explicit_dest(self, mock_sync: MagicMock) -> None:
        RemoteExecutor("windows").sync_outputs("outputs/run-1", "local/dir")
        mock_sync.assert_called_once_with("windows", "outputs/run-1", "local/dir")


def test_module_dunder_main_is_guarded() -> None:
    # The __main__ smoke-test block must never run on import.
    assert execution.__name__ != "__main__"
