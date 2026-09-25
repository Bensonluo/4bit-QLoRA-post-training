"""Unit tests for TrainingRunner (script map, status persistence, pid liveness)."""

import os
from unittest.mock import MagicMock

import pytest

from src.tracking.runner import SCRIPTS, TrainingRunner, _pid_alive


class TestScriptMap:
    def test_all_techniques_present(self) -> None:
        assert {"sft", "dpo", "grpo", "domain"} <= set(SCRIPTS)

    def test_unknown_technique_raises(self, tmp_path: object) -> None:
        runner = TrainingRunner(project_root=str(tmp_path))
        with pytest.raises(ValueError, match="Unknown technique"):
            runner.launch_training(technique="bogus", config_dict={}, run_name="r1")


class TestGetStatus:
    def _proc(self, returncode: int | None) -> MagicMock:
        proc = MagicMock()
        proc.poll.return_value = returncode
        return proc

    def test_running(self, tmp_path: object) -> None:
        runner = TrainingRunner(project_root=str(tmp_path))
        runner._active["r1"] = self._proc(None)
        assert runner.get_status("r1") == "running"

    def test_finished_records_returncode(self, tmp_path: object) -> None:
        runner = TrainingRunner(project_root=str(tmp_path))
        runner._active["r1"] = self._proc(0)
        assert runner.get_status("r1") == "finished"
        assert runner._run_meta["r1"]["returncode"] == 0

    def test_failed(self, tmp_path: object) -> None:
        runner = TrainingRunner(project_root=str(tmp_path))
        runner._active["r1"] = self._proc(1)
        assert runner.get_status("r1") == "failed"
        assert runner._run_meta["r1"]["returncode"] == 1

    def test_returncode_survives_restart(self, tmp_path: object) -> None:
        runner = TrainingRunner(project_root=str(tmp_path))
        runner._active["r1"] = self._proc(0)
        runner.get_status("r1")

        # Simulate a UI restart: fresh runner reloads persisted meta only.
        runner2 = TrainingRunner(project_root=str(tmp_path))
        assert runner2.get_status("r1") == "finished"

    def test_running_survives_restart_via_pid(self, tmp_path: object) -> None:
        runner = TrainingRunner(project_root=str(tmp_path))
        runner._run_meta["r1"] = {"pid": os.getpid()}
        runner._save_meta()

        runner2 = TrainingRunner(project_root=str(tmp_path))
        if os.name == "posix":  # pid liveness is a POSIX-only fallback
            assert runner2.get_status("r1") == "running"

    def test_stale_pid_without_returncode_is_unknown(self, tmp_path: object) -> None:
        runner = TrainingRunner(project_root=str(tmp_path))
        runner._run_meta["r1"] = {"pid": 999_999_999}  # dead pid
        runner._save_meta()

        runner2 = TrainingRunner(project_root=str(tmp_path))
        assert runner2.get_status("r1") == "unknown"

    def test_unknown_run(self, tmp_path: object) -> None:
        runner = TrainingRunner(project_root=str(tmp_path))
        assert runner.get_status("nope") == "unknown"


class TestListActive:
    def test_lists_only_running(self, tmp_path: object) -> None:
        runner = TrainingRunner(project_root=str(tmp_path))
        running = MagicMock()
        running.poll.return_value = None
        runner._active["alive"] = running
        # launch_training always writes meta alongside _active — mirror that here.
        runner._run_meta["alive"] = {"pid": 1}
        done = MagicMock()
        done.poll.return_value = 0
        runner._active["done"] = done
        runner._run_meta["done"] = {"pid": 2}
        runner.get_status("done")  # records returncode

        assert runner.list_active() == ["alive"]

    def test_after_restart_uses_pid_liveness(self, tmp_path: object) -> None:
        runner = TrainingRunner(project_root=str(tmp_path))
        runner._run_meta["r1"] = {"pid": os.getpid()}
        runner._save_meta()

        runner2 = TrainingRunner(project_root=str(tmp_path))
        if os.name == "posix":
            assert runner2.list_active() == ["r1"]


class TestPidAlive:
    def test_self_alive_posix(self) -> None:
        if os.name == "posix":
            assert _pid_alive(os.getpid()) is True

    def test_dead_pid(self) -> None:
        assert _pid_alive(999_999_999) is False
