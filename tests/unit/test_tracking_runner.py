"""Unit tests for the UI-driven training subprocess manager (src/tracking/runner.py).

subprocess.Popen is patched everywhere — no real training launches. The
runner's project_root points at tmp_path so all meta/config/log files are
isolated per test.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from src.tracking.runner import SCRIPTS, TrainingRunner, _pid_alive


def _fake_proc(
    pid: int = 4242,
    poll_value: int | None = None,
    returncode: int | None = None,
) -> MagicMock:
    proc = MagicMock()
    proc.pid = pid
    proc.poll.return_value = poll_value
    proc.returncode = returncode
    return proc


class TestPidAlive:
    def test_own_pid_is_alive(self) -> None:
        assert _pid_alive(os.getpid()) is True

    def test_missing_process_is_dead(self) -> None:
        with patch("os.kill", side_effect=ProcessLookupError()):
            assert _pid_alive(999999) is False

    def test_permission_error_means_alive(self) -> None:
        with patch("os.kill", side_effect=PermissionError()):
            assert _pid_alive(1) is True

    def test_non_posix_reports_dead(self) -> None:
        with patch("src.tracking.runner.os.name", "nt"):
            # os.kill on Windows would terminate the target — must not call it.
            with patch("os.kill") as mock_kill:
                assert _pid_alive(123) is False
            mock_kill.assert_not_called()


class TestInit:
    def test_creates_configs_dir_and_empty_meta(self, tmp_path: Path) -> None:
        runner = TrainingRunner(project_root=str(tmp_path))
        assert (tmp_path / "outputs" / "configs").is_dir()
        assert runner._run_meta == {}

    def test_loads_persisted_meta(self, tmp_path: Path) -> None:
        meta_file = tmp_path / "outputs" / ".run_meta.json"
        meta_file.parent.mkdir(parents=True)
        meta_file.write_text(json.dumps({"run-a": {"pid": 1}}))

        runner = TrainingRunner(project_root=str(tmp_path))
        assert "run-a" in runner._run_meta

    def test_corrupted_meta_falls_back_to_empty(self, tmp_path: Path) -> None:
        meta_file = tmp_path / "outputs" / ".run_meta.json"
        meta_file.parent.mkdir(parents=True)
        meta_file.write_text("{not json")

        runner = TrainingRunner(project_root=str(tmp_path))
        assert runner._run_meta == {}


class TestLaunchTraining:
    def test_unknown_technique_rejected(self, tmp_path: Path) -> None:
        runner = TrainingRunner(project_root=str(tmp_path))
        with pytest.raises(ValueError, match="Unknown technique"):
            runner.launch_training("rlhf", config_dict={}, run_name="r1")

    @patch("src.tracking.runner.subprocess.Popen")
    def test_launch_writes_config_and_starts_subprocess(
        self, mock_popen: MagicMock, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("HF_ENDPOINT", raising=False)
        mock_popen.return_value = _fake_proc(pid=777)
        runner = TrainingRunner(project_root=str(tmp_path))

        run_id = runner.launch_training(
            "sft",
            config_dict={"model": {"name": "qwen"}, "logging": {"mlflow_run_name": "x"}},
            run_name="my-run",
        )

        assert run_id == "my-run"
        # Config file written with use_mlflow force-enabled, other logging kept.
        config_path = tmp_path / "outputs" / "configs" / "my-run.yaml"
        saved = yaml.safe_load(config_path.read_text())
        assert saved["model"]["name"] == "qwen"
        assert saved["logging"]["use_mlflow"] is True
        assert saved["logging"]["mlflow_run_name"] == "x"
        # Command targets the script with the config.
        cmd = mock_popen.call_args.args[0]
        assert cmd[0] == sys.executable
        assert cmd[1].endswith(SCRIPTS["sft"])
        assert cmd[2:] == ["--config", str(config_path)]
        # Env hardening: unbuffered output + China mirror default.
        env = mock_popen.call_args.kwargs["env"]
        assert env["PYTHONUNBUFFERED"] == "1"
        assert env["HF_ENDPOINT"] == "https://hf-mirror.com"
        # Meta persisted with pid.
        meta = runner._run_meta["my-run"]
        assert meta["pid"] == 777
        assert meta["technique"] == "sft"
        persisted = json.loads((tmp_path / "outputs" / ".run_meta.json").read_text())
        assert persisted["my-run"]["pid"] == 777

    @patch("src.tracking.runner.subprocess.Popen")
    def test_existing_hf_endpoint_not_overwritten(
        self, mock_popen: MagicMock, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("HF_ENDPOINT", "https://custom-mirror")
        mock_popen.return_value = _fake_proc()
        runner = TrainingRunner(project_root=str(tmp_path))

        runner.launch_training("dpo", config_dict={}, run_name="r")

        env = mock_popen.call_args.kwargs["env"]
        assert env["HF_ENDPOINT"] == "https://custom-mirror"

    @patch("src.tracking.runner.subprocess.Popen")
    def test_logging_dict_created_when_absent(self, mock_popen: MagicMock, tmp_path: Path) -> None:
        mock_popen.return_value = _fake_proc()
        runner = TrainingRunner(project_root=str(tmp_path))

        runner.launch_training("grpo", config_dict={}, run_name="r")

        saved = yaml.safe_load((tmp_path / "outputs" / "configs" / "r.yaml").read_text())
        assert saved["logging"]["use_mlflow"] is True


class TestLaunchEval:
    """页内评测启动器(R113):launch_eval 走领域评测子进程,与训练同池
    (.run_meta.json)——eval 行进 Activity 列表,30s fragment 门
    (list_active)免费看见评测进度。Popen 依旧全打桩(R104 教义)。"""

    @patch("src.tracking.runner.subprocess.Popen")
    def test_cmd_shape_no_config_flag_cwd_root(
        self, mock_popen: MagicMock, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("HF_ENDPOINT", raising=False)
        mock_popen.return_value = _fake_proc(pid=888)
        runner = TrainingRunner(project_root=str(tmp_path))
        runner._run_meta["src-run"] = {"config_path": "/x/src.yaml", "pid": 1, "returncode": 0}

        rid = runner.launch_eval("src-run", "/models/adapter")

        assert rid == "eval-src-run"
        cmd = mock_popen.call_args.args[0]
        assert cmd[0] == sys.executable
        assert cmd[1:3] == ["-m", "domains.medical_entity.evaluate"]
        assert "--model-path" in cmd and "/models/adapter" in cmd
        # domain evaluate.py 是 argparse,无 --config 选项——挂了即死
        assert "--config" not in cmd
        # 评测脚本以相对路径读 domains/medical_entity/data/(evaluate.py:72),
        # cwd 必须是项目根,否则 seen/unseen 分析静默退化
        assert mock_popen.call_args.kwargs["cwd"] == str(tmp_path)
        # env 惯例与训练一致:unbuffered + 国内镜像默认(用户环境未设时)
        env = mock_popen.call_args.kwargs["env"]
        assert env["PYTHONUNBUFFERED"] == "1"
        assert env["HF_ENDPOINT"] == "https://hf-mirror.com"

    @patch("src.tracking.runner.subprocess.Popen")
    def test_meta_shape_technique_lineage_and_log(
        self, mock_popen: MagicMock, tmp_path: Path
    ) -> None:
        mock_popen.return_value = _fake_proc(pid=888)
        runner = TrainingRunner(project_root=str(tmp_path))
        runner._run_meta["src-run"] = {"config_path": "/x/src.yaml", "pid": 1, "returncode": 0}

        runner.launch_eval("src-run", "/models/adapter")

        meta = runner._run_meta["eval-src-run"]
        assert meta["technique"] == "medical_eval"
        # lineage:eval 行拷贝源训练 run 的 config 路径(面板/审计兜底)
        assert meta["config_path"] == "/x/src.yaml"
        assert meta["model_path"] == "/models/adapter"
        assert meta["log_path"].endswith("eval-src-run.log")
        assert meta["pid"] == 888
        persisted = json.loads((tmp_path / "outputs" / ".run_meta.json").read_text())
        assert persisted["eval-src-run"]["technique"] == "medical_eval"

    def test_eval_run_status_transitions_and_fragment_gate_visibility(self, tmp_path: Path) -> None:
        """eval 行与训练同池:running 时 list_active 可见(fragment 30s 节拍
        门的探测口径),退出后落 finished——评测进度的自动刷新不另造机制。"""
        runner = TrainingRunner(project_root=str(tmp_path))
        runner._active["eval-src"] = _fake_proc(poll_value=None)
        runner._run_meta["eval-src"] = {"pid": 1, "technique": "medical_eval"}
        assert runner.get_status("eval-src") == "running"
        assert runner.list_active() == ["eval-src"]

        runner._active["eval-src"] = _fake_proc(poll_value=0, returncode=0)
        assert runner.get_status("eval-src") == "finished"
        assert runner.list_active() == []


class TestGetStatus:
    def _runner(self, tmp_path: Path) -> TrainingRunner:
        return TrainingRunner(project_root=str(tmp_path))

    def test_active_running(self, tmp_path: Path) -> None:
        runner = self._runner(tmp_path)
        runner._active["r1"] = _fake_proc(poll_value=None)
        assert runner.get_status("r1") == "running"

    def test_active_success_persists_exit_code(self, tmp_path: Path) -> None:
        runner = self._runner(tmp_path)
        runner._active["r1"] = _fake_proc(poll_value=0, returncode=0)

        assert runner.get_status("r1") == "finished"
        assert runner._run_meta["r1"]["returncode"] == 0

    def test_active_failure(self, tmp_path: Path) -> None:
        runner = self._runner(tmp_path)
        runner._active["r1"] = _fake_proc(poll_value=3, returncode=3)
        assert runner.get_status("r1") == "failed"

    def test_unknown_run(self, tmp_path: Path) -> None:
        assert self._runner(tmp_path).get_status("nope") == "unknown"

    def test_restarted_ui_finished_run(self, tmp_path: Path) -> None:
        runner = self._runner(tmp_path)
        runner._run_meta["r1"] = {"returncode": 0}
        assert runner.get_status("r1") == "finished"

    def test_restarted_ui_failed_run(self, tmp_path: Path) -> None:
        runner = self._runner(tmp_path)
        runner._run_meta["r1"] = {"returncode": 2}
        assert runner.get_status("r1") == "failed"

    def test_restarted_ui_live_pid(self, tmp_path: Path) -> None:
        runner = self._runner(tmp_path)
        runner._run_meta["r1"] = {"pid": 12345}
        with patch("src.tracking.runner._pid_alive", return_value=True) as mock_alive:
            assert runner.get_status("r1") == "running"
        mock_alive.assert_called_once_with(12345)

    def test_restarted_ui_dead_pid(self, tmp_path: Path) -> None:
        runner = self._runner(tmp_path)
        runner._run_meta["r1"] = {"pid": 12345}
        with patch("src.tracking.runner._pid_alive", return_value=False):
            assert runner.get_status("r1") == "unknown"


class TestLogs:
    def test_get_log_path_from_meta(self, tmp_path: Path) -> None:
        runner = TrainingRunner(project_root=str(tmp_path))
        runner._run_meta["r1"] = {"log_path": "/tmp/x.log"}
        assert runner.get_log_path("r1") == Path("/tmp/x.log")

    def test_get_log_path_unknown_run(self, tmp_path: Path) -> None:
        assert TrainingRunner(project_root=str(tmp_path)).get_log_path("nope") is None

    def test_read_recent_logs_tails_file(self, tmp_path: Path) -> None:
        log = tmp_path / "train.log"
        log.write_text("\n".join(f"line {i}" for i in range(10)))
        runner = TrainingRunner(project_root=str(tmp_path))
        runner._run_meta["r1"] = {"log_path": str(log)}

        tail = runner.read_recent_logs("r1", tail=3)

        assert tail == "line 7\nline 8\nline 9"

    def test_read_recent_logs_missing_file(self, tmp_path: Path) -> None:
        runner = TrainingRunner(project_root=str(tmp_path))
        runner._run_meta["r1"] = {"log_path": str(tmp_path / "nope.log")}
        assert runner.read_recent_logs("r1") == ""


class TestStopTraining:
    def test_terminates_running_proc(self, tmp_path: Path) -> None:
        runner = TrainingRunner(project_root=str(tmp_path))
        proc = _fake_proc(poll_value=None)
        runner._active["r1"] = proc

        runner.stop_training("r1")

        proc.terminate.assert_called_once()
        proc.wait.assert_called_once_with(timeout=10)
        proc.kill.assert_not_called()

    def test_kills_after_wait_timeout(self, tmp_path: Path) -> None:
        runner = TrainingRunner(project_root=str(tmp_path))
        proc = _fake_proc(poll_value=None)
        proc.wait.side_effect = subprocess.TimeoutExpired(cmd="x", timeout=10)
        runner._active["r1"] = proc

        runner.stop_training("r1")

        proc.kill.assert_called_once()

    def test_already_finished_proc_untouched(self, tmp_path: Path) -> None:
        runner = TrainingRunner(project_root=str(tmp_path))
        proc = _fake_proc(poll_value=0, returncode=0)
        runner._active["r1"] = proc

        runner.stop_training("r1")

        proc.terminate.assert_not_called()

    def test_unknown_run_is_noop(self, tmp_path: Path) -> None:
        TrainingRunner(project_root=str(tmp_path)).stop_training("nope")


class TestDeleteRun:
    def test_unknown_run_raises_keyerror(self, tmp_path: Path) -> None:
        runner = TrainingRunner(project_root=str(tmp_path))
        with pytest.raises(KeyError):
            runner.delete_run("nope")

    def test_running_proc_terminated_then_record_removed(self, tmp_path: Path) -> None:
        runner = TrainingRunner(project_root=str(tmp_path))
        proc = _fake_proc(poll_value=None)
        runner._active["r1"] = proc
        runner._run_meta["r1"] = {"pid": proc.pid}

        runner.delete_run("r1")

        proc.terminate.assert_called_once()
        proc.wait.assert_called_once_with(timeout=10)
        assert "r1" not in runner._active
        assert "r1" not in runner._run_meta
        # Deletion is persisted — a fresh runner (UI restart) sees it gone too.
        assert "r1" not in TrainingRunner(project_root=str(tmp_path))._run_meta

    def test_finished_proc_not_terminated(self, tmp_path: Path) -> None:
        runner = TrainingRunner(project_root=str(tmp_path))
        proc = _fake_proc(poll_value=0, returncode=0)
        runner._active["r1"] = proc
        runner._run_meta["r1"] = {"pid": proc.pid}

        runner.delete_run("r1")

        proc.terminate.assert_not_called()
        assert "r1" not in runner._run_meta

    def test_kills_after_wait_timeout(self, tmp_path: Path) -> None:
        runner = TrainingRunner(project_root=str(tmp_path))
        proc = _fake_proc(poll_value=None)
        proc.wait.side_effect = subprocess.TimeoutExpired(cmd="x", timeout=10)
        runner._active["r1"] = proc
        runner._run_meta["r1"] = {"pid": proc.pid}

        runner.delete_run("r1")

        proc.kill.assert_called_once()
        assert "r1" not in runner._run_meta


class TestListAndMeta:
    def test_list_active_filters_by_status(self, tmp_path: Path) -> None:
        runner = TrainingRunner(project_root=str(tmp_path))
        runner._active["live"] = _fake_proc(poll_value=None)
        runner._active["done"] = _fake_proc(poll_value=0, returncode=0)
        runner._run_meta["live"] = {"pid": 1}
        runner._run_meta["done"] = {"pid": 2}

        assert runner.list_active() == ["live"]

    def test_list_all_runs_and_get_run_info(self, tmp_path: Path) -> None:
        runner = TrainingRunner(project_root=str(tmp_path))
        runner._run_meta = {"a": {"pid": 1}, "b": {"pid": 2}}

        assert runner.list_all_runs() == ["a", "b"]
        assert runner.get_run_info("a") == {"pid": 1}
        assert runner.get_run_info("zzz") is None

    def test_record_exit_none_is_skipped(self, tmp_path: Path) -> None:
        runner = TrainingRunner(project_root=str(tmp_path))
        runner._record_exit("r1", None)
        assert "r1" not in runner._run_meta

    def test_cleanup_records_and_prunes_to_last_20(self, tmp_path: Path) -> None:
        runner = TrainingRunner(project_root=str(tmp_path))
        for i in range(22):
            runner._active[f"run-{i:02d}"] = _fake_proc(poll_value=0, returncode=0)

        runner._cleanup_finished()

        # All exits recorded in meta...
        assert all(runner._run_meta[f"run-{i:02d}"]["returncode"] == 0 for i in range(22))
        # ...but only the 20 most recent processes are kept.
        assert "run-00" not in runner._active
        assert "run-01" not in runner._active
        assert len(runner._active) == 20
