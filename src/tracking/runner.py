"""Training subprocess manager for UI-driven training launch."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import Any
from uuid import uuid4

import yaml

# Supported techniques → entry scripts. Kept in sync with ui/pages/00_Training_Lab.py.
SCRIPTS = {
    "sft": "scripts/train_sft.py",
    "dpo": "scripts/train_dpo.py",
    "grpo": "scripts/train_grpo.py",
    "domain": "scripts/train_sft.py",
    "workbench_sft": "scripts/workbench_train.py",
}


def _pid_alive(pid: int) -> bool:
    """Best-effort process liveness check.

    POSIX only — on Windows os.kill(pid, 0) would TERMINATE the process
    (TerminateProcess with exit code 0), so we report not-alive there.
    """
    if os.name != "posix":
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # exists but owned by another user
    return True


@contextmanager
def _metadata_lock(path: Path):
    """Serialize the small metadata update shared by UI and CLI processes."""
    with path.open("a+b") as handle:
        if os.name == "posix":
            import fcntl

            fcntl.flock(handle, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)
        else:
            import msvcrt

            if handle.tell() == 0:
                handle.write(b"0")
                handle.flush()
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
            try:
                yield
            finally:
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)


class TrainingRunner:
    """Launch and monitor training runs from the Streamlit UI."""

    def __init__(self, project_root: str | None = None):
        self.project_root = Path(project_root or ".")
        self._active: dict[str, subprocess.Popen] = {}
        self._configs_dir = self.project_root / "outputs" / "configs"
        self._configs_dir.mkdir(parents=True, exist_ok=True)
        self._meta_file = self.project_root / "outputs" / ".run_meta.json"
        self._run_meta: dict[str, dict] = self._load_meta()

    def _load_meta(self) -> dict[str, Any]:
        if self._meta_file.exists():
            # A corrupted meta file just means we start tracking afresh.
            with suppress(json.JSONDecodeError, OSError):
                data: dict[str, Any] = json.loads(self._meta_file.read_text())
                return data
        return {}

    def _save_meta(self, changed: set[str] | None = None, deleted: set[str] | None = None) -> None:
        updates = (
            self._run_meta if changed is None else {key: self._run_meta[key] for key in changed}
        )
        with _metadata_lock(self._meta_file.with_suffix(".lock")):
            latest = self._load_meta()
            latest.update(updates)
            for key in deleted or set():
                latest.pop(key, None)
            temporary = self._meta_file.with_name(f".run_meta.{uuid4().hex}.tmp")
            temporary.write_text(json.dumps(latest, indent=2))
            temporary.replace(self._meta_file)
            self._run_meta = latest

    def launch_training(
        self,
        technique: str,
        config_dict: dict,
        run_name: str,
        *,
        python_executable: str | None = None,
        env_remove: list[str] | None = None,
    ) -> str:
        """Start training as a subprocess. Returns run_id."""
        self._cleanup_finished()

        config_path = self._configs_dir / f"{run_name}.yaml"
        config_dict["logging"] = config_dict.get("logging", {})
        config_dict["logging"]["use_mlflow"] = True
        with open(config_path, "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False)

        script = SCRIPTS.get(technique)
        if script is None:
            raise ValueError(f"Unknown technique: {technique!r}. Supported: {sorted(SCRIPTS)}")
        script_path = self.project_root / script

        env = os.environ.copy()
        for name in env_remove or []:
            env.pop(name, None)
        env["PYTHONUNBUFFERED"] = "1"
        if "HF_ENDPOINT" not in env:
            env["HF_ENDPOINT"] = "https://hf-mirror.com"

        cmd = [python_executable or sys.executable, str(script_path), "--config", str(config_path)]

        log_path = self.project_root / "outputs" / "logs" / f"{run_name}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)

        # The child inherits the fd, so closing the parent's handle right
        # after Popen is safe — and leaves no leaked file object behind.
        with open(log_path, "w") as log_file:
            proc = subprocess.Popen(
                cmd,
                env=env,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                cwd=str(self.project_root),
            )

        self._active[run_name] = proc
        self._run_meta[run_name] = {
            "technique": technique,
            "config_path": str(config_path),
            "log_path": str(log_path),
            "pid": proc.pid,
            "start_time": time.time(),
        }
        self._save_meta(changed={run_name})
        return run_name

    def get_status(self, run_id: str) -> str:
        """Check subprocess status: 'running' | 'finished' | 'failed' | 'unknown'."""
        proc = self._active.get(run_id)
        if proc is None:
            meta = self._run_meta.get(run_id)
            if not meta:
                return "unknown"
            if "returncode" in meta:
                return "finished" if meta["returncode"] == 0 else "failed"
            # UI restart: best-effort liveness via the recorded pid.
            pid = meta.get("pid")
            if pid is not None and _pid_alive(int(pid)):
                return "running"
            return "unknown"
        ret = proc.poll()
        if ret is None:
            return "running"
        self._record_exit(run_id, ret)
        return "finished" if ret == 0 else "failed"

    def get_log_path(self, run_id: str) -> Path | None:
        meta = self._run_meta.get(run_id)
        if meta:
            return Path(meta["log_path"])
        return None

    def read_recent_logs(self, run_id: str, tail: int = 50) -> str:
        """Read last N lines of training log."""
        log_path = self.get_log_path(run_id)
        if log_path is None or not log_path.exists():
            return ""
        with open(log_path) as f:
            lines = f.readlines()
        return "".join(lines[-tail:])

    def stop_training(self, run_id: str) -> None:
        """Terminate a running training subprocess."""
        proc = self._active.get(run_id)
        if proc and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()

    def delete_run(self, run_id: str) -> None:
        """Delete a run record: stop it if still running, then forget it.

        Persisted artifacts (config YAML, log file, MLflow data) are kept —
        deletion here only removes the local job record.

        Raises:
            KeyError: If the run_id is unknown.
        """
        if run_id not in self._run_meta:
            raise KeyError(f"Unknown run: {run_id!r}")
        proc = self._active.get(run_id)
        if proc and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
        self._active.pop(run_id, None)
        del self._run_meta[run_id]
        self._save_meta(changed=set(), deleted={run_id})

    def list_active(self) -> list[str]:
        """Return run_ids of currently running runs."""
        known = dict.fromkeys([*self._run_meta, *self._active])
        return [rid for rid in known if self.get_status(rid) == "running"]

    def _record_exit(self, run_id: str, returncode: int | None) -> None:
        """Persist exit code so status survives UI restarts."""
        if returncode is None:
            return
        meta = self._run_meta.setdefault(run_id, {})
        meta["returncode"] = returncode
        self._save_meta(changed={run_id})

    def list_all_runs(self) -> list[str]:
        """Return all run_ids (active + completed) from persisted meta."""
        return list(self._run_meta.keys())

    def get_run_info(self, run_id: str) -> dict | None:
        return self._run_meta.get(run_id)

    def _cleanup_finished(self) -> None:
        """Record exit codes, then drop old finished processes (keeps last 20)."""
        finished = [rid for rid, proc in self._active.items() if proc.poll() is not None]
        for rid in finished:
            self._record_exit(rid, self._active[rid].returncode)
        for rid in finished[:-20]:
            del self._active[rid]
