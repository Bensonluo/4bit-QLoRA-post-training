"""Training subprocess manager for UI-driven training launch."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml


class TrainingRunner:
    """Launch and monitor training runs from the Streamlit UI."""

    def __init__(self, project_root: str | None = None):
        self.project_root = Path(project_root or ".")
        self._active: dict[str, subprocess.Popen] = {}
        self._configs_dir = self.project_root / "outputs" / "configs"
        self._configs_dir.mkdir(parents=True, exist_ok=True)
        self._meta_file = self.project_root / "outputs" / ".run_meta.json"
        self._run_meta: dict[str, dict] = self._load_meta()

    def _load_meta(self) -> dict[str, dict]:
        if self._meta_file.exists():
            try:
                return json.loads(self._meta_file.read_text())
            except (json.JSONDecodeError, OSError):
                pass
        return {}

    def _save_meta(self) -> None:
        self._meta_file.write_text(json.dumps(self._run_meta, indent=2))

    def launch_training(
        self,
        technique: str,
        config_dict: dict,
        run_name: str,
    ) -> str:
        """Start training as a subprocess. Returns run_id."""
        self._cleanup_finished()

        config_path = self._configs_dir / f"{run_name}.yaml"
        config_dict["logging"] = config_dict.get("logging", {})
        config_dict["logging"]["use_mlflow"] = True
        with open(config_path, "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False)

        script_map = {
            "sft": "scripts/train_sft.py",
            "dpo": "scripts/train_dpo.py",
            "domain": "scripts/train_sft.py",
        }
        script = script_map.get(technique, "scripts/train_sft.py")
        script_path = self.project_root / script

        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        if "HF_ENDPOINT" not in env:
            env["HF_ENDPOINT"] = "https://hf-mirror.com"

        cmd = [sys.executable, str(script_path), "--config", str(config_path)]

        log_path = self.project_root / "outputs" / "logs" / f"{run_name}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)

        proc = subprocess.Popen(
            cmd,
            env=env,
            stdout=open(log_path, "w"),
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
        self._save_meta()
        return run_name

    def get_status(self, run_id: str) -> str:
        """Check subprocess status: 'running' | 'finished' | 'failed' | 'unknown'."""
        proc = self._active.get(run_id)
        if proc is None:
            meta = self._run_meta.get(run_id)
            if not meta:
                return "unknown"
            # Check if log file has completion marker
            log_path = Path(meta.get("log_path", ""))
            if log_path.exists() and log_path.stat().st_size > 0:
                return "finished"
            return "unknown"
        ret = proc.poll()
        if ret is None:
            return "running"
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

    def list_active(self) -> list[str]:
        """Return run_ids of currently running processes."""
        self._cleanup_finished()
        active = [
            rid for rid, proc in self._active.items()
            if proc.poll() is None
        ]
        # Also include recently launched runs from persisted meta
        for rid, meta in self._run_meta.items():
            if rid not in self._active:
                log_path = Path(meta.get("log_path", ""))
                if log_path.exists():
                    active.append(rid)
        return active

    def list_all_runs(self) -> list[str]:
        """Return all run_ids (active + completed) from persisted meta."""
        return list(self._run_meta.keys())

    def get_run_info(self, run_id: str) -> dict | None:
        return self._run_meta.get(run_id)

    def _cleanup_finished(self) -> None:
        """Remove long-finished processes from tracking (keeps last 20)."""
        finished = [
            rid for rid, proc in self._active.items()
            if proc.poll() is not None
        ]
        for rid in finished[:-20]:
            del self._active[rid]
