"""Utilities for executing training on remote machines.

This module provides functions to run training on remote machines (e.g., Windows for GPU)
while keeping data processing and development on the local machine (e.g., Mac).
"""

import subprocess
import sys
from pathlib import Path
from typing import List, Optional


def execute_on_remote(
    host: str,
    command: str,
    capture_output: bool = False,
    timeout: Optional[int] = None,
) -> subprocess.CompletedProcess:
    """Execute a command on a remote machine via SSH.

    Args:
        host: Remote hostname (e.g., "windows")
        command: Command to execute (will be run in bash/sh)
        capture_output: Whether to capture stdout/stderr
        timeout: Timeout in seconds

    Returns:
        CompletedProcess with return code and output
    """
    full_command = f"ssh {host} '{command}'"

    if capture_output:
        result = subprocess.run(
            full_command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    else:
        result = subprocess.run(
            full_command,
            shell=True,
            timeout=timeout,
        )

    return result


def train_on_remote(
    host: str,
    script_path: str,
    args: List[str],
    sync_data: bool = True,
    data_path: Optional[str] = None,
) -> None:
    """Execute training script on remote machine.

    This function:
    1. (Optional) Syncs data to remote machine
    2. Executes the training script on remote
    3. Streams output back to local console

    Args:
        host: Remote hostname (e.g., "windows")
        script_path: Path to training script (relative to project root)
        args: Command-line arguments for the script
        sync_data: Whether to sync data before training
        data_path: Path to data directory (if sync_data is True)
    """
    # Ensure we're in the project root
    project_root = Path(__file__).parent.parent.parent
    script_full_path = project_root / script_path

    if not script_full_path.exists():
        raise FileNotFoundError(f"Script not found: {script_full_path}")

    # Sync data if requested
    if sync_data and data_path:
        print(f"Syncing data to {host}...")
        data_full_path = project_root / data_path

        if not data_full_path.exists():
            print(f"Warning: Data path not found: {data_full_path}")
        else:
            # Create remote directory
            execute_on_remote(host, f"mkdir -p {data_path}")

            # Sync data using rsync over SSH
            rsync_cmd = [
                "rsync",
                "-avz",
                "--progress",
                str(data_full_path) + "/",
                f"{host}:{data_path}/",
            ]

            print(f"Running: {' '.join(rsync_cmd)}")
            subprocess.run(rsync_cmd, check=True)
            print("Data sync complete!")

    # Build remote command
    # Activate virtual environment and run script
    remote_cmd = f"cd {project_root} && source venv/bin/activate && python {script_path} {' '.join(args)}"

    print(f"\n{'='*60}")
    print(f"Executing training on {host}")
    print(f"Command: {remote_cmd}")
    print(f"{'='*60}\n")

    # Execute on remote (stream output)
    result = execute_on_remote(host, remote_cmd, capture_output=False)

    if result.returncode != 0:
        print(f"\nTraining failed with return code {result.returncode}")
        sys.exit(1)
    else:
        print(f"\n{'='*60}")
        print("Training completed successfully!")
        print(f"{'='*60}\n")


def sync_from_remote(
    host: str,
    remote_path: str,
    local_path: str,
) -> None:
    """Sync files from remote machine to local.

    Useful for retrieving trained models and checkpoints.

    Args:
        host: Remote hostname
        remote_path: Path on remote machine
        local_path: Destination path on local machine
    """
    local_full_path = Path(local_path)
    local_full_path.parent.mkdir(parents=True, exist_ok=True)

    rsync_cmd = [
        "rsync",
        "-avz",
        "--progress",
        f"{host}:{remote_path}/",
        str(local_full_path) + "/",
    ]

    print(f"Syncing from {host}:{remote_path} to {local_path}")
    print(f"Running: {' '.join(rsync_cmd)}")

    subprocess.run(rsync_cmd, check=True)
    print("Sync complete!")


def check_remote_connection(host: str = "windows") -> bool:
    """Check if remote machine is accessible.

    Args:
        host: Remote hostname

    Returns:
        True if connection successful, False otherwise
    """
    try:
        result = execute_on_remote(host, "echo 'Connection successful'", capture_output=True, timeout=5)
        return result.returncode == 0
    except Exception:
        return False


class RemoteExecutor:
    """Context manager for remote execution setup.

    Usage:
        with RemoteExecutor("windows") as executor:
            executor.train("scripts/train_sft.py", ["--epochs", "3"])
    """

    def __init__(
        self,
        host: str,
        auto_sync: bool = True,
        data_path: Optional[str] = None,
    ):
        """Initialize remote executor.

        Args:
            host: Remote hostname
            auto_sync: Whether to auto-sync data
            data_path: Path to sync if auto_sync is True
        """
        self.host = host
        self.auto_sync = auto_sync
        self.data_path = data_path

    def __enter__(self) -> "RemoteExecutor":
        """Check connection and prepare remote."""
        if not check_remote_connection(self.host):
            raise ConnectionError(f"Cannot connect to remote host: {self.host}")

        print(f"Connected to {self.host}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Cleanup after training."""
        print(f"\nRemote execution on {self.host} finished")

    def train(
        self,
        script_path: str,
        args: List[str],
        sync_back: Optional[str] = None,
    ) -> None:
        """Run training on remote.

        Args:
            script_path: Path to training script
            args: Command-line arguments
            sync_back: Optional path to sync back after training
        """
        train_on_remote(
            host=self.host,
            script_path=script_path,
            args=args,
            sync_data=self.auto_sync,
            data_path=self.data_path,
        )

        if sync_back:
            self.sync_outputs(sync_back)

    def sync_outputs(self, remote_path: str, local_path: Optional[str] = None) -> None:
        """Sync outputs from remote to local.

        Args:
            remote_path: Path on remote
            local_path: Local path (default: same as remote)
        """
        dest_path = local_path or remote_path
        sync_from_remote(self.host, remote_path, dest_path)


if __name__ == "__main__":
    # Test remote connection
    print("Checking remote connection...")
    if check_remote_connection("windows"):
        print("✓ Connected to remote machine")

        # Check GPU
        result = execute_on_remote(
            "windows",
            "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader",
            capture_output=True,
        )
        if result.returncode == 0:
            print(f"✓ Remote GPU: {result.stdout.strip()}")
        else:
            print("✗ Could not query GPU")
    else:
        print("✗ Cannot connect to remote machine")
