#!/usr/bin/env python3
"""Execute training on remote Windows machine via SSH.

This script is designed for the Mac development + Windows training workflow:
- Mac handles data processing and development
- Windows handles GPU training

Usage:
    python scripts/train_remote.py --config configs/finance_sft.yaml
"""

import sys
from pathlib import Path
from typing import Optional, List

import typer
from rich.console import Console

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import RemoteExecutor, check_remote_connection, console

app = typer.Typer(
    name="train-remote",
    help="Execute training on remote Windows machine",
    add_completion=False,
)


@app.command()
def main(
    host: str = typer.Option(
        "windows",
        "--host",
        "-H",
        help="Remote hostname",
    ),
    script: str = typer.Option(
        "scripts/train_sft.py",
        "--script",
        "-s",
        help="Training script to execute",
    ),
    args: List[str] = typer.Argument(
        None,
        help="Arguments to pass to training script",
    ),
    sync_data: bool = typer.Option(
        True,
        "--sync-data/--no-sync",
        help="Sync data directory before training",
    ),
    data_path: str = typer.Option(
        "./data",
        "--data-path",
        help="Local data path to sync",
    ),
    sync_back: Optional[str] = typer.Option(
        None,
        "--sync-back",
        help="Sync outputs back after training (path on remote)",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Show command without executing",
    ):
    """Execute training on remote machine."""

    console.print("\n[bold cyan]=== Remote Training Executor ===[/bold cyan]\n")

    # Check connection
    console.print(f"[cyan]Checking connection to {host}...[/cyan]")
    if not check_remote_connection(host):
        console.print(f"[red]✗ Cannot connect to {host}[/red]")
        console.print("[yellow]Ensure SSH is configured and host is reachable[/yellow]")
        raise typer.Exit(1)

    console.print(f"[green]✓ Connected to {host}[/green]\n")

    # Build script arguments
    if args is None:
        args = []

    # Show what will be executed
    console.print("[bold]Configuration:[/bold]")
    console.print(f"  Host: {host}")
    console.print(f"  Script: {script}")
    console.print(f"  Arguments: {' '.join(args)}")
    console.print(f"  Sync data: {sync_data}")
    if sync_data:
        console.print(f"  Data path: {data_path}")
    console.print(f"  Sync back: {sync_back or 'None'}")
    console.print()

    if dry_run:
        console.print("[yellow]Dry run - not executing[/yellow]")
        console.print(f"[cyan]Command: ssh {host} 'cd {Path.cwd()} && python {script} {' '.join(args)}'[/cyan]")
        return

    try:
        with RemoteExecutor(
            host=host,
            auto_sync=sync_data,
            data_path=data_path,
        ) as executor:
            # Execute training
            executor.train(
                script_path=script,
                args=args,
                sync_back=sync_back,
            )

        console.print("\n[bold green]✓ Remote training completed successfully![/bold green]\n")

    except KeyboardInterrupt:
        console.print("\n[yellow]Remote training interrupted[/yellow]")
        raise typer.Exit(130)
    except Exception as e:
        console.print(f"\n[red]✗ Remote training failed: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def test_connection(
    host: str = typer.Option(
        "windows",
        "--host",
        "-H",
        help="Remote hostname to test",
    ),
):
    """Test connection to remote machine."""

    console.print(f"\n[cyan]Testing connection to {host}...[/cyan]\n")

    if check_remote_connection(host):
        console.print(f"[green]✓ Connection successful to {host}[/green]")

        # Try to get GPU info
        from src.utils import execute_on_remote

        result = execute_on_remote(
            host,
            "nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader",
            capture_output=True,
        )

        if result.returncode == 0:
            console.print(f"\n[bold]Remote GPU:[/bold]")
            console.print(result.stdout.strip())
        else:
            console.print("[yellow]Could not query GPU (nvidia-smi not available?)[/yellow]")
    else:
        console.print(f"[red]✗ Cannot connect to {host}[/red]")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
