#!/usr/bin/env python3
"""Launch the QLoRA Dashboard (MLflow + Streamlit)."""

import os
import subprocess
import sys
from pathlib import Path

import typer
from rich.console import Console

sys.path.insert(0, str(Path(__file__).parent.parent))

app = typer.Typer(name="qlora-dashboard", help="Launch QLoRA Dashboard", add_completion=False)
console = Console()


@app.command()
def main(
    port: int = typer.Option(8501, "--port", "-p", help="Streamlit port"),
    mlflow_port: int = typer.Option(5000, "--mlflow-port", help="MLflow UI port"),
    tracking_uri: str = typer.Option(
        "./outputs/mlruns", "--tracking-uri", help="MLflow tracking URI"
    ),
    mlflow_only: bool = typer.Option(False, "--mlflow-only", help="Start only MLflow server"),
    streamlit_only: bool = typer.Option(False, "--streamlit-only", help="Start only Streamlit"),
):
    """Launch the QLoRA Post-Training Dashboard."""
    project_root = Path(__file__).parent.parent
    ui_path = project_root / "ui" / "app.py"
    processes = []

    try:
        if not streamlit_only:
            console.print(f"[cyan]Starting MLflow server on port {mlflow_port}...[/cyan]")
            console.print(f"  Tracking URI: {tracking_uri}")
            tracking_uri_abs = str(Path(tracking_uri).resolve())
            os.makedirs(tracking_uri_abs, exist_ok=True)
            p_mlflow = subprocess.Popen(
                [
                    sys.executable, "-m", "mlflow", "server",
                    "--host", "0.0.0.0",
                    "--port", str(mlflow_port),
                    "--backend-store-uri", f"file://{tracking_uri_abs}",
                    "--default-artifact-root", f"file://{tracking_uri_abs}/artifacts",
                ],
                cwd=str(project_root),
            )
            processes.append(("MLflow", p_mlflow))
            console.print(f"[green]  MLflow UI: http://localhost:{mlflow_port}[/green]")

        if not mlflow_only:
            console.print(f"[cyan]Starting Streamlit on port {port}...[/cyan]")
            p_streamlit = subprocess.Popen(
                [
                    sys.executable, "-m", "streamlit", "run",
                    str(ui_path),
                    "--server.port", str(port),
                    "--server.headless", "true",
                    "--browser.gatherUsageStats", "false",
                ],
                cwd=str(project_root),
                env={**os.environ, "MLFLOW_TRACKING_URI": f"http://localhost:{mlflow_port}"},
            )
            processes.append(("Streamlit", p_streamlit))
            console.print(f"[green]  Dashboard: http://localhost:{port}[/green]")

        console.print("\n[bold green]Dashboard is running![/bold green]")
        console.print("[yellow]Press Ctrl+C to stop all services.[/yellow]\n")

        # Wait for any process to exit
        for _name, proc in processes:
            proc.wait()

    except KeyboardInterrupt:
        console.print("\n[yellow]Shutting down...[/yellow]")
    finally:
        for _name, proc in processes:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                console.print(f"[dim]  Stopped {_name}[/dim]")
        console.print("[green]Done.[/green]")


if __name__ == "__main__":
    app()
