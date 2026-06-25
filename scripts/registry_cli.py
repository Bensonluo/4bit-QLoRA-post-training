#!/usr/bin/env python3
"""MLflow Model Registry CLI: list, inspect (with lineage), register, transition stages.

Operates on the local MLflow file store (./outputs/mlruns by default). Use this to
manage the model lifecycle without re-running training.

Commands:
    list         — all registered models and their versions
    info         — deep inspect a version: lineage run, params, metrics
    register     — register an already-merged model dir as a new version
    transition   — move a version to Staging / Production / Archived

Examples:
    # List everything
    python scripts/registry_cli.py list

    # List versions of one model
    python scripts/registry_cli.py list --model-name Qwen3-1.7B-QLoRA

    # Inspect a version with lineage
    python scripts/registry_cli.py info --model-name Qwen3-1.7B-QLoRA --version 3

    # Promote to Production
    python scripts/registry_cli.py transition \\
        --model-name Qwen3-1.7B-QLoRA --version 3 --stage Production

    # Register a merged dir manually
    python scripts/registry_cli.py register \\
        --model-dir outputs/merged/run-xxx --name Qwen3-1.7B-QLoRA
"""

import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.base import LoggingConfig
from src.tracking import get_tracker

app = typer.Typer(
    name="registry",
    help="MLflow Model Registry management CLI",
    add_completion=False,
)
console = Console()


def _get_tracker(tracking_uri: str | None = None):
    """Build a LoggingConfig and fetch the tracker."""
    cfg = LoggingConfig(use_mlflow=True)
    if tracking_uri:
        cfg.mlflow_tracking_uri = tracking_uri
    return get_tracker(cfg)


@app.command()
def list(
    model_name: str | None = typer.Option(
        None, "--model-name", "-m", help="Filter to one registered model name"
    ),
    tracking_uri: str | None = typer.Option(
        None, "--tracking-uri", help="MLflow tracking URI (default: ./outputs/mlruns)"
    ),
):
    """List registered models and their versions."""
    tracker = _get_tracker(tracking_uri)
    if not tracker.active:
        console.print("[red]✗ MLflow not active. Is it installed?[/red]")
        raise typer.Exit(1)

    versions = tracker.search_model_versions(name=model_name)
    if not versions:
        console.print("[yellow]No model versions found.[/yellow]")
        return

    table = Table(title="Model Registry", show_lines=True)
    table.add_column("Name", style="cyan", no_wrap=True)
    table.add_column("Version", style="white", justify="right")
    table.add_column("Stage", style="magenta")
    table.add_column("Run ID", style="dim")
    table.add_column("Status", style="green")

    for v in versions:
        stage = v.get("current_stage", "None")
        stage_color = {"Production": "bold green", "Staging": "yellow", "Archived": "dim"}.get(
            stage, "white"
        )
        table.add_row(
            v.get("name", "?"),
            str(v.get("version", "?")),
            f"[{stage_color}]{stage}[/{stage_color}]",
            (v.get("run_id") or "—")[:8],
            v.get("status", "?"),
        )

    console.print(table)


@app.command()
def info(
    model_name: str = typer.Option(..., "--model-name", "-m"),
    version: str = typer.Option(..., "--version", "-v"),
    tracking_uri: str | None = typer.Option(None, "--tracking-uri"),
):
    """Deep-inspect a model version: shows lineage run, params, and metrics."""
    tracker = _get_tracker(tracking_uri)
    if not tracker.active:
        console.print("[red]✗ MLflow not active.[/red]")
        raise typer.Exit(1)

    import mlflow
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    client = mlflow.tracking.MlflowClient()
    try:
        mv = client.get_model_version(model_name, version)
    except Exception as e:
        console.print(f"[red]✗ Version not found: {e}[/red]")
        raise typer.Exit(1) from None

    console.print(Panel.fit(
        f"[bold cyan]{model_name} v{version}[/bold cyan]\n"
        f"Stage: {mv.current_stage}\n"
        f"Status: {mv.status}\n"
        f"Run ID: {mv.run_id or '—'}\n"
        f"Source: {mv.source}\n"
        f"Created: {mv.creation_timestamp}",
        border_style="cyan",
    ))

    if mv.run_id:
        run = client.get_run(mv.run_id)
        console.print("\n[bold]Lineage Run Parameters:[/bold]")
        params_table = Table(show_header=False, box=None)
        for k, v in sorted(run.data.params.items()):
            params_table.add_row(f"[dim]{k}[/dim]", str(v))
        console.print(params_table)

        if run.data.metrics:
            console.print("\n[bold]Lineage Run Metrics:[/bold]")
            metrics_table = Table(show_header=False, box=None)
            for k, v in sorted(run.data.metrics.items()):
                metrics_table.add_row(f"[dim]{k}[/dim]", f"{v:.4f}" if isinstance(v, float) else str(v))
            console.print(metrics_table)
    else:
        console.print("[yellow]\nNo lineage run attached.[/yellow]")


@app.command()
def transition(
    model_name: str = typer.Option(..., "--model-name", "-m"),
    version: str = typer.Option(..., "--version", "-v"),
    stage: str = typer.Option(..., "--stage", "-s", help="Staging / Production / Archived"),
    tracking_uri: str | None = typer.Option(None, "--tracking-uri"),
):
    """Transition a model version to a new stage."""
    if stage not in ("Staging", "Production", "Archived"):
        console.print(f"[red]✗ stage must be Staging/Production/Archived, got '{stage}'[/red]")
        raise typer.Exit(1)

    tracker = _get_tracker(tracking_uri)
    if not tracker.active:
        console.print("[red]✗ MLflow not active.[/red]")
        raise typer.Exit(1)

    tracker.transition_model_stage(name=model_name, version=version, stage=stage)
    console.print(f"[green]✓ {model_name} v{version} → {stage}[/green]")


@app.command()
def register(
    model_dir: str = typer.Option(..., "--model-dir", "-d", help="Merged model directory"),
    name: str = typer.Option(..., "--name", "-n", help="Registered model name"),
    stage: str = typer.Option("Staging", "--stage", "-s"),
    tracking_uri: str | None = typer.Option(None, "--tracking-uri"),
):
    """Register an already-merged model directory as a new model version.

    Unlike the automatic in-training registration, this does NOT need a live run —
    it creates a fresh run, logs the model, and registers it. Useful for registering
    a model you merged with scripts/merge_adapter.py earlier.
    """
    tracker = _get_tracker(tracking_uri)
    if not tracker.active:
        console.print("[red]✗ MLflow not active.[/red]")
        raise typer.Exit(1)

    if not Path(model_dir).exists():
        console.print(f"[red]✗ model-dir not found: {model_dir}[/red]")
        raise typer.Exit(1)

    console.print(f"[cyan]Logging {model_dir} to a new MLflow run...[/cyan]")
    with tracker._mlflow.start_run(run_name=f"register-{name}") as _run:
        tracker.log_params({"registered_via": "registry_cli", "source_dir": model_dir})
        model_uri = tracker.log_model(model_dir=model_dir, artifact_path="model")
        if not model_uri:
            console.print("[red]✗ log_model failed[/red]")
            raise typer.Exit(1)
        info = tracker.register_model(model_uri=model_uri, name=name)
        if stage != "None" and info:
            tracker.transition_model_stage(name=info["name"], version=str(info["version"]), stage=stage)
            info["current_stage"] = stage

    console.print(
        f"[green]✓ Registered {info['name']} v{info['version']} ({info['current_stage']})[/green]"
    )


if __name__ == "__main__":
    app()
