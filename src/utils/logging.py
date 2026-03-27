"""Logging utilities for training monitoring."""

import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from rich.console import Console
from rich.logging import RichHandler

# Console for rich output
console = Console()


def setup_logging(
    log_file: Optional[str] = None,
    level: str = "INFO",
    use_rich: bool = True,
) -> logging.Logger:
    """Setup logging configuration.

    Args:
        log_file: Optional path to log file
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        use_rich: Whether to use Rich for colored console output

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger("qlora")
    logger.setLevel(getattr(logging, level.upper()))

    # Clear existing handlers
    logger.handlers.clear()

    # Console handler
    if use_rich:
        console_handler = RichHandler(
            console=console,
            show_time=True,
            show_path=False,
            rich_tracebacks=True,
        )
    else:
        console_handler = logging.StreamHandler(sys.stdout)

    console_handler.setLevel(getattr(logging, level.upper()))
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def setup_wandb(
    project: str,
    config: Dict[str, Any],
    entity: Optional[str] = None,
    run_name: Optional[str] = None,
    enabled: bool = True,
):
    """Initialize Weights & Biases logging.

    Args:
        project: W&B project name
        config: Training configuration dictionary
        entity: W&B entity/team name
        run_name: Custom run name
        enabled: Whether to enable W&B logging
    """
    if not enabled:
        console.print("[yellow]W&B logging disabled[/yellow]")
        return None

    try:
        import wandb

        wandb.init(
            project=project,
            entity=entity,
            name=run_name,
            config=config,
            reinit=True,
        )
        console.print(f"[green]✓ W&B initialized: {project}[/green]")
        return wandb
    except ImportError:
        console.print("[yellow]⚠ wandb not installed. Skip W&B logging.[/yellow]")
        return None
    except Exception as e:
        console.print(f"[red]✗ Failed to initialize W&B: {e}[/red]")
        return None


def setup_tensorboard(log_dir: str, enabled: bool = True) -> None:
    """Setup TensorBoard logging.

    Args:
        log_dir: Directory for TensorBoard logs
        enabled: Whether to enable TensorBoard
    """
    if not enabled:
        return

    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    console.print(f"[green]✓ TensorBoard logs: {log_dir}[/green]")
    console.print(f"[cyan]View: tensorboard --logdir={log_dir}[/cyan]")


def log_metrics(
    metrics: Dict[str, float],
    step: Optional[int] = None,
    prefix: str = "",
    wandb_run=None,
) -> None:
    """Log training metrics.

    Args:
        metrics: Dictionary of metric names and values
        step: Training step number
        prefix: Prefix for metric names (e.g., "train/", "val/")
        wandb_run: Optional W&B run object
    """
    # Add prefix to metrics
    if prefix:
        metrics = {f"{prefix}{k}": v for k, v in metrics.items()}

    # Log to console
    metric_str = " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
    if step is not None:
        console.print(f"Step {step}: {metric_str}")
    else:
        console.print(metric_str)

    # Log to W&B
    if wandb_run is not None:
        wandb_run.log(metrics, step=step)


def log_gpu_memory(step: int, wandb_run=None) -> None:
    """Log GPU memory usage.

    Args:
        step: Training step number
        wandb_run: Optional W&B run object
    """
    try:
        import torch

        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3

            metrics = {
                "gpu_allocated_gb": allocated,
                "gpu_reserved_gb": reserved,
            }

            log_metrics(metrics, step=step, prefix="memory/", wandb_run=wandb_run)

        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            allocated = torch.mps.current_allocated_memory() / 1024**3

            metrics = {
                "mps_allocated_gb": allocated,
            }

            log_metrics(metrics, step=step, prefix="memory/", wandb_run=wandb_run)

    except Exception as e:
        console.print(f"[yellow]Warning: Could not log GPU memory: {e}[/yellow]")


def print_table(headers: list, rows: list, title: str = "") -> None:
    """Print a formatted table.

    Args:
        headers: Column headers
        rows: List of rows (each row is a list of values)
        title: Optional table title
    """
    from rich.table import Table

    table = Table(title=title)

    for header in headers:
        table.add_column(header)

    for row in rows:
        table.add_row(*[str(v) for v in row])

    console.print(table)


def print_training_summary(config: Dict[str, Any]) -> None:
    """Print training configuration summary.

    Args:
        config: Configuration dictionary
    """
    from rich.panel import Panel
    from rich.text import Text

    text = Text()
    text.append("Training Configuration\n", style="bold magenta")
    text.append("=" * 50 + "\n\n")

    for key, value in config.items():
        text.append(f"{key}:", style="cyan")
        text.append(f" {value}\n")

    panel = Panel(text, border_style="magenta")
    console.print(panel)


if __name__ == "__main__":
    # Test logging
    logger = setup_logging(level="INFO")
    logger.info("Test info message")
    logger.warning("Test warning message")

    # Test table printing
    print_table(
        headers=["Metric", "Value"],
        rows=[["Loss", "2.345"], ["Accuracy", "0.89"], ["LR", "0.0002"]],
        title="Training Metrics",
    )
