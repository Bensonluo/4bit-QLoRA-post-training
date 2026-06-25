"""Custom training callbacks."""

import time
from typing import Any

from transformers import TrainerCallback

from src.utils import console


class ProgressCallback(TrainerCallback):
    """Callback to display training progress."""

    def __init__(self) -> None:
        """Initialize progress callback."""
        super().__init__()
        self.start_time: float | None = None
        self.last_step = 0

    def on_train_begin(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        """Called when training begins."""
        self.start_time = time.time()
        console.print(f"\n[green]{'='*60}[/green]")
        console.print("[bold green]Training Started[/bold green]")
        console.print(f"[green]{'='*60}[/green]\n")

    def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
        """Called at the end of each step."""
        # Log progress every 50 steps
        if state.global_step - self.last_step >= 50:
            elapsed = time.time() - self.start_time if self.start_time is not None else 0
            steps_per_second = state.global_step / elapsed if elapsed > 0 else 0

            last_loss = state.log_history[-1].get('loss') if state.log_history else None
            loss_str = f"{last_loss:.4f}" if last_loss is not None else "N/A"

            console.print(
                f"[cyan]Step {state.global_step}[/cyan] | "
                f"Loss: {loss_str} | "
                f"Speed: {steps_per_second:.2f} steps/s"
            )

            self.last_step = state.global_step

        return control

    def on_train_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        """Called when training ends."""
        elapsed = time.time() - self.start_time if self.start_time is not None else 0
        console.print(f"\n[green]{'='*60}[/green]")
        console.print("[bold green]Training Complete[/bold green]")
        console.print(f"[green]Total time: {elapsed/60:.1f} minutes[/green]")
        console.print(f"[green]{'='*60}[/green]\n")


class LossCallback(TrainerCallback):
    """Callback to track and log training loss."""

    def __init__(self, log_steps: int = 10) -> None:
        """Initialize loss callback.

        Args:
            log_steps: Log loss every N steps
        """
        super().__init__()
        self.log_steps = log_steps
        self.losses: list[tuple[int, float]] = []

    def on_log(self, args: Any, state: Any, control: Any, logs: dict[str, Any] | None = None, **kwargs: Any) -> None:
        """Called when logs are available."""
        if logs is None:
            logs = {}

        loss = logs.get("loss")
        if loss is not None and state.global_step % self.log_steps == 0:
            self.losses.append((state.global_step, loss))

    def get_losses(self) -> list[tuple[int, float]]:
        """Get collected losses."""
        return self.losses


class EarlyStoppingCallback(TrainerCallback):
    """Callback for early stopping based on evaluation loss."""

    def __init__(
        self,
        early_stopping_patience: int = 3,
        early_stopping_threshold: float = 0.0,
    ) -> None:
        """Initialize early stopping callback.

        Args:
            early_stopping_patience: Stop if no improvement for N evaluations
            early_stopping_threshold: Minimum change to qualify as improvement
        """
        super().__init__()
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_threshold = early_stopping_threshold
        self.early_stopping_counter = 0
        self.best_metric: float | None = None

    def on_evaluate(
        self,
        args: Any,
        state: Any,
        control: Any,
        metrics: dict[str, Any] | None = None,
        **kwargs: Any
    ) -> None:
        """Called after evaluation."""
        metric_to_check = "eval_loss"
        if metrics is None or metric_to_check not in metrics:
            return

        current_metric = metrics[metric_to_check]

        # Check if metric improved
        if self.best_metric is None:
            self.best_metric = current_metric
        else:
            best = self.best_metric
            if (
                current_metric < best - self.early_stopping_threshold
                and metric_to_check == "eval_loss"  # Lower is better
            ):
                # Improvement
                self.best_metric = current_metric
                self.early_stopping_counter = 0
                console.print(f"[green]✓ Evaluation improved: {current_metric:.4f}[/green]")
            else:
                # No improvement
                self.early_stopping_counter += 1
                console.print(
                    f"[yellow]No improvement for {self.early_stopping_counter} evals[/yellow]"
                )

                # Check if should stop
                if self.early_stopping_counter >= self.early_stopping_patience:
                    console.print("\n[red]Early stopping triggered![/red]")
                    control.should_training_stop = True


class MemoryMonitorCallback(TrainerCallback):
    """Callback to monitor GPU memory during training."""

    def __init__(self, log_steps: int = 100) -> None:
        """Initialize memory monitor callback.

        Args:
            log_steps: Log memory every N steps
        """
        super().__init__()
        self.log_steps = log_steps

    def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
        """Log memory usage."""
        if state.global_step % self.log_steps == 0:
            try:
                import torch

                if torch.cuda.is_available():
                    allocated = torch.cuda.memory_allocated() / 1024**3
                    reserved = torch.cuda.memory_reserved() / 1024**3

                    console.print(
                        f"[dim]VRAM: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved[/dim]"
                    )
                elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    allocated = torch.mps.current_allocated_memory() / 1024**3

                    console.print(
                        f"[dim]MPS Memory: {allocated:.2f}GB allocated[/dim]"
                    )
            except Exception:
                pass

        return control


class CheckpointCallback(TrainerCallback):
    """Callback to save checkpoints based on metrics."""

    def __init__(
        self,
        save_strategy: str = "best",
        metric_for_best: str = "eval_loss",
        greater_is_better: bool = False,
    ) -> None:
        """Initialize checkpoint callback.

        Args:
            save_strategy: "best", "last", or "all"
            metric_for_best: Metric to monitor for best model
            greater_is_better: Whether higher metric is better
        """
        super().__init__()
        self.save_strategy = save_strategy
        self.metric_for_best = metric_for_best
        self.greater_is_better = greater_is_better
        self.best_metric: float | None = None

    def on_evaluate(
        self,
        args: Any,
        state: Any,
        control: Any,
        metrics: dict[str, Any] | None = None,
        **kwargs: Any
    ) -> None:
        """Called after evaluation."""
        if metrics is None or self.metric_for_best not in metrics:
            return

        current_metric = metrics[self.metric_for_best]

        if self.save_strategy == "best":
            if self.best_metric is None:
                self.best_metric = current_metric
                self._save_checkpoint(state)
            elif (
                current_metric > self.best_metric
                if self.greater_is_better
                else current_metric < self.best_metric
            ):
                self.best_metric = current_metric
                self._save_checkpoint(state)

    def _save_checkpoint(self, state: Any) -> None:
        """Save checkpoint."""
        checkpoint_folder = f"checkpoint-{state.global_step}"
        output_dir = f"{state.output_dir}/{checkpoint_folder}"
        console.print(f"[cyan]Saving best checkpoint to {output_dir}[/cyan]")
