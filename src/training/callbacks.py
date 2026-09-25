"""Custom training callbacks."""

import logging
import time
from typing import Any

from transformers import TrainerCallback

from src.utils import console

logger = logging.getLogger("qlora")


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
        console.print(f"\n[green]{'=' * 60}[/green]")
        console.print("[bold green]Training Started[/bold green]")
        console.print(f"[green]{'=' * 60}[/green]\n")

    def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
        """Called at the end of each step."""
        # Log progress every 50 steps
        if state.global_step - self.last_step >= 50:
            elapsed = time.time() - self.start_time if self.start_time is not None else 0
            steps_per_second = state.global_step / elapsed if elapsed > 0 else 0

            last_loss = state.log_history[-1].get("loss") if state.log_history else None
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
        console.print(f"\n[green]{'=' * 60}[/green]")
        console.print("[bold green]Training Complete[/bold green]")
        console.print(f"[green]Total time: {elapsed / 60:.1f} minutes[/green]")
        console.print(f"[green]{'=' * 60}[/green]\n")


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

    def on_log(
        self, args: Any, state: Any, control: Any, logs: dict[str, Any] | None = None, **kwargs: Any
    ) -> None:
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
        **kwargs: Any,
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

                    console.print(f"[dim]MPS Memory: {allocated:.2f}GB allocated[/dim]")
            except Exception as exc:
                # Best-effort diagnostic print — never break the training log loop.
                logger.debug("memory probe failed: %s", exc)

        return control


class CheckpointCallback(TrainerCallback):
    """Callback to request checkpoint saves when the tracked metric improves.

    Only ``save_strategy="best"`` acts in this callback: it sets
    ``control.should_save`` (the documented TrainerCallback control switch) so
    the Trainer itself writes the checkpoint — callbacks never receive the
    model at ``on_evaluate``, so saving it here was never possible. Earlier
    revisions printed a fake "Saving checkpoint" line built from
    ``state.output_dir`` (an attribute TrainerState does not have), which
    would have raised AttributeError with a real TrainerState.

    ``"last"`` and ``"all"`` delegate to HF Trainer's own
    ``TrainingArguments.save_strategy`` and are no-ops here.
    """

    def __init__(
        self,
        save_strategy: str = "best",
        metric_for_best: str = "eval_loss",
        greater_is_better: bool = False,
    ) -> None:
        """Initialize checkpoint callback.

        Args:
            save_strategy: "best" requests a save on metric improvement;
                "last"/"all" are no-ops (use TrainingArguments.save_strategy).
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
        **kwargs: Any,
    ) -> None:
        """Called after evaluation; requests a checkpoint save on improvement."""
        if metrics is None or self.metric_for_best not in metrics:
            return

        current_metric = metrics[self.metric_for_best]

        if self.save_strategy != "best":
            return

        improved = self.best_metric is None or (
            current_metric > self.best_metric
            if self.greater_is_better
            else current_metric < self.best_metric
        )
        if improved:
            self.best_metric = current_metric
            control.should_save = True
            console.print(
                f"[cyan]New best {self.metric_for_best}: {current_metric:.4f} "
                f"— checkpoint save requested[/cyan]"
            )


class MFUCallback(TrainerCallback):
    """Logs Model FLOPs Utilization at each training log point.

    Uses TRL >= 1.4's pure helpers (compute_flops_per_token / compute_mfu /
    adjusted_mfu) so the arithmetic matches upstream exactly, including MoE
    (Mixtral, Qwen3-MoE, DeepSeek-V2) and the causal-masking correction.
    Observability-only — this callback never touches training behavior.

    Token throughput is approximated as tokens_per_step * steps / wall time
    between log points (windowed, so warmup stalls age out). tokens_per_step
    counts PADDED tokens (batch * grad-accum * world_size * seq_len), so MFU
    is a lower bound on true compute utilization when sequences run shorter
    than seq_len.

    Degradation is graceful: if the model config lacks fields the helper
    needs (e.g. Qwen2 configs carry no head_dim attribute), the callback
    prints one warning and disables itself — training proceeds unaffected.
    """

    def __init__(
        self,
        model_config: Any,
        seq_len: int,
        tokens_per_step: int,
        world_size: int = 1,
        peak_flops_per_device: float | None = None,
    ) -> None:
        """Initialize MFU logging.

        Args:
            model_config: The model's PreTrainedConfig (drives FLOPs/token).
            seq_len: Sequence length used for the FLOPs estimate.
            tokens_per_step: Padded tokens consumed per optimizer step.
            world_size: Number of data-parallel ranks contributing tokens.
            peak_flops_per_device: One device's peak FLOP/s; None keeps
                TRL's default (9.895e14, H100 bf16 dense).
        """
        super().__init__()
        self.model_config = model_config
        self.seq_len = seq_len
        self.tokens_per_step = tokens_per_step
        self.world_size = world_size
        self._peak_kwargs: dict[str, float] = (
            {"peak_flops_per_device": peak_flops_per_device} if peak_flops_per_device else {}
        )
        self.history: list[tuple[int, float]] = []
        self._fpt: int | None = None
        self._t0: float | None = None
        self._step0 = 0

        try:
            from trl.trainer.utils import compute_flops_per_token

            fpt = compute_flops_per_token(model_config, seq_len)
            # Guard both failure modes: configs that raise (Qwen2 has no
            # head_dim) AND duck-typed garbage that silently computes (a
            # MagicMock config returns a MagicMock, not an int).
            if not isinstance(fpt, int) or fpt <= 0:
                raise TypeError(f"unexpected FLOPs/token estimate: {fpt!r}")
            self._fpt = fpt
        except Exception as exc:
            console.print(f"[yellow]MFU logging disabled — {exc}[/yellow]")

    def on_train_begin(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        """Open the first measurement window."""
        self._t0 = time.monotonic()
        self._step0 = state.global_step

    def on_log(
        self, args: Any, state: Any, control: Any, logs: dict[str, Any] | None = None, **kwargs: Any
    ) -> None:
        """Compute windowed MFU and record it (console + history + logs dict)."""
        if self._fpt is None or self._t0 is None or logs is None:
            return
        elapsed = time.monotonic() - self._t0
        steps = state.global_step - self._step0
        if steps <= 0 or elapsed <= 0:
            return

        from trl.trainer.utils import adjusted_mfu, compute_mfu

        tps = self.tokens_per_step * steps / elapsed
        mfu = adjusted_mfu(
            compute_mfu(self._fpt, tps, self.world_size, **self._peak_kwargs),
            self.model_config,
            self.seq_len,
        )
        self.history.append((state.global_step, mfu))
        console.print(f"[dim]MFU: {mfu:.1f}% ({tps:,.0f} padded tok/s)[/dim]")
        # Best-effort: reaches tracker callbacks that run after this one.
        logs["train_mfu_percent"] = mfu
        # Windowed — reset so the next log point measures only its own span.
        self._t0 = time.monotonic()
        self._step0 = state.global_step
