"""Model Registry orchestration: merge → log_model → register → stage.

This module is the single entry point trainers call after save_model(). It
coordinates the full registration flow so SFT/DPO trainers stay thin:

    adapter_dir ──► (merge) ──► merged_dir ──► log_model ──► register ──► stage
                                                      └──► lineage (run_id)

All steps are guarded by config flags so the default (register_model=False) path
is a complete no-op — existing behavior is unchanged.

Lineage: the registered model version points back to the MLflow run, which
already logged the training params (model config, hyperparams) and metrics
(loss curves, eval scores). So from any model version you can trace:
    version → run → params + metrics → training code (via git SHA tag, optional).
"""

from __future__ import annotations

from typing import Any

from config.base import LoggingConfig
from src.models.merger import merge_adapter_to_dir
from src.utils.logging import console

# Tracker protocol — both MLflowTracker and _NoOpTracker satisfy this surface.
TrackerLike = Any  # avoid importing the tracker classes (cyclic with get_tracker)


def register_trained_model(
    adapter_dir: str,
    tracker: TrackerLike,
    logging_config: LoggingConfig,
    base_model_name: str | None = None,
    model_config: Any = None,
) -> dict[str, Any] | None:
    """End-to-end registration of a just-trained adapter to the Model Registry.

    Steps (all optional via config):
      1. Merge the LoRA adapter into the base model (if logging_config.merge_before_register).
      2. Log the merged model as an MLflow artifact under the current run (lineage link).
      3. Register the artifact as a new model version in the Registry.
      4. Transition the version to logging_config.registry_stage (default Staging).

    No-op when the tracker is inactive or logging_config.register_model is False.
    Safe to call from both SFT and DPO trainers without conditionals — they just
    pass through and this function decides whether to act.

    Args:
        adapter_dir: Directory where trainer.save_model() wrote the adapter + tokenizer.
        tracker: The active tracker (MLflowTracker or _NoOpTracker).
        logging_config: LoggingConfig carrying the registry_* flags.
        base_model_name: Override base model id for merge (defaults: read from adapter config).
        model_config: Optional ModelConfig — used for merge_dtype and to derive the
            registry model name when logging_config.registry_model_name is None.

    Returns:
        Dict {'name', 'version', 'current_stage', 'model_dir'} on success, or None
        when registration is disabled / failed / skipped.
    """
    # Guard 1: feature disabled by config — no-op (backward compatible).
    if not getattr(logging_config, "register_model", False):
        return None

    # Guard 2: tracker not active (mlflow disabled) — can't register without MLflow.
    if not getattr(tracker, "active", False):
        console.print(
            "[yellow]register_model=True but MLflow is not active "
            "(use_mlflow=False). Skipping registration.[/yellow]"
        )
        return None

    # Derive the registered model name: explicit override > model config name > fallback.
    registry_name = (
        logging_config.registry_model_name
        or (getattr(model_config, "name", None) if model_config else None)
        or "qlora-finetuned-model"
    )
    # Sanitize: HF model ids contain '/' which is invalid in registry names.
    registry_name = registry_name.replace("/", "-")

    merge_dtype = getattr(model_config, "merge_dtype", "bfloat16") if model_config else "bfloat16"

    console.print("\n[bold cyan]=== Registering Model to MLflow Registry ===[/bold cyan]")
    console.print(f"  Adapter:  {adapter_dir}")
    console.print(f"  Registry: {registry_name}")
    console.print(f"  Stage:    {logging_config.registry_stage}")

    try:
        # Step 1: Merge adapter into base model (default) OR use adapter dir as-is.
        if logging_config.merge_before_register:
            merged_dir = f"{adapter_dir}/../merged_{registry_name}".replace("/../", "/")
            # Keep merged output adjacent to adapter for traceability, but in a clean dir.
            merged_dir = _resolve_merged_dir(adapter_dir, registry_name)
            model_dir = merge_adapter_to_dir(
                adapter_dir=adapter_dir,
                output_dir=merged_dir,
                base_model_name=base_model_name,
                dtype=merge_dtype,
            )
        else:
            model_dir = adapter_dir
            console.print("[cyan]Skipping merge — registering adapter dir as-is[/cyan]")

        # Step 2: Log the model artifact under the current run (creates lineage link).
        console.print("[cyan]Logging model to MLflow run...[/cyan]")
        model_uri = tracker.log_model(model_dir=model_dir, artifact_path="model")
        if not model_uri:
            console.print("[yellow]⚠ log_model returned no URI — aborting registration[/yellow]")
            return None
        console.print(f"[green]✓ Logged as {model_uri}[/green]")

        # Step 3: Register as a new model version.
        console.print(f"[cyan]Registering to '{registry_name}'...[/cyan]")
        version_info = tracker.register_model(model_uri=model_uri, name=registry_name)
        if not version_info:
            console.print("[yellow]⚠ register_model returned no version — aborting[/yellow]")
            return None
        console.print(
            f"[green]✓ Registered: {version_info['name']} v{version_info['version']}[/green]"
        )

        # Step 4: Transition to the requested stage.
        stage = logging_config.registry_stage
        if stage and stage != "None":
            console.print(f"[cyan]Transitioning to '{stage}'...[/cyan]")
            tracker.transition_model_stage(
                name=version_info["name"],
                version=str(version_info["version"]),
                stage=stage,
            )
            version_info["current_stage"] = stage
            console.print(f"[green]✓ Now in {stage}[/green]")

        version_info["model_dir"] = model_dir
        console.print(
            f"\n[bold green]✓ Model registered: {version_info['name']} "
            f"v{version_info['version']} ({version_info['current_stage']})[/bold green]\n"
        )
        return version_info

    except Exception as e:
        # Registration failures should NOT fail the training run — the model is
        # already saved to disk. Log and return None.
        console.print(f"[red]✗ Registration failed: {e}[/red]")
        console.print("[yellow]Model remains saved on disk; you can register manually via "
                      "scripts/registry_cli.py[/yellow]")
        return None


def _resolve_merged_dir(adapter_dir: str, registry_name: str) -> str:
    """Compute a clean sibling directory for the merged model output."""
    from pathlib import Path

    parent = Path(adapter_dir).resolve().parent
    return str(parent / f"merged_{registry_name}")
