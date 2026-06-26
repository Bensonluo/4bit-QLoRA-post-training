"""MLflow tracking wrapper with config-driven activation."""

from __future__ import annotations

from typing import Any

from config.base import LoggingConfig


def _flatten_dict(d: dict[str, Any], parent_key: str = "", sep: str = ".") -> dict[str, Any]:
    """Flatten nested dict with dot notation: {'model': {'name': 'X'}} → {'model.name': 'X'}."""
    items: list[tuple[str, Any]] = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(_flatten_dict(v, new_key, sep).items())
        elif isinstance(v, (list, tuple)):
            items.append((new_key, ", ".join(str(i) for i in v)))
        else:
            items.append((new_key, v))
    return dict(items)


class _NoOpTracker:
    """Stand-in that silently discards all tracking calls."""

    active: bool = False

    def start_run(self, **_: Any) -> None:
        pass

    def log_metrics(self, *_: Any, **__: Any) -> None:
        pass

    def log_params(self, *_: Any, **__: Any) -> None:
        pass

    def log_artifact(self, *_: Any) -> None:
        pass

    def log_artifacts(self, *_: Any, **__: Any) -> None:
        """No-op batch artifact logging (Registry path uses this)."""
        pass

    def log_model(self, *_: Any, **__: Any) -> Any:
        """No-op model logging — returns None so callers can branch on the result."""
        return None

    def register_model(self, *_: Any, **__: Any) -> Any:
        """No-op model registration."""
        return None

    def transition_model_stage(self, *_: Any, **__: Any) -> None:
        """No-op stage transition."""
        pass

    def search_model_versions(self, *_: Any, **__: Any) -> list[dict[str, Any]]:
        """No-op — empty list of model versions."""
        return []

    def end_run(self) -> None:
        pass

    def search_runs(self, **_: Any) -> list[dict[str, Any]]:
        return []


_NO_OP = _NoOpTracker()


class MLflowTracker:
    """Thin wrapper around MLflow tracking API.

    All methods are safe no-ops when mlflow is not installed or not enabled.
    """

    def __init__(self, tracking_uri: str, experiment_name: str):
        try:
            import mlflow

            mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_experiment(experiment_name)
            self._mlflow = mlflow
            self._active = True
        except ImportError:
            self._mlflow = None  # type: ignore[assignment,unused-ignore]
            self._active = False

    @property
    def active(self) -> bool:
        return self._active

    def start_run(
        self,
        run_name: str | None = None,
        config: dict[str, Any] | None = None,
        tags: dict[str, str] | None = None,
    ) -> str | None:
        """Start an MLflow run and optionally log config params."""
        if not self._active:
            return None
        run = self._mlflow.start_run(run_name=run_name)
        if config:
            flat = _flatten_dict(config)
            self._mlflow.log_params(flat)
        if tags:
            self._mlflow.set_tags(tags)
        run_id: str | None = run.info.run_id
        return run_id

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        if not self._active:
            return
        self._mlflow.log_metrics(metrics, step=step)

    def log_params(self, params: dict[str, Any]) -> None:
        if not self._active:
            return
        flat = (
            _flatten_dict(params) if any(isinstance(v, dict) for v in params.values()) else params
        )
        self._mlflow.log_params(flat)

    def log_artifact(self, path: str) -> None:
        if not self._active:
            return
        self._mlflow.log_artifact(path)

    def log_artifacts(self, local_dir: str, artifact_path: str | None = None) -> None:
        """Log an entire directory of artifacts under the current run.

        Used to attach the merged-model output directory to a run for lineage.
        """
        if not self._active:
            return
        self._mlflow.log_artifacts(local_dir, artifact_path=artifact_path)

    def log_model(
        self,
        model_dir: str,
        artifact_path: str = "model",
        registered_model_name: str | None = None,
    ) -> str | None:
        """Log a merged HuggingFace model dir as an MLflow model artifact.

        Uses mlflow.transformers flavor so the model can later be loaded via
        `mlflow.pyfunc.load_model` for serving, or registered to the Model Registry.

        Args:
            model_dir: Directory containing the merged model (config.json + safetensors + tokenizer).
            artifact_path: Path under the run where the model is logged.
            registered_model_name: If given, also creates a Registry entry in one call
                (uses mlflow's implicit-registration path). Leave None to log only;
                call register_model() separately for explicit staging control.

        Returns:
            model_uri like "runs:/<run_id>/model", or None if inactive.
        """
        if not self._active:
            return None
        from transformers import AutoModelForCausalLM, AutoTokenizer

        # Load the merged model + tokenizer, then let MLflow log them via the
        # transformers flavor (preserves architecture + tokenizer for pyfunc loading).
        # device_map=None keeps it on CPU during logging — cheap and avoids GPU OOM.
        model = AutoModelForCausalLM.from_pretrained(model_dir, device_map=None)
        tokenizer = AutoTokenizer.from_pretrained(model_dir)

        components = {"model": model, "tokenizer": tokenizer}
        kwargs: dict[str, Any] = {"artifact_path": artifact_path}
        if registered_model_name:
            kwargs["registered_model_name"] = registered_model_name

        # mlflow.transformers.log_model returns a ModelInfo with .model_uri.
        model_info = self._mlflow.transformers.log_model(transformers_model=components, **kwargs)
        return getattr(model_info, "model_uri", None) or f"runs:/{self._current_run_id()}/{artifact_path}"

    def register_model(self, model_uri: str, name: str) -> dict[str, Any] | None:
        """Register a logged model artifact to the Model Registry as a new version.

        Args:
            model_uri: URI of the logged model, e.g. "runs:/<run_id>/model".
            name: Registered model name. Created if it doesn't exist.

        Returns:
            Dict {'name', 'version', 'current_stage'}, or None if inactive.
        """
        if not self._active:
            return None
        mv = self._mlflow.register_model(model_uri=model_uri, name=name)
        return {
            "name": mv.name,
            "version": mv.version,
            "current_stage": mv.current_status if hasattr(mv, "current_status") else "None",
            "run_id": getattr(mv, "run_id", None),
            "source": mv.source,
        }

    def transition_model_stage(self, name: str, version: str, stage: str) -> None:
        """Move a model version to a new stage: Staging / Production / Archived."""
        if not self._active:
            return
        client = self._mlflow.tracking.MlflowClient()
        client.transition_model_version_stage(
            name=name, version=version, stage=stage, archive_existing_versions=False
        )

    def search_model_versions(self, name: str | None = None) -> list[dict[str, Any]]:
        """List model versions, optionally filtered by registered model name."""
        if not self._active:
            return []
        client = self._mlflow.tracking.MlflowClient()
        if name:
            # Escape single quotes in the filter string to avoid injection.
            escaped = name.replace("'", "''")
            versions = client.search_model_versions(f"name='{escaped}'")
        else:
            versions = client.search_model_versions()
        return [
            {
                "name": v.name,
                "version": v.version,
                "current_stage": v.current_stage,
                "run_id": v.run_id,
                "creation_timestamp": v.creation_timestamp,
                "status": v.status,
            }
            for v in versions
        ]

    def _current_run_id(self) -> str | None:
        """Return the active run id (best-effort)."""
        if not self._active:
            return None
        run = self._mlflow.active_run()
        return run.info.run_id if run else None

    def end_run(self) -> None:
        if not self._active:
            return
        try:
            self._mlflow.end_run()
        except Exception:
            pass

    def search_runs(self, experiment_name: str | None = None) -> list[dict[str, Any]]:
        """Return runs as list of dicts for dashboard consumption."""
        if not self._active:
            return []
        from mlflow.entities import ViewType

        if experiment_name:
            exp = self._mlflow.get_experiment_by_name(experiment_name)
            if exp is None:
                return []
            exp_id = exp.experiment_id
        else:
            exps = list(self._mlflow.search_experiments())
            if not exps:
                return []
            exp_id = exps[0].experiment_id
        runs = self._mlflow.search_runs(experiment_ids=[exp_id], run_view_type=ViewType.ALL)
        return runs.to_dict("records") if hasattr(runs, "to_dict") else []


_tracker_instance: MLflowTracker | None = None


def get_tracker(logging_config: LoggingConfig | None = None) -> MLflowTracker | _NoOpTracker:
    """Factory: returns a singleton tracker, or a no-op if mlflow disabled."""
    global _tracker_instance

    if logging_config is None or not getattr(logging_config, "use_mlflow", False):
        return _NO_OP

    if _tracker_instance is not None and _tracker_instance.active:
        return _tracker_instance

    _tracker_instance = MLflowTracker(
        tracking_uri=getattr(logging_config, "mlflow_tracking_uri", "./outputs/mlruns"),
        experiment_name=getattr(logging_config, "mlflow_experiment_name", "qlora-post-training"),
    )
    return _tracker_instance

    def log_dataset(
        self,
        dataset_path: str,
        dataset_name: str,
        version: str,
        context: str = "training",
    ) -> None:
        """Log dataset lineage information as params/artifacts.

        Args:
            dataset_path: Local path to the dataset.
            dataset_name: Registered dataset name.
            version: Dataset version id.
            context: Dataset context (training/validation/test).
        """
        if not self._active:
            return
        self._mlflow.log_params(
            {
                f"dataset.{context}.name": dataset_name,
                f"dataset.{context}.version": version,
                f"dataset.{context}.path": dataset_path,
            }
        )
