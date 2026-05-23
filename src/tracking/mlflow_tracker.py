"""MLflow tracking wrapper with config-driven activation."""

from __future__ import annotations

from typing import Any, Optional

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
            self._mlflow = None  # type: ignore[assignment]
            self._active = False

    @property
    def active(self) -> bool:
        return self._active

    def start_run(
        self,
        run_name: Optional[str] = None,
        config: Optional[dict[str, Any]] = None,
        tags: Optional[dict[str, str]] = None,
    ) -> Optional[str]:
        """Start an MLflow run and optionally log config params."""
        if not self._active:
            return None
        run = self._mlflow.start_run(run_name=run_name)
        if config:
            flat = _flatten_dict(config)
            self._mlflow.log_params(flat)
        if tags:
            self._mlflow.set_tags(tags)
        return run.info.run_id

    def log_metrics(self, metrics: dict[str, float], step: Optional[int] = None) -> None:
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

    def end_run(self) -> None:
        if not self._active:
            return
        try:
            self._mlflow.end_run()
        except Exception:
            pass

    def search_runs(self, experiment_name: Optional[str] = None) -> list[dict[str, Any]]:
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


_tracker_instance: Optional[MLflowTracker] = None


def get_tracker(logging_config: Optional[LoggingConfig] = None) -> MLflowTracker | _NoOpTracker:
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
