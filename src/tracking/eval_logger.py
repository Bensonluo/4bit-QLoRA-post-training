"""Bridge between domain eval JSON results and MLflow tracking."""

from __future__ import annotations

import json
from pathlib import Path


def log_eval_to_mlflow(json_path: str | Path, experiment_name: str = "domain-evaluation") -> None:
    """Read an eval_detail_*.json and log each model's metrics to MLflow.

    Creates one MLflow run per model. Safe no-op if mlflow not installed.
    """
    try:
        import mlflow
    except ImportError:
        return

    path = Path(json_path)
    if not path.exists():
        return

    with open(path) as f:
        all_models = json.load(f)

    mlflow.set_experiment(experiment_name)

    for model_data in all_models:
        model_name = model_data.get("model", "unknown")
        with mlflow.start_run(run_name=f"eval-{model_name}"):
            # Tag with source file
            mlflow.set_tag("eval_source", str(path.name))
            mlflow.set_tag("model_name", model_name)

            # Core metrics
            for metric_key in [
                "overall_accuracy", "mrr", "avg_confidence",
                "avg_latency_ms", "throughput_per_sec", "total_time_sec",
            ]:
                val = model_data.get(metric_key)
                if isinstance(val, (int, float)):
                    mlflow.log_metric(metric_key, val)

            # Per-difficulty accuracy
            acc_diff = model_data.get("accuracy_by_difficulty", {})
            for diff, val in acc_diff.items():
                if isinstance(val, (int, float)):
                    mlflow.log_metric(f"accuracy_{diff}", val)

            # Per-entity-type accuracy
            acc_type = model_data.get("accuracy_by_type", {})
            for etype, val in acc_type.items():
                if isinstance(val, (int, float)):
                    mlflow.log_metric(f"accuracy_{etype}", val)

            # Params
            mlflow.log_param("total_samples", model_data.get("total", 0))
            mlflow.log_param("correct", model_data.get("correct", 0))

            # Upload the JSON as artifact
            mlflow.log_artifact(str(path))
