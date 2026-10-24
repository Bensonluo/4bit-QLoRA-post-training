"""Load public repository evidence into the cloud display store; never train."""

from __future__ import annotations

import hashlib
import json
import os
import tarfile
from datetime import datetime
from pathlib import Path


def seed(root: Path) -> None:
    if os.environ.get("TUNESMITH_DISPLAY_ONLY") != "1":
        raise RuntimeError("Seeding is only available in the dedicated display environment.")
    import mlflow
    from mlflow.tracking import MlflowClient

    from src.workbench.baseline_analysis import propose_baseline_analysis
    from src.workbench.demo_task import demo_create_payload
    from src.workbench.intake_service import IntakeService

    directory = root / "outputs" / "display"
    directory.mkdir(parents=True, exist_ok=True)
    marker = directory / "evidence-source.json"
    if marker.exists():
        return
    archive = root / "docs/validation/complete-output-handoff-evidence-2026-10-22.tar.gz"
    evidence_path = root / "docs/validation/complete-output-handoff-live-2026-10-22.json"
    evidence = json.loads(evidence_path.read_text())
    with tarfile.open(archive) as bundle:

        def read(suffix: str):
            member = next(m for m in bundle.getmembers() if m.name.endswith("/" + suffix))
            stream = bundle.extractfile(member)
            if stream is None:
                raise ValueError(f"Missing public evidence: {suffix}")
            return json.load(stream)

        snapshot = read("source-run-snapshot.json")
        history = read("workbench_loss_history.json")
        metrics = read("workbench_metrics.json")
    uri = os.environ.get("MLFLOW_TRACKING_URI", f"sqlite:///{directory / 'mlflow.db'}")
    mlflow.set_tracking_uri(uri)
    client = MlflowClient(tracking_uri=uri)
    experiment = client.get_experiment_by_name("Repository validation")
    experiment_id = (
        experiment.experiment_id
        if experiment
        else client.create_experiment("Repository validation")
    )
    finished = int(datetime.fromisoformat(evidence["training"]["finished_at"]).timestamp() * 1000)
    started = finished - int(evidence["training"]["elapsed_seconds"] * 1000)
    run = client.create_run(
        experiment_id,
        start_time=started,
        tags={
            "mlflow.runName": "r149-recorded-local-training",
            "mlflow.note.content": "Imported historical local validation (2026-10-22); not a cloud training run. Source: docs/validation/complete-output-handoff-live-2026-10-22.json",
            "evidence.source_run_id": snapshot["run_id"],
            "evidence.source": evidence_path.name,
            "evidence.recorded_at": evidence["training"]["finished_at"],
        },
    )
    for group, values in snapshot["config"].items():
        if isinstance(values, dict):
            for key, value in values.items():
                if isinstance(value, (str, int, float, bool)):
                    client.log_param(run.info.run_id, f"{group}.{key}", value)
    for point in history:
        for key, value in point.items():
            if key != "step" and isinstance(value, (int, float)):
                client.log_metric(
                    run.info.run_id, key, float(value), timestamp=finished, step=int(point["step"])
                )
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            client.log_metric(run.info.run_id, key, float(value), timestamp=finished)
    client.set_terminated(run.info.run_id, status="FINISHED", end_time=finished)

    intake = IntakeService(root / "outputs/workbench/intake")
    if not intake.list_sessions():
        payload = demo_create_payload(root)
        if payload is None:
            raise ValueError("Repository example data is missing.")
        goal, data, name, description = payload
        session = intake.create(goal, name, data, data_description=description)
        analysis = propose_baseline_analysis(
            session, target_column="类别", excluded_columns=["编号", "处理结果"]
        )
        intake.apply_analysis(session, analysis, model="deterministic-baseline")
    marker.write_text(
        json.dumps(
            {
                "source": evidence_path.name,
                "sha256": hashlib.sha256(evidence_path.read_bytes()).hexdigest(),
                "source_run_id": snapshot["run_id"],
                "mlflow_run_id": run.info.run_id,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    seed(Path(__file__).resolve().parents[1])
