#!/usr/bin/env python3
"""Durable result wrapper around the existing SFT training implementation."""

import argparse
import json
import os
import signal
import subprocess
import sys
import threading
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"


def mlflow_reference(config, record):
    reference = {"mlflow_run_id": None, "mlflow_tracking_uri": None, "mlflow_status": "unavailable"}
    try:
        from mlflow.tracking import MlflowClient
    except ImportError:
        return reference
    try:
        logging = config["logging"]
        client = MlflowClient(tracking_uri=logging["mlflow_tracking_uri"])
        experiment = client.get_experiment_by_name(logging["mlflow_experiment_name"])
        runs = (
            []
            if experiment is None
            else client.search_runs(
                [experiment.experiment_id],
                filter_string=f"tags.mlflow.runName = '{record['run_id']}'",
                max_results=2,
            )
        )
        if len(runs) == 1:
            run_id = runs[0].info.run_id
            reference.update(
                mlflow_run_id=run_id,
                mlflow_tracking_uri=logging["mlflow_tracking_uri"],
                mlflow_status="recorded",
            )
            client.set_tag(run_id, "workbench.run_id", record["run_id"])
            client.set_tag(run_id, "workbench.dataset_version", record["dataset_version"])
            client.set_tag(run_id, "workbench.config_digest", record["config_digest"])
        else:
            reference["mlflow_status"] = "not_started" if not runs else "ambiguous"
    except Exception as exc:
        print(f"MLflow run lookup unavailable: {exc}", flush=True)
        reference["mlflow_status"] = "lookup_failed"
    return reference


def main():
    import yaml

    from src.workbench.intake_models import IntakeSession
    from src.workbench.sources import content_digest
    from src.workbench.training_preflight import preflight_dataset
    from src.workbench.training_runs import (
        file_digest,
        model_identity,
        training_tokenizer,
        write_json,
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    directory = Path(config.pop("workbench")["run_dir"])
    record = json.loads((directory / "run.json").read_text())
    result = {
        "run_id": record["run_id"],
        "status": "running",
        "pid": os.getpid(),
        "artifacts": {},
        "metrics": {},
        "failure": None,
    }
    result_path = directory / "result.json"
    stage = "preflight"
    stopped = threading.Event()

    def terminate(_signum, _frame):
        raise KeyboardInterrupt("用户停止训练")

    def watch_stop():
        while not stopped.wait(0.5):
            if (directory / "stop.request").exists():
                os.kill(os.getpid(), signal.SIGTERM)
                return

    signal.signal(signal.SIGTERM, terminate)
    threading.Thread(target=watch_stop, daemon=True).start()
    write_json(result_path, result)
    try:
        if content_digest(config) != record["config_digest"]:
            raise ValueError("提交配置与已确认的训练交接不一致。")
        if model_identity(Path(record["model_path"])) != record["model_identity"]:
            raise ValueError("本地模型或 tokenizer 在启动前发生变化。")
        session = IntakeSession.model_validate_json((directory / "session.json").read_text())
        preflight = preflight_dataset(
            session, training_tokenizer(record["model_path"]), config["model"]["max_length"]
        )
        if preflight["status"] == "blocked":
            raise ValueError(
                "工作进程重新验证数据失败：" + json.dumps(preflight["issues"], ensure_ascii=False)
            )
        from config.base import DataConfig, LoggingConfig, LoRAConfig, ModelConfig, TrainingConfig
        from src.training.sft_trainer import run_sft_training

        stage = "training"
        trainer = run_sft_training(
            model_config=ModelConfig(**config["model"]),
            training_config=TrainingConfig(**config["training"]),
            lora_config=LoRAConfig(**config["lora"]),
            data_config=DataConfig(**config["data"]),
            logging_config=LoggingConfig(**config["logging"]),
        )
        result.update(mlflow_reference(config, record))
        stage = "artifacts"
        output = Path(config["training"]["output_dir"])
        weights = output / "adapter_model.safetensors"
        if not weights.is_file():
            weights = output / "adapter_model.bin"
        required = {
            "adapter_config": output / "adapter_config.json",
            "adapter_weights": weights,
            "tokenizer_config": output / "tokenizer_config.json",
        }
        if not all(path.is_file() for path in required.values()):
            raise ValueError("训练退出但缺少完整 adapter/tokenizer 产物，不能标记成功。")
        metrics = {
            key: value
            for log in trainer.trainer.state.log_history
            for key, value in log.items()
            if isinstance(value, (int, float))
        }
        write_json(output / "workbench_metrics.json", metrics)
        required["metrics"] = output / "workbench_metrics.json"
        manifest = {
            "run_id": record["run_id"],
            "dataset_version": record["dataset_version"],
            "config_digest": record["config_digest"],
            "mlflow_run_id": result["mlflow_run_id"],
            "mlflow_tracking_uri": result["mlflow_tracking_uri"],
            "model_identity": record["model_identity"],
            "files": {
                key: {"path": str(path), "sha256": file_digest(path)}
                for key, path in required.items()
            },
            "scope_note": "训练完成和产物可追溯，不代表垂类业务目标已经通过独立评测。",
        }
        write_json(output / "workbench_manifest.json", manifest)
        required["manifest"] = output / "workbench_manifest.json"
        result.update(
            status="succeeded",
            artifacts={key: str(path) for key, path in required.items()},
            metrics=metrics,
        )
    except KeyboardInterrupt:
        result.update(
            status="stopped",
            failure={
                "stage": stage,
                "type": "UserStopped",
                "message": "训练已停止；保留日志与已写出的中间产物。",
            },
        )
    except BaseException as exc:
        from src.workbench.training_recovery import failure_evidence

        traceback.print_exc()
        result.update(
            status="failed",
            failure={
                "stage": stage,
                "type": type(exc).__name__,
                "message": str(exc),
                "technical": failure_evidence(exc),
            },
        )
    finally:
        stopped.set()
        if "mlflow_status" not in result:
            result.update(mlflow_reference(config, record))
        from datetime import datetime, timezone

        result.update(finished_at=datetime.now(timezone.utc).isoformat())
        write_json(result_path, result)
    if (
        result["status"] == "failed"
        and record.get("recover_technical_failures")
        and not record.get("recovery_parent_run_id")
    ):
        from src.workbench.training_recovery import launch_dispatcher

        try:
            launch_dispatcher(directory, record)
        except (ValueError, OSError, ImportError, subprocess.SubprocessError) as exc:
            print(f"Technical recovery dispatch failed: {exc}", flush=True)
    return 0 if result["status"] == "succeeded" else 1


if __name__ == "__main__":
    raise SystemExit(main())
