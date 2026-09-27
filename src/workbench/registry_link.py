"""Model-registry linkage for workbench runs: one edge of the lineage triangle.

从一次 workbench 训练反查它在 MLflow 模型库的注册状态;未注册时给出确切的
注册命令,而不是让用户猜。只读查询,不写注册表。
"""

from __future__ import annotations

from typing import Any


def _tags_or_none(client, run_id: str) -> dict | None:
    """读取运行标签;已删除的运行返回 None,不影响其余版本的查询。"""
    try:
        return client.get_run(run_id).data.tags
    except Exception:
        return None


def run_registration_status(run_id: str, tracking_uri: str) -> dict[str, Any]:
    """Look up registry versions traced to this workbench run (read-only).

    版本可能挂在手工注册创建的 MLflow 运行上;血缘以 workbench.run_id 标签为准,
    因此按标签在注册表内全局反查,而不是只查训练运行本身。
    """
    try:
        from mlflow.tracking import MlflowClient
    except ImportError:
        return {
            "run_id": run_id,
            "status": "mlflow_unavailable",
            "message": "未安装 mlflow,无法查询模型库。",
        }
    try:
        client = MlflowClient(tracking_uri=tracking_uri)
        found = []
        for model in client.search_registered_models(max_results=100):
            for version in client.search_model_versions(f"name='{model.name}'"):
                source_run = getattr(version, "run_id", None)
                if not source_run:
                    continue
                tags = _tags_or_none(client, source_run)
                if tags and tags.get("workbench.run_id") == run_id:
                    found.append(
                        {
                            "name": version.name,
                            "version": version.version,
                            "aliases": list(version.aliases or []),
                            "current_stage": version.current_stage,
                        }
                    )
        if not found:
            return {
                "run_id": run_id,
                "status": "not_registered",
                "message": "这次训练尚未注册到模型库。",
                "how_to_register": (
                    "先合并导出为独立模型,再注册(带血缘旗标,注册后可反查本轮训练与数据版本):\n"
                    "python scripts/merge_adapter.py --adapter-dir <本轮输出目录> "
                    "--output-dir outputs/merged/<run_id>\n"
                    "python scripts/registry_cli.py register --model-dir outputs/merged/<run_id> "
                    "--name <模型名> --source-run-id <run_id> "
                    "--dataset-version <数据版本> --config-digest <配置摘要>"
                ),
            }
        return {"run_id": run_id, "status": "registered", "versions": found}
    except Exception as exc:  # 查询失败如实呈现,不吞错
        return {
            "run_id": run_id,
            "status": "lookup_failed",
            "message": f"模型库查询失败:{exc}",
        }


def version_lineage(model_name: str, version: str | int, tracking_uri: str) -> dict[str, Any]:
    """Reverse edge: registry version → workbench run → dataset version (read-only)."""
    try:
        from mlflow.tracking import MlflowClient
    except ImportError:
        return {"model": f"{model_name} v{version}", "status": "mlflow_unavailable"}
    try:
        client = MlflowClient(tracking_uri=tracking_uri)
        model_version = client.get_model_version(name=model_name, version=str(version))
        mlflow_run_id = getattr(model_version, "run_id", None) or ""
        if not mlflow_run_id:
            return {
                "model": f"{model_name} v{version}",
                "status": "no_source_run",
                "message": "该版本没有关联的训练运行记录（可能是手工注册的目录）。",
            }
        run = client.get_run(mlflow_run_id)
        tags = run.data.tags
        params = run.data.params
        if "workbench.run_id" in tags:
            return {
                "model": f"{model_name} v{version}",
                "status": "workbench",
                "workbench_run_id": tags["workbench.run_id"],
                "mlflow_run_id": mlflow_run_id,
                "dataset_version": tags.get("workbench.dataset_version"),
                "config_digest": tags.get("workbench.config_digest"),
                "training_dataset": params.get("data.dataset_name"),
                "metrics": dict(run.data.metrics),
            }
        return {
            "model": f"{model_name} v{version}",
            "status": "external",
            "mlflow_run_id": mlflow_run_id,
            "base_model": params.get("model.name"),
            "message": "该版本来自旧体系或其他训练入口;血缘以 MLflow 参数为准。",
        }
    except Exception as exc:
        return {
            "model": f"{model_name} v{version}",
            "status": "lookup_failed",
            "message": f"血缘查询失败:{exc}",
        }
