"""三角血缘真集成:训练 → 合并 → 带血缘注册 → 双向反查闭环。

用真实 MLflow 文件库与真实合并产物;证明「模型↔实验↔数据版本」三角在
实际操作后仍可互相反查,血缘标签不丢失。
"""

import time

import pytest

pytest.importorskip("datasets")
pytest.importorskip("peft")
pytest.importorskip("mlflow")

from src.models.merger import merge_adapter_to_dir
from src.tracking.registry import register_merged_model
from src.workbench.registry_link import run_registration_status, version_lineage
from tests.unit.test_workbench_training_runs import (  # noqa: F401
    _prepare,
    _verify_labels,
    _wait,
    environment,
)


def test_lineage_triangle_survives_real_merge_and_register(environment, tmp_path):  # noqa: F811
    intake, session, training, model = environment
    record = _prepare(environment)
    training.start(record["run_id"], session)
    result = _wait(training, record["run_id"])
    assert result["status"] == "succeeded", training.read_logs(record["run_id"], tail=50)

    # 合并导出为独立模型
    merged_dir = tmp_path / "merged" / record["run_id"]
    merge_adapter_to_dir(result["output_dir"], merged_dir, base_model_name=str(model))

    # 带血缘注册(真实 MLflow 文件库)
    tracking_uri = f"sqlite:///{training.root / 'mlflow.db'}"
    info = register_merged_model(
        model_dir=str(merged_dir),
        name="工单分类-集成测试",
        stage="None",
        tracking_uri=tracking_uri,
        registered_via="integration-test",
        lineage_tags={
            "workbench.run_id": record["run_id"],
            "workbench.dataset_version": session.dataset.version,
            "workbench.config_digest": record["config_digest"],
        },
    )
    assert info["version"]

    # 正向:训练运行 → 注册状态(找到刚注册的版本)
    deadline = time.monotonic() + 30
    status = None
    while time.monotonic() < deadline:
        status = run_registration_status(record["run_id"], tracking_uri)
        if status["status"] != "tracking_not_found":
            break
        time.sleep(0.5)
    assert status["status"] == "registered", status
    assert status["versions"][0]["name"] == "工单分类-集成测试"

    # 反向:模型版本 → workbench run → 数据版本(三角闭合)
    lineage = version_lineage("工单分类-集成测试", info["version"], tracking_uri)
    assert lineage["status"] == "workbench", lineage
    assert lineage["workbench_run_id"] == record["run_id"]
    assert lineage["dataset_version"] == session.dataset.version
    assert lineage["config_digest"] == record["config_digest"]
