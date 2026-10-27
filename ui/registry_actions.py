"""Apply registry edits against fresh state and report the observed result."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class RegistryNotice:
    level: Literal["success", "warning", "error"]
    message: str


def _binding(version: str | None) -> str:
    return f"v{version}" if version is not None else "未绑定"


def change_alias(
    tracking_uri: str,
    model_name: str,
    alias: str,
    expected_version: str | None,
    target_version: str | None,
) -> RegistryNotice:
    """Check the displayed binding before writing, then read back the result.

    MLflow does not provide compare-and-set for aliases. This detects changes
    already visible before submission; it is not an atomic concurrency lock.
    """
    from mlflow.tracking import MlflowClient

    try:
        client = MlflowClient(tracking_uri=tracking_uri)
        current = (client.get_registered_model(model_name).aliases or {}).get(alias)
    except Exception as exc:
        return RegistryNotice("error", f"无法读取当前别名，未提交变更：{exc}")
    current = str(current) if current is not None else None
    if current != expected_version:
        return RegistryNotice(
            "warning",
            f"{model_name} 的 {alias} 已从 {_binding(expected_version)} 变为 "
            f"{_binding(current)}。本次未提交，请核对刷新后的绑定再操作。",
        )
    if current == target_version:
        return RegistryNotice("warning", "绑定没有变化，无需提交。")
    if target_version is not None:
        try:
            client.get_model_version(model_name, target_version)
        except Exception as exc:
            return RegistryNotice("error", f"无法读取目标版本 v{target_version}，未提交变更：{exc}")
    try:
        if target_version is None:
            client.delete_registered_model_alias(name=model_name, alias=alias)
        else:
            client.set_registered_model_alias(name=model_name, alias=alias, version=target_version)
    except Exception as exc:
        return RegistryNotice("error", f"别名变更未确认：{exc}。请刷新并核对当前绑定后再重试。")
    try:
        observed = (client.get_registered_model(model_name).aliases or {}).get(alias)
    except Exception as exc:
        return RegistryNotice("warning", f"变更请求已提交，但暂时无法读取结果：{exc}。请刷新核对。")
    observed = str(observed) if observed is not None else None
    if observed != target_version:
        return RegistryNotice(
            "warning",
            f"变更请求已提交，但当前 {alias} 为 {_binding(observed)}，"
            f"与目标 {_binding(target_version)} 不同。请核对其他操作是否修改了绑定。",
        )
    return RegistryNotice(
        "success",
        f"已核对：{model_name} 的 {alias}：{_binding(current)} → {_binding(observed)}。",
    )
