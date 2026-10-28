"""A cache entry must identify the model registry as well as tracking storage."""

import pytest

pytest.importorskip("mlflow")
from mlflow.tracking import MlflowClient


def test_registry_cache_separates_real_registries_with_same_tracking_uri(tmp_path):
    import mlflow

    from ui.queries import fetch_model_versions

    previous_uri = mlflow.get_tracking_uri()
    tracking_uri = (tmp_path / "tracking").as_uri()
    first_uri = (tmp_path / "registry-one").as_uri()
    second_uri = (tmp_path / "registry-two").as_uri()
    first = MlflowClient(tracking_uri=tracking_uri, registry_uri=first_uri)
    second = MlflowClient(tracking_uri=tracking_uri, registry_uri=second_uri)
    first.create_registered_model("first-model")
    first.create_model_version("first-model", source="file:///first")
    second.create_registered_model("second-model")
    second.create_model_version("second-model", source="file:///second")
    fetch_model_versions.clear()
    try:
        one = fetch_model_versions(tracking_uri, registry_uri=first_uri)
        two = fetch_model_versions(tracking_uri, registry_uri=second_uri)
        assert one["name"].tolist() == ["first-model"]
        assert two["name"].tolist() == ["second-model"]
    finally:
        fetch_model_versions.clear()
        mlflow.set_tracking_uri(previous_uri)
