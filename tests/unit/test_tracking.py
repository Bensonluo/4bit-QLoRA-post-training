"""Unit tests for MLflow tracker interface (dataset lineage + factory)."""

from src.tracking.mlflow_tracker import MLflowTracker, _NoOpTracker, get_tracker


class TestLogDataset:
    """Regression tests: log_dataset used to be dead code after a return."""

    def test_log_dataset_exists_on_mlflow_tracker(self) -> None:
        assert hasattr(MLflowTracker, "log_dataset")

    def test_log_dataset_noop_on_noop_tracker(self) -> None:
        tracker = _NoOpTracker()
        assert hasattr(tracker, "log_dataset")
        # Must not raise.
        tracker.log_dataset(dataset_path="/data/x.jsonl", dataset_name="x", version="v1")

    def test_factory_noop_has_log_dataset(self) -> None:
        tracker = get_tracker(None)
        assert hasattr(tracker, "log_dataset")
