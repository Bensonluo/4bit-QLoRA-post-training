"""Unit tests for MLflowTracker (src/tracking/mlflow_tracker.py).

The mlflow API surface is faked via sys.modules patching so the suite runs
whether or not the real mlflow package is installed. All tracker methods
have an `if not self._active` early-return guard — each is exercised both
on the active and the inactive (ImportError) path.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

import src.tracking.mlflow_tracker as tracker_mod
from config.base import LoggingConfig
from src.tracking.mlflow_tracker import (
    MLflowTracker,
    _flatten_dict,
    _NoOpTracker,
    get_tracker,
)


def _make_active_tracker() -> MLflowTracker:
    """Build an active tracker with a fresh mock mlflow module."""
    fake_mlflow = MagicMock()
    with patch.dict(sys.modules, {"mlflow": fake_mlflow, "mlflow.entities": MagicMock()}):
        tracker = MLflowTracker(tracking_uri="sqlite:///tmp.db", experiment_name="exp")
    # Give each test a pristine API surface (set_tracking_uri assertions
    # happened against the fake above).
    tracker._mlflow = MagicMock()
    return tracker


def _inactive_tracker() -> MLflowTracker:
    """Build a tracker whose __init__ hit the ImportError branch."""
    with patch.dict(sys.modules, {"mlflow": None}):
        return MLflowTracker(tracking_uri="u", experiment_name="e")


@pytest.fixture(autouse=True)
def _reset_singleton() -> Any:
    """Isolate the get_tracker module-level singleton between tests."""
    tracker_mod._tracker_instance = None
    yield
    tracker_mod._tracker_instance = None


class TestFlattenDict:
    def test_nested_dicts_become_dot_notation(self) -> None:
        assert _flatten_dict({"model": {"name": "Qwen", "size": 1}}) == {
            "model.name": "Qwen",
            "model.size": 1,
        }

    def test_deeply_nested(self) -> None:
        assert _flatten_dict({"a": {"b": {"c": 1}}}) == {"a.b.c": 1}

    def test_lists_and_tuples_are_joined(self) -> None:
        out = _flatten_dict({"mods": ["q_proj", "v_proj"], "t": (1, 2)})
        assert out == {"mods": "q_proj, v_proj", "t": "1, 2"}

    def test_scalars_pass_through(self) -> None:
        assert _flatten_dict({"x": 1, "y": "s", "z": None}) == {"x": 1, "y": "s", "z": None}


class TestNoOpTracker:
    def test_all_methods_are_silent_no_ops(self) -> None:
        t = _NoOpTracker()
        assert t.active is False
        assert t.log_model("dir") is None
        assert t.register_model("uri", "name") is None
        assert t.search_model_versions() == []
        assert t.search_runs() == []
        for call in (
            lambda: t.start_run(run_name="r"),
            lambda: t.log_metrics({"m": 1.0}),
            lambda: t.log_params({"p": 1}),
            lambda: t.log_artifact("f.txt"),
            lambda: t.log_artifacts("dir"),
            lambda: t.transition_model_stage("n", "1", "Staging"),
            lambda: t.set_model_alias("n", "1", "champion"),
            lambda: t.delete_model_alias("n", "champion"),
            lambda: t.end_run(),
            lambda: t.log_dataset("p", "n", "v"),
        ):
            call()  # must not raise


class TestInit:
    def test_success_sets_active_and_configures_mlflow(self) -> None:
        fake = MagicMock()
        with patch.dict(sys.modules, {"mlflow": fake}):
            tracker = MLflowTracker(tracking_uri="sqlite:///x.db", experiment_name="my-exp")
        assert tracker.active is True
        fake.set_tracking_uri.assert_called_once_with("sqlite:///x.db")
        fake.set_experiment.assert_called_once_with("my-exp")

    def test_import_error_sets_inactive(self) -> None:
        tracker = _inactive_tracker()
        assert tracker.active is False
        assert tracker._mlflow is None


class TestStartRun:
    def test_returns_run_id_and_logs_flattened_config_and_tags(self) -> None:
        tracker = _make_active_tracker()
        tracker._mlflow.start_run.return_value = SimpleNamespace(info=SimpleNamespace(run_id="r1"))

        run_id = tracker.start_run(run_name="run", config={"model": {"name": "Q"}}, tags={"a": "b"})

        assert run_id == "r1"
        tracker._mlflow.start_run.assert_called_once_with(run_name="run")
        tracker._mlflow.log_params.assert_called_once_with({"model.name": "Q"})
        tracker._mlflow.set_tags.assert_called_once_with({"a": "b"})

    def test_no_config_or_tags_skips_logging(self) -> None:
        tracker = _make_active_tracker()
        tracker._mlflow.start_run.return_value = SimpleNamespace(info=SimpleNamespace(run_id="r2"))

        assert tracker.start_run() == "r2"
        tracker._mlflow.log_params.assert_not_called()
        tracker._mlflow.set_tags.assert_not_called()

    def test_inactive_returns_none(self) -> None:
        assert _inactive_tracker().start_run(run_name="x") is None


class TestLogMetricsParamsArtifacts:
    def test_log_metrics_forwards_step(self) -> None:
        tracker = _make_active_tracker()
        tracker.log_metrics({"loss": 0.5}, step=10)
        tracker._mlflow.log_metrics.assert_called_once_with({"loss": 0.5}, step=10)

    def test_log_params_flattens_only_when_nested(self) -> None:
        tracker = _make_active_tracker()
        tracker.log_params({"model": {"name": "Q"}})
        tracker._mlflow.log_params.assert_called_once_with({"model.name": "Q"})
        tracker.log_params({"lr": 1e-4})
        tracker._mlflow.log_params.assert_called_with({"lr": 1e-4})

    def test_log_artifact_forwards_path(self) -> None:
        tracker = _make_active_tracker()
        tracker.log_artifact("confusion_matrix.png")
        tracker._mlflow.log_artifact.assert_called_once_with("confusion_matrix.png")

    def test_log_artifacts_forwards_dir_and_path(self) -> None:
        tracker = _make_active_tracker()
        tracker.log_artifacts("outputs/merged", artifact_path="model")
        tracker._mlflow.log_artifacts.assert_called_once_with(
            "outputs/merged", artifact_path="model"
        )

    def test_inactive_paths_do_not_raise(self) -> None:
        t = _inactive_tracker()
        t.log_metrics({"m": 1})
        t.log_params({"p": 1})
        t.log_artifact("f")
        t.log_artifacts("d")


class TestLogModel:
    @patch("transformers.AutoTokenizer")
    @patch("transformers.AutoModelForCausalLM")
    def test_logs_transformers_components_and_returns_uri(
        self, mock_model_cls: MagicMock, mock_tok_cls: MagicMock
    ) -> None:
        tracker = _make_active_tracker()
        tracker._mlflow.transformers.log_model.return_value = SimpleNamespace(
            model_uri="runs:/r1/model"
        )

        uri = tracker.log_model("outputs/merged/x")

        assert uri == "runs:/r1/model"
        mock_model_cls.from_pretrained.assert_called_once_with("outputs/merged/x", device_map=None)
        mock_tok_cls.from_pretrained.assert_called_once_with("outputs/merged/x")
        kwargs = tracker._mlflow.transformers.log_model.call_args.kwargs
        assert kwargs["transformers_model"] == {
            "model": mock_model_cls.from_pretrained.return_value,
            "tokenizer": mock_tok_cls.from_pretrained.return_value,
        }
        assert kwargs["artifact_path"] == "model"
        assert "registered_model_name" not in kwargs

    @patch("transformers.AutoTokenizer")
    @patch("transformers.AutoModelForCausalLM")
    def test_implicit_registration_when_name_given(
        self, mock_model_cls: MagicMock, mock_tok_cls: MagicMock
    ) -> None:
        tracker = _make_active_tracker()
        tracker._mlflow.transformers.log_model.return_value = SimpleNamespace(model_uri="u")

        tracker.log_model("dir", artifact_path="m", registered_model_name="Qwen-QLoRA")

        kwargs = tracker._mlflow.transformers.log_model.call_args.kwargs
        assert kwargs["registered_model_name"] == "Qwen-QLoRA"
        assert kwargs["artifact_path"] == "m"

    @patch("transformers.AutoTokenizer")
    @patch("transformers.AutoModelForCausalLM")
    def test_uri_falls_back_to_active_run(
        self, mock_model_cls: MagicMock, mock_tok_cls: MagicMock
    ) -> None:
        tracker = _make_active_tracker()
        tracker._mlflow.transformers.log_model.return_value = SimpleNamespace()  # no model_uri
        tracker._mlflow.active_run.return_value = SimpleNamespace(
            info=SimpleNamespace(run_id="live-run")
        )

        assert tracker.log_model("dir", artifact_path="model") == "runs:/live-run/model"

    def test_inactive_returns_none(self) -> None:
        assert _inactive_tracker().log_model("dir") is None


class TestRegisterModel:
    def test_returns_version_dict(self) -> None:
        tracker = _make_active_tracker()
        tracker._mlflow.register_model.return_value = SimpleNamespace(
            name="Qwen-QLoRA", version="3", current_status="READY", run_id="r1", source="runs:/s"
        )

        out = tracker.register_model("runs:/r1/model", "Qwen-QLoRA")

        assert out == {
            "name": "Qwen-QLoRA",
            "version": "3",
            "current_stage": "READY",
            "run_id": "r1",
            "source": "runs:/s",
        }
        tracker._mlflow.register_model.assert_called_once_with(
            model_uri="runs:/r1/model", name="Qwen-QLoRA"
        )

    def test_stage_falls_back_when_no_status_attribute(self) -> None:
        tracker = _make_active_tracker()
        tracker._mlflow.register_model.return_value = SimpleNamespace(
            name="m", version="1", run_id=None, source="s"
        )

        out = tracker.register_model("uri", "m")
        assert out is not None
        assert out["current_stage"] == "None"

    def test_inactive_returns_none(self) -> None:
        assert _inactive_tracker().register_model("uri", "n") is None


class TestTransitionModelStage:
    def test_uses_client_without_archiving(self) -> None:
        tracker = _make_active_tracker()
        client = tracker._mlflow.tracking.MlflowClient.return_value

        tracker.transition_model_stage("Qwen-QLoRA", "3", "Production")

        client.transition_model_version_stage.assert_called_once_with(
            name="Qwen-QLoRA", version="3", stage="Production", archive_existing_versions=False
        )

    def test_inactive_does_not_raise(self) -> None:
        _inactive_tracker().transition_model_stage("n", "1", "Staging")


class TestModelAliases:
    def test_set_alias_forwards_to_client(self) -> None:
        tracker = _make_active_tracker()
        client = tracker._mlflow.tracking.MlflowClient.return_value

        tracker.set_model_alias("Qwen-QLoRA", "3", "champion")

        client.set_registered_model_alias.assert_called_once_with(
            name="Qwen-QLoRA", alias="champion", version="3"
        )

    def test_set_alias_inactive_does_not_raise(self) -> None:
        _inactive_tracker().set_model_alias("n", "1", "champion")

    def test_delete_alias_forwards_to_client(self) -> None:
        tracker = _make_active_tracker()
        client = tracker._mlflow.tracking.MlflowClient.return_value

        tracker.delete_model_alias("Qwen-QLoRA", "champion")

        client.delete_registered_model_alias.assert_called_once_with(
            name="Qwen-QLoRA", alias="champion"
        )

    def test_delete_alias_inactive_does_not_raise(self) -> None:
        _inactive_tracker().delete_model_alias("n", "champion")


class TestSearchModelVersions:
    def test_name_filter_with_quote_escaping(self) -> None:
        tracker = _make_active_tracker()
        client = tracker._mlflow.tracking.MlflowClient.return_value
        client.search_model_versions.return_value = [
            SimpleNamespace(
                name="m",
                version="1",
                current_stage="Staging",
                aliases=["champion"],
                run_id="r1",
                creation_timestamp=123,
                status="READY",
            )
        ]

        out = tracker.search_model_versions("acme's model")

        client.search_model_versions.assert_called_once_with("name='acme''s model'")
        assert out == [
            {
                "name": "m",
                "version": "1",
                "current_stage": "Staging",
                "aliases": ["champion"],
                "run_id": "r1",
                "creation_timestamp": 123,
                "status": "READY",
            }
        ]

    def test_missing_aliases_attribute_maps_to_empty_list(self) -> None:
        tracker = _make_active_tracker()
        client = tracker._mlflow.tracking.MlflowClient.return_value
        client.search_model_versions.return_value = [
            SimpleNamespace(
                name="old",
                version="7",
                current_stage="Production",
                run_id="r7",
                creation_timestamp=456,
                status="READY",
            )  # no `aliases` attribute — older MLflow ModelVersion
        ]

        out = tracker.search_model_versions("old")

        assert out[0]["aliases"] == []

    def test_no_name_calls_without_filter(self) -> None:
        tracker = _make_active_tracker()
        client = tracker._mlflow.tracking.MlflowClient.return_value
        client.search_model_versions.return_value = []

        assert tracker.search_model_versions() == []
        client.search_model_versions.assert_called_once_with()

    def test_inactive_returns_empty(self) -> None:
        assert _inactive_tracker().search_model_versions("m") == []


class TestCurrentRunIdAndEndRun:
    def test_current_run_id_from_active_run(self) -> None:
        tracker = _make_active_tracker()
        tracker._mlflow.active_run.return_value = SimpleNamespace(
            info=SimpleNamespace(run_id="rid")
        )
        assert tracker._current_run_id() == "rid"

    def test_current_run_id_none_when_no_active_run(self) -> None:
        tracker = _make_active_tracker()
        tracker._mlflow.active_run.return_value = None
        assert tracker._current_run_id() is None

    def test_end_run_swallows_exceptions(self) -> None:
        tracker = _make_active_tracker()
        tracker._mlflow.end_run.side_effect = RuntimeError("server down")
        tracker.end_run()  # must not raise
        tracker._mlflow.end_run.assert_called_once()

    def test_end_run_inactive_does_not_raise(self) -> None:
        _inactive_tracker().end_run()


class TestSearchRuns:
    def _exp(self, experiment_id: str) -> SimpleNamespace:
        return SimpleNamespace(experiment_id=experiment_id)

    def test_by_experiment_name(self) -> None:
        tracker = _make_active_tracker()
        tracker._mlflow.get_experiment_by_name.return_value = self._exp("e1")
        records = [{"run_id": "r1"}]
        tracker._mlflow.search_runs.return_value.to_dict.return_value = records

        out = tracker.search_runs("my-exp")

        tracker._mlflow.get_experiment_by_name.assert_called_once_with("my-exp")
        kwargs = tracker._mlflow.search_runs.call_args.kwargs
        assert kwargs["experiment_ids"] == ["e1"]
        assert kwargs["run_view_type"] is not None  # ViewType.ALL from mlflow.entities
        assert out == records

    def test_unknown_experiment_name_returns_empty(self) -> None:
        tracker = _make_active_tracker()
        tracker._mlflow.get_experiment_by_name.return_value = None
        assert tracker.search_runs("nope") == []
        tracker._mlflow.search_runs.assert_not_called()

    def test_defaults_to_first_experiment(self) -> None:
        tracker = _make_active_tracker()
        tracker._mlflow.search_experiments.return_value = [self._exp("a"), self._exp("b")]
        tracker._mlflow.search_runs.return_value.to_dict.return_value = []

        tracker.search_runs()

        assert tracker._mlflow.search_runs.call_args.kwargs["experiment_ids"] == ["a"]

    def test_no_experiments_returns_empty(self) -> None:
        tracker = _make_active_tracker()
        tracker._mlflow.search_experiments.return_value = []
        assert tracker.search_runs() == []
        tracker._mlflow.search_runs.assert_not_called()

    def test_runs_without_to_dict_returned_as_empty(self) -> None:
        tracker = _make_active_tracker()
        tracker._mlflow.get_experiment_by_name.return_value = self._exp("e1")
        tracker._mlflow.search_runs.return_value = SimpleNamespace()  # no to_dict

        assert tracker.search_runs("my-exp") == []

    def test_inactive_returns_empty(self) -> None:
        assert _inactive_tracker().search_runs("x") == []


class TestLogDataset:
    def test_logs_lineage_params_under_context(self) -> None:
        tracker = _make_active_tracker()

        tracker.log_dataset(
            dataset_path="/data/train.jsonl", dataset_name="medical-entity", version="v2"
        )

        tracker._mlflow.log_params.assert_called_once_with(
            {
                "dataset.training.name": "medical-entity",
                "dataset.training.version": "v2",
                "dataset.training.path": "/data/train.jsonl",
            }
        )

    def test_custom_context_key(self) -> None:
        tracker = _make_active_tracker()
        tracker.log_dataset("p", "n", "v9", context="validation")
        call = tracker._mlflow.log_params.call_args.args[0]
        assert "dataset.validation.name" in call

    def test_inactive_does_not_raise(self) -> None:
        _inactive_tracker().log_dataset("p", "n", "v")


class TestGetTrackerFactory:
    def test_none_config_returns_shared_no_op(self) -> None:
        assert get_tracker(None) is tracker_mod._NO_OP

    def test_mlflow_disabled_returns_shared_no_op(self) -> None:
        assert get_tracker(LoggingConfig(use_mlflow=False)) is tracker_mod._NO_OP

    @patch.object(tracker_mod, "MLflowTracker")
    def test_enabled_constructs_with_configured_uri_and_experiment(
        self, mock_cls: MagicMock
    ) -> None:
        mock_cls.return_value.active = True
        cfg = LoggingConfig(
            use_mlflow=True,
            mlflow_tracking_uri="sqlite:///prod.db",
            mlflow_experiment_name="finance-sft",
        )

        tracker = get_tracker(cfg)

        assert tracker is mock_cls.return_value
        mock_cls.assert_called_once_with(
            tracking_uri="sqlite:///prod.db", experiment_name="finance-sft"
        )

    @patch.object(tracker_mod, "MLflowTracker")
    def test_enabled_uses_default_uri_and_experiment(self, mock_cls: MagicMock) -> None:
        mock_cls.return_value.active = True
        get_tracker(LoggingConfig(use_mlflow=True))
        mock_cls.assert_called_once_with(
            tracking_uri="./outputs/mlruns", experiment_name="qlora-post-training"
        )

    @patch.object(tracker_mod, "MLflowTracker")
    def test_singleton_reused_while_active(self, mock_cls: MagicMock) -> None:
        mock_cls.return_value.active = True

        first = get_tracker(LoggingConfig(use_mlflow=True))
        second = get_tracker(LoggingConfig(use_mlflow=True))

        assert first is second
        mock_cls.assert_called_once()

    @patch.object(tracker_mod, "MLflowTracker")
    def test_inactive_singleton_is_rebuilt(self, mock_cls: MagicMock) -> None:
        mock_cls.return_value.active = False

        get_tracker(LoggingConfig(use_mlflow=True))
        get_tracker(LoggingConfig(use_mlflow=True))

        assert mock_cls.call_count == 2


class TestCurrentRunId:
    def test_inactive_tracker_returns_none(self) -> None:
        # Inactive guard: no MLflow API call is attempted at all.
        assert _inactive_tracker()._current_run_id() is None

    def test_active_without_run_returns_none(self) -> None:
        tracker = _make_active_tracker()
        tracker._mlflow.active_run.return_value = None
        assert tracker._current_run_id() is None
