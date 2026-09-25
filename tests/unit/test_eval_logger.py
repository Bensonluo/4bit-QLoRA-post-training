"""Unit tests for the eval-to-MLflow bridge (src/tracking/eval_logger.py).

mlflow is injected as a fake module (the function imports it lazily), so no
MLflow server or backend is needed.
"""

from __future__ import annotations

import json
import sys
from types import ModuleType
from typing import Any
from unittest.mock import MagicMock, patch

from src.tracking.eval_logger import log_eval_to_mlflow


def _fake_mlflow() -> ModuleType:
    """Build a fake mlflow module recording every call."""
    mod = ModuleType("mlflow")
    mod.set_experiment = MagicMock()
    # MagicMock supports the context-manager protocol out of the box.
    mod.start_run = MagicMock(return_value=MagicMock())
    mod.set_tag = MagicMock()
    mod.log_metric = MagicMock()
    mod.log_param = MagicMock()
    mod.log_artifact = MagicMock()
    return mod


def _write_eval_json(tmp_path: Any) -> str:
    data = [
        {
            "model": "qwen-8b",
            "overall_accuracy": 0.62,
            "mrr": 0.74,
            "avg_confidence": "high",  # non-numeric → must be skipped
            "accuracy_by_difficulty": {"easy": 0.9, "hard": 0.4},
            "accuracy_by_type": {"drug": 0.8},
            "total": 100,
            "correct": 62,
        },
        {
            "model": "glm-5.1",
            "overall_accuracy": 0.88,
            "accuracy_by_difficulty": {"hard": 0.7},
            "total": 100,
            "correct": 88,
        },
    ]
    path = tmp_path / "eval_detail_test.json"
    path.write_text(json.dumps(data, ensure_ascii=False))
    return str(path)


class TestLogEvalToMlflow:
    def test_one_run_per_model_with_metrics(self, tmp_path: Any) -> None:
        fake = _fake_mlflow()
        path = _write_eval_json(tmp_path)
        with patch.dict(sys.modules, {"mlflow": fake}):
            log_eval_to_mlflow(path, experiment_name="domain-evaluation")

        fake.set_experiment.assert_called_once_with("domain-evaluation")
        assert fake.start_run.call_count == 2  # one per model

        # First run (qwen-8b): numeric core metrics logged, non-numeric skipped
        metric_names = [c.args[0] for c in fake.log_metric.call_args_list]
        assert "overall_accuracy" in metric_names
        assert "mrr" in metric_names
        assert "avg_confidence" not in metric_names
        # Per-difficulty / per-type flattening
        assert "accuracy_easy" in metric_names
        assert "accuracy_hard" in metric_names
        assert "accuracy_drug" in metric_names

        # Params
        param_names = [c.args[0] for c in fake.log_param.call_args_list]
        assert param_names.count("total_samples") == 2
        assert param_names.count("correct") == 2

        # Source JSON uploaded as artifact
        fake.log_artifact.assert_called_with(path)

    def test_tags_carry_model_and_source(self, tmp_path: Any) -> None:
        fake = _fake_mlflow()
        path = _write_eval_json(tmp_path)
        with patch.dict(sys.modules, {"mlflow": fake}):
            log_eval_to_mlflow(path)

        tag_pairs = {(c.args[0], c.args[1]) for c in fake.set_tag.call_args_list}
        assert ("model_name", "qwen-8b") in tag_pairs
        assert ("eval_source", "eval_detail_test.json") in tag_pairs

    def test_missing_file_is_silent_noop(self, tmp_path: Any) -> None:
        fake = _fake_mlflow()
        missing = str(tmp_path / "nope.json")
        with patch.dict(sys.modules, {"mlflow": fake}):
            log_eval_to_mlflow(missing)
        fake.set_experiment.assert_not_called()

    def test_mlflow_not_installed_is_silent_noop(self, tmp_path: Any) -> None:
        path = _write_eval_json(tmp_path)
        # A None entry in sys.modules makes `import mlflow` raise ImportError.
        with patch.dict(sys.modules, {"mlflow": None}):
            log_eval_to_mlflow(path)  # must not raise

    def test_empty_model_list_still_sets_experiment(self, tmp_path: Any) -> None:
        path = tmp_path / "eval_detail_empty.json"
        path.write_text("[]")
        fake = _fake_mlflow()
        with patch.dict(sys.modules, {"mlflow": fake}):
            log_eval_to_mlflow(str(path))
        fake.set_experiment.assert_called_once()
        fake.start_run.assert_not_called()


class TestNonNumericBreakdownValues:
    def test_non_numeric_difficulty_and_type_values_skipped(self, tmp_path: Any) -> None:
        data = [
            {
                "model": "qwen-8b",
                "overall_accuracy": 0.5,
                "accuracy_by_difficulty": {"easy": 0.9, "hard": "n/a"},
                "accuracy_by_type": {"drug": 0.8, "hospital": None},
            }
        ]
        path = tmp_path / "eval_detail_nonnumeric.json"
        path.write_text(json.dumps(data))

        fake = _fake_mlflow()
        with patch.dict(sys.modules, {"mlflow": fake}):
            log_eval_to_mlflow(str(path))

        logged = {c.args[0] for c in fake.log_metric.call_args_list}
        assert "accuracy_easy" in logged
        assert "accuracy_drug" in logged
        assert "accuracy_hard" not in logged  # "n/a" is not int/float
        assert "accuracy_hospital" not in logged  # None is not int/float
