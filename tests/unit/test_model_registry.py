"""Unit tests for the MLflow model registry orchestration (src/tracking/registry.py)."""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

from config.base import LoggingConfig, ModelConfig
from src.tracking.registry import _resolve_merged_dir, register_trained_model

_UNSET: Any = object()  # sentinel so version_info=None is honored, not defaulted


class _FakeTracker:
    """Tracker double satisfying the surface register_trained_model uses."""

    def __init__(
        self,
        active: bool = True,
        model_uri: str | None = "runs:/abc/model",
        version_info: Any = _UNSET,
    ) -> None:
        self.active = active
        self._model_uri = model_uri
        self._version_info = (
            {"name": "Qwen2.5-1.5B-QLoRA", "version": 3, "current_stage": "None"}
            if version_info is _UNSET
            else version_info
        )
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def log_model(self, **kwargs: Any) -> str | None:
        self.calls.append(("log_model", kwargs))
        return self._model_uri

    def register_model(self, **kwargs: Any) -> dict[str, Any] | None:
        self.calls.append(("register_model", kwargs))
        return self._version_info

    def transition_model_stage(self, **kwargs: Any) -> None:
        self.calls.append(("transition", kwargs))


def _config(**overrides: Any) -> LoggingConfig:
    defaults: dict[str, Any] = {
        "register_model": True,
        "registry_model_name": "Qwen2.5-1.5B-QLoRA",
        "merge_before_register": False,  # skip merge in most tests
        "registry_stage": "Staging",
    }
    defaults.update(overrides)
    return LoggingConfig(**defaults)


class TestGuards:
    def test_disabled_by_config_is_none(self) -> None:
        tracker = _FakeTracker()
        result = register_trained_model("/tmp/adapter", tracker, _config(register_model=False))
        assert result is None
        assert tracker.calls == []  # no side effects

    def test_inactive_tracker_is_none(self) -> None:
        tracker = _FakeTracker(active=False)
        result = register_trained_model("/tmp/adapter", tracker, _config())
        assert result is None
        assert tracker.calls == []


class TestNameDerivation:
    @patch("src.tracking.registry.merge_adapter_to_dir")
    def test_explicit_name_wins(self, mock_merge: Any) -> None:
        mock_merge.side_effect = lambda **kw: kw["output_dir"]
        tracker = _FakeTracker()
        register_trained_model(
            "/tmp/adapter",
            tracker,
            _config(registry_model_name=None),  # fall through to model_config
            model_config=ModelConfig(name="Qwen/Qwen2.5-0.5B-Instruct"),
        )
        reg_call = next(c for c in tracker.calls if c[0] == "register_model")
        # model_config.name wins, with '/' sanitized to '-'
        assert reg_call[1]["name"] == "Qwen-Qwen2.5-0.5B-Instruct"

    @patch("src.tracking.registry.merge_adapter_to_dir")
    def test_fallback_name_when_nothing_set(self, mock_merge: Any) -> None:
        mock_merge.side_effect = lambda **kw: kw["output_dir"]
        tracker = _FakeTracker()
        register_trained_model(
            "/tmp/adapter",
            tracker,
            _config(registry_model_name=None, merge_before_register=True),
        )
        reg_call = next(c for c in tracker.calls if c[0] == "register_model")
        assert reg_call[1]["name"] == "qlora-finetuned-model"


class TestRegistrationFlow:
    @patch("src.tracking.registry.merge_adapter_to_dir")
    def test_full_flow_without_merge(self, mock_merge: Any) -> None:
        tracker = _FakeTracker()
        result = register_trained_model(
            "/tmp/adapter", tracker, _config(merge_before_register=False)
        )

        mock_merge.assert_not_called()
        assert result is not None
        assert result["name"] == "Qwen2.5-1.5B-QLoRA"
        assert result["version"] == 3
        assert result["current_stage"] == "Staging"
        assert result["model_dir"] == "/tmp/adapter"
        # Call order: log → register → transition
        assert [c[0] for c in tracker.calls] == ["log_model", "register_model", "transition"]
        transition = tracker.calls[-1][1]
        assert transition["version"] == "3" and transition["stage"] == "Staging"

    @patch("src.tracking.registry.merge_adapter_to_dir")
    def test_merge_path_uses_resolved_dir(self, mock_merge: Any, tmp_path: Any) -> None:
        mock_merge.side_effect = lambda **kw: kw["output_dir"]
        tracker = _FakeTracker()
        adapter_dir = str(tmp_path / "run-1" / "adapter")
        result = register_trained_model(adapter_dir, tracker, _config(merge_before_register=True))

        mock_merge.assert_called_once()
        merge_kwargs = mock_merge.call_args.kwargs
        expected = str(tmp_path / "run-1" / "merged_Qwen2.5-1.5B-QLoRA")
        assert merge_kwargs["adapter_dir"] == adapter_dir
        assert merge_kwargs["output_dir"] == expected
        assert result is not None
        assert result["model_dir"] == expected

    @patch("src.tracking.registry.merge_adapter_to_dir")
    def test_log_model_failure_aborts(self, mock_merge: Any) -> None:
        tracker = _FakeTracker(model_uri=None)
        result = register_trained_model("/tmp/adapter", tracker, _config())
        assert result is None
        assert [c[0] for c in tracker.calls] == ["log_model"]  # nothing after

    @patch("src.tracking.registry.merge_adapter_to_dir")
    def test_register_model_failure_aborts(self, mock_merge: Any) -> None:
        tracker = _FakeTracker(version_info=None)
        result = register_trained_model("/tmp/adapter", tracker, _config())
        assert result is None
        assert [c[0] for c in tracker.calls] == ["log_model", "register_model"]

    @patch("src.tracking.registry.merge_adapter_to_dir")
    def test_stage_none_skips_transition(self, mock_merge: Any) -> None:
        tracker = _FakeTracker()
        result = register_trained_model("/tmp/adapter", tracker, _config(registry_stage="None"))
        assert result is not None
        assert [c[0] for c in tracker.calls] == ["log_model", "register_model"]

    @patch("src.tracking.registry.merge_adapter_to_dir")
    def test_merge_exception_is_swallowed(self, mock_merge: Any) -> None:
        mock_merge.side_effect = RuntimeError("disk full")
        tracker = _FakeTracker()
        # Registration failure must NOT propagate — training result is safe on disk.
        result = register_trained_model(
            "/tmp/adapter", tracker, _config(merge_before_register=True)
        )
        assert result is None
        assert tracker.calls == []


class TestResolveMergedDir:
    def test_sibling_directory(self) -> None:
        assert _resolve_merged_dir("/a/b/adapter", "My-Model") == "/a/b/merged_My-Model"
