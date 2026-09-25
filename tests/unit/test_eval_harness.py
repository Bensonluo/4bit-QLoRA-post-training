"""Tests for the guarded lm-evaluation-harness integration.

The base install deliberately does NOT include lm_eval (opt-in ``[eval]``
extra), so the unavailable path is exercised against the real environment —
no mocks. The available path injects a fake module via ``sys.modules``
(``import`` consults sys.modules before any finder, matching the DCP test
pattern). A MetaPathFinder blocker makes the unavailable path deterministic
even if the extra is installed in the test venv.
"""

from __future__ import annotations

import json
import sys
import types
from collections.abc import Iterator
from typing import Any

import pytest

from src.evaluation.harness import harness_available, run_harness_eval, summarize_results


class _BlockLMEval:
    """MetaPathFinder that raises ImportError for lm_eval — simulates the
    package being absent even when the [eval] extra is installed."""

    def find_spec(self, fullname: str, path: Any = None, target: Any = None) -> None:
        if fullname == "lm_eval" or fullname.startswith("lm_eval."):
            raise ImportError(f"blocked: {fullname}")
        return None


@pytest.fixture()
def blocked_lm_eval(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    blocker = _BlockLMEval()
    sys.meta_path.insert(0, blocker)
    # `import` consults sys.modules before any finder — evict a cached
    # lm_eval so the blocker is actually consulted (DCP-test pattern).
    monkeypatch.delitem(sys.modules, "lm_eval", raising=False)
    yield
    sys.meta_path.remove(blocker)


@pytest.fixture()
def fake_lm_eval(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    calls: dict[str, Any] = {}

    def _simple_evaluate(**kwargs: Any) -> dict[str, Any]:
        calls.update(kwargs)
        return {
            "results": {"hellaswag": {"acc": 0.5, "acc,stderr": 0.01, "alias": "HellaSwag"}},
            "versions": {"hellaswag": 1.0},
        }

    fake = types.ModuleType("lm_eval")
    fake.simple_evaluate = _simple_evaluate
    fake.calls = calls
    monkeypatch.setitem(sys.modules, "lm_eval", fake)
    return fake


class TestGuard:
    def test_harness_available_false_when_blocked(self, blocked_lm_eval: None) -> None:
        assert harness_available() is False

    def test_harness_available_true_when_importable(self, fake_lm_eval: types.ModuleType) -> None:
        assert harness_available() is True

    def test_run_raises_install_hint_when_blocked(self, blocked_lm_eval: None) -> None:
        with pytest.raises(RuntimeError, match=r"\.\[eval\]"):
            run_harness_eval("some/model", ["hellaswag"])


class TestRunHarnessEval:
    def test_forwards_kwargs_to_simple_evaluate(self, fake_lm_eval: types.ModuleType) -> None:
        run_harness_eval(
            "models/x",
            ["hellaswag", "arc_easy"],
            num_fewshot=5,
            limit=100,
            dtype="bfloat16",
            device="mps",
        )
        calls = fake_lm_eval.calls
        assert calls["model"] == "hf"
        assert calls["model_args"] == "pretrained=models/x,dtype=bfloat16"
        assert calls["tasks"] == ["hellaswag", "arc_easy"]
        assert calls["num_fewshot"] == 5
        assert calls["limit"] == 100
        assert calls["device"] == "mps"

    def test_defaults_match_documented_api(self, fake_lm_eval: types.ModuleType) -> None:
        run_harness_eval("m", ["t"])
        calls = fake_lm_eval.calls
        assert calls["model_args"] == "pretrained=m,dtype=auto"
        assert calls["num_fewshot"] is None
        assert calls["limit"] is None
        assert calls["batch_size"] == "auto"
        assert calls["device"] is None
        assert calls["apply_chat_template"] is False

    def test_returns_full_result_dict(self, fake_lm_eval: types.ModuleType) -> None:
        results = run_harness_eval("m", ["hellaswag"])
        assert results["results"]["hellaswag"]["acc"] == 0.5
        assert "versions" in results

    def test_output_json_written(self, fake_lm_eval: types.ModuleType, tmp_path: Any) -> None:
        out = tmp_path / "sub" / "results.json"
        results = run_harness_eval("m", ["t"], output_path=out)
        assert out.exists()
        assert json.loads(out.read_text()) == results

    def test_no_output_file_without_output_path(
        self, fake_lm_eval: types.ModuleType, tmp_path: Any
    ) -> None:
        run_harness_eval("m", ["t"])
        assert list(tmp_path.iterdir()) == []


class TestSummarizeResults:
    def test_stderr_becomes_sibling_metric(self) -> None:
        raw = {"results": {"hellaswag": {"acc": 0.5, "acc,stderr": 0.01, "alias": "HellaSwag"}}}
        assert summarize_results(raw) == {"hellaswag": {"acc": 0.5, "acc_stderr": 0.01}}

    def test_missing_stderr_stays_none_not_zero(self) -> None:
        raw = {"results": {"arc_easy": {"acc": 1, "acc_norm": 0.75}}}
        summary = summarize_results(raw)["arc_easy"]
        assert summary["acc"] == 1.0
        assert summary["acc_stderr"] is None
        assert summary["acc_norm"] == 0.75
        assert summary["acc_norm_stderr"] is None

    def test_bool_values_are_skipped(self) -> None:
        # bool is an int subclass — the numeric filter excludes it on purpose.
        raw = {"results": {"t": {"acc": True, "acc,stderr": 0.1}}}
        assert summarize_results(raw) == {"t": {}}

    def test_empty_or_missing_results(self) -> None:
        assert summarize_results({}) == {}
        assert summarize_results({"results": {}}) == {}
