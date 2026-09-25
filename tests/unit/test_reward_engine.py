"""Unit tests for the reward engine."""

from typing import Any
from unittest.mock import patch

import pytest

from src.training.reward_engine import (
    accuracy_reward,
    build_reward_functions,
    combine_rewards,
    cosine_reward,
    format_reward,
    get_reward_function,
    length_reward,
    list_reward_functions,
    llm_judge_reward,
)


class TestRewardFunctions:
    def test_format_reward_json(self) -> None:
        completions = ['{"answer": 42}', "not json"]
        scores = format_reward(["p"] * 2, completions, require_json=True)
        assert scores == [1.0, 0.0]

    def test_format_reward_tag(self) -> None:
        completions = ["<reasoning>ok</reasoning>", "ok"]
        scores = format_reward(["p"] * 2, completions, required_tag="<reasoning>")
        assert scores == [1.0, 0.0]

    def test_accuracy_reward(self) -> None:
        completions = ["1081", "1082"]
        scores = accuracy_reward(["p"] * 2, completions, answer="1081")
        assert scores == [1.0, 0.0]

    def test_accuracy_reward_list(self) -> None:
        completions = ["a", "b"]
        scores = accuracy_reward(["p"] * 2, completions, answer=["a", "b"])
        assert scores == [1.0, 1.0]

    def test_accuracy_reward_mismatched_length(self) -> None:
        with pytest.raises(ValueError, match="must match"):
            accuracy_reward(["p"] * 2, ["a", "b"], answer=["a"])

    def test_length_reward(self) -> None:
        completions = ["short", "a" * 2000]
        scores = length_reward(["p"] * 2, completions, min_length=5, max_length=1000)
        assert scores == [1.0, 0.0]

    def test_unknown_reward_function(self) -> None:
        with pytest.raises(KeyError, match="Unknown reward function"):
            get_reward_function("nonexistent")

    def test_list_reward_functions(self) -> None:
        names = list_reward_functions()
        assert "accuracy" in names
        assert "format" in names


class TestCombineRewards:
    def test_combine_equal_weights(self) -> None:
        rewards = [[1.0, 0.0], [0.0, 1.0]]
        combined = combine_rewards(rewards)
        assert combined == [1.0, 1.0]

    def test_combine_weighted(self) -> None:
        rewards = [[1.0, 0.0], [1.0, 1.0]]
        combined = combine_rewards(rewards, weights=[2.0, 1.0])
        assert combined == [3.0, 1.0]

    def test_combine_mismatched_length(self) -> None:
        with pytest.raises(ValueError, match="same length"):
            combine_rewards([[1.0], [1.0, 0.0]])


class _StubJudgeClient:
    """Deterministic judge for tests (avoids loading a real model)."""

    def judge(self, prompt: str, completion: str, **_: Any) -> float:
        return 1.0 if "good" in completion else 0.0


class TestBindKwargs:
    """Regression tests: bound kwargs must reach functions with **kwargs catch-alls.

    _bind_kwargs used to return such functions unbound, so configured kwargs
    (judge_client, answer_key, ...) were silently dropped.
    """

    def test_bound_kwargs_reach_catchall_function(self) -> None:
        funcs = build_reward_functions(
            ["llm_judge"],
            judge_client=_StubJudgeClient(),
        )
        scores = funcs[0][0](["p", "p"], ["good answer", "bad answer"])
        assert scores == [1.0, 0.0]

    def test_bound_answer_key_is_delivered(self) -> None:
        funcs = build_reward_functions(["accuracy"], answer_key="gold")
        # No runtime `answer` passed — bound answer_key makes accuracy_reward
        # look up "gold" instead and find nothing → all zeros, no crash.
        scores = funcs[0][0](["p"], ["anything"])
        assert scores == [0.0]

    def test_runtime_kwargs_override_bound(self) -> None:
        funcs = build_reward_functions(["accuracy"], answer_key="gold")
        # Runtime kwargs (TRL dataset columns) win over bound config.
        scores = funcs[0][0](["p"], ["42"], answer="42")
        assert scores == [1.0]


class TestLLMJudgeReward:
    def test_stub_client_scores(self) -> None:
        scores = llm_judge_reward(["p", "p"], ["good", "bad"], judge_client=_StubJudgeClient())
        assert scores == [1.0, 0.0]

    def test_missing_client_and_model_raises(self) -> None:
        with pytest.raises(ValueError, match="judge_client or judge_model"):
            llm_judge_reward(["p"], ["c"])

    def test_lazy_builds_local_judge_client_from_model_name(self) -> None:
        # judge_model-only path: LocalJudgeClient is constructed lazily.
        stub = _StubJudgeClient()
        with patch("src.training.reward_engine.LocalJudgeClient", return_value=stub) as mock_build:
            scores = llm_judge_reward(
                ["p"],
                ["good"],
                judge_model="Qwen/Qwen2.5-0.5B",
                judge_prompt_template="Rate: {completion}",
            )

        assert scores == [1.0]
        mock_build.assert_called_once_with(
            model_name="Qwen/Qwen2.5-0.5B", prompt_template="Rate: {completion}"
        )


class TestCosineReward:
    def test_string_reference_broadcast(self) -> None:
        scores = cosine_reward(["p", "p"], ["hello world", "zzzz"], reference="hello world")
        assert scores[0] == pytest.approx(1.0)
        assert scores[1] == pytest.approx(0.0)

    def test_reference_looked_up_from_kwargs(self) -> None:
        scores = cosine_reward(
            ["p"],
            ["the cat sat"],
            reference="the cat sat",  # via kwargs? no — explicit
        )
        assert scores == [pytest.approx(1.0)]
        # kwargs path: reference_key lookup when no explicit reference.
        scores2 = cosine_reward(["p"], ["the cat sat"], ref="the cat sat", reference_key="ref")
        assert scores2 == [pytest.approx(1.0)]

    def test_no_reference_scores_zero(self) -> None:
        assert cosine_reward(["p", "q"], ["a", "b"]) == [0.0, 0.0]

    def test_list_reference_mismatched_length(self) -> None:
        with pytest.raises(ValueError, match="must match"):
            cosine_reward(["p", "p"], ["a", "b"], reference=["only-one"])

    def test_partial_similarity_ratio(self) -> None:
        scores = cosine_reward(["p"], ["abcdef"], reference="abcdxyz")
        assert 0.0 < scores[0] < 1.0


class TestResidualBranches:
    """Branches the first wave missed."""

    def test_format_reward_plain_completion_scores_one(self) -> None:
        # Neither require_json nor required_tag → score stays 1.0.
        assert format_reward(["p"], ["anything at all"]) == [1.0]

    def test_format_reward_valid_json_still_one(self) -> None:
        # json.loads succeeds → no penalty.
        assert format_reward(["p"], ['{"k": 1}'], require_json=True) == [1.0]

    def test_accuracy_no_answer_scores_zero(self) -> None:
        # Neither explicit answer nor kwargs[answer_key] → zeros, no crash.
        assert accuracy_reward(["p", "q"], ["a", "b"]) == [0.0, 0.0]

    def test_accuracy_case_insensitive_and_whitespace(self) -> None:
        scores = accuracy_reward(["p"], ["  Hello   World  "], answer="hello world")
        assert scores == [1.0]
        scores = accuracy_reward(["p"], ["Hello World"], answer="hello world", case_sensitive=True)
        assert scores == [0.0]

    def test_combine_empty_rewards(self) -> None:
        assert combine_rewards([]) == []

    def test_combine_weights_length_mismatch(self) -> None:
        with pytest.raises(ValueError, match="weights must match"):
            combine_rewards([[1.0]], weights=[1.0, 2.0])


class TestCosineRewardReferenceList:
    def test_reference_list_matching_length_scores_each(self) -> None:
        scores = cosine_reward(["p1", "p2"], ["abc", "x"], reference=["ab", "xyz"])

        assert len(scores) == 2
        assert all(s is not None for s in scores)
        assert scores[0] > scores[1]  # "abc" vs "ab" scores higher than "x" vs "xyz"


class TestNormalizeTextWhitespace:
    def test_whitespace_preserved_when_normalization_off(self) -> None:
        from src.training.reward_engine import _normalize_text

        result = _normalize_text("  Foo   Bar  ", case_sensitive=False, normalize_whitespace=False)

        # Only strip + lower run; the inner triple space survives.
        assert result == "foo   bar"
