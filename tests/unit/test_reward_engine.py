"""Unit tests for the reward engine."""

import pytest

from src.training.reward_engine import (
    accuracy_reward,
    combine_rewards,
    format_reward,
    get_reward_function,
    length_reward,
    list_reward_functions,
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
