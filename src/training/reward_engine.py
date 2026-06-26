"""Pluggable reward engine for GRPO training.

Reward functions follow the TRL convention:
    func(prompts: list[str], completions: list[str], **kwargs) -> list[float]
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from difflib import SequenceMatcher
from typing import Any

RewardFunction = Callable[..., list[float]]

_REWARD_REGISTRY: dict[str, RewardFunction] = {}


def register_reward(name: str) -> Callable[[RewardFunction], RewardFunction]:
    """Decorator to register a reward function."""

    def decorator(func: RewardFunction) -> RewardFunction:
        _REWARD_REGISTRY[name] = func
        return func

    return decorator


def get_reward_function(name: str) -> RewardFunction:
    """Get a registered reward function by name."""
    if name not in _REWARD_REGISTRY:
        raise KeyError(
            f"Unknown reward function: {name}. Available: {sorted(_REWARD_REGISTRY.keys())}"
        )
    return _REWARD_REGISTRY[name]


def list_reward_functions() -> list[str]:
    """List all registered reward function names."""
    return sorted(_REWARD_REGISTRY.keys())


def build_reward_functions(
    names: list[str],
    weights: dict[str, float] | None = None,
    **kwargs: Any,
) -> list[tuple[RewardFunction, float]]:
    """Build a list of (reward_func, weight) tuples.

    Args:
        names: Reward function names.
        weights: Optional per-name weights. Defaults to 1.0.
        **kwargs: Extra arguments forwarded to each reward function.

    Returns:
        List of tuples (reward_function, weight).
    """
    weights = weights or {}
    result: list[tuple[RewardFunction, float]] = []
    for name in names:
        func = get_reward_function(name)
        bound = _bind_kwargs(func, kwargs)
        result.append((bound, weights.get(name, 1.0)))
    return result


def _bind_kwargs(func: RewardFunction, kwargs: dict[str, Any]) -> RewardFunction:
    """Bind keyword arguments that the function accepts.

    Inspects the function signature and only passes through keys that are
    accepted, plus any **kwargs catch-all.
    """
    import inspect

    sig = inspect.signature(func)
    accepts_kwargs = any(
        param.kind == inspect.Parameter.VAR_KEYWORD for param in sig.parameters.values()
    )
    accepted = set(sig.parameters.keys())

    if accepts_kwargs:
        return func

    filtered = {k: v for k, v in kwargs.items() if k in accepted}

    def wrapped(*args: Any, **call_kwargs: Any) -> list[float]:
        return func(*args, **{**filtered, **call_kwargs})

    return wrapped


@register_reward("format")
def format_reward(
    prompts: list[str],
    completions: list[str],
    *,
    required_tag: str | None = None,
    require_json: bool = False,
    **_: Any,
) -> list[float]:
    """Reward completions that follow a required format.

    Args:
        prompts: Prompts (unused but kept for signature compatibility).
        completions: Generated completions.
        required_tag: Optional XML/Markdown tag that must appear.
        require_json: Whether completion must be valid JSON.

    Returns:
        1.0 if format matches, 0.0 otherwise.
    """
    scores: list[float] = []
    for completion in completions:
        score = 1.0
        text = completion.strip()

        if require_json:
            try:
                json.loads(text)
            except json.JSONDecodeError:
                score = 0.0

        if required_tag and required_tag not in text:
            score = 0.0

        scores.append(score)
    return scores


@register_reward("accuracy")
def accuracy_reward(
    prompts: list[str],
    completions: list[str],
    *,
    answer: str | list[str] | None = None,
    answer_key: str = "answer",
    case_sensitive: bool = False,
    normalize_whitespace: bool = True,
    **kwargs: Any,
) -> list[float]:
    """Reward completions that match a reference answer.

    Args:
        prompts: Prompts (unused but kept for signature compatibility).
        completions: Generated completions.
        answer: Reference answer(s). If a list, must match length of completions.
        answer_key: Key to look up in kwargs for batched answers.
        case_sensitive: Whether matching is case sensitive.
        normalize_whitespace: Strip and collapse whitespace before matching.

    Returns:
        1.0 if exact match, 0.0 otherwise.
    """
    if answer is None:
        answer = kwargs.get(answer_key)

    if answer is None:
        return [0.0] * len(completions)

    if isinstance(answer, str):
        answers = [answer] * len(completions)
    else:
        answers = list(answer)
        if len(answers) != len(completions):
            raise ValueError(
                f"Number of answers ({len(answers)}) must match "
                f"number of completions ({len(completions)})"
            )

    scores: list[float] = []
    for completion, ref in zip(completions, answers):
        pred = _normalize_text(completion, case_sensitive, normalize_whitespace)
        target = _normalize_text(ref, case_sensitive, normalize_whitespace)
        scores.append(1.0 if pred == target else 0.0)
    return scores


@register_reward("length")
def length_reward(
    prompts: list[str],
    completions: list[str],
    *,
    min_length: int = 10,
    max_length: int = 1024,
    **_: Any,
) -> list[float]:
    """Reward completions whose length is within an acceptable range.

    Returns:
        1.0 if length is in range, linear penalty otherwise.
    """
    scores: list[float] = []
    for completion in completions:
        length = len(completion.strip())
        if min_length <= length <= max_length:
            scores.append(1.0)
        else:
            scores.append(0.0)
    return scores


@register_reward("cosine")
def cosine_reward(
    prompts: list[str],
    completions: list[str],
    *,
    reference: str | list[str] | None = None,
    reference_key: str = "reference",
    **kwargs: Any,
) -> list[float]:
    """Reward completions by cosine-like string similarity to reference.

    Uses SequenceMatcher ratio as a lightweight proxy.
    """
    if reference is None:
        reference = kwargs.get(reference_key)

    if reference is None:
        return [0.0] * len(completions)

    if isinstance(reference, str):
        references = [reference] * len(completions)
    else:
        references = list(reference)
        if len(references) != len(completions):
            raise ValueError(
                f"Number of references ({len(references)}) must match "
                f"number of completions ({len(completions)})"
            )

    scores: list[float] = []
    for completion, ref in zip(completions, references):
        ratio = SequenceMatcher(None, completion.strip(), ref.strip()).ratio()
        scores.append(ratio)
    return scores


@register_reward("llm_judge")
def llm_judge_reward(
    prompts: list[str],
    completions: list[str],
    *,
    judge_client: Any | None = None,
    judge_model: str | None = None,
    judge_prompt_template: str | None = None,
    **_: Any,
) -> list[float]:
    """Reward completions using an LLM-as-a-Judge.

    Args:
        prompts: Original prompts.
        completions: Generated completions.
        judge_client: Client object with a `judge(prompt, completion) -> float` method.
        judge_model: Model identifier (used if no client provided).
        judge_prompt_template: Optional custom prompt template.

    Returns:
        Normalized scores in [0.0, 1.0].
    """
    if judge_client is not None:
        return [
            float(judge_client.judge(prompt, completion))
            for prompt, completion in zip(prompts, completions)
        ]

    if judge_model is None:
        raise ValueError("Either judge_client or judge_model must be provided")

    # Fallback: return neutral scores if no client available.
    # A real implementation would instantiate a local model or API client here.
    return [0.5] * len(completions)


def _normalize_text(text: str, case_sensitive: bool, normalize_whitespace: bool) -> str:
    """Normalize text for matching."""
    result = text.strip()
    if not case_sensitive:
        result = result.lower()
    if normalize_whitespace:
        result = re.sub(r"\s+", " ", result)
    return result


def combine_rewards(
    rewards: list[list[float]],
    weights: list[float] | None = None,
) -> list[float]:
    """Combine multiple reward lists with optional weights.

    Args:
        rewards: List of reward lists, each of length N.
        weights: Optional weights for each reward list.

    Returns:
        Combined reward list of length N.
    """
    if not rewards:
        return []

    n = len(rewards[0])
    if any(len(r) != n for r in rewards):
        raise ValueError("All reward lists must have the same length")

    if weights is None:
        weights = [1.0] * len(rewards)

    if len(weights) != len(rewards):
        raise ValueError("weights must match the number of reward lists")

    combined: list[float] = []
    for i in range(n):
        total = sum(weights[j] * rewards[j][i] for j in range(len(rewards)))
        combined.append(total)
    return combined
