"""Build preference pairs from group completions and rewards."""

from __future__ import annotations

import uuid
from typing import Any

from src.data_flywheel.schemas import PreferencePair


class PreferenceBuilder:
    """Build preference pairs from GRPO-style group completions."""

    def __init__(
        self,
        min_margin: float = 0.0,
        top_k: int = 1,
        dedup: bool = True,
    ) -> None:
        """Initialize preference builder.

        Args:
            min_margin: Minimum reward difference required to create a pair.
            top_k: Number of (chosen, rejected) pairs to generate per prompt.
            dedup: Whether to skip identical chosen/rejected pairs.
        """
        self.min_margin = min_margin
        self.top_k = top_k
        self.dedup = dedup

    def build(
        self,
        prompt: str,
        completions: list[str],
        rewards: list[float],
        generation_policy: str = "",
        judge_model: str | None = None,
        lineage_id: str = "",
    ) -> list[PreferencePair]:
        """Build preference pairs from completions and rewards.

        Args:
            prompt: Original prompt.
            completions: List of generated completions.
            rewards: Reward for each completion.
            generation_policy: Model/run that generated the completions.
            judge_model: Judge model used to score (if any).
            lineage_id: Lineage ID to attach.

        Returns:
            List of PreferencePair objects.
        """
        if len(completions) != len(rewards):
            raise ValueError("completions and rewards must have the same length")

        indexed = list(enumerate(zip(completions, rewards)))
        indexed.sort(key=lambda x: x[1][1], reverse=True)

        pairs: list[PreferencePair] = []
        seen: set[tuple[str, str]] = set()

        for i in range(min(self.top_k, len(indexed) // 2)):
            chosen_idx, (chosen, reward_chosen) = indexed[i]
            rejected_idx, (rejected, reward_rejected) = indexed[-(i + 1)]

            if reward_chosen - reward_rejected < self.min_margin:
                continue

            if self.dedup:
                key = (chosen.strip(), rejected.strip())
                if key in seen or chosen.strip() == rejected.strip():
                    continue
                seen.add(key)

            pairs.append(
                PreferencePair(
                    id=str(uuid.uuid4()),
                    prompt=prompt,
                    chosen=chosen,
                    rejected=rejected,
                    generation_policy=generation_policy,
                    judge_model=judge_model,
                    reward_chosen=reward_chosen,
                    reward_rejected=reward_rejected,
                    lineage_id=lineage_id,
                )
            )

        return pairs

    def build_batch(
        self,
        prompts: list[str],
        completions: list[list[str]],
        rewards: list[list[float]],
        **kwargs: Any,
    ) -> list[PreferencePair]:
        """Build preference pairs for a batch of prompts."""
        pairs: list[PreferencePair] = []
        for prompt, comps, rews in zip(prompts, completions, rewards):
            pairs.extend(self.build(prompt, comps, rews, **kwargs))
        return pairs
