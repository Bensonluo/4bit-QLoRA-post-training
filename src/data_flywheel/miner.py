"""Bad-case mining from evaluation results for data flywheel."""

from __future__ import annotations

import uuid
from typing import Any

from src.data_flywheel.schemas import DatasetItem


class BadCaseMiner:
    """Mine bad cases from evaluation results for retraining."""

    def __init__(
        self,
        judge: Any | None = None,
        threshold: float = 0.3,
        min_reward: float | None = None,
    ) -> None:
        """Initialize bad case miner.

        Args:
            judge: Optional judge client to rescore responses.
            threshold: Score threshold below which examples are considered bad.
            min_reward: Alias for threshold.
        """
        self.judge = judge
        self.threshold = min_reward if min_reward is not None else threshold

    def mine(
        self,
        eval_results: list[dict[str, Any]],
        generation_policy: str = "",
        lineage_id: str = "",
    ) -> list[DatasetItem]:
        """Mine bad cases from evaluation results.

        Expected eval_result format:
            {
                "prompt": "...",
                "response": "...",
                "score": 0.2,  # or "reward"
            }
        """
        bad_cases: list[DatasetItem] = []

        for result in eval_results:
            prompt = result.get("prompt", "")
            response = result.get("response", "")
            score = result.get("score", result.get("reward", 0.0))

            if score < self.threshold:
                bad_cases.append(
                    DatasetItem(
                        id=str(uuid.uuid4()),
                        prompt=prompt,
                        response=response,
                        source="bad_case_mining",
                        metadata={
                            "eval_score": score,
                            "generation_policy": generation_policy,
                        },
                        lineage_id=lineage_id,
                    )
                )

        return bad_cases
