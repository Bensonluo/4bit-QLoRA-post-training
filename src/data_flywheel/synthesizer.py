"""Data synthesis strategies for expanding seed datasets."""

from __future__ import annotations

import uuid
from collections.abc import Callable
from typing import Any

from src.data_flywheel.schemas import DatasetItem


class DataSynthesizer:
    """Synthesize new training examples from seed data.

    Supports multiple strategies:
      - evol_instruct: rewrite instructions to be more complex/diverse
      - self_instruct: generate new instructions similar to seed
      - paraphrase: paraphrase seed responses
    """

    STRATEGIES: set[str] = {"evol_instruct", "self_instruct", "paraphrase"}

    def __init__(
        self,
        model_client: Any,
        strategy: str = "evol_instruct",
        config: dict[str, Any] | None = None,
    ) -> None:
        """Initialize synthesizer.

        Args:
            model_client: Client with a `generate(prompt: str) -> str` method.
            strategy: Synthesis strategy name.
            config: Strategy-specific configuration.
        """
        if strategy not in self.STRATEGIES:
            raise ValueError(f"Unknown strategy: {strategy}. Valid: {sorted(self.STRATEGIES)}")

        self.model_client = model_client
        self.strategy = strategy
        self.config = config or {}

    def synthesize(
        self,
        seed_items: list[DatasetItem],
        n_outputs: int | None = None,
    ) -> list[DatasetItem]:
        """Generate synthetic items from seed items.

        Args:
            seed_items: Seed examples.
            n_outputs: Number of synthetic items to generate. If None, generates
                one per seed item.

        Returns:
            List of synthetic DatasetItem objects.
        """
        if n_outputs is None:
            n_outputs = len(seed_items)

        outputs: list[DatasetItem] = []
        strategy_fn = self._get_strategy_fn()

        for i in range(n_outputs):
            seed = seed_items[i % len(seed_items)]
            prompt, response = strategy_fn(seed)
            outputs.append(
                DatasetItem(
                    id=str(uuid.uuid4()),
                    prompt=prompt,
                    response=response,
                    source=f"synthetic_{self.strategy}",
                    metadata={
                        "strategy": self.strategy,
                        "seed_id": seed.id,
                        **self.config,
                    },
                )
            )

        return outputs

    def _get_strategy_fn(self) -> Callable[[DatasetItem], tuple[str, str | None]]:
        """Return the strategy implementation."""
        if self.strategy == "evol_instruct":
            return self._evol_instruct
        if self.strategy == "self_instruct":
            return self._self_instruct
        if self.strategy == "paraphrase":
            return self._paraphrase
        raise ValueError(f"Unhandled strategy: {self.strategy}")

    def _evol_instruct(self, seed: DatasetItem) -> tuple[str, str | None]:
        """Rewrite instruction to be more complex."""
        template = (
            "Rewrite the following instruction to make it more challenging "
            "and detailed, while keeping the answerable.\n\n"
            "Original instruction:\n{prompt}\n\n"
            "Rewritten instruction:"
        )
        prompt = self.model_client.generate(template.format(prompt=seed.prompt))
        return prompt.strip(), seed.response

    def _self_instruct(self, seed: DatasetItem) -> tuple[str, str | None]:
        """Generate a new instruction in the same domain."""
        template = (
            "Here is an example instruction:\n{prompt}\n\n"
            "Generate a new, different instruction on a similar topic:"
        )
        prompt = self.model_client.generate(template.format(prompt=seed.prompt))
        return prompt.strip(), None

    def _paraphrase(self, seed: DatasetItem) -> tuple[str, str | None]:
        """Paraphrase the seed response while keeping the prompt."""
        if seed.response is None:
            return seed.prompt, None

        template = (
            "Paraphrase the following response without changing its meaning.\n\n"
            "Response:\n{response}\n\n"
            "Paraphrased response:"
        )
        response = self.model_client.generate(template.format(response=seed.response))
        return seed.prompt, response.strip()
