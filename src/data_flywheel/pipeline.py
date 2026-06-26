"""End-to-end data flywheel pipeline orchestration."""

from __future__ import annotations

from typing import Any

from src.data_flywheel.dataset_registry import LocalDatasetRegistry, new_lineage_id
from src.data_flywheel.judge import JudgeClient
from src.data_flywheel.miner import BadCaseMiner
from src.data_flywheel.preference_builder import PreferenceBuilder
from src.data_flywheel.schemas import DatasetItem, LineageRecord, PreferencePair
from src.data_flywheel.synthesizer import DataSynthesizer


class DataFlywheelPipeline:
    """Orchestrate one iteration of the data flywheel."""

    def __init__(
        self,
        synthesizer: DataSynthesizer,
        preference_builder: PreferenceBuilder,
        registry: LocalDatasetRegistry,
        judge: JudgeClient,
        miner: BadCaseMiner | None = None,
    ) -> None:
        """Initialize pipeline."""
        self.synthesizer = synthesizer
        self.preference_builder = preference_builder
        self.registry = registry
        self.judge = judge
        self.miner = miner

    def run_iteration(
        self,
        seed_data: list[DatasetItem],
        prompt_completions: list[dict[str, Any]],
        generation_policy: str = "",
        run_id: str | None = None,
        n_synthetic: int | None = None,
    ) -> dict[str, str]:
        """Run one flywheel iteration.

        Args:
            seed_data: Seed examples for synthesis.
            prompt_completions: List of dicts with keys:
                {
                    "prompt": str,
                    "completions": list[str],
                    "rewards": list[float],  # optional
                }
            generation_policy: Policy model that generated completions.
            run_id: Optional MLflow run_id.
            n_synthetic: Number of synthetic items to generate.

        Returns:
            Dict with dataset version ids:
                {
                    "sft_dataset_version": str,
                    "dpo_dataset_version": str,
                }
        """
        input_hash = self._hash_seed(seed_data)

        # 1. Synthesize new SFT data
        synthetic_items = self.synthesizer.synthesize(seed_data, n_outputs=n_synthetic)
        synth_lineage = LineageRecord(
            lineage_id=new_lineage_id(),
            operation="synthesize",
            input_hash=input_hash,
            output_hash="",
            config={"strategy": self.synthesizer.strategy},
            run_id=run_id,
        )
        sft_version = self.registry.register(
            "sft_synthetic",
            synthetic_items,
            synth_lineage,
        )

        # 2. Build preference pairs from completions
        preference_pairs: list[PreferencePair] = []
        for item in prompt_completions:
            prompt = item["prompt"]
            completions = item["completions"]
            rewards = item.get("rewards")

            if rewards is None:
                rewards = [self.judge.judge(prompt, completion) for completion in completions]

            pairs = self.preference_builder.build(
                prompt=prompt,
                completions=completions,
                rewards=rewards,
                generation_policy=generation_policy,
                judge_model=getattr(self.judge, "model_name", None),
                lineage_id=sft_version,
            )
            preference_pairs.extend(pairs)

        dpo_lineage = LineageRecord(
            lineage_id=new_lineage_id(),
            operation="preference_generation",
            input_hash=input_hash,
            output_hash="",
            parent_lineage_ids=[sft_version],
            config={"num_pairs": len(preference_pairs)},
            run_id=run_id,
        )
        dpo_version = self.registry.register(
            "dpo_preferences",
            preference_pairs,
            dpo_lineage,
        )

        return {
            "sft_dataset_version": sft_version,
            "dpo_dataset_version": dpo_version,
            "num_synthetic": len(synthetic_items),
            "num_preferences": len(preference_pairs),
        }

    @staticmethod
    def _hash_seed(seed_data: list[DatasetItem]) -> str:
        """Compute deterministic hash of seed data."""
        import hashlib
        import json

        content = json.dumps(
            [item.to_dict() for item in seed_data],
            sort_keys=True,
            ensure_ascii=False,
            default=str,
        )
        return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]
