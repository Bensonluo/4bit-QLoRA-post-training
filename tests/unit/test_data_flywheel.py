"""Unit tests for data flywheel modules."""

import tempfile
from unittest.mock import MagicMock

import pytest

from src.data_flywheel import (
    BadCaseMiner,
    DataFlywheelPipeline,
    DatasetItem,
    DataSynthesizer,
    LineageRecord,
    LocalDatasetRegistry,
    PreferenceBuilder,
    RuleJudgeClient,
)


class TestSchemas:
    def test_dataset_item_roundtrip(self) -> None:
        item = DatasetItem(id="1", prompt="hello", response="world")
        data = item.to_dict()
        restored = DatasetItem.from_dict(data)
        assert restored.prompt == "hello"
        assert restored.response == "world"


class TestPreferenceBuilder:
    def test_build_basic_pair(self) -> None:
        builder = PreferenceBuilder()
        pairs = builder.build(
            prompt="p",
            completions=["good", "bad"],
            rewards=[1.0, 0.0],
        )
        assert len(pairs) == 1
        assert pairs[0].chosen == "good"
        assert pairs[0].rejected == "bad"

    def test_min_margin_filters(self) -> None:
        builder = PreferenceBuilder(min_margin=0.5)
        pairs = builder.build(
            prompt="p",
            completions=["a", "b"],
            rewards=[0.6, 0.4],
        )
        assert len(pairs) == 0

    def test_dedup_identical(self) -> None:
        builder = PreferenceBuilder(dedup=True)
        pairs = builder.build(
            prompt="p",
            completions=["same", "same"],
            rewards=[1.0, 0.0],
        )
        assert len(pairs) == 0


class TestDatasetRegistry:
    def test_register_and_load(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            registry = LocalDatasetRegistry(base_dir=tmp)
            items = [
                DatasetItem(id="1", prompt="p1", response="r1"),
                DatasetItem(id="2", prompt="p2", response="r2"),
            ]
            lineage = LineageRecord(
                lineage_id="v1",
                operation="test",
                input_hash="abc",
                output_hash="",
            )
            version = registry.register("sft", items, lineage)

            loaded = registry.load("sft", version)
            assert len(loaded) == 2
            assert loaded[0]["prompt"] == "p1"

    def test_list_versions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            registry = LocalDatasetRegistry(base_dir=tmp)
            items = [DatasetItem(id="1", prompt="p1")]
            lineage = LineageRecord(
                lineage_id="v1",
                operation="test",
                input_hash="abc",
                output_hash="",
            )
            registry.register("sft", items, lineage)
            versions = registry.list_versions("sft")
            assert versions == ["v1"]


class TestSynthesizer:
    def test_evol_instruct(self) -> None:
        client = MagicMock()
        client.generate.return_value = "Rewritten prompt"
        synth = DataSynthesizer(client, strategy="evol_instruct")
        seed = [DatasetItem(id="1", prompt="Original", response="Answer")]
        outputs = synth.synthesize(seed)
        assert len(outputs) == 1
        assert outputs[0].prompt == "Rewritten prompt"
        assert outputs[0].response == "Answer"

    def test_self_instruct(self) -> None:
        client = MagicMock()
        client.generate.return_value = "New prompt"
        synth = DataSynthesizer(client, strategy="self_instruct")
        seed = [DatasetItem(id="1", prompt="Original", response="Answer")]
        outputs = synth.synthesize(seed)
        assert outputs[0].prompt == "New prompt"
        assert outputs[0].response is None

    def test_unknown_strategy(self) -> None:
        with pytest.raises(ValueError, match="Unknown strategy"):
            DataSynthesizer(MagicMock(), strategy="unknown")


class TestBadCaseMiner:
    def test_mine_low_scores(self) -> None:
        miner = BadCaseMiner(threshold=0.5)
        eval_results = [
            {"prompt": "p1", "response": "r1", "score": 0.2},
            {"prompt": "p2", "response": "r2", "score": 0.8},
        ]
        bad_cases = miner.mine(eval_results)
        assert len(bad_cases) == 1
        assert bad_cases[0].prompt == "p1"


class TestRuleJudgeClient:
    def test_exact_match(self) -> None:
        judge = RuleJudgeClient()
        assert judge.judge("p", "answer", answer="answer") == 1.0
        assert judge.judge("p", "wrong", answer="answer") == 0.0


class TestDataFlywheelPipeline:
    def test_run_iteration(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            registry = LocalDatasetRegistry(base_dir=tmp)

            client = MagicMock()
            client.generate.return_value = "Synthetic prompt"
            synthesizer = DataSynthesizer(client, strategy="self_instruct")
            judge = RuleJudgeClient(answer_key="answer")
            builder = PreferenceBuilder()

            pipeline = DataFlywheelPipeline(
                synthesizer=synthesizer,
                preference_builder=builder,
                registry=registry,
                judge=judge,
            )

            seed = [DatasetItem(id="1", prompt="What is 2+2?", response="4")]
            prompt_completions = [
                {
                    "prompt": "What is 2+2?",
                    "completions": ["4", "5"],
                    "rewards": [1.0, 0.0],
                }
            ]

            result = pipeline.run_iteration(
                seed_data=seed,
                prompt_completions=prompt_completions,
                generation_policy="test-policy",
            )

            assert "sft_dataset_version" in result
            assert "dpo_dataset_version" in result
            assert result["num_synthetic"] == 1
            assert result["num_preferences"] == 1
