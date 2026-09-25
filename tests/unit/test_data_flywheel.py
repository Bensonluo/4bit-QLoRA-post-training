"""Unit tests for the data flywheel module (schemas, registry, judge,
synthesizer, preference builder, pipeline). No model weights are loaded."""

from __future__ import annotations

from collections import deque
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from src.data_flywheel.dataset_registry import LocalDatasetRegistry, new_lineage_id
from src.data_flywheel.judge import JudgeClient, LocalJudgeClient, RuleJudgeClient
from src.data_flywheel.miner import BadCaseMiner
from src.data_flywheel.pipeline import DataFlywheelPipeline
from src.data_flywheel.preference_builder import PreferenceBuilder
from src.data_flywheel.schemas import DatasetItem, LineageRecord, PreferencePair
from src.data_flywheel.synthesizer import DataSynthesizer


def make_item(prompt: str = "q", response: str | None = "a", **kw: Any) -> DatasetItem:
    return DatasetItem(id=kw.pop("id", "s1"), prompt=prompt, response=response, **kw)


def make_lineage(operation: str = "test") -> LineageRecord:
    return LineageRecord(
        lineage_id=new_lineage_id(),
        operation=operation,
        input_hash="abc",
        output_hash="",
    )


# ─── schemas ─────────────────────────────────────────────────────


class TestSchemas:
    def test_dataset_item_roundtrip(self) -> None:
        item = DatasetItem(id="i1", prompt="q", response="a", source="seed")
        d = item.to_dict()
        restored = DatasetItem.from_dict(d)
        assert restored == item
        assert isinstance(d["created_at"], str)  # ISO-serialized

    def test_preference_pair_roundtrip(self) -> None:
        pair = PreferencePair(
            id="p1",
            prompt="q",
            chosen="good",
            rejected="bad",
            reward_chosen=0.9,
            reward_rejected=0.1,
        )
        assert PreferencePair.from_dict(pair.to_dict()) == pair

    def test_lineage_record_defaults(self) -> None:
        rec = LineageRecord(lineage_id="v1", operation="op", input_hash="h", output_hash="o")
        restored = LineageRecord.from_dict(rec.to_dict())
        assert restored == rec
        assert restored.parent_lineage_ids == []
        assert restored.run_id is None


# ─── LocalDatasetRegistry ────────────────────────────────────────


class TestLocalDatasetRegistry:
    def test_register_writes_jsonl_and_manifest(self, tmp_path: Any) -> None:
        reg = LocalDatasetRegistry(str(tmp_path / "reg"))
        lineage = make_lineage()
        version = reg.register("ds", [make_item()], lineage)

        assert version == lineage.lineage_id
        manifest = tmp_path / "reg" / "ds" / "manifest.json"
        assert manifest.exists()
        assert lineage.output_hash != ""  # filled by register
        records = reg.load("ds", version)
        assert len(records) == 1 and records[0]["prompt"] == "q"

    def test_load_latest_version_by_default(self, tmp_path: Any) -> None:
        reg = LocalDatasetRegistry(str(tmp_path / "reg"))
        v1 = reg.register("ds", [make_item(prompt="first")], make_lineage())
        v2 = reg.register("ds", [make_item(prompt="second")], make_lineage())
        assert v1 != v2
        assert reg.load("ds")[-1]["prompt"] == "second"  # last manifest key

    def test_load_unknown_dataset_raises(self, tmp_path: Any) -> None:
        reg = LocalDatasetRegistry(str(tmp_path / "reg"))
        with pytest.raises(FileNotFoundError):
            reg.load("nope")

    def test_load_unknown_version_raises(self, tmp_path: Any) -> None:
        reg = LocalDatasetRegistry(str(tmp_path / "reg"))
        reg.register("ds", [make_item()], make_lineage())
        with pytest.raises(ValueError, match="not found"):
            reg.load("ds", "v_missing")

    def test_lineage_roundtrip_and_list_versions(self, tmp_path: Any) -> None:
        reg = LocalDatasetRegistry(str(tmp_path / "reg"))
        lineage = make_lineage("synthesize")
        version = reg.register("ds", [make_item()], lineage)

        stored = reg.get_lineage("ds", version)
        assert stored.operation == "synthesize"
        assert stored.output_hash == lineage.output_hash
        assert reg.list_versions("ds") == [version]
        assert reg.list_versions("never-registered") == []

    def test_get_lineage_unknown_version_raises(self, tmp_path: Any) -> None:
        reg = LocalDatasetRegistry(str(tmp_path / "reg"))
        reg.register("ds", [make_item()], make_lineage())
        with pytest.raises(ValueError, match="not found"):
            reg.get_lineage("ds", "v_missing")


# ─── judges ──────────────────────────────────────────────────────


class TestJudgeClients:
    def test_rule_judge_exact_match_case_insensitive(self) -> None:
        judge = RuleJudgeClient(answer_key="answer")
        assert judge.judge("p", "Paris", answer="paris") == 1.0
        assert judge.judge("p", "London", answer="paris") == 0.0

    def test_rule_judge_missing_answer_is_neutral(self) -> None:
        judge = RuleJudgeClient()
        assert judge.judge("p", "anything") == 0.5

    def test_judge_pair_default_scores_both(self) -> None:
        class _Fixed(JudgeClient):
            def judge(self, prompt: str, completion: str, **_: Any) -> float:
                return 0.7

        chosen, rejected = _Fixed().judge_pair("p", "x", "y")
        assert (chosen, rejected) == (0.7, 0.7)

    def _local_judge(
        self, mock_tok: MagicMock, mock_model: MagicMock, decoded: str
    ) -> LocalJudgeClient:
        tokenizer = MagicMock()
        tokenizer.return_value = {"input_ids": MagicMock()}
        tokenizer.decode.return_value = decoded
        model = MagicMock()
        model.device.type = "cpu"
        model.generate.return_value = [[1, 2, 3]]
        mock_tok.return_value = tokenizer
        mock_model.return_value = model
        return LocalJudgeClient(model_name="fake-model")

    @patch("transformers.AutoModelForCausalLM.from_pretrained")
    @patch("transformers.AutoTokenizer.from_pretrained")
    def test_local_judge_extracts_score(self, mock_tok: MagicMock, mock_model: MagicMock) -> None:
        client = self._local_judge(mock_tok, mock_model, "full prompt text 8")
        assert client.judge("p", "c") == pytest.approx(0.8)  # 8 / 10

    @patch("transformers.AutoModelForCausalLM.from_pretrained")
    @patch("transformers.AutoTokenizer.from_pretrained")
    def test_local_judge_no_number_is_neutral(
        self, mock_tok: MagicMock, mock_model: MagicMock
    ) -> None:
        client = self._local_judge(mock_tok, mock_model, "no digits at all")
        assert client.judge("p", "c") == 0.5

    @patch("transformers.AutoModelForCausalLM.from_pretrained")
    @patch("transformers.AutoTokenizer.from_pretrained")
    def test_local_judge_clamps_and_lazy_loads_once(
        self, mock_tok: MagicMock, mock_model: MagicMock
    ) -> None:
        client = self._local_judge(mock_tok, mock_model, "score: 15")
        assert client.judge("p", "c") == 1.0  # 15/10 clamped
        client.judge("p", "c2")
        assert mock_model.call_count == 1  # lazy-load cached

    @patch("transformers.AutoModelForCausalLM.from_pretrained")
    @patch("transformers.AutoTokenizer.from_pretrained")
    def test_local_judge_moves_inputs_to_gpu_device(
        self, mock_tok: MagicMock, mock_model: MagicMock
    ) -> None:
        tokenizer = MagicMock()
        ids, mask = MagicMock(), MagicMock()
        tokenizer.return_value = {"input_ids": ids, "attention_mask": mask}
        tokenizer.decode.return_value = "full prompt text 8"
        model = MagicMock()
        model.device.type = "cuda"  # non-CPU: tensors must follow the model
        model.generate.return_value = [[1, 2, 3]]
        mock_tok.return_value = tokenizer
        mock_model.return_value = model

        client = LocalJudgeClient(model_name="fake-model")
        assert client.judge("p", "c") == pytest.approx(0.8)

        ids.to.assert_called_once_with(model.device)
        mask.to.assert_called_once_with(model.device)


# ─── synthesizer ─────────────────────────────────────────────────


class _FakeGenClient:
    def __init__(self, output: str = "generated text") -> None:
        self.output = output
        self.calls: list[str] = []

    def generate(self, prompt: str) -> str:
        self.calls.append(prompt)
        return self.output


class TestSynthesizer:
    def test_unknown_strategy_rejected(self) -> None:
        with pytest.raises(ValueError, match="Unknown strategy"):
            DataSynthesizer(_FakeGenClient(), strategy="bogus")

    def test_mutated_strategy_rejected_at_synthesize(self) -> None:
        # Post-construction mutation bypasses __init__ validation — the
        # strategy lookup still guards with a clear error.
        synth = DataSynthesizer(_FakeGenClient(), strategy="paraphrase")
        synth.strategy = "bogus"
        with pytest.raises(ValueError, match="Unhandled strategy"):
            synth.synthesize([make_item()])

    def test_default_n_outputs_is_one_per_seed(self) -> None:
        synth = DataSynthesizer(_FakeGenClient("evolved"), strategy="evol_instruct")
        seeds = [make_item(prompt=f"q{j}") for j in range(3)]
        items = synth.synthesize(seeds)
        assert len(items) == 3
        assert all(i.prompt == "evolved" for i in items)
        assert all(i.source == "synthetic_evol_instruct" for i in items)
        assert {i.metadata["seed_id"] for i in items} == {s.id for s in seeds}

    def test_n_outputs_cycles_seed_items(self) -> None:
        synth = DataSynthesizer(_FakeGenClient("x"), strategy="paraphrase")
        items = synth.synthesize([make_item(response="r")], n_outputs=3)
        assert len(items) == 3
        assert {i.metadata["seed_id"] for i in items} == {"s1"}

    def test_self_instruct_drops_response(self) -> None:
        synth = DataSynthesizer(_FakeGenClient("new instruction"), strategy="self_instruct")
        (item,) = synth.synthesize([make_item(response="original")])
        assert item.prompt == "new instruction"
        assert item.response is None

    def test_paraphrase_keeps_prompt_rewrites_response(self) -> None:
        synth = DataSynthesizer(_FakeGenClient("rephrased"), strategy="paraphrase")
        (item,) = synth.synthesize([make_item(prompt="q", response="r")])
        assert (item.prompt, item.response) == ("q", "rephrased")

    def test_paraphrase_with_no_response_is_passthrough(self) -> None:
        synth = DataSynthesizer(_FakeGenClient("never called"), strategy="paraphrase")
        (item,) = synth.synthesize([make_item(prompt="q", response=None)])
        assert (item.prompt, item.response) == ("q", None)


# ─── preference builder ──────────────────────────────────────────


class TestPreferenceBuilder:
    def test_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="same length"):
            PreferenceBuilder().build("p", ["a"], [0.1, 0.2])

    def test_best_vs_worst_pair(self) -> None:
        builder = PreferenceBuilder(top_k=1)
        (pair,) = builder.build("p", ["bad", "good"], [0.1, 0.9])
        assert (pair.chosen, pair.rejected) == ("good", "bad")
        assert pair.reward_chosen == 0.9
        assert pair.reward_rejected == 0.1

    def test_min_margin_filters(self) -> None:
        builder = PreferenceBuilder(min_margin=0.5)
        assert builder.build("p", ["a", "b"], [0.5, 0.4]) == []

    def test_top_k_two_makes_nested_pairs(self) -> None:
        builder = PreferenceBuilder(top_k=2)
        pairs = builder.build("p", ["w", "second", "third", "best"], [0.0, 0.3, 0.6, 1.0])
        assert [(p.chosen, p.rejected) for p in pairs] == [("best", "w"), ("third", "second")]

    def test_single_completion_yields_no_pairs(self) -> None:
        assert PreferenceBuilder().build("p", ["only"], [0.9]) == []

    def test_dedup_skips_identical_texts(self) -> None:
        builder = PreferenceBuilder()
        # Same text twice with different rewards → chosen.strip() == rejected.strip()
        assert builder.build("p", ["same", "same"], [0.9, 0.1]) == []

    def test_build_batch_flattens(self) -> None:
        builder = PreferenceBuilder()
        pairs = builder.build_batch(
            ["p1", "p2"],
            [["a", "b"], ["c", "d"]],
            [[0.1, 0.9], [0.8, 0.2]],
        )
        assert [p.chosen for p in pairs] == ["b", "c"]


# ─── bad-case miner ─────────────────────────────────────────────


class TestBadCaseMiner:
    def _results(self, *scores: float, key: str = "score") -> list[dict[str, Any]]:
        return [{"prompt": f"q{i}", "response": f"a{i}", key: s} for i, s in enumerate(scores)]

    def test_below_threshold_is_mined_with_metadata(self) -> None:
        miner = BadCaseMiner(threshold=0.3)
        items = miner.mine(self._results(0.1, 0.9), generation_policy="qwen-1.5b", lineage_id="v1")
        assert len(items) == 1
        item = items[0]
        assert item.source == "bad_case_mining"
        assert item.metadata["eval_score"] == 0.1
        assert item.metadata["generation_policy"] == "qwen-1.5b"
        assert item.lineage_id == "v1"

    def test_boundary_score_equal_threshold_not_mined(self) -> None:
        miner = BadCaseMiner(threshold=0.3)
        assert miner.mine(self._results(0.3)) == []  # strict <

    def test_reward_key_fallback(self) -> None:
        miner = BadCaseMiner(threshold=0.5)
        items = miner.mine(self._results(0.2, key="reward"))
        assert len(items) == 1
        assert items[0].metadata["eval_score"] == 0.2

    def test_missing_score_defaults_to_zero_and_mines(self) -> None:
        miner = BadCaseMiner(threshold=0.3)
        items = miner.mine([{"prompt": "q", "response": "a"}])
        assert len(items) == 1
        assert items[0].metadata["eval_score"] == 0.0

    def test_min_reward_alias_overrides_threshold(self) -> None:
        miner = BadCaseMiner(threshold=0.9, min_reward=0.2)
        # 0.5 < 0.2? No. 0.5 < 0.9? Yes. Alias must win: NOT mined.
        assert miner.mine(self._results(0.5)) == []
        assert miner.mine(self._results(0.1)) != []


# ─── pipeline ────────────────────────────────────────────────────


class _SeqJudge(JudgeClient):
    """Judge stub serving queued scores in order."""

    def __init__(self, *scores: float) -> None:
        self._scores = deque(scores)
        self.calls = 0

    def judge(self, prompt: str, completion: str, **_: Any) -> float:
        self.calls += 1
        return self._scores.popleft()


class TestDataFlywheelPipeline:
    def _pipeline(self, tmp_path: Any, judge: JudgeClient) -> DataFlywheelPipeline:
        return DataFlywheelPipeline(
            synthesizer=DataSynthesizer(_FakeGenClient("evolved prompt")),
            preference_builder=PreferenceBuilder(),
            registry=LocalDatasetRegistry(str(tmp_path / "reg")),
            judge=judge,
        )

    def test_full_iteration_registers_both_datasets(self, tmp_path: Any) -> None:
        pipe = self._pipeline(tmp_path, _SeqJudge(0.9, 0.1))
        result = pipe.run_iteration(
            seed_data=[make_item(prompt="q", response="a")],
            prompt_completions=[{"prompt": "q", "completions": ["good one", "bad one"]}],
        )

        assert result["num_synthetic"] == 1
        assert result["num_preferences"] == 1
        # SFT dataset loadable and carries synthesis metadata
        sft = pipe.registry.load("sft_synthetic", result["sft_dataset_version"])
        assert sft[0]["prompt"] == "evolved prompt"
        assert sft[0]["source"] == "synthetic_evol_instruct"
        # DPO dataset holds the judged preference
        dpo = pipe.registry.load("dpo_preferences", result["dpo_dataset_version"])
        assert dpo[0]["chosen"] == "good one"
        assert dpo[0]["rejected"] == "bad one"

    def test_dpo_lineage_parents_sft_version(self, tmp_path: Any) -> None:
        pipe = self._pipeline(tmp_path, _SeqJudge(0.9, 0.1))
        result = pipe.run_iteration(
            seed_data=[make_item()],
            prompt_completions=[{"prompt": "q", "completions": ["a", "b"]}],
        )
        lineage = pipe.registry.get_lineage("dpo_preferences", result["dpo_dataset_version"])
        assert lineage.parent_lineage_ids == [result["sft_dataset_version"]]
        assert lineage.operation == "preference_generation"

    def test_provided_rewards_skip_judge(self, tmp_path: Any) -> None:
        class _ExplodingJudge(JudgeClient):
            def judge(self, prompt: str, completion: str, **_: Any) -> float:
                raise AssertionError("judge must not be called when rewards given")

        pipe = self._pipeline(tmp_path, _ExplodingJudge())
        result = pipe.run_iteration(
            seed_data=[make_item()],
            prompt_completions=[{"prompt": "q", "completions": ["x", "y"], "rewards": [0.2, 0.8]}],
        )
        assert result["num_preferences"] == 1

    def test_seed_hash_is_deterministic(self) -> None:
        seeds_a = [make_item(prompt="q"), make_item(id="s2", prompt="r")]
        seeds_b = [make_item(prompt="q"), make_item(id="s2", prompt="r")]
        assert DataFlywheelPipeline._hash_seed(seeds_a) == DataFlywheelPipeline._hash_seed(seeds_b)
        assert DataFlywheelPipeline._hash_seed(seeds_a) != DataFlywheelPipeline._hash_seed(
            [make_item(prompt="different")]
        )


class TestRegistryBlankLines:
    def test_blank_jsonl_lines_skipped_on_load(self, tmp_path: Any) -> None:
        reg = LocalDatasetRegistry(str(tmp_path / "reg"))
        version = reg.register("ds", [make_item()], make_lineage())

        data_file = next((tmp_path / "reg" / "ds").glob("*.jsonl"))
        with open(data_file, "a", encoding="utf-8") as f:
            f.write("\n   \n")

        records = reg.load("ds", version)

        assert [r["prompt"] for r in records] == ["q"]


class TestPreferenceBuilderDedupOff:
    def test_dedup_false_keeps_identical_pairs(self) -> None:
        builder = PreferenceBuilder(top_k=2, dedup=False)

        pairs = builder.build("p", ["a", "a", "b", "b"], [1.0, 0.9, 0.1, 0.0])

        # Without dedup the second (identical) pair is kept instead of skipped.
        assert [(p.chosen, p.rejected) for p in pairs] == [("a", "b"), ("a", "b")]

    def test_dedup_true_skips_second_identical_pair(self) -> None:
        builder = PreferenceBuilder(top_k=2, dedup=True)

        pairs = builder.build("p", ["a", "a", "b", "b"], [1.0, 0.9, 0.1, 0.0])

        assert [(p.chosen, p.rejected) for p in pairs] == [("a", "b")]
