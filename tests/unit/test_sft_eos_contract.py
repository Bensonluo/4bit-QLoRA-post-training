"""Complete SFT records teach EOS, including when pad_token_id equals eos_token_id."""

from copy import deepcopy

import pytest
from datasets import Dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from transformers import PreTrainedTokenizerFast

from src.data.loaders import AlpacaDataset, render_alpaca_prompt, tokenize_alpaca_record
from src.data.sft_collator import AttentionMaskCausalCollator

EXAMPLE = {"instruction": "classify", "input": "question", "output": "answer"}


def tokenizer(*, auto_eos=False, side="right", pad_equals_eos=True):
    vocab = {
        token: index
        for index, token in enumerate(
            [
                "[UNK]",
                "[BOS]",
                "[EOS]",
                "[PAD]",
                "###",
                "Instruction",
                ":",
                "Input",
                "Response",
                "classify",
                "question",
                "answer",
            ]
        )
    }
    raw = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    raw.pre_tokenizer = Whitespace()
    raw.post_processor = TemplateProcessing(
        single="[BOS] $A [EOS]" if auto_eos else "[BOS] $A",
        special_tokens=[("[BOS]", vocab["[BOS]"]), ("[EOS]", vocab["[EOS]"])],
    )
    return PreTrainedTokenizerFast(
        tokenizer_object=raw,
        unk_token="[UNK]",
        bos_token="[BOS]",
        eos_token="[EOS]",
        pad_token="[EOS]" if pad_equals_eos else "[PAD]",
        padding_side=side,
    )


@pytest.mark.parametrize("auto_eos", [False, True])
def test_complete_record_has_exactly_one_terminal_eos_and_aligned_offsets(auto_eos):
    tok = tokenizer(auto_eos=auto_eos)
    plain = tok(render_alpaca_prompt(EXAMPLE), return_offsets_mapping=True)
    encoded = tokenize_alpaca_record(EXAMPLE, tok, return_offsets_mapping=True)
    assert encoded["input_ids"][0] == tok.bos_token_id
    assert encoded["input_ids"][-1] == tok.eos_token_id
    assert encoded["input_ids"].count(tok.eos_token_id) == 1
    assert (
        len(encoded["input_ids"])
        == len(encoded["attention_mask"])
        == len(encoded["offset_mapping"])
    )
    assert encoded["offset_mapping"][-1] == (0, 0)
    assert encoded["input_ids"] == plain["input_ids"] + ([] if auto_eos else [tok.eos_token_id])
    assert all(encoded["attention_mask"])


@pytest.mark.parametrize("side", ["left", "right"])
@pytest.mark.parametrize("pad_equals_eos", [True, False])
def test_formatter_and_collator_mask_only_padding_preserving_actual_eos(side, pad_equals_eos):
    tok = tokenizer(side=side, pad_equals_eos=pad_equals_eos)
    full = tokenize_alpaca_record(EXAMPLE, tok)
    length = len(full["input_ids"]) + 3
    dataset = AlpacaDataset("fixture")
    dataset.dataset = Dataset.from_list([EXAMPLE])
    formatted = dict(dataset.format_for_training(tok, max_length=length)[0])
    # Stale labels, including a masked real EOS, must not survive or affect padding.
    formatted["labels"] = [-100]
    before = deepcopy(formatted)
    batch = AttentionMaskCausalCollator(tok)([formatted, full | {"labels": [999, 998]}])
    assert formatted == before
    assert batch["input_ids"].shape[1] % 8 == 0
    for ids, mask, labels in zip(batch["input_ids"], batch["attention_mask"], batch["labels"]):
        assert labels[mask == 0].tolist() == [-100] * int((mask == 0).sum())
        assert labels[mask == 1].tolist() == ids[mask == 1].tolist() == full["input_ids"]
        assert labels[mask == 1][-1].item() == tok.eos_token_id


@pytest.mark.parametrize("auto_eos", [False, True])
def test_right_truncation_does_not_replace_answer_with_false_ending(auto_eos):
    tok = tokenizer(auto_eos=auto_eos)
    full = tokenize_alpaca_record(EXAMPLE, tok, return_offsets_mapping=True)
    truncated = tokenize_alpaca_record(
        EXAMPLE, tok, max_length=len(full["input_ids"]) - 1, return_offsets_mapping=True
    )
    assert truncated["input_ids"] == full["input_ids"][:-1]
    assert truncated["input_ids"][-1] != tok.eos_token_id
    assert truncated["offset_mapping"] == full["offset_mapping"][:-1]
    assert all(truncated["attention_mask"])


def test_left_truncation_and_padding_preserve_token_offset_alignment():
    tok = tokenizer(side="left")
    tok.truncation_side = "left"
    full = tokenize_alpaca_record(EXAMPLE, tok, return_offsets_mapping=True)
    clipped = tokenize_alpaca_record(EXAMPLE, tok, max_length=3, return_offsets_mapping=True)
    assert clipped == {key: value[-3:] for key, value in full.items()}
    padded = tokenize_alpaca_record(
        EXAMPLE, tok, max_length=len(full["input_ids"]) + 2, return_offsets_mapping=True
    )
    assert padded["attention_mask"][:2] == [0, 0]
    assert padded["offset_mapping"][:2] == [(0, 0), (0, 0)]
    assert {key: value[2:] for key, value in padded.items()} == full


def test_missing_eos_and_missing_attention_mask_fail_explicitly():
    tok = tokenizer()
    tok.eos_token = None
    with pytest.raises(ValueError, match="eos_token_id"):
        tokenize_alpaca_record(EXAMPLE, tok)
    with pytest.raises(ValueError, match="attention_mask"):
        AttentionMaskCausalCollator(tok)([{"input_ids": [1, 2]}])
