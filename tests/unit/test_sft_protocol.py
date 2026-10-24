"""Strict single-turn contracts use full-sequence token proofs, never offsets."""

import importlib.util
from copy import deepcopy
from pathlib import Path

import pytest

from src.data.sft_protocol import (
    ANSWER_PROTOCOL,
    LEGACY_PROTOCOL,
    NATIVE_PROTOCOL,
    protocol_identity,
    render_sft_prompt,
    tokenize_sft_record,
    validate_protocol,
)

ROW = {"instruction": "classify", "input": "question", "output": "answer"}


class SlowTokenizer:
    """Character tokenizer with native controls and no offset/mask support."""

    eos_token, eos_token_id = "<E>", 1
    pad_token, pad_token_id = "<P>", 2
    bos_token, bos_token_id = "<B>", 3
    unk_token, unk_token_id = "<UNK>", 4
    all_special_tokens = ["<E>", "<P>", "<B>", "<UNK>", "<U>", "<A>"]
    all_special_ids = [1, 2, 3, 4, 5, 6]
    chat_template = "native-v1"
    padding_side = "right"
    truncation_side = "right"
    is_fast = False

    def __init__(self, *, transform=None, merge=None, auto_eos=False):
        self.transform = transform
        self.merge = merge
        self.auto_eos = auto_eos
        self.calls = []

    def get_chat_template(self):
        return self.chat_template

    def apply_chat_template(
        self, messages, *, chat_template, tokenize, add_generation_prompt, enable_thinking
    ):
        assert chat_template == self.chat_template
        assert not tokenize and enable_thinking is False
        prompt = "<U>" + messages[0]["content"] + "<E>\n<A>:"
        if len(messages) == 1:
            assert add_generation_prompt
            return prompt
        assert not add_generation_prompt
        answer = messages[1]["content"]
        if self.transform and answer:
            answer = self.transform(answer)
        return prompt + answer + "<E>\n"

    def __call__(
        self,
        text,
        *,
        add_special_tokens=True,
        truncation=False,
        padding=False,
        return_attention_mask=True,
    ):
        assert truncation is False and padding is False and return_attention_mask
        self.calls.append((text, add_special_tokens))
        ids = []
        while text:
            control = next((s for s in self.all_special_tokens if text.startswith(s)), None)
            if control:
                ids.append(self.all_special_ids[self.all_special_tokens.index(control)])
                text = text[len(control) :]
            elif self.merge and text.startswith(self.merge):
                ids.append(9999)
                text = text[len(self.merge) :]
            else:
                ids.append(4 if text[0] == "?" else ord(text[0]) + 100)
                text = text[1:]
        if add_special_tokens:
            ids.insert(0, self.bos_token_id)
            if self.auto_eos:
                ids.append(self.eos_token_id)
        return {"input_ids": ids, "attention_mask": [1] * len(ids)}


def supervised(record):
    return [token for token in record["labels"] if token != -100]


@pytest.mark.parametrize("protocol", [LEGACY_PROTOCOL, ANSWER_PROTOCOL, NATIVE_PROTOCOL])
def test_validate_protocol_preserves_exact_explicit_version(protocol):
    assert validate_protocol(protocol) == protocol


@pytest.mark.parametrize("value", [None, "", "native", "alpaca_answer_only_v2", [], 1])
def test_unknown_protocol_never_silently_falls_back(value):
    with pytest.raises(ValueError, match="Unknown SFT protocol"):
        validate_protocol(value)


def test_legacy_prompt_and_identity_allow_no_tokenizer_or_output():
    row = {"instruction": "classify", "input": "question"}
    expected = "### Instruction:\nclassify\n\n### Input:\nquestion\n\n### Response:\n"
    assert render_sft_prompt(row, None) == expected
    assert render_sft_prompt(row, None, ANSWER_PROTOCOL) == expected
    assert protocol_identity(None)["protocol"] == LEGACY_PROTOCOL
    assert protocol_identity(None)["eos_token_id"] is None


def test_legacy_delegates_and_keeps_original_truncation(monkeypatch):
    calls = []

    def legacy(row, tokenizer, max_length):
        calls.append((row, tokenizer, max_length))
        return {"input_ids": [10, 1, 1], "attention_mask": [1, 1, 0]}

    monkeypatch.setattr("src.data.loaders.tokenize_alpaca_record", legacy)
    tok = SlowTokenizer()
    result = tokenize_sft_record(ROW, tok, 3)
    assert calls == [(ROW, tok, 3)]
    assert result == {"input_ids": [10, 1, 1], "attention_mask": [1, 1, 0], "labels": [10, 1, -100]}


@pytest.mark.parametrize("protocol", [ANSWER_PROTOCOL, NATIVE_PROTOCOL])
@pytest.mark.parametrize("side", ["left", "right"])
@pytest.mark.parametrize("same_pad", [True, False])
def test_answer_and_original_eos_only_with_left_right_padding(protocol, side, same_pad):
    tok = SlowTokenizer()
    tok.padding_side = side
    if same_pad:
        tok.pad_token, tok.pad_token_id = tok.eos_token, tok.eos_token_id
    before = deepcopy(ROW)
    full = tokenize_sft_record(ROW, tok, protocol=protocol)
    padded = tokenize_sft_record(ROW, tok, len(full["input_ids"]) + 3, protocol=protocol)
    assert before == ROW
    answer_ids = [ord(char) + 100 for char in ROW["output"]]
    assert supervised(full) == supervised(padded) == answer_ids + [tok.eos_token_id]
    assert padded["attention_mask"].count(0) == 3
    pad_slice = slice(0, 3) if side == "left" else slice(-3, None)
    assert padded["input_ids"][pad_slice] == [tok.pad_token_id] * 3
    assert padded["labels"][pad_slice] == [-100] * 3
    prompt = render_sft_prompt(ROW, tok, protocol)
    prefix = tok(prompt, add_special_tokens=protocol != NATIVE_PROTOCOL)["input_ids"]
    assert full["input_ids"][: len(prefix)] == prefix
    assert full["labels"][: len(prefix)] == [-100] * len(prefix)


def test_native_preserves_post_eos_newline_without_supervising_or_adding_eos():
    tok = SlowTokenizer(auto_eos=True)
    result = tokenize_sft_record(ROW, tok, protocol=NATIVE_PROTOCOL)
    assert result["input_ids"][-2:] == [tok.eos_token_id, ord("\n") + 100]
    assert result["labels"][-2:] == [tok.eos_token_id, -100]
    assert tok.bos_token_id not in result["input_ids"]
    assert all(not add_special_tokens for _, add_special_tokens in tok.calls)
    assert tok.is_fast is False


@pytest.mark.parametrize("protocol", [ANSWER_PROTOCOL, NATIVE_PROTOCOL])
def test_overlength_refuses_to_remove_prompt_answer_or_eos(protocol):
    tok = SlowTokenizer()
    full = tokenize_sft_record(ROW, tok, protocol=protocol)
    for side in ("left", "right"):
        tok.truncation_side = side
        with pytest.raises(ValueError, match="exceeds max_length"):
            tokenize_sft_record(ROW, tok, len(full["input_ids"]) - 1, protocol=protocol)


@pytest.mark.parametrize("length", [0, -1, True, 2.5])
def test_invalid_max_length_rejected(length):
    with pytest.raises(ValueError, match="max_length"):
        tokenize_sft_record(ROW, SlowTokenizer(), length, protocol=NATIVE_PROTOCOL)


@pytest.mark.parametrize("answer", [None, "", " \n ", 42])
@pytest.mark.parametrize("protocol", [ANSWER_PROTOCOL, NATIVE_PROTOCOL])
def test_empty_or_nonstring_answer_is_not_an_eos_only_example(answer, protocol):
    with pytest.raises(ValueError, match="nonempty"):
        tokenize_sft_record(ROW | {"output": answer}, SlowTokenizer(), protocol=protocol)


@pytest.mark.parametrize("answer", ["<E>", "a<P>b", "<U>", "<A>", "<UNK>"])
@pytest.mark.parametrize("protocol", [ANSWER_PROTOCOL, NATIVE_PROTOCOL])
def test_special_control_token_answers_are_rejected_before_encoding(answer, protocol):
    tok = SlowTokenizer()
    with pytest.raises(ValueError, match="special control token"):
        tokenize_sft_record(ROW | {"output": answer}, tok, protocol=protocol)
    assert not tok.calls


@pytest.mark.parametrize("protocol", [ANSWER_PROTOCOL, NATIVE_PROTOCOL])
def test_unknown_answer_token_fails_closed(protocol):
    with pytest.raises(ValueError, match="unknown token"):
        tokenize_sft_record(ROW | {"output": "?"}, SlowTokenizer(), protocol=protocol)


@pytest.mark.parametrize(
    "transform", [lambda a: "", str.upper, lambda a: a + a, lambda a: a.strip()]
)
def test_native_rejects_template_drop_replace_duplicate_or_trim(transform):
    tok = SlowTokenizer(transform=transform)
    with pytest.raises(ValueError, match="changes, drops or duplicates"):
        tokenize_sft_record(ROW | {"output": " answer "}, tok, protocol=NATIVE_PROTOCOL)


def test_native_rejects_answer_dependent_header():
    class HeaderTokenizer(SlowTokenizer):
        def apply_chat_template(self, messages, **kwargs):
            text = super().apply_chat_template(messages, **kwargs)
            if len(messages) == 2 and messages[-1]["content"]:
                text = "unexpected header" + text
            return text

    with pytest.raises(ValueError, match="changes, drops or duplicates"):
        tokenize_sft_record(ROW, HeaderTokenizer(), protocol=NATIVE_PROTOCOL)


def test_text_prefix_is_insufficient_when_bpe_merges_prompt_and_answer():
    tok = SlowTokenizer(merge=":a")
    prompt = render_sft_prompt(ROW, tok, NATIVE_PROTOCOL)
    assert (prompt + ROW["output"]).startswith(prompt)
    with pytest.raises(ValueError, match="exact training token prefix"):
        tokenize_sft_record(ROW, tok, protocol=NATIVE_PROTOCOL)


def test_alpaca_bpe_boundary_and_automatic_prompt_eos_are_rejected():
    tok = SlowTokenizer(merge="\na")
    with pytest.raises(ValueError, match="exact training token prefix"):
        tokenize_sft_record(ROW, tok, protocol=ANSWER_PROTOCOL)
    with pytest.raises(ValueError, match="exact training token prefix"):
        tokenize_sft_record(ROW, SlowTokenizer(auto_eos=True), protocol=ANSWER_PROTOCOL)


def test_native_answer_closure_boundary_must_be_exact():
    class BoundaryTokenizer(SlowTokenizer):
        def __call__(self, text, **kwargs):
            encoded = super().__call__(text, **kwargs)
            if text.endswith("answer<E>\n"):
                encoded["input_ids"][-3] = 9999
            return encoded

    with pytest.raises(ValueError, match="exact training token prefix"):
        tokenize_sft_record(ROW, BoundaryTokenizer(), protocol=NATIVE_PROTOCOL)


@pytest.mark.parametrize("suffix", ["", "\n<E>", "<E>nonwhitespace", "<E><E>\n"])
def test_native_unsupported_closure_is_rejected(suffix):
    class ClosureTokenizer(SlowTokenizer):
        def apply_chat_template(self, messages, **kwargs):
            text = super().apply_chat_template(messages, **kwargs)
            return text[:-4] + suffix if len(messages) == 2 else text

    with pytest.raises(ValueError, match="closure"):
        tokenize_sft_record(ROW, ClosureTokenizer(), protocol=NATIVE_PROTOCOL)


def test_native_requires_template_and_valid_eos():
    with pytest.raises(ValueError, match="chat template"):
        render_sft_prompt(ROW, None, NATIVE_PROTOCOL)
    tok = SlowTokenizer()
    tok.chat_template = None
    with pytest.raises(ValueError, match="chat template"):
        tokenize_sft_record(ROW, tok, protocol=NATIVE_PROTOCOL)
    tok.chat_template = "native-v1"
    tok.eos_token_id = None
    with pytest.raises(ValueError, match="eos_token_id"):
        tokenize_sft_record(ROW, tok, protocol=NATIVE_PROTOCOL)


def test_native_prompt_does_not_use_reference_answer_and_user_mapping_is_stable():
    tok = SlowTokenizer()
    prompt = render_sft_prompt(ROW | {"output": object()}, tok, NATIVE_PROTOCOL)
    assert "classify\n\nquestion" in prompt
    assert prompt == render_sft_prompt(ROW, tok, NATIVE_PROTOCOL)
    assert "\n\n" not in render_sft_prompt(ROW | {"input": ""}, tok, NATIVE_PROTOCOL)
    with pytest.raises(ValueError, match="must be strings"):
        render_sft_prompt(ROW | {"input": None}, tok, NATIVE_PROTOCOL)


def test_protocol_identity_captures_template_and_fixed_options():
    tok = SlowTokenizer()
    native = protocol_identity(tok, NATIVE_PROTOCOL)
    assert native["enable_thinking"] is False
    assert native["add_special_tokens"] is False
    assert native["supervision_strategy"] == "answer_and_eos_only"
    assert len(native["chat_template_sha256"]) == 64
    tok.chat_template += " changed"
    assert (
        protocol_identity(tok, NATIVE_PROTOCOL)["chat_template_sha256"]
        != native["chat_template_sha256"]
    )
    legacy = protocol_identity(tok)
    answer = protocol_identity(tok, ANSWER_PROTOCOL)
    assert legacy["prompt_renderer"] == answer["prompt_renderer"]
    assert legacy["add_special_tokens"] is True
    assert legacy["chat_template_sha256"] is None
    assert legacy["supervision_strategy"] != answer["supervision_strategy"]
    assert legacy["prompt_identity"] == answer["prompt_identity"]
    assert "protocol" not in native["prompt_identity"]
    assert "supervision_strategy" not in native["prompt_identity"]


def test_protocol_module_can_load_without_model_or_dataset_imports(monkeypatch):
    import builtins

    original = builtins.__import__

    def checked_import(name, *args, **kwargs):
        assert name.split(".")[0] not in {"torch", "transformers", "datasets"}
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", checked_import)
    path = Path(__file__).parents[2] / "src/data/sft_protocol.py"
    spec = importlib.util.spec_from_file_location("standalone_sft_protocol", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.protocol_identity(None)["protocol"] == LEGACY_PROTOCOL
