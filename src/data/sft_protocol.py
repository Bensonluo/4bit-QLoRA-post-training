"""Versioned, fail-closed single-turn SFT prompt and supervision contracts.

This module deliberately imports only the standard library. Native v1 supports
templates whose final assistant content is preserved verbatim and closed by the
tokenizer's EOS followed by optional whitespace. Other templates require an
explicit protocol extension, never an automatic fallback to Alpaca.
"""

import hashlib
from typing import Any

LEGACY_PROTOCOL = "alpaca_full_prompt_v1"
ANSWER_PROTOCOL = "alpaca_answer_only_v1"
NATIVE_PROTOCOL = "native_chat_answer_only_v1"
SFT_PROTOCOLS = (LEGACY_PROTOCOL, ANSWER_PROTOCOL, NATIVE_PROTOCOL)


def validate_protocol(value: Any) -> str:
    """Validate an explicit versioned protocol; legacy defaults belong to callers."""
    if not isinstance(value, str) or value not in SFT_PROTOCOLS:
        raise ValueError(f"Unknown SFT protocol {value!r}; choose one of {SFT_PROTOCOLS}.")
    return value


def _alpaca_prompt(row: dict[str, Any]) -> str:
    instruction = row["instruction"]
    input_text = row.get("input", "")
    if input_text:
        return f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n"
    return f"### Instruction:\n{instruction}\n\n### Response:\n"


def _user_content(row: dict[str, Any]) -> str:
    instruction, input_text = row.get("instruction"), row.get("input", "")
    if not isinstance(instruction, str) or not isinstance(input_text, str):
        raise ValueError("SFT instruction and input must be strings.")
    return instruction + ("\n\n" + input_text if input_text else "")


def _chat_template(tokenizer: Any) -> str:
    if tokenizer is None:
        raise ValueError("Native SFT requires a tokenizer with a chat template.")
    getter = getattr(tokenizer, "get_chat_template", None)
    template = getter() if callable(getter) else getattr(tokenizer, "chat_template", None)
    if not isinstance(template, str) or not template.strip():
        raise ValueError("Native SFT requires an unambiguous nonempty chat template.")
    return template


def _render_chat(tokenizer: Any, user: str, answer: str | None) -> str:
    template = _chat_template(tokenizer)
    messages = [{"role": "user", "content": user}]
    if answer is not None:
        messages.append({"role": "assistant", "content": answer})
    rendered = tokenizer.apply_chat_template(
        messages,
        chat_template=template,
        tokenize=False,
        add_generation_prompt=answer is None,
        enable_thinking=False,
    )
    if not isinstance(rendered, str):
        raise ValueError("The chat template must render a single text string.")
    return rendered


def render_sft_prompt(row: dict[str, Any], tokenizer: Any, protocol: str = LEGACY_PROTOCOL) -> str:
    """Render only the generation prompt, independently of the reference answer."""
    protocol = validate_protocol(protocol)
    if protocol == NATIVE_PROTOCOL:
        return _render_chat(tokenizer, _user_content(row), None)
    return _alpaca_prompt(row)


def _token_id(tokenizer: Any, name: str) -> int:
    value = getattr(tokenizer, name, None)
    if type(value) is not int or value < 0:
        raise ValueError(f"SFT requires a valid tokenizer {name}.")
    return value


def _encode(tokenizer: Any, text: str, *, add_special_tokens: bool) -> list[int]:
    encoded = tokenizer(
        text,
        add_special_tokens=add_special_tokens,
        truncation=False,
        padding=False,
        return_attention_mask=True,
    )
    ids = list(encoded["input_ids"])
    attention = list(encoded.get("attention_mask", [1] * len(ids)))
    if any(type(token) is not int or token < 0 for token in ids):
        raise ValueError("Tokenizer must return one-dimensional nonnegative integer token IDs.")
    if len(attention) != len(ids) or any(type(mask) is not int or mask != 1 for mask in attention):
        raise ValueError("Unpadded SFT tokens must have aligned attention_mask values of one.")
    return ids


def _special_ids(tokenizer: Any) -> set[int]:
    ids = set(getattr(tokenizer, "all_special_ids", None) or [])
    for name in ("eos_token_id", "bos_token_id", "pad_token_id", "unk_token_id"):
        value = getattr(tokenizer, name, None)
        if type(value) is int:
            ids.add(value)
    return ids


def _answer(row: dict[str, Any], tokenizer: Any) -> str:
    answer = row.get("output")
    if not isinstance(answer, str) or not answer.strip():
        raise ValueError("Answer-only SFT requires a nonempty string answer.")
    controls = set(getattr(tokenizer, "all_special_tokens", None) or [])
    for name in ("eos_token", "bos_token", "pad_token", "unk_token"):
        token = getattr(tokenizer, name, None)
        if isinstance(token, str) and token:
            controls.add(token)
    if any(control and control in answer for control in controls):
        raise ValueError("The SFT answer contains a special control token.")
    return answer


def _require_prefix(prefix: list[int], full: list[int]) -> None:
    if not prefix or full[: len(prefix)] != prefix:
        raise ValueError("Generation/answer token boundary is not an exact training token prefix.")


def _pad(
    ids: list[int], labels: list[int], tokenizer: Any, max_length: int | None
) -> dict[str, list[int]]:
    result = {"input_ids": ids, "attention_mask": [1] * len(ids), "labels": labels}
    if max_length is None:
        return result
    if len(ids) > max_length:
        raise ValueError(
            "Answer-only SFT exceeds max_length; truncating the prompt, answer or EOS is forbidden."
        )
    missing = max_length - len(ids)
    if not missing:
        return result
    pad_id = _token_id(tokenizer, "pad_token_id")
    side = getattr(tokenizer, "padding_side", "right")
    if side not in {"left", "right"}:
        raise ValueError("Tokenizer padding_side must be left or right.")
    fill = {"input_ids": pad_id, "attention_mask": 0, "labels": -100}
    return {
        key: ([fill[key]] * missing + values if side == "left" else values + [fill[key]] * missing)
        for key, values in result.items()
    }


def tokenize_sft_record(
    row: dict[str, Any],
    tokenizer: Any,
    max_length: int | None = None,
    *,
    protocol: str = LEGACY_PROTOCOL,
) -> dict[str, list[int]]:
    """Encode one record, proving answer boundaries before creating loss labels.

    Native v1 keeps the complete templated input, including post-EOS whitespace,
    while supervising only the verbatim answer and its original EOS. It uses no
    character offsets or assistant masks, so slow tokenizers are supported.
    """
    protocol = validate_protocol(protocol)
    if max_length is not None and (type(max_length) is not int or max_length <= 0):
        raise ValueError("max_length must be a positive integer or None.")
    if protocol == LEGACY_PROTOCOL:
        from src.data.loaders import tokenize_alpaca_record

        result = tokenize_alpaca_record(row, tokenizer, max_length)
        result["labels"] = [
            token if mask else -100
            for token, mask in zip(result["input_ids"], result["attention_mask"])
        ]
        return result

    if tokenizer is None:
        raise ValueError("Answer-only SFT requires a tokenizer.")
    _user_content(row)
    answer = _answer(row, tokenizer)
    eos_id = _token_id(tokenizer, "eos_token_id")
    prompt = render_sft_prompt(row, tokenizer, protocol)
    add_special_tokens = protocol != NATIVE_PROTOCOL
    prompt_ids = _encode(tokenizer, prompt, add_special_tokens=add_special_tokens)
    content_ids = _encode(tokenizer, prompt + answer, add_special_tokens=add_special_tokens)

    if protocol == NATIVE_PROTOCOL:
        user = _user_content(row)
        empty = _render_chat(tokenizer, user, "")
        if not empty.startswith(prompt):
            raise ValueError("Native assistant header does not match the generation prompt.")
        suffix = empty[len(prompt) :]
        eos_token = getattr(tokenizer, "eos_token", None)
        if (
            not isinstance(eos_token, str)
            or not eos_token
            or not suffix.startswith(eos_token)
            or suffix[len(eos_token) :].strip()
        ):
            raise ValueError("Native closure must be tokenizer EOS followed only by whitespace.")
        full = _render_chat(tokenizer, user, answer)
        if full != prompt + answer + suffix:
            raise ValueError(
                "The native template changes, drops or duplicates the original answer."
            )
        ids = _encode(tokenizer, full, add_special_tokens=False)
        _require_prefix(content_ids, ids)
        answer_end = len(content_ids)
        if answer_end >= len(ids) or ids[answer_end] != eos_id:
            raise ValueError(
                "Native EOS must be an exact token boundary immediately after the answer."
            )
        # Prove all trailing input is the original EOS plus whitespace, without
        # relying on independently tokenizing the answer or decoder cleanup.
        closure_ids = _encode(tokenizer, suffix, add_special_tokens=False)
        if not closure_ids or closure_ids[0] != eos_id or ids[answer_end:] != closure_ids:
            raise ValueError("Native closure tokenization is not stable at the answer boundary.")
        if any(token in _special_ids(tokenizer) for token in closure_ids[1:]):
            raise ValueError("Native post-EOS whitespace contains another special control token.")
    else:
        # Preserve the legacy tokenizer's automatic special-token behavior. An
        # automatically inserted prompt EOS fails the prefix proof below.
        ids = content_ids.copy()
        if not ids or ids[-1] != eos_id:
            ids.append(eos_id)
        answer_end = len(ids) - 1

    _require_prefix(prompt_ids, ids)
    answer_start = len(prompt_ids)
    answer_ids = ids[answer_start:answer_end]
    if not answer_ids:
        raise ValueError("The answer has no independently supervised tokens.")
    if any(token in _special_ids(tokenizer) for token in answer_ids):
        raise ValueError("The encoded answer contains a special control or unknown token.")
    labels = [-100] * len(ids)
    labels[answer_start : answer_end + 1] = ids[answer_start : answer_end + 1]
    return _pad(ids, labels, tokenizer, max_length)


def protocol_identity(tokenizer: Any, protocol: str = LEGACY_PROTOCOL) -> dict[str, Any]:
    """Return JSON-safe protocol facts without importing model/data libraries."""
    protocol = validate_protocol(protocol)
    native = protocol == NATIVE_PROTOCOL
    template = _chat_template(tokenizer) if native else None
    prompt_identity = {
        "prompt_renderer": (
            "native_single_turn_chat_v1" if native else "render_alpaca_prompt_without_answer"
        ),
        "add_special_tokens": not native,
        "enable_thinking": False if native else None,
        "chat_template_sha256": (
            hashlib.sha256(template.encode("utf-8")).hexdigest() if template is not None else None
        ),
        "eos_token_id": getattr(tokenizer, "eos_token_id", None),
    }
    return {
        "protocol": protocol,
        "supervision_strategy": (
            "full_prompt_padding_masked" if protocol == LEGACY_PROTOCOL else "answer_and_eos_only"
        ),
        **prompt_identity,
        "prompt_identity": prompt_identity,
    }
