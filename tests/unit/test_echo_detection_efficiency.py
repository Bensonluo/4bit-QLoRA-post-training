"""Echo detection preserves the exact 12-character rule on long inputs."""

import random

import pytest

from src.workbench.evaluation_diagnostics import (
    ECHO_MIN_OUTPUT_CHARS,
    ECHO_MIN_SHARED_CHARS,
    output_echoes_prompt,
)


def reference(output, prompt):
    if not isinstance(output, str) or not isinstance(prompt, str):
        return False
    if len(output) < ECHO_MIN_OUTPUT_CHARS:
        return False
    return any(
        output[i : i + ECHO_MIN_SHARED_CHARS] in prompt
        for i in range(len(output) - ECHO_MIN_SHARED_CHARS + 1)
    )


@pytest.mark.parametrize(
    "output,prompt",
    [
        (None, "abcdef"),
        ("abcdefghijk", "abcdefghijk"),
        ("abcdefghijkX", "abcdefghijkY"),
        ("abcdefghijkL", "abcdefghijkL"),
        ("一个中文字符🧪序列用于边界检查", "其他前文一个中文字符🧪序列用于边界检查尾"),
        ("\u0065\u0301" * 8, "é" * 8),
    ],
)
def test_exact_threshold_and_unicode(output, prompt):
    assert output_echoes_prompt(output, prompt) == reference(output, prompt)


def test_randomized_matches_simple_exact_reference():
    randomizer = random.Random(20261028)
    alphabet = "abcd甲乙🧪\n"
    for _ in range(300):
        output = "".join(randomizer.choices(alphabet, k=randomizer.randrange(5, 75)))
        prompt = "".join(randomizer.choices(alphabet, k=randomizer.randrange(5, 75)))
        if len(output) >= ECHO_MIN_SHARED_CHARS and randomizer.randrange(2):
            start = randomizer.randrange(len(output) - ECHO_MIN_SHARED_CHARS + 1)
            prompt += output[start : start + ECHO_MIN_SHARED_CHARS]
        assert output_echoes_prompt(output, prompt) == reference(output, prompt)


def test_long_inputs_check_entire_output_including_suffix():
    shared = "唯一共享的完整连续片段12345"
    assert output_echoes_prompt("甲" * 20_000 + shared, "乙" * 20_000 + shared)
    assert not output_echoes_prompt("甲" * 20_000, "乙" * 20_000)
