"""LLM-as-a-Judge clients for scoring completions and preference pairs."""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import Any


class JudgeClient(ABC):
    """Abstract base class for judge clients."""

    @abstractmethod
    def judge(self, prompt: str, completion: str, **kwargs: Any) -> float:
        """Score a single completion in [0.0, 1.0]."""
        raise NotImplementedError

    def judge_pair(
        self,
        prompt: str,
        chosen: str,
        rejected: str,
        **kwargs: Any,
    ) -> tuple[float, float]:
        """Score a preference pair.

        Default implementation scores each completion independently.
        """
        return (
            self.judge(prompt, chosen, **kwargs),
            self.judge(prompt, rejected, **kwargs),
        )


class LocalJudgeClient(JudgeClient):
    """Judge using a local HuggingFace model.

    This is a minimal implementation. A production version would use a dedicated
    reward model or an instruct model with a structured scoring prompt.
    """

    DEFAULT_PROMPT_TEMPLATE = """You are an expert evaluator. Score the following response on a scale of 0 to 10, where 10 is excellent.

Prompt: {prompt}
Response: {completion}

Score (0-10):"""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
        prompt_template: str | None = None,
        device: str = "auto",
    ) -> None:
        """Initialize local judge client."""
        self.model_name = model_name
        self.prompt_template = prompt_template or self.DEFAULT_PROMPT_TEMPLATE
        self.device = device
        self._model: Any = None
        self._tokenizer: Any = None

    def _load(self) -> None:
        """Lazy-load model and tokenizer."""
        if self._model is not None:
            return

        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map=self.device,
        )

    def judge(self, prompt: str, completion: str, **_: Any) -> float:
        """Score a completion using the local model.

        Returns a score in [0.0, 1.0].
        """
        self._load()

        text = self.prompt_template.format(prompt=prompt, completion=completion)
        inputs = self._tokenizer(text, return_tensors="pt")
        if self._model.device.type != "cpu":
            inputs = {k: v.to(self._model.device) for k, v in inputs.items()}

        output = self._model.generate(**inputs, max_new_tokens=5, do_sample=False)
        decoded = self._tokenizer.decode(output[0], skip_special_tokens=True)

        # Extract trailing number
        match = re.search(r"(\d+(?:\.\d+)?)", decoded.replace(text, "").strip())
        if match:
            score = float(match.group(1)) / 10.0
            return max(0.0, min(1.0, score))
        return 0.5


class RuleJudgeClient(JudgeClient):
    """Simple rule-based judge for testing and deterministic rewards."""

    def __init__(
        self,
        answer_key: str = "answer",
        case_sensitive: bool = False,
    ) -> None:
        """Initialize rule-based judge."""
        self.answer_key = answer_key
        self.case_sensitive = case_sensitive

    def judge(self, prompt: str, completion: str, **kwargs: Any) -> float:
        """Score by exact match to reference answer."""
        answer = kwargs.get(self.answer_key)
        if answer is None:
            return 0.5

        pred = completion.strip() if self.case_sensitive else completion.strip().lower()
        target = answer.strip() if self.case_sensitive else answer.strip().lower()
        return 1.0 if pred == target else 0.0
