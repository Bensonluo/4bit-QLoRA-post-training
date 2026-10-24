"""Causal labels preserve real EOS even when its ID is also used for padding."""

from dataclasses import dataclass
from numbers import Integral
from typing import Any

from src.data.sft_protocol import ANSWER_PROTOCOL, LEGACY_PROTOCOL, validate_protocol


@dataclass
class AttentionMaskCausalCollator:
    tokenizer: Any
    pad_to_multiple_of: int = 8
    sft_protocol: str = LEGACY_PROTOCOL

    def __post_init__(self) -> None:
        validate_protocol(self.sft_protocol)

    def _answer_labels(self, features: list[dict[str, Any]]) -> list[list[int]]:
        """Validate supplied masks before padding can coerce invalid label types."""
        eos = getattr(self.tokenizer, "eos_token_id", None)
        if type(eos) is not int or eos < 0:
            raise ValueError("Answer-only SFT requires a valid tokenizer eos_token_id.")
        if not features:
            raise ValueError("Answer-only SFT requires a nonempty batch.")
        results = []
        special_ids = set(getattr(self.tokenizer, "all_special_ids", None) or [])
        special_ids.add(eos)
        for feature in features:
            ids, mask, labels = (
                feature.get("input_ids"),
                feature.get("attention_mask"),
                feature.get("labels"),
            )
            if (
                not isinstance(ids, (list, tuple))
                or not isinstance(mask, (list, tuple))
                or not isinstance(labels, (list, tuple))
            ):
                raise ValueError(
                    "Answer-only SFT requires aligned input_ids, attention_mask and labels."
                )
            if not ids or len(ids) != len(mask) or len(ids) != len(labels):
                raise ValueError("Answer-only SFT requires aligned nonempty labels.")
            if any(
                not isinstance(value, Integral) or isinstance(value, bool) or value < 0
                for value in ids
            ):
                raise ValueError("Answer-only input_ids must be integers.")
            if any(not isinstance(value, Integral) or isinstance(value, bool) for value in labels):
                raise ValueError("Answer-only labels must be integers.")
            if any(value not in (0, 1) for value in mask):
                raise ValueError("Answer-only attention_mask must contain only 0 or 1.")
            active = [index for index, value in enumerate(mask) if value]
            if not active or active != list(range(active[0], active[-1] + 1)):
                raise ValueError("Answer-only SFT requires contiguous active tokens.")
            supervised = []
            for index, (token, attention, label) in enumerate(zip(ids, mask, labels)):
                if label == -100:
                    continue
                if not attention or label != token:
                    raise ValueError(
                        "Answer-only labels must equal input_ids; padding must be -100."
                    )
                supervised.append(index)
            if len(supervised) < 2:
                raise ValueError("Answer-only SFT requires a nonempty causal answer and EOS.")
            first, last = supervised[0], supervised[-1]
            if first <= active[0] or supervised != list(range(first, last + 1)):
                raise ValueError(
                    "Answer-only labels require a masked prompt followed by the complete answer."
                )
            if ids[last] != eos or any(ids[index] in special_ids for index in supervised[:-1]):
                raise ValueError(
                    "Answer-only supervision must end with EOS after nonempty answer tokens."
                )
            if self.sft_protocol == ANSWER_PROTOCOL and last != active[-1]:
                raise ValueError("Alpaca answer-only requires supervised terminal EOS.")
            results.append([int(label) for label in labels])
        return results

    def __call__(self, features: list[dict[str, Any]]) -> Any:
        answer_labels = (
            self._answer_labels(features) if self.sft_protocol != LEGACY_PROTOCOL else None
        )
        # Existing labels can be stale or ragged; only the actual padded mask defines padding.
        inputs = [
            {key: value for key, value in feature.items() if key != "labels"}
            for feature in features
        ]
        if any("attention_mask" not in feature for feature in inputs):
            raise ValueError(
                "SFT records must provide attention_mask to distinguish EOS from padding."
            )
        batch = self.tokenizer.pad(
            inputs, padding=True, pad_to_multiple_of=self.pad_to_multiple_of, return_tensors="pt"
        )
        labels = batch["input_ids"].clone()
        if answer_labels is None:
            labels[batch["attention_mask"] == 0] = -100
        else:
            labels.fill_(-100)
            side = getattr(self.tokenizer, "padding_side", "right")
            if side not in {"left", "right"}:
                raise ValueError("Tokenizer padding_side must be left or right.")
            for index, values in enumerate(answer_labels):
                missing = labels.shape[1] - len(values)
                if missing < 0:
                    raise ValueError("Answer-only collation cannot truncate records.")
                start = missing if side == "left" else 0
                labels[index, start : start + len(values)] = labels.new_tensor(values)
            labels[batch["attention_mask"] == 0] = -100
        batch["labels"] = labels
        return batch
