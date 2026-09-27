"""Causal labels preserve real EOS even when its ID is also used for padding."""

from dataclasses import dataclass
from typing import Any


@dataclass
class AttentionMaskCausalCollator:
    tokenizer: Any
    pad_to_multiple_of: int = 8

    def __call__(self, features):
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
        labels[batch["attention_mask"] == 0] = -100
        batch["labels"] = labels
        return batch
