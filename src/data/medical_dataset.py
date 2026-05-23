"""医疗实体匹配数据加载器。"""

import json
from pathlib import Path
from typing import Optional

from datasets import Dataset
from transformers import PreTrainedTokenizer

from src.data.base import BaseDataset
from src.utils.logging import console


class MedicalEntityDataset(BaseDataset):
    """医疗实体匹配数据集 (instruction 格式)。

    数据格式 (alpaca):
    {
        "instruction": "从候选列表中选出匹配的标准名称...",
        "input": "输入实体: xxx\\n候选:\\n1. ...",
        "output": "{\\"standard_name\\": \\"xxx\\", ...}",
        "metadata": {"entity_type": "drug", "difficulty": "hard"}
    }
    """

    def __init__(
        self,
        data_path: str,
        max_samples: Optional[int] = None,
        difficulty_filter: Optional[str] = None,
    ):
        self.difficulty_filter = difficulty_filter
        super().__init__(data_path, max_samples)

    def load(self) -> Dataset:
        data_path = Path(self.data_path)
        if not data_path.exists():
            raise FileNotFoundError(f"数据文件不存在: {self.data_path}")

        with open(data_path) as f:
            data = json.load(f)

        if self.difficulty_filter:
            data = [
                d for d in data
                if d.get("metadata", {}).get("difficulty") == self.difficulty_filter
            ]

        if self.max_samples:
            data = data[: self.max_samples]

        self.dataset = Dataset.from_list(data)
        console.print(f"[green]✓ 加载医疗实体数据: {len(self.dataset)} 条 ({data_path.name})[/green]")
        return self.dataset

    def format_for_training(
        self,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 1024,
    ) -> Dataset:
        def tokenize_fn(examples):
            prompts = []
            for inst, inp, out in zip(
                examples["instruction"], examples["input"], examples["output"]
            ):
                prompt = f"### Instruction:\n{inst}\n\n### Input:\n{inp}\n\n### Response:\n{out}"
                prompts.append(prompt)

            return tokenizer(
                prompts,
                truncation=True,
                max_length=max_length,
                padding="max_length",
            )

        self.dataset = self.dataset.map(
            tokenize_fn,
            batched=True,
            remove_columns=self.dataset.column_names,
        )
        return self.dataset
