#!/usr/bin/env python3
"""Convert train.json to MLX LoRA format (train.jsonl + valid.jsonl)."""

import json
import random
from pathlib import Path

random.seed(42)

DOMAIN_ROOT = Path(__file__).resolve().parent.parent

def main():
    with open(DOMAIN_ROOT / "data" / "train" / "train.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    random.shuffle(data)

    # 90/10 split
    split = int(len(data) * 0.9)
    train_data = data[:split]
    valid_data = data[split:]

    mlx_dir = DOMAIN_ROOT / "data" / "mlx_train"
    mlx_dir.mkdir(parents=True, exist_ok=True)

    with open(mlx_dir / "train.jsonl", "w", encoding="utf-8") as f:
        for item in train_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    with open(mlx_dir / "valid.jsonl", "w", encoding="utf-8") as f:
        for item in valid_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"MLX data converted: {len(train_data)} train + {len(valid_data)} valid")

if __name__ == "__main__":
    main()
