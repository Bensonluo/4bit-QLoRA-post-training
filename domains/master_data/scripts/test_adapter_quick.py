#!/usr/bin/env python3
"""快速测试 MLX LoRA adapter 效果（不干扰训练进程）。"""

import json
import time
from pathlib import Path

from mlx_lm import load, generate

DOMAIN_ROOT = Path(__file__).resolve().parent.parent
ADAPTER_PATH = str(DOMAIN_ROOT / "outputs" / "adapters-gemma-26b")
MODEL_ID = "mlx-community/gemma-4-26b-a4b-it-4bit"


def main() -> None:
    print(f"[加载模型] {MODEL_ID}")
    print(f"[加载adapter] {ADAPTER_PATH}")
    model, tokenizer = load(MODEL_ID, adapter_path=ADAPTER_PATH)

    # 读测试数据
    test_file = DOMAIN_ROOT / "data" / "test.jsonl"
    with open(test_file) as f:
        test_data = [json.loads(line) for line in f]

    # 找1条机构和1条产品
    inst_sample = None
    prod_sample = None
    for d in test_data:
        if "institution" in d.get("task", "") and inst_sample is None:
            inst_sample = d
        elif "product" in d.get("task", "") and prod_sample is None:
            prod_sample = d
        if inst_sample and prod_sample:
            break

    for name, sample in [("机构", inst_sample), ("产品", prod_sample)]:
        if sample is None:
            continue
        messages = sample["messages"]
        prompt = tokenizer.apply_chat_template(messages[:2], tokenize=False, add_generation_prompt=True)

        print(f"\n{'='*60}")
        print(f"【{name}匹配】输入：{messages[1]['content'][:100]}...")
        print(f"{'='*60}")

        t0 = time.time()
        response = generate(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=1024,
            temp=0.1,
            verbose=False,
        )
        latency = time.time() - t0

        print(f"[输出] ({latency:.1f}s):")
        print(response)
        print(f"\n[期望]:")
        print(messages[2]["content"])


if __name__ == "__main__":
    main()
