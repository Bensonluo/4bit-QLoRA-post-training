#!/usr/bin/env python3
"""快速测试 MLX LoRA adapter 效果（不干扰训练进程）。"""

import json
import time
from pathlib import Path

from mlx_lm import generate, load
from mlx_lm.sample_utils import make_sampler

DOMAIN_ROOT = Path(__file__).resolve().parent.parent
ADAPTER_PATH = str(DOMAIN_ROOT / "outputs" / "adapters-gemma-26b")
MODEL_ID = "mlx-community/gemma-4-26b-a4b-it-4bit"


def main() -> None:
    print(f"[加载模型] {MODEL_ID}")
    print(f"[加载adapter] {ADAPTER_PATH}")
    model, tokenizer = load(MODEL_ID, adapter_path=ADAPTER_PATH)

    # 读测试数据
    inst_file = DOMAIN_ROOT / "data" / "test" / "eval_institution.json"
    prod_file = DOMAIN_ROOT / "data" / "test" / "eval_product.json"

    with open(inst_file) as f:
        inst_data = json.load(f)
    with open(prod_file) as f:
        prod_data = json.load(f)

    # 各取第一条
    inst_sample = inst_data[0] if inst_data else None
    prod_sample = prod_data[0] if prod_data else None

    for name, sample in [("机构", inst_sample), ("产品", prod_sample)]:
        if sample is None:
            continue
        messages = sample["messages"]
        prompt = tokenizer.apply_chat_template(messages[:2], tokenize=False, add_generation_prompt=True)

        print(f"\n{'='*60}")
        print(f"【{name}匹配】输入：{messages[1]['content'][:100]}...")
        print(f"{'='*60}")

        t0 = time.time()
        sampler = make_sampler(temp=0.1)
        response = generate(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=1024,
            sampler=sampler,
            verbose=False,
        )
        latency = time.time() - t0

        print(f"[输出] ({latency:.1f}s):")
        print(response)
        print("\n[期望]:")
        print(messages[2]["content"])


if __name__ == "__main__":
    main()
