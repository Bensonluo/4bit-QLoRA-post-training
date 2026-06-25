#!/usr/bin/env python3
"""精确分析数据质量 v2：正确区分 D 级中的 hard/easy。"""

import json
from collections import Counter
from pathlib import Path

DOMAIN_ROOT = Path(__file__).resolve().parent.parent


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def extract_query(messages):
    for m in messages:
        if m["role"] == "user":
            content = m["content"]
            if "】：" in content:
                return content.split("】：", 1)[1].strip().split("\n")[0]
    return None


def jaccard(a, b):
    sa, sb = set(a), set(b)
    inter = len(sa & sb)
    union = len(sa | sb)
    return inter / union if union > 0 else 0


def analyze_products(data, name):
    print(f"\n{'='*60}")
    print(f"=== 产品数据精确分析 {name} ===")
    print(f"{'='*60}")

    # D 级分层
    d_hard = 0   # Jaccard >= 0.15 或共同字符 >= 2
    d_medium = 0  # 有一些关联但不够强
    d_easy = 0   # 完全无关

    total_d = 0
    d_examples = []

    for _idx, item in enumerate(data):
        messages = item["messages"]
        query = extract_query(messages)
        if not query:
            continue

        # 提取 query 核心名（去掉规格）
        query_core = query.split()[0] if " " in query else query

        # 提取 candidates 和 labels
        for m in messages:
            if m["role"] == "assistant":
                try:
                    labels = json.loads(m["content"])
                except Exception:
                    continue

                # 从 user msg 提取 candidates
                for um in messages:
                    if um["role"] == "user":
                        lines = um["content"].split("\n")
                        candidates = []
                        for line in lines:
                            if line.startswith("[") and "名称:" in line:
                                name_part = line.split("名称:", 1)[1].split(", ")[0].strip()
                                candidates.append(name_part)
                        break

                for i, label in enumerate(labels):
                    if i >= len(candidates):
                        continue
                    if label.get("match_grade") == "D":
                        total_d += 1
                        cand_core = candidates[i].split()[0] if " " in candidates[i] else candidates[i]
                        sim = jaccard(query_core, cand_core)
                        common_chars = len(set(query_core) & set(cand_core))

                        if sim >= 0.15 or common_chars >= 2:
                            d_hard += 1
                        elif common_chars >= 1:
                            d_medium += 1
                        else:
                            d_easy += 1
                            if len(d_examples) < 10:
                                d_examples.append((query_core, cand_core, sim, common_chars))
                        break  # 只分析第一个 D 级，避免重复计数

    print(f"\n[D 级候选硬度分布] (总计 {total_d} 个)")
    print(f"  Hard (Jaccard>=0.15 或共同字符>=2): {d_hard} ({d_hard/total_d*100:.1f}%)")
    print(f"  Medium (共同字符=1): {d_medium} ({d_medium/total_d*100:.1f}%)")
    print(f"  Easy (完全无关): {d_easy} ({d_easy/total_d*100:.1f}%)")

    print("\nEasy D 级示例:")
    for qc, cc, sim, common in d_examples:
        print(f"  '{qc}' vs '{cc}' -> Jaccard={sim:.2f}, 共同字符={common}")


def analyze_institutions(data, name):
    print(f"\n{'='*60}")
    print(f"=== 机构数据精确分析 {name} ===")
    print(f"{'='*60}")

    queries = []
    for item in data:
        q = extract_query(item["messages"])
        if q:
            queries.append(q)

    query_counts = Counter(queries)
    duplicates = {q: c for q, c in query_counts.items() if c > 1}

    print(f"\n总样本: {len(queries)}")
    print(f"唯一 query: {len(query_counts)}")
    print(f"重复 query 数: {len(duplicates)}")
    print(f"重复 query 涉及样本: {sum(duplicates.values())}")
    print(f"重复率: {sum(duplicates.values())/len(queries)*100:.1f}%")

    if duplicates:
        print("\n重复次数 top 10:")
        for q, c in sorted(duplicates.items(), key=lambda x: -x[1])[:10]:
            print(f"  重复 {c} 次: {q[:60]}")


def main():
    train = load_json(DOMAIN_ROOT / "data" / "train" / "train.json")
    eval_i = load_json(DOMAIN_ROOT / "data" / "test" / "eval_institution.json")
    eval_p = load_json(DOMAIN_ROOT / "data" / "test" / "eval_product.json")

    # 产品分析
    analyze_products(train[:2000], "训练集 (前2000条)")
    analyze_products(eval_p, "评测集-产品")

    # 机构分析
    analyze_institutions(train, "训练集")
    analyze_institutions(eval_i, "评测集-机构")


if __name__ == "__main__":
    main()
