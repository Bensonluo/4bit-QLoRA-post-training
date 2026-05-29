#!/usr/bin/env python3
"""深入分析数据质量问题根因。"""

import json
from collections import Counter, defaultdict
from pathlib import Path

DOMAIN_ROOT = Path(__file__).resolve().parent.parent


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_query(messages):
    for m in messages:
        if m["role"] == "user":
            content = m["content"]
            if "】：" in content:
                return content.split("】：", 1)[1].strip().split("\n")[0]
    return None


def main():
    train_data = load_json(DOMAIN_ROOT / "data" / "train" / "train.json")

    print("=" * 70)
    print("问题 1: 重复 Query 分析")
    print("=" * 70)

    query_counts = Counter()
    query_items = defaultdict(list)

    for idx, item in enumerate(train_data):
        q = extract_query(item["messages"])
        if q:
            query_counts[q] += 1
            query_items[q].append(idx)

    duplicates = {q: c for q, c in query_counts.items() if c > 1}
    print(f"\n重复 query 数量: {len(duplicates)}")
    print(f"重复 query 涉及样本数: {sum(duplicates.values())}")

    # 分析重复的模式
    print(f"\n重复次数分布:")
    dup_dist = Counter(duplicates.values())
    for times in sorted(dup_dist.keys()):
        print(f"  重复 {times} 次: {dup_dist[times]} 个 query")

    # 看一个具体例子
    print(f"\n一个重复 query 的示例:")
    for q, count in sorted(duplicates.items(), key=lambda x: -x[1])[:1]:
        print(f"  Query: {q}")
        print(f"  出现 {count} 次，位置: {query_items[q]}")
        for pos in query_items[q][:2]:
            item = train_data[pos]
            for m in item["messages"]:
                if m["role"] == "user":
                    print(f"  位置 {pos} 的候选数: {m['content'].count('编码:')}")
                    break

    print(f"\n{'='*70}")
    print("问题 2: eval_institution 混入产品样本分析")
    print(f"{'='*70}")

    eval_inst = load_json(DOMAIN_ROOT / "data" / "test" / "eval_institution.json")
    prod_like = []
    for idx, item in enumerate(eval_inst):
        q = extract_query(item["messages"])
        if q and ("片" in q or "胶囊" in q or "注射液" in q or "颗粒" in q or "口服液" in q or "g" in q or "ml" in q):
            prod_like.append((idx, q))

    print(f"\n产品-like 的 query 数量: {len(prod_like)}")
    for idx, q in prod_like[:10]:
        print(f"  [{idx}] {q}")

    print(f"\n{'='*70}")
    print("问题 3: Easy 候选分析")
    print(f"{'='*70}")

    # 随机抽 100 条产品样本，看 D 级候选的核心名
    prod_items = []
    for item in train_data:
        q = extract_query(item["messages"])
        if q and ("片" in q or "胶囊" in q or "注射液" in q):
            prod_items.append(item)
            if len(prod_items) >= 100:
                break

    easy_examples = []
    for item in prod_items:
        messages = item["messages"]
        q = extract_query(messages)
        for m in messages:
            if m["role"] == "user":
                lines = m["content"].split("\n")
                candidates = []
                for line in lines:
                    if line.startswith("[") and "名称:" in line:
                        name = line.split("名称:", 1)[1].split(", ")[0].strip()
                        candidates.append(name)
                break
        for m in messages:
            if m["role"] == "assistant":
                try:
                    labels = json.loads(m["content"])
                    for i, label in enumerate(labels):
                        if label.get("match_grade") == "D" and i < len(candidates):
                            easy_examples.append((q, candidates[i]))
                            if len(easy_examples) >= 20:
                                break
                except:
                    pass
                break
        if len(easy_examples) >= 20:
            break

    print(f"\nD 级候选示例 (输入 vs 候选):")
    for q, cand in easy_examples[:15]:
        print(f"  输入: {q[:40]}")
        print(f"  候选: {cand[:40]}")
        # 判断是否一眼假
        q_core = q.split()[0] if q else ""
        c_core = cand.split()[0] if cand else ""
        print(f"  核心名: '{q_core}' vs '{c_core}' -> {'一眼假' if q_core != c_core else '可能是hard'}")
        print()


if __name__ == "__main__":
    main()
