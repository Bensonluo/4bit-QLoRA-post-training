#!/usr/bin/env python3
"""全面检查主数据匹配训练/评测数据质量。"""

import json
import random
from collections import Counter
from pathlib import Path

random.seed(42)

DOMAIN_ROOT = Path(__file__).resolve().parent.parent


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def extract_query_and_candidates(messages):
    """从 messages 中提取 query 和 candidates。"""
    user_msg = None
    for m in messages:
        if m["role"] == "user":
            user_msg = m["content"]
            break
    if not user_msg:
        return None, []

    lines = user_msg.split("\n")
    query = None
    candidates = []
    for line in lines:
        if line.startswith("【输入") and "】：" in line:
            query = line.split("】：", 1)[1].strip()
        elif line.startswith("[") and "]" in line and "编码:" in line:
            # 提取候选名称
            parts = line.split(", ")
            for p in parts:
                if "名称:" in p:
                    name = p.split("名称:", 1)[1].strip()
                    candidates.append(name)
                    break
    return query, candidates


def extract_labels(messages):
    """从 assistant 输出中提取标签。"""
    for m in messages:
        if m["role"] == "assistant":
            try:
                return json.loads(m["content"])
            except Exception:
                return None
    return None


def is_institution(query, candidates):
    """判断是机构还是产品。"""
    # 通过 system prompt 判断更准确，但这里简单通过内容判断
    if query and ("医院" in query or "诊所" in query or "药店" in query
                  or "卫生" in query or "药房" in query or "社区" in query):
        return True
    return False


def check_dataset(data, name, train_queries=None):
    """检查单个数据集的质量。"""
    print(f"\n{'='*60}")
    print(f"=== 检查 {name} ({len(data)} 条) ===")
    print(f"{'='*60}")

    issues = []

    # 1. 基本统计
    inst_count = 0
    prod_count = 0
    candidate_counts = Counter()
    label_counts = Counter()
    true_positions = []

    # 机构：true 位置
    inst_true_positions = []
    # 产品：A/B/D 分布
    prod_grade_counts = Counter()

    # 用于重叠检测
    all_queries = set()
    dup_queries = []

    # 一眼假检测
    total_candidates = 0

    for idx, item in enumerate(data):
        messages = item.get("messages", [])
        query, candidates = extract_query_and_candidates(messages)
        labels = extract_labels(messages)

        if not query or not candidates:
            issues.append(f"[{idx}] 无法提取 query 或 candidates")
            continue

        # 检查重复 query
        if query in all_queries:
            dup_queries.append((idx, query))
        all_queries.add(query)

        # 判断 domain
        is_inst = is_institution(query, candidates)
        if is_inst:
            inst_count += 1
        else:
            prod_count += 1

        # 候选数量
        candidate_counts[len(candidates)] += 1
        total_candidates += len(candidates)

        # 检查标签
        if not labels or not isinstance(labels, list):
            issues.append(f"[{idx}] 标签格式错误")
            continue

        if len(labels) != len(candidates):
            issues.append(f"[{idx}] 标签数量({len(labels)}) != 候选数量({len(candidates)})")

        # 检查 true/A 的位置
        true_found = False
        for i, label in enumerate(labels):
            if isinstance(label, dict):
                if "matched" in label:
                    # 机构
                    if label["matched"]:
                        true_found = True
                        true_positions.append(i)
                        if is_inst:
                            inst_true_positions.append(i)
                elif "match_grade" in label:
                    # 产品
                    grade = label.get("match_grade", "D")
                    label_counts[grade] += 1
                    if is_inst:
                        pass  # 不应该出现
                    else:
                        prod_grade_counts[grade] += 1
                    if grade == "A":
                        true_found = True
                        true_positions.append(i)

        if not true_found:
            issues.append(f"[{idx}] 没有找到 true/A 标签")

        # 检查一眼假候选
        for _cand in candidates:
            if is_inst:
                # 机构：简单判断，如果候选和 query 完全不在一个城市
                pass  # 比较复杂，暂不自动判断
            else:
                # 产品：提取核心名
                pass  # 需要更复杂的逻辑

    # 2. 输出统计
    print("\n[基本统计]")
    print(f"  机构样本: {inst_count} ({inst_count/len(data)*100:.1f}%)")
    print(f"  产品样本: {prod_count} ({prod_count/len(data)*100:.1f}%)")

    print("\n[候选数量分布]")
    for k in sorted(candidate_counts.keys()):
        print(f"  {k} 个候选: {candidate_counts[k]} 条 ({candidate_counts[k]/len(data)*100:.1f}%)")

    print("\n[正确答案位置分布]")
    pos_dist = Counter(true_positions)
    for i in range(max(pos_dist.keys()) + 1 if pos_dist else 0):
        print(f"  位置 {i}: {pos_dist.get(i, 0)} 条 ({pos_dist.get(i, 0)/len(data)*100:.1f}%)")

    # 检查位置偏差
    if len(true_positions) >= 10:
        first_pos_ratio = true_positions.count(0) / len(true_positions)
        if first_pos_ratio > 0.4:
            issues.append(f"位置偏差警告: {first_pos_ratio*100:.1f}% 的正确答案在位置 0")
        print(f"  位置 0 占比: {first_pos_ratio*100:.1f}%")

    print("\n[产品匹配等级分布]")
    for grade in ["A", "B", "D"]:
        print(f"  {grade}级: {prod_grade_counts.get(grade, 0)} ({prod_grade_counts.get(grade, 0)/max(prod_count,1)*100:.1f}%)")

    # 3. 重复 query
    if dup_queries:
        print(f"\n[重复 Query 警告] {len(dup_queries)} 条")
        for idx, q in dup_queries[:5]:
            print(f"  [{idx}] {q[:50]}...")

    # 4. 与训练集重叠
    if train_queries is not None:
        overlap = all_queries & train_queries
        if overlap:
            print(f"\n[与训练集重叠警告] {len(overlap)} 条")
            for q in list(overlap)[:5]:
                print(f"  - {q[:50]}...")
        else:
            print("\n[与训练集重叠] 0 条 ✅")

    # 5. 输出问题
    if issues:
        print(f"\n[发现 {len(issues)} 个问题]")
        for issue in issues[:20]:
            print(f"  ⚠️ {issue}")
        if len(issues) > 20:
            print(f"  ... 还有 {len(issues)-20} 个问题")
    else:
        print("\n[质量问题] 0 个 ✅")

    return all_queries, issues


def check_hardness(data, name):
    """检查候选列表的'硬度'——有多少一眼假的候选。"""
    print(f"\n{'='*60}")
    print(f"=== 候选硬度分析 {name} ===")
    print(f"{'='*60}")

    easy_count = 0  # 完全无关的候选
    medium_count = 0  # 有一定关联但明显不同
    hard_count = 0  # 近似候选
    total = 0

    for _idx, item in enumerate(data):
        messages = item.get("messages", [])
        query, candidates = extract_query_and_candidates(messages)
        labels = extract_labels(messages)

        if not query or not candidates or not labels:
            continue

        is_inst = is_institution(query, candidates)

        for i, cand in enumerate(candidates):
            if i >= len(labels):
                continue
            label = labels[i]
            total += 1

            if is_inst:
                # 机构：判断硬度
                # 简单启发式：如果候选和 query 有较长的共同子串
                # 或者候选包含 query 中的城市名
                query_city = None
                for city in ["北京", "上海", "广州", "深圳", "成都", "杭州", "武汉", "南京", "重庆", "西安",
                             "天津", "长沙", "郑州", "苏州", "沈阳", "昆明", "济南", "哈尔滨", "长春",
                             "石家庄", "贵阳", "南宁", "福州", "合肥", "南昌", "太原", "兰州", "海口",
                             "银川", "西宁"]:
                    if city in query:
                        query_city = city
                        break

                if query_city and query_city in cand:
                    # 同一城市
                    if label.get("matched", False):
                        hard_count += 1
                    else:
                        # 同一城市但不匹配 = hard negative
                        hard_count += 1
                else:
                    # 不同城市
                    if label.get("matched", False):
                        medium_count += 1  # 不应该出现
                    else:
                        # 检查是否有共同子串
                        common = set(query) & set(cand)
                        if len(common) > len(query) * 0.3:
                            medium_count += 1
                        else:
                            easy_count += 1
            else:
                # 产品：根据 grade 判断
                grade = label.get("match_grade", "D")
                if grade == "A":
                    hard_count += 1  # 完全匹配
                elif grade == "B":
                    hard_count += 1  # 近似匹配
                else:
                    # D 级需要判断是 hard 还是 easy
                    # 简单判断：核心名是否相同
                    # 这里简化处理：假设 D 都是 easy（实际可能有 hard）
                    easy_count += 1

    print(f"\n[候选硬度分布] (总计 {total} 个候选)")
    print(f"  Hard (近似/匹配): {hard_count} ({hard_count/total*100:.1f}%)")
    print(f"  Medium (有一定关联): {medium_count} ({medium_count/total*100:.1f}%)")
    print(f"  Easy (一眼假): {easy_count} ({easy_count/total*100:.1f}%)")

    if easy_count / total > 0.3:
        print("  ⚠️ Easy 候选占比过高 (>30%)")


def main():
    train_path = DOMAIN_ROOT / "data" / "train" / "train.json"
    eval_inst_path = DOMAIN_ROOT / "data" / "test" / "eval_institution.json"
    eval_prod_path = DOMAIN_ROOT / "data" / "test" / "eval_product.json"

    print("=" * 70)
    print("主数据匹配数据质量全面检查")
    print("=" * 70)

    # 加载数据
    train_data = load_json(train_path) if train_path.exists() else []
    eval_inst_data = load_json(eval_inst_path) if eval_inst_path.exists() else []
    eval_prod_data = load_json(eval_prod_path) if eval_prod_path.exists() else []

    print("\n数据集大小:")
    print(f"  训练集: {len(train_data)} 条")
    print(f"  评测-机构: {len(eval_inst_data)} 条")
    print(f"  评测-产品: {len(eval_prod_data)} 条")

    # 检查训练集
    train_queries, train_issues = check_dataset(train_data, "训练集")

    # 检查评测集
    _, inst_issues = check_dataset(eval_inst_data, "评测集-机构", train_queries)
    _, prod_issues = check_dataset(eval_prod_data, "评测集-产品", train_queries)

    # 机构评测和产品评测之间是否有重叠
    inst_queries = set()
    for item in eval_inst_data:
        messages = item.get("messages", [])
        q, _ = extract_query_and_candidates(messages)
        if q:
            inst_queries.add(q)

    prod_queries = set()
    for item in eval_prod_data:
        messages = item.get("messages", [])
        q, _ = extract_query_and_candidates(messages)
        if q:
            prod_queries.add(q)

    cross_overlap = inst_queries & prod_queries
    if cross_overlap:
        print(f"\n[评测集交叉重叠] 机构和产品评测集有 {len(cross_overlap)} 条重复 query ⚠️")
    else:
        print("\n[评测集交叉重叠] 0 条 ✅")

    # 候选硬度分析
    check_hardness(train_data[:500], "训练集 (前500条抽样)")

    # 总结
    print(f"\n{'='*70}")
    print("检查总结")
    print(f"{'='*70}")
    total_issues = len(train_issues) + len(inst_issues) + len(prod_issues)
    if total_issues == 0:
        print("✅ 所有检查通过，未发现质量问题")
    else:
        print(f"⚠️ 共发现 {total_issues} 个问题")
        print(f"   训练集: {len(train_issues)}")
        print(f"   评测-机构: {len(inst_issues)}")
        print(f"   评测-产品: {len(prod_issues)}")


if __name__ == "__main__":
    main()
