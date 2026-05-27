#!/usr/bin/env python3
"""生成医疗实体匹配训练数据（大规模版）。

从 drug_knowledge_base.json 加载 14K+ 药品，动态生成变体和负采样，
合并原有 10 家医院数据，产出 Alpaca 指令格式训练集。

用法:
    cd 4bit-QLoRA-post-training
    python -m domains.medical_entity.prepare_data
    python -m domains.medical_entity.prepare_data --max-drugs 5000   # 限制药品数
"""

import json
import random
import re
import sys
from pathlib import Path

random.seed(42)

DOMAIN_ROOT = Path(__file__).resolve().parent
KB_PATH = DOMAIN_ROOT / "data" / "drug_knowledge_base.json"

# ─── 原有医院数据库（保留） ──────────────────────────────────────────

HOSPITAL_DATABASE = [
    {"standard_name": "北京大学第三医院", "code": "H110108001", "city": "北京",
     "variants": ["北医三院", "北大三院", "北医三", "北京大学第三附属医院", "PUTH"]},
    {"standard_name": "北京协和医院", "code": "H110101001", "city": "北京",
     "variants": ["协和", "协和医院", "北京协和", "PUMCH", "中国医学科学院北京协和医院"]},
    {"standard_name": "四川大学华西医院", "code": "H510104001", "city": "成都",
     "variants": ["华西医院", "华西", "川大华西", "West China Hospital", "四川华西"]},
    {"standard_name": "复旦大学附属中山医院", "code": "H310104001", "city": "上海",
     "variants": ["中山医院", "复旦中山", "上海中山", "ZS Hospital", "复旦大学中山医院"]},
    {"standard_name": "浙江大学医学院附属第一医院", "code": "H330102001", "city": "杭州",
     "variants": ["浙一", "浙一医院", "浙大附一", "浙江大学第一附属医院", "浙大附一院"]},
    {"standard_name": "中山大学附属第一医院", "code": "H440106001", "city": "广州",
     "variants": ["中山一院", "中大附一", "广州中山一院", "FAH-SYSU"]},
    {"standard_name": "华中科技大学同济医学院附属同济医院", "code": "H420106001", "city": "武汉",
     "variants": ["同济医院", "武汉同济", "华科同济", "Tongji Hospital"]},
    {"standard_name": "中南大学湘雅医院", "code": "H430103001", "city": "长沙",
     "variants": ["湘雅医院", "湘雅", "中南湘雅", "Xiangya Hospital"]},
    {"standard_name": "上海交通大学医学院附属瑞金医院", "code": "H310103001", "city": "上海",
     "variants": ["瑞金医院", "瑞金", "上海瑞金", "Ruijin Hospital"]},
    {"standard_name": "广东省人民医院", "code": "H440106002", "city": "广州",
     "variants": ["省医", "广东省医", "广省人民医院", "GDPPH"]},
]


# ─── 工具函数 ─────────────────────────────────────────────────────────

def _edit_distance(s1: str, s2: str) -> int:
    if len(s1) < len(s2):
        return _edit_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    prev = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr = [i + 1]
        for j, c2 in enumerate(s2):
            curr.append(min(prev[j + 1] + 1, curr[j] + 1, prev[j] + (c1 != c2)))
        prev = curr
    return prev[-1]


def classify_difficulty(query: str, standard: str) -> str:
    if query == standard:
        return "easy"
    q = query.lower().replace(" ", "")
    s = standard.lower().replace(" ", "")
    if q == s or q in s or s in q:
        return "easy"
    if _edit_distance(q, s) <= 2:
        return "medium"
    return "hard"


def add_noise(text: str) -> str:
    """对文本注入 1-2 个字符噪声（替换/删除/插入）。"""
    if len(text) < 3:
        return text
    ops = []
    chars = list(text)
    n_ops = random.choice([1, 1, 2])
    for _ in range(n_ops):
        pos = random.randint(0, len(chars) - 1)
        op = random.choice(["replace", "delete", "insert"])
        if op == "replace":
            chars[pos] = random.choice("的一是不了在人我有这他中大来")
        elif op == "delete" and len(chars) > 3:
            chars.pop(pos)
        elif op == "insert":
            chars.insert(pos, random.choice("的一是不了"))
    return "".join(chars)


# ─── 知识库加载 ──────────────────────────────────────────────────────

def load_drug_kb(max_drugs: int | None = None) -> tuple[list[dict], dict[str, list[str]]]:
    """加载药品知识库。返回 (drugs_list, generic_groups)。"""
    if not KB_PATH.exists():
        print(f"错误: 知识库不存在 {KB_PATH}")
        print("请先运行: python scripts/build_drug_knowledge_base.py")
        sys.exit(1)

    with open(KB_PATH, encoding="utf-8") as f:
        data = json.load(f)

    drugs = data["drugs"]
    generic_groups = data.get("generic_groups", {})

    if max_drugs and max_drugs < len(drugs):
        # 优先选择有多剂型的通用名对应的药品（信息更丰富）
        multi_form = {name for names in generic_groups.values() for name in names}
        priority = [d for d in drugs if d["standard_name"] in multi_form]
        rest = [d for d in drugs if d["standard_name"] not in multi_form]
        drugs = priority[:max_drugs]
        if len(drugs) < max_drugs:
            drugs += rest[: max_drugs - len(drugs)]

    print(f"加载药品知识库: {len(drugs)} 种药物, {len(generic_groups)} 个多剂型通用名")
    return drugs, generic_groups


# ─── 负采样 ──────────────────────────────────────────────────────────

def pick_drug_negatives(
    query_standard: str,
    query_generic: str,
    all_standards: list[tuple[str, str]],
    generic_groups: dict[str, list[str]],
    n: int = 8,
) -> list[tuple[str, str]]:
    """药品负采样：同类硬负例 + 随机负例。

    硬负例策略：
    1. 同 generic_name 的其他剂型（最相似）
    2. generic_name 前缀相似的其他药物
    3. 随机负例
    """
    negatives = []
    used = {query_standard}

    # 1. 同通用名不同剂型（最强硬负例）
    same_generic = generic_groups.get(query_generic, [])
    for name in same_generic:
        if name != query_standard and name not in used:
            # 找到对应的 code
            for s, c in all_standards:
                if s == name:
                    negatives.append((s, c))
                    used.add(name)
                    break
        if len(negatives) >= n // 3:
            break

    # 2. 前缀相似的药物（名称前 2-4 字相同）
    prefix = query_generic[:min(4, len(query_generic))]
    if prefix:
        similar = [
            (s, c) for s, c in all_standards
            if s != query_standard and s not in used and s.startswith(prefix)
        ]
        random.shuffle(similar)
        for s, c in similar:
            negatives.append((s, c))
            used.add(s)
            if len(negatives) >= n // 2:
                break

    # 3. 随机负例
    pool = [(s, c) for s, c in all_standards if s not in used]
    if pool:
        negatives.extend(random.sample(pool, min(n - len(negatives), len(pool))))

    return negatives[:n]


def pick_hospital_negatives(
    standard: str,
    all_standards: list[tuple[str, str]],
    n: int = 8,
) -> list[tuple[str, str]]:
    """医院负采样（保留原逻辑）。"""
    same_cat = [(s, c) for s, c in all_standards if s != standard]
    similar = sorted(same_cat, key=lambda x: _edit_distance(standard, x[0]))[: n // 2]
    others = [s for s in all_standards if s[0] != standard and s not in similar]
    rand_negs = random.sample(others, min(n - len(similar), len(others)))
    return similar + rand_negs


# ─── 样本生成 ────────────────────────────────────────────────────────

def generate_drug_samples(
    drugs: list[dict],
    generic_groups: dict[str, list[str]],
    augment_noise: bool = True,
) -> list[dict]:
    """从药品知识库生成训练样本。"""
    all_standards = [(d["standard_name"], d["code"]) for d in drugs]
    samples = []

    for drug in drugs:
        standard = drug["standard_name"]
        code = drug["code"]
        generic = drug["generic_name"]
        known_variants = drug.get("variants", [])

        # 收集所有可用的 query
        queries = set()

        # 1. 通用名（最常见的查询方式）
        if generic and generic != standard:
            queries.add(generic)

        # 2. 已知变体/品牌名
        for v in known_variants:
            queries.add(v)

        # 3. 标准名本身（easy 难度）
        queries.add(standard)

        # 4. 同通用名的其他剂型名作为查询
        for other_name in generic_groups.get(generic, []):
            if other_name != standard:
                queries.add(other_name)

        # 5. 噪声注入（每个 query 最多生成 1 个噪声变体）
        noise_queries = set()
        if augment_noise:
            for q in list(queries):
                if len(q) >= 4 and random.random() < 0.3:
                    noise_queries.add(add_noise(q))
        queries.update(noise_queries)

        # 为每个 query 生成样本
        for query in queries:
            negatives = pick_drug_negatives(
                standard, generic, all_standards, generic_groups
            )
            if not negatives:
                continue
            candidates = (
                [{"name": standard, "code": code, "label": 1}]
                + [{"name": n[0], "code": n[1], "label": 0} for n in negatives]
            )
            random.shuffle(candidates)
            samples.append({
                "query": query,
                "standard_name": standard,
                "code": code,
                "entity_type": "drug",
                "difficulty": classify_difficulty(query, standard),
                "candidates": candidates,
            })

    return samples


def generate_hospital_samples() -> list[dict]:
    """生成医院实体匹配样本（保留原逻辑）。"""
    all_standards = [(h["standard_name"], h["code"]) for h in HOSPITAL_DATABASE]
    samples = []
    for item in HOSPITAL_DATABASE:
        standard, code = item["standard_name"], item["code"]
        for variant in item["variants"]:
            negatives = pick_hospital_negatives(standard, all_standards)
            candidates = (
                [{"name": standard, "code": code, "label": 1}]
                + [{"name": n[0], "code": n[1], "label": 0} for n in negatives]
            )
            random.shuffle(candidates)
            samples.append({
                "query": variant,
                "standard_name": standard,
                "code": code,
                "entity_type": "hospital",
                "difficulty": classify_difficulty(variant, standard),
                "candidates": candidates,
            })
        # 标准名本身
        negatives = pick_hospital_negatives(standard, all_standards)
        candidates = (
            [{"name": standard, "code": code, "label": 1}]
            + [{"name": n[0], "code": n[1], "label": 0} for n in negatives]
        )
        random.shuffle(candidates)
        samples.append({
            "query": standard,
            "standard_name": standard,
            "code": code,
            "entity_type": "hospital",
            "difficulty": "easy",
            "candidates": candidates,
        })
    return samples


# ─── 指令格式化 ──────────────────────────────────────────────────────

def format_as_instruction(sample: dict) -> dict:
    candidates_text = "\n".join(
        f"{i + 1}. {c['name']} ({c['code']})" for i, c in enumerate(sample["candidates"])
    )
    match_idx = next(
        i for i, c in enumerate(sample["candidates"]) if c["label"] == 1
    ) + 1
    return {
        "instruction": (
            '从候选列表中选出与输入实体匹配的标准名称。'
            '输出JSON：{"match_index": 序号, "standard_name": "标准名", '
            '"code": "编码", "confidence": 置信度}'
        ),
        "input": f"输入实体: {sample['query']}\n候选:\n{candidates_text}",
        "output": json.dumps(
            {
                "match_index": match_idx,
                "standard_name": sample["standard_name"],
                "code": sample["code"],
                "confidence": 0.95,
            },
            ensure_ascii=False,
        ),
        "metadata": {
            "entity_type": sample["entity_type"],
            "difficulty": sample["difficulty"],
        },
    }


# ─── 主流程 ──────────────────────────────────────────────────────────

def main(max_drugs: int | None = None):
    raw_dir = DOMAIN_ROOT / "data" / "raw"
    train_dir = DOMAIN_ROOT / "data" / "train"
    val_dir = DOMAIN_ROOT / "data" / "val"
    test_dir = DOMAIN_ROOT / "data" / "test"
    for d in [raw_dir, train_dir, val_dir, test_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # 加载药品
    drugs, generic_groups = load_drug_kb(max_drugs)

    # 按药品编码分组，确保 train/val/test 的药品完全不重叠
    random.shuffle(drugs)
    n_drugs = len(drugs)
    train_drugs = drugs[: int(n_drugs * 0.8)]
    val_drugs = drugs[int(n_drugs * 0.8): int(n_drugs * 0.9)]
    test_drugs = drugs[int(n_drugs * 0.9):]

    train_codes = {d["code"] for d in train_drugs}
    val_codes = {d["code"] for d in val_drugs}
    test_codes = {d["code"] for d in test_drugs}
    assert not (train_codes & val_codes), "train/val 药品重叠!"
    assert not (train_codes & test_codes), "train/test 药品重叠!"

    # 各自生成样本（负采样从全量药品中取）
    print(f"按药品划分: 训练 {len(train_drugs)} 种 | 验证 {len(val_drugs)} 种 | 测试 {len(test_drugs)} 种")

    print("生成训练样本...")
    train_samples = generate_drug_samples(train_drugs, generic_groups, augment_noise=True)
    print(f"  训练样本: {len(train_samples)}")

    print("生成验证样本...")
    val_samples = generate_drug_samples(val_drugs, generic_groups, augment_noise=True)
    print(f"  验证样本: {len(val_samples)}")

    print("生成测试样本...")
    test_samples = generate_drug_samples(test_drugs, generic_groups, augment_noise=True)
    print(f"  测试样本: {len(test_samples)}")

    # 医院数据全部加入训练集（量小，不影响评估）
    print("生成医院训练样本...")
    hospital_samples = generate_hospital_samples()
    print(f"  医院样本: {len(hospital_samples)}")
    train_samples = train_samples + hospital_samples

    # 打乱
    random.shuffle(train_samples)
    random.shuffle(val_samples)
    random.shuffle(test_samples)

    # 统计
    for name, samples in [("训练", train_samples), ("验证", val_samples), ("测试", test_samples)]:
        stats: dict[str, int] = {}
        type_stats: dict[str, int] = {}
        for s in samples:
            stats[s["difficulty"]] = stats.get(s["difficulty"], 0) + 1
            type_stats[s["entity_type"]] = type_stats.get(s["entity_type"], 0) + 1
        print(f"\n{name}: {len(samples)} 条")
        print(f"  难度: easy={stats.get('easy', 0)}, medium={stats.get('medium', 0)}, hard={stats.get('hard', 0)}")
        print(f"  类型: {type_stats}")

    # 保存
    with open(train_dir / "train.json", "w") as f:
        json.dump([format_as_instruction(s) for s in train_samples], f, ensure_ascii=False, indent=2)
    with open(val_dir / "val.json", "w") as f:
        json.dump([format_as_instruction(s) for s in val_samples], f, ensure_ascii=False, indent=2)
    with open(test_dir / "test_instruction.json", "w") as f:
        json.dump([format_as_instruction(s) for s in test_samples], f, ensure_ascii=False, indent=2)
    with open(test_dir / "test_raw.json", "w") as f:
        json.dump(test_samples, f, ensure_ascii=False, indent=2)

    # 保存原始数据（合并）
    with open(raw_dir / "all_samples.json", "w") as f:
        json.dump(train_samples + val_samples + test_samples, f, ensure_ascii=False, indent=2)

    print(f"\n保存至: {DOMAIN_ROOT}/data/")
    print(f"训练集药品: {len(train_codes)} 种 (codes 已写入 train)")
    print(f"测试集药品: {len(test_codes)} 种 (与训练集完全无重叠)")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-drugs", type=int, default=None, help="限制药品数量（测试用）")
    args = parser.parse_args()
    main(max_drugs=args.max_drugs)
