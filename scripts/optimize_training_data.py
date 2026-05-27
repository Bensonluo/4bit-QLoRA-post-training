"""优化训练数据：砍 easy、过滤一眼假候选、补充 hard 样本、重新划分训练/测试集。"""

import json
import random
from pathlib import Path
from collections import defaultdict

SEED = 42
random.seed(SEED)

DOMAIN_ROOT = Path("domains/medical_entity/data")
KB_PATH = DOMAIN_ROOT / "drug_knowledge_base.json"
TRAIN_PATH = DOMAIN_ROOT / "train/train.json"
TEST_BRAND_PATH = DOMAIN_ROOT / "test/test_brand_names.json"


def char_overlap(a: str, b: str) -> int:
    return sum(1 for ch in a if ch in b)


def parse_candidates(input_text: str) -> list[dict]:
    candidates = []
    for line in input_text.split("\n"):
        line = line.strip()
        if line and line[0].isdigit() and ". " in line:
            rest = line.split(". ", 1)[1]
            name = rest.split(" (")[0]
            code = rest.split("(")[1].rstrip(")") if "(" in rest else ""
            candidates.append({"name": name, "code": code})
    return candidates


def build_candidate_text(candidates: list[dict], correct_idx: int) -> str:
    lines = []
    for i, c in enumerate(candidates):
        lines.append(f"{i + 1}. {c['name']} ({c['code']})")
    return "\n".join(lines)


def find_replacement_candidates(query: str, drug_code: str, drugs: list[dict],
                                 generic_groups: dict, exclude_codes: set,
                                 max_replacements: int = 3) -> list[dict]:
    replacements = []
    query_lower = query.lower()

    for generic_name, forms in generic_groups.items():
        if generic_name in query_lower or query_lower in generic_name:
            for form_name in forms:
                form_drug = next((d for d in drugs if d["standard_name"] == form_name), None)
                if form_drug and form_drug["code"] not in exclude_codes:
                    replacements.append({"name": form_drug["standard_name"], "code": form_drug["code"]})

    for d in drugs:
        if d["code"] in exclude_codes:
            continue
        name = d["standard_name"]
        overlap = char_overlap(query, name)
        if overlap >= max(2, len(query) * 0.4) and name not in [r["name"] for r in replacements]:
            replacements.append({"name": name, "code": d["code"]})

    random.shuffle(replacements)
    return replacements[:max_replacements]


def filter_obvious_candidates(query: str, candidates: list[dict],
                                min_overlap: int = 2) -> list[dict]:
    filtered = []
    for c in candidates:
        if char_overlap(query, c["name"]) >= min_overlap:
            filtered.append(c)
        elif len(query) <= 2:
            filtered.append(c)
    return filtered


def main():
    print("=== 优化训练数据 ===\n")

    with open(KB_PATH) as f:
        kb = json.load(f)
    drugs = kb["drugs"]
    generic_groups = kb.get("generic_groups", {})
    code_to_drug = {d["code"]: d for d in drugs}
    print(f"知识库: {len(drugs)} 种药品, {len(generic_groups)} 个通用名分组")

    with open(TRAIN_PATH) as f:
        train_data = json.load(f)
    print(f"现有训练数据: {len(train_data)} 条")

    with open(TEST_BRAND_PATH) as f:
        brand_data = json.load(f)
    print(f"品牌名测试集: {len(brand_data)} 条\n")

    # === Step 1: 优化现有训练数据 ===
    print("--- Step 1: 优化现有训练数据 ---")

    easy_samples = [s for s in train_data if s.get("metadata", {}).get("difficulty") == "easy"]
    medium_samples = [s for s in train_data if s.get("metadata", {}).get("difficulty") == "medium"]
    hard_samples = [s for s in train_data if s.get("metadata", {}).get("difficulty") == "hard"]
    print(f"  easy: {len(easy_samples)}, medium: {len(medium_samples)}, hard: {len(hard_samples)}")

    # 砍 easy 到 3000 条
    random.shuffle(easy_samples)
    easy_kept = easy_samples[:3000]
    print(f"  easy 砍到: {len(easy_kept)}")

    # 过滤一眼假候选（对所有样本）
    def optimize_sample(sample: dict) -> dict | None:
        input_text = sample["input"]
        query = input_text.split("输入实体: ")[1].split("\n")[0] if "输入实体: " in input_text else ""
        candidates = parse_candidates(input_text)

        correct_output = json.loads(sample["output"])
        correct_code = correct_output.get("code", "")
        correct_name = correct_output.get("standard_name", "")

        correct_cand = next((c for c in candidates if c["code"] == correct_code), None)
        if not correct_cand:
            correct_cand = {"name": correct_name, "code": correct_code}

        filtered = [c for c in candidates if c["code"] != correct_code]
        filtered = filter_obvious_candidates(query, filtered, min_overlap=2)

        if len(filtered) < 2:
            exclude_codes = {c["code"] for c in filtered} | {correct_code}
            replacements = find_replacement_candidates(
                query, correct_code, drugs, generic_groups, exclude_codes, max_replacements=3
            )
            filtered.extend(replacements)

        if len(filtered) < 2:
            return None

        all_cands = [correct_cand] + filtered
        random.shuffle(all_cands)
        new_correct_idx = next(i for i, c in enumerate(all_cands) if c["code"] == correct_code) + 1

        cand_text = build_candidate_text(all_cands, new_correct_idx)
        new_input = f"输入实体: {query}\n候选:\n{cand_text}"
        new_output = json.dumps({
            "match_index": new_correct_idx,
            "standard_name": correct_name,
            "code": correct_code,
            "confidence": correct_output.get("confidence", 0.95)
        }, ensure_ascii=False)

        return {
            "instruction": sample["instruction"],
            "input": new_input,
            "output": new_output,
            "metadata": sample["metadata"],
        }

    all_to_optimize = easy_kept + medium_samples + hard_samples
    optimized = []
    skipped = 0
    for s in all_to_optimize:
        result = optimize_sample(s)
        if result:
            optimized.append(result)
        else:
            skipped += 1

    print(f"  优化后: {len(optimized)} 条 (跳过 {skipped} 条候选不足)")

    opt_diff = defaultdict(int)
    for s in optimized:
        opt_diff[s["metadata"].get("difficulty", "?")] += 1
    for k, v in sorted(opt_diff.items()):
        print(f"    {k}: {v}")

    # === Step 2: 从 test_brand_names.json 补充 hard 样本 ===
    print("\n--- Step 2: 从品牌名测试集补充 hard 样本 ---")

    code_to_brand_samples = defaultdict(list)
    for s in brand_data:
        code_to_brand_samples[s["code"]].append(s)

    all_codes = sorted(code_to_brand_samples.keys())
    random.shuffle(all_codes)
    split_idx = int(len(all_codes) * 0.6)
    train_codes = set(all_codes[:split_idx])
    test_codes = set(all_codes[split_idx:])

    print(f"  训练药品: {len(train_codes)} 种, 测试药品: {len(test_codes)} 种")

    brand_train_samples = []
    for code in train_codes:
        for s in code_to_brand_samples[code]:
            query = s["query"]
            correct_cand = {"name": s["standard_name"], "code": s["code"]}

            filtered_negs = []
            for c in s["candidates"]:
                if c.get("label") == 1:
                    continue
                if char_overlap(query, c["name"]) >= 2 or len(query) <= 2:
                    filtered_negs.append(c)

            if len(filtered_negs) < 2:
                exclude_codes = {c["code"] for c in filtered_negs} | {code}
                replacements = find_replacement_candidates(
                    query, code, drugs, generic_groups, exclude_codes, max_replacements=3
                )
                filtered_negs.extend(replacements)

            if len(filtered_negs) < 2:
                continue

            all_cands = [correct_cand] + filtered_negs
            random.shuffle(all_cands)
            new_correct_idx = next(i for i, c in enumerate(all_cands) if c["code"] == code) + 1

            cand_text = build_candidate_text(all_cands, new_correct_idx)
            new_input = f"输入实体: {query}\n候选:\n{cand_text}"
            new_output = json.dumps({
                "match_index": new_correct_idx,
                "standard_name": s["standard_name"],
                "code": code,
                "confidence": 0.95
            }, ensure_ascii=False)

            brand_train_samples.append({
                "instruction": "从候选列表中选出与输入实体匹配的标准名称。输出JSON：{\"match_index\": 序号, \"standard_name\": \"标准名\", \"code\": \"编码\", \"confidence\": 置信度}",
                "input": new_input,
                "output": new_output,
                "metadata": {
                    "entity_type": s.get("entity_type", "drug"),
                    "difficulty": "hard",
                    "source": "brand_names",
                },
            })

    print(f"  补充 hard 样本: {len(brand_train_samples)} 条")

    # 新测试集
    new_test_brand = []
    for code in test_codes:
        new_test_brand.extend(code_to_brand_samples[code])
    print(f"  新测试集: {len(new_test_brand)} 条")

    # === Step 3: 合并 + 划分 ===
    print("\n--- Step 3: 合并 + 划分 ---")

    combined = optimized + brand_train_samples
    random.shuffle(combined)

    final_diff = defaultdict(int)
    for s in combined:
        final_diff[s["metadata"].get("difficulty", "?")] += 1
    print(f"  合并总计: {len(combined)} 条")
    for k, v in sorted(final_diff.items()):
        print(f"    {k}: {v} ({v / len(combined) * 100:.1f}%)")

    # 按 code 分组划分 train/val/test
    all_drug_codes = set()
    for s in combined:
        out = json.loads(s["output"])
        if out.get("code"):
            all_drug_codes.add(out["code"])

    code_list = sorted(all_drug_codes)
    random.shuffle(code_list)
    n = len(code_list)
    train_codes_final = set(code_list[: int(n * 0.8)])
    val_codes_final = set(code_list[int(n * 0.8): int(n * 0.9)])
    test_codes_final = set(code_list[int(n * 0.9):])

    train_split, val_split, test_split = [], [], []
    for s in combined:
        out = json.loads(s["output"])
        code = out.get("code", "")
        if code in train_codes_final:
            train_split.append(s)
        elif code in val_codes_final:
            val_split.append(s)
        elif code in test_codes_final:
            test_split.append(s)
        else:
            train_split.append(s)

    print(f"\n  train: {len(train_split)} 条")
    print(f"  val: {len(val_split)} 条")
    print(f"  test: {len(test_split)} 条")

    # 断言零重叠
    train_c = set(json.loads(s["output"]).get("code") for s in train_split if json.loads(s["output"]).get("code"))
    val_c = set(json.loads(s["output"]).get("code") for s in val_split if json.loads(s["output"]).get("code"))
    test_c = set(json.loads(s["output"]).get("code") for s in test_split if json.loads(s["output"]).get("code"))
    assert not (train_c & val_c), f"train/val 重叠: {train_c & val_c}"
    assert not (train_c & test_c), f"train/test 重叠: {train_c & test_c}"
    print("  零重叠验证: PASS")

    # === Step 4: 保存 ===
    print("\n--- Step 4: 保存 ---")

    (DOMAIN_ROOT / "train").mkdir(parents=True, exist_ok=True)
    (DOMAIN_ROOT / "val").mkdir(parents=True, exist_ok=True)
    (DOMAIN_ROOT / "test").mkdir(parents=True, exist_ok=True)

    # 备份原文件
    import shutil
    for path in [TRAIN_PATH, TEST_BRAND_PATH]:
        backup = path.with_suffix(path.suffix + ".bak")
        if not backup.exists():
            shutil.copy2(path, backup)
            print(f"  备份: {backup}")

    with open(TRAIN_PATH, "w") as f:
        json.dump(train_split, f, ensure_ascii=False, indent=2)
    print(f"  保存: {TRAIN_PATH} ({len(train_split)} 条)")

    with open(DOMAIN_ROOT / "val/val.json", "w") as f:
        json.dump(val_split, f, ensure_ascii=False, indent=2)
    print(f"  保存: val/val.json ({len(val_split)} 条)")

    # test_raw 格式
    test_raw = []
    for s in test_split:
        out = json.loads(s["output"])
        query = s["input"].split("输入实体: ")[1].split("\n")[0]
        candidates = parse_candidates(s["input"])
        test_raw.append({
            "query": query,
            "standard_name": out.get("standard_name", ""),
            "code": out.get("code", ""),
            "entity_type": s.get("metadata", {}).get("entity_type", "drug"),
            "difficulty": s.get("metadata", {}).get("difficulty", "hard"),
            "candidates": candidates,
        })

    with open(DOMAIN_ROOT / "test/test_raw.json", "w") as f:
        json.dump(test_raw, f, ensure_ascii=False, indent=2)
    print(f"  保存: test/test_raw.json ({len(test_raw)} 条)")

    with open(DOMAIN_ROOT / "test/test_instruction.json", "w") as f:
        json.dump(test_split, f, ensure_ascii=False, indent=2)
    print(f"  保存: test/test_instruction.json ({len(test_split)} 条)")

    # brand_names test 稍后保存（需要先去重叠）

    # 从 brand_names test 中移除与训练集重叠的药品
    train_all_codes = train_c | val_c | test_c
    brand_test_codes = set(s["code"] for s in new_test_brand)
    overlap = train_all_codes & brand_test_codes
    if overlap:
        print(f"\n  brand_names test 与 train 重叠 {len(overlap)} 种药品，移除中...")
        new_test_brand = [s for s in new_test_brand if s["code"] not in overlap]
        print(f"  移除后 brand_names test: {len(new_test_brand)} 条")

    # 重新保存 brand_names test
    with open(TEST_BRAND_PATH, "w") as f:
        json.dump(new_test_brand, f, ensure_ascii=False, indent=2)
    print(f"  重新保存: test/test_brand_names.json ({len(new_test_brand)} 条)")

    # 最终验证
    final_brand_codes = set(s["code"] for s in new_test_brand)
    final_overlap = train_all_codes & final_brand_codes
    print(f"  最终验证 - code 重叠: {len(final_overlap)} (应为 0)")

    print("\n=== 完成 ===")


if __name__ == "__main__":
    main()
