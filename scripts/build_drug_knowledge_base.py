#!/usr/bin/env python3
"""从 chinese-medical-kg + yuanyanyao 构建药品实体归一化知识库。

输出格式: { metadata, drugs: [{ standard_name, code, generic_name, variants }] }
variants 来源: generic_name(去剂型) + 清洗后的别名 + 原研药品牌名

用法:
    source venv/bin/activate
    python scripts/build_drug_knowledge_base.py
"""

import json
import re
from collections import defaultdict
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXTERNAL_DIR = PROJECT_ROOT / "data" / "external"
OUTPUT_PATH = PROJECT_ROOT / "domains" / "medical_entity" / "data" / "drug_knowledge_base.json"
CMKG_DIR = EXTERNAL_DIR / "chinese-medical-kg"
YYY_DIR = EXTERNAL_DIR / "yuanyanyao"

# 常见剂型后缀（从 chinese-medical-kg 的 extract_generic_name_and_dosage 移植）
DOSAGE_FORMS = sorted(
    [
        "注射液", "注射剂", "针剂", "肠溶片", "肠溶胶囊", "缓释片", "缓释胶囊",
        "控释片", "控释胶囊", "分散片", "咀嚼片", "泡腾片", "口含片", "舌下片",
        "薄膜衣片", "糖衣片", "片", "片剂", "胶囊", "胶囊剂", "颗粒", "颗粒剂",
        "散", "散剂", "丸", "丸剂", "栓", "栓剂", "软膏", "软膏剂", "乳膏",
        "乳膏剂", "凝胶", "凝胶剂", "贴", "贴剂", "喷雾", "喷雾剂", "吸入",
        "吸入剂", "滴眼液", "滴耳液", "滴鼻液", "溶液", "溶液剂", "混悬液",
        "混悬剂", "乳剂", "糖浆", "糖浆剂", "口服液", "合剂",
    ],
    key=len,
    reverse=True,
)


def strip_dosage(name: str) -> str:
    """去除剂型后缀，返回通用名。"""
    for form in DOSAGE_FORMS:
        if name.endswith(form) and len(name) > len(form):
            return name[: -len(form)]
    return name


def clean_alias(alias: str) -> str | None:
    if not alias:
        return None
    alias = alias.strip()
    if alias.lower() in ("nan", "none", "", "-"):
        return None
    if re.match(r"^869\d{11}", alias):
        return None
    if re.match(r"^\d+$", alias):
        return None
    if len(alias) < 2 or len(alias) > 60:
        return None
    return alias


def load_yuanyanyao_brands() -> dict[str, list[str]]:
    brands: dict[str, list[str]] = defaultdict(list)
    data_dir = YYY_DIR / "data"
    if not data_dir.exists():
        return {}
    for md_file in data_dir.glob("*.md"):
        text = md_file.read_text(encoding="utf-8")
        m = re.search(r"^---\n(.*?)\n---", text, re.DOTALL)
        if not m:
            continue
        try:
            fm = yaml.safe_load(m.group(1))
        except yaml.YAMLError:
            continue
        if not isinstance(fm, dict):
            continue
        generic = (fm.get("genericName") or "").strip()
        if not generic:
            continue
        for field in ("brandName", "brandNameEn", "inn"):
            val = (fm.get(field) or "")
            if isinstance(val, str):
                val = val.strip()
                if val and val != generic:
                    brands[generic].append(val)
    return dict(brands)


def main():
    drugs_path = CMKG_DIR / "ontology" / "data" / "drugs.json"
    if not drugs_path.exists():
        raise FileNotFoundError(
            f"请先运行: cd {CMKG_DIR} && python scripts/parse_official_medical_excel.py"
        )

    print("加载 chinese-medical-kg 药品数据...")
    with open(drugs_path, encoding="utf-8") as f:
        raw_drugs = json.load(f)
    print(f"  原始药物: {len(raw_drugs)} 条")

    print("加载原研药品牌名...")
    yyy_brands = load_yuanyanyao_brands()
    print(f"  品牌名映射: {len(yyy_brands)} 个通用名")

    # 按 generic_name 分组（跨剂型）
    generic_groups: dict[str, list[str]] = defaultdict(list)

    drugs = []
    for name, data in raw_drugs.items():
        standard_name = data.get("standard_name", name)
        generic_name = data.get("generic_name", "") or strip_dosage(standard_name)
        approvals = data.get("approval_numbers", [])
        code = approvals[0] if approvals else ""

        # 收集变体
        variants = set()
        # 通用名
        if generic_name != standard_name:
            variants.add(generic_name)
        # 清洗别名
        for a in data.get("aliases", []):
            c = clean_alias(a)
            if c and c != standard_name:
                variants.add(c)
        # 品牌名
        for b in yyy_brands.get(generic_name, []):
            if b != standard_name:
                variants.add(b)

        drugs.append({
            "standard_name": standard_name,
            "code": code,
            "generic_name": generic_name,
            "variants": sorted(variants),
        })
        generic_groups[generic_name].append(standard_name)

    # 过滤：只保留有变体或通用名!=标准名的药物
    drugs = [d for d in drugs if d["variants"] or d["generic_name"] != d["standard_name"]]

    multi_form = {k: v for k, v in generic_groups.items() if len(v) > 1}

    metadata = {
        "total_drugs": len(drugs),
        "unique_generic_names": len(set(d["generic_name"] for d in drugs)),
        "multi_form_generics": len(multi_form),
        "generic_groups_sample": {k: v for i, (k, v) in enumerate(multi_form.items()) if i < 20},
    }

    print(f"\n知识库统计:")
    print(f"  总药物实体: {metadata['total_drugs']}")
    print(f"  不同通用名: {metadata['unique_generic_names']}")
    print(f"  多剂型通用名: {metadata['multi_form_generics']}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump({"metadata": metadata, "drugs": drugs, "generic_groups": multi_form}, f, ensure_ascii=False, indent=2)

    print(f"\n已保存: {OUTPUT_PATH} ({OUTPUT_PATH.stat().st_size / 1024 / 1024:.1f} MB)")


if __name__ == "__main__":
    main()
