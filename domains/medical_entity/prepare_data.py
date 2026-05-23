#!/usr/bin/env python3
"""生成医疗实体匹配训练数据。

用法:
    cd 4bit-QLoRA-post-training
    python -m domains.medical_entity.prepare_data
"""

import json
import random
import sys
from pathlib import Path

random.seed(42)

DOMAIN_ROOT = Path(__file__).resolve().parent

# 模拟药品数据库
DRUG_DATABASE = [
    {"standard_name": "阿莫西林胶囊", "code": "H44023614", "category": "青霉素类",
     "variants": ["阿莫西林", "阿莫仙", "Amoxicillin Capsules", "amoxicillin胶囊", "阿莫西林胶襄"]},
    {"standard_name": "头孢克洛分散片", "code": "H20050649", "category": "头孢类",
     "variants": ["可福乐", "头孢克洛", "cefaclor分散片", "头孢克洛分散片(可福乐)", "头包克洛分散片"]},
    {"standard_name": "头孢克肟分散片", "code": "H20050650", "category": "头孢类",
     "variants": ["世福素", "头孢克肟", "cefixime分散片", "头孢克肟分散片(世福素)", "头孢克肟胶襄"]},
    {"standard_name": "阿莫西林克拉维酸钾片", "code": "H20044234", "category": "青霉素类",
     "variants": ["安灭菌", "阿莫西林克拉维酸钾", "阿莫西林/克拉维酸钾", "安奇", "Amoxicillin and Clavulanate"]},
    {"standard_name": "氨苄西林钠舒巴坦钠", "code": "H20044235", "category": "青霉素类",
     "variants": ["安苄西林钠舒巴坦钠", "氨苄西林舒巴坦", "凯德林", "舒氨新", "ampicillin sulbactam"]},
    {"standard_name": "阿托伐他汀钙片", "code": "H19990258", "category": "调脂药",
     "variants": ["立普妥", "阿托伐他汀", "atorvastatin", "阿乐", "阿托伐他汀钙", "立普妥阿托伐他汀钙片"]},
    {"standard_name": "硝苯地平控释片", "code": "J20040031", "category": "降压药",
     "variants": ["拜新同", "硝苯地平", "nifedipine控释片", "倪福达", "硝苯地平缓释片"]},
    {"standard_name": "盐酸二甲双胍片", "code": "H20023371", "category": "降糖药",
     "variants": ["格华止", "二甲双胍", "metformin", "美迪康", "盐二甲双胍", "二甲双胍盐酸盐片"]},
    {"standard_name": "氯吡格雷片", "code": "H20000542", "category": "抗凝药",
     "variants": ["波立维", "氯吡格雷", "clopidogrel", "泰嘉", "硫酸氯吡格雷片"]},
    {"standard_name": "奥美拉唑肠溶胶囊", "code": "H20059414", "category": "质子泵抑制剂",
     "variants": ["洛赛克", "奥美拉唑", "omeprazole", "奥克", "奥美拉唑肠溶片"]},
    {"standard_name": "布洛芬缓释胶囊", "code": "H10900089", "category": "解热镇痛药",
     "variants": ["芬必得", "布洛芬", "ibuprofen", "美林", "布洛芬缓释胶囊(芬必得)"]},
    {"standard_name": "阿司匹林肠溶片", "code": "H20065051", "category": "解热镇痛药",
     "variants": ["拜阿司匹灵", "阿司匹林", "ASA", "aspirin", "乙酰水杨酸", "巴米尔"]},
    {"standard_name": "左氧氟沙星片", "code": "H20040091", "category": "喹诺酮类",
     "variants": ["可乐必妥", "左氧氟沙星", "levofloxacin", "利复星", "左氧", "左氧氟沙星氯化钠"]},
    {"standard_name": "甲硝唑片", "code": "H32023112", "category": "硝基咪唑类",
     "variants": ["灭滴灵", "甲硝唑", "metronidazole", "甲硝唑氯化钠注射液", "甲消唑"]},
    {"standard_name": "盐酸安罗替尼胶囊", "code": "H20180004", "category": "靶向药",
     "variants": ["正大天晴安罗替尼", "安罗替尼", "anlotinib", "福可维", "盐酸安罗替尼"]},
    {"standard_name": "注射用紫杉醇(白蛋白结合型)", "code": "H20180005", "category": "抗肿瘤药",
     "variants": ["白蛋白紫杉醇", "凯素", "Abraxane", "紫杉醇白蛋白", "注射用紫杉醇"]},
    {"standard_name": "地塞米松磷酸钠注射液", "code": "H32021541", "category": "糖皮质激素",
     "variants": ["地塞米松", "DXM", "dexamethasone", "地米", "地塞米松磷酸钠"]},
    {"standard_name": "盐酸氨溴索口服溶液", "code": "H20059205", "category": "祛痰药",
     "variants": ["沐舒坦", "氨溴索", "ambroxol", "痰易净", "盐酸氨溴索"]},
    {"standard_name": "蒙脱石散", "code": "H20068320", "category": "止泻药",
     "variants": ["思密达", "蒙脱石", "smectite", "蒙脱石散剂", "猛脱石散"]},
    {"standard_name": "复方甘草片", "code": "H32025541", "category": "镇咳药",
     "variants": ["甘草片", "复方甘草", "brown mixture", "复甘草片", "复方甘草合剂"]},
]

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


def _pick_negatives(standard, variant, database, all_standards, n=8):
    same_cat = [(item["standard_name"], item["code"]) for item in database if item["standard_name"] != standard]
    similar = sorted(same_cat, key=lambda x: _edit_distance(variant, x[0]))[: n // 2]
    others = [s for s in all_standards if s[0] != standard and s not in similar]
    rand_negs = random.sample(others, min(n - len(similar), len(others)))
    return similar + rand_negs


def generate_samples(database, entity_type):
    all_standards = [(item["standard_name"], item["code"]) for item in database]
    samples = []
    for item in database:
        standard, code = item["standard_name"], item["code"]
        for variant in item["variants"]:
            negatives = _pick_negatives(standard, variant, database, all_standards)
            samples.append({
                "query": variant, "standard_name": standard, "code": code,
                "entity_type": entity_type,
                "difficulty": classify_difficulty(variant, standard),
                "candidates": [{"name": standard, "code": code, "label": 1}] +
                              [{"name": n[0], "code": n[1], "label": 0} for n in negatives],
            })
        samples.append({
            "query": standard, "standard_name": standard, "code": code,
            "entity_type": entity_type, "difficulty": "easy",
            "candidates": [{"name": standard, "code": code, "label": 1}] +
                          [{"name": n[0], "code": n[1], "label": 0}
                           for n in _pick_negatives(standard, standard, database, all_standards)],
        })
    return samples


def format_as_instruction(sample):
    candidates_text = "\n".join(
        f"{i+1}. {c['name']} ({c['code']})" for i, c in enumerate(sample["candidates"])
    )
    return {
        "instruction": '从候选列表中选出与输入实体匹配的标准名称。输出JSON：{"match_index": 序号, "standard_name": "标准名", "code": "编码", "confidence": 置信度}',
        "input": f"输入实体: {sample['query']}\n候选:\n{candidates_text}",
        "output": json.dumps({"match_index": 1, "standard_name": sample["standard_name"],
                              "code": sample["code"], "confidence": 0.95}, ensure_ascii=False),
        "metadata": {"entity_type": sample["entity_type"], "difficulty": sample["difficulty"]},
    }


def main():
    raw_dir = DOMAIN_ROOT / "data" / "raw"
    train_dir = DOMAIN_ROOT / "data" / "train"
    val_dir = DOMAIN_ROOT / "data" / "val"
    test_dir = DOMAIN_ROOT / "data" / "test"
    for d in [raw_dir, train_dir, val_dir, test_dir]:
        d.mkdir(parents=True, exist_ok=True)

    all_samples = generate_samples(DRUG_DATABASE, "drug") + generate_samples(HOSPITAL_DATABASE, "hospital")
    random.shuffle(all_samples)

    stats = {}
    for s in all_samples:
        stats[s["difficulty"]] = stats.get(s["difficulty"], 0) + 1
    print(f"总样本: {len(all_samples)} (easy={stats.get('easy',0)}, medium={stats.get('medium',0)}, hard={stats.get('hard',0)})")

    with open(raw_dir / "all_samples.json", "w") as f:
        json.dump(all_samples, f, ensure_ascii=False, indent=2)

    n = len(all_samples)
    train_raw, val_raw, test_raw = all_samples[:int(n*0.8)], all_samples[int(n*0.8):int(n*0.9)], all_samples[int(n*0.9):]

    with open(train_dir / "train.json", "w") as f:
        json.dump([format_as_instruction(s) for s in train_raw], f, ensure_ascii=False, indent=2)
    with open(val_dir / "val.json", "w") as f:
        json.dump([format_as_instruction(s) for s in val_raw], f, ensure_ascii=False, indent=2)
    with open(test_dir / "test_instruction.json", "w") as f:
        json.dump([format_as_instruction(s) for s in test_raw], f, ensure_ascii=False, indent=2)
    with open(test_dir / "test_raw.json", "w") as f:
        json.dump(test_raw, f, ensure_ascii=False, indent=2)

    print(f"训练: {len(train_raw)}, 验证: {len(val_raw)}, 测试: {len(test_raw)}")
    print(f"保存至: {DOMAIN_ROOT}/data/")


if __name__ == "__main__":
    main()
