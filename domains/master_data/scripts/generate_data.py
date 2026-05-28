#!/usr/bin/env python3
"""主数据匹配 SFT 训练数据生成脚本。

生成机构匹配和产品匹配的训练/评测数据，使用 messages chat 格式。
"""

import json
import random
from pathlib import Path

random.seed(42)

DOMAIN_ROOT = Path(__file__).resolve().parent.parent
MEDICAL_ENTITY_ROOT = DOMAIN_ROOT.parent / "medical_entity"

# ── System Prompts ──
INST_SYSTEM_PROMPT = (
    "角色：专业的医药机构主数据匹配审核员\n"
    "任务：逐一判断【输入机构】与列表中的每个【候选机构】是否代表同一物理实体。\n"
    "严格规则：\n"
    "1. 必须对每个候选机构【独立】进行验证，候选之间绝不能互相干扰。\n"
    "2. 严格按优先级1-6执行短路验证（精确>地理>大学>粒度>修饰词>辅助）。只要任何高优先级冲突，该候选即为false。\n"
    "3. 强制失败规则：若出现核心冲突，或符合四项严格失败条件之一，直接判false。\n"
    "验证流程（对每个候选独立执行）：\n"
    "Step 1 — 输入清洗：在内心去除人名、联系方式、测试标记、无意义数字。\n"
    "Step 2 — 括号评估：判断核心机构名在括号内还是括号外。\n"
    "Step 3 — 短路匹配验证：\n"
    "- 优先级1：精确匹配（完全一致直接判true）\n"
    "- 优先级2：地理信息层级（街道>区>市>省，存在层级冲突则判false）\n"
    "- 优先级3：大学/研究机构（上下级关系必须明确，不可错配）\n"
    "- 优先级4：最小粒度匹配（院区、分院不可与总院混淆）\n"
    "- 优先级5：精确修饰词（数字、分院、子类型、区域后缀冲突则判false）\n"
    "- 优先级6：辅助信息综合\n"
    "输出要求：\n"
    "严格输出标准JSON数组，数组长度必须与候选列表一致。不要输出任何思考过程或其他字符。\n"
    '格式：\n'
    '[\n'
    '  {"index": 1, "reasoning": "P1(通过)->P2(冲突:输入A区,候选B区)->判定false", "matched": false, "confidence": "Low"},\n'
    '  {"index": 2, "reasoning": "P1(通过)->P2(通过)->P3(通过)->全通过", "matched": true, "confidence": "High"}\n'
    ']'
)

PROD_SYSTEM_PROMPT = (
    "角色：专业的产品数据匹配专家\n"
    "任务：逐一判定【输入产品】与列表中的每个【候选产品】是否为同一核心产品，并评估匹配等级。\n"
    "严格规则：\n"
    "1. 必须对每个候选产品【独立】进行验证，候选之间绝不能互相干扰。\n"
    "2. 核心名称拥有绝对一票否决权。\n"
    "3. 不需要计算具体分数，只需根据差异情况判定匹配等级（A/B/C/D）。\n"
    "4. 必须先提取核心名，再比对修饰词，最后比对规格。\n"
    "判定流程（对每个候选独立执行）：\n"
    "Step 1 — 一票否决（核心名称一致性）：\n"
    "去除修饰词，提取核心产品名。若核心名不一致，直接判定为 D级。\n"
    "Step 2 — 差异提取与定级（仅在核心名一致时执行）：\n"
    "比对修饰词（材质、方法、品牌、型号）和规格（尺寸、包装数、容量），判定匹配等级：\n"
    "- A级：核心名一致，且修饰词完全一致，规格完全一致。\n"
    "- B级：核心名一致，修饰词一致或缺失可忽略，规格存在微小差异但不影响主体（如10片装vs12片装）。\n"
    "- C级：核心名一致，但修饰词有关键差异（如材质：棉 vs 化纤），或规格有明显量级差异（如0.5g vs 0.25g）。\n"
    "- D级：核心名不一致。\n"
    "输出要求：\n"
    "严格输出标准JSON数组，数组长度必须与候选列表一致。不要输出任何思考过程或其他字符。\n"
    '格式：\n'
    '[\n'
    '  {"index": 1, "core_name_match": false, "modifier_diff": "无", "spec_diff": "无", "match_grade": "D"},\n'
    '  {"index": 2, "core_name_match": true, "modifier_diff": "无", "spec_diff": "10片装vs12片装", "match_grade": "B"}\n'
    ']'
)

# ── Geography Data ──
CITIES = {
    "北京市": {"province": "北京市", "districts": ["东城区", "西城区", "朝阳区", "海淀区", "丰台区", "石景山区", "通州区", "大兴区", "昌平区", "顺义区"]},
    "上海市": {"province": "上海市", "districts": ["黄浦区", "徐汇区", "长宁区", "静安区", "普陀区", "虹口区", "杨浦区", "浦东新区", "闵行区", "宝山区"]},
    "广州市": {"province": "广东省", "districts": ["越秀区", "海珠区", "荔湾区", "天河区", "白云区", "黄埔区", "番禺区", "花都区", "南沙区", "增城区"]},
    "深圳市": {"province": "广东省", "districts": ["罗湖区", "福田区", "南山区", "宝安区", "龙岗区", "盐田区", "龙华区", "坪山区", "光明区"]},
    "成都市": {"province": "四川省", "districts": ["锦江区", "青羊区", "金牛区", "武侯区", "成华区", "龙泉驿区", "青白江区", "新都区", "温江区", "双流区"]},
    "杭州市": {"province": "浙江省", "districts": ["上城区", "拱墅区", "西湖区", "滨江区", "萧山区", "余杭区", "临平区", "钱塘区", "富阳区"]},
    "武汉市": {"province": "湖北省", "districts": ["江岸区", "江汉区", "硚口区", "汉阳区", "武昌区", "青山区", "洪山区", "东西湖区", "蔡甸区"]},
    "南京市": {"province": "江苏省", "districts": ["玄武区", "秦淮区", "建邺区", "鼓楼区", "浦口区", "栖霞区", "雨花台区", "江宁区"]},
    "重庆市": {"province": "重庆市", "districts": ["渝中区", "大渡口区", "江北区", "沙坪坝区", "九龙坡区", "南岸区", "北碚区", "渝北区", "巴南区"]},
    "西安市": {"province": "陕西省", "districts": ["新城区", "碑林区", "莲湖区", "灞桥区", "未央区", "雁塔区", "阎良区", "临潼区", "长安区"]},
    "天津市": {"province": "天津市", "districts": ["和平区", "河东区", "河西区", "南开区", "河北区", "红桥区", "东丽区", "西青区", "津南区", "北辰区"]},
    "长沙市": {"province": "湖南省", "districts": ["芙蓉区", "天心区", "岳麓区", "开福区", "雨花区", "望城区", "长沙县"]},
    "郑州市": {"province": "河南省", "districts": ["中原区", "二七区", "管城区", "金水区", "上街区", "惠济区"]},
    "苏州市": {"province": "江苏省", "districts": ["虎丘区", "吴中区", "相城区", "姑苏区", "吴江区", "昆山市", "常熟市"]},
    "沈阳市": {"province": "辽宁省", "districts": ["和平区", "沈河区", "大东区", "皇姑区", "铁西区", "苏家屯区", "浑南区", "沈北新区"]},
}

# ── Pharmacy Chain Data ──
PHARMACY_CHAINS = [
    "大参林", "一心堂", "老百姓大药房", "国大药房", "海王星辰",
    "益丰大药房", "桐君阁大药房", "华氏大药房", "成大方圆",
    "众康大药房", "健客大药房", "同仁堂", "雷允上",
    "养和堂", "保和堂", "济民大药房", "康泽大药房",
    "好药师大药房", "仁和药房", "九洲大药房",
]

HOSPITAL_TYPES = ["人民医院", "中心医院", "第一医院", "第二医院", "第三医院", "中医院", "妇幼保健院"]

STREETS = [
    "中山路", "解放路", "建设路", "人民路", "和平路", "光明路",
    "长江路", "黄河路", "文化路", "民主路", "新华路", "胜利路",
    "前进路", "幸福路", "健康路", "朝阳路", "学府路", "科技路",
]


def _gen_code(prefix: str, idx: int) -> str:
    return f"{prefix}{idx:06d}"


# ════════════════════════════════════════════
# Institution Knowledge Base
# ════════════════════════════════════════════
def generate_institution_kb() -> list[dict]:
    """Generate synthetic institution KB: pharmacies, hospitals, clinics."""
    kb = []
    idx = 0

    for city, info in CITIES.items():
        prov = info["province"]
        districts = info["districts"]

        # Chain pharmacies: 2-4 chains per city, 2-5 branches per chain
        for chain in random.sample(PHARMACY_CHAINS, min(random.randint(2, 4), len(PHARMACY_CHAINS))):
            for _ in range(random.randint(2, 5)):
                dist = random.choice(districts)
                street = random.choice(STREETS)
                branch_suffix = random.choice([
                    f"({dist}{street}店)",
                    f"({dist}店)",
                    f"第{random.randint(1, 200)}分店",
                    f"({street}店)",
                    f"({dist}{random.choice(['旗舰店', '中心店', '总店'])})",
                ])
                standard_name = f"{chain}{branch_suffix}"
                full_name = f"{chain}连锁有限公司{city}{dist}{street}药店"
                address = f"{city}{dist}{street}{random.randint(1, 300)}号"

                idx += 1
                kb.append({
                    "code": _gen_code("P", idx),
                    "standard_name": standard_name,
                    "full_name": full_name,
                    "short_name": chain,
                    "address": address,
                    "city": city,
                    "district": dist,
                    "province": prov,
                    "type": "pharmacy_chain",
                    "chain": chain,
                })

        # Independent pharmacies: 2-4 per district
        for dist in random.sample(districts, min(3, len(districts))):
            for _ in range(random.randint(1, 3)):
                street = random.choice(STREETS)
                prefixes = ["康乐", "济民", "仁心", "安康", "健民", "祥和", "德心", "益康", "惠民"]
                suffixes = ["药店", "药房", "大药房", "医药商店"]
                name = f"{random.choice(prefixes)}{random.choice(suffixes)}"
                address = f"{city}{dist}{street}{random.randint(1, 500)}号"

                idx += 1
                kb.append({
                    "code": _gen_code("P", idx),
                    "standard_name": name,
                    "full_name": f"{name}({address})",
                    "short_name": name,
                    "address": address,
                    "city": city,
                    "district": dist,
                    "province": prov,
                    "type": "pharmacy_independent",
                    "chain": None,
                })

        # Hospitals: 1-3 per city
        for htype in random.sample(HOSPITAL_TYPES, min(random.randint(1, 3), len(HOSPITAL_TYPES))):
            dist = random.choice(districts)
            standard_name = f"{city}{htype}"
            address = f"{city}{dist}{random.choice(STREETS)}{random.randint(1, 100)}号"

            idx += 1
            kb.append({
                "code": _gen_code("H", idx),
                "standard_name": standard_name,
                "full_name": standard_name,
                "short_name": htype,
                "address": address,
                "city": city,
                "district": dist,
                "province": prov,
                "type": "hospital",
                "chain": None,
            })

        # Community health centers: 1-2 per district
        for dist in random.sample(districts, min(2, len(districts))):
            street = random.choice(STREETS)
            name = f"{city}{dist}{street}社区卫生服务中心"

            idx += 1
            kb.append({
                "code": _gen_code("C", idx),
                "standard_name": name,
                "full_name": name,
                "short_name": f"{street}社区服务中心",
                "address": f"{city}{dist}{street}{random.randint(1, 100)}号",
                "city": city,
                "district": dist,
                "province": prov,
                "type": "community",
                "chain": None,
            })

    return kb


# ════════════════════════════════════════════
# Product Knowledge Base
# ════════════════════════════════════════════
def load_product_kb() -> list[dict]:
    """Load drug KB and add synthetic specs."""
    kb_path = MEDICAL_ENTITY_ROOT / "data" / "drug_knowledge_base.json"
    with open(kb_path) as f:
        raw = json.load(f)

    drugs = raw["drugs"]
    # Group by generic_name for creating similar candidates
    generic_groups = {}
    for drug in drugs:
        gn = drug["generic_name"]
        if gn not in generic_groups:
            generic_groups[gn] = []
        # Add synthetic spec based on drug form
        name = drug["standard_name"]
        spec = _gen_spec(name)
        drug["spec"] = spec
        generic_groups[gn].append(drug)

    return drugs, generic_groups


def _gen_spec(drug_name: str) -> str:
    """Generate realistic drug spec based on name."""
    if "胶囊" in drug_name:
        specs = ["0.25g*24粒/盒", "0.5g*20粒/盒", "0.25g*36粒/盒", "0.125g*12粒/盒"]
    elif "片" in drug_name:
        specs = ["0.25g*24片/盒", "0.5g*30片/盒", "0.25g*12片/盒", "0.5g*12片/盒", "0.1g*30片/盒"]
    elif "颗粒" in drug_name:
        specs = ["10g*6袋/盒", "3g*12袋/盒", "5g*10袋/盒", "15g*6袋/盒"]
    elif "注射液" in drug_name:
        specs = ["2ml:0.1g", "5ml:0.25g", "10ml:0.5g", "100ml:0.5g"]
    elif "口服液" in drug_name:
        specs = ["10ml*6支/盒", "10ml*10支/盒", "20ml*6支/盒"]
    elif "丸" in drug_name:
        specs = ["6g*10丸/盒", "9g*10丸/盒", "3g*12丸/盒", "60g/瓶"]
    elif "散" in drug_name:
        specs = ["3g*6袋/盒", "6g*3袋/盒", "10g/袋"]
    elif "膏" in drug_name:
        specs = ["20g/支", "10g/支", "30g/支", "50g/盒"]
    elif "滴眼液" in drug_name:
        specs = ["5ml:15mg", "8ml:24mg", "10ml:30mg"]
    elif "栓" in drug_name:
        specs = ["0.5g*7枚/盒", "0.2g*10枚/盒", "1g*6枚/盒"]
    else:
        specs = ["10片/盒", "30片/盒", "100ml/瓶", "50g/瓶"]
    return random.choice(specs)


# ════════════════════════════════════════════
# Institution Training Pair Generation
# ════════════════════════════════════════════
def generate_inst_pairs(kb: list[dict], n: int) -> list[dict]:
    """Generate institution matching training pairs."""
    pairs = []
    # Index by type for efficient lookup
    by_chain = {}
    by_city = {}
    by_district = {}
    for inst in kb:
        by_chain.setdefault(inst.get("chain"), []).append(inst)
        by_city.setdefault(inst["city"], []).append(inst)
        by_district.setdefault(inst["city"] + inst["district"], []).append(inst)

    for _ in range(n):
        target = random.choice(kb)
        candidates = []
        answers = []

        # 1. Add a TRUE match (variant of the same entity)
        true_variants = _gen_inst_variants(target)
        true_cand = random.choice(true_variants)
        candidates.append(true_cand)
        answers.append(_gen_inst_reasoning(target, true_cand, is_match=True, difficulty=random.choice(["easy", "medium"])))

        # 2. Add FALSE matches (similar but different)
        n_false = random.randint(3, 6)
        false_pool = []

        # Same chain, different branch
        if target.get("chain") and len(by_chain.get(target["chain"], [])) > 1:
            same_chain = [x for x in by_chain[target["chain"]] if x["code"] != target["code"]]
            false_pool.extend(same_chain[:5])

        # Same district, different entity
        dist_key = target["city"] + target["district"]
        same_dist = [x for x in by_district.get(dist_key, []) if x["code"] != target["code"]]
        false_pool.extend(same_dist[:5])

        # Same city, different district
        same_city = [x for x in by_city.get(target["city"], []) if x["code"] != target["code"] and x["district"] != target["district"]]
        false_pool.extend(same_city[:5])

        # Random different entities
        random_others = [x for x in kb if x["city"] != target["city"]]
        false_pool.extend(random.sample(random_others, min(5, len(random_others))))

        # Deduplicate
        seen_codes = {target["code"]}
        unique_false = []
        for x in false_pool:
            if x["code"] not in seen_codes:
                seen_codes.add(x["code"])
                unique_false.append(x)

        selected_false = random.sample(unique_false, min(n_false, len(unique_false)))
        for false_cand in selected_false:
            difficulty = "hard" if (false_cand.get("chain") == target.get("chain") and false_cand["city"] == target["city"]) else "medium"
            candidates.append(false_cand)
            answers.append(_gen_inst_reasoning(target, false_cand, is_match=False, difficulty=difficulty))

        # Shuffle candidates and answers together
        combined = list(zip(candidates, answers))
        random.shuffle(combined)
        candidates, answers = zip(*combined)
        candidates, answers = list(candidates), list(answers)

        # Re-index after shuffle
        for i, ans in enumerate(answers):
            ans["index"] = i + 1

        # Build query text
        query_text = random.choice([
            target["standard_name"],
            target.get("full_name", target["standard_name"]),
            target.get("short_name", target["standard_name"]),
        ])

        pair = {
            "query": query_text,
            "candidates": [{"code": c["code"], "name": c.get("standard_name") or c.get("name", "")} for c in candidates],
            "answers": answers,
        }
        pairs.append(pair)

    return pairs


def _gen_inst_variants(inst: dict) -> list[dict]:
    """Generate name variants for a true match."""
    variants = []
    std = inst["standard_name"]

    # Exact match
    variants.append({"code": inst["code"], "name": std})

    # With address
    if inst.get("address"):
        variants.append({"code": inst["code"], "name": f"{std}({inst['address']})"})

    # Full registered name
    if inst.get("full_name"):
        variants.append({"code": inst["code"], "name": inst["full_name"]})

    # Short name
    if inst.get("short_name") and inst["short_name"] != std:
        variants.append({"code": inst["code"], "name": inst["short_name"]})

    # With city prefix removed or added
    city = inst.get("city", "")
    if std.startswith(city):
        variants.append({"code": inst["code"], "name": std[len(city):].lstrip()})

    return variants


def _gen_inst_reasoning(query_inst: dict, cand: dict, is_match: bool, difficulty: str) -> dict:
    """Generate reasoning chain for institution matching."""
    if is_match:
        if difficulty == "easy":
            reasonings = [
                "P1(精确匹配)->判定true",
                "P1(完全一致)->判定true",
            ]
            return {"reasoning": random.choice(reasonings), "matched": True, "confidence": "High"}
        else:
            reasonings = [
                "P1(非精确)->P2(通过,同城同区)->P4(通过,同一机构)->全通过",
                "P1(非精确)->P2(通过)->P5(通过,简写/全称差异)->全通过",
                "P1(非精确)->P2(通过,地理一致)->P6(辅助信息一致,地址匹配)->全通过",
            ]
            return {"reasoning": random.choice(reasonings), "matched": True, "confidence": random.choice(["High", "Medium"])}
    else:
        # False match - different reasons
        q_city = query_inst.get("city", "")
        c_city = cand.get("city", "")
        q_dist = query_inst.get("district", "")
        c_dist = cand.get("district", "")
        q_chain = query_inst.get("chain")
        c_chain = cand.get("chain")

        if q_chain and c_chain and q_chain == c_chain and q_city == c_city and q_dist != c_dist:
            return {"reasoning": f"P1(非精确)->P2(地理冲突:输入{q_dist},候选{c_dist})->判定false", "matched": False, "confidence": "Medium"}
        elif q_chain and c_chain and q_chain == c_chain:
            return {"reasoning": "P1(非精确)->P4(粒度冲突:不同分店)->判定false", "matched": False, "confidence": "Medium"}
        elif q_city == c_city and q_dist != c_dist:
            return {"reasoning": f"P1(非精确)->P2(地理冲突:输入{q_dist},候选{c_dist})->判定false", "matched": False, "confidence": "Low"}
        elif q_city != c_city:
            return {"reasoning": f"P1(非精确)->P2(地理冲突:输入{q_city},候选{c_city})->判定false", "matched": False, "confidence": "Low"}
        else:
            return {"reasoning": "P1(非精确)->核心名冲突(不同实体)->判定false", "matched": False, "confidence": "Low"}


# ════════════════════════════════════════════
# Product Training Pair Generation
# ════════════════════════════════════════════
def generate_prod_pairs(drugs: list[dict], generic_groups: dict, n: int) -> list[dict]:
    """Generate product matching training pairs."""
    pairs = []
    # Only use generic groups with 2+ entries (multi-form drugs for interesting matches)
    multi_form = {k: v for k, v in generic_groups.items() if len(v) >= 2}
    single_form = {k: v for k, v in generic_groups.items() if len(v) == 1}
    all_generics = list(generic_groups.keys())

    for _ in range(n):
        # Pick a target drug
        if random.random() < 0.7 and multi_form:
            gn = random.choice(list(multi_form.keys()))
        else:
            gn = random.choice(all_generics)

        group = generic_groups[gn]
        target = random.choice(group)
        candidates = []
        answers = []

        # 1. A-grade: exact match or same form + same spec
        a_cand = {"code": target["code"], "name": target["standard_name"], "spec": target["spec"]}
        candidates.append(a_cand)
        answers.append({"core_name_match": True, "modifier_diff": "无", "spec_diff": "无", "match_grade": "A"})

        # 2. B-grade: same core, minor spec difference
        if len(group) >= 2:
            same_form = [d for d in group if d["code"] != target["code"]]
            if same_form:
                b_target = random.choice(same_form)
                candidates.append({"code": b_target["code"], "name": b_target["standard_name"], "spec": b_target["spec"]})
                answers.append({
                    "core_name_match": True,
                    "modifier_diff": "无",
                    "spec_diff": f"{target['spec']}vs{b_target['spec']}",
                    "match_grade": "B",
                })

        # 3. C-grade: same core, different form (e.g., capsule vs tablet)
        if len(group) >= 2:
            diff_form = [d for d in group if d["standard_name"] != target["standard_name"] and d["code"] != target["code"]]
            if diff_form:
                c_target = random.choice(diff_form)
                candidates.append({"code": c_target["code"], "name": c_target["standard_name"], "spec": c_target["spec"]})
                answers.append({
                    "core_name_match": True,
                    "modifier_diff": f"剂型差异",
                    "spec_diff": f"{target['spec']}vs{c_target['spec']}",
                    "match_grade": "C",
                })

        # 4. D-grade: different core name
        other_generics = [g for g in all_generics if g != gn]
        for _ in range(random.randint(2, 4)):
            other_gn = random.choice(other_generics)
            other_drug = random.choice(generic_groups[other_gn])
            # Avoid duplicate codes
            if other_drug["code"] not in [c["code"] for c in candidates]:
                candidates.append({"code": other_drug["code"], "name": other_drug["standard_name"], "spec": other_drug["spec"]})
                answers.append({"core_name_match": False, "modifier_diff": "无", "spec_diff": "无", "match_grade": "D"})

        # Shuffle
        combined = list(zip(candidates, answers))
        random.shuffle(combined)
        candidates, answers = zip(*combined)
        candidates, answers = list(candidates), list(answers)

        for i, ans in enumerate(answers):
            ans["index"] = i + 1

        pair = {
            "query_name": target["standard_name"],
            "query_spec": target["spec"],
            "candidates": [{"code": c["code"], "name": c["name"], "spec": c["spec"]} for c in candidates],
            "answers": answers,
        }
        pairs.append(pair)

    return pairs


# ════════════════════════════════════════════
# Format to Messages
# ════════════════════════════════════════════
def format_inst_messages(pair: dict) -> dict:
    """Format institution pair to messages chat format."""
    cand_text = "\n".join(f"[{i+1}] 编码: {c['code']}, 名称: {c['name']}" for i, c in enumerate(pair["candidates"]))

    user_content = f"【输入机构】：{pair['query']}\n【候选机构列表】：\n{cand_text}\n请逐个独立验证并输出JSON数组："

    # Build assistant content (JSON array)
    assistant_items = []
    for ans in pair["answers"]:
        assistant_items.append(ans)
    assistant_content = json.dumps(assistant_items, ensure_ascii=False, indent=2)

    return {
        "messages": [
            {"role": "system", "content": INST_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ]
    }


def format_prod_messages(pair: dict) -> dict:
    """Format product pair to messages chat format."""
    cand_text = "\n".join(
        f"[{i+1}] 编码: {c['code']}, 名称: {c['name']}, 规格: {c['spec']}"
        for i, c in enumerate(pair["candidates"])
    )

    user_content = f"【输入产品】：{pair['query_name']} {pair['query_spec']}\n【候选产品列表】：\n{cand_text}\n请逐个独立验证并输出JSON数组："

    assistant_items = []
    for ans in pair["answers"]:
        assistant_items.append(ans)
    assistant_content = json.dumps(assistant_items, ensure_ascii=False, indent=2)

    return {
        "messages": [
            {"role": "system", "content": PROD_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ]
    }


# ════════════════════════════════════════════
# Main
# ════════════════════════════════════════════
def main():
    output_dir = DOMAIN_ROOT / "data"

    print("=== 生成机构知识库 ===")
    inst_kb = generate_institution_kb()
    print(f"  机构总数: {len(inst_kb)}")
    by_type = {}
    for inst in inst_kb:
        by_type[inst["type"]] = by_type.get(inst["type"], 0) + 1
    for t, c in sorted(by_type.items()):
        print(f"    {t}: {c}")

    print("\n=== 加载产品知识库 ===")
    drugs, generic_groups = load_product_kb()
    print(f"  药品总数: {len(drugs)}")
    print(f"  通用名组: {len(generic_groups)}")
    multi = sum(1 for v in generic_groups.values() if len(v) >= 2)
    print(f"  多剂型组: {multi}")

    print("\n=== 生成训练数据 ===")
    inst_train = generate_inst_pairs(inst_kb, 3000)
    prod_train = generate_prod_pairs(drugs, generic_groups, 3000)
    print(f"  机构匹配: {len(inst_train)} 条")
    print(f"  产品匹配: {len(prod_train)} 条")

    print("\n=== 生成评测数据 ===")
    inst_eval = generate_inst_pairs(inst_kb, 800)
    prod_eval = generate_prod_pairs(drugs, generic_groups, 800)
    print(f"  机构匹配: {len(inst_eval)} 条")
    print(f"  产品匹配: {len(prod_eval)} 条")

    print("\n=== 格式化并保存 ===")
    # Format all data
    inst_train_msgs = [format_inst_messages(p) for p in inst_train]
    prod_train_msgs = [format_prod_messages(p) for p in prod_train]
    inst_eval_msgs = [format_inst_messages(p) for p in inst_eval]
    prod_eval_msgs = [format_prod_messages(p) for p in prod_eval]

    # Save
    train_all = inst_train_msgs + prod_train_msgs
    random.shuffle(train_all)

    eval_inst_path = output_dir / "test" / "eval_institution.json"
    eval_prod_path = output_dir / "test" / "eval_product.json"
    train_path = output_dir / "train" / "train.json"

    for path in [eval_inst_path, eval_prod_path, train_path]:
        path.parent.mkdir(parents=True, exist_ok=True)

    with open(train_path, "w") as f:
        json.dump(train_all, f, ensure_ascii=False, indent=2)
    print(f"  训练集: {train_path} ({len(train_all)} 条)")

    with open(eval_inst_path, "w") as f:
        json.dump(inst_eval_msgs, f, ensure_ascii=False, indent=2)
    print(f"  机构评测: {eval_inst_path} ({len(inst_eval_msgs)} 条)")

    with open(eval_prod_path, "w") as f:
        json.dump(prod_eval_msgs, f, ensure_ascii=False, indent=2)
    print(f"  产品评测: {eval_prod_path} ({len(prod_eval_msgs)} 条)")

    # Also save raw pairs for debugging
    raw_dir = output_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    with open(raw_dir / "inst_kb.json", "w") as f:
        json.dump(inst_kb, f, ensure_ascii=False, indent=2)
    with open(raw_dir / "inst_train_pairs.json", "w") as f:
        json.dump(inst_train, f, ensure_ascii=False, indent=2)
    with open(raw_dir / "prod_train_pairs.json", "w") as f:
        json.dump(prod_train, f, ensure_ascii=False, indent=2)

    print(f"\n=== 完成 ===")
    print(f"训练: {len(train_all)} 条 (机构 {len(inst_train_msgs)} + 产品 {len(prod_train_msgs)})")
    print(f"评测: 机构 {len(inst_eval_msgs)} + 产品 {len(prod_eval_msgs)} = {len(inst_eval_msgs) + len(prod_eval_msgs)} 条")

    # Stats
    inst_matched = sum(1 for p in inst_train for a in p["answers"] if a.get("matched"))
    inst_total_answers = sum(len(p["answers"]) for p in inst_train)
    print(f"\n机构训练正负比例: matched={inst_matched}/{inst_total_answers} ({inst_matched/inst_total_answers:.1%})")

    prod_grades = {"A": 0, "B": 0, "C": 0, "D": 0}
    for p in prod_train:
        for a in p["answers"]:
            prod_grades[a.get("match_grade", "D")] += 1
    prod_total = sum(prod_grades.values())
    print(f"产品训练等级分布: " + " ".join(f"{k}={v}({v/prod_total:.1%})" for k, v in sorted(prod_grades.items())))


if __name__ == "__main__":
    main()
