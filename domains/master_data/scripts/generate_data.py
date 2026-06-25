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
    "3. 不需要计算具体分数，只需根据差异情况判定匹配等级（A/B/D）。\n"
    "4. 必须先提取核心名，再比对修饰词，最后比对规格。\n"
    "判定流程（对每个候选独立执行）：\n"
    "Step 1 — 一票否决（核心名称一致性）：\n"
    "去除修饰词，提取核心产品名。若核心名不一致，直接判定为 D级。\n"
    "Step 2 — 差异提取与定级（仅在核心名一致时执行）：\n"
    "比对修饰词（材质、方法、品牌、型号）和规格（尺寸、包装数、容量），判定匹配等级：\n"
    "- A级：核心名一致，且修饰词完全一致，规格完全一致。\n"
    "- B级：核心名一致，但修饰词或规格存在差异（如剂型不同、剂量不同、数量不同等）。\n"
    "- D级：核心名不一致。\n"
    "输出要求：\n"
    "严格输出标准JSON数组，数组长度必须与候选列表一致。不要输出任何思考过程或其他字符。\n"
    '格式：\n'
    '[\n'
    '  {"index": 1, "core_name_match": false, "modifier_diff": "无", "spec_diff": "无", "match_grade": "D"},\n'
    '  {"index": 2, "core_name_match": true, "modifier_diff": "剂型差异", "spec_diff": "0.25g*24片/盒vs0.5g*20粒/盒", "match_grade": "B"}\n'
    ']'
)

# ── Geography Data (expanded to 30 cities) ──
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
    "昆明市": {"province": "云南省", "districts": ["五华区", "盘龙区", "官渡区", "西山区", "呈贡区", "晋宁区", "东川区"]},
    "济南市": {"province": "山东省", "districts": ["历下区", "市中区", "槐荫区", "天桥区", "历城区", "长清区", "章丘区"]},
    "哈尔滨市": {"province": "黑龙江省", "districts": ["道里区", "南岗区", "道外区", "平房区", "松北区", "香坊区", "呼兰区"]},
    "长春市": {"province": "吉林省", "districts": ["南关区", "宽城区", "朝阳区", "二道区", "绿园区", "双阳区", "九台区"]},
    "石家庄市": {"province": "河北省", "districts": ["长安区", "桥西区", "新华区", "裕华区", "井陉矿区", "藁城区", "鹿泉区"]},
    "贵阳市": {"province": "贵州省", "districts": ["南明区", "云岩区", "花溪区", "乌当区", "白云区", "观山湖区", "开阳县"]},
    "南宁市": {"province": "广西壮族自治区", "districts": ["兴宁区", "青秀区", "江南区", "西乡塘区", "良庆区", "邕宁区", "武鸣区"]},
    "福州市": {"province": "福建省", "districts": ["鼓楼区", "台江区", "仓山区", "马尾区", "晋安区", "长乐区", "闽侯县"]},
    "合肥市": {"province": "安徽省", "districts": ["瑶海区", "庐阳区", "蜀山区", "包河区", "长丰县", "肥东县", "肥西县"]},
    "南昌市": {"province": "江西省", "districts": ["东湖区", "西湖区", "青云谱区", "青山湖区", "新建区", "红谷滩区", "南昌县"]},
    "太原市": {"province": "山西省", "districts": ["小店区", "迎泽区", "杏花岭区", "尖草坪区", "万柏林区", "晋源区", "清徐县"]},
    "兰州市": {"province": "甘肃省", "districts": ["城关区", "七里河区", "西固区", "安宁区", "红古区", "永登县", "皋兰县"]},
    "海口市": {"province": "海南省", "districts": ["秀英区", "龙华区", "琼山区", "美兰区"]},
    "银川市": {"province": "宁夏回族自治区", "districts": ["兴庆区", "西夏区", "金凤区", "永宁县", "贺兰县", "灵武市"]},
    "西宁市": {"province": "青海省", "districts": ["城东区", "城中区", "城西区", "城北区", "湟中区", "大通县", "湟源县"]},
}

# ── Pharmacy Chain Data (expanded with real brands) ──
PHARMACY_CHAINS = [
    "大参林", "一心堂", "老百姓大药房", "国大药房", "海王星辰",
    "益丰大药房", "桐君阁大药房", "华氏大药房", "成大方圆",
    "众康大药房", "健客大药房", "同仁堂", "雷允上",
    "养和堂", "保和堂", "济民大药房", "康泽大药房",
    "好药师大药房", "仁和药房", "九洲大药房",
    "健之佳", "漱玉平民大药房", "怡康医药", "张仲景大药房",
    "吉林大药房", "一树药业", "贵州一品药业", "昌盛大药房",
    "重庆和平药房", "重庆万和药房", "重庆鑫斛药房",
    "养天和大药房", "千金大药房", "达嘉维康",
    "楚济堂", "诺舟大药房", "恒康大药房",
    "龙马药业", "东飞药业", "杏林医药", "泉源堂",
    "正和祥", "全泰堂", "马应龙大药房", "宜草堂",
    "用心人大药房", "同济堂", "南京医药国药",
    "百佳惠瑞丰", "大众医药", "康济大药房",
    "震元医药", "英特集团", "华通医药",
    "人民同泰医药", "鑫世一医药", "齐泰医药",
    "德生堂", "佛慈大药房", "普济堂", "康宁医药",
    "桂中大药房", "一心药业", "康全药业",
    "鹭燕医药", "嘉事堂", "德信行", "圆心科技",
    "爱心大药房", "百源堂", "佛心医药", "中智大药房",
    "南北药行", "燕喜堂", "信宏仁", "同方药业",
    "医保城", "幸福人大药房", "葆春堂",
    "乡亲大药房", "咸阳百姓乐", "乐榕融",
    "高济长坂坡", "吴都药业",
    "开开心心大药房", "布衣大药房",
    "天士力", "天益堂", "百和堂",
    "汇仁堂", "洪兴大药房", "赣州昌盛",
]

HOSPITAL_TYPES = ["人民医院", "中心医院", "第一医院", "第二医院", "第三医院", "中医院", "妇幼保健院", "肿瘤医院", "骨科医院", "眼科医院", "口腔医院", "儿童医院", "胸科医院", "脑科医院"]

STREETS = [
    "中山路", "解放路", "建设路", "人民路", "和平路", "光明路",
    "长江路", "黄河路", "文化路", "民主路", "新华路", "胜利路",
    "前进路", "幸福路", "健康路", "朝阳路", "学府路", "科技路",
    "青年路", "东风路", "工业路", "友谊路", "建国路", "复兴路",
    "迎宾路", "环湖路", "滨海路", "广场路", "花园路", "林荫路",
    "金水路", "银海路", "锦绣路", "春华路", "秋实路", "冬梅路",
    "夏荷路", "松柏路", "梧桐路", "银杏路", "樱花路", "枫叶路",
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

        # Chain pharmacies: 4-6 chains per city, 3-6 branches per chain
        n_chains = min(random.randint(4, 6), len(PHARMACY_CHAINS))
        for chain in random.sample(PHARMACY_CHAINS, n_chains):
            n_branches = random.randint(3, 6)
            for _ in range(n_branches):
                dist = random.choice(districts)
                street = random.choice(STREETS)
                branch_suffix = random.choice([
                    f"({dist}{street}店)",
                    f"({dist}店)",
                    f"第{random.randint(1, 500)}分店",
                    f"({street}店)",
                    f"({dist}{random.choice(['旗舰店', '中心店', '总店', '形象店'])})",
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

        # Independent pharmacies: 5-8 per district
        n_dist_indep = min(random.randint(5, 8), len(districts))
        for dist in random.sample(districts, n_dist_indep):
            n_indep = random.randint(2, 4)
            for _ in range(n_indep):
                street = random.choice(STREETS)
                prefixes = ["康乐", "济民", "仁心", "安康", "健民", "祥和", "德心", "益康", "惠民",
                           "福康", "瑞康", "华康", "同康", "顺康", "宁康", "永乐", "广济", "博爱",
                           "平安", "万寿", "长青", "永安", "仁和", "正和", "德和", "泰和",
                           "民生", "康泰", "康宁", "康复", "康健", "康源", "康达", "康盛",
                           "济世", "济生", "济康", "济安", "济仁", "济众", "济华", "济民",
                           "仁德", "仁济", "仁术", "仁义", "仁厚", "仁善", "仁美", "仁诚",
                           "安泰", "安和", "安宁", "安怡", "安瑞", "安祥", "安顺", "安盛",
                           "健安", "健泰", "健宁", "健和", "健生", "健源", "健达", "健丰",
                           "祥瑞", "祥和", "祥泰", "祥康", "祥安", "祥宁", "祥盛", "祥乐",
                           "德馨", "德润", "德华", "德泰", "德和", "德宁", "德安", "德康",
                           "益民", "益生", "益康", "益泰", "益和", "益安", "益宁", "益达",
                           "惠康", "惠安", "惠民", "惠和", "惠宁", "惠泰", "惠达", "惠生"]
                suffixes = ["药店", "药房", "大药房", "医药商店", "平价药房", "连锁药店", "药品超市", "医保药店"]
                # 50% 概率加区名前缀，大幅降低同城市内重名概率
                if random.random() < 0.5:
                    name = f"{dist}{random.choice(prefixes)}{random.choice(suffixes)}"
                else:
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

        # Hospitals: 3-5 per city
        n_hospitals = min(random.randint(3, 5), len(HOSPITAL_TYPES))
        for htype in random.sample(HOSPITAL_TYPES, n_hospitals):
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

        # Community health centers: 3-5 per district
        n_dist_comm = min(random.randint(3, 5), len(districts))
        for dist in random.sample(districts, n_dist_comm):
            n_comm = random.randint(2, 3)
            for _ in range(n_comm):
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
def load_product_kb() -> tuple[list[dict], dict, dict]:
    """Load drug KB, add synthetic specs, and create per-standard_name variants for B-grade.

    Results are cached to disk to avoid re-processing 17k+ drugs on every run.

    Returns:
        drugs: list of all drug entries with specs assigned
        generic_groups: {generic_name: [drug, ...]}
        std_variants: {standard_name: [drug_variant, ...]} including synthetic variants
    """
    cache_path = DOMAIN_ROOT / "data" / "raw" / "product_kb_processed.json"
    if cache_path.exists():
        with open(cache_path) as f:
            cached = json.load(f)
        return cached["drugs"], cached["generic_groups"], cached["std_variants"]

    kb_path = MEDICAL_ENTITY_ROOT / "data" / "drug_knowledge_base.json"
    with open(kb_path) as f:
        raw = json.load(f)

    drugs = raw["drugs"]
    generic_groups = {}
    std_variants = {}  # standard_name -> list of variants (for B-grade)

    for drug in drugs:
        gn = drug["generic_name"]
        if gn not in generic_groups:
            generic_groups[gn] = []
        name = drug["standard_name"]
        spec = _gen_spec(name)
        drug["spec"] = spec
        generic_groups[gn].append(drug)

        if name not in std_variants:
            std_variants[name] = []
        std_variants[name].append(drug)

    # Generate synthetic spec variants for B-grade (same standard_name, different spec)
    for name, variants in std_variants.items():
        original = variants[0]
        existing_specs = {v["spec"] for v in variants}
        # Try to generate 2 additional variants with different specs
        attempts = 0
        while len(variants) < 3 and attempts < 10:
            attempts += 1
            new_spec = _gen_spec(name)
            if new_spec not in existing_specs:
                new_drug = {
                    "code": f"VAR{len(variants)}_{original['code']}",
                    "standard_name": name,
                    "generic_name": original["generic_name"],
                    "spec": new_spec,
                }
                variants.append(new_drug)
                existing_specs.add(new_spec)
                # Also add to generic_groups so C-grade can see them
                generic_groups[original["generic_name"]].append(new_drug)

    # Save cache
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump({"drugs": drugs, "generic_groups": generic_groups, "std_variants": std_variants}, f, ensure_ascii=False)

    return drugs, generic_groups, std_variants


def _extract_formulation(drug_name: str) -> str | None:
    """Extract formulation type from drug name (e.g. '片', '胶囊', '注射液')."""
    formulations = ["注射液", "口服液", "滴眼液", "胶囊", "颗粒", "片剂", "片", "丸", "散", "膏", "栓"]
    for f in formulations:
        if f in drug_name:
            return f
    return None


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
    used_queries = set()
    # Index by type for efficient lookup
    by_chain = {}
    by_city = {}
    by_district = {}
    for inst in kb:
        by_chain.setdefault(inst.get("chain"), []).append(inst)
        by_city.setdefault(inst["city"], []).append(inst)
        by_district.setdefault(inst["city"] + inst["district"], []).append(inst)

    max_attempts = n * 10
    attempts = 0
    while len(pairs) < n and attempts < max_attempts:
        attempts += 1
        target = random.choice(kb)
        candidates = []
        answers = []

        # 1. Add a TRUE match (variant of the same entity)
        true_variants = _gen_inst_variants(target)
        true_cand = random.choice(true_variants)
        candidates.append(true_cand)
        answers.append(_gen_inst_reasoning(target, true_cand, is_match=True, difficulty=random.choice(["easy", "medium"])))

        # 2. Add FALSE matches with分层 difficulty
        n_false = random.randint(3, 6)

        # Collect candidates by difficulty tier
        # Heavy hard: same chain, different branch (same city)
        hard_heavy = []
        if target.get("chain") and len(by_chain.get(target["chain"], [])) > 1:
            hard_heavy = [x for x in by_chain[target["chain"]]
                          if x["code"] != target["code"] and x["city"] == target["city"]]

        # Light hard: same district different entity OR same city different district
        hard_light = []
        dist_key = target["city"] + target["district"]
        same_dist = [x for x in by_district.get(dist_key, []) if x["code"] != target["code"]]
        same_city_diff_dist = [x for x in by_city.get(target["city"], [])
                               if x["code"] != target["code"] and x["district"] != target["district"]]
        hard_light = same_dist + same_city_diff_dist

        # Easy: different city (completely unrelated)
        easy_pool = [x for x in kb if x["city"] != target["city"]]

        # Deduplicate each tier
        seen_codes = {target["code"]}
        seen_names = {target.get("standard_name", "")}

        def dedup(pool, _seen_codes=seen_codes, _seen_names=seen_names):
            out = []
            for x in pool:
                if x["code"] not in _seen_codes and x.get("standard_name", "") not in _seen_names:
                    _seen_codes.add(x["code"])
                    _seen_names.add(x.get("standard_name", ""))
                    out.append(x)
            return out

        hard_heavy = dedup(hard_heavy)
        hard_light = dedup(hard_light)
        easy_pool = dedup(easy_pool)

        # Stratified sampling for vector-retrieval-like distribution:
        # ~40% heavy hard (same chain/very similar) / ~50% light hard (same city) / ~10% easy (different city)
        # Vector recall rarely returns completely unrelated items
        n_heavy = min(max(1, int(n_false * 0.4)), len(hard_heavy)) if hard_heavy else 0
        n_light = min(n_false - n_heavy, len(hard_light)) if hard_light else 0
        n_easy = max(0, n_false - n_heavy - n_light)
        # Ensure at least 1 easy if pool exists (simulates occasional bad recall)
        if n_easy == 0 and easy_pool and n_false >= 4:
            n_easy = 1
            if n_light > n_heavy and n_light > 1:
                n_light -= 1
            elif n_heavy > 1:
                n_heavy -= 1

        selected_false = []
        if n_heavy > 0 and hard_heavy:
            selected_false.extend(random.sample(hard_heavy, min(n_heavy, len(hard_heavy))))
        if n_light > 0 and hard_light:
            selected_false.extend(random.sample(hard_light, min(n_light, len(hard_light))))
        if n_easy > 0 and easy_pool:
            selected_false.extend(random.sample(easy_pool, min(n_easy, len(easy_pool))))

        for false_cand in selected_false:
            is_heavy = (false_cand.get("chain") == target.get("chain") and false_cand["city"] == target["city"])
            difficulty = "hard" if is_heavy else ("medium" if false_cand["city"] == target["city"] else "easy")
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

        # Build query text - always prefix with city to avoid cross-city overlap
        # Institution names (especially independent pharmacies) may not include city
        # in their standard_name, so we must add it for the query.
        # short_name may not include city, so prefix it to prevent train/eval leakage.
        short = target.get("short_name", "")
        if short and not short.startswith(target["city"]):
            short = f"{target['city']}{short}"
        query_variants = [
            f"{target['city']}{target['standard_name']}",
            target.get("full_name", target["standard_name"]),
            short if short else f"{target['city']}{target['standard_name']}",
            f"{target['district']}{target['standard_name']}",
        ]
        # Ensure query uniqueness: try variants in order, skip target if all used
        query_text = None
        for variant in query_variants:
            if variant and variant not in used_queries:
                query_text = variant
                break
        if query_text is None:
            continue  # All variants used for this target, skip and pick another
        used_queries.add(query_text)

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
def generate_prod_pairs(drugs: list[dict], generic_groups: dict, std_variants: dict, n: int, exclude_queries: set | None = None) -> list[dict]:
    """Generate product matching training pairs (3-level: A/B/D).

    A-grade: exact match (same standard_name, same spec).
    B-grade: same generic group (same core name), any difference in standard_name or spec.
    D-grade: different generic group (different core name).
    """
    pairs = []
    multi_form = {k: v for k, v in generic_groups.items() if len(v) >= 2}
    all_generics = list(generic_groups.keys())
    exclude_queries = exclude_queries or set()

    # Pre-compute indices to avoid O(N) scans inside the pair loop
    gn_formulation = {}
    formulation_to_gns = {}
    prefix_to_gns = {}   # 2-char prefix -> list of generic names
    length_to_gns = {}   # length -> list of generic names
    for g in all_generics:
        form = _extract_formulation(generic_groups[g][0]["standard_name"])
        gn_formulation[g] = form
        if form:
            formulation_to_gns.setdefault(form, []).append(g)
        if len(g) >= 2:
            prefix_to_gns.setdefault(g[:2], []).append(g)
        length_to_gns.setdefault(len(g), []).append(g)

    for _ in range(n):
        # Pick a target drug (prefer multi-form groups for B variety)
        # Retry if query is in exclude_queries
        target = None
        for _attempt in range(100):
            if random.random() < 0.7 and multi_form:
                gn = random.choice(list(multi_form.keys()))
            else:
                gn = random.choice(all_generics)
            group = generic_groups[gn]
            candidate_target = random.choice(group)
            q = candidate_target["standard_name"] + " " + candidate_target["spec"]
            if q not in exclude_queries:
                target = candidate_target
                break
        if target is None:
            # Fallback: pick any target
            gn = random.choice(all_generics)
            group = generic_groups[gn]
            target = random.choice(group)

        candidates = []
        answers = []
        used_codes = {target["code"]}

        # 1. A-grade: exact match
        a_cand = {"code": target["code"], "name": target["standard_name"], "spec": target["spec"]}
        candidates.append(a_cand)
        answers.append({"core_name_match": True, "modifier_diff": "无", "spec_diff": "无", "match_grade": "A"})

        # 2. B-grade: same generic group (same core name), any difference
        b_pool = [d for d in group if d["code"] != target["code"] and d["code"] not in used_codes]
        if b_pool:
            b_target = random.choice(b_pool)
            used_codes.add(b_target["code"])
            # Determine modifier_diff based on whether standard_name differs
            mod_diff = "剂型差异" if b_target["standard_name"] != target["standard_name"] else "无"
            candidates.append({"code": b_target["code"], "name": b_target["standard_name"], "spec": b_target["spec"]})
            answers.append({
                "core_name_match": True,
                "modifier_diff": mod_diff,
                "spec_diff": f"{target['spec']}vs{b_target['spec']}",
                "match_grade": "B",
            })

        # 3. D-grade: different core name, but simulate vector retrieval (mostly similar-looking)
        other_generics = [g for g in all_generics if g != gn]

        # Build similarity tiers for D-grade selection using character Jaccard similarity
        # Heavy hard: high character overlap (vector recall would return these)
        target_chars = set(gn)
        similarities = []
        for g in other_generics:
            g_chars = set(g)
            inter = len(target_chars & g_chars)
            union = len(target_chars | g_chars)
            if union > 0:
                sim = inter / union
                if sim >= 0.25:  # at least 25% character overlap for hard negative
                    similarities.append((sim, g))
        # Sort by similarity descending, take top 60 as heavy hard pool
        similarities.sort(reverse=True)
        heavy_hard_set = set(g for sim, g in similarities[:60])
        heavy_hard_set.discard(gn)

        # Also include same-formulation-type similarity using pre-computed mapping
        target_formulation = gn_formulation.get(gn)
        if target_formulation and target_formulation in formulation_to_gns:
            for g in formulation_to_gns[target_formulation]:
                if g != gn and g not in heavy_hard_set:
                    heavy_hard_set.add(g)
                    if len(heavy_hard_set) >= 80:
                        break

        # Light hard: moderate similarity (same length range ±2)
        light_hard_set = set()
        for length in range(len(gn) - 2, len(gn) + 3):
            for g in length_to_gns.get(length, []):
                if g != gn and g not in heavy_hard_set:
                    light_hard_set.add(g)
        # Keep only those with some character overlap (>= 10%) to avoid complete randoms
        light_hard_set = {g for g in light_hard_set if len(target_chars & set(g)) / max(len(target_chars | set(g)), 1) >= 0.1}

        # Easy: completely unrelated (low or no character overlap)
        easy_set = set(other_generics) - heavy_hard_set - light_hard_set

        heavy_hard_gn = list(heavy_hard_set)
        light_hard_gn = list(light_hard_set)
        easy_gn = list(easy_set)

        n_d = random.randint(2, 4)
        # Vector retrieval distribution: ~40% heavy / ~50% light / ~10% easy
        n_d_heavy = min(max(1, int(n_d * 0.4)), len(heavy_hard_gn)) if heavy_hard_gn else 0
        n_d_light = min(n_d - n_d_heavy, len(light_hard_gn)) if light_hard_gn else 0
        n_d_easy = max(0, n_d - n_d_heavy - n_d_light)
        # Optional: ensure at least 1 easy candidate per sample for variety
        # Disabled to prioritize hard negatives (realistic vector retrieval)
        # if n_d_easy == 0 and easy_gn and n_d >= 5:
        #     n_d_easy = 1
        #     if n_d_light > n_d_heavy and n_d_light > 1:
        #         n_d_light -= 1
        #     elif n_d_heavy > 1:
        #         n_d_heavy -= 1

        d_selected = []
        for _gns, _n in [(heavy_hard_gn, n_d_heavy), (light_hard_gn, n_d_light), (easy_gn, n_d_easy)]:
            pool = [d for g in _gns for d in generic_groups[g] if d["code"] not in used_codes]
            if _n > 0 and pool:
                picks = random.sample(pool, min(_n, len(pool)))
                for d in picks:
                    used_codes.add(d["code"])
                    d_selected.append(d)

        for d_drug in d_selected:
            candidates.append({"code": d_drug["code"], "name": d_drug["standard_name"], "spec": d_drug["spec"]})
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
    drugs, generic_groups, std_variants = load_product_kb()
    print(f"  药品总数: {len(drugs)}")
    print(f"  通用名组: {len(generic_groups)}")
    multi = sum(1 for v in generic_groups.values() if len(v) >= 2)
    print(f"  多剂型组: {multi}")
    n_std_with_variants = sum(1 for v in std_variants.values() if len(v) >= 2)
    print(f"  有规格变体的标准名: {n_std_with_variants}")

    print("\n=== 生成训练数据 ===")
    # Split cities into train/eval groups to ensure zero query overlap for institutions
    all_cities = list(CITIES.keys())
    train_cities = set(random.sample(all_cities, 20))
    eval_cities = set(all_cities) - train_cities
    print(f"  训练城市: {len(train_cities)} 个, 评测城市: {len(eval_cities)} 个")

    inst_kb_train = [x for x in inst_kb if x['city'] in train_cities]
    inst_kb_eval = [x for x in inst_kb if x['city'] in eval_cities]
    print(f"  训练机构KB: {len(inst_kb_train)} 条, 评测机构KB: {len(inst_kb_eval)} 条")

    inst_train = generate_inst_pairs(inst_kb_train, 3000)
    prod_train = generate_prod_pairs(drugs, generic_groups, std_variants, 3000)
    print(f"  机构匹配: {len(inst_train)} 条")
    print(f"  产品匹配: {len(prod_train)} 条")

    print("\n=== 生成评测数据 ===")
    inst_eval = generate_inst_pairs(inst_kb_eval, 800)

    # Product eval: exclude training queries directly in generator
    prod_train_queries = {p['query_name'] + ' ' + p['query_spec'] for p in prod_train}
    prod_eval = generate_prod_pairs(drugs, generic_groups, std_variants, 800, exclude_queries=prod_train_queries)

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

    print("\n=== 完成 ===")
    print(f"训练: {len(train_all)} 条 (机构 {len(inst_train_msgs)} + 产品 {len(prod_train_msgs)})")
    print(f"评测: 机构 {len(inst_eval_msgs)} + 产品 {len(prod_eval_msgs)} = {len(inst_eval_msgs) + len(prod_eval_msgs)} 条")

    # Stats
    inst_matched = sum(1 for p in inst_train for a in p["answers"] if a.get("matched"))
    inst_total_answers = sum(len(p["answers"]) for p in inst_train)
    print(f"\n机构训练正负比例: matched={inst_matched}/{inst_total_answers} ({inst_matched/inst_total_answers:.1%})")

    prod_grades = {"A": 0, "B": 0, "D": 0}
    for p in prod_train:
        for a in p["answers"]:
            prod_grades[a.get("match_grade", "D")] += 1
    prod_total = sum(prod_grades.values())
    print("产品训练等级分布: " + " ".join(f"{k}={v}({v/prod_total:.1%})" for k, v in sorted(prod_grades.items())))


if __name__ == "__main__":
    main()
