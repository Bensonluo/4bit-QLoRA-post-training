# 主数据匹配项目参考文档

## 项目结构

```
domains/master_data/
├── prompts/                          # Prompt 模板（最核心）
│   ├── institution_matching.md       # 机构匹配 System Prompt + User Prompt 格式
│   └── product_matching.md           # 产品匹配 System Prompt + User Prompt 格式
├── scripts/
│   └── generate_data.py              # 数据生成脚本
├── data/
│   ├── train/train.json              # 训练集 (6000条: 机构3000+产品3000)
│   ├── test/
│   │   ├── eval_institution.json     # 机构评测集 (800条)
│   │   └── eval_product.json         # 产品评测集 (800条)
│   └── raw/                          # 原始知识库和对
├── eval/                             # 评测脚本
└── REFERENCE.md                      # 本文件
```

## 数据格式 (Messages Chat Format)

```json
{
  "messages": [
    {
      "role": "system",
      "content": "角色：专业的医药机构主数据匹配审核员..."
    },
    {
      "role": "user",
      "content": "【输入机构】：北京协和医院\n【候选机构列表】：\n[1] 编码: 1001, 名称: 中国医学科学院北京协和医院\n[2] 编码: 1002, 名称: 北京协和医学院"
    },
    {
      "role": "assistant",
      "content": "[{\"index\": 1, \"reasoning\": \"P1(通过)->...\", \"matched\": true, \"confidence\": \"High\"}, ...]"
    }
  ]
}
```

## 两个任务的输出格式

### 机构匹配
```json
{"index": 1, "reasoning": "P1(通过)->P2(冲突:输入A区,候选B区)->判定false", "matched": false, "confidence": "Low"}
{"index": 2, "reasoning": "P1(通过)->P2(通过)->P3(通过)->全通过", "matched": true, "confidence": "High"}
```

### 产品匹配
```json
{"index": 1, "core_name_match": false, "modifier_diff": "无", "spec_diff": "无", "match_grade": "D"}
{"index": 2, "core_name_match": true, "modifier_diff": "无", "spec_diff": "10片装vs12片装", "match_grade": "B"}
```

## 数据统计

| 数据集 | 条数 | 说明 |
|---|---|---|
| 训练集 | 6,000 | 机构 3000 + 产品 3000 |
| 机构评测 | 800 | 全难度 |
| 产品评测 | 800 | A/B/D 各级 |

机构训练正负比例: matched=18.3% (每个查询约 1 个 true + 4-6 个 false)
产品训练等级分布: A/B/D 三级（简化后）

## 机构知识库来源

- 合成数据，基于 15 个城市、20+ 连锁药店品牌
- 覆盖：连锁药店(171)、单体药店(88)、医院(31)、社区中心(30)
- 总计 320 个机构实体

## 产品知识库来源

- 复用 `domains/medical_entity/data/drug_knowledge_base.json`
- 14,101 个药品，9,926 个通用名组，2,415 个多剂型组
- 规格（spec）为合成数据，基于剂型生成合理规格

## Prompt 模板位置

- 机构匹配: `domains/master_data/prompts/institution_matching.md`
- 产品匹配: `domains/master_data/prompts/product_matching.md`

这两个文件包含完整的 System Prompt、User Prompt 格式、候选列表格式、结果解析逻辑。

## 训练方案

- 基座模型：Qwen3.5-4B
- 数据格式：messages chat format（非 Alpaca）
- 一个模型混合训练（靠 system prompt 区分任务）
- 无验证集，跳过 eval
- 评测集各 800 条
