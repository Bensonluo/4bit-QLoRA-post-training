# 主数据匹配 — 核心业务逻辑

**⚠️ 必读：方向错了，任何努力全是白费**

---

## 一、业务目标（唯一核心）

> **给定一个查询（机构名 / 产品名+规格）+ N 个候选，从中选出唯一正确匹配的那个。**

- **不是**多标签分类
- **不是**对每个候选独立打分的准确率竞赛
- **是** "N 选 1" 的精确选择问题

---

## 二、机构匹配

### 2.1 模型输出

对每个候选输出：
- `matched`: true / false
- `confidence`: High / Medium / Low
- `reasoning`: 判定理由（可选）

### 2.2 选1逻辑（后处理）

```python
best_inst = None
for res in results:
    if res["matched"]:
        if best_inst is None or priority_map[res["confidence"]] > priority_map[best_inst["confidence"]]:
            best_inst = res
```

- 从 `matched=true` 的候选中，按 confidence 优先级选出最佳匹配
- confidence 优先级：**High > Medium > Low**
- 每个样本 ground truth 中**有且仅有 1 个** `matched=true`

### 2.3 核心评测指标

| 指标 | 说明 | 优先级 |
|---|---|---|
| **Top-1 选择准确率** | 选中的候选 index 与 ground truth 一致 | ⭐ 唯一核心 |
| 独立判定准确率 | 每个候选的 matched 判得对不对 | 辅助参考 |
| Precision / Recall / F1 | 二分类统计 | 辅助参考 |

---

## 三、产品匹配

### 3.1 模型输出

对每个候选输出：
- `match_grade`: A / B / D
- `core_name_match`: true / false
- `modifier_diff`: 修饰词差异描述
- `spec_diff`: 规格差异描述

### 3.2 选1逻辑（后处理）

```python
grade_to_score = {"A": 95, "B": 75, "D": 0}
best_prod = None
max_score = -1
for res in results:
    current_score = grade_to_score.get(res["match_grade"], 0)
    if current_score > max_score:
        max_score = current_score
        best_prod = res
```

- 按等级分数选出最高分候选
- 等级优先级：**A(95) > B(75) > D(0)**
- 每个样本 ground truth 中**有且仅有 1 个** A 级（完全匹配）

### 3.3 核心评测指标

| 指标 | 说明 | 优先级 |
|---|---|---|
| **Top-1 选择准确率** | 选中的候选 index 与 ground truth 一致 | ⭐ 唯一核心 |
| 等级准确率 | 每个候选的 A/B/D 判得对不对 | 辅助参考 |
| 核心名准确率 | core_name_match 对不对 | 辅助参考 |
| A/B/D 各级准确率 | 分等级统计 | 辅助参考 |

---

## 四、为什么独立判定准确率不是核心

**反例**：模型可能把 5 个候选中的 4 个都判对了（80% 独立准确率），但**唯一正确的那个 A 级被它判成了 D**。Top-1 选择完全失败，业务上就是错的。

独立判定准确率只反映模型对每个候选的理解能力，**不反映**它能否在候选中定位正确答案。

---

## 五、评测脚本使用规范

```bash
# 评测脚本已内置选1逻辑，核心输出指标为 Top-1 选择准确率
python -m domains.master_data.eval.evaluate \
  --local-model qwen3-8b \
  --task both \
  --max-samples 100
```

输出结果中 **"Top-1选择"** 列是唯一需要关注的指标，其余列仅作参考。

---

## 六、微调目标

| 任务 | Top-1 目标 | 对标基准 |
|---|---|---|
| 机构匹配 | ≥ 94.8% | 超过 GLM-5.1 |
| 产品匹配 | ≥ 99.6% | 超过 qwen3-8b |
| 产品 B级 | ≥ 98% | 核心难点指标 |

> 所有目标均以 **Top-1 选择准确率** 为准，不是独立判定准确率。
