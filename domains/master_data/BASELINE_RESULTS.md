# 主数据匹配基线评测结果

**评测时间**: 2026-05-29 ~ 2026-05-30
**评测样本**: 400条/任务（institution + product），云端 API 模型；本地模型 50条。31b 为 50条。
**评测脚本**: `domains/master_data/eval/evaluate.py`
**本地推理**: LM Studio (port 1234)，并发=2

---

## 机构匹配 (Institution Matching)

| 模型 | 规模 | 样本 | Top-1 | 准确率 | Precision | Recall | F1 | 解析失败 | 延迟 |
|------|------|------|-------|--------|-----------|--------|-----|---------|------|
| **gemma-4-26b** | 26B | 50 | **90.0%** | 98.1% | 100.0% | 90.0% | 94.7% | 0 | 12.6s |
| GLM-5.1 | 云端 | 400 | **85.2%** | 97.1% | 97.2% | 86.5% | 91.5% | 0 | 5.9s |
| gemma-4-31b | 31B | 50 | 88.0% | 97.7% | 100.0% | 88.0% | 93.6% | 0 | 46.9s* |
| MiniMax-M2.7 | 云端 | 400 | **83.8%** | 95.2% | 87.3% | 86.0% | 86.6% | 6 | 9.3s |
| qwen3.6-35b | 35B | 50 | 76.0% | 95.4% | 100.0% | 76.0% | 86.4% | 0 | 9.4s |
| qwen3-30b | 30B | 50 | 72.0% | 94.6% | 100.0% | 72.0% | 83.7% | 0 | 12.5s |
| **qwen3-8b** | **8B** | **50** | **62.0%** | **89.3%** | **75.0%** | **66.0%** | **70.2%** | **2** | **7.7s** |

\* 31b 并发 2 实际为串行排队（单条约 19s），延迟不具备可比性。

### 洞察
- **gemma-4-26b 机构 Top-1 = 90%**，超越 GLM-5.1 (88%)，为当前所有评测模型最高
- GLM / gemma-26b / gemma-31b / qwen3.6-35b / qwen3-30b Precision 均为 100%：判断为 true 的全对
- 独立判定准确率（89-98%）远高于 Top-1 选择准确率（62-90%）→ 模型"能看出来"但不会"做选择"
- qwen3-8b Precision=75%：不仅漏判，还会错判 false 为 true

---

## 产品匹配 (Product Matching)

| 模型 | 样本 | Top-1 | 等级准确率 | 核心名准确率 | A级 | B级 | D级 | 解析失败 | 延迟 |
|------|------|-------|-----------|-------------|-----|-----|-----|---------|------|
| gemma-4-31b | 50 | 100.0% | 99.2% | 99.2% | 100% | 96% | 100% | 0 | 47.9s* |
| qwen3-8b | 50 | 100.0% | 99.2% | 99.2% | 100% | 96% | 100% | 0 | 10.5s |
| gemma-4-26b | 50 | 100.0% | 98.8% | 98.8% | 100% | 94% | 100% | 0 | 11.2s |
| MiniMax-M2.7 | 400 | **99.8%** | **98.8%** | **98.8%** | 100% | 95% | 100% | 0 | 5.8s |
| qwen3.6-35b | 50 | 100.0% | 98.8% | 98.8% | 100% | 94% | 100% | 0 | 9.0s |
| GLM-5.1 | 400 | **100.0%** | **98.0%** | **98.0%** | 100% | 90% | 100% | 0 | 4.0s |
| qwen3-30b | 50 | 100.0% | 95.7% | 96.9% | 94% | 84% | 100% | 0 | 5.9s |
| qwen3-30b | 50 | 100.0% | 95.7% | 96.9% | 94% | 84% | 100% | 0 | 5.9s |

\* 31b 并发 2 实际为串行排队，延迟不具备可比性。

### 洞察
- 所有模型 Top-1 = 100% → 核心名一票否决机制非常有效
- B级（剂型/规格差异）是唯一失分点；qwen3-30b B级仅 84%，其余均 ≥92%
- **产品匹配已天花板，微调无意义**

---

## 微调目标

| 任务 | 当前 qwen3-8b | 目标 (GLM) | 当前最佳 (gemma-26b) | 差距 |
|------|--------------|-----------|---------------------|------|
| 机构 Top-1 | 62.0% | 85.2% | **90.0%** | +28% |
| 机构 Precision | 75.0% | 97.2% | 100.0% | +25% |
| 机构 Recall | 66.0% | 86.5% | **90.0%** | +24% |
| 产品 Top-1 | 100.0% | 100.0% | 100.0% | 0% |

**结论**: 机构匹配是唯一有意义的微调目标。gemma-4-26b 已证明本地 26B 模型可达 90% Top-1。

---

## 原始结果文件位置

### 2026-05-29 基线
- `domains/master_data/eval/results_glm_50.txt`
- `domains/master_data/eval/results_minimax_50.txt`
- `domains/master_data/eval/results_local_50.txt` (qwen3-8b)
- `domains/master_data/eval/results_35b_50.txt` (qwen3.6-35b)
- `domains/master_data/data/results/eval_20260529_152139.json` (GLM 逐条)
- `domains/master_data/data/results/eval_20260529_152145.json` (8B 逐条)
- `domains/master_data/data/results/eval_20260529_152255.json` (MiniMax 逐条)
- `domains/master_data/data/results/eval_20260529_153850.json` (35B 逐条)

### 2026-05-30 补充评测
- `domains/master_data/data/results/eval_20260530_015152.json` (qwen3-30b product, 50条)
- `domains/master_data/data/results/eval_20260530_015720.json` (qwen3-30b institution, 50条)
- `domains/master_data/data/results/eval_20260530_023020.json` (gemma-4-26b product, 50条)
- `domains/master_data/data/results/eval_20260530_023603.json` (gemma-4-26b institution, 50条)
- `domains/master_data/data/results/eval_20260530_180151.json` (GLM-5.1 product, 400条)
- `domains/master_data/data/results/eval_20260530_180749.json` (MiniMax-M2.7 product, 400条)
- `domains/master_data/data/results/eval_20260530_182741.json` (GLM-5.1 institution, 400条)
- `domains/master_data/data/results/eval_20260530_183914.json` (MiniMax-M2.7 institution, 400条)
- `domains/master_data/data/results/eval_20260530_101925.json` (gemma-4-31b institution, 50条)
- `domains/master_data/data/results/eval_20260530_103923.json` (gemma-4-31b product, 50条)
