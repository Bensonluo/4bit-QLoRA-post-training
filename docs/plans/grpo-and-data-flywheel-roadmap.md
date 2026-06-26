# 4bit-QLoRA-post-training：GRPO + 数据飞轮模块实施规划

> 目标：在现有 SFT / DPO 管道基础上，新增 GRPO（Group Relative Policy Optimization）训练能力，并搭建一个可自举的数据飞轮（数据合成 → 偏好生成 → 版本血缘 → 反馈闭环）。

---

## 一、当前架构盘点（已确认）

| 层级 | 现有实现 | 关键文件 |
|------|---------|---------|
| 配置 | dataclass + YAML 组合：`ModelConfig` / `LoRAConfig` / `TrainingConfig` / `DataConfig` / `LoggingConfig` / `DPOConfig` | `config/base.py`, `config/sft.py`, `config/dpo.py` |
| SFT | `SFTTrainer` 包装 HF `Trainer` + QLoRA | `src/training/sft_trainer.py` |
| DPO | `DPOTrainer` 包装 TRL `DPOTrainer` + reference model | `src/training/dpo_trainer.py` |
| 数据 | `BaseDataset` + `AlpacaDataset` / `FinanceDataset` / `PreferenceDataset` | `src/data/base.py`, `src/data/loaders.py` |
| 模型 | 4-bit / 8-bit QLoRA 加载，merge adapter | `src/models/loader.py`, `src/models/merger.py` |
| 追踪 | MLflow tracker + Model Registry + W&B/TensorBoard | `src/tracking/mlflow_tracker.py`, `src/tracking/registry.py` |
| 评估 | perplexity、accuracy、side-by-side generation | `src/evaluation/metrics.py`, `src/evaluation/comparisons.py` |
| 入口 | `scripts/train_sft.py`, `scripts/train_dpo.py`, `scripts/train_domain.py` | `scripts/` |
| UI | Streamlit Training Lab / Experiments / Evaluation | `ui/` |

**关键依赖版本约束**：
- `transformers>=4.36`
- `trl>=0.7.10`
- `peft>=0.7`
- `bitsandbytes>=0.41`

> GRPO 在 TRL 中由 `GRPOTrainer` 提供（TRL ≥ 0.11 较完整，≥ 0.15 支持 `log_completions`、multi-GPU `vllm`、offset bias 等）。因此升级 `trl` 是必要前置。

---

## 二、GRPO 模块设计

### 2.1 核心概念映射

| GRPO 概念 | 项目中的对应 |
|-----------|-------------|
| Policy | 当前 LoRA adapter  attached 的 LLM |
| Reference model | SFT checkpoint（或 base model） |
| Group sampling | 对同一 prompt 生成 `num_generations` 个 responses |
| Reward function | 可插拔：rule-based + LLM-as-a-Judge + metric-based |
| Advantage | group 内 reward 归一化：`(r - mean(r)) / std(r)` |
| Loss | GRPO objective = policy gradient + KL penalty |

### 2.2 新增配置

新增 `config/grpo.py`：

```python
@dataclass
class GRPOConfig:
    beta: float = 0.04                       # KL penalty coefficient
    num_generations: int = 8                 # group size G
    max_completion_length: int = 512         # 生成长度
    use_vllm: bool = False                   # 是否用 vLLM 加速采样
    vllm_device: str | None = None           # vLLM 专用 device
    reward_funcs: list[str] = field(default_factory=lambda: ["format", "accuracy"])
    judge_model: str | None = None           # LLM-as-a-Judge 模型
    judge_prompt_template: str | None = None

@dataclass
class GRPOTrainingConfig:
    model_config: ModelConfig
    training_config: TrainingConfig
    lora_config: LoRAConfig
    grpo_config: GRPOConfig
    reference_config: ModelConfig | None      # 默认复用 SFT checkpoint
    logging_config: LoggingConfig
```

`config/base.py` 扩展：
- `DataConfig.format` 增加 `"grpo"` 合法值。

### 2.3 新增模块与文件布局

```
src/
├── training/
│   ├── grpo_trainer.py          # GRPOTrainer 封装，协调模型、采样、奖励
│   └── reward_engine.py         # 奖励函数注册与组合
├── data/
│   └── grpo_dataset.py          # GRPO 数据集加载与格式化
├── generation/
│   ├── __init__.py
│   ├── sampler.py               # group sampling（HF generate / vLLM）
│   └── judge_client.py          # LLM-as-a-Judge 客户端（本地 / API）
└── evaluation/
    └── grpo_metrics.py          # reward 分布、KL、response 长度等指标
```

### 2.4 GRPOTrainer 实现要点

选择两条路（建议分阶段）：

**阶段 A：基于 TRL `GRPOTrainer`（推荐）**
- 升级 `trl>=0.15`。
- 直接复用 `GRPOTrainer`，传入：
  - `model`：带 LoRA 的 policy
  - `ref_model`：参考模型（默认加载 SFT adapter merged 后的 checkpoint）
  - `reward_funcs`：`src/training/reward_engine.py` 中注册的函数列表
  - `args`：HF `TrainingArguments`
  - `train_dataset` / `eval_dataset`

**阶段 B：自实现 GRPO loop（如果 TRL 不满足）**
- 仅当需要自定义 GAE、PPO-style 变体或离线 RL 时考虑。
- 成本更高，暂不建议。

### 2.5 Reward Engine 设计

```python
# src/training/reward_engine.py
REWARD_REGISTRY: dict[str, Callable] = {}

def register(name: str):
    def decorator(fn: Callable):
        REWARD_REGISTRY[name] = fn
        return fn
    return decorator

@register("format")
def format_reward(completions: list[str], **kwargs) -> list[float]:
    """检查 JSON/Markdown/特定标签格式。"""
    ...

@register("accuracy")
def accuracy_reward(completions: list[str], answer: str, **kwargs) -> list[float]:
    """与标准答案比对（如 exact match / fuzzy match）。"""
    ...

@register("llm_judge")
def judge_reward(completions: list[str], prompt: str, judge_client, **kwargs) -> list[float]:
    """LLM-as-a-Judge 打分。"""
    ...

def get_reward_functions(names: list[str]) -> list[Callable]:
    return [REWARD_REGISTRY[n] for n in names]
```

奖励函数签名统一：
```python
Callable[[list[str], dict[str, Any]], list[float]]
```
输入 `completions`（长度 = group size），输出等长 reward list。

### 2.6 数据格式

新增 `GRPODataset` 支持两种输入格式：

**格式 1：带标准答案（rule reward）**
```json
{
  "prompt": "计算 23 * 47",
  "answer": "1081"
}
```

**格式 2：带参考 response（judge reward）**
```json
{
  "prompt": "...",
  "reference": "..."
}
```

**格式 3：纯 prompt（无监督 reward / judge only）**
```json
{
  "prompt": "..."
}
```

`GRPOTrainer` 的 dataset 需要字段：
- `prompt`：string or list of token ids
- 可选 `answer`, `reference` 作为 reward function 的额外输入

### 2.7 与现有管道集成

1. **CLI 入口**：`scripts/train_grpo.py`， Typer app，类似 `train_sft.py`。
2. **训练流程**：
   - 加载 policy（SFT checkpoint + LoRA）
   - 加载 reference model（SFT merged checkpoint，冻结）
   - 加载 GRPO dataset
   - 配置 `GRPOTrainer` + `TrainingArguments` + MLflow callback
   - 训练、保存 adapter、注册模型
3. **MLflow 追踪**：
   - 记录 `grpo_config` 参数
   - 记录 reward mean/std、KL、response length 等指标
   - 保存 sample completions 为 artifact
4. **模型注册**：复用 `register_trained_model`，注册 GRPO 训练后的 adapter。

### 2.8 分布式与显存

- GRPO 需要同时加载 policy + reference，并做 group generation，显存压力 ≈ DPO 的 1.5~2 倍。
- 推荐：
  - 单卡/小显存：QLoRA + gradient checkpointing + `num_generations=4`
  - 多卡：FSDP full_shard 或 DeepSpeed ZeRO-3
  - 加速采样：`use_vllm=True`（TRL ≥ 0.15 支持）

---

## 三、数据飞轮模块设计

### 3.1 模块定位

数据飞轮负责：
1. **数据合成**：基于 seed / 知识库生成新的训练样本。
2. **偏好生成**：基于当前 policy 生成 responses，产出 preference pairs。
3. **数据集版本与血缘**：追踪每条数据从哪来、由哪个模型/配置生成。
4. **反馈闭环**：评估失败样本回流，自动触发重生成或加入下一轮训练。

### 3.2 新增文件布局

```
src/
├── data_flywheel/
│   ├── __init__.py
│   ├── schemas.py              # DatasetItem / PreferencePair / LineageRecord 数据模型
│   ├── synthesizer.py          # 数据合成器（seed expansion, evol-instruct, self-instruct）
│   ├── preference_builder.py   # 由 group completions 生成 preference pairs
│   ├── dataset_registry.py     # 本地/MLflow 数据集注册、版本、hash
│   ├── judge.py                # LLM-as-a-Judge 判定 + 偏好评分
│   ├── miner.py                # 从评估失败/低分样本中挖掘 bad cases
│   └── pipeline.py             # 端到端飞轮 orchestration
```

### 3.3 数据模型

```python
# src/data_flywheel/schemas.py
from dataclasses import dataclass
from datetime import datetime

@dataclass
class DatasetItem:
    id: str
    prompt: str
    response: str | None
    metadata: dict
    source: str                # "synthetic", "human", "grpo_generated"
    lineage_id: str
    created_at: datetime

@dataclass
class PreferencePair:
    id: str
    prompt: str
    chosen: str
    rejected: str
    judge_model: str | None
    reward_chosen: float
    reward_rejected: float
    lineage_id: str
    generation_policy: str     # 生成该 pair 的模型 checkpoint/run_id

@dataclass
class LineageRecord:
    lineage_id: str
    parent_lineage_ids: list[str]
    operation: str             # "synthesize", "grpo_sample", "judge", "filter"
    config: dict
    run_id: str | None         # MLflow run_id
    input_hash: str
    output_hash: str
    timestamp: datetime
```

### 3.4 数据合成器（Synthesizer）

支持多种策略，可开关：

| 策略 | 说明 | 适用场景 |
|------|------|---------|
| `evol_instruct` | 用 LLM 改写/加深 seed instruction | 通用指令增强 |
| `self_instruct` | 用 LLM 基于 seed 生成新 instruction | 扩充覆盖 |
| `qa_from_kb` | 从知识库生成 QA pairs | 领域适配 |
| `backtranslation` | answer → question 生成 | 闭卷问答 |

```python
class DataSynthesizer:
    def __init__(self, model_client, strategy: str, config: dict):
        ...

    def synthesize(self, seed_items: list[DatasetItem]) -> list[DatasetItem]:
        ...
```

### 3.5 偏好生成器（Preference Builder）

输入：prompt + group completions + rewards
输出：`PreferencePair` 列表

```python
class PreferenceBuilder:
    def __init__(self, min_margin: float = 0.0, top_k: int = 1):
        ...

    def build(
        self,
        prompt: str,
        completions: list[str],
        rewards: list[float],
        policy_run_id: str,
    ) -> list[PreferencePair]:
        # 排序后取 top/bottom 构造 pair
        ...
```

### 3.6 数据集注册表（Dataset Registry）

两种后端：
- **本地**：JSONL + manifest JSON，基于内容 hash 去重/版本。
- **MLflow**：用 `mlflow.data` 或自定义 artifact 记录 dataset version 与 run 关联。

核心接口：
```python
class DatasetRegistry:
    def register(self, name: str, items: list[DatasetItem], lineage: LineageRecord) -> str:
        """返回 dataset version id"""
        ...

    def load(self, name: str, version: str | None = None) -> list[DatasetItem]:
        ...

    def get_lineage(self, version_id: str) -> LineageRecord:
        ...
```

### 3.7 评估失败样本回流（Bad Case Miner）

```python
class BadCaseMiner:
    def __init__(self, judge_client, threshold: float = 0.3):
        ...

    def mine(
        self,
        eval_results: list[dict],
        policy_model: str,
    ) -> list[DatasetItem]:
        """从评估结果中挖出需要重训的 bad cases。"""
        ...
```

### 3.8 飞轮 Pipeline

```python
# src/data_flywheel/pipeline.py
class DataFlywheelPipeline:
    def __init__(
        self,
        synthesizer: DataSynthesizer,
        preference_builder: PreferenceBuilder,
        registry: DatasetRegistry,
        judge: JudgeClient,
        miner: BadCaseMiner,
    ):
        ...

    def run_iteration(
        self,
        seed_data: list[DatasetItem],
        policy_path: str,
        n_synthetic: int = 100,
        n_grpo_samples: int = 100,
    ) -> dict:
        """一次飞轮迭代：
        1. 合成新 SFT 数据
        2. 用当前 policy 采样 group completions
        3. 计算 rewards，生成 preference pairs
        4. 注册数据集，记录血缘
        5. 返回 {sft_dataset_version, dpo_dataset_version}
        """
        ...
```

---

## 四、端到端训练流程

```
Seed Data
   │
   ▼
SFT (现有) ──────────────► SFT Checkpoint
   │
   ▼
GRPO ──► Policy aligned by rewards
   │
   ├──────► Save adapter + Register model
   │
   ▼
Data Flywheel
   ├─ 采样 group completions
   ├─ 奖励打分
   ├─ 生成 preference pairs
   └─ 注册 DPO/GRPO 下一轮数据集
   │
   ▼
DPO / GRPO 迭代训练
```

### 推荐默认训练流水线

1. **SFT**：在 domain seed 数据上微调，得到 domain SFT checkpoint。
2. **GRPO Round 1**：用 rule/judge rewards 做第一轮对齐。
3. **Data Flywheel**：基于 GRPO policy 生成 preference data + synthetic SFT data。
4. **DPO Round 1**：在生成的 preference data 上继续优化。
5. **迭代**：评估 → bad case mining → 重新生成 → GRPO/DPO。

---

## 五、实施阶段（推荐顺序）

### Phase 0：依赖升级与验证（1-2 天）

- [ ] 升级 `trl>=0.15`，验证 SFT/DPO 现有脚本仍可运行。
- [ ] 验证 `transformers`/`peft` 与 `trl` 版本兼容（必要时升级）。
- [ ] 添加 `vllm` 为可选依赖（`[grpo]` extras）。

### Phase 1：Reward Engine + GRPOTrainer（3-5 天）

- [ ] 新增 `config/grpo.py`。
- [ ] 实现 `src/training/reward_engine.py`（含 `format`, `accuracy`, `llm_judge` 示例）。
- [ ] 实现 `src/data/grpo_dataset.py`。
- [ ] 实现 `src/training/grpo_trainer.py`（基于 TRL GRPOTrainer）。
- [ ] 新增 `scripts/train_grpo.py` CLI。
- [ ] 单元测试：reward engine、GRPO dataset formatting、config validation。

### Phase 2：数据飞轮核心（4-6 天）

- [ ] 实现 `src/data_flywheel/schemas.py`。
- [ ] 实现 `src/data_flywheel/judge.py`（本地模型 / OpenAI API 兼容接口）。
- [ ] 实现 `src/data_flywheel/synthesizer.py`（至少 `evol_instruct` + `self_instruct`）。
- [ ] 实现 `src/data_flywheel/preference_builder.py`。
- [ ] 实现 `src/data_flywheel/dataset_registry.py`（本地 backend + MLflow backend）。
- [ ] 实现 `src/data_flywheel/miner.py`。
- [ ] 实现 `src/data_flywheel/pipeline.py`。

### Phase 3：评估与血缘闭环（2-3 天）

- [ ] 新增 `src/evaluation/grpo_metrics.py`。
- [ ] 扩展 MLflow tracker 记录 dataset version（`log_dataset`）。
- [ ] 扩展 registry：注册模型时附带训练数据版本血缘。
- [ ] 实现 `scripts/run_flywheel_iteration.py`。

### Phase 4：UI 与示例（2-3 天）

- [ ] Streamlit 新增 "GRPO Training" 页面。
- [ ] Streamlit 新增 "Data Flywheel" 页面（展示 lineage、合成状态、偏好对）。
- [ ] 添加 finance / medical_entity 的 GRPO 示例配置。
- [ ] 补充 `docs/tutorials/grpo_training.md` 和 `docs/tutorials/data_flywheel.md`。

### Phase 5：分布式与性能优化（2-4 天）

- [ ] 验证 FSDP 下 GRPOTrainer 运行。
- [ ] 可选集成 `vLLM` 加速 group sampling。
- [ ] 实现 sampling cache / reuse 降低重复生成成本。
- [ ] 性能 benchmark：SFT vs DPO vs GRPO 显存/耗时对比。

---

## 六、关键依赖变更

```toml
[project.dependencies]
# 升级 trl 到支持 GRPOTrainer 的版本
"trl>=0.15.0"

[project.optional-dependencies]
grpo = [
    "vllm>=0.6.0",            # 可选，加速采样
]

flywheel = [
    "jsonlines>=4.0",
    "xxhash>=3.4",            # 快速内容 hash
]

all = [
    "qlora-post-training[dev,notebook,ui,grpo,flywheel]"
]
```

---

## 七、风险与回退方案

| 风险 | 影响 | 回退方案 |
|------|------|---------|
| `trl` 升级破坏现有 DPO/SFT | 高 | 先 pin `trl` 版本在独立分支验证；保留旧 trainer 接口兼容 |
| GRPO 显存不足 | 高 | 减小 `num_generations`、用更小模型、启用 CPU offloading |
| LLM-as-a-Judge 不稳定 | 中 | 增加 rule-based reward 权重；judge 结果做温度/重复采样平均 |
| 数据合成质量差 | 中 | 引入人工 seed curation + 多样性过滤 + dedup |
| 数据血缘依赖 MLflow | 低 | registry 本地 backend 作为默认，MLflow 可选 |

---

## 八、文件变更清单（预估）

**新增**：
- `config/grpo.py`
- `src/training/grpo_trainer.py`
- `src/training/reward_engine.py`
- `src/data/grpo_dataset.py`
- `src/generation/sampler.py`
- `src/generation/judge_client.py`
- `src/evaluation/grpo_metrics.py`
- `src/data_flywheel/` 全部
- `scripts/train_grpo.py`
- `scripts/run_flywheel_iteration.py`
- `docs/plans/grpo-and-data-flywheel-roadmap.md`（本文档）
- `docs/tutorials/grpo_training.md`
- `docs/tutorials/data_flywheel.md`
- 对应测试文件

**修改**：
- `pyproject.toml`：升级 `trl`，新增 extras
- `config/base.py`：`DataConfig.format` 增加 `"grpo"`
- `config/__init__.py`：导出 GRPO 配置
- `src/tracking/mlflow_tracker.py`：增加 dataset 日志方法
- `src/tracking/registry.py`：注册时记录数据版本
- `ui/app.py` / `ui/pages/`：新增页面入口

---

## 九、立即可以开始的最小可验证步骤

1. 在独立分支升级 `trl` 到 0.15+，跑通现有 `train_sft.py` 和 `train_dpo.py` 的 quick test。
2. 新增 `config/grpo.py` 和最小 `GRPOTrainer` 包装，用单条 prompt + `accuracy` reward 跑通一次训练循环。
3. 实现 `reward_engine.py` 并写单元测试验证奖励函数注册与组合。
4. 用上述结果生成第一批 preference pairs，验证数据飞轮 schemas 与 registry 的写入/读取。

---

*文档版本：2026-06-26*
*基于 4bit-QLoRA-post-training 当前 main 分支架构规划*
