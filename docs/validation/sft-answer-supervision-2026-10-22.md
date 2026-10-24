# SFT 答案监督：2026-10-22 真实本地机制实验

本轮完成了一次三臂受控实验和一次真实产品训练集成验证。去除第一版数据中的 ID 捷径后，三臂都能在开发题上完整生成到 EOS；仅答案监督的 Alpaca 臂答对 16/16，完整提示监督和原生聊天仅答案监督均为 12/16。独立的产品训练适配器在产品默认推理精度下答对 12/16，基座为 0/16 且全部达到长度上限。

这些是**合成任务的机制证据**：说明本地模型在明确的小任务上，经真实更新可以学会完整作答，并验证产品实际消费了指定的答案标签。它们不是业务成功、最终验收或北极星指标达标的证据，也没有证明旧 R146 财报任务零分的根因。

## 可审阅证据

- [自包含逐题原始证据 JSON](sft-answer-supervision-live-2026-10-22.json)：v2 数据、配置、所有题目的提示词、输入和生成 token IDs、原始输出、EOS/截断/评分、产品 token proof、来源及适配器哈希、独立审计摘要。
- [完整无权重原始补件](sft-answer-supervision-evidence-2026-10-22.tar.gz)：235,763 bytes，74 个 manifest 文件条目，含实际执行的原脚本、完整训练标签、逐步 loss/LR、真实 collator 消费记录、日志、产品源文件快照，以及保留的 v1 与失败记录。原本地包逐字节复制，未裁剪文件。
- 补件 SHA-256：`89f5ccbacbee62ff4114edd3bf5953b08d1b23b2f78eac1ee27435e7a4dfaaf6`。包内 `manifest.json` 记录脱敏后各文件的大小和哈希；自包含 JSON 同时列出所引用原始输出文件的哈希与路径。

所有数据均为本轮合成，包内用户主目录和仓库绝对路径分别替换为 `<USER_HOME>`、`<REPO_ROOT>`。不包含基座、适配器或优化器权重。实际本地基座文件的运行后 SHA-256 清单、各适配器 SHA-256 保留在证据中；这是运行后内容身份记录，不冒充运行前后双向校验。

## 问题和实验边界

旧的两条财报训练样例和两次更新不足以回答“监督目标或提示格式是否能让模型完整作答”。本轮使用一个更明确、可核验的小任务：从 `signal` 字段学习四个任意处理码映射，然后输出短分类标签或严格 JSON。映射为 `amber→P4`、`birch→K7`、`cobalt→Q9`、`dahlia→M2`，不在提示词中给出。允许的四个标签在提示词中明确列出。

两种输出契约：

1. 分类：只输出一个处理码。评分对整个原始输出去除首尾空白后做精确比较，不抽取子串。
2. JSON：整个输出必须能解析成对象，恰好有 `route`、`ticket` 两个字符串字段，值与参考一致。拒绝重复键、解释文字和代码围栏，不从输出中提取对象。

最终答对还要求真实生成了 tokenizer EOS，且未被长度上限截断。分类上限 16 个新 token，JSON 上限 48 个；采用 greedy、单 beam，不采样，不继承模型仓库额外 generation-config 处理器。保存实际生成 token IDs，以 `skip_special_tokens=True` 解码供评分；没有补 EOS、答案修复或生成后提取。

没有读取现有业务 holdout。这里的 16 题称为**开发验证**，不是业务保留集或独立业务泛化测试。四个映射、单模型、单种子的规模不能支持普遍方法优越性结论。

## v1 混淆和 v2 数据冻结

第一版有 32 条训练、8 条开发数据，但工单号按类别索引构造，十位数携带类别。其好成绩不能证明模型使用了 `signal` 字段。发现后完整保留 v1 的脚本、数据、结果和适配器哈希，标为 `known_identifier_confound`，没有覆盖或择优报告。

v2 在重跑前冻结如下设计，**没有根据 v1 成绩更改学习率、种子或更新步数**：

- 训练 32 条 = 4 个 signal × 2 种输出形式 × 4 组干扰项组合。
- 每个 signal、每种输出形式都出现相同的 `TK-1000` 至 `TK-1003` 和相应备注模板；ID、备注和语体均不携带类别信息。
- 开发 16 条 = 4 个 signal × 2 种输出形式 × 2 组未见组合，使用 `TK-9000`、`TK-9001` 和两种新备注；同一开发 ID/备注跨每个 signal 和两种输出形式出现。
- 训练与开发完整提示不重复，ID 集合分开。组内重复 ID 是刻意构造的反事实配对，不能把这些行视作独立真实工单。

独立审计核对了提示不重复及每个 `(form, signal)` 的干扰项集合完全相等。v2 的全数据已放入自包含 JSON；完整逐条 token/label 见补件中的 `outputs/validation/r148/v2/supervision.json`。

## 三臂受控方法

使用本机已有完整 Qwen3-1.7B，Apple M3 Max / 64 GB / MPS，Python 3.12.13、PyTorch 2.11.0、Transformers 5.9.0、PEFT 0.19.1。没有下载模型、安装依赖、修改基座或已有训练产物。

三臂分别为：

- `alpaca_full_prompt_v1`：原始 Alpaca 提示，监督提示和答案。
- `alpaca_answer_only_v1`：相同 Alpaca 提示和完整输入 token，只监督答案及原本的 EOS。
- `native_chat_answer_only_v1`：实际本地 tokenizer 原生聊天模板，显式 `enable_thinking=False`，只监督答案和模板原本的 EOS；这是实验臂。

三臂均从同一基座重新加载，以 seed 42 初始化相同的 LoRA 参数；可训练初始张量哈希均为 `a26c0de5d49c5266ba3c7f94e5d3f4dbb26a06794a293f59cf6946c14de410cf`。配置为 r8、alpha16、dropout0、all-linear、bias none、FP32、batch1、64 次真实 AdamW 更新、常量 LR `2e-4`、warmup0、weight_decay0、梯度裁剪1。32 条训练样本各两遍，三臂逐步样本顺序完全相同。

训练编码最大长度 192，不截断。三臂使用显式 labels，右侧补齐到 8 的倍数；padding labels 为 `-100`。不是让另一个 collator 从 input IDs 重建监督。统计按 causal shift 后的有效标签计算：Alpaca 完整监督每行 98–130 个，两个仅答案监督臂为 3–16 个。原始完整输入最长 Alpaca 131、原生聊天 135 个 token。

每行都证明“推理提示的 token IDs 是完整训练输入的严格前缀”，随后才确定答案边界；不根据字符长度猜测 token 边界。答案 EOS 必须真实保留并受监督。当前 Qwen3 原生模板的 assistant EOS 后还有一个换行，该 token 在原生臂的 label 是 `-100`，没有把尾换行误当成答案，也没有额外附加第二个 EOS。

先在基座上分别评估原始 Alpaca 和原生非思考聊天提示；两个 Alpaca 监督臂共享同一基座提示及基线。再依次训练和评估三臂。脚本始终只有一个重模型进程，不把并发资源竞争混入对照。

## 三臂真实结果

下表均为 v2。EOS 和格式列按全部 16 题统计；答对同时要求分数正确、EOS、未截断。

| 阶段与协议 | 分类答对 | JSON 答对 | 合计答对 | EOS 完整 | 格式有效 | 截断 |
|---|---:|---:|---:|---:|---:|---:|
| 基座 / 原始 Alpaca | 0/8 | 0/8 | 0/16 | 0/16 | 0/16 | 16/16 |
| 基座 / 原生非思考聊天 | 2/8 | 0/8 | 2/16 | 16/16 | 16/16 | 0/16 |
| 训练 / Alpaca 完整提示监督 | 6/8 | 6/8 | 12/16 | 16/16 | 16/16 | 0/16 |
| 训练 / Alpaca 仅答案监督 | 8/8 | 8/8 | 16/16 | 16/16 | 16/16 | 0/16 |
| 训练 / 原生聊天仅答案监督 | 6/8 | 6/8 | 12/16 | 16/16 | 16/16 | 0/16 |

该数据上，完整提示监督也学会完整结束，因此不能声称“只有新增仅答案监督才会停止”。原生聊天基座本来就能完整输出，但映射准确率低，说明格式完整与学习任务正确是两个维度。

三臂 loss 监督范围不同，不能横向比较数值大小。完整提示监督、Alpaca 仅答案监督、原生仅答案监督的前 8 步/末 8 步平均 loss 分别为 `3.3601→0.07377`、`3.5864→0.0003012`、`2.0945→0.000002394`。原生臂极低的训练 loss 并没有得到更高开发准确率；报告不以 loss 替代原始生成评分。

## 真实产品训练和推理集成

独立使用 v2 的同 32 条训练和同 16 条开发 JSON，实际调用产品 `SFTTrainer.prepare_model()`、`prepare_data()`、`setup_trainer()`、`train()`。设置 `DataConfig(format='alpaca', dataset_loader='alpaca', sft_protocol='alpaca_answer_only_v1', validation_split=0)`，并通过单独 `validation_file` 加载开发集。

训练仍为 r8/all-linear/FP32/seed42、batch1、accumulation1、2 epochs 共 64 次更新、warmup0、constant LR `2e-4`。保留真实 TrainingArguments、Trainer state、完整 loss/LR 日志、适配器内容哈希及保存的训练产物。逐行比较产品准备的 IDs/labels 和协议编码；在实际产品 collator 外加观察包装，**不改其输出 batch**。记录首个完整 batch 及全部实际消费行标签计数，确认 32 条训练各消费两次、每行答案 EOS 受监督、提示及 padding label 为 `-100`。

真实框架在最后一步自动进行 16 条开发的 teacher-forced 评估和 checkpoint；这不是额外优化更新。最后开发 loss `0.167728`，训练平均 loss `0.695552`，均不作为生成准确率。此次没有通过反复查看分数选择参数。

训练完成释放模型后，分别用真实 `LocalEvaluationRuntime(enforce_token_contract=True)` 加载原基座和这次适配器。两者保留产品默认精度，测得首个基座参数均为 `torch.bfloat16`；没有手动 `.float()`。同一 16 题用相同 Alpaca 提示、greedy、16/48 上限，逐题核对输入 token IDs 与 EOS IDs 完全一致。原始输出与完整 token proof 均保存。

| 产品实际推理对象 | 分类答对 | JSON 答对 | 合计答对 | EOS 完整 | 格式有效 | 截断 |
|---|---:|---:|---:|---:|---:|---:|
| 原基座，默认 BF16 基座参数 | 0/8 | 0/8 | 0/16 | 0/16 | 0/16 | 16/16 |
| 本次产品训练适配器，默认 BF16 基座参数 | 6/8 | 6/8 | 12/16 | 16/16 | 16/16 | 0/16 |

这是产品集成 smoke，不是三臂的第四个严格对称重复：产品由框架采样，样本顺序与显式实验不同；产品固定长度 padding 与受控动态 padding 不同；产品默认推理的基座精度为 BF16，而受控三臂推理是 FP32。没有通过本轮实验拆分这些因素，不能把 `16/16→12/16` 归因于其中任何一个。

产品运行时的源文件快照**早于随后 warmup_ratio=1 上界语义的修复**。原源文件 SHA 和内容快照保留在补件；本次测量使用 `warmup_ratio=0`，后续修复保留其零 warmup 语义。没有把后修源文件冒充此次实际执行版本，也没有为此重新推理。Git 中新增的复现脚本将使用运行时 checkout 的源文件并重新记录哈希，未来输出不应冒充原快照的逐位复现。

## 失败和成本完整保留

首次受控启动在导入 `src` 时失败，尚未加载模型。原日志保留为 `outputs/validation/r148/run-initial-import-error.log`，修正脚本仓库路径后才完成 v1。

首次产品训练确实完成 64 步并保存适配器，但实验脚本错误地断言 `train()` 返回前只消费了训练行。产品最后自动评估了 16 条开发数据，所以该取证断言失败。第一次完整目录、日志和原脚本保留为 `product-sft-first-instrumentation-failure*`。修正为按 split 计数、实时保存消费记录后，按原参数重新训练；未调 LR、种子、步数或评分。该失败不是产品训练失败，也不被隐去当成一次无成本成功。

| 运行 | 完成运行墙钟秒 | 训练秒 | 生成秒 | 说明 |
|---|---:|---:|---:|---|
| v1，已知 ID 混淆 | 109.563 | 61.771 | 29.027 | 不作为干净结论 |
| v2，平衡干扰项三臂 | 113.867 | 53.331 | 48.731 | 含三臂及两种基座提示 |
| 成功产品集成 | 56.408 | 22.592 | 逐题保留 | 训练秒含产品结束处理；框架 train_runtime 为 22.2504 秒 |
| 首次产品取证断言失败 | 未单独记录完整墙钟 | 框架 train_runtime 22.5591 | 未运行 | 已完成真实训练，后置取证失败 |

完整墙钟还包含模型加载、保存、编码和释放，不能与训练/生成部分强行等同。未使用实际电表，电量保持 `null`，没有估算成真实测量。首次导入失败的成本未单独测量。成功运行数字不包括失败尝试；不把这些表格值当作整个迭代的总成本。

## 复现与默认行为

使用已存在的完整本地模型和一个全新输出目录：

```sh
./venv/bin/python -B scripts/validate_sft_supervision.py \
  --model-path /absolute/path/to/local/Qwen3-1.7B \
  --output-dir outputs/validation/sft-supervision-new \
  --steps 64

./venv/bin/python -B scripts/validate_sft_supervision_smoke.py \
  --model-path /absolute/path/to/local/Qwen3-1.7B \
  --data outputs/validation/sft-supervision-new/data.json \
  --output-dir outputs/validation/sft-supervision-product-new
```

[三臂脚本](../../scripts/validate_sft_supervision.py)从当前 checkout 的 `src/data/sft_protocol.py` 复制协议快照到新输出并记录 SHA。[产品脚本](../../scripts/validate_sft_supervision_smoke.py)按脚本所在仓库计算产品源文件哈希，不依赖调用者当前目录。两者要求明确的本地模型和输出路径，拒绝覆盖现有输出；仅当前进程开启离线加载，不安装依赖或下载模型。第二个脚本固定相同产品配置，要求前一个脚本生成的 32/16 数据。

Git 中脚本是实际输出脚本的路径参数化版本，进行了 `--help`、Ruff、编译检查，没有因整理交付再次运行模型。原始实际执行脚本仍保留在补件和本地输出中。

产品默认仍是 `alpaca_full_prompt_v1`。`alpaca_answer_only_v1` 需要明确选择；`native_chat_answer_only_v1` 只用于本轮实验，不因模型名猜测 thinking 开关，也不作为用户默认训练配置。这里依据实际本地 Qwen3 tokenizer 模板和严格前缀证明来构建原生标签；不能把该模型的模板规律推广到任意 tokenizer。相关官方方法背景见 [Transformers chat templating](https://huggingface.co/docs/transformers/v5.9.0/en/chat_templating)、[TRL SFTTrainer](https://huggingface.co/docs/trl/sft_trainer) 和 [Qwen3-1.7B 模型说明](https://huggingface.co/Qwen/Qwen3-1.7B)；本轮分数和标签结论以保存的本地实际执行证据为依据。

## 自动检查

最终源码全量 `venv/bin/python -m pytest --no-cov -q`：**2715 passed, 169 warnings, 326.99s**。场景矩阵 **47/47 as_expected**，Ruff、372 份 Python 文件格式及 diff 检查通过。新增训练回归包含实际 toy CE/梯度、真正的独立 CPU worker、协议在方案/迭代/技术重试中的继承、原始 EOS 与左右 padding，以及按 HF 实际更新总数计算 warmup 比例 0、0.03、0.5、1。最后方案 token 消费摘要仅统计训练分区；未知答案边界保持 `null`，不把保留集统计用作监督目标选择的依据。

全仓 mypy 仍有 **301 项存量错误**。对照 R147 的 329 项，按文件及诊断内容比较并忽略行列号、保留重复计数：新增 0、消除 28；不能宣称类型检查全绿。原始日志在 `outputs/validation/r148/pytest-verified.log`、`mypy-verified.log`、`scenario-matrix.json`。

本轮没有测量真实用户成功率、业务价值或北极星指标，没有完成业务最终验收。下一次真实业务验证需要独立定义任务契约、足够且干净的训练/开发数据、冻结同题基线与保留集；不能继承这里的合成分数作为业务评分。
