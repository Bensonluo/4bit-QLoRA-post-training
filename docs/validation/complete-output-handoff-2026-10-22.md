# 完整输出到 HF / GGUF 交接：2026-10-22 真实 workbench 集成证据

本轮从普通零 key intake 入口开始，经过真实业务确认、全量验证、分组物化、盲标门禁、固定题集和正式 workbench worker，完成了一次 64 更新的训练。随后从真实成功运行导出合并 HF 和 Q4_K_M GGUF，并对原基座、适配器、合并 HF、GGUF 在同一组开发题上实际生成、独立评分。

基座 8 题均达到 16 个新 token 上限，未完整结束；适配器、合并 HF、GGUF 均为 8/8。三个训练衍生模型逐题生成的 token IDs 完全相同，都是两个标签 token 加真实 EOS `151645`。**原报告保留 `completed_with_failures`**，没有删除基座失败行或改成全成功状态。

这是开发者编写的合成集成证据。开发集只有两个 ticket 组、每组四个反事实行，不能称为八个独立业务样本。数据作者知道全量合成标签；本轮没有业务最终验收、真实用户试用或北极星指标验证。

## 证据位置与内容身份

- [自包含原始证据 JSON](complete-output-handoff-live-2026-10-22.json)：完整 8×4 行提示词、原始输出、输入/生成 IDs、EOS、token proofs、指标与错误；实际训练配置、来源、导出及工具身份、开发数据、固定 suite / 测试封存身份、真实成本和独立核验。
- [详细无权重补件](complete-output-handoff-evidence-2026-10-22.tar.gz)：原输出执行脚本、真实训练与启动日志、64 步 loss/LR、config/run/preflight/worker/训练 manifest、实际比较报告、导出回执和独立验证。包内 `manifest.json` 覆盖其他全部 74 个文件条目，分别记录原始与脱敏后的大小、SHA-256。
- 补件大小：167,899 bytes；SHA-256：`8ab3090a745007edfbb9141602afa55e96dc5b5eb6c763236b8c9b6bea45af85`。

将用户主目录和仓库绝对路径替换成 `<USER_HOME>`、`<REPO_ROOT>`，另明确移除 test 的逐行预检项；开发题生成文本、IDs、错误、分数和状态未修改。原回执中的哈希、内容摘要仍指原始文件，不以脱敏占位符文件冒充原始哈希的可重算副本。包内 manifest 另行绑定实际脱敏字节。

补件不包含基座、适配器、GGUF、checkpoint、优化器或二进制训练状态权重。不复制包含全量/测试逐行内容的原 source、session、full-data、SQLite、suite 和 test 文件；`omitted-files.json` 保留这些原文件的 SHA、路径及遗漏原因。为保守封存，脱敏 run/preflight 副本连 test 的逐行 row IDs、长度和监督数量也不保留，只保留测试汇总、seal 和明确遗漏计数；原文件 SHA 及结构化遗漏路径在 manifest / omitted-files 中记录。完整产品源代码已提交于 `dff1c73e2629faf3efc32c39967981fa15a3d0a7`。

## 合成任务及分区

任务是根据 `signal` 学习四个任意处理码：`amber→L8`、`birch→R3`、`cobalt→V6`、`dahlia→N1`。输入和 goal 中不提供这个映射；goal 只声明输出 `route`，允许 `L8、R3、V6、N1` 四个标签，并禁止解释、标点和代码块。对应关系只通过训练标签提供给模型。

数据为 4 signals × 20 组相同 ticket/note 组合，共 80 个反事实行。每个 ticket 的四个 signal 共用同一备注，工单号与备注不能指示类别。每个 ticket 的备注又有独立的干扰编号，避免不同 ticket 因完全相同的模型输入被连成同一大组。普通基础分析明确选择 `route` 为答案列、`ticket` 为分组；实际模型输入是 `signal` 与 `note`，不包含答案列。

以 seed 42 进行普通 80%/10%/10% 分组切分，没有为了盲标题或分数改变种子、移动样本：

| 分区 | 行数 | ticket 组数 | 每组结构 |
|---|---:|---:|---|
| 训练 | 64 | 16 | 每个 signal 各一行 |
| 开发 | 8 | 2 | 每个 signal 各一行 |
| 测试 | 8 | 2 | 保留封存，不生成或评分 |

开发组为 `R149-T00`、`R149-T16`；测试组名称由全量组集合减去训练/开发组得到，仅作为分组元数据记录。先物化普通分组版本，再通过真实 `EvalSuiteService.freeze` 冻结原开发与测试题，随后沿该 suite 再物化固定题版本。没有根据模型效果重分区。

原始 source 的标签是数据作者按规则生成的，作者当然知道它们，不能宣称测试对作者盲。实验脚本没有逐行解析物化后的 test 或 suite 测试题，没有测试生成或评分。正常产品物化、freeze、训练前预检和内部完整性核验会读取所有分区以检查来源、隔离、长度和监督，这是正常门禁；不能将其写成“整个系统从未读过测试”。封存承诺是固定题与字节不变，以及不使用测试生成分数选参。

## 真实普通零 key 门禁

全程未调用 Agent / BYOK 分析。使用真实 `propose_baseline_analysis`，由开发者作为模拟资料提供方明确选择答案、分组及指令，经 `IntakeService.apply_analysis → confirm → validate_full_data → confirm_full_data → materialize_dataset` 落入普通产品状态。

预览检查了每行输入和标签、类别均衡、同 ticket 分组及输入无答案泄漏。全量没有 blocking issue，开发者通过真实确认 API 确认。本轮是模拟业务方的集成验证，不把自动执行这些确认描述成真实业务用户认可。

在盲标前实际调用 `TrainingRunService.prepare`，得到 `blocked`，原因是尚未完成盲标，保留该运行 `wb-fff2ca9bad0647aebdd26d6185f6f229`。随后真实抽取 8 条隐藏输入题，脚本只读取 challenge 的 input，按预先固定的 signal 规则生成答案，不读取结果中的 `data_label` 来作答；提交后 8/8 一致，得到 `verified`。这一轮自然没有抽到测试 ticket，没有为此调整分区。

这证明真实门禁被执行，不证明数据作者是独立盲标人员。产品的常规 verdict 文案不能把这个开发者模拟提升为独立业务标注证据。

最终正式运行：

- Session：`d69fc5181a014671985b671dc698fcf3`，revision 6。
- Dataset：`fcd98e451ba85fcab820424b56b2bbd9bf4752789eead0c917a965adf59de94d`。
- Fixed suite：`021d0c50b8036f475b5f3fa29448349c22cb27b6d4fddbcc24ce663a7258b387`。
- Run：`wb-0dfbfbee65f040e3bf4e2d023646783e`。
- Config digest：`f9361ed7cb4d7419980313924433aae7a6bed3dd370b5707f226f7dc1d5079dd`。

`prepare` 预检通过，仅有答案监督策略 info，无需确认 warning。通过真实 `TrainingRunService.start` 启动已准备的不可变配置；没有直接 SFT 替身回执，没有跳过启动前模型/数据校验，没有授权自动 recovery。

## 实际训练与封存核对

使用本机已有完整 Qwen3-1.7B，Apple M3 Max / MPS。`alpaca_answer_only_v1`，最大训练长度 192，不截断，严格验证推理提示 token 是完整训练输入的前缀，只监督答案及 EOS；提示和 padding label 为 `-100`。

训练为 FP32、seed 42、r8 / alpha16 / dropout0 / all-linear / bias none，1 epoch、batch1、accumulation1，AdamW、constant LR `2e-4`、warmup0、weight_decay0、梯度裁剪1。实际 Trainer state 和日志确认 64 次 optimizer updates，64 条 loss 日志的学习率均为 `0.0002`。正式 worker 自然退出，PID 23761，return code 0，产品状态 `succeeded`。

产品在最后一步自动计算了 8 条开发数据的 teacher-forced loss：`0.0000872601`；训练平均 loss `0.943416`。它们不是生成准确率，实验没有另行用测试计算 loss 或生成分数。导出回执中的 `verified_rows=80`、`supervised_tokens=240` 是训练/开发/测试全部分区的**预检监督总量**，不是训练消费了 80 行；实际优化只消费 64 条训练行、64 步。

开发文件 SHA `7f0006b732c1bc8347441b6c2017c9bfec7df2a403876ac30679410a61f1589f`，测试文件 SHA `2e1109a30ca26a7148b95c6e9bd245968b2aa2c977066166805d286723907910`。训练前后，以及后续交接内部核验后，开发/测试字节 seal、固定 suite 和不可变配置保持一致。运行涉及的七个训练源文件 SHA 前后相同。

适配器权重 SHA：`175ddbf9b6cf15d93c4e35fa4e2ace95f44b503fb5c2b838cd4497247d40244e`；真实训练 manifest SHA：`b08f875f78f14038cfc642fc96959eb84b7049c8267c3973507e5aea12d3a590`。这些是正式 worker 生成并经产品重新校验的产物。

## HF / GGUF 导出及来源边界

只有真实训练成功并完成产物校验后，才运行产品 `plan_model_export / export_model`。合并 HF 与 GGUF 均绑定同一 run、session、dataset、config digest、基础模型内容身份、实际 adapter 和训练 manifest。两份导出结果独立重新盘点验证为已导出；HF 的 `output_files` 与 GGUF 的 `merged_source_files` 完全一致，证明这次 GGUF 使用了字节相同的合并 HF 来源。

合并默认 BF16，导出 HF config 与 safetensors header 均为 BF16；GGUF 为 Q4_K_M，复用已有 llama.cpp 固定 revision `436f6f89e1e581249900b37a5b8a12a36a6d0912`、已有 quantizer 与本项目静态 CPU native runtime。工具和 runtime 文件 SHA 在执行前后核对，没有构建、安装或下载。

| 产物 | 权重大小 bytes | 权重 SHA-256 |
|---|---:|---|
| 合并 HF | 3,441,185,608 | `a77af9d3cb74b6dc9cc52ddbe0d4a7787990367b7a4db4df6ad4ca9d7d1428ae` |
| GGUF Q4_K_M | 1,107,408,640 | `04ea3e4b2783f44d21f8bdb50e8c927e42e7b384d80ce21a229c66d3f74e76c4` |

来源诊断与独立导出核验是两种结论：

| 对象 | 训练来源诊断入口 | 本轮实际证据 |
|---|---|---|
| 原基座 | `not_available` | 没有本轮适配器来源身份 |
| 训练适配器 | `available` | 真实成功 run、配置、分区、manifest 和内容身份 |
| 合并 HF | **`not_available`** | 该入口未记录可验证的适配器身份；另经真实 HF export receipt、完整输出哈希与 source identity 独立核验为 `verified` |
| GGUF | `available` | 实际 GGUF / runtime / tokenizer / export-evidence 内容身份及明确 `source_training` 回溯 |

不把合并 HF 写成诊断入口 `available`，也不把它伪装为 adapter。独立核验补足的是这次实际导出来源证据，不是修改产品入口的能力声明。

GGUF 导出回执的 `business_evaluation_status` 在导出时为 `not_run`，后续比较后仍原样为 `not_run`，receipt SHA 保持 `bbd03079392f0a3db18f27965ae15c9b62c87c295e5daa25948d7abc81f6886d`。HF receipt 没有该评分字段，其 SHA 为 `972af6902bb1238c06f659de31e2d0635467afa122e5fb8a1a45338daed28576`。后续真实生成评分是独立报告，未回写导出回执，也没有继承 HF 分数充当 GGUF 分数。

## 四模型同题原始生成

比较报告 ID 为 `88dac05c7add42ac9a0b496e5bf25b9c`，原文件 SHA `90e77797ac6df8a5d90c64a08b2d2624f65907d59196a4254f9620b5790f204c`。使用固定开发集 8 题、原始 Alpaca 提示、`classification_exact`、首尾空白剥离后整段精确比较；`greedy_token_contract_v2`，单 beam、max_new_tokens16、Transformers decoder 5.9.0。

每题保存完整输入 IDs、实际生成 IDs、EOS 集合、finish reason、prompt/output digest。四个模型逐题完整输入 token IDs 和 EOS IDs 相同；GGUF 由 native runtime 实际检查原始输入 tokenization，最终生成 IDs 使用绑定的 HF tokenizer 解码。没有客户端自动套 chat 模板，没有抽取标签、去掉解释或补 EOS。

| 实际模型 | 生成覆盖 | 可计分行 | 答对/固定分母 | EOS 完整 | 长度截断 |
|---|---:|---:|---:|---:|---:|
| 原基座 | 8/8 | 0/8 | 0/8 | 0/8 | 8/8 |
| 本轮训练适配器 | 8/8 | 8/8 | 8/8 | 8/8 | 0/8 |
| 实际合并 HF | 8/8 | 8/8 | 8/8 | 8/8 | 0/8 |
| 实际 GGUF Q4_K_M | 8/8 | 8/8 | 8/8 | 8/8 | 0/8 |

基座不是模型加载失败：它真实生成 8 次，每行 16 token，继续输出解释而未到 EOS，因此每行保留 `truncated`、`correct=false` 和长度限制 error；模型级 `errors` 数组原本为空，也按原样保留。不能把可计分 0/8 换成空分母删除失败。原报告状态 `completed_with_failures` 正是这一事实，没有因后三个模型通过而改写。

三个训练衍生模型逐题 token IDs 相同：`L8=[43,23,151645]`、`R3=[49,18,151645]`、`V6=[53,21,151645]`、`N1=[45,16,151645]`。只说明这八题的输出 token 一致，不说明任意输入、logits、架构、精度或客户端行为等价。

FP32 训练后，产品默认 BF16 推理、BF16 HF 合并、Q4_K_M 转换以及 native CPU runtime 构成一条综合路径。本轮没有单独测量精度变化或纯量化损失，也不能推广到任意模型架构、任意量化或 LM Studio / 其他客户端自动 chat 配置。实际 GGUF 在指定 raw Alpaca 协议下可完整生成，不能据此宣称客户端自动聊天入口已经验证。

## 独立核验与真实成本

保存报告后，在新的 Python 进程运行只读验证：产品 `get_report / verify_evaluation_report`、冻结开发题绑定、实际 tokenization 和 EOS、各来源与内容哈希、导出回执与输出文件、训练回溯、工具字节、报告再次读回稳定性。该验证只加载 tokenizer、读取文件和哈希，不加载模型进行推理。交付时另按原始保存内容复算了所有 prompt/output digest、IDs 长度、8×4 分母/状态/得分及三衍生产物逐题 IDs 一致性。

| 阶段 | 实际墙钟秒 | 范围 |
|---|---:|---|
| 准备 | 5.032 | 零 key intake 至真实 prepared；只加载 tokenizer / 哈希模型，不训练 |
| 训练监控 | 40.016 | 真实 start、worker 运行至退出和核验；框架 train_runtime 为 21.0435 秒 |
| HF 导出 | 9.744 | 实际 progress 阶段边界 |
| GGUF 导出 | 29.989 | 实际合并、转换和发布阶段 |
| 四模型开发比较 | 35.304 | 实际逐题生成及报告保存阶段 |
| 比较后核验与收尾 | 30.961 | 独立验证及最终元数据核对 |
| 导出至比较/核验整体 | 111.034 | 包含上述导出/比较阶段及加载、来源检查等间隙，不可再与子阶段相加 |

未使用电表，电量保持未知；没有把这些时间估成实测电耗，也不把它们当作全部历史迭代成本。

## 复现脚本与结论范围

补件保留 `outputs/validation/r149/prepare_workbench_experiment.py`、`start_prepared_training.py`、`export_and_compare.py` 的实际执行版本及哈希。它们是本次 one-off 取证脚本，其中含固定 run/session/commit、输出路径与原工具位置。新实验必须重新通过真实准备流程创建新的 source、session、分区、suite 和 run，再明确替换对应身份与新目录；不能把原运行 ID 套到新模型，也不能将这些脚本宣传为一键可移植工具。缺少原本地资产时，去权重证据包只能审阅，不能独立重新运行模型。

本轮只训练一次，没有看到分数后重训、调参或重分区，没有改评分、补 EOS、修改 LM Studio、安装工具或下载模型。正式训练与评测 worker 已退出。结果补足了“完整作答任务经过真实 intake→训练→HF/GGUF 交接”的本地证据，但没有验证真实业务价值、北极星指标、独立保留集效果或旧 R146 财报任务零分的原因；这些边界保持未测。
