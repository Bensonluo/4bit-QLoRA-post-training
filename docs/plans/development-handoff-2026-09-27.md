# TuneSmith 开发暂停交接与恢复计划

更新日期：2026-09-27。用户要求暂停开发，持续目标已设为 `paused`，并行开发任务已停止。本文是下次恢复的入口；产品目标以[北极星权威版](north-star.md)为准，产品行为契约见 [agent-setup.md](../agent-setup.md) 与 [business-evaluation.md](../business-evaluation.md)。

## 1. 产品目标与已确认约束

**北极星(权威全文见 [north-star.md](north-star.md),2026-09-27 定稿):让任意行业、任意业务场景中雇不起算法工程师的人,以可承担的成本、在自己的消费级硬件上,独立运转模型后训练的完整闭环(数据版本→方案→训练→指标→可视化→实验/模型管理→迭代),产品用通用语义验证机制替代不在场的专家。**

核心工作顺序：用户描述业务目标并上传样例 → Agent 联合理解目标与数据结构 → 判断资料缺口、澄清业务含义 → 设计并执行数据处理 → 展示真实转换 → 确认全量数据 → 复用训练、评测和版本管理 → 根据坏例改进。

- 首要难点在训练前的数据接入与专业判断，不能用增加图表、模板或训练次数代替。
- 两个旧领域模板只是示例，产品需要配置、组件组合和隔离代码支持长尾。
- 基本 BYOK 必须有；不建设 Agent 额度、预算或复杂恢复平台。
- 已确认范围内自动执行；业务含义变化、阻断问题和需核对的提示应交还用户。
- 当前使用**虚构数据**验证软件流程。美股财报与未来约 20 个交易日“上涨／未上涨”只是选定的简单试点；不再调研外部金融资料、采购行情或以证明股价预测能力作为开发前置。
- 不反复在线试探 GLM。不得把真实训练运行成功、loss 下降或虚构样例通过写成业务效果达标。

## 2. 当前完成情况

| 环节 | 已落地、已验证的能力 | 证据边界 |
|---|---|---|
| 目标与样例入口（P0） | 同页 UI/CLI、CSV/Excel/JSONL、原始值和来源保留、画像、Agent 业务提问、目标与字段角色联合分析、真实转换预览、修订与业务确认 | 真实 GLM 已验证虚构任务；更广泛业务理解准确性仍需检验 |
| 数据管道（P1） | 多源关联、展开、会话组合、受限适配代码、真实隔离和正反例校验、独立全量验证、分区、版本与行血缘 | 任意长尾并未全部支持；隔离不可用时明确阻断 |
| 基本 BYOK | 共用供应商配置、环境/会话密钥、工具调用连接验证、GLM Coding Plan 实际调用 | 使用 `glm-5.3`，Coding Plan 地址 `https://open.bigmodel.cn/api/coding/paas/v4`；不是所有兼容服务均实测。密钥不写入本文或配置文件 |
| 训练交接（P2） | Agent 根据业务、数据、本地模型和硬件推荐配置；真实 tokenizer 预检；接入已有 SFT、日志、MLflow 与产物 | 已有本机 Qwen3/MPS 短训练及真实产物，不等于业务成功 |
| 业务评测 | 同协议基座/微调模型对照、逐样本输出、失败与截断保留分母、固定开发/最终题集、坏例诊断、自定义隔离评分 | 自定义评分真实 GLM 起草仍未成功验证；本地协议与隔离验证不能替代它 |
| 迭代（P3） | 父子轮次、改进假设、数据/配置差异、固定题集、第二轮训练、三模型对照、采用/继续/停止/证据不足记录 | 第二轮真实运行过，但旧路径靠手动交接，自动推进尚未完成 |
| 最终验收 | 独立保留测试、运行前确定标准、结果单独保存且不交 Agent 调参 | 实测只有一道题且生成截断，结论为证据不足；没有真实客户业务达标结论 |

主要实测文件：

- [真实 GLM 样例分析](../validation/data-intake-glm-2026-09-26.json)
- [长尾适配与训练评测](../validation/adapter-training-evaluation-2026-09-26.json)
- [Agent 推荐训练方案](../validation/agent-training-plan-2026-09-26.json)
- [固定题集第二轮](../validation/second-round-fixed-suite-2026-09-26.json)
- [Agent 数据修订](../validation/agent-data-revision-2026-09-26.json)：原配方已有 `strip`，主要更新验收样例，不能说成已验证新的转换逻辑修改。
- [最终验收](../validation/final-acceptance-2026-09-26.json)

## 3. 最近完成且已验证的修复

1. **虚构财报核心链路贯穿。** GLM 生成三来源处理方案后，实际执行得到训练/开发/测试各两条，三条明确排除。缺公开时间、缺行情变体均阻断，本地 tokenizer 预检通过。没有新增大模型训练，测试驱动的业务确认不冒充客户认可。见[实测记录](../validation/synthetic-agent-intake-2026-09-27.json)。
2. **Agent 上下文与时间提示。** 来源引用从 375 条压缩为保留各来源首尾的 43 条，完整血缘仍保留本地；预览返回真实时间分区与排除原因，减少询问是否放宽固定规则。这些后续修改通过协议测试，未重新调用 GLM 验证效果。
3. **训练消费正确性。** 完整 Alpaca 记录保留 EOS；截断不伪造完整答案；collator 按 attention mask 屏蔽填充，真实 EOS 即使与 pad 同 ID 也受监督。训练和预检共用分词及 collator。相关 120 项测试通过，六条虚构记录本地预检通过；未进行修正后的新一轮大模型训练。见[EOS 实测](../validation/sft-eos-consumption-2026-09-27.json)。
4. **长正文 CSV 上传。** 修复默认单字段 131,072 字符限制；保留引号、换行、前导零和原始行号，坏格式仍明确失败。上传、任务重开、Agent 分页重组及入口回归 52 项通过。
5. **缺标签记录可定位。** 样例和全量转换支持问题优先、状态筛选和分页；补标签提示指向实际重传入口；未成熟未来标签不要求人工补造。业务确认绑定当前展示的行，翻页需重新核对。页面及时间入口 20 项回归通过，覆盖样例第 25 条、全量第 45 条问题及第二页确认。

这些测试数是各次针对性验证，不应相加称作整个仓库测试全部通过。**上述页面测试是在本次未完成的自动执行 UI 修改之前通过的，不能覆盖暂停时的新增草稿。**

## 4. P3.4 自动执行交接（2026-09-27 已完成并通过真实贯穿测试）

要解决的实际问题：用户已确认改进方向与数据含义后，仍要逐次点击物化、训练准备、启动、训练后对照。现已改为一次明确授权后执行到固定开发集对照，最终采用决定仍由用户作出。

### 已落地的实现

| 文件 | 状态 |
|---|---|
| `src/workbench/iteration_execution.py` | `IterationExecutionService`：`start`（授权/恢复/幂等重复提交）、`get`（纯只读）、`stop`（停止交接并停关联训练）、`run_worker`（后台状态机：核对授权范围 → 绑定固定题集并物化 → 预检/准备 → 启动 → 等训练进程真正退出 → 三模型同协议对照 → 绑定结果）。授权快照绑定 goal/来源/方案/全量确认，授权后任一变化即阻断 |
| `scripts/workbench_iterate.py` | 脱离终端会话的后台 worker 入口（`start_new_session`，关闭页面不影响执行）；worker 可在任意时刻被杀，记录不损坏 |
| `src/workbench/iterations.py` | 新增 `claim_execution`（一轮只归属一个自动执行；归属后手动 prepare/start/bind 被拒）；`prepare/start/bind_evaluation` 的 `execution_id` 归属检查已生效 |
| `scripts/data_intake.py` | `iteration-execute` / `iteration-execution-status` / `iteration-execution-stop` 可用 |
| `ui/pages/07_Data_Intake.py` | 原 UI 草稿联通：确认轮次显示「按确认方案执行到开发集对照」；进度/停止/刷新；预检警告暂停后需勾选确认再继续；手动路径保留 |

执行记录状态：`queued/materializing/preparing/awaiting_warning_ack/training/waiting_for_release/evaluating/completed/blocked/failed/stopped`，记录 `run_id/evaluation_id/message/issues/session_revision`。终态后重复提交：`completed` 幂等返回原报告，其余终态拒绝并提示提新轮次。后台 worker 进程死亡时提交会如实标记 `failed`，不静默重排。

### 验证证据（全部真实运行）

- `tests/unit/test_iteration_execution_integration.py`：极小本地 GPT2 全真贯穿——一次授权 → 真实物化 → 真实第二轮训练 → 基座/父轮/本轮三模型固定开发集对照 → 绑定后轮次停在「待业务决策」（无 decision）；重复提交不新增训练（run 数保持 2）。
- `tests/unit/test_iteration_execution.py`（10 项）：授权范围绑定与阻断、陈旧 revision、未确认轮次、重复提交不重训、死进程如实上报、警告确认恢复需显式勾选、终态终结、停止联动训练、worker 尊重预先存在的停止请求。
- `tests/unit/test_iteration_execution_cli.py`（2 项）：CLI 授权/revision 过期退出码 2/status 与 stop 只读不推进。
- `tests/unit/test_iteration_execution_ui.py`（3 项）：页面仅显式点击触发执行、执行中可停止、警告恢复必须勾选。

### 顺带修正（与 P3.4 同步，均为暂停前遗留的不一致）

1. **评测报告携带完整固定题集引用。** 报告 `dataset.evaluation_suite` 现为与数据产物相同的完整引用（含 root/manifest_path/case_counts，可从报告独立溯源套件）；每份报告的按 split 题目身份 `evaluation_key` 移入 `protocol`，比较键语义不变。`comparison_key`/`EvaluationDiagnostics`/`bind_evaluation` 的协议比较同步更新（`evaluation_key` 与 `answers_digest` 同样按「套件绑定而非评分规则」排除父轮比较）。此前 outputs 下无存量报告，无迁移负担。
2. **过时测试更正。** `test_workbench_training_runs.py::test_warning_confirmation_is_specific_to_actual_tokenizer` 原期望 pad==eos 产生警告——EOS 消费修正后 pad==eos 是已验证安全配置（`test_sft_eos_contract`、`test_intake_training_preflight` 均断言 passed）。该测试改为用真实 tokenizer 特定警告（声明上下文长度低于 max_length）检验同一条「警告需显式确认才能启动」链路。

**P3.4 完成标准核对：** 一次授权即得真实三模型开发集报告 ✓；范围变化阻断 ✓；预检暂停与显式确认 ✓；停止联动 ✓；重复提交不重训 ✓；结果只进入待业务判断状态 ✓；后台独立于页面推进 ✓（worker 为独立进程，测试进程只读状态文件）。

已约定的服务接口（**已实现并按上方测试验证**，保留作记录）：

```python
IterationExecutionService(
    root, intake_root, iteration_root, training_root, evaluation_root,
    project_root=None, python_executable=None,
)
start(iteration_id, session,
      acknowledge_warnings=False, independent_rows_confirmed=False)
get(iteration_id)   # 纯读取，不推动执行
stop(iteration_id)
```

执行记录使用：`queued/materializing/preparing/awaiting_warning_ack/training/waiting_for_release/evaluating/completed/blocked/failed/stopped`，记录 `run_id/evaluation_id/message/session_revision`。

## 5. 后续实施顺序与验收

### 第一优先：完成当前自动交接（✅ 已完成，见第 4 节）

~~1. 先读本交接与工作区，补齐缺失服务/worker，解决上述导入断点；保留原有手动路径。~~
~~2. 复用 `IntakeService`、`IterationService`、`TrainingRunService`、`BusinessEvaluationService` 和现有锁/进程管理。~~
~~3. 同轮重复提交不得重复训练；刷新只读；停止阻止后续交接并停止关联训练。授权后目标、资料或方案改变应阻断，不能沿用旧授权。~~
~~4. 新出现的预检提示先展示再由用户核对；不能初次提交便默认接受未知提示。~~
~~5. 验证后台独立推进，关闭页面不影响既定交接。~~

以上五条均已按第 4 节的测试证据落实。

### 第二优先：补齐当前核心用例的证据

- ~~核验从坏例出发的一次**实际转换规则修改**，经真实预览和确认后交给上述流程~~ **✅ 已完成（2026-09-27）**，见[验证记录](../validation/rule-revision-auto-execution-2026-09-27.json)。要点：父轮真实训练+对照拿到真实坏例（两模型均复述任务定义并截断、零分）→ 在披露正文绑定上真实新增 replace 转换（`次公开披露：收入为100`→`次披露：收入为100`，字节核对仅命中训练期两行，固定开发/测试题逐字节不变以满足套件约束）→ 真实预览 2/9 行变化并经样例与全量两级业务确认 → 一次授权经 P3.4 自动执行完成三模型同题对照，轮次停在 `evaluated` 无 decision。三模型 exact_match 仍为 0——结论是软件流程已核验，业务效果未达标也未宣称。
- 先用少量虚构正反例检查含义。只有需要验证真实 Agent 判断时再作有目的的单次在线验证；连接失败应保留清晰原因，不反复盲试。（本次规则修改由操作者按坏例确定性做出、未经 GLM，符合"先用虚构正反例检查含义"。**单次在线验证已备好一键入口**：`TUNESMITH_AGENT_API_KEY=… venv/bin/python scripts/verify_agent_revision_live.py` ——本地提出并确认一个以坏例诊断为方向的 data_change 轮次后，对真实 GLM 做一次 `revise_data_for_iteration` 调用，不重试，结果如实写入 `docs/validation/agent-revision-live-2026-09-27.json`。当前会话环境无密钥，状态为 `not_attempted`（未尝试连接，非连接失败），已按"不盲试"要求记录；取得密钥后重跑该脚本即完成此项。）
- 检查 EOS 修正后的实际训练消费；极小模型可证明交接与执行，大模型效果仍需独立证据，不把它混为一项。（本轮两次真实 Qwen3-1.7B 训练均经修正后的 prepare/start/tokenizer 路径启动并成功产出 adapter，运行层无异常；但"训练消费正确性"的专项证据仍以 2026-09-27 的 EOS 记录为准，本轮不重复宣称。）

### 第三优先：用户试用与业务验收

软件核心通路稳定后，让目标用户以自己的业务含义核对方案和结果，记录需要算法专家介入的判断点。没有获得用户认可或可比较证据前，保持“未验收/证据不足”。不为等客户资料而扩建 P4；DPO/GRPO、更多后端和图表按实际需求再排。

**移交状态（2026-09-27）：** 软件核心通路已稳定并经真实证据核验（P3.4 自动执行、坏例驱动的实际规则修改全链路）。用户试用可直接从当前工作区开始：

1. **面板已在本机运行**：Streamlit http://localhost:8501（MLflow UI http://localhost:5001——默认 5000 被 macOS AirPlay 占用，启动时用 `--mlflow-port 5001`）。任务 `cd0ffd458e024e3da2436e8cf7267c35` 已有完整历史：父轮/本轮训练、三模型对照报告、待业务决策的改进轮次（`it-af3ecde…`，决策按钮：采用/继续/停止/证据不足）。
2. 试用记录模板：[user-trial-log.md](../validation/user-trial-log.md) ——按建议动线逐步记录"能否独立判断"，试用后汇总**需算法专家介入的判断点**，即产品"零专家介入"目标的差距清单。
3. 如需先补单次 GLM 在线验证：在**您自己的终端**执行 `TUNESMITH_AGENT_API_KEY=<密钥> venv/bin/python scripts/verify_agent_revision_live.py`（一次调用、不重试、结果自动写入 `docs/validation/agent-revision-live-2026-09-27.json`）。恢复会话已穷尽本机全部许可来源（环境、shell 历史、历史会话日志、钥匙串、配置目录）确认密钥从未持久化——符合设计；密钥不要经对话或文件传递。
4. 试点仍为虚构资料；更换真实业务资料前不宣称任何业务达标。

## 6. 恢复所需资料与工作区注意事项

- 虚构试点文件：`data/custom/examples/synthetic_filings/`，包含事件、价格、日历、异常变体与 `scenario.txt`。
- 已完成的虚构财报任务：`outputs/workbench/intake`，session ID `cd0ffd458e024e3da2436e8cf7267c35`。恢复先读取当前 revision，不复用旧 revision 写入。
- 历史验证文件在 `docs/validation/`；部分真实模型运行产物位于 `/tmp/tunesmith-*`，可能被系统清理，须检查存在性后再引用或执行。
- GLM 密钥不保存到交接、日志或源码。通过环境或会话重新取得可用配置，不把曾经连接成功当成当前可用。
- 未提交、未推送、未创建 PR；大量已实现文件仍是 Git 未跟踪文件。不要 `git clean`、重置工作区或误删未跟踪文件。
- 保留与本任务无关的既有变动，尤其 `scripts/registry_cli.py`、`src/tracking/registry.py`、`.zcodeignore`，以及两个旧文档的删除，不顺手还原。
- 暂停时本轮没有启动新训练、GLM 请求或后台执行 worker。两位并行代理均确认无未结束工具/测试/训练进程句柄；父任务也没有此类活动句柄。系统级 `ps` 被沙箱限制，未作全机进程扫描，不能据此声称其他用户进程均停止。

## 7. 产品就绪迭代轮次（2026-09-27 恢复会话持续进行）

按「产品 ready」目标定义就绪缺口并逐项闭合，每项带真实测试与（涉 UI 时）运行中面板的浏览器实测：

### 第 1 轮（已完成，全量回归 1503 项通过）

| 项 | 内容 | 证据 |
|---|---|---|
| R1 零 Agent 通路 | `src/workbench/baseline_analysis.py`：确定性基础分析（答案列/分组/排除字段 → 合法 IntakeAnalysis，内置反复述指令模板，findings 如实声明"不判断业务含义"）；页面新增「没有 Agent 服务？用基础分析开始」入口，新用户零密钥可走通到真实预览 | `test_baseline_analysis.py`（3）+ `test_baseline_analysis_ui.py`（1） |
| R2 指令回声诊断 | 服务层 `output_echoes_prompt/count_instruction_echo`（最长公共子串 ≥12 字，逐字前缀与**改述式**复述都命中；短标签不误判）；诊断 summary 新增 `instruction_echo` 假设与核查方向；对照表新增「复述指令」列与警告 | `test_evaluation_diagnostics.py`（+2）；运行中面板对真实财报三模型报告实测显示「检测到指令回声（基座 2 题、父轮模型 2 题、本轮微调 2 题）」 |
| R3 决策 UI | 补齐此前缺失的迭代决策入口：已评测轮次展示三模型结果表 + 采用/继续/停止/证据不足单选 + 必填业务理由；已决策轮次显示记录 | `test_iteration_decide_ui.py`（2）；试点轮次 it-af3ecde 已按真实结果记录「证据不足」决策 |
| R5 快速上手 | QUICKSTART.md 顶部新增产品主路径（含零 Agent 入口与诚实边界声明） | 文档 |
| 死按钮修复（走查发现） | 旧数据版本的 prepared 训练方案不再渲染可点击的启动按钮（点击必被服务拒绝），显示原因 | `test_workbench_training_ui.py`（+1） |
| 浏览器实测 | 首页导航、任务选择、决策渲染、回声警告、对照表全部在运行中的面板上用浏览器实测通过；发现并解决 Streamlit 模块热缓存导致的 ImportError（重启进程即可，非代码缺陷） | 本节记录 |

已知非缺陷：`outputs/workbench/training` 中 wb-d12aa/wb-d052 为早期 CLI 调用留下的 prepared 方案（数据版本已过期），页面现在如实标注「旧数据版本，启动会被拒绝」，保留作历史。

### 第 2 轮 = 里程碑 M1 语义安全层（2026-09-27 完成，提交 5650eb7，分支 dev/m1-semantic-safety）

北极星核心痛点「语义错误无声通过」的三道通用关卡全部落地并验证：

| 关卡 | 性质 | 证据 |
|---|---|---|
| 盲标核验 | 硬门禁：隐藏答案抽样作答，全一致才允许准备训练；摘要绑定，修订自动失效；P3.4 授权前置校验 | 8 项测试 + CLI 烟雾 + 浏览器实测（真实财报会话 5/5 判定通过） |
| 对比核验 | 样例确认防盲点头：两输入+打乱答案配对，配错留档 | 3 项测试 + 页面测试 |
| 可学性探针 | 可选证据：基座零样本 vs 瞎猜多数类基线，带样本量限制声明 | 2 项测试 + CLI + 页面入口 |

契约文档：agent-setup.md 新增「语义安全层」段。全量回归 **1517 项全绿**（+14）。

### 第 3 轮 = 里程碑 M2 旅程串联与非专家可视化（2026-09-27 完成）

| 任务 | 交付 | 提交 |
|---|---|---|
| t1 零密钥旅程集成测试 | test_product_journey:建任务→基础分析→对比核验→确认→全量→盲标→物化→训练产物,一段不断;修复 capability_gaps 误阻断真断点 | 9097c95 |
| t2 对照报告语言化 | report_summary.summarize_comparison + 页面「大白话解读」区,固定「不是达标结论」结尾 | 5e976a4 |
| t3 训练/预检语言化 | summarize_preflight / summarize_training_run + 「大白话看这轮训练与检查」区 | 51289c0 |
| t4 页面旅程走查 | test_page_journey:六阶段每段断言内容+下一步指引+无异常,双路径并存 | 06e252d |
| t5 核验失效可感知 | 修订后旧核验带 stale 标记,页面明确「已失效请重验」,门禁照拦 | 695a7e9 |
| t6 收尾 | UI 失效提示测试 + 运行面板实测语言化渲染 | 本提交 |

基线:回归 1528 项全绿。M3(三角血缘+成本叙事)为下一里程碑。

### 第 4 轮 = 里程碑 M3 成本叙事 + 三角血缘（2026-09-27 完成）

| 任务 | 交付 | 提交 |
|---|---|---|
| t1 成本账 | cost_summary(时长实测/功耗电价明示估计/API 对比口径)+worker finished_at+页面成本区 | dc74056 |
| t3 血缘正向 | run_registration_status(按 workbench.run_id 标签全局反查注册状态) | 305b9de |
| t4 血缘反向 | version_lineage + registry_cli lineage(workbench/external/no_source 如实) | 2d9e1f6 |
| t5 真集成 | 训练→合并→带血缘注册→双向反查闭环;修 transformers flavor 本地模型崩溃与注册血缘丢失 | b58be06 |

北极星三差距(旅程串联/非专家可视化/三角血缘+成本)全部闭合。回归 1536 项全绿。

### 第 5 轮 = 里程碑 M4 场景矩阵测量台（2026-09-27 完成）

场景矩阵核心(ScenarioSpec/run_scenario/run_matrix)+6 内置场景+CLI+首跑 6/6 as_expected
(报告 docs/validation/scenario-matrix-latest.json)。新增预测型目标的时间分区泄漏
预警。已知缺口入 M5 队列:高基数目标早期警告。提交 c480380、b72498f。

### 第 6 轮 = 里程碑 M5 known-gap 清偿（2026-09-27 进行中）

高基数目标早期警告(27ff83e)、标签变体检出+矩阵 8 场景(9ba317d)、开放任务诚实
声明+value_kind 误判修复(30a0c28)。矩阵保持 8/8 as_expected,回归 1544 全绿。

### 第 6 轮补记 = 里程碑 M5 完成（2026-09-27）

known-gap 清偿完毕:高基数早期警告、标签变体检出、开放任务诚实声明+value_kind
误判修复(30a0c28)、对比核验二连对换题防蒙(c1d848f)。矩阵 8/8,回归 1544 全绿。

### 第 7 轮 = Ralph PRD 循环(2026-09-27,US-001..005 全部 passes)

US-001 探针候选页面呈现(74aba11)/US-002 预检+训练语言化页面验证/US-003 候选导出
CSV(e198db3)/US-004 标点变体场景 16(b72498f 后续)/US-005 全量回归 1560 全绿。
矩阵计数测试改为增长自适应(全 as_expected 即可)。

### 候选下一轮（按用户试用反馈排序）

- 试用模板（user-trial-log.md）回收后按专家介入点清单迭代
- 评测对照区对「截断比例高」给出 max_new_tokens 核查提示（已有回声方向，截断方向同构）
- 基础分析支持时间分区字段选择（当前仅分组；预测类任务仍需 Agent 或补充说明）


下次可直接说：**"继续产品就绪迭代（第 7 节候选清单），或回收试用记录按专家介入点迭代。"** 不必重新讨论北极星、客群或购买真实金融资料。

### 第 8 轮 = Ralph PRD 循环(2026-09-27,US-006..009 全部 passes)

US-007 对比核验三轮可选(连胜真实计数+二连对后可选入口,37547b3)/US-008 README
主路径与北极星对齐+补齐 LICENSE(9cf6430)/矩阵场景 16→18:Excel UTF-16 导出
BOM 识别+超长单行边界(59fd01c)/语言化摘要接入 train/eval/plan CLI(e1291ef)/
US-009 探针结果存盘可回读——页面刷新不丢证据,UI+CLI 双路径(afc33b8、fa9f703)。
全量回归 1573 全绿;矩阵 18/18 as_expected。

### 第 9 轮 = 并行矩阵扩展（场景 19-20）

两个脏数据场景先实测、后定局,expect 全部按真实行为记录(提交 3b50601、d986dc7):

- **场景 19「空答案行混入」**(`empty-label-rows-in-full`):样例两条均有标签,全量
  005/008 行答案为空。实测结局:既不被静默跳过,也不带病通过——空标签行按
  `needs_label` 计为 blocking,在 `validate_full` 硬拦,报错点名
  「全量存在缺少监督答案的记录…(2 条)」;expect=`blocked_at:validate_full`,
  用户须补标签或删行后重新验证。
- **场景 20「列名前后空格」**(`spaced-header-names`):表头如「 编号 ,客户描述, 类别」。
  实测结局:入口读表不剥空格、create 阶段不拦;用户按业务口径选「类别」时
  精确匹配失败,在 `baseline_analysis` 即被拦,报错把带空格的真实列名原样列出
  (可用:[' 编号 ','客户描述',' 类别']),用户能看见差异;
  expect=`blocked_at:baseline_analysis`。边界如实记录:若改选确切带空格列名
  可全程通过,空格随之进入指令与标签——不做自动剥空格,是否归一由用户决定。

矩阵 20/20 as_expected;test_scenario_matrix 9 项全绿。

### 候选下一轮（按价值排序）

- 评测对照区对「截断比例高」给出 max_new_tokens 核查提示（与指令回声提示同构）
- 基础分析支持时间分区字段选择（当前仅分组；预测类任务仍需 Agent 或补充说明）
- 矩阵场景继续扩展：JSONL 超长行、重复表头、全角数字编号
- 试用模板回收后按专家介入点清单迭代

### 第 10 轮 = Ralph PRD 循环(2026-09-27,US-010..014 全部 passes)

US-010 评测对照区高比例截断给 max_new_tokens 核查提示——`high_truncation_models`
共享口径(单模型截断占比≥20%)接入对照区警告与 `summarize_comparison`,带当前
max_new_tokens 值,与回声提示同构、不认定原因(bdbd994)。US-011 基础分析支持
时间分区字段选择——零密钥路径可选三个时间字段+带时区边界,边界不完整明确报错
不退回随机切分;端到端物化 `split_method=temporal`;配时间方案后预测型目标不再
要求 Agent(d33e992)。US-012 矩阵场景 20→23:JSONL 超长行(单行数十 KB)/重复
表头/全角数字,期望结局以实测为准(855116f)。US-013 清偿场景 21 暴露的真缺口:
重复表头行原本静默成为训练样本,现被全量验证 `repeated_header_rows` blocking
硬拦并点名行号,不自动删行(0bbf86d)。US-014 手工训练参数大白话指引与推荐起步
值(协调员追加,791c3bf)。全量回归 1588 全绿(+15);矩阵 23/23 as_expected。

### 候选下一轮（按价值排序）

- 试用模板（user-trial-log.md）回收后按专家介入点清单迭代（需真实试用数据）
- 复核 PRD known-gap 清单:高基数目标(`high-cardinality-target`)仍是唯一 known-gap 场景
- JSONL 样例与全量列顺序/字段不一致时的行为可补一个边界场景
- CLI eval 子命令核对语言化摘要与对照区提示口径一致性

### 第 11 轮 = 并行矩阵扩展(场景 24-26)

三个新场景全部先实测真实行为、后定期望,expect 无一凭猜(提交 f89a071、90f3ada、30eb4ad):

- **场景 24「只有一行的样例」**(`single-row-sample`):样例仅 1 行数据(2 行含表头)且该行有标签。
  实测结局:create 与基础分析都不拦(分析如实观察「答案列非空取值 1/1 行,共 1 类」),
  即使全量数据正常,旅程也在**对比核验**被拦——「对比核验需要至少两条答案不同的已标注行。」
  这是该关卡的第一个矩阵场景:一条样例不足以让用户完成配对核验;
  expect=`blocked_at:contrast_check`,用户须至少提供 2 条答案不同的已标注样例。
- **场景 25「目标列全空」**(`all-empty-target-column`):「类别」表头存在但样例与全量所有值均为空
  (导出漏了标签值列)。实测结局:基础分析**不拦**,finding 如实观察「非空取值 0/2 行,共 0 类」,
  预览逐行标 `needs_label`;旅程同样在**对比核验**被拦(同一条报错)。不静默跳过、不自动补值;
  expect=`blocked_at:contrast_check`,用户须先补标签再重走旅程。与场景 19(样例有标签、
  全量混空行→`validate_full` 拦)互补:全空形态更早暴露(样例阶段即拦)。
- **场景 26「无 BOM 的 UTF-16」**(`utf16-no-bom-rejected`,自选脏数据形态):Excel「Unicode 文本」
  导出经其他工具转存丢 BOM。实测结局:utf-8-sig 解不开,gb18030 兜底也解不开(前导字节后跟
  交错的 NUL/换行字节),**create 即明确拒绝**——「无法解码文件，请明确指定编码；原始数据未修改。」
  恢复路径已实测:入口显式指定编码(如 utf-16-le)即可正确解码并继续旅程;
  expect=`blocked_at:create`,是矩阵首个 create 关拦截,与 GBK 自动回退(通过)、带 BOM UTF-16
  按 BOM 证据识别(通过)构成编码家族的完整边界。测试钉住夹具「两个自动兜底都解不开」,
  防止夹具漂移成碰巧可解码的形态。

矩阵 23→26,26/26 as_expected;test_scenario_matrix 13 项全绿(全集下限断言随之
16→26)。全量回归见本轮提交记录。
