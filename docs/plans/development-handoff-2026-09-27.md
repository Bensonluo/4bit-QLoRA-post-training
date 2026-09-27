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

### 第 12 轮 = 并行矩阵扩展(场景 27-29)

三个新场景全部先实测真实行为、后定期望,expect 无一凭猜(临时探针测完即删;
提交 e11bf28、31f4948、611c375):

- **场景 27「答案列单一取值」**(`constant-target`):样例与全量 10 行全部「质量」。
  实测结局:create 与基础分析都不拦,基础分析如实发出**非阻断预警**——「答案分布
  严重不均衡:「质量」占 100%(10 行)。模型只要全猜这一类就有 100% 准确率」;
  旅程在**对比核验**被拦:「对比核验需要至少两条答案不同的已标注行。」单一类别
  连配对核验都组不成,模型没有可学习的区分边界;expect=`blocked_at:contrast_check`,
  用户须让数据覆盖至少两个类别或确认答案列选错。夹具样例取满 10 行,正是为了让
  不均衡检出(≥10 行才触发)有机会说话——有提示(不拦)+有关卡拦,双层如实。
- **场景 28「答案列全是空格」**(`whitespace-only-values`):表头保留「类别」,
  每行答案是 3 个空格。实测结局:**空格不算真值**——预览层按 strip 判空,逐行标
  `needs_label`(「缺少监督答案，需要补充或确认标签；没有自动生成真值。」),
  target 为 None,与真空值同路;旅程在**对比核验**被拦(与场景 24/25/27 同一条
  报错)。边界如实记录:基础分析的「非空取值」计数按不等于空串统计,空格被算作
  非空(finding 显示「非空取值 2/2 行,共 1 类」)——分析层与预览/验证层判空口径
  不一致,但既不静默跳过也不带病通过,拦截关卡与报错同真空值完全一致;
  expect=`blocked_at:contrast_check`,用户须补真标签。
- **场景 29「单格 3 万字符」**(`extreme-long-single-cell`):单个输入单元格精确
  30 000 字符(按 CSV 规范加引号),样例与全量各含一条,其余行普通短文本;行号
  嵌入日志开头保证长格内容唯一(探针实测:内容全同的长格会被「输入完全重复」
  检出、更会因相同输入不同答案在确认样例关被拦——夹具因此按行差异化)。实测结局:
  入口读取完整无截断(单元格 30 000 字符原样)、预览原样进入(零密钥数据层不截断)、
  全量验证无 blocking,**八关全过**;expect=`passes`。截断风险边界如实记录:
  「2000 字符展示截断」只是 Agent 工具的展示层惯例(原始值保留本地,
  `read_cell_content` 按 offset/limit 分页可读全),不是数据层截断;训练期是否
  截断由训练前预检用真实 tokenizer 测量,语义是否受影响由用户在真实预览核对。
  对照事实:未按 CSV 规范加引号的长格(逗号裸奔)在 create 即按「列数与表头
  不一致」拒绝——规范内完整通过,规范外入口诚实报错。与场景 18(所有行超长)
  互补,本场景钉「单格极限」。

矩阵 26→29,29/29 as_expected(意外通过/意外拦截/错误均为 0);test_scenario_matrix
16 项全绿,全集下限断言随之 26→29。全量回归见本轮提交记录。

### 第 13 轮 = 变体一键归一(并行 agent A 第四批)

三个独立提交(88a9c2c、d18d914、6c3cbbd),把变体检出从「提示用户手动归一」推进到
「草案已就位,用户只裁决」:

- **变体一键归一草案**(88a9c2c):基础分析检出到标签变体后,自动生成 map_values
  归一转换草案——每组变体映射到该组出现次数最多的写法(平票取数据中先出现的写法,
  重复运行结果一致),直接预置进 propose_baseline_analysis 方案的答案列 transforms;
  finding 消息附「已生成归一规则草案：质量。→质量」等可直接采用的规则说明。关键
  设计事实:map_values 是严格映射,映射之外的值会在预览报错,草案因此覆盖答案列
  全部已见取值(变体→规范写法,非变体原样保留);用户在预览确认时可修改或删除
  草案,归一写法的选择仍由用户裁决;干净标签不生成草案,答案列转换保持为空。
- **矩阵场景变体归一端到端**(d18d914):实测发现 dirty-label-variants 原夹具样例
  只有「质量/物流。」各一行,归一后每组只剩一种写法——变体检出实际不触发,
  expect_note 声称的「标签多种写法」预警从未出现(夹具与声称不符,本轮修正)。样例
  改为 6 行、同时含两组变体写法(带句号与不带)且规范写法各占多数,归一草案从样例
  生成,恰好覆盖全量出现的全部四种写法;新测试断言方案预置草案、finding 附规则、
  预览答案集合归一后只剩「质量/物流」、无问题行、八关全过;矩阵回跑 29/29
  as_expected。
- **平票决策显式化+证据行引用**(6c3cbbd,自选任务,缺口来自代码走查):①平票时
  草案「取数据中先出现的写法」此前只藏在代码注释里,用户无从知道草案依据,现在
  finding 消息明说——「「质量。」「质量」出现次数相同（各 3 行），草案暂取数据中
  先出现的「质量。」，请在预览确认时改选你认定的规范写法」,并钉住确定性(同一数据
  重复生成草案完全一致,不因集合迭代顺序漂移);②变体 finding 此前不引用任何证据
  行,用户须自己翻数据找样例,现在变体组内每种写法各引用其首次出现的真实样例行
  (上限 6 行),经 apply_analysis 的证据行核对真实可用。

test_baseline_analysis 15→19 项、test_scenario_matrix 16→17 项全绿;全量回归
pytest tests/unit 1614 passed / 0 failed;ruff clean。

### 第 14 轮 = 第五批并行(两 agent 分线交付)

**B 线:eval-compare / train-status 语言化摘要的 CLI 分派路径钉死**(3 提交:
e118c97、9d285a1、本记录)。背景核对:e1291ef 已把 summarize_comparison /
summarize_training_run 接进两个子命令,且各有一条最小 CLI 断言(eval 侧仅空报告
占位句「该报告没有模型结果。」,train 侧仅 succeeded 无预检一层)——本线不重做
接线,补上真实分派路径的行为缺口,新建 tests/unit/test_cli_summaries.py:

- **eval-compare 真实对照报告全句断言**(e118c97):mocked
  TrainingRunService/BusinessEvaluationService(复用 evaluation_cli 夹具模式),
  报告携带基座 3/10、本轮微调 8/10 两个模型结果走完整分派;stdout 仍为纯 JSON,
  stderr 逐句断言:总题数句「这次对照在固定开发集的 10 道题上进行」、逐模型
  答对句、最优句「答对最多的是本轮微调(8/10)」、增益句「本轮微调比基座答对更多
  (8/10 vs 3/10)——但要注意样本量」、10 题仍触发的小样本提示「任何百分比都受
  单题影响很大」、收尾「以上是观察事实,不是业务达标结论」。
- **train-status 摘要两层各就位**(9d285a1):记录附带 preflight 时,stderr 在
  训练状态句(模型名 Qwen3-1.7B、轮数、「训练完成，产出了微调适配器。」、
  loss 0.4200 与「不代表业务效果」、「要用同一套开发题与基座对照」)之外追加
  预检大白话(「训练前检查通过：6 行数据」「没有内容因长度超限被截断」「不代表
  训练效果或业务达标」);记录没带预检时只给状态句(running 的「正在训练中」
  「关闭页面不影响后台训练」),stderr 不出现任何「训练前检查」字样——没有的
  证据不编造,与摘要层的诚实边界一致。

**A 线(另一并行 agent,页面与探针方向):**(待其交付后由协调员或本人补记——
预期涉及 ui/pages/07_Data_Intake.py、learnability_probe 及其测试。)

### 第 15 轮 = 并行矩阵扩展(场景 30-32)

(编号说明:任务原文写「第 13 轮」,但本文档第 13/14 轮已被占用,按既有顺序续号为
第 15 轮。)三个新场景全部先以临时探针实测真实行为、再定期望(探针测完即删,
禁止猜):

- **场景 30「样例中部混入重复表头行」**(`duplicate-header-row-in-sample`):
  与场景 23(`duplicate-header-rows-in-full`)方向相反的脏数据——重复表头行混在
  **样例**(非全量)中部,全量干净。实测结局:**样例侧不拦**——入口按普通数据行
  读入(3 行数据),该行以「就绪」面目进入预览:输入是列名「客户描述」、答案是
  列名「类别」,分布 finding 如实把「类别」计为一类(「共 3 类:质量×1、类别×1、
  物流×1」),对比核验也可能拿它出题(探针实测:配对项与选项都出现列名);
  全旅程八关通过。边界如实记录:与全量侧硬拦(点名行号、要求删行)构成不对称
  ——样例侧没有对称检查,靠用户在预览逐行核对自行识别;该行不进入物化数据集
  (物化只消费全量预览,已钉断言:全量预览仍为干净 10 行)。
- **场景 31「引号内含换行的多行单元格」**(`multiline-quoted-cells`):输入与标签
  单元格按 CSV 规范加引号、引号内有换行。实测结局:入口按规范完整读入——多行
  输入与多行标签原样保留,行号记录的是记录起始物理行(钉断言:第 2 条记录起始
  物理行=5),分隔符嗅探不受换行干扰;预览输入/答案逐字含换行,对比核验与盲标按
  精确原文匹配,八关全过。两个对照事实均实测并钉住:①同样的换行不加引号(裸
  换行)在 create 即按「第 3 行有 1 列」诚实拒绝;②标签**首尾**带换行(如
  「\n质量\n」)时盲标必拦——提交侧按 strip 比对、数据侧保留原文,用户照抄预览
  答案也对不上(探针实测,未入夹具,记入 expect_note);含内部换行的标签
  (本场景夹具形态)不受影响。
- **场景 32「目标列大小写变体」**(`target-case-variants`):Yes/yes/YES 混用。
  实测结局:**被当成 3 个不同类别原样通过**——入口已有 strip,大小写无归一;
  变体检出只覆盖「strip/末尾标点归一后相同」的写法,大小写差异不在归一范围,
  因此不发「标签多种写法」预警、不生成 map_values 草案(测试断言 transforms==[]
  且无变体 finding)——与场景 8(dirty-label-variants,句号变体检出+草案预置)
  形成明确分界;分布 finding 如实列出 3 类,全量无新增类别,八关全过。边界如实
  记录:大小写变体会被模型当不同答案学习,如需归一须用户手工加 map_values 规则
  或在数据侧统一写法。

矩阵 29→32,32/32 as_expected(意外通过/意外拦截/错误均为 0);test_scenario_matrix
17→20 项全绿,全集下限断言随之 29→32;ruff clean。

### 第 16 轮 = 并行矩阵扩展(场景 33-35)

(B 线第七批。)三个新场景全部先以临时探针实测真实行为、再定期望(探针测完即删,
禁止猜):

- **场景 33「样例与全量目标列含义反转」**(`target-meaning-reversal`):样例确认
  「类别」=售后类别(质量/物流),全量是另一份导出——同名列装的是优先级(高/低),
  输入行与样例不重叠。实测结局:apply_analysis/样例预览不检测漂移(全量那时还不在
  场);漂移的既有守卫在 validate_full 触发但**不硬拦**——`new_categories` 是
  review 级预警:「类别字段 类别 出现样例未覆盖的 2 种答案,请核对是否属于目标类别」
  点名全部 10 行,`new_target_values` 如实记录 ['高','低'],全量预览原样携带反转后
  标签,确认全量不被 review 拦;含义反转最终**拦在盲标关**——按样例确认的业务语义
  作答的用户复现不出 高/低(盲标 0/5 insufficient_agreement)。对照事实(探针实测):
  若全量还含与样例同输入的行,`sample_answer_disagreement`(blocking)会在
  validate_full 更早硬拦。边界如实记录:照抄数据标签的「全知用户」会全程通过——
  自动检出=review 预警,语义裁决靠用户在预览与盲标两处人工核对;期望定为
  blocked_at:blind_verification,若日后把 new_categories 升级为 blocking(拦截前移),
  本场景需随之改期望。
- **场景 34「单格 10 万字符」**(`100k-single-cell`):extreme-long-single-cell
  (精确 3 万字符)的 3.3 倍同形态夹具(含逗号、按 CSV 规范加引号)。实测结局:
  入口读取完整无截断(读入口按文件长度抬高 csv 字段上限,不依赖 128KB 默认值),
  预览原样进入(输入 100 006 字符=单元格+「客户描述: 」前缀),全量验证无 blocking,
  八关全过——与 3 万字符场景同结论:零密钥数据层不设长度上限,训练期截断风险由
  训练前预检用真实 tokenizer 测量。
- **场景 35「Excel『CSV UTF-8』导出」**(`excel-utf8-bom-csv`,自选形态):UTF-8 带
  BOM(EF BB BF)+ CRLF 行尾——Windows Excel 2016+ 默认 UTF-8 导出的真实形态。
  实测结局:全程无碍——utf-8-sig 解码剥掉 BOM,CRLF 由 csv 规范消化,列名与单元格
  值都不残留 \ufeff/\r,八关全过。对照事实(探针实测):同样的字节用纯 utf-8 解码,
  首列名是「\ufeff编号」——若入口不做 utf-8-sig 兜底,用户按业务口径选「编号」作
  分组列就会像 spaced-header-names 一样在基础分析被拦;与 gbk-encoded-upload(编码
  回退)、utf16-excel-export(BOM 证据识别)、utf16-no-bom-rejected(明确拒绝)共同
  构成入口编码家族边界。探针同时实测了两个落选形态,记录在案:仅 CRLF(无 BOM)
  同样干净通过(不单列场景);全量答案值带前后空格(样例干净)会因「同输入不同答案」
  在 validate_full 被 sample_answer_disagreement 硬拦——本质与场景 33 对照变体同守卫,
  不重复入列。

矩阵 32→35,35/35 as_expected(意外通过/意外拦截/错误均为 0);test_scenario_matrix
20→23 项全绿,全集下限断言随之 32→35;ruff clean。docs/validation/scenario-matrix-
latest.json 留给跑台例行刷新,本批未动。

### 第 17 轮 = 并行矩阵扩展(场景 36-38)

(B 线第八批。)三个新场景全部先以临时探针实测真实行为、再定期望(探针测完即删,
禁止猜):

- **场景 36「多 Sheet Excel 上传」**(`excel-multi-sheet`):xlsx 含两个 sheet——
  第一个「工单表」是业务数据,第二个「员工表」是完全不同的表(一份工作簿装多个
  业务表的常见形态)。实测结局:入口按 pd.read_excel 默认 sheet_name=0 **只读第一个
  sheet**,第二个 sheet 被静默忽略——不报错、不提示其存在,列名/数据/全旅程全部只
  来自工单表,八关全过。对照事实(探针实测):把数据放在第二个 sheet(第一个 sheet
  是员工表)时,入口把员工表当数据读入(列名 员工号/姓名/部门),同样不报错——
  用户得不到「数据在其他 sheet」的提示。**定局:单 sheet 隐含限制是已知边界,不是
  被拦**——多 sheet 工作簿须用户自行把要分析的表放在第一个 sheet,入口不做 sheet
  选择。夹具由 openpyxl 现场生成(固定文档时间戳保证字节可复现;openpyxl 是入口读
  xlsx 的既有引擎依赖,未新增第三方依赖)。
- **场景 37「目标列数字型连续值」**(`numeric-continuous-target`):答案列是
  1.0/2.5/3.7 这类连续测量值(回归形态,非离散类别),全量含样例未覆盖的新测量值。
  实测结局:value_kind 按「均长<40 且去重数≤20」判成 **categorical——数值语义不被
  识别,回归任务当前被当多类别分类对待**:分布 finding 把每个测量值各计一类
  (「共 2 类:1.0×1、2.5×1」),预览/对比核验/盲标按精确字符串匹配(「1.0」原样
  进答案),全量 8 个新测量值触发 new_categories review 预警(对连续值而言是新值的
  常态提示,不阻断),八关全过。**定局:连续值回归未被如实对待为回归**——已知边界,
  需用户在数据侧离散化答案列或配置 Agent 走回归方案;场景钉住当前判定,日后若引入
  数值型 value_kind 须随之改期望。
- **场景 38「行序颠倒重传」**(`row-order-reversed-full`):全量与样例同源
  (编号 001-010)但行序完全颠倒。实测结局:内容级守卫**全部按内容而非行号**工作
  ——sample_answer_disagreement 按输入文本比对、new_categories 按答案值比对、冲突/
  重复检出按 canonical 内容,行序不影响任何判断,全量验证无 blocking,八关全过。
  血缘/版本(探针实测同一会话重传路径):行 ID 按物理行序重新编号(r000001 从 001
  变成 010)——**行身份是位置的,不是内容的**;重传后旧盲标核验按 full_source_digest
  失效(stale 如实标记,不静默复用),full/ 目录按内容寻址同时保留正序与倒序两份
  原始字节,重新确认+重新盲标(对新绑定从第 1 轮开始)后照常 verified,物化产生
  新版本并绑定重传文件摘要——**盲标核验按内容摘要稳定工作,行序不影响身份判断**。

矩阵 35→38,38/38 as_expected(意外通过/意外拦截/错误均为 0);test_scenario_matrix
23→26 项全绿,全集下限断言随之 35→38;ruff check/format clean。本批只动
scenario_specs.py 与 test_scenario_matrix.py(scenario_matrix.py 无需改动),
未触碰 A 线并行文件。

### 第 18 轮 = 多 Sheet 提示清偿 + 并行矩阵扩展(场景 39-40)

(B 线第九批。)上批场景 36 实测发现的真缺口——多 Sheet Excel 上传第二个 sheet 被
静默忽略、用户得不到任何提示——本批清偿,并以两个新场景钉住边界。全部结论先以
临时探针实测、再定期望(探针测完即删,禁止猜)。

- **多 Sheet 读取范围标注上线**(`sources.py`):读 xlsx/xls 时探测 sheet 清单
  (改用 `pd.ExcelFile` + `book.parse`,与原 `read_excel(BytesIO, sheet_name=0)`
  同一解析路径,**读取行为不变**——探针验证两种写法解析结果逐格相等),含多个
  sheet 时把「该文件含 N 个 sheet,仅读取第一个(名称);其余 N-1 个(名称)未读取」
  按内容摘要记下,`profile_source` 以 `sheet_note` 键如实呈现(单 sheet/CSV 无此键,
  profile 形状不变)。实现要点:SampleSource 契约不可增字段(extra="forbid",本批
  文件域不含 intake_models),故用摘要键控的进程内记忆——内容寻址、确定性一致,
  重启后历史会话已序列化的 profile 快照仍带标注,仅重新计算依赖同进程读取记录
  (局限如实记录)。全量侧同样标注;其余 sheet 超 5 个不逐一罗列、以「等」收尾。
  新增 tests/unit/test_sources.py(openpyxl 现场生成单/双/9-sheet 夹具)。
- **场景 36 真相更新**:`excel-multi-sheet` 的 expect_note 从「第二个 sheet 被静默
  忽略(不报错、不提示)」更新为「提示已上线,读取即可见」——矩阵记录的是当前真相。
- **场景 39「多 Sheet 数据在第二个 sheet」**(`excel-data-on-second-sheet`):
  第一个 sheet 是员工表、工单数据在第二个 sheet。实测结局:入口仍按 sheet_name=0
  把员工表当数据读入(列名 员工号/姓名/部门,2 行员工记录),create 不拦;旅程在
  **baseline_analysis 被拦**——「答案列『类别』不在数据字段中(可用:['员工号',
  '姓名', '部门'])」,被读入的列名原样列出。配合 sheet_note,数据放错 sheet 从
  「无声吞掉」变为 create 即告知「只读了员工表,工单表未读取」。定局:入口只读
  第一个 sheet、不做 sheet 选择是已知边界;修好数据后旅程可续。
- **场景 40「重复表头行两侧同现」**(`duplicate-header-rows-in-both`):任务初衷
  「样例(非全量)中部混入重复表头、与全量侧是否对称」**已由既有场景 30
  (duplicate-header-row-in-sample)实测定局**——不对称:样例侧不拦、按普通数据行
  读入,靠用户预览逐行核对。为不在矩阵里重复计一条行为,40 号覆盖此前未测的组合:
  样例与全量同时含重复表头行。实测结局:样例侧四关照常通过(表头行以「就绪」面目
  进预览并参与对比核验),全量侧 validate_full 硬拦「1 条记录与表头完全相同…」,
  **组合结局由全量侧决定**——全量硬拦兜底,两侧同时脏也不会带病物化。

矩阵 38→40,40/40 as_expected(意外通过/意外拦截/错误均为 0);test_scenario_matrix
26→28 项、test_sources 0→3 项全绿,全集下限断言随之 38→40;ruff check/format
clean。本批只动 sources.py、scenario_specs.py、test_scenario_matrix.py、新增
test_sources.py,未触碰 A 线并行文件(intake_service/页面/基线分析域均只读)。

### 第 19 轮 = 多 Sheet 可选读取 + 创建入口接 sheet 选择 + 并行矩阵扩展(场景 41)

(B 线第十批。)上批场景 39 的定局是「入口只读第一个 sheet、不做 sheet 选择」是已知
边界;本批清偿:sheet 选择上线,数据放在非第一个 sheet 的工作簿不再要求用户改文件。
全部结论先以临时探针实测、再定期望(探针测完即删,禁止猜)。

- **多 Sheet 可选读取**(`sources.py`):`read_source` 新增 `sheet` 参数——仅对
  Excel 有效,按名称或 1 起始序号指定工作表(CLI 传参皆为字符串,纯数字串同义序号;
  名称精确匹配优先),`None` 默认读第一个 sheet,读取行为不变。解析不到即报错并如实
  列出全部 sheet 名与序号口径,不静默回退;CSV/JSONL 传 sheet 明确拒绝。
- **标注正道修复**(撤上批的摘要键控进程内记忆):SampleSource 契约新增可选字段
  `sheet`(实际读取的工作表名)与 `sheet_note`(读取范围说明),标注随来源对象在
  读取时生成并持久化。上批按摘要键控的记忆有两个如实记录过的局限——重启后重算依赖
  进程内记录、且同一文件按不同 sheet 重复读取会串味(本批引入选择后成为真 bug):
  字段方案下标注随会话序列化,重启、apply_analysis 重算 profile、同摘要不同选择
  全部各自如实;旧存档缺字段按默认空值回读,既有 profile 快照原样保留。
  `profile_source` 直接取用来源携带的 `sheet_note`,默认路径措辞不变。
- **创建入口接 sheet 选择**(服务层+CLI,页面入口不动,页面改动属其他 agent 域):
  `IntakeService.create` 透传 sheet;`validate_full_data` 同样接收,且复用已声明
  全量的原文件时沿用来源持久化的 sheet(用户显式传入优先)——按第二个 sheet 建的
  任务做全量复验不会静默退回第一个 sheet。CLI create 与 full-validate 新增
  `--sheet`(名称或序号,1 表示第一个;默认第一个)。
- **场景 41「数据在第二个 sheet + --sheet 指定后旅程走通」**
  (`excel-second-sheet-selected`):与场景 39 同源字节(员工表在前、工单数据在第二个
  sheet),sample_sheet/full_sheet 指定序号 2。实测结局:入口按序号读取第二个 sheet
  「工单表」,八关全过——create 不拦,基础分析/对比核验/样例确认照常,全量验证 10 行
  工单记录无 blocking/review,盲标 5/5 一致,物化 train 8/validation 1/test 1。读取
  范围如实标注:样例与全量 profile sheet_note 均为「该文件含 2 个 sheet,按指定读取
  『工单表』;其余 1 个(员工表)未读取」;同一份字节不带 --sheet 时标注仍是「仅读取
  第一个『员工表』」——同摘要不同选择不串味(标注随来源对象走的直接证据)。边界如实
  记录:sheet 选择当前在服务层与 CLI,页面入口待接入。
- **场景 36/39 expect_note 真相更新**:「入口只读第一个 sheet、不做 sheet 选择」的
  已知边界改写为当前真相——默认仍读第一个,sheet 选择已上线(服务层/CLI --sheet),
  指向场景 41;矩阵记录的是现在,不是历史。

矩阵 40→41,41/41 as_expected(意外通过/意外拦截/错误均为 0);test_scenario_matrix
28→29 项、test_sources 3→11 项,全集下限断言随之 40→41;ruff check/format clean。
本批按任务要求动了 sources.py、intake_models.py、intake_service.py(仅 create/
validate_full_data 接线)、scripts/data_intake.py(仅 create/full-validate 的
--sheet 参数与透传)、scenario_matrix.py(仅 ScenarioSpec 字段与透传)、scenario_specs
.py、test_sources.py、test_scenario_matrix.py;与 A 线同在 data_intake.py 的
label-verify 提交(c257ce6/337328c/1f06ea2)先后落库无冲突。全量回归 tests/unit
1672 passed(--no-cov)。遗留下一轮候选:页面入口接 sheet 选择(A/页面域)、
add-source/full-sources 的 sheet 参数对称补齐、服务层 sheet 选择的契约字段
`sheet` 若日后需序号回显可另议。

### 第 20 轮 = 盲标题目清单 CSV 导出 + agent-setup 盲标 CLI 用法逐字校对 + 页面域并行(第十一批)

(B 线第十一批。)承接第十批 A 线把盲标 CLI 的证据说明补齐(evidence_note 预告、
判定行下界、submit_hint 对齐),本批清偿「抽题只能在终端看题」的 offline 缺口,
并把文档里盲标 CLI 的用法从简写升级为与实现逐字一致的完整用法。

- **label-verify 加可选 `--export-csv`**(05ef05a,scripts/data_intake.py):与
  learnability-probe 的候选清单导出同款口径——`--export-csv 路径` 抽题成功后把
  题目清单落盘(BOM 表头、Excel 直开、父目录自动创建、stderr 告知导出路径),
  供线下作答后再逐条照抄回 label-verify-submit 提交。关键差别是盲标清单**只含
  行ID、题目输入、留空待填的盲标答案三列,绝不含数据标签**——导出文件一旦泄露
  答案,盲标核验「独立复现业务含义」的价值就失效了;抽题被拒(如 revision 过期)
  时不写文件也不装作导出成功。测试 test_label_verify_cli 5→7 项:清单内容与
  抽样逐行一致(含真实输入列)、数据标签不泄露、行数=样本量+表头、失败不落盘。
- **agent-setup.md 语义安全层段补盲标核验完整 CLI 用法**(9bc0382):新增
  「盲标核验的完整 CLI 用法」小节——label-verify(--size 1–50、--export-csv)
  与 label-verify-submit(每行一个 --answer、恰好覆盖抽样行、不收 --revision)
  全参数;stderr 输出顺序逐字引用(提示句、evidence_note 证据预告、
  shortfall_note、逐题清单),判定行「判定：verified（5/5 一致，95% 置信下界约
  57%）」与通过/未通过两态注记照实写明;导出 CSV 三列结构与「清单不含数据答案」
  边界一并入文。顺带清偿两处文档漂移:①关卡条目早期口径「系统随机抽取最多
  5 条已标注行」改为当前真相(样本量 1–50 自选,默认 5),末尾过渡注记随之删除;
  ②原条目 `[--size 5]` 的写法暗示样本量固定。全部引用字符串均以真实 CLI 进程
  跑 label-verify/label-verify-submit 逐字比对(stderr 与 JSON 键),不凭记忆书写。
- **本记录**(第三提交)。

**A 线(页面域并行 agent,交付后补记):**页面三处 sheet 入口全部落地
(ca2a09d/94345f6/b5d7d48),第 19 轮遗留候选中的「页面入口接 sheet 选择」与
「add-source 的 sheet 参数对称补齐」就此清偿;full-sources 的 sheet 参数仍留作
后续候选。三处同款约束:上传控件移到表单外——表单内部件要到提交才提交值,
放里面就无法在提交前按上传的文件类型显示 sheet 选择;仅 Excel 上传显示可选
sheet 输入,留空读第一个 sheet,按名称或 1 起始序号指定。

- **新建任务表单接 sheet 选择**(ca2a09d):「文件读取设置」expander 内新增可选
  sheet 输入,透传 `service.create`。
- **补充资料入口对称接 sheet**(94345f6):「保存补充资料」与新建任务同款;
  `service.add_source` 增加可选 sheet 参数(签名与 read_source 透传,已确认的
  最小增量改动),主来源摘要不受影响。
- **全量文件读取设置接 sheet**(b5d7d48):走查发现的真缺口——validate_full_data
  服务层与 CLI --sheet 已就绪,页面「全量文件读取设置」却不能指定 sheet
  (全量数据在第二个 sheet 时页面上无解);与新建任务同款处理,透传
  `validate_full_data`。

测试 test_data_intake_ui 用 openpyxl 现场生成多 Sheet 工作簿,三处入口各实测:
选第二个 sheet 后新建/补充资料读到员工表数据、主来源摘要不受影响,全量验证
通过并进入 review_full_data(列与就绪行数均来自全量表);并覆盖未上传、CSV
上传时不显示 sheet 输入。预期中的 test_workbench_training_ui 未涉及——本批
交付只落在数据入口域,页面盲标/训练工作台方向无改动。A 线三提交各自
ruff check/format clean;并入后全量回归 tests/unit 1680 passed(--no-cov,
较 B 线记录的 1674 净增 6 项 sheet 入口测试)。

回归:tests/unit 1674 passed(--no-cov;第 19 轮 1672 → +2 为本轮新增 label-verify
导出测试);ruff check/format clean。本批只动 scripts/data_intake.py(仅 label-verify
的参数、导出 helper 与分支接线)、tests/unit/test_label_verify_cli.py、
docs/agent-setup.md 与本记录;未触碰 A 线并行文件(页面与 UI 测试域只读)。

### 第 21 轮 = full-sources CLI 补齐 --sheet 对称参数(第十二批后、恢复循环第 1 轮)

(恢复的北极星打磨循环,第 1 轮。)第 20 轮 A 线遗留候选「full-sources 的 sheet
参数仍留作后续候选」就此清偿。sheet 选择至此四个入口全对称:create / add-source /
full-validate / full-sources(CLI+页面)。缺口收窄过程有证据:服务层
`validate_full_sources(sheets=...)` 与页面表单早已就绪(test_composed_intake 的
服务级 sheet 测试、07 页面表单透传),唯独 CLI 子命令没有 `--sheet` 参数——多
Sheet Excel 的全量资料在 CLI 路径上无解。

- **full-sources 加 `--sheet ALIAS=名称或序号`(可重复)**:与同命令 `--source
  ALIAS=PATH` 同款解析风格——`partition("=")` 拆别名与值,格式错误、别名重复
  逐一给出中文提示(分派级 `raise ValueError` → 上游捕获 → exit 2 无 Traceback,
  会话 revision 不变)。help 文案与 full-validate 的 `--sheet` 同口径。序号走
  read_source 的 1 起始序号解析,无需 CLI 侧特判。
- **测试 test_multisource_cli 11→15 项**:新增本地 `_workbook_bytes`(与
  test_composed_intake 同款 openpyxl 内存工作簿)——正路径用 main 数据在第二个
  sheet(不指定就读到说明表)、labels 用序号 `labels=1` 指定,断言 ready==3、
  `full_data.sources` 各自记录实际读到的 sheet 名、目标类别集合正确;参数化
  4 种非法 `--sheet`(缺 `=`、别名重复、指向未提供的资料、不带 `--source`
  单用 sheet)全部 exit 2、revision 不变、无 Traceback。
- 回归:test_multisource_cli + test_composed_intake 17 passed、test_full_data_cli +
  test_sources 16 passed(--no-cov);ruff check/format clean。本批只动
  scripts/data_intake.py(full-sources 的参数与分派)、tests/unit/test_multisource_cli.py
  与本记录。

### 第 22 轮 = 连续数值目标如实标注 numeric_continuous(恢复循环第 2 轮)

(恢复的北极星打磨循环,第 2 轮。)场景 37(numeric-continuous-target)此前钉住的
是**不诚实现状**:回归式答案列(1.0/2.5/3.7 这类测量值)被静默判成 categorical
按逐字分类学习,全量新测量值被报成「新类别」——spec 里「日后若引入数值型
value_kind 需随之改期望」的欠条本轮兑现。核心原则与盲标/对比核验同源:业务语义
判断不静默通过。

- **value_kind 增加 `numeric_continuous`**(intake_models.py):FieldBinding 描述同步
  ——「带小数的连续数值」「只如实标注答案形态,当前训练仍按逐字字符串学习,不是
  数值回归」。四个消费方(business_evaluation/acceptance/data_intake/07 页面)都按
  `== "categorical"` 精确匹配走分类严格评分,numeric 自然落入开放任务人工核对口径
  ——不伪造回归准确率,与既有 open_text 行为一致。
- **保守判定**(baseline_analysis.py):`_is_continuous_number` 要求带小数点且可
  float——整数编码(0/1/2)不算;`_value_kind` 顺序 open_text(平均长度)→
  numeric_continuous(**全部**非空取值均为此形态)→ categorical(去重 ≤20)→
  unspecified。判定即给诚实 finding:逐字字符串学习(「1.0」≠「1.00」)、不是数值
  回归、评测只能逐字比对、需数值误差容差请先在数据侧离散化;小数若是离散编码
  (如版本号)照常逐字核对即可。
- **全量验证新问题码 `numeric_new_values`**(full_data.py,review 级不阻断):
  连续目标全量出现新测量值是常态,不再按「新类别」表述,但如实列出并重申逐字学习
  边界;new_target_values 照常记录,不静默跳过。
- **Agent 提示词同步**(agent/intake.py):value_kind 三分类规则 + 「Agent 不得把
  数值回归承诺成可按误差评分的任务」。
- **场景 37 重写为诚实版**(scenario_specs.py + test_scenario_matrix.py):value_kind
  断言 numeric_continuous、诚实 finding 含「不是数值回归/逐字」、numeric_new_values
  含「8 个样例未覆盖的新测量值/逐字」、new_target_values 八个值原样、分布 finding
  「共 2 类」保留、八关全过 verdict as_expected。剩余边界如实入 spec:整数测量值/
  编码仍走类别路径。
- 回归:目标套件(scenario_matrix+baseline_analysis+full_data+forecast+temporal)
  68 passed;全量 tests/unit 在本机环境排除 46 个缺可选依赖(datasets/peft/plotly)
  的模块后 621 passed 37 skipped 0 failed。环境注记:本机 venv 未装 datasets 等可选
  依赖,46 个模块收集期/运行期 ModuleNotFoundError(全部先于本批存在,与本批无关;
  其中 test_tracking/test_models_base 依赖其他模块导入暖场,排除暖场模块后连带
  失败,已一并列入环境排除清单)。ruff check/format clean。本批只动
  src/workbench/intake_models.py、src/workbench/baseline_analysis.py、
  src/workbench/full_data.py、src/agent/intake.py、src/workbench/scenario_specs.py、
  tests/unit/test_scenario_matrix.py 与本记录。

### 第 23 轮 = add-source CLI 补 --sheet + sheet/numeric_continuous 文档补齐(恢复循环第 3 轮)

(恢复的北极星打磨循环,第 3 轮。)起点是文档补齐——第 19-21 轮的 sheet 选择与
第 22 轮的 numeric_continuous 在 agent-setup.md 里零记载;核查文档的过程中发现
一处真实代码缺口并入本批:CLI add-source 没有 `--sheet`,而服务层
(`add_source(sheet=...)`,第 20 轮 A 线)与页面(excel_sheet_input)早已支持——
第 21 轮记录的「sheet 选择至此四个入口全对称:create / add-source /
full-validate / full-sources」对 add-source CLI 而言当时并不成立(该轮只补了
full-sources)。历史记录不改写,本条如实更正:对称性在本轮补齐 add-source 后
才真正成立。

- **add-source CLI 补 `--sheet`**(scripts/data_intake.py):parser 与 create/
  full-validate 同款 help 串「读取 Excel 的指定 sheet（名称或序号，1 表示第一个；
  默认第一个）」,分派透传 `sheet=args.sheet`。多 Sheet Excel 的补充资料在 CLI
  路径上从此可指定 sheet,不再只读第一个。
- **测试 test_multisource_cli 11→12 项**:新增 add-source --sheet 正路径——补充
  资料真实数据在第二个 sheet(首 sheet 是说明表),按名称指定后会话记录的
  columns/sheet/sheet_note 全部来自指定表(「按指定读取「类别表」」);同一份字节
  不带 --sheet 再补充一次,读到说明表、标注「仅读取第一个「说明」」——同摘要
  不同选择不串味在 add-source 路径同样成立。
- **agent-setup.md 文档补齐**(三处):①「多份资料一起分析」的 full-sources 示例
  加 `--sheet main=工单表 --sheet labels=1`,新增一段 sheet 选择说明——四个入口
  (create/add-source/full-validate/full-sources)+页面可选输入、默认只读第一个
  sheet、读取范围随来源持久化并如实标注、复用原文件沿用持久化选择、指定不存在
  报错列全部 sheet 名、CSV/JSONL 明确拒绝;②「单份资料」的 full-validate 支持
  参数列表补 `--sheet`;③新增小节「连续数值答案的如实边界」——numeric_continuous
  如实标注、逐字字符串学习(「1.0」≠「1.00」)、不是数值回归、需数值误差先离散化、
  整数编码不受影响、numeric_new_values 常态提示不阻断、评测落入开放任务人工核对
  口径。QUICKSTART/README 经查无读取选项内容,agent-setup.md 是唯一文档目标。
- **测试 test_readme_alignment 14→16 项**:新增两个钉测试——(a)sheet 文档与
  真实 CLI help 同步:create/add-source/full-validate 三入口同款 canonical help
  串、full-sources 的 `--sheet ALIAS=名称或序号` 与「1 起始序号」、agent-setup
  的「四个入口/如实标注/CSV/JSONL 明确拒绝/full-validate 三参数」关键句;
  (b)连续数值边界:numeric_continuous/不是数值回归/逐字/1.0≠1.00/
  numeric_new_values/离散化/整数编码关键句,并由场景矩阵
  numeric-continuous-target 场景真实存在背书。
- 回归:test_multisource_cli 12 passed、test_full_data_cli+test_sources+
  test_composed_intake 22 passed、test_readme_alignment 16 passed(均 --no-cov);
  ruff check/format clean。本批只动 scripts/data_intake.py(add-source 的参数与
  透传)、tests/unit/test_multisource_cli.py、tests/unit/test_readme_alignment.py、
  docs/agent-setup.md 与本记录。

### 第 24 轮 = Excel 合并单元格如实点名 + 矩阵场景 42(恢复循环第 4 轮)

(恢复的北极星打磨循环,第 4 轮。)核心痛点:Excel 里「同一类别只写一次然后下拉
合并」是业务人员做表的常见形态,读取时合并区除左上角外均读为空串——用户在样例
确认或全量验证被「缺少监督答案」拦下时,看到的只是一堆空值,无从知道根因是自己
的 Excel 合并。与多 Sheet 静默忽略(第 18 轮)同源:事实不可见,不是判断错误。

- **merged_note 如实点名**(`sources.py` + `intake_models.py`):探针实测 pd.ExcelFile
  走 openpyxl 只读模式,ReadOnlyWorksheet 没有 merged_cells——检测需要第二次
  `load_workbook(read_only=False)` 完整加载(仅 xlsx;xls 的 xlrd 引擎不提供合并
  范围,如实不检测)。读取的 sheet 存在与数据区相交的合并区(行 1..行数+1、列
  1..列数;完全在数据区之外或未读取 sheet 的合并不列入)时,来源携带 `merged_note`:
  「该 sheet 含 1 处合并单元格(类别 C2:C4):合并区除左上角外均读为空值,涉及答案列
  时这些行会按缺少监督答案处理。请取消合并并逐行填写受影响的值;没有自动填充。」
  点名受影响列的表头与坐标(超过 5 处以「等」收尾,与 sheet_note 同口径);随
  SampleSource 契约字段持久化(第 19 轮正道方案,不用进程内记忆),profile 同步呈现,
  存档回读仍在。读取行为不变:合并区除锚点外读空串,**不自动填充**——是否取消合并
  由用户决定(语义安全原则:不猜业务语义)。
- **矩阵场景 42「目标列纵向合并」**(`merged-cells-in-target-column`,先探针后定局):
  三变体探针实测——样例合并(C2:C4,锚点 质量 保留、002/003 空读):create 不拦、
  merged_note 即时点名,对比核验不拦(两条已标注行 001/004 答案不同足以配对),旅程
  **在样例确认被拦**「当前方案仍有业务问题、缺标签或转换问题,不能确认数据就绪。」
  ——这是矩阵首个 confirm_sample 关场景(此前 24/25/27 都拦在更早的对比核验);
  全量合并(C6:C8)且样例干净:走到全量验证被硬拦「全量存在缺少监督答案的记录,
  需要补充标签;没有自动生成真值。(2 条)」(与场景 19 同守卫,根因不同);对照探针
  (同样空答案但不合并)拦在同一关同一条报错——**拦截本身是通用缺标签门,与合并
  无关;合并场景的独有价值是 merged_note 把根因点名给用户**。expect=`blocked_at:
  confirm_sample`,测试另钉:profile 与来源的 merged_note 一致、全量来源空读 2 行
  且点名 类别 C6:C8。矩阵 41→42,42/42 as_expected。
- **顺手清偿一处过时标签**:high-cardinality-target 的 tags 含 "known-gap",但其
  expect_note 早已写明「高基数目标早期警告已上线(M5/task-001)…预警不阻断」——
  标签与自己的说明矛盾,移除。grep 确认 specs 内不再有 known-gap;forecast 场景的
  known-limitation 保留(泄漏预警+无时间契约仍是该夹具的真实结局)。
- **文档钉死**(agent-setup.md + test_readme_alignment 16→17):多份资料小节新增
  合并单元格段——merged_note 逐字示例、合并区除左上角外读空值、没有自动填充、
  数据区之外/未读取 sheet 不列入、xls 不检测;钉测试断言关键句且由场景矩阵
  merged-cells-in-target-column 真实存在背书。
- 回归:test_sources 11→15(空读不填充/区域与 sheet 过滤/前五上限/服务层持久化)、
  test_scenario_matrix 29→30(全集下限 41→42)、test_readme_alignment 16→17;
  邻接套件(multisource/full_data/composed/data_intake/baseline_analysis/full_data 域)
  83 passed;ruff check/format clean。全量回归 **tests/unit 1706 passed / 0 failed**
  (--no-cov,无排除,171.96s)。**环境更正(本轮核实,第 22 轮环境注记就此作废):**
  本机 venv 现已装 datasets/peft/plotly(venv/bin/python 逐一 import 干净),
  第 22 轮的 46 模块排除清单(/tmp/exclude_final.txt)已过时——收集期对比实测
  1706(无排除)vs 1705(带该清单,标志实际未生效),真实全集 1706 项即本轮回归
  基线;此前「621 passed 37 skipped」是排除清单生效时代的数字,不是当前真相。
  本批只动 src/workbench/intake_models.py、src/workbench/sources.py、
  src/workbench/scenario_specs.py、tests/unit/test_sources.py、
  tests/unit/test_scenario_matrix.py、tests/unit/test_readme_alignment.py、
  docs/agent-setup.md 与本记录。

### 第 25 轮 = Excel 公式无缓存如实点名 formula_note + 矩阵场景 43(恢复循环第 5 轮)

(恢复的北极星打磨循环,第 5 轮。)核心痛点:报表工具/脚本写出的 xlsx(以及任何
未经真实 Excel 打开保存过的文件)公式格没有缓存计算结果——openpyxl 不计算公式,
这些格读为空串。用户在样例确认或全量验证被「缺少监督答案」拦下时,看到的只是
一堆空值,无从知道根因是公式无缓存。与多 Sheet 静默忽略(第 18 轮)、合并单元格
静默空读(第 24 轮)同源:事实不可见,不是判断错误。

- **formula_note 如实点名**(`sources.py` + `intake_models.py`):探针实测
  `load_workbook(read_only=False)`(data_only 默认 False)暴露 `cell.data_type=='f'`
  与坐标;再开一次 `data_only=True` 比对,值非 None 的即有缓存——真实 Excel 保存
  过的公式格按缓存值正常读取、不列入(无缓存误报为零)。数据区存在无缓存公式格时
  来源携带 formula_note:「该 sheet 含 2 个没有缓存计算结果的公式单元格(类别 C2、
  类别 C3):这些公式读为空值,涉及答案列时这些行会按缺少监督答案处理。请用 Excel
  等软件打开并保存以生成计算结果;没有自动计算。」(超过 5 个以「等」收尾,与
  sheet_note/merged_note 同口径);随 SampleSource 契约字段持久化(第 19 轮正道),
  profile 同步呈现。实现与 merged_note 共享一次完整加载(`_excel_notes` 一次返回
  两注),data_only 二次加载仅在实际存在公式格时发生;仅 xlsx(xls 引擎不提供
  公式清单,如实不检测)。读取行为不变:**不自动计算**——是否用 Excel 打开保存
  生成缓存值由用户决定(语义安全原则:不猜业务语义)。
- **矩阵场景 43「目标列公式无缓存」**(`formula-cells-in-target-column`,先探针后
  定局):样例侧 C2/C3 无缓存公式(空值在前两条)——create 不拦、formula_note
  即时点名,对比核验不拦(两条已标注行 003/004 答案不同足以配对),旅程**在样例
  确认被拦**「当前方案仍有业务问题、缺标签或转换问题,不能确认数据就绪。」;全量
  侧公式(C6:C8)且样例干净:走到全量验证被硬拦「全量存在缺少监督答案的记录,
  需要补充标签;没有自动生成真值。(3 条)」(与场景 19 同守卫,根因不同);对照
  探针(同样空答案但不含公式)拦在同一关同一条报错——**拦截本身是通用缺标签门,
  与公式无关;公式场景的独有价值是 formula_note 把根因点名给用户**。
  expect=`blocked_at:confirm_sample`,测试另钉:profile 与来源的 formula_note 一致、
  全量来源空读 3 行且点名 类别 C6/C8。两个对照事实均探针实测并入 expect_note:
  ①有缓存值不误报——测试用 XML 补丁夹具钉死:解压 openpyxl 写出的 xlsx,把公式格
  的空 `<v />` 补成 `t="str"` 与缓存值(正是 Excel 保存后的形态),断言找不到目标格
  即失败,防 openpyxl 输出漂移让夹具静默失效;②分组列(编号)的公式同样读空——
  影响不止答案列。矩阵 42→43,43/43 as_expected。
- **文档钉死**(agent-setup.md + test_readme_alignment 17→18):多份资料小节新增
  公式格段——formula_note 逐字示例、这些公式读为空值、没有自动计算、带缓存值
  正常读取不列入、分组列公式同样读空、数据区之外/未读取 sheet 不列入、xls 不检测;
  钉测试断言关键句且由场景矩阵 formula-cells-in-target-column 真实存在背书。
- 回归:test_sources 15→20(空读不计算/缓存值不误报/区域与 sheet 过滤/前五上限/
  服务层持久化)、test_scenario_matrix 30→31(全集下限 42→43)、test_readme_alignment
  17→18;邻接套件(multisource/full_data/composed/data_intake/baseline_analysis 域)
  83 passed;ruff check/format clean。全量回归 **tests/unit 1713 passed / 0 failed**
  (--no-cov,无排除,168.49s)。本批只动 src/workbench/intake_models.py、src/workbench/
  sources.py、src/workbench/scenario_specs.py、tests/unit/test_sources.py、
  tests/unit/test_scenario_matrix.py、tests/unit/test_readme_alignment.py、
  docs/agent-setup.md 与本记录。

### 第 26 轮 = Excel 隐藏行/列如实点名 hidden_note + 矩阵场景 44(恢复循环第 7 轮)

(恢复的北极星打磨循环,第 7 轮。)核心痛点:AutoFilter 筛选后直接保存、或手工隐藏
旧行/列的 Excel,读取时隐藏行/列照常进数据——用户以为「筛选掉的就是排除了」,
实际 Excel 里看不到的行全部进入分析与训练,隐藏列也仍出现在可用字段中。与多
Sheet 静默忽略(第 18 轮)、合并单元格静默空读(第 24 轮)、公式无缓存静默空读
(第 25 轮)同源:事实不可见,不是判断错误。

- **hidden_note 如实点名**(`sources.py` + `intake_models.py`):探针实测
  pd.read_excel/ExcelFile 把隐藏行/列当普通数据读入,openpyxl 完整加载
  (`load_workbook(read_only=False)`,与 merged/formula 检测共享同一次加载,
  `_excel_notes` 因此从二元组扩为三元组,隐藏检测零额外 IO)暴露
  `row_dimensions[N].hidden` 与 `column_dimensions[字母].hidden`。数据区(行
  2..行数+1、列 1..列数——表头行与数据区之外不列入)存在隐藏行/列时,来源携带
  hidden_note:「该 sheet 含 2 个隐藏行（第 3 行、第 4 行）：隐藏行照常读入——
  Excel 中看不到的行也会进入分析与训练。请取消隐藏并删除不需要的行；没有自动
  排除。」隐藏列按表头名列出、追加第二段(隐藏列照常读入、仍出现在可用字段中);
  行/列各自超过 5 个以「等」收尾(与 sheet_note/merged_note/formula_note 同口径);
  未读取 sheet 的隐藏不列入;随 SampleSource 契约字段持久化(第 19 轮正道),
  profile 同步呈现。读取行为不变:**不自动排除**——隐藏行可能是有意保留的数据
  (筛选只是视图),是否取消隐藏、删除不需要的行列由用户决定(语义安全原则:不猜
  业务语义)。仅 xlsx(xls 引擎不提供隐藏标志,如实不检测)。
- **矩阵场景 44「隐藏行照常读入」**(`hidden-rows-in-sheet`,先探针后定局):
  样例第 3/4 行隐藏(002/003,AutoFilter 筛选后保存的形态)且隐藏行带有效标签
  (物流/质量),全量第 7/8 行隐藏(006/007)。实测结局:create 不拦、hidden_note
  即时点名,隐藏行照常读入参与全部旅程——对比核验用答案不同的已标注行配对,
  样例确认/全量验证/盲标/物化全部通过,**八关全过,expect=passes**。定局:**披露
  不阻断**——与前两轮(合并/公式→缺标签硬拦)方向相反但原则同一:合并区非首格
  与无缓存公式读为空值,走通用缺标签门拦下,根因披露是锦上添花;隐藏行是完整
  有效数据,拦下反而错(用户可能有意保留),hidden_note 的价值是把「Excel 里
  看不到却在训练里」的事实点名给用户,去留由用户裁决。对照事实(探针实测并入
  expect_note):隐藏列同样照常读入且按表头名列出;表头行/数据区之外/未读取
  sheet 的隐藏不列入;xls 不检测。测试另钉:夹具真实隐藏由 openpyxl 重读复核
  (防夹具漂移成普通表)、create 后 002/003 都在数据里、profile 与来源的
  hidden_note 一致。矩阵 43→44,44/44 as_expected。
- **文档钉死**(agent-setup.md + test_readme_alignment 18→19):多份资料小节新增
  隐藏行/列段——hidden_note 逐字示例、隐藏行照常读入(Excel 中看不到的行也会
  进入分析与训练)、隐藏列仍出现在可用字段中、没有自动排除、表头行/数据区
  之外/未读取 sheet 不列入、xls 不检测;钉测试断言关键句且由场景矩阵
  hidden-rows-in-sheet 真实存在(expect=passes)背书。
- 回归:test_sources 20→25(照常读入/隐藏列按表头名/区域与 sheet 过滤/前五上限/
  服务层持久化)、test_scenario_matrix 31→32(全集下限 43→44)、test_readme_
  alignment 18→19;邻接套件(multisource/full_data/composed/data_intake/
  baseline_analysis 域)83 passed;ruff check/format clean。全量回归 **tests/unit
  1720 passed / 0 failed**(--no-cov,无排除,170.44s)。本批只动 src/workbench/
  intake_models.py、src/workbench/sources.py、src/workbench/scenario_specs.py、
  tests/unit/test_sources.py、tests/unit/test_scenario_matrix.py、
  tests/unit/test_readme_alignment.py、docs/agent-setup.md 与本记录。

### 第 27 轮 = Excel 四条如实标注上页面渲染(恢复循环第 8 轮)

(恢复的北极星打磨循环,第 8 轮。)核心痛点:第 18/24/25/26 轮建成的四条 Excel
如实标注(sheet_note/merged_note/formula_note/hidden_note)只在 CLI 输出与
profile JSON 里可见——页面任务视图只渲染 scope_note,全量报告的标注埋在
「全量原始记录与结构变化」expander 的 st.json 深处。grep 核实页面零渲染。
对非专家目标用户(北极星:雇不起算法工程师的人)等于披露链路断了最后一公里:
服务层如实点名了,用户看不见。

- **show_excel_fact_notes 渲染助手**(07_Data_Intake.py):固定键序遍历
  (sheet_note→merged_note→formula_note→hidden_note),sheet_note 用 st.info
  (说明读取范围,信息级),其余三条用 st.warning(影响数据事实的告知级);
  profile(dict,.get)与 SampleSource(对象,getattr——pydantic 无 .get)同键
  统一取用;空标注跳过。三个渲染位点:(1) 任务视图主来源画像 scope_note 下
  (session.profile);(2) 全量验证报告来源 caption 下(report.profile,报告
  stale 与否都渲染——披露不因失效而消失);(3)「原始资料与补充文件」区每份
  资料各自渲染,多资料(>1)时每条标注带资料名前缀「labels:…」区分归属。
- **UI 测试三件**(test_data_intake_ui.py 22→25):双 sheet+合并区+隐藏行
  夹具一簿触发三注;(a) 新建任务后主来源渲染——CSV 任务四条负例 + Excel 任务
  merged/hidden warning 与 sheet info 正例;(b) 全量报告位点——隐藏行带有效
  标签的 xlsx 全量验证通过(expect=passes 形态)且 hidden_note 在 warning 里;
  (c) 补充资料位点——labels 别名前缀断言。**AppTest 教训**:selectbox 选项
  来自上一次渲染,run 之后才 service.create 的会话不在旧选项里,.select 新 id
  会静默渲染回旧会话——负例断言「零 warning」全过而正例落空,极具迷惑性;
  夹具会话必须在首次 page.run() 前建好(测试内已留注释钉死)。
- **文档钉死**(agent-setup.md + test_readme_alignment 19→20):多份资料小节
  新增渲染位点段——四条标注不埋进 JSON、三处位点(主来源画像下/原始资料区
  带前缀/全量报告来源行下)、info/warning 分级、CSV/JSONL 一条都不渲染;钉
  测试断言关键句。本轮纯 UI,不改读取行为,无矩阵场景新增。
- 回归:test_data_intake_ui 22→25、test_readme_alignment 19→20;ruff
  check/format clean。全量回归 **tests/unit 1724 passed / 0 failed**
  (--no-cov,无排除,169.98s)。本批只动 ui/pages/07_Data_Intake.py、
  tests/unit/test_data_intake_ui.py、tests/unit/test_readme_alignment.py、
  docs/agent-setup.md 与本记录。

### 第 28 轮 = 空行/全空行如实点名 blank_note + 矩阵场景 45(恢复循环第 9 轮)

(恢复的北极星打磨循环,第 9 轮。)核心痛点:空行处理是披露系列里唯一跨格式、
且三种格式行为互不相同的事实——CSV/JSONL 的空行读取时被静默跳过(不进数据,
用户不知道有没有丢行);Excel 数据区中部的全空行照常读入为全空记录(每列空串),
旅程在样例确认被通用缺标签门拦下、全量侧被 invalid+missing_group_values 双
blocking 拦下;Excel 尾部空行在解析时自然消失。三种形态都不在记录里留下痕迹,
用户被拦时无从知道根因。与多 Sheet 静默忽略(第 18 轮)、合并/公式/隐藏静默
(第 24-26 轮)同源:事实不可见,不是判断错误。

- **blank_note 跨格式如实点名**(`sources.py` + `intake_models.py`):探针实测
  三态定局后实现。双口径对应两种真实行为——CSV/JSONL「已跳过」:「该文件有
  3 个空行（第 3 行、第 5 行、第 6 行）已跳过——空行不进入分析与训练。请核对
  空行位置是否丢了数据；没有自动补行。」(读取行为不变,csv.reader 空行
  `not values` 跳过时收集行号、JSONL 纯空白行 `not line.strip()` 同计);
  Excel「照常读入」:「该 sheet 有 1 个全空行（第 3 行）照常读入——全空行按
  缺少监督答案与分组标识处理，会在样例确认或全量验证被拦下。请删除空行或补全
  数据；没有自动排除。」(基于解析后记录 `all(is_missing(v))` 判定,因此 **xls
  同样检测**——与 merged/formula/hidden 仅 xlsx 不同)。行号超过 5 个以「等」
  收尾(与前四注同口径);Excel 尾部空行解析时自然消失、无从检测,不列入
  (如实边界)。随 SampleSource 契约字段持久化,profile 同步呈现;不自动补行、
  不自动排除,处理由用户决定。
- **UI 助手升级 show_fact_notes**(07_Data_Intake.py):原 show_excel_fact_notes
  改名——blank_note 跨格式后「Excel 事实」名不副实;键序元组扩为五注
  (sheet→merged→formula→hidden→blank),sheet 用 info、其余四条 warning,
  三个渲染位点与签名不变。
- **矩阵场景 45「全空行照常读入」**(`blank-rows-in-sheet`,先探针后定局):
  样例第 3 行全空、全量 005 后插全空行(复用 `_hidden_xlsx` 生成器,无隐藏参数
  时即普通 xlsx 夹具)。实测结局:create 不拦、blank_note 即时点名,旅程在样例
  确认被拦「当前方案仍有业务问题、缺标签或转换问题,不能确认数据就绪。」
  (通用缺标签门,与公式/纯缺标签同关同错——拦截本身与空行无关,blank_note 的
  价值是点名根因),**expect=blocked_at:confirm_sample**。全量侧对照实测:样例
  干净时全量含全空行在 validate_full 被双 blocking 拦下(invalid「无法按已确认
  规则转换」+ missing_group_values「缺少方案声明的分组标识」);两侧同现时样例
  侧更早拦截。对照事实(入 expect_note):CSV/JSONL 跳过口径点名行号(纯空白行
  也计);尾部空行无从检测;xls 同样检测。测试另钉:夹具全空行由 openpyxl 重读
  复核(整行 cell.value 全 None,防夹具漂移)、create 后全空记录在第 3 行每列
  空串、profile 与来源的 blank_note 一致。矩阵 44→45,45/45 as_expected。
- **文档钉死**(agent-setup.md + test_readme_alignment 20→21):多份资料小节
  新增空行段——blank_note 双口径(已跳过/照常读入为全空记录)、空行不进入分析
  与训练、缺少监督答案与分组标识关联、没有自动排除、尾部空行自然消失不列入、
  xls 同样检测;渲染段「四条」改「五条」并修正 CSV/JSONL 边界句(sheet/merged/
  formula/hidden 一条都不渲染,blank_note 跨格式、空行事实同样渲染);钉测试
  断言关键句且由场景矩阵 blank-rows-in-sheet 真实存在背书;渲染位点钉测试补
  「空行事实同样渲染」断言。
- 回归:test_sources 25→31(CSV 跳过+行号/JSONL 含纯空白/Excel 照常读入+尾部
  边界/前五上限等/三格式干净负例/服务层持久化)、test_scenario_matrix 32→33
  (全集下限 44→45)、test_data_intake_ui 25(夹具升级四注一簿+blank 断言)、
  test_readme_alignment 20→21;ruff check/format clean。全量回归 **tests/unit
  1732 passed / 0 failed**(--no-cov,无排除,171.68s)。本批只动
  src/workbench/intake_models.py、src/workbench/sources.py、
  src/workbench/scenario_specs.py、ui/pages/07_Data_Intake.py、
  tests/unit/test_sources.py、tests/unit/test_scenario_matrix.py、
  tests/unit/test_data_intake_ui.py、tests/unit/test_readme_alignment.py、
  docs/agent-setup.md 与本记录。

### 第 29 轮 = 样例侧重复表头行如实点名 dup_header_note(恢复循环第 10 轮)

(恢复的北极星打磨循环,第 10 轮。)核心痛点:导出拼接产生的重复表头行是读取层
最后一个「有据可查的静默事实」——全量侧早已硬拦(`repeated_header_rows`,场景 22
blocking「与表头完全相同」点名行号),样例侧却完全无声:场景 30 实测该行以
「就绪」面目进入预览(输入是列名「客户描述」、答案是列名「类别」)、分布 finding
把列名计为一类、对比核验还可能拿它出题,expect_note 白纸黑字写着「样例侧没有
对称检查,靠用户在预览逐行核对自行识别」。读取层披露系列(sheet/merged/formula/
hidden/blank,第 18/24/25/26/28 轮)至此只剩这一处静默。

- **dup_header_note 跨格式如实点名**(`intake_models.py` + `sources.py`):契约
  字段 + `_dup_header_note` 文案助手 + 检测块(位于 blank_note 组装之后、
  「文件没有数据行」拦截之前)+ 构造注入 + profile 同步 + `read_source`
  docstring,六处接线。**判定口径与全量侧完全一致**(列数 ≥ 2 且每格等于
  列名,基于解析后的 `(行号, 记录)` 元组),披露与拦截永不互相矛盾;单列文件
  不检测(整列同值是合法业务数据,如类别列全是「质量」);因此 **xlsx 与 xls
  均检测**、CSV/JSONL 同样检测(JSONL 数值型值不等于字符串列名,不误报;
  自命名记录 `{"编号": "编号"}` 如实点名)。note 文案分述两侧行为:「该文件有
  1 行与表头完全相同（第 3 行）照常读入为普通数据行——通常是导出拼接产生的
  重复表头，输入与答案都会是列名；样例侧不拦（以「就绪」进入预览、可能参与
  对比核验），全量侧会被全量验证硬拦。请删除重复表头行；没有自动删行。」行号
  超过 5 个以「等」收尾(与各注同口径)。读取行为不变:该行照常读入为普通数据
  行(测试钉死 rows[1].values 即列名);不自动删行,删除由用户决定。
- **UI 第六键**(`ui/pages/07_Data_Intake.py`):`show_fact_notes` 键元组补
  `dup_header_note`(warning 级,与 blank 同),三处渲染位点(session.profile、
  原始资料与补充文件区带别名前缀、全量报告来源行)自动覆盖,签名不变。
- **场景矩阵**(45 不变,无新场景——形态已由 30/40 覆盖,只升 expect_note):
  场景 30「样例侧没有对称检查,靠用户在预览逐行核对自行识别」改写为「样例侧
  不拦但已如实点名——dup_header_note(第 29 轮)按与全量侧完全一致的判据指认
  第 3 行…识别不再只靠用户逐行核对」;场景 40「样例侧至今没有对称检查」改写为
  「拦截不对称(样例侧披露不拦、全量侧硬拦),两侧判据同源,披露与拦截不互相
  矛盾」。矩阵测试场景 30 扩展探针:note 含「1 行与表头完全相同」「第 3 行」
  「样例侧不拦」「全量验证硬拦」「没有自动删行」,profile 与来源一致;场景 40
  测试注释同步。45/45 as_expected 不变。
- **文档钉死**(agent-setup.md + test_readme_alignment 21→22):多份资料小节
  新增重复表头段——dup_header_note 点名行号、输入与答案都会是列名、样例侧
  不拦/全量验证硬拦两侧行为分述、判据与全量侧完全一致、单列文件不检测、
  没有自动删行、xlsx 与 xls 均检测;渲染段「五条」改「六条」,跨格式句升为
  「空行与重复表头事实同样渲染」(钉测试同步改句)。新钉测试
  `test_dup_header_rows_docs_pinned_and_backed_by_matrix`:关键句 + 场景矩阵
  背书(duplicate-header-row-in-sample=passes、duplicate-header-rows-in-full=
  blocked_at:validate_full 双场景钉死)。
- **回归**:test_sources 31→37(CSV 检测+读取行为不变/JSONL 自命名+数值负例/
  Excel 物理行号/前五上限「等」/三格式干净负例+单列守卫/服务层持久化)、
  test_scenario_matrix 33(场景 30 探针扩展)、test_data_intake_ui 25(夹具
  升级五注一簿+主例/别名/负例断言)、test_readme_alignment 21→22;顺带清偿
  既有 lint 债(test_page_journey.py 两个 F401,commit 06e252d 引入,ruff
  --fix,套件 2 passed 不变);ruff check/format clean。全量回归 **tests/unit
  1739 passed / 0 failed**(--no-cov,无排除,170.99s,算术对账 1732+6+1)。
  本批只动 src/workbench/intake_models.py、src/workbench/sources.py、
  src/workbench/scenario_specs.py、ui/pages/07_Data_Intake.py、
  tests/unit/test_sources.py、tests/unit/test_scenario_matrix.py、
  tests/unit/test_data_intake_ui.py、tests/unit/test_readme_alignment.py、
  tests/unit/test_page_journey.py、docs/agent-setup.md 与本记录。

### 第 30 轮 = 物化层分区答案覆盖如实点名 answer_coverage_note(恢复循环第 11 轮)

(恢复的北极星打磨循环,第 11 轮。)核心痛点:分组隔离切分可把稀有类别的整组
记录全部分进验证或测试(分组隔离优先于比例、未做类别分层)——训练按逐字学习
答案,模型无法输出没学过的值,而验证与测试照常打分。此前 statistics 对分区
答案构成完全无声:默认 seed 42 下 18 个单行组(17 条「yes」+ 末行唯一一条
「screen」)切分 14/2/2、稀有行落入测试集,用户在评测看到该类全错之前毫无
线索;更刺眼的是 **UI 默认夹具(FULL 3 行:100质量/101物流/102质量)本身就
触发同一形态**——train 只见「物流」、「质量」整组落验证/测试,旅程测试早已
路过却不曾点名。

- **三统计键**(materialize.py):`answer_counts_by_split`(各分区答案 Counter
  dict)、`train_missing_answers`(`{值: {"validation"/"test": 条数}}`)、
  `answer_coverage_note`(人话披露)。**门控:全部答案不同取值 ≤ 20 才计算**
  (逐字学习下未见值不可学,点名才有信息量);开放文本/大量类别时逐值点名没有
  信息量,统计缺键即这一如实边界(不看 value_kind,单看取值数——categorical
  也可能 30 类,numeric_continuous 也可能恰好 8 个离散值)。三种切分方式
  (分组随机/temporal/fixed_evaluation_suite)同口径,成因句按 split_method
  分支:temporal「时间边界先于比例,窗口内只出现一次的类别会整体落在单一
  分区」、固定题集「固定题集把既定评分题保留在原分区,训练段新增类别可能
  只出现在验证或测试」、分组随机「分组隔离优先于比例且未做类别分层,稀有
  类别的整组记录可能全部落在验证或测试」。note 文案:「验证/测试集中有 N 类
  答案（值×条数（测试M 条）…）从未出现在训练集——训练按逐字学习答案,模型
  没有学过这些值,验证与测试仍会照常打分。{成因}补充该类别的独立业务对象后
  可重新生成分区版本;没有自动重新切分,也不会把记录挪回训练集。」值超过
  5 个以「等」收尾(与各注同口径)。**披露不阻断、不自动重切、不挪记录**
  ——物化照常完成,分区文件与 row_counts 不变。
- **人话摘要与 CLI**(report_summary.py + data_intake.py):summarize_dataset
  在边界句「分区就绪只说明…」前渲染 coverage_note(UI 数据集版本区经
  st.write 自动可见,披露不埋进 JSON);CLI materialize 输出沿用 temporal
  stderr 打印先例。
- **场景矩阵 46**(`rare-category-only-in-holdout`):全量 18 行(17 质量 +
  末行屏幕,`_rows_as_csv` 夹具)、样例含两类答案(对比可配对)、编号单行组;
  expect=passes(八关全过,披露不阻断)。expect_note 实测口径:14/2/2、屏幕
  落测试集、note 原文、≤20 门控与三切分同口径。矩阵下限 45→46。
- **文档钉死**(agent-setup.md + test_readme_alignment 22→23):「生成独立
  数据分区与版本」段新增答案覆盖披露段——三统计键、note 原文例、逐字学习
  与照常打分的反差、没有自动重新切分、20 种门限双侧口径(不超过/超过)、
  时间分区与固定题集同口径、页面与 CLI 两个展示位点。新钉测试
  `test_answer_coverage_docs_pinned_and_backed_by_matrix`(关键句 + 场景矩阵
  背书 expect=="passes")。
- **回归**:test_data_materialize 8→10(稀有答案正例:14/2/2 + 三键 + note
  短语;开放答案空间负例:23 种取值、19/2/2、三键缺位)、test_report_summary
  +1(coverage_note 渲染位置在边界句前 + 无键负例)、test_scenario_matrix
  下限 46 + 场景 46 钉测试(夹具真实性 + run_scenario as_expected 八关全过 +
  手动探针断言 statistics)、test_data_intake_ui 既有流测试扩展
  (train_missing_answers=={"质量": {validation:1, test:1}} + page.markdown
  渲染断言——默认夹具形态被正向点名)。教训重申:row_counts 断言必须探针
  先行——开放负例初版照抄 n=24 探针值 19/3/2,而夹具实为 23 行(19/2/2),
  定向套件一次拦下后修正。全量回归 **tests/unit 1744 passed / 0 failed**
  (--no-cov,无排除;基线 1739 + 新增 5)。本批只动 src/workbench/
  materialize.py、src/workbench/report_summary.py、src/workbench/
  scenario_specs.py、scripts/data_intake.py、docs/agent-setup.md、
  tests/unit/test_data_materialize.py、tests/unit/test_data_intake_ui.py、
  tests/unit/test_report_summary.py、tests/unit/test_scenario_matrix.py、
  tests/unit/test_readme_alignment.py 与本记录。

### 第 31 轮 = 完全相同例题如实点名 duplicate_note(恢复循环第 12 轮)

(恢复的北极星打磨循环,第 12 轮。)核心痛点:物化统计一直携带双口径重复行计数
——`rendered_exact_duplicate_rows`(渲染后输入与答案逐字一致的重复条数)与
`source_exact_duplicate_rows`(原始行每个字段完全一致的重复条数)——但这两个
数字只躺在 statistics 里,人话摘要、页面与 CLI 从不翻译(round 27 勘察时记录的
缺口)。重复例题等于训练隐式加权(同一道例题出现多次),用户在看到模型对个别
例题过拟合前毫无线索;连 UI 默认夹具(FULL 9 行,rendered=2/source=1)都触发
同一形态。选题前先排除候选 (a)「相同渲染输入不同答案」:full_data.py 的
conflict status blocking + materialize 非 ready 行 raise 早已硬拦,非静默——
重复行是答案一致时的兄弟形态,披露缺口才是真缺口。

- **`_duplicate_note` 助手 + statistics 附件**(materialize.py):门控
  `rendered_extra > 0`(计数为零缺键;与 20 值门控不同,计数本身总是有信息量,
  无需上限)。note 分述两种成因——`source_extra < rendered_extra` 时「其中 N 条
  原始行完全重复(每个字段都一致,多见于导出拼接或关联重复);另有 M 条是不同
  原始行渲染成同一例题(原始字段不同、例题相同)」,只有原始重复时「均为原始行
  完全重复…」,只有渲染重复时「均为不同原始行渲染成同一例题…」。点名隐式加权
  效应与独立例题数(「N 条记录去重后只有 N-M 道独立例题」);写明与 conflict
  硬拦的分界(「相同输入配不同答案已被全量验证拦下,不会出现在任何分区」)、
  同分区事实(「完全相同的记录保持在同一分区」)与语义安全边界(「全部记录
  原样保留;没有自动去重,去留由你决定」)。三种切分方式同口径。
- **人话摘要与 CLI**(report_summary.py + data_intake.py):summarize_dataset
  在 coverage_note 之后、边界句「分区就绪只说明…」前渲染 duplicate_note(UI
  数据集版本区经 st.write 自动可见);CLI materialize 输出沿用 coverage_note
  stderr 先例。
- **场景矩阵 47**(`exact-duplicate-rows-in-full`,先探针后定局):全量 12 行
  =干净 10 行(001-010)+ 005 原始行完全重复 + 011(不同编号、同一「开不了机/
  质量」例题);样例两类答案(001 质量/002 物流,对比可配对)。探针实测:
  validate_full 对重复行不拦(答案一致不触发 conflict,唯一 review 是与重复无关
  的 split_not_validated);「开不了机」三条记录经相同输入连接成同一分组、
  全落训练集;10 组贪心切分 10/1/1;rendered=2/source=1;answer_coverage_note
  正确沉默(训练集见过全部答案类别)。expect=passes(八关全过,披露不阻断、
  不自动去重)。矩阵下限 46→47。
- **文档钉死**(agent-setup.md + test_readme_alignment 23→24):「生成独立
  数据分区与版本」段在答案覆盖段后新增完全相同例题披露段——双计数键、note
  原文例、两种成因分述、隐式加权、与冲突守卫分界、永不跨分区、缺键即零重复、
  不自动去重、页面与 CLI 位点。新钉测试
  `test_duplicate_rows_docs_pinned_and_backed_by_matrix`(关键句 + 场景矩阵
  背书 expect=="passes")。
- **回归**:test_data_materialize 首测扩展 duplicate_note 断言(「2 条记录」
  「去重后只有 7 道独立例题」「其中 1 条原始行完全重复」「另有 1 条」「没有
  自动去重」)+ rare 测试加无键负例;test_report_summary +1(双注顺序:
  coverage→duplicate→边界句,负例不多说);test_scenario_matrix +1(场景 47
  钉测试:005 两次且每字段一致、011 同例题不同编号的夹具真实性 + run_scenario
  八关 + 手动探针 statistics + 「开不了机」同分区断言)。定向套件 102 passed;
  全量回归 **tests/unit 1747 passed / 0 failed**(--no-cov,无排除,169.66s;
  基线 1744 + 新增 3)。本批只动 src/workbench/materialize.py、src/workbench/
  report_summary.py、src/workbench/scenario_specs.py、scripts/data_intake.py、
  docs/agent-setup.md、tests/unit/test_data_materialize.py、
  tests/unit/test_report_summary.py、tests/unit/test_scenario_matrix.py、
  tests/unit/test_readme_alignment.py 与本记录。
