# 配置数据分析 Agent（BYOK）

Agent 使用你选择的模型服务，联合分析业务目标和上传的数据样例。这里配置的是分析助手使用的模型，不是之后要微调的基础模型。服务需要支持 OpenAI Chat Completions 的工具调用协议。

## 在页面中配置

打开「目标与数据」入口，在「分析模型设置」选择供应商，核对 API 地址和模型名称，输入 API Key，点击「测试模型连接」。探针只发送固定测试内容并检查工具调用，不会读取或发送数据任务；通过连接测试也不代表业务分析质量已验收。

「保存模型配置」只保存供应商、地址和模型名。页面输入的密钥只保留在当前 Streamlit 会话中；重新建立会话后需再次输入，或由运行进程的环境变量提供。切换供应商或修改地址会清除输入密钥及业务数据发送授权，环境密钥也不会跟随到另一个供应商或地址。

正式分析远程服务时，需勾选页面的数据发送授权。发送内容包括业务说明、回答、数据画像和选取的证据行；连接探针不会代替这项授权。

## 供应商预设

| 选择 | OpenAI Base URL | 说明 |
| --- | --- | --- |
| `local` | `http://localhost:11434/v1` | 本地兼容服务，填写已部署且支持工具调用的模型名 |
| `glm-coding` | `https://open.bigmodel.cn/api/coding/paas/v4` | 智谱中国 GLM Coding Plan |
| `glm` | `https://open.bigmodel.cn/api/paas/v4` | 智谱中国普通 API |
| `compatible` | 自行填写 | 其他 OpenAI 兼容服务，包括 Z.ai 国际端点 |

模型名可以修改，以账户实际可用模型为准。Base URL 不包含末尾 `/chat/completions`，程序会补上该路径。远程服务必须使用 HTTPS。

Coding Plan 与普通 API 的端点不互换，失败时程序不会自动改用普通 API。官方注明 Coding Plan 端点用于编码场景；选用前需确认当前用途符合服务范围。Z.ai 国际 Coding Plan 地址为 `https://api.z.ai/api/coding/paas/v4`，普通 API 地址为 `https://api.z.ai/api/paas/v4`，可通过自定义服务填写。端点依据：[官方模型配置说明](https://zcode.z.ai/en/docs/configuration)（核对于 2026-09-26）。

## CLI 配置与检查

以下命令在项目根目录执行。密钥只从 `TUNESMITH_AGENT_API_KEY` 读取，不提供命令行密钥参数，也不保存到配置文件、任务快照或导出结果。

```sh
python scripts/data_intake.py agent-config --provider glm-coding --model glm-5.3
```

通过当前 shell 的隐藏输入设置密钥，避免把密钥直接写进命令历史。例如 zsh：

```zsh
read -rs 'TUNESMITH_AGENT_API_KEY?输入 API Key: '
export TUNESMITH_AGENT_API_KEY
python scripts/data_intake.py agent-check
```

`agent-check` 只发固定工具探针，不建立数据任务。完成使用后可运行 `unset TUNESMITH_AGENT_API_KEY`；已启动的页面进程仍持有启动时继承的环境，需要重新启动才能更新。

创建数据任务不会调用模型：

```sh
python scripts/data_intake.py create --input ./sample.csv \
  --goal '根据用户首次描述判断工单类别' \
  --description '一行一个工单，类别由质检人员审核，关闭原因是事后信息' \
  --scope sample
```

命令会返回 `session_id`。分析远程服务时显式允许业务数据发送：

```sh
python scripts/data_intake.py analyze SESSION_ID --allow-remote-data
python scripts/data_intake.py analyze SESSION_ID --answer '质检类别才是需要学习的答案' --allow-remote-data
python scripts/data_intake.py show SESSION_ID
```

凡返回任务记录的命令（`show`、`create`、`add-source`、`analyze`、`confirm`、`full-*`、`materialize`）在 stderr 尾行输出「下一步状态: 枚举（人话对照）」——枚举保留供脚本解析，人话由 `next_action_phrase` 单一来源翻译（13 个状态全覆盖，未知状态只显枚举不编造），与页面提示同词汇。awaiting_analysis 的尾行同时点名两条路径——配置了 Agent 的 `analyze` 与零密钥的 `baseline-analyze`（与页面「没有 Agent 服务？用基础分析开始」入口同词汇），零密钥用户不会被指去配置密钥才能跑的命令。Agent 产物三态同样点名出口——needs_business_answers 用 `analyze --answer` 一次完成回答与重新分析（与页面「回答问题或修正理解」同一动作，不是页面专属），needs_capability 如实说明资料侧出口——调整目标或 `add-source` 补充资料后重新 `analyze`，不伪造零密钥兜底，needs_recipe 点名重新 `analyze` 生成处理规则。

needs_data_revision 的尾行同样点名两条重分析路径——配置了 Agent 的 `analyze` 与「此前的基础分析可调整字段重跑 baseline-analyze」，零密钥用户在问题行修复后不会走进死胡同。needs_labels 的尾行点名零密钥修复工具链——`answer-sheet` 导出待补清单、补齐后「替换原文件重新分析（基础分析可重跑 baseline-analyze）」，「补齐标注」不是一句没有出口的提醒。review_preview 的尾行点名配对工具链——`contrast-check` 抽两道配对题、`contrast-check-submit` 提交配对（二连对）、之后 `confirm` 确认，「完成对比核验」不是页面专属动作。

awaiting_full_data／awaiting_full_validation 的尾行点名 `full-validate`（多资料任务用 `full-sources`；已声明全量可省略 `--input`）、review_full_data 的尾行点名 `full-confirm`、needs_full_data_revision 的尾行点名修正后重跑——全量数据段每一步都有可照抄的命令入口。至此 13 个状态的尾行全部有出口或如实边界。

五个清单命令（`train-list`、`plan-list`、`iteration-list`、`acceptance-list`、`scoring-list`）同样在 stderr 追加一行清单尾行（`summarize_listing` 单一来源）：空清单点名该走的第一步入口——用户分不清「还没有」和「查错了任务」；非空给计数，不逐条灌业务人话。

`--base-url` 和 `--model` 可临时覆盖分析或连接检查的配置。若已有环境密钥，临时地址与解析后的地址不同，CLI 会拒绝请求。要改用另一个服务，应同时配置该服务的 `TUNESMITH_AGENT_BASE_URL`、`TUNESMITH_AGENT_MODEL` 和配套密钥，再重试，避免旧密钥被带到新服务。

`analyze` 在返回任务 JSON 之外，还会在 stderr 逐行打印与页面「数据判断与待确认问题」区同词汇的人话摘要（`summarize_analysis` 单一来源）：先给计数行「这份分析给出数据判断与待确认问题：发现 N 条、待确认问题 M 个。」，随后逐条翻译发现——kind 按页面同一词汇译出（已观察／待验证推断／需要业务解释／需要全量验证），带证据行的发现附「（证据：行ID…）」原文引用；待确认问题按原文＋为什么＋可选解释翻译；另如实给出当前能力缺口、暂定微调思路与下一步；固定收尾句写明：以上发现中「已观察」是数据里的事实，其余是待确认的推断或业务解释；分析待你确认并经真实预览核对，不代表业务效果达标。没有可读内容时只给缺位句，不编造发现。摘要尾行另以评测解读同一格式渲染工具核查轨迹（`summarize_tool_trace` 单一来源，与 eval-analyze 共用）：「工具核查轨迹：N 次调用，成功 X 次、失败 Y 次——失败的调用没有取到证据，分析只依赖成功的调用」。页面「处理规则与工具记录」区在同一位置渲染同一行（原始 JSON 留在下方），CLI 与页面同源同词汇。

`funnel-report` 输出全部任务在闭环各环节的当前停点计数（只读快照，不发起计算）：stdout 纯 JSON、stderr 人话摘要（`summarize_funnel` 单一来源）。数据准备段按旅程顺序给出五个停点（分析与方案／样例预览确认／全量验证／独立分区／预检就绪），随后是训练运行、对照评测、最终业务验收与改进轮次的状态计数，停得最多的段会点名「下一个最值得查看具体卡点的入口」。未收录的停点状态按原样列出，不硬塞进相近段；某段记录读取失败时点名跳过（如「训练记录读取失败，该段未计入（其余段照常统计）」），不假装为零。数据来自本机 workbench 记录目录（`--store` 与四个 `--*-root` 参数，默认项目内路径）。页面「目标与数据」侧栏的「全部任务停点快照」折叠区与该命令同源同词汇。固定收尾句写明：这是各任务当前停点的快照，不是历史通过率；旅程可以回退，回退后按新停点重新计数；计数只描述进度，不代表业务效果。

## 从样例继续到全量数据

### 内置演示任务

新建任务页提供「第一次使用？用内置演示任务开始」入口：一键用仓库自带的虚构售后工单样例（2 条）创建任务（`src/workbench/demo_task.py` 单一来源，目标与数据说明与《用户试用记录》的试点任务同一份），代替「先找一份 CSV 再填表」的冷启动。入口只代劳找文件与填表这一步——创建后的每一步（基础分析、预览核对、对比核验、盲标核验）与真实任务完全相同，没有预设结论，也不跳过任何门禁。样例确认后需要全量数据时，来源就是演示样例的任务（按文件内容摘要比对，同名不同内容不算）会在全量上传框旁多一个「使用配套演示全量数据（虚构，10 条）」按钮；摘要对不上的真实任务看不到这个按钮，演示数据不会混进任何真实任务。演示文件不在场时入口如实隐藏（`demo_sample` 返回 None），不编造演示数据。演示数据与真实业务数据无关，仅为流程演示。

CLI 有同源的两个零密钥入口。`create --demo` 用同一份演示目标与样例创建任务，不收 `--input`/`--goal`，也不收 `--description`/`--encoding`/`--delimiter`/`--sheet`/`--scope full`——单一来源不被自定义值污染，给出的自定义参数逐一点名并直接报错，不静默忽略；演示文件不在场时如实报错退出，不编造演示数据；创建后 stderr 给出对演示数据逐字可用的下一步命令。`baseline-analyze SESSION_ID --target 答案列 [--group 分组列] [--exclude 排除列] [--instruction 补充指令]` 与页面「没有 Agent 服务？用基础分析开始」区同一来源（`propose_baseline_analysis` 确定性生成方案，`model` 记为 `baseline-deterministic`），时间分区用 `--temporal` 加三个时间字段与三个边界参数。已有 Agent 分析结果时命令如实拒绝（页面在已有 Agent 分析时不再显示该入口，CLI 同口径，不悄悄覆盖）；已有的分析也是基础分析时，可调整字段重新生成——替换旧方案、预览与确认状态随之失效（与 Agent 重分析同一套失效语义），这是 needs_data_revision 的零密钥出口：问题行修复后不必配置密钥就能重跑。页面同口径：已有基础分析时入口保持可见并提示「重新生成会替换它」。`--temporal` 缺参数时逐一点名缺哪些，不静默退回随机切分。stdout 任务 JSON 与 stderr 的 `summarize_analysis` 人话摘要与 `analyze` 同一格式；替换时 stderr 先说明「已替换此前的基础分析」。演示任务的 CLI 冷启动组合：`create --demo` 之后直接照抄 stderr 提示的 `baseline-analyze SESSION_ID --target 类别 --group 编号 --exclude 处理结果`，零密钥进入真实预览。

### 多份资料一起分析

首次上传的资料命名为 `main`。在页面「原始资料与补充文件」上传其他资料，给它一个便于识别的名称，并说明用途和关系，例如「labels 是质检审核的类别，通过工单编号对应 main」。同名上传会替换该份资料，页面要求明确勾选替换；补充或替换后，旧方案需重新分析确认。

Agent 可以检查多个来源，试运行关联、列表展开、嵌套字段提取，以及按会话和顺序生成「历史输入 → 当前答案」。页面展示实际处理步骤、问题和每个组合行的原始来源。用户无需先拼表，也无需填写 JSON 配方。会话历史只包含该回答之前的信息；生成的数据仍走当前 Alpaca 训练入口。

```sh
python scripts/data_intake.py add-source SESSION_ID --revision CURRENT_REVISION \
  --alias labels --input ./labels-sample.csv --scope sample \
  --description '人工审核类别，通过工单编号与 main 对应'
python scripts/data_intake.py analyze SESSION_ID --allow-remote-data
```

确认组合后的真实样例预览后，提供方案所需的每份全量原始文件。页面分别显示所需资料的上传框，不要求先手工关联。CLI 用重复的 `--source` 指定资料名称与路径：

```sh
python scripts/data_intake.py full-sources SESSION_ID --revision CURRENT_REVISION \
  --source main=./tickets-full.csv --source labels=./labels-full.csv \
  --sheet main=工单表 --sheet labels=1
```

`full-sources` 不提供 `--source` 时，只能复用已保存且明确标为全量的所需来源；样例不会自动升级为全量。全量继续执行已经确认的组合与转换规则，缺失关联、字段冲突或会话顺序异常会进入报告。

Excel 多 Sheet 工作簿默认只读第一个 sheet。四个入口都可以指定读取的工作表（名称或 1 起始序号，1 表示第一个）：`create`、`add-source`、`full-validate` 各收一个 `--sheet` 参数，`full-sources` 按资料收可重复的 `--sheet ALIAS=名称或序号`；页面上传 Excel 时同样提供可选 sheet 输入。实际读取范围随来源持久化并在画像中如实标注，例如「该文件含 2 个 sheet，仅读取第一个「员工表」；其余 1 个（工单表）未读取」——数据在第二个 sheet 时用户能看见读的是哪张表。首次声明全量的文件复用原文件做全量验证时，沿用当时持久化的 sheet 选择。指定的 sheet 不存在会报错并列出全部可用 sheet 名，不静默回退；CSV/JSONL 传 sheet 会被明确拒绝。

xlsx 读取的 sheet 存在与数据区相交的合并单元格时，来源携带 `merged_note` 如实点名，例如「该 sheet 含 1 处合并单元格（类别 C2:C4）：合并区除左上角外均读为空值，涉及答案列时这些行会按缺少监督答案处理。请取消合并并逐行填写受影响的值；没有自动填充。」——合并区除左上角外均读为空值，用户被「缺少监督答案」拦下时能看到根因是 Excel 合并而不是数据缺失。取消合并并逐行填写由用户决定，不自动填充；数据区之外的合并、未读取 sheet 的合并不列入；xls 引擎不提供合并范围，不检测。

xlsx 读取的 sheet 数据区存在没有缓存计算结果的公式单元格时，来源携带 `formula_note` 如实点名，例如「该 sheet 含 2 个没有缓存计算结果的公式单元格（类别 C2、类别 C3）：这些公式读为空值，涉及答案列时这些行会按缺少监督答案处理。请用 Excel 等软件打开并保存以生成计算结果；没有自动计算。」——报表工具或脚本写出的 xlsx 公式格常常没有缓存值（openpyxl 不计算公式），这些格读为空值，用户被「缺少监督答案」拦下时能看到根因是公式无缓存而不是数据缺失；真实 Excel 保存过的公式格带缓存值，按缓存值正常读取、不列入。用 Excel 打开保存生成缓存值由用户决定，没有自动计算；分组列等其他列的公式同样读空，不止答案列；数据区之外的公式格、未读取 sheet 的公式格不列入；xls 引擎不提供公式清单，不检测。

xlsx 读取的 sheet 数据区存在隐藏行或隐藏列时，来源携带 `hidden_note` 如实点名，例如「该 sheet 含 2 个隐藏行（第 3 行、第 4 行）：隐藏行照常读入——Excel 中看不到的行也会进入分析与训练。请取消隐藏并删除不需要的行；没有自动排除。」——AutoFilter 筛选后保存或手工隐藏的行/列，读取时照常进数据，Excel 里看不到的行也会进入分析与训练、隐藏列仍出现在可用字段中（`hidden_note` 按表头名列出）；不自动排除，隐藏行可能是有意保留的数据（筛选只是视图），是否取消隐藏、删除不需要的行列由用户决定。表头行的隐藏、数据区之外的隐藏、未读取 sheet 的隐藏不列入；xls 引擎不提供隐藏标志，不检测。

空行处理跨格式如实点名：来源携带 `blank_note`，两种口径对应两种真实行为。CSV/JSONL 的空行（含纯空白行）读取时被跳过、不进数据（读取行为不变），note 以「已跳过」口径点名行号，例如「该文件有 3 个空行（第 3 行、第 5 行、第 6 行）已跳过——空行不进入分析与训练。请核对空行位置是否丢了数据；没有自动补行。」Excel 数据区的全空行（整行无值）则照常读入为全空记录，note 以「照常读入」口径点名行号并说明会按缺少监督答案与分组标识处理、在样例确认或全量验证被拦下，没有自动排除——用户被拦时能看到根因是全空行。Excel 尾部空行在解析时自然消失、无从检测，不列入（如实边界）；与 merged/formula/hidden 仅 xlsx 检测不同，全空行基于解析后的记录判定，xls 同样检测。

数据区存在与表头完全相同的行（每个单元格都等于其列名，通常是导出拼接产生的重复表头）时，来源携带 `dup_header_note` 如实点名行号，例如「该文件有 1 行与表头完全相同（第 3 行）照常读入为普通数据行——通常是导出拼接产生的重复表头，输入与答案都会是列名；样例侧不拦（以「就绪」进入预览、可能参与对比核验），全量侧会被全量验证硬拦。请删除重复表头行；没有自动删行。」——样例侧读取行为不变：该行按普通数据行读入、以「就绪」面目进入预览并可能参与对比核验，此前对该行完全无声；全量侧由全量验证硬拦（`repeated_header_rows` 点名行号）。两侧判据完全一致（列数 ≥ 2 且每格等于列名），披露与拦截不互相矛盾；单列文件不检测（整列同值是合法业务数据）。删除重复表头行由用户决定，没有自动删行；跨格式，xlsx 与 xls 均检测（基于解析后的记录，与 blank_note 同口径）。

这六条来源如实标注（sheet 读取范围、合并单元格、公式无缓存、隐藏行列、空行/全空行、重复表头）不埋进 JSON：页面在任务视图的主来源画像下、「原始资料与补充文件」区（多资料时每条标注带资料名前缀，如「labels：…」）和全量验证报告的来源行下三处渲染——sheet 级标注用 info 说明读取范围，其余五条用 warning 提示影响数据事实的情况。CSV/JSONL 来源没有 Excel 事实，sheet/merged/formula/hidden 一条都不渲染；`blank_note` 与 `dup_header_note` 跨格式，CSV/JSONL 的空行与重复表头事实同样渲染。

### 长尾字段解析与受限适配

当已有转换组件无法表达某个解析规则时，Agent 可以起草受限适配代码，先用真实业务期望样例和独立反例在操作系统隔离环境中验证，再用于真实资料。当前适配只支持保留原字段和原行、一对一新增声明字段；不能静默删除记录或覆盖原值。

页面展示实际隔离后端、业务样例与反例的测试结果、新增字段以及源行引用。源码、配置和验收样例可折叠查看，不要求业务用户先人工审代码。用户仍通过真实转换预览确认模型输入和答案是否符合目标。全量验证单独显示其适配报告，样例测试通过不等于全量已经通过。

执行依赖可用的真实操作系统隔离后端，例如 Docker 或受支持的 macOS 沙盒；不可用时返回失败，不回退到宿主直接执行。Docker 的内存限制依赖容器控制机制；macOS 模式采用进程 RSS 抽样检查，不能视为同等的容器硬内存上限。页面仅展示执行报告，不直接运行源码。

### 单份资料

核对样例转换后，页面提供「全量数据验证」入口：上传全量 CSV、Excel 或 JSONL，沿已确认方案进行本地转换与诊断；首次就声明为全量的文件可直接复用。报告独立保存全量来源和问题行，样例证据与确认记录不会被替换。

缺少必需字段、监督答案、转换失败或答案冲突时，先根据报告修正资料或业务规则。新增类别和字段需要核对；开放文本不会仅因出现新答案而被判错。「保存业务补充，稍后分析」可先保存修正，旧全量报告随即失效，重新确认方案后须重验全量。

CLI 使用每次输出中的最新 `revision`，确认操作表示已经查看相应报告：

```sh
python scripts/data_intake.py confirm SESSION_ID --revision SAMPLE_REVISION
python scripts/data_intake.py full-validate SESSION_ID --revision CURRENT_REVISION --input ./full.csv
python scripts/data_intake.py full-confirm SESSION_ID --revision FULL_REPORT_REVISION
```

`full-validate` 支持 `--encoding`、`--delimiter`、`--sheet`。首次上传已声明 `--scope full` 时可省略 `--input`，样例任务必须显式提供全量文件。这两个命令只处理本地文件，不调用模型或消耗模型服务额度。验证发现问题仍会保存报告，下一步状态为 `needs_full_data_revision`；问题解决后核对并确认，进入 `awaiting_dataset_split`。这表示可以继续准备独立训练与评测分区，尚未认定可以正式训练。

`full-validate`、`full-sources` 与 `full-confirm` 在 stdout 输出 JSON 的同时，会向 stderr 追加全量验证报告的人话摘要（`summarize_full_report`；任务已有报告时，`show` 等返回任务记录的命令同样追加）：先说清报告针对哪份资料（名称、条数、内容摘要）并注明行 ID 仅属于这份全量文件，再给当前结论——存在阻断问题需先修正再重验、没有阻断待核对后确认、已确认可准备分区（尚未开始训练）、或业务理解已变化报告失效；随后按阻断在前逐条渲染报告问题原文（附涉及证据行条数），报全量真实转换的四态计数（已生成预览／缺少答案／需修正处理／答案有冲突），并以「全量验证只核对数据事实与已确认方案的一致性，不代表模型效果或业务达标」收尾。状态与问题词汇与页面全量验证区同源。

缺答案的行不再只是一段页面提示。页面在「需补齐 N 条记录的答案」提示下方提供「导出待补答案清单（交给填写人）」下载按钮，CLI 用 `answer-sheet SESSION_ID [--export-csv 清单.csv]`（本地读取任务记录，不调用网络、不消耗模型额度）——两者输出同一份填写表与同三条交接规则（`src/workbench/answer_sheet.py` 单一来源）。清单口径与页面提示一致：全量真实转换预览存在且未失效时按全量行导出，否则按样例行——样例有缺答案会拦确认，全量报告仍存在即说明样例当时是干净的。填写表只含「行ID、题目输入（模型将看到的内容）、每字段一个待填答案列」：待填格为空，行ID 与题目输入原样保留，回传后按行ID 对号回填；表内不包含任何已有答案，结构上不可能泄露监督标签（与盲标题目清单同一原则）。三条规则随清单同源输出：交给填写人后只填「待填答案」列，回传后按行ID 对号回填；填写依据是业务事实（查业务系统或档案都可以），不要凭猜测填——答案会直接成为模型的监督信号，错一条教错一条；填写人认为题目输入不足以判断答案时把该行留空并在回传时说明——输入信息不足是任务定义问题（与盲标核验、可学性探针的同类提示同方向），比编一个答案更有价值。CLI stdout 为纯 JSON（`scope`／`missing_count`／`row_ids`／`answer_fields`），stderr 先给「清单口径：…，待补 N 行。」再给规则与导出行。没有缺答案行时不写文件、明说「当前没有缺答案的行，无需导出清单」；任务还没有真实转换预览时报错退出（先运行 analyze，或零密钥的 baseline-analyze）。填写表是交接物不是回传文件——填写人线下填好后，仍按原提示在页面替换对应原始文件（或全量文件）后重新分析，没有直接上传填写表自动回填的通道。行筛选（`missing_answer_rows`）与 `needs_labels` 状态判定共用同一来源，时间方案的未成熟标签行是「标签窗口未结束」的既定保留，不会混进交给填写人的清单。

### 连续数值答案的如实边界

答案列是带小数的连续测量值（如 1.0/2.5/3.7）时，基础分析会把该目标的答案形态如实标注为 `numeric_continuous`，并同时给出边界说明：当前训练仍按逐字字符串学习答案（「1.0」与「1.00」算两个不同答案），不是数值回归；评测只能逐字比对，无法按数值误差评分。需要数值误差口径时，先在数据侧把答案列离散化。小数若只是离散编码（如版本号），照常逐字核对即可；整数编码（0/1/2）不受此判定影响，仍按类别处理。全量出现样例未覆盖的新测量值是连续目标的常态：报告以 `numeric_new_values` 如实列出这些值并重申逐字学习边界，不按「新类别」表述、不阻断。评测、验收与页面都不会把 `numeric_continuous` 目标伪造成分类严格评分——它落入与开放任务相同的人工核对口径。

## 语义安全层：确认不再盲点头

北极星的核心痛点是语义错误无声通过。三道通用关卡（与任务类型无关、零或低成本、非专家可独立完成）已接入流程：

- **对比核验（样例确认时）**：系统抽取两条答案不同的输入，把两个答案打乱后由你配对。配对正确才说明真正看清了转换含义；配错会留档并提示重新查看预览。预览或方案变化后核验自动失效。页面在「确认当前转换含义」前提供入口；CLI 用 `contrast-check` / `contrast-check-submit`（完整用法见下文）。
- **盲标核验（训练准备的硬门禁）**：全量确认后、准备训练前，系统按你选择的样本量（1–50 条，默认 5 条）随机抽取已标注行并隐藏答案，你仅根据输入作答，与数据标签全部一致才允许准备训练。核验与全量来源和处理方案摘要绑定：数据或方案修订后自动失效，需重新核验。P3.4 自动执行的授权入口同样前置校验。完整 CLI 用法见下。
- **可学性探针（可选证据，非门禁）**：用基座模型对开发集抽样做零样本探测，与「瞎猜多数类」基线对比并给出带样本量限制的说明。显著低于基线通常意味着提示格式或任务定义需要先核查；高于基线也不能预测微调效果。CLI：`learnability-probe SESSION --revision R --model-path 目录 [--size 8]`；页面在「训练前检查」区提供入口（会实际加载本地模型）。两个 CLI 命令的 stderr 都先给判定行（`describe_probe_verdict`）：亮出基座零样本、瞎猜多数类基线与差异三组数字及三态判定词汇（零样本高于／不低于／低于瞎猜基线——低于时指向先核查提示格式与任务定义），再原样复述 note 的样本量与不预测微调效果边界，然后才列候选清单；`learnability-probe-show` 回读同样先给判定行。三态词汇由 `probe_verdict_phrase` 单一来源输出，页面与 CLI 不各说各话。判定行与 note 之后，低于瞎猜基线时还会按记录内事实逐行输出核查方向分辨（`low_baseline_triage_lines` 单一来源，页面探针区与两个 CLI 命令的 stderr 渲染同一份行）——「提示模板问题还是任务定义问题」由可观察事实分流：有生成被截断→被截断的输出已按不匹配计、分数被截断压低，先加大 `max_new_tokens` 重测再判断方向；基座输出不在这份开发集的标签里出现过→模型没有用任务的答案词汇作答，先核对提示模板是否讲清作答口径与词汇（补全式模板与对话型基座不匹配是常见形态）；全部未截断输出完全相同→没有按输入区分作答，模板没把任务讲清与输入本身缺少区分信息两种可能都在，先补清指令或换对话式模板重测、仍同答再核对输入；输出用任务答案词汇且按输入区分作答仍低于基线→更像任务定义或标注口径的问题，先核对类别边界与标注规则。尾行给出对号处理映射：模板没讲清就补指令或换模板后重测（不动数据）；输入缺信息或类别边界不清就补输入字段、澄清标注口径（改任务定义）；两边都核对过分数仍低，如实在记录里保留低分证据，不硬修。分辨行只给核查方向，不认定原因。探针记录现在携带 `label_vocabulary`（开发集标签全集，供词汇分辨），早期存档没有该字段时跳过词汇检查、其余分辨照常，如实降级不报错。弱信号候选的改标签门槛（`WEAK_SIGNAL_RULE` 单一来源）同源出现在候选 note、CLI 清单表头与弱信号证据列：单凭模型不认同不改标签，人工核对后仍不认同才修正数据。

这三道关卡都不使用 LLM 判断、不消耗模型服务额度，判定全部确定性可复现。

对比核验的 CLI 用法与 `confirm` 的如实提示：`contrast-check SESSION_ID --revision CURRENT_REVISION` 抽出两条答案不同的输入与打乱后的两个候选答案——stderr 逐条列出「[行ID] 题目输入」与候选清单，stdout 为纯 JSON（`check_id`、`items`、`options` 与可照抄的 `submit_hint`）；`contrast-check-submit SESSION_ID --check-id CHECK_ID --answer 行ID=候选答案 --answer 行ID=候选答案` 提交配对，需恰好覆盖两条输入、答案来自候选。判定行如实亮出配对计数与连胜口径：verified 但连胜只有一轮时明确提示「还需再连续配对正确一轮（二连对）才算真正看清」，二连对及以上提示达标——防瞎蒙靠的是连胜不是单轮；三轮及以上的连胜在达标句后追加「对比核验N轮连胜：转换的业务含义经多组不同题目反复配对核对」（`contrast-check-submit` 判定行与 `confirm` 提示同样追加，由 `contrast_streak_banner` 单一来源输出，页面横幅与 CLI 不各说各话）；配错判定行报 mismatch 配对计数并提示此前的确认可能是盲点头。`confirm` 在 stderr 如实报告当前配对状态：尚未核验（建议先运行 `contrast-check`）、已连续 N 轮配对正确（达标或还差一轮）、或最近一次配对错误——对比核验是软门禁，配错留档但不阻断确认，去留由你决定。核验与预览及方案绑定：提交不收 `--revision`，预览或方案变化后原核验自动失效。

### 盲标核验的完整 CLI 用法

```sh
python scripts/data_intake.py label-verify SESSION_ID --revision CURRENT_REVISION --size 5
# --size 在 1–50 之间自选（默认 5）；加 --export-csv 路径 可把题目清单导出为 CSV 供线下作答。
python scripts/data_intake.py label-verify-submit SESSION_ID --verification-id VERIFICATION_ID \
  --answer 行ID=你的答案 --answer 行ID=你的答案
```

`label-verify` 抽题后依次输出：提示「请仅根据输入作答，不要查看数据中的现有答案。」；`evidence_note` 证据预告——本轮 N 条即使全部一致，95% 置信下真实一致率下界也只约多少（例如 5 条约 57%），小样本下下界才是你能依赖的数；已标注行不足所选条数时再输出 `shortfall_note`，说明按现有全部已标注行抽取、统计按实际条数口径；随后逐题列出「[行ID] 题目输入」。标准输出为 JSON，含 `verification_id`、`sample_size`、`row_ids` 与可照抄的 `submit_hint`。`--export-csv` 导出的清单带 BOM、Excel 直开，只含「行ID、题目输入、留空待填的盲标答案」三列，不含数据答案——导出文件泄露答案，盲标核验就失效；线下作答后逐条照抄回 `label-verify-submit` 提交。

`label-verify-submit` 不收 `--revision`（核验结论由全量来源与处理方案摘要绑定，数据或方案变化后原核验自动失效）；每条抽样行一个 `--answer`，行ID=你的答案，需恰好覆盖全部抽样行。人读判定行形如「判定：verified（5/5 一致，95% 置信下界约 57%）」，通过与未通过两种判定同口径亮出下界数字，观测一致率不冒充真实水平；通过时另附「盲标核验通过：监督信号的业务含义经独立复现。」，未通过时另附「存在不一致，训练不会开始；请核对数据标签或业务定义后重新核验。」JSON 结果同样携带 `agreement_lower_bound` 与 `evidence_note`。

未通过时，判定行之后还会按记录内事实逐行输出三因分辨（`src/workbench/intake_service.py` 的 `mismatch_triage_lines` 单一来源，随结论存进记录的 `mismatch_triage` 键，stderr、页面未通过区与记录渲染同一份行）：答案词汇在这份数据的全部标签里没有出现过——先核对双方是否在用同一套类别词汇（任务定义不清的典型形态）；同一对「数据标签→你的答案」重复出现、或几处不一致都给了同一个答案——同一方向的错位不像随机记错，更像两个类别的口径或边界没对齐；仅 1 处不一致——先展开这条记录核对输入信息是否足以判断；不一致分散在不同类别之间、没有共同方向——更像逐条的问题，逐条展开核对。分辨行只给核查方向，不认定原因；尾行给出对号修正映射：核对后你的答案对就修正数据标签（改数据），数据标签对就把类别边界或输入信息补清（改方案），并提醒重新核验会换一组题——照抄上一轮公布的答案无效。通过态与早期存档记录没有该键，如实缺席。

## 生成独立数据分区与版本

全量确认后，页面可按已确认分组字段生成训练、验证和独立测试集。没有分组字段时，需要先明确确认每行属于独立业务对象；相关客户、会话或文档不能为了凑比例而拆开。页面展示实际数量、数据集版本及本地文件，并提供三个分区和训练数据配置的下载。

```sh
python scripts/data_intake.py materialize SESSION_ID --revision CURRENT_REVISION \
  --name my-domain --validation-fraction 0.1 --test-fraction 0.1 --seed 42
```

可用 `--registry-root` 指定数据注册目录；无分组字段且已完成业务核对时加 `--independent-rows-confirmed`。不足三个独立组会要求补充资料，不能据几条相关样例宣称完成独立评测。分组隔离优先于比例，报告显示实际比例；目前不做类别分层。

分区统计如实披露答案覆盖：当全部答案的不同取值不超过 20 种时，统计携带 `answer_counts_by_split`（各分区的答案构成）与 `train_missing_answers`（训练集没见过的答案及其落在验证/测试的条数）；存在这类答案时另附 `answer_coverage_note` 如实点名，例如「验证/测试集中有 1 类答案（屏幕×1（测试1 条））从未出现在训练集——训练按逐字学习答案，模型没有学过这些值，验证与测试仍会照常打分」。分组隔离可以把稀有类别的整组记录全部分进验证或测试（未做类别分层），训练按逐字学习意味着模型无法输出没学过的值，而验证与测试照常打分——此前这一形态完全无声。补充该类别的独立业务对象后可重新生成分区版本；没有自动重新切分，也不会把记录挪回训练集。时间分区与固定题集沿用同一披露口径，成因分别按时间边界与固定题集说明。答案不同取值超过 20 种（开放文本、大量类别）时不计算该统计——答案几乎必然互不相同，逐条点名没有信息量；统计里缺这两个键就是这一如实边界。页面数据集版本区的摘要与 CLI `materialize` 的输出都会展示这条披露。

分区统计同样如实披露完全相同的例题：统计一直携带 `rendered_exact_duplicate_rows`（渲染后输入与答案逐字一致的重复条数）与 `source_exact_duplicate_rows`（原始行每个字段完全一致的重复条数），存在渲染重复时另附 `duplicate_note` 如实点名，例如「本版本有 2 条记录与前面的记录渲染后完全相同（输入与答案逐字一致）——同一道例题会出现多次，训练等效于给这些例题加权；12 条记录去重后只有 10 道独立例题。其中 1 条原始行完全重复（每个字段都一致，多见于导出拼接或关联重复）；另有 1 条是不同原始行渲染成同一例题（原始字段不同、例题相同）。」重复例题等于训练隐式加权，此前这两个计数只躺在统计里、不进任何人话摘要。两种成因分述：原始行完全重复多见于导出拼接或关联重复；不同原始行渲染成同一例题说明字段选择让多行携带了相同信息。与冲突守卫的边界：相同输入配不同答案会被全量验证硬拦，不会出现在任何分区；完全相同的记录经相同输入连接成同一分组、永不跨分区。没有重复条数（计数为零）时统计缺 `duplicate_note` 键。披露不阻断、全部记录原样保留、没有自动去重，去留由用户决定。页面数据集版本区的摘要与 CLI `materialize` 的输出都会展示这条披露。

CLI `materialize` 在 stdout 输出 JSON 任务记录的同时，向 stderr 追加与页面数据集版本区同一口径的分区人话摘要（`summarize_dataset`，三种切分方式统一）：分法句（按业务对象隔离划分／按已确认的时间边界划分／沿用固定开发/测试题集）、各分区计数与独立分组数、比例受分组大小影响的如实说明、答案覆盖与完全相同例题披露，并以边界句「分区就绪只说明数据已按规则隔离、可以进入训练前检查；不代表模型效果或业务达标」收尾。此前分组与固定题集路径在 CLI 上零人话，时间路径则是 CLI 自造句式、与页面词汇不一致。时间方案在摘要之外仅补一行逐原因排除计数（`exclusion_counts`）与 `dataset.paths.manifest` 的 `metadata.excluded_rows` 原行明细指引——摘要句说明总数与成因，逐条计数与原行入口由补充行承载。

分区设置的「何时该改」引导与独立测试集条数提醒同样统一出自 `src/workbench/training_guidance.py`（`split_settings_guidance_lines` / `small_test_set_line` 单一来源）：页面「分区设置」折叠区与「数据集版本与分区产物」区、CLI `materialize` 决策点 stderr 与 argparse help 输出同一份行——页面与 CLI 同源同词汇。独立测试集少于 30 条时按 1/N 诚实算术点名每题约占的百分点，30 条是粗略经验阈值、不是统计保证；比例与种子何时该改（几百条记录时比例可提到 0.15–0.2、种子只在想换一种切分时改）由同一来源给出，不替用户决定比例。

物化后状态为 `ready_for_training_preflight`，表示继续训练前检查，模型还未训练。输出的 `dataset.data_config` 是完整训练配置中 `data` 部分：使用固定的 `train_file` 和 `validation_file`，`validation_split: 0` 防止再次随机切分，`dataset_loader: alpaca` 明确采用已生成的数据格式，避免由文件路径猜测领域并再次过滤记录。独立测试文件单独用于最终评测。单独把训练路径交给会重新切分的旧入口不能替代这套配置。

## 时间预测任务：先核对来源与标签窗口

只有目标涉及未来结果时，Agent 才会提出时间方案；普通分类、抽取或文本任务继续原流程。先说明模型在什么时点作预测、当时实际可得什么资料、未来答案如何观察。报告期或文件日期不能替代公开可得时间，系统不猜时区、不补伪真值。

时间方案有两条并行路径。配置了 Agent 时由 Agent 核对来源与业务后提出方案（见下文）；没有配置 Agent 时，页面「没有 Agent 服务？用基础分析开始」区可勾选「按时间分区切分」由用户直接指定：数据须是单表，且已存在三个互不相同的时间字段——信息实际可得时间、作出预测时间、标签窗口结束时间（每行为带时区的 ISO 时间，且满足 可得时间 ≤ 预测时间 < 标签窗口结束时间）——再填写验证起点、测试起点、观察截止三个递增边界。基础分析只核验字段存在、时间格式与行内先后顺序（选错字段或边界格式错误在生成时就地报错，不会退回随机切分），不判断业务时间含义：边界是否符合真实业务节奏、标签窗口结束时间在预测时是否确实未知，由用户在真实预览中确认。零密钥路径的能力边界：这是单表三字段方案，不派生标签——监督答案列必须已在数据中存在；需要按行情与日历派生未来标签（`forecast_labels`）、多源组合或业务问答时，仍需配置 Agent。两条路径此后的真实预览、全量验证与分区物化遵循同一套时间分区规则。

如果目标是根据公开事件预测之后若干交易日的市场方向，可分别上传三份本地资料：事件表、明确价格口径的行情表、独立的交易所收盘日历。日历 CSV 每行是明确带时区的实际收盘时点，需要覆盖事件之前至预测窗口结束；不能把行情行数当作交易日数，也不能用自然日推算缺少的交易日。事件表需有单一事件标识、标的和实际公开可得时间；行情需有标的、带时区的收盘时点、明确的正价格。

Agent 通过 `forecast_labels` 调用现有真实组合预览，按用户确认的预测交易日数、观察截止及价格口径生成标签。价格口径明确选择拆股调整收盘价（`split_adjusted_close`）或含分红总回报调整价（`total_return_adjusted_close`），并与提供的行情来源一致。日历和价格完整性需要业务核对；缺行情、缺时间或日历覆盖不足会报告具体问题，不跳过缺失交易日计算方向。目前直接读取本地上传文件，不连接或下载行情平台。

组合预览保留每个事件及其来源，派生 `prediction_at`、`label_end_at`、参考/目标价格、实际收益、方向和状态等字段（默认加 `forecast_` 前缀）。未来价格、收益、方向、标签窗口结束时间和标签成熟状态不能进入模型输入。窗口尚未结束的方向保留为空，状态为 `label_not_observed`，不会填成“未上涨”。若缺任何必需来源，页面会分别要求事件、行情、日历的全量文件，不要求用户自行拼表。

```sh
python scripts/data_intake.py create --goal '描述实际预测目标、时点和可得资料' --input events.csv
python scripts/data_intake.py add-source SESSION_ID --revision CURRENT_REVISION \
  --alias prices --input prices.csv --description '说明标的、收盘时间和真实价格调整口径'
python scripts/data_intake.py add-source SESSION_ID --revision CURRENT_REVISION \
  --alias calendar --input exchange-closes.csv --description '交易所实际收盘时点，含时区及完整覆盖范围'
python scripts/data_intake.py analyze SESSION_ID --allow-remote-data
# 核对 Agent 方案与实际预览后确认；每次修改后用最新 revision。
python scripts/data_intake.py full-sources SESSION_ID --revision CURRENT_REVISION \
  --source main=full-events.csv --source prices=full-prices.csv --source calendar=full-exchange-closes.csv
```

`group_columns` 中多个字段表示“任一共同字段把记录关联为同组”，不是复合主键。因此不能把 `[ticker, period]` 当作一个独立事件标识；需要先根据真实业务定义生成并核对单一 `event_id`，再按它分组，防止无意将所有同标的或同报告期连成大组。

## 按已确认时间边界生成分区

时间预测配方的 `temporal_split` 指定三个不同字段：信息实际可得时间、预测时间、标签窗口结束时间，以及验证起点、测试起点、观察截止三个递增边界。字段值和边界均须包含明确时区，且每行满足“可得时间 ≤ 预测时间 < 标签窗口结束时间”。界面展示这些字段、日期和业务含义，随真实预览由用户确认；缺字段或含义不清时，Agent 应先提问。

时间方案按已确认窗口分配：训练标签须在验证起点前成熟，验证标签须在测试起点前成熟，测试标签须在观察截止前可见。跨窗口、截至观察期仍未成熟及与这些记录相连的行明确排除并保留原行、输入、标签和原因。同事件或相同模型输入跨时间分区会阻断，不能靠随机换组绕过。已成熟却缺标签、时间非法或预测时尚不可得的信息仍是问题，不能借“尚未成熟”排除掩盖。

含合法未成熟行的样例可以继续核对，但样例确认和全量确认至少需要一条真实有标签且已成熟的记录；仅有未来待观察数据不能宣称具备训练监督。未成熟行不保存为已认可的答案样例。正式物化还要求训练、验证、测试三个分区都非空。

```sh
python scripts/data_intake.py materialize SESSION_ID --revision CURRENT_REVISION
```

无需额外填写时间参数，物化使用已确认的时间方案配方（Agent 提出或零密钥基础分析中用户指定）。时间方案不展示随机比例/种子设置，也不会随机回退；CLI 的 `--validation-fraction`、`--test-fraction` 和 `--seed` 对时间方案不生效，并给出明确提示。页面和 CLI 展示纳入与排除数量及原因；完整排除记录保存在 `dataset.paths.manifest` 指向文件的 `metadata.excluded_rows`，页面可查看与下载。已冻结题集仍须与时间方案兼容，不会通过随机重分配原测试题。

## 检查实际 token 与答案保留

生成数据分区后，在页面「训练前检查」填写本地 tokenizer 目录或已缓存标识以及计划采用的最大 token 长度，点击「检查实际截断与答案保留」。只有点击按钮才会加载 tokenizer；不会自动下载文件、加载模型权重或启动训练。

```sh
python scripts/data_intake.py preflight SESSION_ID --revision CURRENT_REVISION \
  --tokenizer ./models/my-base-tokenizer --max-length 2048
```

本地目录缺少文件或缓存不存在时，先准备对应基础模型的 tokenizer，再执行检查。`2048` 仅为命令示例，应根据实际模型和业务上下文选取。

报告保存为 `training_preflight`，状态包括 `blocked`（存在阻断问题）、`warnings`（需核对风险）、`passed`（训练前检查通过）。人话状态句由 `summarize_preflight` 单一来源产出（CLI 与页面同句，R95 起页面不再手抄第二套）。报告列出各分区的截断和答案丢失统计、问题行以及实际 token 消费；答案完全丢失会阻断。tokenizer 无法提供答案边界时明确标记未验证。检查依据当前训练模板、因果位移和 padding 屏蔽方式，不据此声称业务效果已验收或硬件足以训练。

## 准备本地基础模型

上一节的检查只读取 tokenizer，可学性探针与训练需要完整权重；两者都要求模型文件已在本地——工作台不自动下载模型。用 `pip install -e ".[ui]"` 自带的 huggingface_hub `hf` 命令把模型下载到项目下的 `models/` 目录（与 README 快速开始同一命令）：

```sh
hf download Qwen/Qwen3-0.6B --local-dir models/Qwen3-0.6B
```

国内网络先 `export HF_ENDPOINT=https://hf-mirror.com`。下载完成后，页面「用当前数据微调模型」的「本机已准备的候选模型」与 `model-list` 都会自动发现它——发现由 `src/workbench/local_models.py` 的 `discover_local_models` 完成，扫描 HF/ModelScope 缓存目录与项目下的 `models/`，只读文件、不加载权重。文件完整只表示可以进一步检查；模型是否兼容、训练长度和机器是否适合，仍由训练前检查与方案检查判断。模型放在其他位置时，页面「高级：补充本地模型路径与发现详情」与 `model-list --root LOCAL_DIRECTORY` 可指定目录。训练前检查只需要 tokenizer 时，任何已下载模型的目录都可直接填写；探针与训练则需等权重文件下载完整（发现结果会标注文件是否完整）。

## 让 Agent 推荐训练方案

确认数据并生成分区后，「用当前数据微调模型」会发现本机已准备的模型，列出文件完整的候选供点选。只选择本轮需要比较的模型，避免核查不相关的大型权重；其他位置的已有模型可在「高级：补充本地模型路径与发现详情」中填写，每行一个。发现只检查已知缓存和模型目录的本地文件，不下载、加载或计算权重哈希；文件完整也不等于训练兼容，Agent 后续还会检查候选事实和真实 tokenizer。点击「让 Agent 推荐训练方案」后，Agent 读取业务目标、处理方案、实际数据统计、本机条件及候选模型事实，并对选择的模型执行真实 tokenizer 预检。它不会自动下载模型、加载训练权重或启动训练。Agent 读取的数据就绪说明（`data_readiness.required_actions`）同样带出口行：每条门禁句之后附 `next_action_phrase` 单一来源的下一步状态人话——Agent 判断「需要先完善数据」时读到的出口行与 CLI 尾行、页面提示同词汇，不为 Agent 另造一套出口；就绪态没有门禁句，`required_actions` 为空。该说明仍只含状态、字段和计数，不包含原始行。

方案显示推荐理由、训练长度、轮数、batch size、学习率等关键参数，以及实际检查记录。`ready` 表示可供确认；`needs_data` 提示先完善数据；`unsupported` 表示当前模型或机器条件不支持。预检通过不保证内存一定足够，也不代表业务效果达标。核对方案后点击「确认推荐方案并准备训练」，随后使用已有训练记录的「启动这轮训练」。已保存方案绑定任务版本、实际数据及模型内容，发生变化需重新推荐。熟悉训练参数的用户仍可展开「高级：手工配置训练参数」。

远程 Agent 需要独立确认发送任务与处理方案、数据统计、模型配置及本机硬件摘要；不会发送 API 密钥或训练、开发、测试原文。BYOK 继续使用公共配置和环境密钥，无额度或计费设置。

```sh
python scripts/data_intake.py model-list
python scripts/data_intake.py plan-recommend SESSION_ID --revision CURRENT_REVISION --allow-remote-data
# 需要指定其他已有模型时，可追加 --model-path ./models/local-base-a（可重复）。
python scripts/data_intake.py plan-list SESSION_ID
python scripts/data_intake.py plan-show PLAN_ID
python scripts/data_intake.py plan-prepare SESSION_ID PLAN_ID --revision CURRENT_REVISION
python scripts/data_intake.py train-start SESSION_ID RUN_ID --revision CURRENT_REVISION
```

省略 `--model-path` 时自动采用发现的完整本地候选；没有完整模型时会提示缺少文件，不会自动下载。`model-list --root LOCAL_DIRECTORY` 可只查看指定目录，`--root` 可重复。`model-list` 同时在 stderr 追加发现尾行（`summarize_model_discovery` 单一来源）：空态点名先准备本地候选模型或用 `--root` 指定目录；非空给文件完整/不完整分档计数（缺什么看 JSON 的 issues 字段），并复述「文件完整只表示可以进一步检查」的边界——与页面候选模型区同词汇。`--model-path` 可以重复；本地 Agent 服务不需要 `--allow-remote-data`。方案默认保存在 `outputs/workbench/training-plans`，全局 `--plan-root` 可覆盖目录。`plan-prepare` 表示已经审阅并确认保存的推荐方案，只准备训练，不自动启动。

上述 plan 子命令在 stdout 输出 JSON 的同时向 stderr 追加人话（`plan-list` 只追加一行清单尾行：`summarize_listing` 单一来源——空清单点名下一步入口，非空给计数）：`plan-recommend` 与 `plan-show` 先翻译方案本身——建议的基础模型与关键参数（最大长度、训练轮数、batch size、学习率、LoRA rank、量化位数）、状态三态（方案可供确认／需要先完善数据／当前条件不支持）、推荐理由与尚未验证的限制原文、需要你先回答的业务问题；ready 方案再追加其携带的真实预检证据的人话翻译（预检记录在方案记录的 `probe` 里）。`plan-prepare` 的结果复用训练记录摘要：方案已准备好并通过检查，还没有开始训练。确认这份方案只会准备训练、不会自动启动，方案就绪与推荐理由也不构成训练效果或业务达标的判断。方案摘要另以评测解读同一格式渲染工具核查轨迹（`summarize_tool_trace` 单一来源）：「工具核查轨迹：N 次调用，成功 X 次、失败 Y 次——失败的调用没有取到证据，方案只依赖成功的调用」该行由摘要函数自动带上；页面在方案记录的「候选事实与真实检查记录」折叠区渲染同一份摘要，CLI 与页面同源同词汇。

## 训练启动前对齐：任务规约投影

启动训练前，可先看这份任务的「规约」——我们在教模型什么、按什么口径验收。规约不是新的登记表，而是由既有确认记录只读汇编的投影（设计文档 ADR-1）：业务目标（目标与成功标准）、答案语义（监督来源与盲标核验结论）、评分口径（已确认的自定义规则，无则默认严格匹配）、验收标准（冻结条款或如实显示「未冻结」）、时间约束（时间预测任务的时间字段与窗口），外加最新改进轮假设。投影只反映已确认的事实——草稿评分规则不进入规约、未冻结验收不假装存在标准，这是特性不是缺陷；查看规约不写任何状态。

```sh
python scripts/data_intake.py task-spec-show SESSION_ID
```

`task-spec-show SESSION_ID` 在 stdout 输出规约 JSON、stderr 逐行追加人话摘要（`summarize_task_spec` 单一来源），页面在「独立数据分区已生成」之后、「训练前检查」之前的「📋 任务规约投影」折叠区渲染同一份内容——页面与 CLI 同源同词汇。数据来自 `--store` 与各 `--*-root` 记录目录（session + analysis + scoring + acceptance + 最新 iteration），读取失败会如实报错而不是返回半份规约。固定收尾句写明：这是训练启动前的对齐视图，只汇编已确认事实，不代表模型效果达标。

训练启动这个用户决策点同样读这份投影。页面在「启动这轮训练」按钮旁提供「📋 任务规约（启动本轮训练前的口径）」折叠区，渲染同一份规约摘要（训练准备完成后、原「📋 任务规约投影」卡不再可见时，规约在决策点仍在场）；`train-start` 在启动时输出同一份规约摘要至 stderr（`summarize_task_spec` 单一来源，位于原有运行摘要之前），页面与 CLI 同源同词汇。预检警告确认勾选文案为「已核对任务规约与预检提示，按当前方案开始训练。」；勾选仍只表示已核对，不代表模型效果达标。

## 在同一任务中启动真实训练

数据版本生成后，页面「用当前数据微调模型」可填写本地基础模型目录，选择最大 token 长度、训练轮数和 batch size。基础参数还包括学习率、梯度累积、LoRA rank，以及兼容 NVIDIA CUDA 环境下可选的 4-bit 量化。

点击「准备本轮训练方案」后，系统使用所选模型自己的 tokenizer 重新预检，并保存与当前数据版本绑定的训练配置。展开本轮配置与预检记录即可核对；有阻断问题时不能启动，有风险提示时先确认已经核对。点击「启动这轮训练」才会运行已有 SFT 训练链路，无需手工编写 YAML 或复制命令。

同一页面保存每轮记录，可刷新状态和最近日志、停止运行中的训练，以及查看输出目录和已经存在的产物。训练失败显示阶段及错误信息；训练成功仍需使用独立测试集检验业务效果。准备方案后如果数据或业务方案已变化，需重新准备，不能把旧训练方案直接应用到新数据。

CLI 也使用同一套训练记录，默认目录为项目下的 `outputs/workbench/training`，可用子命令前的全局 `--training-root` 改变目录：

```sh
python scripts/data_intake.py train-prepare SESSION_ID --revision CURRENT_REVISION \
  --model-path ./models/my-local-base --max-length 1024 --epochs 1 --batch-size 1
python scripts/data_intake.py train-start SESSION_ID RUN_ID --revision CURRENT_REVISION
python scripts/data_intake.py train-status RUN_ID
python scripts/data_intake.py train-logs RUN_ID --tail 100
python scripts/data_intake.py train-stop RUN_ID
python scripts/data_intake.py train-list SESSION_ID
python scripts/data_intake.py train-lineage RUN_ID
python scripts/data_intake.py train-export RUN_ID
python scripts/data_intake.py train-cost RUN_ID
```

`RUN_ID` 来自准备结果；有预检风险且已完成核对时，启动命令增加 `--acknowledge-warnings`。`train-prepare` 另支持 `--learning-rate`、`--gradient-accumulation`、`--lora-rank` 和 `--load-in-4bit`。这里需要已准备好的本地基础模型目录，工作台不会悄悄更换底座或下载另一个模型。

手工配置训练参数的判断辅助统一出自 `src/workbench/training_guidance.py` 单一来源：参数大白话与推荐起步值由 `manual_training_parameter_lines` 给出，页面「高级：手工配置训练参数」表单的逐参数说明、CLI `train-prepare` 的 argparse help 与决策点 stderr 先看的引导输出同一份行——页面与 CLI 同源同词汇。推荐起步值（小数据：训练轮数 1–2、每设备 batch size 1、梯度累积步数 4、学习率 2e-4、LoRA rank 8）与学习率分档（<2,000 条 5e-5~1e-4、≥2,000 条 2e-4）是外部指南的汇总启发（Unsloth 指南、Raschka 实践笔记等），非本产品实测，以你自己的同题对照结果为准。

返回训练记录的命令（`train-prepare`、`train-start`、`train-status`，以及方案入口 `plan-prepare`）在 stderr 追加同一份运行摘要（`summarize_training_run` 单一来源）。运行摘要现以评测解读同一格式渲染方案阶段的工具核查轨迹：经方案准备时，方案执行把方案 trace 快照进运行记录（`plan_trace` 键），轨迹行「工具核查轨迹：N 次调用，成功 X 次、失败 Y 次——失败的调用没有取到证据，训练方案只依赖成功的调用」由 `summarize_tool_trace` 单一来源产出；不经方案、直接 `train-prepare` 准备的运行没有这一行——如实缺席，不编造轨迹。

训练完成后的指标呈现面向非专家：训练工作进程把逐 step 的 loss 序列单独落盘为产物目录里的 `workbench_loss_history.json`（flat metrics 压平后只剩每个键的末值，曲线必须由这份序列重建）。这份序列也是「实时」的写入侧：工作进程通过 `LiveLossWriter` 回调在每次训练日志点后立即把已有点位重写进同一份文件（`run_sft_training` 的 `extra_callbacks` 参数在训练开始前注册），训练进行中页面与 CLI 读到的就是已训练部分的曲线；训练结束时工作进程再用完整 `log_history` 重写同一文件作为权威序列。页面训练记录的「本轮训练指标」区用这份序列渲染 loss 曲线并配趋势人话，原始 flat JSON 收进「查看原始指标 JSON」折叠区保透明，不再裸倾倒；`train-status RUN_ID` 在记录带训练指标或处于未完成状态时向 stderr 追加同一份趋势人话（`loss_trend_lines` 单一来源，页面与 CLI 同源同词汇）。趋势句先给首末位置与记录点数，再按首段/末段均值对比给出三态结论（整体在下降 / 整体基本持平 / 末段反而更高）；持平提醒「loss 没降下来不代表训练失败，先核对配置是否按方案执行」，不降反升点名「需要核查——常见方向是学习率过大或数据里有异常样本」，都只给观察事实与核查方向、不认定原因。每条趋势都以「loss 下降只说明模型在逐步记住训练题，不代表业务效果；效果要看同一套开发题上的对照报告」收尾。旧产物目录没有这份序列文件时如实说明没有逐条记录，不编造曲线。三态判定只在训练已结束时给出：训练进行中（running/stopping）页面在「训练中的 loss 曲线」区画到最近一次日志点并提示「刷新页面查看最新进度」，趋势句只给跨度和「训练进行中，趋势判定等训练完成后再看」——半程数据不足以支持整场结论；CLI `train-status` 在未完成状态采用同一口径。训练中断（stopped/failed）时页面在「训练未完成时已记录的 loss 曲线」区给出完整趋势人话并注明「只代表已训练的部分」。

`train-lineage RUN_ID` 在 stdout 输出 JSON 的同时向 stderr 追加这次训练在模型库的注册状态人话摘要（`summarize_registration` 单一来源）：已注册时点名全部版本与别名（如 champion）；尚未注册时如实说明，并给出可照抄的合并与注册命令（带血缘旗标，注册后可反查本轮训练与数据版本）；查询失败如实报告原因，不编造状态。已注册与未注册两种状态都以「注册只说明模型库记录了这次训练的产物与血缘，不代表业务效果达标」收尾。页面训练记录区渲染同一份摘要，页面与 CLI 同源同词汇。反方向用 `python scripts/registry_cli.py lineage --model-name NAME --version N` 从模型库版本查回训练运行、数据版本与配置摘要（`summarize_lineage` 单一来源，缺项如实显示「-」，不显示 None）——模型、实验与数据三点由此可以互相追溯。

`train-export RUN_ID` 是旅程的最后一环（交接）：把这次训练的适配器合并进它使用的基础模型，产出可被 vLLM、Ollama、LM Studio 直接加载的完整模型目录（默认 `outputs/workbench/merged/RUN_ID`，可用 `--output-dir` 覆盖）。合并复用框架层 `merge_adapter_to_dir`，基座路径取自训练记录，不下载、不更换底座；导出完成后在模型目录写入 `export_evidence.json` 证据链（来自哪次训练、哪个数据版本、哪个基础模型、何时导出）。命令在 stdout 输出 JSON、stderr 追加人话摘要（`summarize_export` 单一来源）：导出完成时点名输出目录、可直接加载的事实与证据文件；此前已导出则如实说明目录已完整、重复导出不会改变模型内容（幂等）；盘点不过关（训练未成功、产物目录缺少 adapter 文件、基础模型目录不存在、导出目录已被其他内容占用）时命令直接失败并逐条点名原因，不静默降级。每条摘要都以「导出只产出模型文件与证据记录，不代表业务效果达标，也不会自动部署」收尾——与验收/采用记录同口径。页面在成功训练记录的「📦 合并导出」折叠区渲染同一份摘要（只读盘点，不在页面执行合并），页面与 CLI 同源同词汇。

`train-cost RUN_ID` 把训练成本账带到 CLI（度量体系「成本可计算」的平权入口）：stdout 输出 JSON 成本账、stderr 逐行追加人话摘要，与页面在成功训练记录下渲染的成本账同一来源——`summarize_run_cost` 出账、`cost_lines` 出人话，页面与 CLI 同源同词汇。可选 `--api-price`（元/百万 token）与 `--monthly-queries`（次）给出 API 对比口径，两者留 0 不对比（与页面输入同款默认）；`--device` 可指定 cuda/mps/cpu，默认自动检测本机设备。口径边界与页面一致：时长是实测，功耗与电费是估计值、不是电表读数（按设备类别估计功耗与居民电价，硬件购置成本不含在内）；API 对比与本地训练两者口径不同，仅供量级比较，不是精确账单。

## 允许一次显存不足技术恢复

启动前可选「允许显存不足后自动技术重试一次」，默认关闭。开启后，只有真实训练步骤提供内存不足证据时，后台才会尝试缩小微批次并增加梯度累积以保持有效 batch，或在微批次已为 1 时启用梯度检查点。业务目标、数据分区、模型、LoRA、学习率和训练轮数继续沿已确认方案；没有可用调整、非内存不足、准备阻断或重试仍失败时，都保留失败信息。

原训练保留 `failed` 和实际错误，恢复训练另建关联记录，页面显示恢复依据、参数变化、子训练状态和产物。一次技术恢复不算新的业务假设，成功后仍需用相同固定题集检验效果。查询和刷新只读状态，不会触发恢复。点击「停止此训练及其自动恢复」，或对原训练运行 `train-stop`，会取消后续恢复并停止已关联的子训练。

```sh
python scripts/data_intake.py train-start SESSION_ID RUN_ID --revision CURRENT_REVISION --recover-technical-failures
# 已确认改进轮次也可采用相同授权：
python scripts/data_intake.py iteration-start SESSION_ID ITERATION_ID --revision CURRENT_REVISION --recover-technical-failures
python scripts/data_intake.py train-status RUN_ID
python scripts/data_intake.py train-stop RUN_ID
```

父记录的 `recovery.child_run_id` 与子记录的 `recovery_parent_run_id` 相互关联；子训练不再自动重试。改进轮次的子训练成功后，可用该子训练 ID 运行 `eval-compare ... --iteration-id ITERATION_ID`，系统核验双向关联后比较基座、父轮业务模型和恢复产物。这里没有额度或预算管理。

## 比较基座与本轮微调效果

成功训练的记录下提供「基座与本轮微调：同开发集对照」。系统按训练记录选择基础模型和本轮 adapter，在同一个固定开发集上顺序加载、生成和比较；独立测试集留待最终业务验收。当前任务的数据版本需对应本轮训练；固定题集模式允许父轮使用自己的历史训练数据快照与模型证据，在同一套开发题上与新模型对照。

评分沿已确认的数据目标选择：单字段类别使用严格匹配，JSON 答案使用全部已声明字段逐项匹配；开放任务只生成并保留输出，明确标记待业务评分，不自动宣布效果通过。可以设置统一生成长度，以及严格评分是否忽略首尾空白。

页面显示总样本数、已评分数、生成失败与截断数量、严格准确率或字段准确率。失败与截断保留在分母中。逐样本选择器可查看每一条的完整输入、期望答案和各个模型的完整输出；完整报告也可以下载。模型释放失败或评测未完成时，不将部分结果当作完整对照。

JSON 答案的对照额外携带逐字段准确率（`field_accuracy`，按每个已声明字段单独计算）：整题严格准确率只统计全部字段都对的题，模型可能每个字段都答对 9/10 却一道整题也不算答对，只看答对总数无从知道该先核对哪个字段。页面「大白话解读」与 CLI 评测输出因此会用一句人话点名每个模型最弱的字段，例如 基座最弱的是「日期」(2/10 题答对)。失败与截断保留在分母中的口径对逐字段统计同样成立：生成失败、截断或无法按 JSON 解析的题按该字段错误计入。分类、开放任务没有逐字段统计，摘要保持沉默；某模型全部字段全对时也不多说一句。

输出高度重复（输出坍缩）同样如实点名（`dominant_output_models` 单一来源，语言化摘要与页面对照区警告共用口径）：当一个模型的非空输出中同一内容占 80% 以上、且至少 4 条输出时，摘要写明哪个模型、多少条输出完全相同、重复的内容片段，并提示先对照开发集答案分布——分布本身集中时模型可能只是复述多数类，不一定是学坏了；分布不集中时需确认每题是否本该有不同答案（R101 起逐题查看出口收敛为摘要帧尾单命令，语境句不再内联查看指令）。这是小规模微调的常见失败形态（模型在所有题上输出同一答案）；此前两条截断提示里「输出是否在重复生成」的提醒没有检测器支撑，零分用户更无从知道根因。披露不认定原因、不阻断；零分归因句在观察到该形态时点名「有模型在反复输出同一答案」，没有观察到时如实说「没有观察到截断、生成失败、复述或重复输出」。生成失败（无输出）不计入占比分母。

指令回声预警的核查顺序按报告内事实分流，统一出自 `src/workbench/evaluation_diagnostics.py`（`echo_triage_lines` 单一来源）：页面对照区警告与 CLI 对照摘要（`summarize_comparison`，经 `eval-compare`／`eval-show` 的 stderr 流出）输出同一份行——页面与 CLI 同源同词汇。三个核查方向中两个按报告内可观察事实分流：回声题是否同时触及生成长度上限（是→优先核查 max_new_tokens 是否小于最短合法答案），回声题输入平均长度是否达到其余题的 1.5 倍（是→支持「过长内容淹没答案信号」方向，两个平均值同时给出供人自行核对）；模板方向只陈述报告已记录的补全式（Alpaca）模板事实，报告未记录模板类型时如实说明无法用报告内事实分流。每段引导固定以「以上是按报告内事实排出的核查顺序，不认定原因；每改一项后用同一题集复测一次」收尾——只给观察事实与核查顺序，不认定原因。

零分对照的失败原因与下一步对号，统一出自 `src/workbench/report_summary.py`（`summarize_comparison` 零分分支，页面「大白话解读」与 CLI 对照摘要同源同词汇）：摘要点名失败形态后追加一行「失败原因对号处理」——每类失败的第一步各不相同（复述题目——补数据治不了回声，先核对提示模板与指令长度；截断——先加生成长度重测，当前分数低估了模型；生成失败——先修失败原因，失败题没有测到模型；重复输出——先对照答案分布披露判断），补数据是这些技术原因逐一排除后的选项，改任务定义还是停止在排除后再按业务判断；没有观察到任何失败形态时如实说零分更可能来自答案格式不匹配，格式对得上之前补数据和改任务定义都还不是下一步。所有题在所有模型里都没写完或生成失败（全技术性零分）时，「学不出这个任务」的定性被撤回——零分只说明生成环节没走通，先修生成长度与失败原因后重测，再谈补数据、改任务定义或停止。三问（补数据、改任务定义、停止）的决定权仍在用户，软件只排核查顺序、不作统计结论。

```sh
python scripts/data_intake.py eval-compare SESSION_ID RUN_ID --revision CURRENT_REVISION \
  --max-new-tokens 256
python scripts/data_intake.py eval-show EVALUATION_ID
```

`EVALUATION_ID` 来自比较结果；加 `--keep-whitespace` 可保留首尾空白差异。报告默认保存在 `outputs/workbench/evaluations`，全局参数 `--evaluation-root` 可指定其他目录。这一步比较开发集表现，不代表已完成最终独立测试或业务上线验收。

## 让 Agent 解读结果与下一步

每份评测报告下可以点击「让 Agent 分析结果与下一步」。Agent 使用已配置的 BYOK 模型，结合目标、当前处理方案、实际模型输出和坏例核查证据；有微调 adapter 时还必须读取对应训练配置、指标及监督检查，不能把软件已有的排查信息交给用户查询。远程服务需要原有资料发送授权，并额外确认发送本次评测细节。这里发送的内容包括业务文本、真实输出及训练配置和检查摘要，不只是汇总指标；不会发送 API 密钥或完整训练/最终测试原文。

解读分别展示有证据的观察、仍需验证的原因及验证方式、建议先处理的事项、局限和必要业务问题。它不会自动改标签、删除坏例、采纳方案变更或启动下一轮训练。页面保留解读及证据引用，重新打开仍可查看；当前业务方案已变化时会标明旧解读的上下文。

```sh
python scripts/data_intake.py eval-analyze SESSION_ID EVALUATION_ID \
  --revision CURRENT_REVISION --allow-remote-data
```

CLI 同样读取现有供应商、模型和环境密钥配置。使用本地模型服务时不需 `--allow-remote-data`；远程服务必须明确允许。解读保存在评测根目录的 `assessments` 子目录，与对应评测标识及任务版本关联。

`eval-analyze` 在 stdout 输出 JSON 的同时向 stderr 追加人话摘要（`summarize_assessment`）：解读由 Agent 核查真实工具证据后给出，先复述总述原文，再把有证据的观察与待核查原因分开计数（假设不是事实，每条附验证方式）并统计共引用的工具证据处数；随后给出建议优先处理（先核查数据／先修订处理方案／先核查训练行为／先补充证据／需要业务核对）、建议下一步、解读自己声明的局限原文、需要你先回答的业务问题，以及工具核查轨迹的调用与失败计数——失败的调用没有取到证据，解读只依赖成功的调用。每条摘要都以不自动执行边界收尾：软件不会据此自动改标签、删除坏例或采纳方案变更，也不会自动启动下一轮训练；是否有效仍需你的业务判断，不代表业务效果达标。页面在每份已保存解读的核查记录下方渲染同一份摘要，页面与 CLI 同源同词汇。

## 定义并确认自定义业务评分

默认类别严格匹配、JSON 字段匹配和开放任务人工判断继续可用。需要自己的业务规则时，在「自定义业务评分」描述可判定的要求与不能接受的错误。Agent 仅读取所需开发样例及当前目标/处理方案，拟定评分代码、参数、单题通过分数及正反例，并在真实操作系统隔离环境中执行验证；隔离不可用或例子验证失败时不能确认，不会直接在宿主环境运行评分代码。

例如，业务要求可以明确规定结构中哪些信息必须存在、允许怎样的数值误差、什么违规内容必须扣分。只有“更专业”“质量好”等主观要求时，Agent 会提出必要问题，用户补充可判断的业务边界后再拟定规则；不会用几个关键词伪造专业质量评估，也没有自动引入另一个模型作为裁判。

页面展示真实正反例得分与理由、隔离验证结果和单题通过分数，源码和预期样例可折叠查看。用户先核对这些实际结果，再勾选确认并点击「确认这套业务评分规则」。Agent 不会自动确认。已确认规则绑定业务任务及语义；数据修订后可继续使用兼容规则，业务目标或输入/答案含义变更时需重新确认。

开发对照和最终验收都可从「本次使用的评分规则」选择已确认规则。自定义对照显示**业务分均值**和**业务通过率**：前者是规则得分的平均数，后者是达到已确认单题通过分数的比例；两者不称为严格准确率。失败与截断仍计入总题数，不能当作通过。逐条输出同时显示实际分数与评分理由。

页面「大白话解读」与 CLI 评测输出对自定义对照同样用这一口径翻译：逐模型句报「业务评分均值 X、通过 Y/N」，不把通过数写成「答对」，也不触发「零分/答对最多」这类只对严格准确率成立的结论句；想定位扣分点，摘要帧尾单处给出「逐题查看评分理由:eval-show <evaluation_id>。」命令（R101 起由句中内联改为帧尾单命令，严格/自定义帧整帧至多一条，多语境叠加时目标保序去重拼合为「完整输出与评分理由」；开放任务帧不点名命令——r98-reviewer nit-1 既定决策）。开放任务的对照在人话摘要里只报生成事实（生成了 N 条回答与失败、截断计数），明确说明开放任务不做自动评分、每题通过与否由用户逐题判断——生成失败、缺失或截断的回答不能人工标为通过。

最终验收采用这套规则时，用户另外填写的「最低通过率」是整套测试的业务门槛，区别于规则中每一题的通过分数。规则引用与评分协议一起冻结，最终题和坏例仍不发送给 Agent 优化。

```sh
python scripts/data_intake.py scoring-propose SESSION_ID --revision CURRENT_REVISION \
  --business-standard '说明可以实际判定的业务标准与扣分情况' --allow-remote-data
python scripts/data_intake.py scoring-show SCORING_ID
python scripts/data_intake.py scoring-confirm SESSION_ID SCORING_ID --revision CURRENT_REVISION
python scripts/data_intake.py scoring-list SESSION_ID
python scripts/data_intake.py eval-compare SESSION_ID RUN_ID --revision CURRENT_REVISION --scoring-id SCORING_ID
python scripts/data_intake.py acceptance-prepare SESSION_ID RUN_ID --revision CURRENT_REVISION \
  --scoring-id SCORING_ID --business-standard '业务方确认的最终验收标准' \
  --minimum-score 0.9 --minimum-cases 100
```

最后一条中的通过率和题数仅为参数格式示例，必须替换成真实业务要求。CLI 的自定义验收指标为 `pass_rate`，单题分数门槛来自已确认规则；未确认规则或其他任务的规则会被拒绝。规则默认位于 `outputs/workbench/business-scoring`，可用全局 `--scoring-root` 覆盖。远程 Agent 需要明确允许发送所需开发样例；本地服务不需要远程授权。需要澄清时 `scoring-propose` 返回 `needs_business_input` 和问题，不返回可确认的规则 ID。

上述 scoring 子命令在 stdout 输出 JSON 的同时，会向 stderr 追加人话摘要（`scoring-list` 只追加一行清单尾行：`summarize_listing` 单一来源——空清单点名下一步入口，非空给计数）：草稿复述这套规则要判断的业务标准、单题通过分数（达到才计为通过，均分与通过率分开展示）、业务正例与同题反例条数及真实隔离验证状态，并写明草稿待你核对实际正反例分数与理由后确认、软件不会自动确认评分规则；`scoring-confirm` 的摘要从记录本身读取，已确认规则写明绑定当前业务目标与输入/答案语义——数据修订后兼容规则可继续用，业务目标或含义变更需重新确认；`scoring-propose` 返回 `needs_business_input` 时，摘要点名澄清原因与待补业务问题原文，并写明当前没有可确认的评分方案。已成形规则的摘要都以「业务评分均值与通过率是规则口径的描述，不等于严格准确率，也不构成业务达标的判断」收尾。评分摘要现以评测解读同一格式渲染工具核查轨迹（`summarize_tool_trace` 单一来源）：「工具核查轨迹：N 次调用，成功 X 次、失败 Y 次——失败的调用没有取到证据，评分只依赖成功的调用」该行由摘要函数自动带上，CLI 与页面同源同词汇。页面自定义业务评分区在每条规则的验证状态下方渲染同一份摘要（`summarize_scoring`），页面与 CLI 同源同词汇。

## 用独立测试题做单模型业务验收

开发集用于比较和改进，最终独立测试用于判断一个已选定的模型是否达到业务标准。成功模型下展开「按业务标准做最终验收」，填写标准说明、最低通过率和最低测试题数。题数按测试记录计，不代表独立业务组数或统计置信度。评分规则沿已确认任务：类别严格匹配、JSON 全部声明字段匹配，或开放任务人工逐题判断。界面不预设业务分数或题数门槛；这些应由实际交付要求决定。

点击「冻结此模型与业务验收标准」保存模型、固定测试题、评分与生成协议及业务门槛。核对冻结条款后，再点击「按冻结标准执行最终验收」进行单模型生成。结果区分达到标准、未达到标准、证据不足和待人工判断；即使少数题全部答对，未满足最低测试题数也不能认定可交付。生成失败和截断保留在分母中。

开放任务会逐题显示输入、参考答案与实际回答，用户选择通过或不通过并填写业务理由。生成失败、缺失或截断的回答不能人工标为通过。判断会保存到该次验收，累积完成后再按冻结标准形成结论。最终记录可从页面下载，也可用 CLI 查看。

执行验收会留下题目暴露记录。同套件不能通过更改阈值、评分或生成协议变成新的验收标准；已经揭示的测试题或同业务组也不能通过更换套件标识伪装成新盲测。服务会明确标记或阻断重复暴露，并要求真正独立的新资料。最终题及坏例只向用户展示，不提供 Agent 分析或优化入口，也不会混入默认开发集报告列表。`eval-analyze` 拒绝最终测试报告。

```sh
# 以下分数和题数仅演示参数格式，必须替换为自己的业务要求。
python scripts/data_intake.py acceptance-prepare SESSION_ID RUN_ID --revision CURRENT_REVISION \
  --business-standard '业务方确认的交付要求' --minimum-score 0.9 --minimum-cases 100
python scripts/data_intake.py acceptance-show ACCEPTANCE_ID
python scripts/data_intake.py acceptance-run SESSION_ID ACCEPTANCE_ID --revision CURRENT_REVISION
python scripts/data_intake.py acceptance-list SESSION_ID
# 开放任务：row-index 使用报告中零起始的 index，每次提交一题判断。
python scripts/data_intake.py acceptance-review SESSION_ID ACCEPTANCE_ID --revision CURRENT_REVISION \
  --row-index 0 --decision accepted --reason '说明为何符合已冻结业务标准'
```

`--minimum-score` 取 0 到 1，`--minimum-cases` 为正整数，两者都必须显式提供。`acceptance-prepare` 可用 `--max-new-tokens` 设置统一生成长度、`--keep-whitespace` 保留严格匹配中的首尾空白差异；这些会随条款冻结。全局 `--acceptance-root` 指定验收记录目录，默认为 `outputs/workbench/acceptance`。最终验收通过表示达到该次冻结标准，不会自动部署模型。

上述验收子命令在 stdout 输出 JSON 的同时，会向 stderr 追加人话摘要：先复述运行前冻结的标准（评分口径名与门槛、最低测试题数），再按记录当前状态给结论——尚未执行（条款已冻结、留出题执行后即被占用，生成失败也不能再伪装成首次盲测）、被阻断（测试题或同任务业务对象已经揭示）、执行失败、待逐题业务判断（生成失败、缺失或截断的回答已按未通过锁定，不能人工改标为通过），或最终结论三态（达到标准／未达到标准／证据不足，不能确认可交付）。最终结论附通过数与分母口径：失败与截断保留在全部题目分母中，按未通过计；可完整核查的有效输出不足时单独点名。训练资料与留出题的隔离未能核验时，摘要如实写明「单看通过率本会判为通过／未通过」，但数值不能作为独立业务验收的结论。自定义规则的验收附业务评分均值，仅作描述，结论按冻结的逐题通过门槛与整体通过率计算。每条摘要都以「达到标准也不会自动部署模型」收尾——结论只对这次冻结的条款与固定测试题负责。页面最终验收区在每条记录的结论下方渲染同一份摘要（`summarize_acceptance`）：条款刚冻结尚未执行、被阻断、待逐题判断或已出最终结论，页面用户与 CLI 用户读到同一口径的冻结标准、结论与边界收尾句，不再各说各话。

验收冻结这个用户决策点同样读这份投影。`acceptance-prepare` 冻结前先向 stderr 打印同一份规约摘要（`summarize_task_spec` 单一来源），冻结时把任务规约四要素（业务目标、答案语义、评分口径、时间约束）快照进验收记录（`task_spec` 键）；`acceptance-show` 与页面记录的摘要随后以 `spec_anchor_lines` 单一来源渲染「冻结时引用的任务规约口径」段——冻结时按什么标准判有处可查；页面在冻结表单旁提供「📋 任务规约（冻结验收条款前的口径）」折叠区渲染同一份摘要，页面与 CLI 同源同词汇。旧记录没有 `task_spec` 键时如实没有该段，不回填。

最终验收门槛的分辨率算术统一出自 `src/workbench/report_summary.py`（`acceptance_gate_lines` 单一来源）：把「最低通过率 T%」换算成「N 道最终测试题需通过多少道、最多容错多少道未通过」，并给出每题占通过率多少个百分点（如「按 90% 通过率门槛与 10 道最终测试题算：需通过 9 道、最多容错 1 道未通过」）；容错为 0 道时单列警示（任何一题未通过都判未达标），最低测试题数超过固定测试题实际题数时预告执行必判「证据不足」。算术与验收执行的判定用同一式（k/N 浮点商与门槛比较），页面冻结表单与验收摘要（CLI acceptance 子命令 stderr）同源同词汇；门槛数值本身仍由用户设定，软件不代设，也不作统计结论。

## 从评测结果进入下一轮改进

评测报告下展开「将结果转成下一轮改进假设」，可从 Agent 解读带入待核查原因与建议，再填写希望看到的业务变化、具体变更范围，以及是否会修改数据。训练配置默认继承父轮，按需只调整轮数、学习率或长度。保存提案后，在同页的改进记录中点击「确认本轮假设与变更范围」；保存提案本身不会启动训练。

提案会冻结父轮开发题和独立测试题，并保留父轮模型及证据。后续都在同一个数据任务中修订：需要改数据时，通过原资料入口补充或修订文件，由 Agent 重新分析，重新确认转换及全量结果；不需要手工编写 JSON 或另建任务。确认的数据改进轮次提供「让 Agent 按确认方向修改数据方案」：它读取父轮坏例与确认范围，调用已有数据工具执行组合、受限适配或转换预览，保存前后方案及改动摘要。生成的新预览仍需业务确认，不自动编造监督标签。远程服务需同时确认发送当前业务资料及父轮实际评测输出与坏例。修订记录的工具核查轨迹现以评测解读同一格式渲染（`summarize_tool_trace` 单一来源）：页面修订记录的「修改前后方案与工具记录」折叠区直接渲染「工具核查轨迹：N 次调用，成功 X 次、失败 Y 次——失败的调用没有取到证据，修订只依赖成功的调用」，原始 JSON 留在下方。原开发和测试题的输入、答案不能改写、缺失或进入训练集；新增独立业务对象进入训练，同组资料仍留在原保留分区。

已有数据版本也可直接点击「用本轮固定题集准备数据版本」，为本轮绑定题集。没有分组字段时，仍需确认每行属于独立业务对象。数据绑定完成后点击「按已确认范围准备下一轮训练」，通过预检后在下方对应记录启动。每轮从同一基础模型重新微调，父轮 adapter 保留作对照，不作为续训起点。

本轮训练成功后的比较会自动选择基座、已确认父轮、本轮三个模型，使用固定开发题、相同实际输入与评分/生成协议。每个历史 adapter 使用它自己的基础模型路径与训练证据。页面逐条展示完整输出、期望、失败与截断，并把对照关联到当前改进假设。然后填写业务理由，记录「采用本轮结果」「继续改进」「停止本轮路线」或「证据不足」。采用记录不会自动部署模型；固定题数太少、输出截断或开放任务尚无评分时，应明确保留证据局限。

CLI 提供相同的确认流程；每次改变资料或物化后使用返回的最新 `revision`：

```sh
python scripts/data_intake.py iteration-propose SESSION_ID --revision CURRENT_REVISION \
  --parent-run-id PARENT_RUN_ID --evaluation-id PARENT_EVALUATION_ID \
  --hypothesis '已有监督覆盖不足可能影响类别判断' \
  --expected-outcome '固定开发题上的严格错误减少' \
  --change '补充经过审核的独立训练样例，保持原开发和测试题不变' --data-change
python scripts/data_intake.py iteration-confirm SESSION_ID ITERATION_ID --revision CURRENT_REVISION
# 按需先经 add-source 补充资料，再让 Agent 按确认方向修订处理方案。
python scripts/data_intake.py iteration-revise SESSION_ID ITERATION_ID --revision CURRENT_REVISION --allow-remote-data
# 核对新预览并沿 confirm / full-sources / full-confirm 确认实际数据。
python scripts/data_intake.py materialize SESSION_ID --revision CURRENT_REVISION --iteration-id ITERATION_ID
python scripts/data_intake.py iteration-prepare SESSION_ID ITERATION_ID --revision CURRENT_REVISION
python scripts/data_intake.py iteration-start SESSION_ID ITERATION_ID --revision CURRENT_REVISION
python scripts/data_intake.py train-status NEW_RUN_ID
python scripts/data_intake.py eval-compare SESSION_ID NEW_RUN_ID --revision CURRENT_REVISION --iteration-id ITERATION_ID
python scripts/data_intake.py iteration-decide ITERATION_ID --decision insufficient_evidence --reason '题数仍少，暂不据此采用'
python scripts/data_intake.py iteration-list SESSION_ID
```

`iteration-propose` 可重复 `--change`，可选 `--epochs`、`--learning-rate`、`--max-length`；不改数据则省略 `--data-change`。`iteration-start` 有预检风险且已核对时加 `--acknowledge-warnings`。`eval-compare --iteration-id` 会自动选择父轮，并绑定三模型结果；已有合格报告也可用 `iteration-bind SESSION_ID ITERATION_ID --revision CURRENT_REVISION --evaluation-id EVALUATION_ID` 关联。全局 `--iteration-root` 可覆盖改进记录目录。

上述轮次子命令（iteration-propose/confirm/prepare/start/bind/revise/decide）与自动执行子命令（iteration-execute/execution-status/execution-stop）在 stdout 输出 JSON 的同时，都会向 stderr 追加人话摘要（iteration-list 只追加一行清单尾行：`summarize_listing` 单一来源——空清单点名下一步入口，非空给计数）。轮次摘要先复述这轮改进的假设，再说明当前停在哪一步：提案已保存尚未确认、已确认尚未准备训练、正在准备、方案已准备尚未启动、训练已启动、三模型同题对照已完成正等待业务决定；已记录决定时回显决定名（「采用本轮结果」「继续改进」「停止本轮路线」「证据不足」）与业务理由，采用补「采用记录不会自动部署模型」、证据不足补证据局限提示，被阻断时如实点名原因。自动执行摘要翻译执行状态机：进行中各态说明后台推进到哪一步，并写明「关闭页面不影响执行」；暂停等待确认时点名需要你在原入口勾选确认继续才会恢复，不会跳过提示自动训练，并附可照抄的命令行恢复命令（`iteration-execute SESSION_ID ITERATION_ID --revision N --acknowledge-warnings`——「勾选」是页面词汇，命令行没有可勾的框）；completed 点名训练与评测记录、轮次停在待业务决定，重复提交不会再次训练、只返回原报告；被阻断或失败时逐条列出问题并指向 worker.log（执行记录在启动时落盘日志路径，摘要据此给出 worker.log 的完整位置；旧记录缺该路径时回退泛指句）；已停止说明本轮不再推进。执行记录的 `message` 字段只报事实（R100 起六处收敛：授权失效、训练准备未通过、启动前数据校验未通过、等确认暂停、对照未完整完成、对照已完成均为纯事实短句），行动建议与恢复指引只在摘要单处输出，不与 message 同屏堆叠；历史记录自带的长句 message 由摘要原样嵌入，不重写。每条摘要都以流程状态不代表业务效果达标收尾——以上只是流程状态与已记录的决定，不代表业务效果达标。页面改进轮次区渲染同一对摘要函数（`summarize_iteration`／`summarize_execution`）：待决策轮次在三模型结果下方读到「对照已完成，正等待你的业务决定」与流程边界收尾，已决策轮次回显决定名与业务理由；后台执行进度在刷新按钮上方实时翻译当前状态，训练中写明「关闭页面不影响执行」、暂停等确认写明「在原入口勾选确认继续才会恢复」。页面与 CLI 同源同词汇。

三模型对照的分数差在这份轮次摘要里换算成题数差，统一出自 `src/workbench/report_summary.py`（`three_model_delta_lines` 单一来源）：轮次处于三模型同题对照完成、等待业务决定的状态时，摘要紧随状态行输出基座对照与父轮对照的题数差（如「父轮对照：本轮微调比父轮多答对 2 题（7/8 vs 5/8）」；业务评分口径下「答对」写作「通过」），页面改进轮次决策卡与 CLI 轮次子命令的 stderr 输出同一份行——页面与 CLI 同源同词汇。末行给出单题分辨率：「开发集共 N 道题，每题约占 X 个百分点」，并写明差距在 1 题量级时方向参考价值有限，是否值得再投一轮还要结合失败形态（截断/生成失败/复述指令）与逐题核对判断。这些行只做确定性题数算术（分数差换算成题数差、每题百分点分辨率），不作统计结论；对照缺基座或父轮结果、两侧题数不一致时对应行不出现，不编造算术。

若只需先固定题集，可点击「固定当前开发与测试题集」，或运行 `suite-freeze SESSION_ID --revision CURRENT_REVISION`；这一步不修改父轮数据版本。`suite-show SUITE_ID` 查看来源与题数，后续 `materialize ... --suite-id SUITE_ID` 显式复用。全局 `--suite-root` 控制独立题集目录；改进提案自带题集引用，使用 `--iteration-id` 时无需手工查找目录。固定题集时，新独立行进入训练，原题保持不变，分区比例及随机种子不再重新分配原题。

`suite-freeze` 与 `suite-show` 在 stdout 输出 JSON 的同时向 stderr 追加人话摘要（`summarize_suite`，冻结返回的引用与展示读出的完整清单两种形态同口径）：题数（开发题与最终测试题分列）、题目内容按摘要锁定且原评分题不能修改、同对象新增行不自动扩充评分题、materialize 用 `--suite-id` 复用，并以「固定题集只保证各轮比较基线一致，不代表业务效果达标」收尾；`suite-show` 的完整清单另附锚定数据版本名与版本号。页面「分区设置」选择固定题集后渲染同一份摘要，页面与 CLI 同源同词汇。

## 配置文件与优先级

页面与 CLI 默认共享项目目录下的 `outputs/workbench/agent-settings.json`，文件仅包含 `provider`、`base_url`、`model`。CLI 可用全局参数覆盖文件路径，须放在子命令前：

```sh
python scripts/data_intake.py --agent-config-path /tmp/tunesmith-agent.json \
  agent-config --provider local --model my-tool-model
```

读取顺序为：CLI 临时参数（如果提供）→ `TUNESMITH_AGENT_PROVIDER` / `TUNESMITH_AGENT_BASE_URL` / `TUNESMITH_AGENT_MODEL` 环境变量 → 保存的公共配置 → 本地默认配置。环境选择新供应商时，未明确设置的地址和模型取该供应商预设，不沿用旧供应商的地址。页面初次加载同样采用环境优先，随后可以在当前会话修改。

`agent-config` 保存时只依据已有文件和显式参数，不会把环境覆盖值意外写入文件。切换供应商取新预设；自定义服务需明确填写地址。环境覆盖不会因保存文件自动消失，需要清除对应环境变量才能使用文件值。

环境密钥由启动者配给当前有效供应商和地址；保存或切换公共配置后也应核对配套密钥。当前实现没有密钥库、额度管理或计费系统。

## 长单元格的证据读取

普通画像、行检查和预览会将单个长文本展示为前 2000 个字符，原资料保持完整。Agent 可用 `read_cell_content` 核实财报尾注、合同末段等后续内容：`source_kind="current"` 读取当前组合或适配结果，`source_kind="source"` 加 `alias` 读取已上传的具名原始文件，`source_kind="full"` 读取已有全量报告的来源。三种选择均使用各自来源的短 `row_id` 与实际 `column`，不会将别名解释为磁盘路径或网址。

`offset` 默认 0，`limit` 默认 4096、最多 16384 个字符。返回内容包括来源指纹、完整行证据引用、字段类型、总字符数、当前片段及 `next_offset` / `has_more`；结构化值以确定性 JSON 编码分页。该工具的片段不会再次被截成 2000 字符。Agent 可按需跳到指定位置，但只能把实际读过的部分称为观察证据；全量报告若已过期，返回结果也会标明 `stale`。工具读取遵循本次数据分析已有的供应商和资料发送授权，不改变源文件，不要求为提出缺资料问题读完所有长字段。

## 盲标核验的样本量与统计口径

盲标核验的样本量由你按业务风险决定：页面抽取表单可在 1–50 条之间自选（默认 5 条），CLI `label-verify --size` 同口径，超出范围的请求在抽题前就被拒绝。样本量是证据强度的主要决定因素：同样全部一致，5 条的 95% Wilson 置信下界约 57%，20 条约 84%，30 条约 89%——观测一致率在小样本下会高估真实水平，高风险业务建议加大样本量后再下结论。

系统对结论如实标注统计口径，不冒充确定性：

- 抽题结果自带 `evidence_note`：本轮 N 条即使全部一致，95% 置信下真实一致率下界也只约多少——抽题前就预告证据上限。
- 提交结论与存档带 `agreement_lower_bound` 和 `evidence_note`，通过与未通过两种判定都展示；CLI `label-verify-submit` 输出的 JSON 同样携带这两个字段。样本 <30 条时文案明说「下界才是你能依赖的数」；≥30 条改说「下界接近观测值」。
- 已标注行少于所选样本量时，结果带 `shortfall_note` 如实说明按现有全部行抽取；统计说明按实际抽样条数计算，不按请求值夸大证据。
- 早期版本的存档记录没有这两个统计字段时，页面回读按同一口径现算统计说明，不会静默丢掉局限提示。
