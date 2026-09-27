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

`--base-url` 和 `--model` 可临时覆盖分析或连接检查的配置。若已有环境密钥，临时地址与解析后的地址不同，CLI 会拒绝请求。要改用另一个服务，应同时配置该服务的 `TUNESMITH_AGENT_BASE_URL`、`TUNESMITH_AGENT_MODEL` 和配套密钥，再重试，避免旧密钥被带到新服务。

## 从样例继续到全量数据

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

### 连续数值答案的如实边界

答案列是带小数的连续测量值（如 1.0/2.5/3.7）时，基础分析会把该目标的答案形态如实标注为 `numeric_continuous`，并同时给出边界说明：当前训练仍按逐字字符串学习答案（「1.0」与「1.00」算两个不同答案），不是数值回归；评测只能逐字比对，无法按数值误差评分。需要数值误差口径时，先在数据侧把答案列离散化。小数若只是离散编码（如版本号），照常逐字核对即可；整数编码（0/1/2）不受此判定影响，仍按类别处理。全量出现样例未覆盖的新测量值是连续目标的常态：报告以 `numeric_new_values` 如实列出这些值并重申逐字学习边界，不按「新类别」表述、不阻断。评测、验收与页面都不会把 `numeric_continuous` 目标伪造成分类严格评分——它落入与开放任务相同的人工核对口径。

## 语义安全层：确认不再盲点头

北极星的核心痛点是语义错误无声通过。三道通用关卡（与任务类型无关、零或低成本、非专家可独立完成）已接入流程：

- **对比核验（样例确认时）**：系统抽取两条答案不同的输入，把两个答案打乱后由你配对。配对正确才说明真正看清了转换含义；配错会留档并提示重新查看预览。预览或方案变化后核验自动失效。页面在「确认当前转换含义」前提供入口；CLI 用 `contrast-check` / `contrast-check-submit`（完整用法见下文）。
- **盲标核验（训练准备的硬门禁）**：全量确认后、准备训练前，系统按你选择的样本量（1–50 条，默认 5 条）随机抽取已标注行并隐藏答案，你仅根据输入作答，与数据标签全部一致才允许准备训练。核验与全量来源和处理方案摘要绑定：数据或方案修订后自动失效，需重新核验。P3.4 自动执行的授权入口同样前置校验。完整 CLI 用法见下。
- **可学性探针（可选证据，非门禁）**：用基座模型对开发集抽样做零样本探测，与「瞎猜多数类」基线对比并给出带样本量限制的说明。显著低于基线通常意味着提示格式或任务定义需要先核查；高于基线也不能预测微调效果。CLI：`learnability-probe SESSION --revision R --model-path 目录 [--size 8]`；页面在「训练前检查」区提供入口（会实际加载本地模型）。

这三道关卡都不使用 LLM 判断、不消耗模型服务额度，判定全部确定性可复现。

对比核验的 CLI 用法与 `confirm` 的如实提示：`contrast-check SESSION_ID --revision CURRENT_REVISION` 抽出两条答案不同的输入与打乱后的两个候选答案——stderr 逐条列出「[行ID] 题目输入」与候选清单，stdout 为纯 JSON（`check_id`、`items`、`options` 与可照抄的 `submit_hint`）；`contrast-check-submit SESSION_ID --check-id CHECK_ID --answer 行ID=候选答案 --answer 行ID=候选答案` 提交配对，需恰好覆盖两条输入、答案来自候选。判定行如实亮出配对计数与连胜口径：verified 但连胜只有一轮时明确提示「还需再连续配对正确一轮（二连对）才算真正看清」，二连对及以上提示达标——防瞎蒙靠的是连胜不是单轮；配错判定行报 mismatch 配对计数并提示此前的确认可能是盲点头。`confirm` 在 stderr 如实报告当前配对状态：尚未核验（建议先运行 `contrast-check`）、已连续 N 轮配对正确（达标或还差一轮）、或最近一次配对错误——对比核验是软门禁，配错留档但不阻断确认，去留由你决定。核验与预览及方案绑定：提交不收 `--revision`，预览或方案变化后原核验自动失效。

### 盲标核验的完整 CLI 用法

```sh
python scripts/data_intake.py label-verify SESSION_ID --revision CURRENT_REVISION --size 5
# --size 在 1–50 之间自选（默认 5）；加 --export-csv 路径 可把题目清单导出为 CSV 供线下作答。
python scripts/data_intake.py label-verify-submit SESSION_ID --verification-id VERIFICATION_ID \
  --answer 行ID=你的答案 --answer 行ID=你的答案
```

`label-verify` 抽题后依次输出：提示「请仅根据输入作答，不要查看数据中的现有答案。」；`evidence_note` 证据预告——本轮 N 条即使全部一致，95% 置信下真实一致率下界也只约多少（例如 5 条约 57%），小样本下下界才是你能依赖的数；已标注行不足所选条数时再输出 `shortfall_note`，说明按现有全部已标注行抽取、统计按实际条数口径；随后逐题列出「[行ID] 题目输入」。标准输出为 JSON，含 `verification_id`、`sample_size`、`row_ids` 与可照抄的 `submit_hint`。`--export-csv` 导出的清单带 BOM、Excel 直开，只含「行ID、题目输入、留空待填的盲标答案」三列，不含数据答案——导出文件泄露答案，盲标核验就失效；线下作答后逐条照抄回 `label-verify-submit` 提交。

`label-verify-submit` 不收 `--revision`（核验结论由全量来源与处理方案摘要绑定，数据或方案变化后原核验自动失效）；每条抽样行一个 `--answer`，行ID=你的答案，需恰好覆盖全部抽样行。人读判定行形如「判定：verified（5/5 一致，95% 置信下界约 57%）」，通过与未通过两种判定同口径亮出下界数字，观测一致率不冒充真实水平；通过时另附「盲标核验通过：监督信号的业务含义经独立复现。」，未通过时另附「存在不一致，训练不会开始；请核对数据标签或业务定义后重新核验。」JSON 结果同样携带 `agreement_lower_bound` 与 `evidence_note`。

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

报告保存为 `training_preflight`，状态包括 `blocked`（存在阻断问题）、`warnings`（需核对风险）、`passed`（当前数据与 token 消费检查通过）。报告列出各分区的截断和答案丢失统计、问题行以及实际 token 消费；答案完全丢失会阻断。tokenizer 无法提供答案边界时明确标记未验证。检查依据当前训练模板、因果位移和 padding 屏蔽方式，不据此声称业务效果已验收或硬件足以训练。

## 让 Agent 推荐训练方案

确认数据并生成分区后，「用当前数据微调模型」会发现本机已准备的模型，列出文件完整的候选供点选。只选择本轮需要比较的模型，避免核查不相关的大型权重；其他位置的已有模型可在「高级：补充本地模型路径与发现详情」中填写，每行一个。发现只检查已知缓存和模型目录的本地文件，不下载、加载或计算权重哈希；文件完整也不等于训练兼容，Agent 后续还会检查候选事实和真实 tokenizer。点击「让 Agent 推荐训练方案」后，Agent 读取业务目标、处理方案、实际数据统计、本机条件及候选模型事实，并对选择的模型执行真实 tokenizer 预检。它不会自动下载模型、加载训练权重或启动训练。

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

省略 `--model-path` 时自动采用发现的完整本地候选；没有完整模型时会提示缺少文件，不会自动下载。`model-list --root LOCAL_DIRECTORY` 可只查看指定目录，`--root` 可重复。`--model-path` 可以重复；本地 Agent 服务不需要 `--allow-remote-data`。方案默认保存在 `outputs/workbench/training-plans`，全局 `--plan-root` 可覆盖目录。`plan-prepare` 表示已经审阅并确认保存的推荐方案，只准备训练，不自动启动。

上述 plan 子命令在 stdout 输出 JSON 的同时向 stderr 追加人话（`plan-list` 只列清单，不追加）：`plan-recommend` 与 `plan-show` 先翻译方案本身——建议的基础模型与关键参数（最大长度、训练轮数、batch size、学习率、LoRA rank、量化位数）、状态三态（方案可供确认／需要先完善数据／当前条件不支持）、推荐理由与尚未验证的限制原文、需要你先回答的业务问题；ready 方案再追加其携带的真实预检证据的人话翻译（预检记录在方案记录的 `probe` 里）。`plan-prepare` 的结果复用训练记录摘要：方案已准备好并通过检查，还没有开始训练。确认这份方案只会准备训练、不会自动启动，方案就绪与推荐理由也不构成训练效果或业务达标的判断。

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
```

`RUN_ID` 来自准备结果；有预检风险且已完成核对时，启动命令增加 `--acknowledge-warnings`。`train-prepare` 另支持 `--learning-rate`、`--gradient-accumulation`、`--lora-rank` 和 `--load-in-4bit`。这里需要已准备好的本地基础模型目录，工作台不会悄悄更换底座或下载另一个模型。

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

页面「大白话解读」与 CLI 评测输出对自定义对照同样用这一口径翻译：逐模型句报「业务评分均值 X、通过 Y/N」，不把通过数写成「答对」，也不触发「零分/答对最多」这类只对严格准确率成立的结论句；想定位扣分点，逐题查看评分理由。开放任务的对照在人话摘要里只报生成事实（生成了 N 条回答与失败、截断计数），明确说明开放任务不做自动评分、每题通过与否由用户逐题判断——生成失败、缺失或截断的回答不能人工标为通过。

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

上述 scoring 子命令在 stdout 输出 JSON 的同时，会向 stderr 追加人话摘要（`scoring-list` 只列清单，不追加）：草稿复述这套规则要判断的业务标准、单题通过分数（达到才计为通过，均分与通过率分开展示）、业务正例与同题反例条数及真实隔离验证状态，并写明草稿待你核对实际正反例分数与理由后确认、软件不会自动确认评分规则；`scoring-confirm` 的摘要从记录本身读取，已确认规则写明绑定当前业务目标与输入/答案语义——数据修订后兼容规则可继续用，业务目标或含义变更需重新确认；`scoring-propose` 返回 `needs_business_input` 时，摘要点名澄清原因与待补业务问题原文，并写明当前没有可确认的评分方案。已成形规则的摘要都以「业务评分均值与通过率是规则口径的描述，不等于严格准确率，也不构成业务达标的判断」收尾。页面自定义业务评分区在每条规则的验证状态下方渲染同一份摘要（`summarize_scoring`），页面与 CLI 同源同词汇。

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

## 从评测结果进入下一轮改进

评测报告下展开「将结果转成下一轮改进假设」，可从 Agent 解读带入待核查原因与建议，再填写希望看到的业务变化、具体变更范围，以及是否会修改数据。训练配置默认继承父轮，按需只调整轮数、学习率或长度。保存提案后，在同页的改进记录中点击「确认本轮假设与变更范围」；保存提案本身不会启动训练。

提案会冻结父轮开发题和独立测试题，并保留父轮模型及证据。后续都在同一个数据任务中修订：需要改数据时，通过原资料入口补充或修订文件，由 Agent 重新分析，重新确认转换及全量结果；不需要手工编写 JSON 或另建任务。确认的数据改进轮次提供「让 Agent 按确认方向修改数据方案」：它读取父轮坏例与确认范围，调用已有数据工具执行组合、受限适配或转换预览，保存前后方案及改动摘要。生成的新预览仍需业务确认，不自动编造监督标签。远程服务需同时确认发送当前业务资料及父轮实际评测输出与坏例。原开发和测试题的输入、答案不能改写、缺失或进入训练集；新增独立业务对象进入训练，同组资料仍留在原保留分区。

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

上述轮次子命令（iteration-propose/confirm/prepare/start/bind/revise/decide）与自动执行子命令（iteration-execute/execution-status/execution-stop）在 stdout 输出 JSON 的同时，都会向 stderr 追加人话摘要（iteration-list 只列清单，不追加）。轮次摘要先复述这轮改进的假设，再说明当前停在哪一步：提案已保存尚未确认、已确认尚未准备训练、正在准备、方案已准备尚未启动、训练已启动、三模型同题对照已完成正等待业务决定；已记录决定时回显决定名（「采用本轮结果」「继续改进」「停止本轮路线」「证据不足」）与业务理由，采用补「采用记录不会自动部署模型」、证据不足补证据局限提示，被阻断时如实点名原因。自动执行摘要翻译执行状态机：进行中各态说明后台推进到哪一步，并写明「关闭页面不影响执行」；暂停等待确认时点名需要你在原入口勾选确认继续才会恢复，不会跳过提示自动训练；completed 点名训练与评测记录、轮次停在待业务决定，重复提交不会再次训练、只返回原报告；被阻断或失败时逐条列出问题并指向 worker.log；已停止说明本轮不再推进。每条摘要都以流程状态不代表业务效果达标收尾——以上只是流程状态与已记录的决定，不代表业务效果达标。页面改进轮次区渲染同一对摘要函数（`summarize_iteration`／`summarize_execution`）：待决策轮次在三模型结果下方读到「对照已完成，正等待你的业务决定」与流程边界收尾，已决策轮次回显决定名与业务理由；后台执行进度在刷新按钮上方实时翻译当前状态，训练中写明「关闭页面不影响执行」、暂停等确认写明「在原入口勾选确认继续才会恢复」。页面与 CLI 同源同词汇。

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
