# 长尾数据解析适配

当字段中包含现有转换组件不能表达的业务编码或嵌套文本时，Agent 可以起草解析代码，先用真实业务样例和反例验证，再展示实际转换结果。用户核对的是新增字段是否符合业务含义，以及它如何参与模型输入和答案设计。

当前支持一对一新增字段：例如把 `code:0012` 解析为新列 `number="0012"`，保留前导零、原文、其他字段和来源标识。适配器不能静默删行、复制行、覆盖原字段或替换来源 ID。关联、展开、会话组织使用组合管道，不借这条接口绕过来源约束。

## 真实执行边界

适配代码只在已探测成功的操作系统或容器隔离环境中执行。没有可用后端时返回 `unavailable`，不会在宿主 Python 中直接执行，也不会把语法检查当作隔离验证成功。

- Docker：只使用本地已有镜像，不自动拉取；网络关闭、根文件系统只读、非 root 用户、清除 capabilities、禁止权限提升，并限制进程数、CPU、内存和临时空间。只挂载本次暂存的代码与输入，不挂载项目、凭据目录或 Docker socket。
- macOS：使用 `sandbox-exec` 的默认拒绝规则，仅允许解释器、必要标准库、系统库与本次临时输入读取；只允许写入受控输出文件。网络、其他用户文件内容和其他程序执行均不放行。解释器的父目录仅按需允许读取路径元数据。
- 请求环境不包含 Agent 密钥。代码接口只允许 `re`、`json`、`math` 导入；这项静态限制只是接口约束，真正的文件和网络保护依赖操作系统隔离。

默认单次执行限制为 5 秒墙钟时间、3 秒 CPU、1 MiB 输入和 2 MiB 输出。macOS 无法可靠设置 Python 的 DATA/AS 硬内存上限，因此采用宿主进程采样 RSS 并终止越界进程；验证报告明确记录 `memory_enforcement: sampled_rss`，不把它描述为容器级硬内存限制。Docker 使用容器内存边界。以上是隔离保护，与 Agent 额度管理无关。

## 验收与来源

代码接口为 `transform(rows, config) -> list[dict]`。每条输入携带保留字段 `__row_id`，返回的来源 ID 集合必须与输入完全一致。Agent 声明新增列，实际输出必须保留原始列并只增加这些列。

适配方案至少包含一个引用真实输入及行 ID 的业务期望样例，以及一个反例。正例比较实际 JSON 输出；反例可以要求拒绝无效格式，也可以声明边界输入的正确结果。只有逐例通过后才处理真实资料；真实处理结果还会再次检查行集合、字段和值的保留情况。

验证结果记录源码 SHA-256、样例集 SHA-256、后端、执行边界和逐例结果。适配方案与输出来源另有内容摘要；修改源码或配置会重新执行验证，不复用旧预览。样例和全量分别执行，同样保留原始文件。生成的数据版本保留原文件摘要及原始行 ID，可从训练记录追溯来源。

## 开发接口

```python
from src.workbench.sandbox import TransformCase, TransformSandbox

runner = TransformSandbox(backend="auto")
result = runner.run(source_code, rows, config)
# result.status: passed / failed / unavailable
# result.rows 仅在成功且输出结构合格时可用

report = runner.validate(source_code, [
    TransformCase("业务样例", rows, expected_rows),
    TransformCase("无效格式", invalid_rows,
                  kind="counterexample", expect_error=True),
])
```

产品层使用 `src.workbench.adapters.apply_adapter`：在上述隔离结果之外，强制真实样例引用、一对一来源、原字段不变与新增字段声明。其结果随后进入 `DataRecipe` 预览、全量诊断和数据分区。

## 当前验证证据

macOS 真实隔离回归覆盖了正则新增字段、外部临时哨兵文件读写被拒、绕过 Python 导入约束后网络仍被系统拒绝、密钥环境不继承、超时中止、输出约束及内存监督。集成测试覆盖样例分析到全量确认、固定数据分区和原始来源追溯，也覆盖丢行、修改原字段及源码变更使旧验证失效。

受限制的宿主环境可能禁止建立嵌套 macOS 沙箱；这时真实隔离测试明确跳过，产品能力返回不可用，不能把普通测试通过当作隔离成功。此机器的 macOS 后端已在允许建立系统沙箱的环境完成实际验证；Docker 后端实现尚未在本机完成容器运行验证。

```sh
python -m pytest tests/unit/test_intake_sandbox.py tests/unit/test_intake_adapters.py -o addopts='' -q
```
