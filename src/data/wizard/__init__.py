"""数据准备向导（TuneSmith guided data prep）。

原始表格（CSV/Excel/JSONL）→ 列映射 → 垂类模板生成 → 数据体检 →
泄漏安全切分 → Alpaca 格式导出。面向非算法工程师：专家决策已固化为默认值，
检查结果用可执行的语言解释。

典型用法::

    from src.data.wizard import WizardPipeline, WizardSpec, import_table, suggest_mapping

    table = import_table("raw.xlsx")
    spec = WizardSpec(mapping=suggest_mapping(table.columns))
    report = WizardPipeline(spec).run(table, out_dir="outputs/wizard/run1")
"""

from src.data.wizard.checks import CheckResult, run_checks
from src.data.wizard.exporter import ExportReport, SplitResult, export_splits, split_by_entity
from src.data.wizard.importers import RawTable, import_table, suggest_mapping
from src.data.wizard.pipeline import WizardPipeline, WizardReport
from src.data.wizard.spec import FieldMapping, WizardError, WizardSpec
from src.data.wizard.templates import (
    Candidate,
    DomainTemplate,
    MatchingSample,
    available_templates,
    get_template,
    register_template,
)

__all__ = [
    # spec
    "FieldMapping",
    "WizardError",
    "WizardSpec",
    # importers
    "RawTable",
    "import_table",
    "suggest_mapping",
    # templates
    "Candidate",
    "DomainTemplate",
    "MatchingSample",
    "register_template",
    "get_template",
    "available_templates",
    # checks
    "CheckResult",
    "run_checks",
    # exporter
    "SplitResult",
    "ExportReport",
    "split_by_entity",
    "export_splits",
    # pipeline
    "WizardPipeline",
    "WizardReport",
]
