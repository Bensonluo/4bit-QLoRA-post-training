"""向导流水线：导入 → 建样本 → 去重 → 切分 → 体检 → 导出。

编排层只做流程控制与报告汇总；领域逻辑在模板，质量规则在 checks，
切分/写盘在 exporter。CLI（scripts/data_wizard.py）与后续 Streamlit
页面（ui/pages/05_Data_Wizard.py）都只调 WizardPipeline。
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

from src.data.wizard.checks import CheckResult, run_checks
from src.data.wizard.exporter import (
    ExportReport,
    dedup_samples,
    export_splits,
    split_by_entity,
)
from src.data.wizard.importers import RawTable
from src.data.wizard.spec import SPLITS, WizardError, WizardSpec
from src.data.wizard.templates import (
    BuildResult,
    DomainTemplate,
    RowIssue,
    get_template,
)


@dataclass
class WizardReport:
    """一次向导运行的完整结果（CLI 打印 / UI 展示 / report.json 三用）。

    随流水线各阶段逐步填充，因此是可变 dataclass。
    """

    template: str
    source: str
    total_rows: int
    built_samples: int
    dedup_removed: int
    dropped_rows: list[RowIssue] = field(default_factory=list)
    split_counts: dict[str, int] = field(default_factory=dict)
    checks: list[CheckResult] = field(default_factory=list)
    export: ExportReport | None = None

    @property
    def blocking_errors(self) -> list[CheckResult]:
        return [c for c in self.checks if not c.passed and c.severity == "error"]

    @property
    def passed(self) -> bool:
        return not self.blocking_errors

    def summary_lines(self) -> list[str]:
        """人类可读的运行摘要（Rich console 打印用）。"""
        lines = [
            f"模板: {self.template} | 来源: {self.source} | 原始行: {self.total_rows}",
            f"生成样本: {self.built_samples}（去重去除 {self.dedup_removed}，跳过行 {len(self.dropped_rows)}）",
        ]
        if self.split_counts:
            lines.append("切分: " + " | ".join(f"{k}={v}" for k, v in self.split_counts.items()))
        for c in self.checks:
            icon = "✓" if c.passed else ("✗" if c.severity == "error" else "⚠")
            lines.append(f"{icon} [{c.severity}] {c.check_id}: {c.message}")
        if self.export is not None:
            lines.append(f"已导出: {self.export.out_dir}")
        else:
            lines.append("未导出：存在阻断错误，请按上方 ✗ 项修复后重跑。")
        return lines

    def to_dict(self) -> dict[str, object]:
        return {
            "template": self.template,
            "source": self.source,
            "total_rows": self.total_rows,
            "built_samples": self.built_samples,
            "dedup_removed": self.dedup_removed,
            "dropped_rows": [{"row": d.row, "reason": d.reason} for d in self.dropped_rows],
            "split_counts": self.split_counts,
            "checks": [
                {
                    "check_id": c.check_id,
                    "severity": c.severity,
                    "passed": c.passed,
                    "message": c.message,
                }
                for c in self.checks
            ],
            "export": self.export.to_dict() if self.export else None,
        }


class WizardPipeline:
    """向导主流程。用法：

    table = import_table("raw.xlsx")
    spec = WizardSpec(mapping=suggest_mapping(table.columns))
    report = WizardPipeline(spec).run(table, out_dir="outputs/wizard/run1")
    for line in report.summary_lines():
        console.print(line)
    """

    def __init__(self, spec: WizardSpec) -> None:
        self.spec = spec
        self.template: DomainTemplate = get_template(spec.template)

    def run(self, table: RawTable, out_dir: str | Path) -> WizardReport:
        built: BuildResult = self.template.build_samples(table, self.spec.mapping, self.spec)

        samples = built.samples
        if self.spec.dedup:
            samples, removed = dedup_samples(samples)
        else:
            removed = 0

        report = WizardReport(
            template=self.spec.template,
            source=table.source,
            total_rows=len(table.rows),
            built_samples=len(samples) + removed,
            dedup_removed=removed,
            dropped_rows=built.dropped,
        )
        if not samples:
            # 全部行被跳过：无法切分导出，报告落盘说明原因
            report.split_counts = dict.fromkeys(SPLITS, 0)
            report.checks = run_checks({s: [] for s in SPLITS}, built.dropped)
            report.checks.append(
                CheckResult(
                    "no_samples",
                    "error",
                    False,
                    "没有生成任何训练样本——所有原始行都被跳过。"
                    "请检查标准名列映射是否指向了正确的列，以及表中是否有有效数据行。",
                )
            )
            self._write_report(report, Path(out_dir))
            return report

        try:
            splits = split_by_entity(samples, self.spec)
        except WizardError as exc:
            # 实体过少等切分失败：无法导出，报告仍落盘说明原因
            report.split_counts = {"train": len(samples), "val": 0, "test": 0}
            report.checks = run_checks({s: [] for s in SPLITS}, built.dropped)
            report.checks.append(CheckResult("split", "error", False, f"切分失败: {exc}"))
            self._write_report(report, Path(out_dir))
            return report

        split_map = splits.as_dict()
        report.split_counts = {s: len(split_map[s]) for s in SPLITS}
        report.checks = run_checks(split_map, built.dropped)

        if not report.passed:
            self._write_report(report, Path(out_dir))
            return report

        report.export = export_splits(splits, self.template, out_dir)
        self._write_report(report, Path(out_dir))
        return report

    @staticmethod
    def _write_report(report: WizardReport, out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "wizard_report.json", "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, ensure_ascii=False, indent=2)
