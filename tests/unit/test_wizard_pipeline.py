"""数据向导流水线端到端测试（内存表格 → 导出产物）。"""

import json

from src.data.wizard.exporter import split_by_entity  # noqa: F401  (保证可导入)
from src.data.wizard.importers import RawTable
from src.data.wizard.pipeline import WizardPipeline, WizardReport
from src.data.wizard.spec import FieldMapping, WizardSpec
from src.data.wizard.templates import (
    BuildResult,
    Candidate,
    DomainTemplate,
    MatchingSample,
    available_templates,
    register_template,
)


def make_table(rows: list[dict], columns: list[str]) -> RawTable:
    return RawTable(source="mem://test", columns=columns, rows=rows)


DRUG_ROWS = [{"标准名": f"药{i}", "别名": f"别名{i}", "编码": f"Z{i}"} for i in range(12)]
DRUG_COLUMNS = ["标准名", "别名", "编码"]


def make_spec(**overrides) -> WizardSpec:
    mapping = FieldMapping(standard_name="标准名", query="别名", code="编码")
    return WizardSpec(mapping=mapping, **overrides)


class TestHappyPath:
    def test_full_run_exports(self, tmp_path) -> None:
        report = WizardPipeline(make_spec()).run(
            make_table(DRUG_ROWS, DRUG_COLUMNS), tmp_path / "run"
        )
        assert report.passed, report.summary_lines()
        assert report.export is not None
        assert report.total_rows == 12
        assert report.built_samples == 12
        assert report.dedup_removed == 0
        # 三个 split 文件 + 报告
        out = tmp_path / "run"
        assert {p.name for p in out.iterdir()} == {
            "train.json",
            "val.json",
            "test.json",
            "wizard_report.json",
        }
        with open(out / "train.json", encoding="utf-8") as f:
            records = json.load(f)
        assert records, "train 非空"
        assert set(records[0]) == {"instruction", "input", "output", "metadata"}

    def test_output_compatible_with_medical_dataset(self, tmp_path) -> None:
        """导出格式必须与 MedicalEntityDataset 的键约定一致（instruction/input/output/metadata）。"""
        report = WizardPipeline(make_spec()).run(make_table(DRUG_ROWS, DRUG_COLUMNS), tmp_path)
        assert report.passed
        for split in ("train", "val", "test"):
            with open(tmp_path / f"{split}.json", encoding="utf-8") as f:
                for rec in json.load(f):
                    assert rec["metadata"]["difficulty"] in {"easy", "medium", "hard"}
                    assert rec["metadata"]["entity_type"]

    def test_summary_lines_and_to_dict(self, tmp_path) -> None:
        report = WizardPipeline(make_spec()).run(make_table(DRUG_ROWS, DRUG_COLUMNS), tmp_path)
        lines = "\n".join(report.summary_lines())
        assert "模板: medical_entity" in lines
        assert "切分: " in lines
        assert "已导出" in lines
        d = report.to_dict()
        assert d["template"] == "medical_entity"
        assert d["export"] is not None
        assert len(d["checks"]) == 6

    def test_dedup_removes_duplicate_queries(self, tmp_path) -> None:
        rows = DRUG_ROWS + [DRUG_ROWS[0]]  # 完全重复行
        report = WizardPipeline(make_spec()).run(make_table(rows, DRUG_COLUMNS), tmp_path)
        assert report.dedup_removed == 1
        # built_samples 统计去重前的生成总数（13 行 → 13 条，其中 1 条被去重）
        assert report.built_samples == 13

    def test_dedup_disabled(self, tmp_path) -> None:
        rows = DRUG_ROWS + [DRUG_ROWS[0]]
        report = WizardPipeline(make_spec(dedup=False)).run(
            make_table(rows, DRUG_COLUMNS), tmp_path
        )
        assert report.dedup_removed == 0
        assert report.built_samples == 13


class _SingleCandidateStub(DomainTemplate):
    """体检失败路径专用桩模板：每样本只有 1 个候选（触发 candidate_counts error）。

    切分本身可成功（6 个实体组），因此失败发生在「体检 → 阻断导出」阶段。
    """

    name = "stub_single_candidate"

    def describe(self) -> str:
        return "桩模板：单候选"

    def build_samples(
        self, table: RawTable, mapping: FieldMapping, spec: WizardSpec
    ) -> BuildResult:
        samples = [
            MatchingSample(
                query=f"q{i}",
                standard_name=f"S{i}",
                code=f"C{i}",
                entity_type="entity",
                difficulty="easy",
                candidates=[Candidate(f"S{i}", f"C{i}", True)],
                source_row=i + 1,
            )
            for i in range(6)
        ]
        return BuildResult(samples=samples, dropped=[])

    def format_record(self, sample: MatchingSample) -> dict[str, object]:
        return {"instruction": "stub", "input": sample.query, "output": "", "metadata": {}}


if "stub_single_candidate" not in available_templates():
    register_template(_SingleCandidateStub())


class TestBlockedPaths:
    def test_failing_check_after_split_blocks_export(self, tmp_path) -> None:
        """切分成功但体检报 error → 只写报告不导出（155-156 路径）。"""
        spec = WizardSpec(
            mapping=FieldMapping(standard_name="标准名"), template="stub_single_candidate"
        )
        report = WizardPipeline(spec).run(make_table(DRUG_ROWS[:6], DRUG_COLUMNS), tmp_path)
        assert not report.passed
        assert report.export is None
        assert [c.check_id for c in report.blocking_errors] == ["candidate_counts"]
        assert (tmp_path / "wizard_report.json").exists()
        assert not (tmp_path / "train.json").exists()
        assert "未导出" in "\n".join(report.summary_lines())

    def test_all_rows_dropped(self, tmp_path) -> None:
        rows = [{"标准名": None, "别名": "x", "编码": "Z"} for _ in range(3)]
        report = WizardPipeline(make_spec()).run(make_table(rows, DRUG_COLUMNS), tmp_path)
        assert not report.passed
        assert report.export is None
        assert report.split_counts == {"train": 0, "val": 0, "test": 0}
        assert (tmp_path / "wizard_report.json").exists()
        assert not (tmp_path / "train.json").exists()
        assert "未导出" in "\n".join(report.summary_lines())

    def test_too_few_entities_blocks_export(self, tmp_path) -> None:
        rows = [
            {"标准名": "药A", "别名": "a1", "编码": "Z1"},
            {"标准名": "药A", "别名": "a2", "编码": "Z1"},
        ]
        report = WizardPipeline(make_spec()).run(make_table(rows, DRUG_COLUMNS), tmp_path)
        assert not report.passed
        split_errors = [c for c in report.checks if c.check_id == "split"]
        assert len(split_errors) == 1
        assert "切分失败" in split_errors[0].message
        assert report.export is None
        assert report.split_counts == {"train": 2, "val": 0, "test": 0}

    def test_dropped_rows_warning_does_not_block(self, tmp_path) -> None:
        rows = DRUG_ROWS + [{"标准名": None, "别名": "孤儿", "编码": None}]
        report = WizardPipeline(make_spec()).run(make_table(rows, DRUG_COLUMNS), tmp_path)
        # 跳过行是 warning：12 个好样本仍应导出
        assert report.passed, report.summary_lines()
        assert report.export is not None
        assert len(report.dropped_rows) == 1

    def test_blocking_errors_property(self, tmp_path) -> None:
        rows = [{"标准名": None, "别名": "x", "编码": "Z"}]
        report = WizardPipeline(make_spec()).run(make_table(rows, DRUG_COLUMNS), tmp_path)
        assert [c.check_id for c in report.blocking_errors] == ["no_samples"]
        assert all(c.severity == "error" for c in report.blocking_errors)


class TestReportShapes:
    def test_empty_report_defaults(self) -> None:
        report = WizardReport(
            template="t", source="s", total_rows=0, built_samples=0, dedup_removed=0
        )
        assert report.passed  # 无 error 即通过
        assert report.blocking_errors == []
        d = report.to_dict()
        assert d["export"] is None
        assert d["dropped_rows"] == []

    def test_summary_lines_omit_split_when_empty(self) -> None:
        # split_counts 为空（报告刚构造、尚未切分）→ 摘要不输出「切分:」行
        report = WizardReport(
            template="t", source="s", total_rows=0, built_samples=0, dedup_removed=0
        )
        lines = report.summary_lines()
        assert not any(line.startswith("切分:") for line in lines)
        assert "未导出" in "\n".join(lines)
