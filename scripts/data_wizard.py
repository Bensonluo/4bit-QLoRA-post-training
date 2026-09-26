#!/usr/bin/env python3
"""数据准备向导 CLI：原始表格 → 体检合格的训练集。

用法示例::

    # 先看看能识别到什么（不生成）
    python scripts/data_wizard.py --input data/raw.xlsx --suggest

    # 用自动识别的列映射直接生成
    python scripts/data_wizard.py --input data/raw.xlsx --out-dir outputs/wizard/run1

    # 显式指定映射
    python scripts/data_wizard.py --input data/raw.csv \
        --standard-col 标准名 --query-col 别名 --code-col 编码 \
        --out-dir outputs/wizard/run1

输出 train.json / val.json / test.json（Alpaca 指令格式，可直接接
scripts/train_medical_entity.py 等训练链路）+ wizard_report.json（体检报告）。
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.wizard import (  # noqa: E402
    WizardPipeline,
    WizardSpec,
    available_templates,
    get_template,
    import_table,
    suggest_mapping,
)
from src.data.wizard.spec import FieldMapping, WizardError  # noqa: E402
from src.utils.logging import console  # noqa: E402


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="TuneSmith 数据准备向导：原始表格 → 体检合格的训练集"
    )
    parser.add_argument("--input", required=True, help="原始表格路径 (.csv/.xlsx/.xls/.jsonl)")
    parser.add_argument("--out-dir", default=None, help="输出目录（默认 outputs/wizard/<文件名>）")
    parser.add_argument("--template", default="medical_entity", help="垂类模板名")
    parser.add_argument("--suggest", action="store_true", help="只打印列映射建议与模板说明，不生成")
    parser.add_argument("--standard-col", default=None, help="标准名列（默认自动识别）")
    parser.add_argument("--query-col", default=None, help="查询/别名列（默认自动识别）")
    parser.add_argument("--code-col", default=None, help="编码列（默认自动识别）")
    parser.add_argument("--variants-col", default=None, help="变体列（一格多个别名，默认自动识别）")
    parser.add_argument("--type-col", default=None, help="实体类型列（默认自动识别）")
    parser.add_argument("--spec-col", default=None, help="规格列（产品匹配任务用，默认自动识别）")
    parser.add_argument("--candidates", type=int, default=8, help="每个样本的候选数（默认 8）")
    parser.add_argument(
        "--ratios",
        nargs=3,
        type=float,
        default=[0.8, 0.1, 0.1],
        metavar=("TRAIN", "VAL", "TEST"),
        help="切分比例（默认 0.8 0.1 0.1）",
    )
    parser.add_argument("--seed", type=int, default=42, help="随机种子（同种子同产出）")
    parser.add_argument(
        "--noise-augment",
        action="store_true",
        help="噪音增强：每个样本追加一条带错别字/漏字的查询副本（标签不变，难度重估）",
    )
    parser.add_argument("--keep-duplicates", action="store_true", help="不去重（默认去重）")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)

    try:
        table = import_table(args.input)
    except WizardError as exc:
        console.print(f"[red]导入失败: {exc}[/red]")
        return 2

    auto = suggest_mapping(table.columns)
    mapping = FieldMapping(
        standard_name=args.standard_col or auto.standard_name,
        query=args.query_col if args.query_col is not None else auto.query,
        code=args.code_col if args.code_col is not None else auto.code,
        variants=args.variants_col if args.variants_col is not None else auto.variants,
        entity_type=args.type_col if args.type_col is not None else auto.entity_type,
        spec=args.spec_col if args.spec_col is not None else auto.spec,
    )

    console.print(f"[bold]模板:[/bold] {args.template}")
    console.print(
        f"[bold]来源:[/bold] {table.source}（{len(table.rows)} 行 × {len(table.columns)} 列）"
    )
    console.print(
        f"[bold]列映射:[/bold] standard={mapping.standard_name} | query={mapping.query} | "
        f"code={mapping.code} | variants={mapping.variants} | type={mapping.entity_type}"
        f" | spec={mapping.spec}"
        + ("（未显式指定的项为自动识别）" if not args.standard_col else "")
    )

    if args.suggest:
        console.print(f"\n[dim]{get_template(args.template).describe()}[/dim]")
        console.print(f"\n可用模板: {', '.join(available_templates())}")
        return 0

    out_dir = args.out_dir or str(Path("outputs/wizard") / Path(args.input).stem)
    r_train, r_val, r_test = args.ratios
    try:
        spec = WizardSpec(
            mapping=mapping,
            template=args.template,
            split_ratios=(r_train, r_val, r_test),
            n_candidates=args.candidates,
            noise_augment=args.noise_augment,
            dedup=not args.keep_duplicates,
            seed=args.seed,
        )
        report = WizardPipeline(spec).run(table, out_dir=out_dir)
    except WizardError as exc:
        console.print(f"[red]向导失败: {exc}[/red]")
        return 2

    for line in report.summary_lines():
        console.print(line)
    return 0 if report.passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
