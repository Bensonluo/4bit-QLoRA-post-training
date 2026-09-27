#!/usr/bin/env python3
"""Run the built-in scenario matrix and write the honest coverage report."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> int:
    parser = argparse.ArgumentParser(description="运行内置场景矩阵,输出覆盖率报告")
    parser.add_argument("--root", default="/tmp/tunesmith-scenario-matrix")
    parser.add_argument("--out", default="docs/validation/scenario-matrix-latest.json")
    args = parser.parse_args()

    from src.workbench.scenario_matrix import run_matrix, save_matrix_report
    from src.workbench.scenario_specs import builtin_scenarios

    report = run_matrix(builtin_scenarios(), args.root)
    path = save_matrix_report(report, args.out)
    summary = report["summary"]
    for scenario in report["scenarios"]:
        mark = {
            "as_expected": "✓",
            "unexpected_pass": "✗(意外通过)",
            "unexpected_block": "✗(意外拦截)",
            "error": "✗(错误)",
        }.get(scenario["verdict"], "?")
        print(f"{mark} {scenario['scenario_id']}: {scenario['verdict']}")
        if scenario["verdict"] != "as_expected":
            print(f"    {scenario['detail'] or scenario['blocked_message']}")
    print(
        f"\n合计 {summary['total']}: 符合预期 {summary['as_expected']}, "
        f"意外通过 {summary['unexpected_pass']}, 意外拦截 {summary['unexpected_block']}, "
        f"错误 {summary['error']}\n报告: {path}"
    )
    return (
        0 if summary["unexpected_pass"] + summary["unexpected_block"] + summary["error"] == 0 else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
