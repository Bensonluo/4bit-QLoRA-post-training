#!/usr/bin/env python3
"""Detached worker: advance one authorized iteration execution to its end.

Launched by IterationExecutionService with start_new_session so closing the UI
page never interrupts the handoff. All state lives in the execution record;
this process can be killed at any point without corrupting it.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--execution-id", required=True)
    parser.add_argument("--execution-root", required=True)
    args = parser.parse_args()
    record = json.loads(
        (Path(args.execution_root) / args.execution_id / "execution.json").read_text(
            encoding="utf-8"
        )
    )
    from src.workbench.iteration_execution import IterationExecutionService

    service = IterationExecutionService(
        args.execution_root,
        record["intake_root"],
        record["iteration_root"],
        record["training_root"],
        record["evaluation_root"],
        project_root=record["project_root"],
        python_executable=record["python_executable"],
    )
    return 0 if service.run_worker(args.execution_id) else 1


if __name__ == "__main__":
    raise SystemExit(main())
