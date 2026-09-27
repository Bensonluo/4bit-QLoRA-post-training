#!/usr/bin/env python3
"""Wait for a failed training worker to exit before its single authorized retry."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    from src.workbench.training_recovery import dispatch_after_exit

    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--parent-pid", required=True, type=int)
    parser.add_argument("--parent-created", required=True, type=float)
    parser.add_argument("--timeout", type=float, default=120)
    args = parser.parse_args()
    result = dispatch_after_exit(args.run_dir, args.parent_pid, args.parent_created, args.timeout)
    print(result, flush=True)
    return 1 if result["status"] in {"dispatch_failed", "retry_blocked"} else 0


if __name__ == "__main__":
    raise SystemExit(main())
