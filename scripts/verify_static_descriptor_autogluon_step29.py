#!/usr/bin/env python3
"""Verify step29 AutoGluon static-descriptor manifest and completion state."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import run_static_descriptor_autogluon_matrix as runner


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-dir", default=runner.DEFAULT_PLAN_DIR.as_posix())
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    plan_dir = Path(args.plan_dir)
    plan = runner.verify_plan(plan_dir)
    completion = runner.completion_summary(plan_dir, require_complete=args.require_complete)
    result = {
        "status": "passed" if completion["status"] in {"complete", "incomplete"} else completion["status"],
        "plan": plan,
        "completion": completion,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
