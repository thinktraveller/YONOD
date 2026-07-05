#!/usr/bin/env python3
"""Run different_order YONOD tasks sequentially."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


TASK_NAMES = [
    "12",
    "123",
    "124",
    "125",
    "126",
    "127",
    "1234",
    "1235",
    "1236",
    "1237",
    "1245",
    "1246",
    "1247",
    "1256",
    "1257",
    "1267",
    "12345",
    "12346",
    "12347",
    "12356",
    "12357",
    "12367",
    "12456",
    "12457",
    "12467",
    "12567",
    "123456",
    "123457",
    "123467",
    "123567",
    "124567",
    "1234567",
]

BASELINE_TASK = "1234567"
DEFAULT_CSV = "different_order/amide-coupling(additive_fixed)_normalized_dataset.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch-run the YONOD different_order task configs."
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=None,
        help="Task names to run. Defaults to all tasks listed in project-goal.md.",
    )
    parser.add_argument(
        "--new-only",
        action="store_true",
        help="Run the 31 generated task folders only, excluding the 1234567 baseline.",
    )
    parser.add_argument(
        "--csv",
        default=DEFAULT_CSV,
        help="Normalized dataset path passed through to main.py.",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable used to invoke main.py.",
    )
    parser.add_argument(
        "--main",
        default="main.py",
        help="Path to the YONOD main.py entrypoint.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing them.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue running later tasks if one task fails.",
    )
    return parser.parse_args()


def resolve_tasks(args: argparse.Namespace) -> list[str]:
    tasks = args.tasks if args.tasks is not None else list(TASK_NAMES)
    unknown = [task for task in tasks if task not in TASK_NAMES]
    if unknown:
        raise SystemExit(f"Unknown task name(s): {', '.join(unknown)}")
    if args.new_only:
        tasks = [task for task in tasks if task != BASELINE_TASK]
    return tasks


def command_for(task: str, args: argparse.Namespace) -> list[str]:
    config_path = f"different_order/{task}/{task}_yonod_config.json"
    return [
        args.python,
        args.main,
        "--config",
        config_path,
        "--csv",
        args.csv,
    ]


def main() -> int:
    args = parse_args()
    tasks = resolve_tasks(args)

    failed: list[tuple[str, int]] = []
    for index, task in enumerate(tasks, start=1):
        cmd = command_for(task, args)
        printable = " ".join(f'"{part}"' if " " in part else part for part in cmd)
        print(f"[{index}/{len(tasks)}] {task}: {printable}", flush=True)

        if args.dry_run:
            continue

        result = subprocess.run(cmd)
        if result.returncode != 0:
            failed.append((task, result.returncode))
            print(f"[error] task {task} failed with exit code {result.returncode}", file=sys.stderr)
            if not args.continue_on_error:
                return result.returncode

    if failed:
        print("[done] completed with failures:", file=sys.stderr)
        for task, code in failed:
            print(f"  - {task}: exit code {code}", file=sys.stderr)
        return 1

    print("[done] all requested tasks completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
