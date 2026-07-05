"""Run the 31 different_order modeling tasks.

By default this executes each command in sequence and stops on the first
failure. Use --dry-run to print commands without executing them.
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path


TASKS = [
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
    "124567"
]

CSV_PATH = "different_order/amide-coupling(additive_fixed)_normalized_dataset.csv"


def build_command(task: str, python_executable: str) -> list[str]:
    config_path = f"different_order/{task}/{task}_yonod_config.json"
    return [
        python_executable,
        "main.py",
        "--config",
        config_path,
        "--csv",
        CSV_PATH,
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="print commands only")
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="continue running later tasks when one task fails",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable used to run main.py",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    failures: list[tuple[str, int]] = []

    for index, task in enumerate(TASKS, start=1):
        command = build_command(task, args.python)
        printable = shlex.join(command)
        print(f"[{index:02d}/{len(TASKS)}] {task}: {printable}", flush=True)

        if args.dry_run:
            continue

        result = subprocess.run(command, cwd=repo_root)
        if result.returncode != 0:
            failures.append((task, result.returncode))
            if not args.continue_on_error:
                break

    if failures:
        for task, code in failures:
            print(f"FAILED {task}: exit code {code}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
