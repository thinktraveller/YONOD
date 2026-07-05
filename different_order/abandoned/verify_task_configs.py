"""Validate generated different_order task configs."""

from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent
GOAL_PATH = ROOT / "project-goal.md"
COMPLETED_TASK = "1234567"

COLUMN_MAP = {
    "1": "reactant-amide",
    "2": "reactant-acid",
    "3": "product",
    "4": "activation",
    "5": "additive",
    "6": "base",
    "7": "solvent",
}


def load_tasks() -> list[str]:
    text = GOAL_PATH.read_text(encoding="utf-8")
    tasks: list[str] = []
    for line in text.splitlines():
        match = re.match(r"^(\d+)(?:（.*）)?$", line.strip())
        if match:
            task = match.group(1)
            if task != COMPLETED_TASK:
                tasks.append(task)
    return tasks


def main() -> int:
    tasks = load_tasks()
    errors: list[str] = []
    config_count = 0

    if len(tasks) != 31:
        errors.append(f"Expected 31 pending tasks, got {len(tasks)}")

    for task in tasks:
        path = ROOT / task / f"{task}_yonod_config.json"
        if not path.exists():
            errors.append(f"Missing config: {path.relative_to(ROOT.parent)}")
            continue

        config_count += 1
        config = json.loads(path.read_text(encoding="utf-8"))
        expected_columns = [COLUMN_MAP[digit] for digit in task]

        if config.get("project_name") != task:
            errors.append(
                f"{task}: project_name={config.get('project_name')!r}, expected {task!r}"
            )

        descriptors = config.get("descriptors", [])
        if len(descriptors) != 6:
            errors.append(f"{task}: descriptor count={len(descriptors)}, expected 6")

        for descriptor in descriptors:
            if descriptor.get("columns") != expected_columns:
                errors.append(
                    f"{task}/{descriptor.get('descriptor')}: "
                    f"columns={descriptor.get('columns')!r}, "
                    f"expected={expected_columns!r}"
                )

    print(f"tasks={len(tasks)}")
    print(f"config_files={config_count}")
    if tasks:
        first = tasks[0]
        last = tasks[-1]
        print(f"first_task={first} columns={[COLUMN_MAP[d] for d in first]}")
        print(f"last_task={last} columns={[COLUMN_MAP[d] for d in last]}")

    if errors:
        print("validation=failed")
        for error in errors:
            print(error)
        return 1

    print("validation=passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
