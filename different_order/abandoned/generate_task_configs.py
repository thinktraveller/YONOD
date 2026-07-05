"""Generate different_order task config files from project-goal.md.

This script copies the completed 1234567 config as a template, then rewrites
project_name and every descriptor's columns according to the task digit order.
"""

from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent
GOAL_PATH = ROOT / "project-goal.md"
SOURCE_CONFIG_PATH = ROOT / "1234567" / "1234567_yonod_config.json"
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
    if len(tasks) != 31:
        raise RuntimeError(f"Expected 31 pending tasks, got {len(tasks)}: {tasks}")
    return tasks


def main() -> None:
    tasks = load_tasks()
    source_config = json.loads(SOURCE_CONFIG_PATH.read_text(encoding="utf-8"))

    for task in tasks:
        config = json.loads(json.dumps(source_config, ensure_ascii=False))
        columns = [COLUMN_MAP[digit] for digit in task]

        config["project_name"] = task
        for descriptor in config.get("descriptors", []):
            descriptor["columns"] = columns

        task_dir = ROOT / task
        task_dir.mkdir(parents=True, exist_ok=True)
        out_path = task_dir / f"{task}_yonod_config.json"
        out_path.write_text(
            json.dumps(config, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(out_path.relative_to(ROOT.parent))


if __name__ == "__main__":
    main()
