"""Shared schema-2 task output layout (paths are relative to the YAML)."""

from pathlib import Path
from typing import Any, Mapping
from urllib.parse import quote

from .contracts import resolve_config_path


def task_root(config_path: Path, config: Mapping[str, Any]) -> Path:
    outputs = config.get("outputs") or {}
    if outputs.get("root"):
        return resolve_config_path(config_path, str(outputs["root"]))
    reference = (config.get("artifacts") or {}).get("output_dir")
    if reference:
        feature_root = resolve_config_path(config_path, str(reference))
        return feature_root.parent
    return resolve_config_path(config_path, "result")


def create_task_layout(root: Path) -> None:
    for name in ("feature", "pictures", "report", "runs"):
        (root / name).mkdir(parents=True, exist_ok=True)


def combination_name(feature_id: str, model: str) -> str:
    # Percent encoding is reversible and does not collapse distinct IDs with
    # slashes, punctuation or Unicode into the same Windows-safe filename.
    return f"run_{quote(feature_id, safe='-_')}-{quote(model, safe='-_')}"
