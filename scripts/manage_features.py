#!/usr/bin/env python3
"""Public CLI for read-only inspection and immutable feature derivations."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from yonod.artifacts.operations import ArtifactOperationError, derive_features, inspect_feature_artifact


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="查看或另存 schema-2 特征版本")
    commands = parser.add_subparsers(dest="command", required=True)
    inspect_parser = commands.add_parser("inspect", help="只读校验并显示 manifest 摘要")
    inspect_parser.add_argument("--manifest", required=True, type=Path)
    derive_parser = commands.add_parser("derive", help="执行 derive_features YAML，绝不改写父版本")
    derive_parser.add_argument("--config", required=True, type=Path, help="operation: derive_features 的 YAML")
    args = parser.parse_args(argv)
    try:
        if args.command == "inspect":
            inspection = inspect_feature_artifact(args.manifest)
            print(json.dumps({
                "artifact_id": inspection.artifact_id,
                "feature_id": inspection.feature_id,
                "lifecycle": inspection.lifecycle,
                "matrix_shape": inspection.matrix_shape,
                "matrix_dtype": inspection.matrix_dtype,
                "sample_count": inspection.sample_count,
                "valid_sample_count": inspection.valid_sample_count,
                "parent_artifact_id": inspection.parent_artifact_id,
                "content_identity": inspection.content_identity,
            }, ensure_ascii=False, indent=2))
            return 0
        result = derive_features(args.config)
        print(json.dumps({
            "output_root": str(result.output_root),
            "manifest": str(result.manifest_path),
            "operations": [
                {"index": item.index, "kind": item.kind, "artifact_id": item.artifact_id, "manifest": str(item.manifest_path)}
                for item in result.operations
            ],
        }, ensure_ascii=False, indent=2))
        return 0
    except (ArtifactOperationError, FileNotFoundError, ValueError) as exc:
        print(f"[error] 特征操作失败：{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
