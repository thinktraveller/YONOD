"""CLI: rebuild a benchmark report from an existing run directory only."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from yonod.benchmark.report import generate_benchmark_report


def main() -> int:
    parser = argparse.ArgumentParser(description="从已落盘 benchmark artefacts 重建报告，不训练模型")
    parser.add_argument("--run-dir", required=True, type=Path, help="包含 manifests/、metrics/ 和 predictions/ 的 run 目录")
    args = parser.parse_args()
    result = generate_benchmark_report(args.run_dir)
    print("HTML:", result.html_path)
    print("Markdown:", result.markdown_path)
    for figure in result.figures:
        print("Figure:", figure)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
