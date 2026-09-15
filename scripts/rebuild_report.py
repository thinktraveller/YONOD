"""Rebuild a schema-2 training report without features, data, or model fitting."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from yonod.pipeline.reporting import ReportServiceError, rebuild_schema2_report


def main() -> int:
    parser = argparse.ArgumentParser(description="从完成的 schema-2 run_manifest.yaml 重建普通报告；不训练模型。")
    parser.add_argument("--run-dir", required=True, type=Path, help="包含 run_manifest.yaml 的普通训练目录")
    parser.add_argument("--output-root", type=Path, help="可选的新报告根目录；省略则原子替换该 run 的 pictures/report")
    parser.add_argument("--format", dest="formats", action="append", choices=("html", "markdown"), help="覆盖 YAML 的 report_formats；可重复指定")
    args = parser.parse_args()
    try:
        result = rebuild_schema2_report(args.run_dir, output_root=args.output_root, formats=args.formats)
    except ReportServiceError as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 2
    print("run_id:", result.run_id)
    for picture in result.pictures: print("picture:", picture)
    if result.html_path: print("HTML:", result.html_path)
    if result.markdown_path: print("Markdown:", result.markdown_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
