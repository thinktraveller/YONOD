from __future__ import annotations

from pathlib import Path
import argparse
import json
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe VJETHBKM official source URLs and local checksums.")
    parser.add_argument("--timeout", type=int, default=20)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Return a non-zero exit code when required public URLs cannot be probed successfully.",
    )
    args = parser.parse_args()

    repro_root = Path(__file__).resolve().parents[1]
    src_dir = repro_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    from vjethbkm_repro.sources import probe_sources, write_probe_manifest

    records = probe_sources(timeout_s=args.timeout)
    output_path = write_probe_manifest(records)
    print(json.dumps({"output": str(output_path), "records": records}, ensure_ascii=False, indent=2))

    required_public_sources = {"vjethbkm-main", "vjethbkm-si", "vjethbkm-data-code"}
    failed = [
        record
        for record in records
        if record["source_id"] in required_public_sources and not record["ok"]
    ]
    if failed:
        print(
            "source probe recorded blocked public sources; use --strict to fail CI on these records",
            file=sys.stderr,
        )
    return 1 if args.strict and failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
