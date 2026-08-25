from __future__ import annotations

from pathlib import Path
import argparse
import json
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description="Run VJETHBKM reproduction benchmarks.")
    parser.add_argument("--stage", default="smoke", choices=["smoke", "core_rf_5x5"])
    parser.add_argument("--dataset", default=None)
    parser.add_argument(
        "--allow-smoke-dataset",
        action="store_true",
        help="Allow formal 5x5 workflow validation on the local smoke dataset.",
    )
    args = parser.parse_args()

    repro_root = Path(__file__).resolve().parents[1]
    src_dir = repro_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    from vjethbkm_repro.benchmark import run_core_rf_5x5, run_smoke

    if args.stage == "smoke":
        result = run_smoke()
    elif args.stage == "core_rf_5x5":
        result = run_core_rf_5x5(
            dataset_id=args.dataset or "smoke_local_yonod",
            allow_smoke_dataset=args.allow_smoke_dataset,
        )
    else:
        raise ValueError(f"Unsupported stage: {args.stage}")

    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
