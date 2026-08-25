from __future__ import annotations

from pathlib import Path
import argparse
import json
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description="Run VJETHBKM reproduction benchmarks.")
    parser.add_argument("--stage", default="smoke", choices=["smoke"])
    args = parser.parse_args()

    repro_root = Path(__file__).resolve().parents[1]
    src_dir = repro_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    from vjethbkm_repro.benchmark import run_smoke

    if args.stage == "smoke":
        result = run_smoke()
    else:
        raise ValueError(f"Unsupported stage: {args.stage}")

    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

