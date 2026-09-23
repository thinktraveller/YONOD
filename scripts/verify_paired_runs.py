"""Fail-closed paired ΔP summary for new schema-2 training runs.

Historical strict results lack the saved molecular/numeric block identities
required by this gate. They must remain explicitly legacy/unverified rather
than being silently treated as equivalent to a verified schema-2 pair.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from yonod.pipeline.paired import verified_paired_delta_p


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-run", type=Path, required=True)
    parser.add_argument("--reduced-run", type=Path, required=True)
    parser.add_argument("--full-feature", type=Path, required=True)
    parser.add_argument("--reduced-feature", type=Path, required=True)
    parser.add_argument("--allowed-extra", action="append", required=True)
    parser.add_argument("--required-shared", action="append", default=[])
    args = parser.parse_args()
    summary = verified_paired_delta_p(
        args.full_run, args.reduced_run, args.full_feature,
        args.reduced_feature, allowed_extra_columns=args.allowed_extra,
        required_shared_columns=args.required_shared,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
