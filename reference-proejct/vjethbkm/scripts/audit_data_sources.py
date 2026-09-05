from __future__ import annotations

from pathlib import Path
import sys


def main() -> int:
    repro_root = Path(__file__).resolve().parents[1]
    src_dir = repro_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    from vjethbkm_repro.manifest import (
        dataset_config_from_sources,
        dataset_stats,
        read_dataset,
        write_stats_csv,
    )

    config = dataset_config_from_sources("smoke_local_yonod")
    df = read_dataset(config)
    stats = dataset_stats(df, config)
    output_path = repro_root / "data" / "manifest" / "dataset_stats.csv"
    write_stats_csv([stats], output_path)
    print(f"wrote {output_path}")
    print(f"rows={stats['n_rows_read']} target={stats['target_min']:.3f}..{stats['target_max']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

