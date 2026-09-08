"""Prepare YieldMaster paper-exact materials and rerun RF alignment.

This command intentionally stops after RF.  It does not start AutoGluon; the
next stage must only proceed if the RF alignment file says every row passed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from yonod.benchmark.paper_exact_pipeline import (  # noqa: E402
    PAPER_EXACT_RF_PARAMS,
    PAPER_EXACT_STAMP,
    prepare_all_paper_exact_materials,
    run_paper_exact_rf_matrix,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="生成 YieldMaster paper_exact 5x5 材料并只重跑 RF 对齐矩阵"
    )
    parser.add_argument(
        "--reference-root",
        type=Path,
        default=PROJECT_ROOT / "reference-proejct" / "vjethbkm",
        help="vjethbkm 参考工程根目录",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "configs",
        help="paper_exact 输出根；会在其中创建新的 paper_exact 目录",
    )
    parser.add_argument("--stamp", default=PAPER_EXACT_STAMP, help="输出目录日期戳")
    parser.add_argument(
        "--population",
        action="append",
        dest="populations",
        default=None,
        help="只处理指定 population_id；可重复传入。默认处理全部五个 population",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="只生成 population/feature/split/task 材料，不训练 RF",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="允许覆盖同一路径下已有 paper_exact 材料；默认拒绝覆盖不同内容",
    )
    parser.add_argument("--tolerance", type=float, default=1e-9, help="RF 对齐容差")
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=int(PAPER_EXACT_RF_PARAMS["n_estimators"]),
        help="RF 树数；正式对齐必须为 500",
    )
    parser.add_argument(
        "--max-features",
        type=float,
        default=float(PAPER_EXACT_RF_PARAMS["max_features"]),
        help="RF max_features；正式对齐必须为 0.30",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=int(PAPER_EXACT_RF_PARAMS["n_jobs"]),
        help="RF n_jobs；正式对齐按论文 wrapper 使用 -1",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.prepare_only:
        materials = prepare_all_paper_exact_materials(
            args.reference_root,
            args.output_root,
            population_ids=args.populations,
            stamp=args.stamp,
            overwrite=args.overwrite,
        )
        print(json.dumps({
            "status": "prepared",
            "population_count": len(materials),
            "population_dirs": [str(item.population_dir) for item in materials],
            "descriptor_task_count": sum(len(item.feature_paths) for item in materials),
        }, ensure_ascii=False, indent=2))
        return 0

    result = run_paper_exact_rf_matrix(
        args.reference_root,
        args.output_root,
        population_ids=args.populations,
        stamp=args.stamp,
        overwrite=args.overwrite,
        tolerance=args.tolerance,
        n_estimators=args.n_estimators,
        max_features=args.max_features,
        n_jobs=args.n_jobs,
    )
    print(json.dumps({
        "status": "rf_alignment_passed",
        "output_root": str(result.output_root),
        "fold_metrics": str(result.fold_metrics_path),
        "summary": str(result.summary_path),
        "alignment": str(result.alignment_path),
        "max_abs_metric_diff": result.max_abs_metric_diff,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
