
"""Prepare or run YieldMaster paper-exact AutoGluon 5x5 jobs.

Default formal execution requires the RF gate produced by
``run_yieldmaster_paper_exact_rf_20260908.py``.  Use ``--prepare-only`` to write
configs and launch manifests without training; use ``--allow-partial`` together
with ``--max-folds-per-task`` only for smoke tests.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from yonod.benchmark.paper_exact_autogluon import (  # noqa: E402
    PAPER_EXACT_AUTOGLOON_PARAMS,
    PAPER_EXACT_STAMP,
    run_paper_exact_autogluon_matrix,
    write_paper_exact_autogluon_launch_materials,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="运行 YieldMaster paper_exact 5x5 AutoGluon；默认要求 RF gate 已通过"
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
        help="paper_exact 输出根；会在其中使用隔离的 paper_exact 目录",
    )
    parser.add_argument("--stamp", default=PAPER_EXACT_STAMP, help="输出目录日期戳")
    parser.add_argument(
        "--population",
        action="append",
        dest="populations",
        default=None,
        help="只处理指定 population_id；可重复传入。正式默认处理全部五个 population",
    )
    parser.add_argument(
        "--feature",
        action="append",
        dest="features",
        default=None,
        choices=("mfp", "ohe"),
        help="只处理指定 feature_id；可重复传入。正式默认使用每个 population 声明的全部特征",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="只写 AutoGluon 配置、任务 manifest 和启动材料，不训练",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="允许覆盖同一路径下已有 paper_exact/AutoGluon 输出；默认拒绝覆盖不同内容",
    )
    parser.add_argument(
        "--time-limit",
        type=int,
        default=int(PAPER_EXACT_AUTOGLOON_PARAMS["time_limit"]),
        help="每个 AutoGluon 外层 fold 的 time_limit；正式默认 300 秒",
    )
    parser.add_argument(
        "--presets",
        default=str(PAPER_EXACT_AUTOGLOON_PARAMS["presets"]),
        help="AutoGluon presets；正式默认 medium_quality",
    )
    parser.add_argument(
        "--num-cpus",
        type=int,
        default=int(PAPER_EXACT_AUTOGLOON_PARAMS["num_cpus"]),
        help="每个 AutoGluon fold 可用 CPU；正式默认 19",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=int(PAPER_EXACT_AUTOGLOON_PARAMS["random_state"]),
        help="记录到 AutoGluon 适配器的 random_state；外层 seed 仍由 split_manifest 控制",
    )
    parser.add_argument(
        "--keep-autogluon-artifacts",
        action="store_true",
        help="保留每折 AutoGluon 模型目录；默认训练后清理大体量模型包，只保留预测和 metadata",
    )
    parser.add_argument(
        "--skip-rf-gate",
        action="store_true",
        help="跳过 RF alignment gate；仅限调试，不得用于正式结论",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="允许写出 partial smoke 结果；正式运行不要开启",
    )
    parser.add_argument(
        "--max-folds-per-task",
        type=int,
        default=None,
        help="每个 descriptor task 最多跑几个 fold；仅用于 smoke，必须配合 --allow-partial",
    )
    args = parser.parse_args()
    if args.max_folds_per_task is not None and not args.allow_partial:
        parser.error("--max-folds-per-task 只能与 --allow-partial 一起使用，避免 partial 结果冒充正式 25 折")
    if args.time_limit < 1:
        parser.error("--time-limit 必须 >= 1")
    if args.num_cpus < 1:
        parser.error("--num-cpus 必须 >= 1")
    return args


def main() -> int:
    args = _parse_args()
    cleanup = not bool(args.keep_autogluon_artifacts)
    if args.prepare_only:
        prepared = write_paper_exact_autogluon_launch_materials(
            args.reference_root,
            args.output_root,
            population_ids=args.populations,
            stamp=args.stamp,
            time_limit=args.time_limit,
            presets=args.presets,
            num_cpus=args.num_cpus,
            random_state=args.random_state,
            cleanup=cleanup,
            overwrite=args.overwrite,
        )
        print(json.dumps({
            "status": "autogluon_launch_materials_prepared",
            "batch_dir": str(prepared["batch_dir"]),
            "tasks_path": str(prepared["tasks_path"]),
            "manifest_path": str(prepared["manifest_path"]),
            "descriptor_task_count": prepared["descriptor_task_count"],
            "expected_autogluon_fold_fits": prepared["expected_autogluon_fold_fits"],
            "config_paths": prepared["config_paths"],
        }, ensure_ascii=False, indent=2))
        return 0

    result = run_paper_exact_autogluon_matrix(
        args.reference_root,
        args.output_root,
        population_ids=args.populations,
        feature_ids=args.features,
        stamp=args.stamp,
        overwrite=args.overwrite,
        time_limit=args.time_limit,
        presets=args.presets,
        num_cpus=args.num_cpus,
        random_state=args.random_state,
        cleanup=cleanup,
        require_rf_alignment=not args.skip_rf_gate,
        allow_partial=args.allow_partial,
        max_folds_per_task=args.max_folds_per_task,
    )
    print(json.dumps({
        "status": result.status,
        "output_root": str(result.output_root),
        "expected_fold_count": result.expected_fold_count,
        "completed_fold_count": result.completed_fold_count,
        "fold_metrics": str(result.fold_metrics_path),
        "summary": str(result.summary_path),
        "paired_input": str(result.paired_input_path) if result.paired_input_path else None,
        "run_manifest": str(result.manifest_path),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
