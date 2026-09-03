"""Run the serial, resumable strict benchmark pipeline from a YAML contract."""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path
from typing import Any, Dict

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from yonod.benchmark import (
    BenchmarkConfig,
    TaskSpec,
    TaskStateStore,
    create_benchmark_contract,
    execute_fold,
    generate_benchmark_report,
    paired_comparisons,
    rebuild_fold_metrics,
    summarize_combinations,
    tukey_hsd_comparisons,
    write_metric_tables,
    write_run_manifest,
)
from yonod.benchmark.layout import resolve_benchmark_output_layout
from yonod.splits import create_split_manifest, write_split_manifest
from yonod.universal.feature_builder import build_universal_features


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行 YONOD 严谨、可恢复的外部-fold benchmark")
    parser.add_argument("--config", required=True, type=Path, help="版本化 benchmark YAML 配置")
    parser.add_argument("--rerun-failed", action="store_true", help="显式重新执行 failed/interrupted 任务")
    parser.add_argument("--task-key", action="append", default=None, help="只执行指定任务键；可重复传入")
    parser.add_argument("--stale-seconds", type=float, default=3600.0, help="running 心跳超过此秒数标为 interrupted")
    parser.add_argument("--bootstrap-n", type=int, default=2000, help="组合/比较 CI 的 bootstrap 重采样次数，至少 100")
    return parser.parse_args()


def _feature_matrices(config: BenchmarkConfig, frame, descriptor: str):
    X_smiles, X_numeric, mask = build_universal_features(
        smiles_cols=list(config.smiles_cols), numeric_cols=[], df=frame, desc_name=descriptor,
    )
    if not mask.all():
        raise RuntimeError(
            "描述符 {0!r} 过滤了 {1} 行。严格 benchmark 的 split manifest 已绑定全量 sample_id；"
            "请先修复输入 SMILES，不能静默缩小数据集。".format(descriptor, int((~mask).sum()))
        )
    return X_smiles, X_numeric


def _complete_metrics(rebuilt):
    complete = rebuilt.completeness.loc[rebuilt.completeness["is_complete"], ["split_id", "descriptor", "model"]]
    return rebuilt.fold_metrics.merge(complete, on=["split_id", "descriptor", "model"], how="inner")


def main() -> int:
    args = _parse_args()
    config = BenchmarkConfig.from_file(args.config)
    contract = create_benchmark_contract(config)
    write_run_manifest(contract)
    split_manifest = create_split_manifest(contract)
    write_split_manifest(contract, split_manifest)
    split_id = str(split_manifest["split_id"].iloc[0])
    specs = [
        TaskSpec(contract.run_id, contract.config_hash, split_id, descriptor, model, repeat, fold)
        for descriptor, model in itertools.product(config.descriptors, config.models)
        for repeat, fold in split_manifest[["repeat", "fold"]].drop_duplicates().itertuples(index=False, name=None)
    ]
    state = TaskStateStore(
        resolve_benchmark_output_layout(contract.run_dir).state / "tasks.sqlite"
    )
    state.sync_tasks(specs)
    requeued = state.audit_succeeded_outputs()
    interrupted = state.recover_stale_running(args.stale_seconds)
    print("[plan] run_id={0} theoretical_tasks={1} requeued={2} interrupted={3}".format(contract.run_id, len(specs), requeued, interrupted))

    frame = config.validate_dataset()
    sample_ids = frame[config.sample_id_col].astype(str).tolist()
    labels = frame[config.label_col].astype(float).to_numpy()
    model_params: Dict[str, Dict[str, Any]] = dict(config.raw.get("model_params", {}))
    feature_cache: Dict[str, tuple[Any, Any]] = {}
    while claimed := state.claim_next(rerun_failed=args.rerun_failed, task_keys=args.task_key):
        try:
            if claimed.spec.descriptor not in feature_cache:
                feature_cache[claimed.spec.descriptor] = _feature_matrices(config, frame, claimed.spec.descriptor)
            X_smiles, X_numeric = feature_cache[claimed.spec.descriptor]
            result = execute_fold(
                contract, split_manifest, sample_ids, X_smiles, labels,
                claimed.spec.descriptor, claimed.spec.model, claimed.spec.repeat, claimed.spec.fold,
                X_numeric=X_numeric, model_kwargs=model_params.get(claimed.spec.model, {}),
            )
            state.mark_succeeded(claimed, result.prediction_path, result.metadata_path)
            print("[succeeded]", claimed.spec.task_key)
        except BaseException as exc:
            state.mark_failed(claimed, exc)
            print("[failed] {0}: {1}".format(claimed.spec.task_key, exc), file=sys.stderr)
            if isinstance(exc, KeyboardInterrupt):
                raise

    rebuilt = rebuild_fold_metrics(contract.run_dir)
    complete_metrics = _complete_metrics(rebuilt)
    summary = summarize_combinations(complete_metrics, n_bootstrap=args.bootstrap_n, seed=int(config.cv["seed"]))
    model_comparisons, model_exclusions = paired_comparisons(
        complete_metrics, rebuilt.completeness, dimension="model", n_bootstrap=args.bootstrap_n, seed=int(config.cv["seed"]),
    )
    descriptor_comparisons, descriptor_exclusions = paired_comparisons(
        complete_metrics, rebuilt.completeness, dimension="descriptor", n_bootstrap=args.bootstrap_n, seed=int(config.cv["seed"]),
    )
    model_tukey = tukey_hsd_comparisons(complete_metrics, dimension="model")
    descriptor_tukey = tukey_hsd_comparisons(complete_metrics, dimension="descriptor")
    write_metric_tables(
        contract.run_dir, rebuilt, summary, model_comparisons, descriptor_comparisons,
        pd.concat([model_exclusions, descriptor_exclusions], ignore_index=True), model_tukey, descriptor_tukey,
    )
    report = generate_benchmark_report(contract.run_dir)
    counts = state.summary()
    print("[state]", counts)
    print("[report]", report.html_path)
    return 0 if counts["failed"] == 0 and counts["interrupted"] == 0 and counts["pending"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
