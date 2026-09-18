"""Run the serial, resumable strict benchmark pipeline from a YAML contract."""

from __future__ import annotations

import argparse
import importlib.util
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
from yonod.splits import create_split_manifest, load_external_split_manifest, write_split_manifest
from yonod.universal.feature_builder import build_universal_features
from yonod.universal.descriptor_artifact import (
    descriptor_artifact_path,
    load_descriptor_artifact,
    prepare_descriptor_artifact,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行 YONOD 严谨、可恢复的外部-fold benchmark")
    parser.add_argument("--config", required=True, type=Path, help="版本化 benchmark YAML 配置")
    parser.add_argument("--rerun-failed", action="store_true", help="显式重新执行 failed/interrupted 任务")
    parser.add_argument("--task-key", action="append", default=None, help="只执行指定任务键；可重复传入")
    parser.add_argument("--stale-seconds", type=float, default=3600.0, help="running 心跳超过此秒数标为 interrupted")
    parser.add_argument("--bootstrap-n", type=int, default=2000, help="组合/比较 CI 的 bootstrap 重采样次数，至少 100")
    return parser.parse_args()


def _require_parquet_engine() -> None:
    """Fail before publishing a run when strict parquet evidence is unavailable."""
    if importlib.util.find_spec("pyarrow") is not None or importlib.util.find_spec("fastparquet") is not None:
        return
    raise RuntimeError(
        "严格 benchmark 需要 pyarrow 或 fastparquet 来原子发布/验证 split、预测与指标 parquet；"
        "当前 yonod 环境均未安装。请安装 requirements.txt 中的 pyarrow 后，通过 yonod.py 重新启动。"
    )


def _prepare_feature_artifacts(config: BenchmarkConfig, frame, run_dir: Path):
    """Precompute only global descriptors; defer fold transformers to execution."""
    layout = resolve_benchmark_output_layout(run_dir)
    sample_ids = frame[config.sample_id_col].astype(str).tolist()
    features: Dict[str, tuple[Any, Any]] = {}
    failures: Dict[str, str] = {}
    statuses = []
    print("[phase 1/2] precompute or reuse eligible feature sets")
    feature_sets = getattr(config, "feature_sets", None)
    if feature_sets is None:
        # Keep the helper callable by older integrations/tests that pass the
        # original lightweight config shape instead of BenchmarkConfig.
        feature_sets = tuple({
            "name": descriptor,
            "algorithm": descriptor,
            "kind": "precomputed_descriptor",
            "component_cols": list(config.smiles_cols),
            "params": {},
        } for descriptor in config.descriptors)
    for feature_set in feature_sets:
        descriptor = str(feature_set["name"])
        algorithm = str(feature_set.get("algorithm", descriptor))
        kind = str(feature_set["kind"])
        component_cols = list(feature_set["component_cols"])
        mode = str(feature_set.get("mode", "concat"))
        if kind == "fold_transform":
            # OHE (and any future transformer with this lifecycle) must never
            # be fitted globally or persisted as a whole-dataset artifact.
            features[descriptor] = (None, None)
            statuses.append({
                "descriptor": descriptor, "status": "deferred",
                "stage": "fold_transform", "artifact_path": None,
                "n_total": len(frame), "n_valid": len(frame), "feature_dim": None,
                "skipped_model_count": 0,
                "reason": "将在每个外部 fold 的训练样本上单独 fit",
            })
            print("[descriptor:deferred] {0} will fit inside each fold".format(descriptor))
            continue
        artifact_path = descriptor_artifact_path(layout.descriptors, descriptor)
        try:
            preparation = prepare_descriptor_artifact(
                layout.descriptors,
                descriptor=algorithm,
                feature_id=descriptor,
                smiles_cols=component_cols,
                df=frame,
                sample_ids=sample_ids,
                sample_id_name=config.sample_id_col,
                compute=build_universal_features,
                mode=mode,
                descriptor_config=dict(feature_set["params"]),
            )
            # Reload even newly-created artifacts so modelling has one input path.
            artifact = load_descriptor_artifact(
                preparation.path,
                expected_descriptor_config=preparation.descriptor_config,
                expected_dataset_fingerprint=preparation.dataset_fingerprint,
            )
            if not artifact.valid_mask.all():
                raise RuntimeError(
                    "描述符 {0!r} 过滤了 {1} 行。严格 benchmark 的 split manifest 已绑定全量 sample_id；"
                    "请先修复输入 SMILES，不能静默缩小数据集。".format(
                        descriptor, int((~artifact.valid_mask).sum())
                    )
                )
            features[descriptor] = (artifact.X_smiles, None)
            statuses.append({
                "descriptor": descriptor, "status": preparation.status,
                "stage": "precompute", "artifact_path": str(preparation.path),
                "n_total": preparation.n_total, "n_valid": preparation.n_valid,
                "feature_dim": preparation.feature_dim, "skipped_model_count": 0,
                "reason": preparation.reason,
            })
            print("[descriptor:{0}] {1} -> {2}".format(preparation.status, descriptor, preparation.path))
        except Exception as exc:
            reason = "{0}: {1}".format(type(exc).__name__, exc)
            failures[descriptor] = reason
            statuses.append({
                "descriptor": descriptor, "status": "failed", "stage": "precompute",
                "artifact_path": str(artifact_path), "n_total": len(frame),
                "n_valid": None, "feature_dim": None,
                "skipped_model_count": len(config.models), "reason": reason,
            })
            print("[descriptor:failed] {0}: {1}".format(descriptor, reason), file=sys.stderr)
    layout.docs.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(statuses).to_csv(layout.docs / "descriptor_status.csv", index=False, encoding="utf-8-sig")
    print("[phase 2/2] folds consume artifacts or fold-local transformers by declared lifecycle")
    return features, failures


def _complete_metrics(rebuilt):
    complete = rebuilt.completeness.loc[rebuilt.completeness["is_complete"]].copy()
    fold_metrics = rebuilt.fold_metrics.copy()
    key_columns = ["split_id", "descriptor", "model"]
    if "evaluation_protocol" in complete.columns and "evaluation_protocol" in fold_metrics.columns:
        complete = complete.loc[complete["evaluation_protocol"].astype(str).eq("manifest_outer_cv")].copy()
        key_columns.insert(1, "evaluation_protocol")
    complete = complete.loc[:, key_columns]
    return fold_metrics.merge(complete, on=key_columns, how="inner")


def run_benchmark(
    config_path: Path | str,
    *,
    rerun_failed: bool = False,
    task_keys: list[str] | None = None,
    stale_seconds: float = 3600.0,
    bootstrap_n: int = 2000,
) -> int:
    """Run one already-validated schema-2 strict benchmark configuration.

    This function is called by the public ``yonod.py`` YAML path.  Keeping the
    orchestration here lets the script remain a thin developer CLI without
    creating a second configuration parser or changing the manifest/state
    protocol.
    """
    config = BenchmarkConfig.from_file(config_path)
    _require_parquet_engine()
    contract = create_benchmark_contract(config)
    write_run_manifest(contract)
    if config.split_manifest_path is None:
        split_manifest = create_split_manifest(contract)
        print("[split] generated a new manifest from the declared grouping strategy")
    else:
        frame_for_split = config.validate_dataset()
        split_manifest = load_external_split_manifest(
            config.split_manifest_path,
            sample_ids=frame_for_split[config.sample_id_col].astype(str).tolist(),
            dataset_sha256=contract.dataset_sha256,
            n_splits=int(config.cv["n_splits"]),
            n_repeats=int(config.cv["n_repeats"]),
            run_id=contract.run_id,
        )
        print("[split] imported and revalidated external manifest: {0}".format(config.split_manifest_path))
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
    interrupted = state.recover_stale_running(stale_seconds)
    print("[plan] run_id={0} theoretical_tasks={1} requeued={2} interrupted={3}".format(contract.run_id, len(specs), requeued, interrupted))

    frame = config.validate_dataset()
    sample_ids = frame[config.sample_id_col].astype(str).tolist()
    labels = frame[config.label_col].astype(float).to_numpy()
    model_params: Dict[str, Dict[str, Any]] = dict(config.legacy_model_kwargs)
    feature_cache, descriptor_failures = _prepare_feature_artifacts(config, frame, contract.run_dir)
    feature_sets = {str(item["name"]): item for item in config.feature_sets}
    while claimed := state.claim_next(rerun_failed=rerun_failed, task_keys=task_keys):
        try:
            if claimed.spec.descriptor in descriptor_failures:
                raise RuntimeError(
                    "描述符预计算失败，已跳过该描述符的模型任务：{0}".format(
                        descriptor_failures[claimed.spec.descriptor]
                    )
                )
            feature_set = feature_sets[claimed.spec.descriptor]
            X_smiles, X_numeric = feature_cache[claimed.spec.descriptor]
            component_frame = None
            fold_transformer = None
            if feature_set["kind"] == "fold_transform":
                if str(feature_set.get("algorithm", claimed.spec.descriptor)) != "ohe":
                    raise RuntimeError(
                        "尚未实现 fold_transform 特征集 {0!r}".format(
                            feature_set.get("algorithm", claimed.spec.descriptor)
                        )
                    )
                from yonod.benchmark.fold_preprocessors import ReactionComponentOHE
                component_frame = frame.loc[:, list(feature_set["component_cols"])].copy()
                fold_transformer = ReactionComponentOHE(feature_set["component_cols"])
            result = execute_fold(
                contract, split_manifest, sample_ids, X_smiles, labels,
                claimed.spec.descriptor, claimed.spec.model, claimed.spec.repeat, claimed.spec.fold,
                X_numeric=X_numeric, model_kwargs=model_params.get(claimed.spec.model, {}),
                component_frame=component_frame, fold_transformer=fold_transformer,
                schema2_model_config=(
                    config.model_configs.get(claimed.spec.model)
                    if claimed.spec.model != "autogluon" else None
                ),
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
    summary = summarize_combinations(complete_metrics, n_bootstrap=bootstrap_n, seed=int(config.cv["seed"]))
    model_comparisons, model_exclusions = paired_comparisons(
        complete_metrics, rebuilt.completeness, dimension="model", n_bootstrap=bootstrap_n, seed=int(config.cv["seed"]),
    )
    descriptor_comparisons, descriptor_exclusions = paired_comparisons(
        complete_metrics, rebuilt.completeness, dimension="descriptor", n_bootstrap=bootstrap_n, seed=int(config.cv["seed"]),
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


def main() -> int:
    args = _parse_args()
    return run_benchmark(
        args.config,
        rerun_failed=args.rerun_failed,
        task_keys=args.task_key,
        stale_seconds=args.stale_seconds,
        bootstrap_n=args.bootstrap_n,
    )


if __name__ == "__main__":
    raise SystemExit(main())
