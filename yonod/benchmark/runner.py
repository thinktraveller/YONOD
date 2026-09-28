"""Run the serial, resumable strict benchmark pipeline from a YAML contract."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import itertools
import json
import sys
from pathlib import Path
from typing import Any, Dict

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def _reject_incompatible_numeric_folds(config: BenchmarkConfig) -> None:
    """Prevent a resumed run from accepting folds made without declared inputs."""
    numeric_contract = config.raw["numeric_contract"]
    if not numeric_contract["columns"]:
        return
    folds_dir = resolve_benchmark_output_layout(config.outputs_root).folds
    for path in folds_dir.glob("*.json"):
        try:
            metadata = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"无法检查已有折级元数据：{path}") from exc
        features = metadata.get("feature_transformer") or {}
        numeric = features.get("numeric_conditions") if isinstance(features, dict) else None
        state = numeric.get("state") if isinstance(numeric, dict) else None
        if not isinstance(state, dict) or state.get("contract") != numeric_contract:
            raise RuntimeError(
                f"已有折级结果缺少匹配的声明式数值特征：{path}；请使用新的任务输出目录重新运行"
            )


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
    hpo = config.raw.get("hpo", {})
    if hpo.get("enabled"):
        from yonod.hpo.contracts import require_search_engine
        require_search_engine()
    from contextlib import nullcontext
    from yonod.hpo.budget import ActiveBudgetLedger
    budget_context = (
        ActiveBudgetLedger(
            config.outputs_root, {
                "hpo": hpo,
                "dataset_sha256": _sha256_file(config.dataset_path),
            },
            hpo["budget"]["task_timeout_s"],
        ) if hpo.get("enabled") and hpo.get("budget", {}).get("task_timeout_s") is not None
        else nullcontext(None)
    )
    with budget_context as task_budget:
        return _run_benchmark_impl(
            config, rerun_failed=rerun_failed, task_keys=task_keys,
            stale_seconds=stale_seconds, bootstrap_n=bootstrap_n,
            task_budget=task_budget,
        )


def _run_benchmark_impl(
    config: BenchmarkConfig, *, rerun_failed: bool,
    task_keys: list[str] | None, stale_seconds: float,
    bootstrap_n: int, task_budget: Any,
) -> int:
    _require_parquet_engine()
    contract = create_benchmark_contract(config)
    _reject_incompatible_numeric_folds(config)
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
    numeric_contract = config.raw["numeric_contract"]
    numeric_frame = (
        frame.loc[:, [entry["source"] for entry in numeric_contract["columns"]]].copy()
        if numeric_contract["columns"] else None
    )
    model_params: Dict[str, Dict[str, Any]] = dict(config.legacy_model_kwargs)
    feature_cache, descriptor_failures = _prepare_feature_artifacts(config, frame, contract.run_dir)
    feature_sets = {str(item["name"]): item for item in config.feature_sets}
    while True:
        if task_budget is not None and task_budget.remaining_s() <= 0:
            print("[hpo] task_timeout_s 已耗尽，剩余外层任务保持 pending", flush=True)
            break
        claimed = state.claim_next(rerun_failed=rerun_failed, task_keys=task_keys)
        if claimed is None:
            break
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
                numeric_frame=numeric_frame,
                numeric_contract=numeric_contract if numeric_frame is not None else None,
                hpo_raw=config.raw.get("hpo"),
                task_budget=task_budget,
            )
            state.mark_succeeded(claimed, result.prediction_path, result.metadata_path)
            print("[succeeded]", claimed.spec.task_key)
        except BaseException as exc:
            from yonod.hpo.budget import TaskBudgetExpired
            if isinstance(exc, TaskBudgetExpired):
                state.defer_claimed(claimed, str(exc))
                print("[hpo] task_timeout_s 已耗尽，当前外层任务保持 pending", flush=True)
                break
            state.mark_failed(claimed, exc)
            print("[failed] {0}: {1}".format(claimed.spec.task_key, exc), file=sys.stderr)
            if isinstance(exc, KeyboardInterrupt):
                raise

    final_status = None
    hpo = config.raw.get("hpo", {})
    if hpo.get("enabled") and hpo.get("final_model", {}).get("enabled"):
        from contextlib import nullcontext
        from yonod.hpo.budget import ActiveBudgetLedger, TaskBudgetExpired
        from yonod.hpo.finalize import fit_strict_final_model
        from yonod.hpo.storage import _atomic_json
        from yonod.hpo.strict import select_strict_outer_parameters

        expected = [(descriptor, model) for descriptor, model in itertools.product(config.descriptors, hpo["models"])]
        final_status = {"schema": "yonod-strict-final-status/v1", "expected": len(expected),
                        "completed": [], "failed": [], "pending": []}
        if any(state.summary()[key] for key in ("pending", "running", "failed", "interrupted")):
            final_status["pending"] = [f"{descriptor}__{model}" for descriptor, model in expected]
        else:
            for descriptor, model in expected:
                combination = f"{descriptor}__{model}"
                feature_set = feature_sets[descriptor]
                feature_matrix, _ = feature_cache[descriptor]
                categorical_frame = (frame.loc[:, list(feature_set["component_cols"])].copy()
                                     if feature_set["kind"] == "fold_transform" else None)
                if categorical_frame is not None:
                    from yonod.benchmark.fold_preprocessors import ReactionComponentOHE
                    ohe_factory = lambda columns=list(feature_set["component_cols"]): ReactionComponentOHE(columns)
                else:
                    ohe_factory = None
                final_timeout = hpo["final_model"]["budget"].get("task_timeout_s")
                final_context = (
                    ActiveBudgetLedger(
                        contract.run_dir,
                        {"run_id": contract.run_id, "purpose": "strict_final", "combination": combination},
                        final_timeout,
                        study_dir=contract.run_dir / "hpo" / "final_task_budget" / combination,
                    ) if final_timeout is not None else nullcontext(None)
                )
                try:
                    with final_context as final_budget:
                        if task_budget is not None:
                            task_budget.require_start("严格最终模型搜索")
                        selected = select_strict_outer_parameters(
                            output_root=contract.run_dir, sample_ids=sample_ids,
                            labels=labels, split_manifest=split_manifest,
                            repeat=0, fold=0, descriptor=descriptor, model=model,
                            model_config=config.model_configs[model], hpo_raw=hpo,
                            models=config.models,
                            grouping_strategy=str(config.grouping.get("strategy", "")),
                            feature_matrix=feature_matrix,
                            categorical_frame=categorical_frame, ohe_factory=ohe_factory,
                            numeric_frame=numeric_frame,
                            numeric_contract=numeric_contract if numeric_frame is not None else None,
                            source_frame=frame, task_budget=task_budget,
                            final_budget=final_budget, purpose="final_model",
                        )
                        if task_budget is not None:
                            task_budget.require_start("严格完整开发集最终重训")
                        if final_budget is not None:
                            final_budget.require_start("严格完整开发集最终重训")
                        final_info = fit_strict_final_model(
                            contract=contract, frame=frame, feature_set=feature_set,
                            feature_matrix=feature_matrix, numeric_frame=numeric_frame,
                            numeric_contract=numeric_contract, model=model,
                            search_audit=selected,
                        )
                    final_status["completed"].append({"combination": combination, **final_info})
                    print("[hpo] final_model complete", combination, flush=True)
                except TaskBudgetExpired as exc:
                    final_status["pending"].append(combination)
                    print(f"[hpo] final_model pending {combination}: {exc}", flush=True)
                    final_status["pending"].extend(
                        f"{other_descriptor}__{other_model}" for other_descriptor, other_model in expected
                        if f"{other_descriptor}__{other_model}" not in [
                            item["combination"] for item in final_status["completed"]
                        ] and f"{other_descriptor}__{other_model}" not in final_status["pending"]
                    )
                    break
                except Exception as exc:
                    final_status["failed"].append({"combination": combination,
                                                   "reason": f"{type(exc).__name__}: {exc}"})
                    print(f"[hpo] final_model failed {combination}: {exc}", file=sys.stderr)
        _atomic_json(contract.run_dir / "final_models" / "final_status.json", final_status)

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
    return 0 if (counts["failed"] == 0 and counts["interrupted"] == 0 and counts["pending"] == 0 and
                 (final_status is None or (not final_status["failed"] and not final_status["pending"]))) else 2


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
