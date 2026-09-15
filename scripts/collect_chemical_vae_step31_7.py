#!/usr/bin/env python
"""Collect the bounded, paired evidence for Chemical VAE step 31.7.

This is deliberately a read-mostly verifier: modelling is performed only by
the saved schema-2 YAML configurations through ``yonod.py``.  The collector
re-reads their hash-verified artifacts and training outputs, proves the
strict-all-role common subset and split alignment, then writes small derived
summaries without changing model artifacts or source data.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import re
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

import numpy as np
import pandas as pd
import yaml
from scipy.stats import kendalltau

# ``python scripts/<tool>.py`` makes ``scripts/`` the first import location;
# retain the documented repository-root invocation without depending on an
# externally configured PYTHONPATH.
_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from yonod.artifacts.reader import FeatureArtifact, load_feature_artifact


EXPECTED_FEATURES = frozenset({"chemical-vae-zinc-v5", "morgan-ecfp4", "mfp-r3-count"})
TIMED_FEATURES = {
    "chemical-vae-zinc-v5": "chemical_vae",
    "morgan-ecfp4": "morgan",
    "mfp-r3-count": "mfp",
}
_REAL_RE = re.compile(r"^real\t(?P<minutes>\d+)m(?P<seconds>[0-9.]+)s$", re.MULTILINE)


class ComparisonEvidenceError(ValueError):
    """Raised when the purported comparison no longer meets its frozen scope."""


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="\n", dir=path.parent,
        prefix=f".{path.name}.", suffix=".tmp", delete=False,
    ) as handle:
        temporary = Path(handle.name)
        handle.write(content)
    try:
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    _atomic_text(path, json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def _atomic_csv(path: Path, rows: Iterable[Mapping[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="", dir=path.parent,
        prefix=f".{path.name}.", suffix=".tmp", delete=False,
    ) as handle:
        temporary = Path(handle.name)
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    try:
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _only_manifest(root: Path, feature_id: str) -> Path:
    matches = sorted(root.glob("features/*/manifest.yaml"))
    selected = [path for path in matches if yaml.safe_load(path.read_text(encoding="utf-8")).get("feature_id") == feature_id]
    if len(selected) != 1:
        raise ComparisonEvidenceError(f"{feature_id} 应恰有一个 feature manifest，实际为 {len(selected)}")
    return selected[0]


def _load_artifacts(artifacts_root: Path) -> dict[str, FeatureArtifact]:
    result: dict[str, FeatureArtifact] = {}
    for path in sorted(artifacts_root.glob("features/*/manifest.yaml")):
        artifact = load_feature_artifact(path)
        feature_id = str(artifact.manifest["feature_id"])
        if feature_id in result:
            raise ComparisonEvidenceError(f"重复 feature_id：{feature_id}")
        result[feature_id] = artifact
    if frozenset(result) != EXPECTED_FEATURES:
        raise ComparisonEvidenceError(f"比较特征集合不匹配：{sorted(result)}")
    return result


def _load_runs(outputs_root: Path) -> dict[str, tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, Path]]:
    result: dict[str, tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, Path]] = {}
    for manifest_path in sorted(outputs_root.glob("runs/*/run_manifest.yaml")):
        manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
        feature_id = str(manifest.get("feature_id"))
        if feature_id in result:
            raise ComparisonEvidenceError(f"重复训练结果 feature_id：{feature_id}")
        run_dir = manifest_path.parent
        predictions = pd.read_csv(run_dir / "predictions.csv").sort_values(
            ["repeat", "fold", "sample_id"], kind="stable"
        ).reset_index(drop=True)
        folds = pd.read_csv(run_dir / "fold_metrics.csv").sort_values(
            ["repeat", "fold"], kind="stable"
        ).reset_index(drop=True)
        result[feature_id] = (manifest, predictions, folds, run_dir)
    if frozenset(result) != EXPECTED_FEATURES:
        raise ComparisonEvidenceError(f"比较训练结果集合不匹配：{sorted(result)}")
    return result


def _strict_common_subset(
    artifacts: Mapping[str, FeatureArtifact], fixture: pd.DataFrame
) -> list[dict[str, str]]:
    expected_ids = fixture["reaction_id"].astype(str).to_numpy(dtype=np.str_)
    records: list[dict[str, str]] = []
    for feature_id, artifact in artifacts.items():
        if not np.array_equal(artifact.sample_ids, expected_ids):
            raise ComparisonEvidenceError(f"{feature_id} sample_id 与 fixture 不一致")
        if not artifact.valid_mask.all():
            raise ComparisonEvidenceError(f"{feature_id} 含无效行，不能作为严格共同子集")
        if artifact.matrix.shape[0] != len(expected_ids):
            raise ComparisonEvidenceError(f"{feature_id} feature matrix 未覆盖共同样本")

    vae = artifacts["chemical-vae-zinc-v5"]
    if vae.diagnostics is None:
        raise ComparisonEvidenceError("Chemical VAE artifact 缺少诊断 sidecar")
    rows = vae.diagnostics.get("rows")
    if not isinstance(rows, list) or len(rows) != len(expected_ids):
        raise ComparisonEvidenceError("Chemical VAE diagnostics 行数不等于 fixture")
    for fixture_row, diagnostic in zip(fixture.itertuples(index=False), rows):
        sample_id = str(getattr(fixture_row, "reaction_id"))
        if diagnostic.get("sample_id") != sample_id:
            raise ComparisonEvidenceError("Chemical VAE diagnostic sample_id 顺序不一致")
        roles = diagnostic.get("roles")
        if not isinstance(roles, Mapping):
            raise ComparisonEvidenceError("Chemical VAE diagnostic 缺少角色记录")
        for column in ("reactant_1_smiles", "reactant_2_smiles"):
            record = roles.get(column)
            if not isinstance(record, Mapping) or record.get("status") != "encoded":
                raise ComparisonEvidenceError(f"{sample_id}/{column} 不是严格可编码输入")
            raw_value = str(getattr(fixture_row, column))
            if record.get("input_text") != raw_value or record.get("preprocessed_text") != raw_value:
                raise ComparisonEvidenceError(f"{sample_id}/{column} 不满足 identity 输入契约")
        records.append({
            "sample_id": sample_id,
            "chemical_vae_reactant_1": "encoded",
            "chemical_vae_reactant_2": "encoded",
            "strict_all_roles_encodable": "true",
        })
    return records


def _parse_wall_seconds(path: Path) -> float:
    content = path.read_text(encoding="utf-8")
    match = _REAL_RE.search(content)
    if match is None:
        raise ComparisonEvidenceError(f"计时日志缺少 POSIX time real 行：{path}")
    return int(match.group("minutes")) * 60 + float(match.group("seconds"))


def _feature_status(path: Path, feature_id: str) -> str:
    content = path.read_text(encoding="utf-8")
    match = re.search(rf"- {re.escape(feature_id)}: (?P<status>ready|reused) ", content)
    if match is None:
        raise ComparisonEvidenceError(f"日志中没有 {feature_id} 的 ready/reused 状态：{path}")
    return str(match.group("status"))


def _timing_evidence(logs_root: Path) -> dict[str, Any]:
    records: dict[str, Any] = {}
    for feature_id, stem in TIMED_FEATURES.items():
        cold_log = logs_root / f"step31_7_timing_{stem}_cold.log"
        warm_log = logs_root / f"step31_7_timing_{stem}_warm.log"
        cold_status = _feature_status(cold_log, feature_id)
        warm_status = _feature_status(warm_log, feature_id)
        if cold_status != "ready" or warm_status != "reused":
            raise ComparisonEvidenceError(
                f"{feature_id} 计时 cache 状态不正确：cold={cold_status}, warm={warm_status}"
            )
        records[feature_id] = {
            "scope": "sequential end-to-end yonod.py stage:features wall time, including process import, YAML validation, artifact I/O and cache validation",
            "cold_status": cold_status,
            "cold_wall_seconds": _parse_wall_seconds(cold_log),
            "warm_status": warm_status,
            "warm_wall_seconds": _parse_wall_seconds(warm_log),
            "cold_log": str(cold_log.relative_to(logs_root.parent)),
            "warm_log": str(warm_log.relative_to(logs_root.parent)),
        }
    return records


def _metric_records(
    artifacts: Mapping[str, FeatureArtifact],
    runs: Mapping[str, tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, Path]],
    repository_root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    paired_reference: pd.DataFrame | None = None
    shared: dict[str, Any] | None = None
    for feature_id in sorted(runs):
        manifest, predictions, folds, run_dir = runs[feature_id]
        columns = ["repeat", "fold", "sample_id", "y_true"]
        current_alignment = predictions.loc[:, columns]
        if paired_reference is None:
            paired_reference = current_alignment
            shared = {
                "dataset_identity": manifest["dataset_identity"],
                "label_identity": manifest["label_identity"],
                "split_identity": manifest["split_identity"],
                "evaluation": manifest["evaluation"],
            }
        else:
            pd.testing.assert_frame_equal(paired_reference, current_alignment, check_exact=True)
            assert shared is not None
            for field in ("dataset_identity", "label_identity", "split_identity", "evaluation"):
                if manifest[field] != shared[field]:
                    raise ComparisonEvidenceError(f"{feature_id} 的 {field} 与配对基准不一致")
        y_true = predictions["y_true"].to_numpy(dtype=float)
        y_pred = predictions["y_pred"].to_numpy(dtype=float)
        if not np.isfinite(y_pred).all():
            raise ComparisonEvidenceError(f"{feature_id} OOF 预测包含非有限值")
        tau = kendalltau(y_true, y_pred)
        mae = float(np.mean(np.abs(y_true - y_pred)))
        rmse = float(math.sqrt(float(np.mean((y_true - y_pred) ** 2))))
        centered = y_true - float(np.mean(y_true))
        r2 = float(1.0 - float(np.sum((y_true - y_pred) ** 2)) / float(np.sum(centered ** 2)))
        artifact = artifacts[feature_id]
        summary.append({
            "feature_id": feature_id,
            "artifact_id": artifact.manifest["artifact_id"],
            "run_id": manifest["run_id"],
            "feature_dim": int(artifact.matrix.shape[1]),
            "n_strict_common_samples": int(len(y_true)),
            "coverage_fraction": 1.0,
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
            "kendall_tau": float(tau.statistic),
            "kendall_tau_pvalue": float(tau.pvalue),
            "oof_predictions": str((run_dir / "predictions.csv").relative_to(repository_root)),
            "fold_metrics": str((run_dir / "fold_metrics.csv").relative_to(repository_root)),
        })
        for row in folds.to_dict(orient="records"):
            fold_rows.append({"feature_id": feature_id, "run_id": manifest["run_id"], **row})
    if shared is None:
        raise ComparisonEvidenceError("没有可比较训练结果")
    return summary, fold_rows, shared


def _platform_evidence(repository_root: Path) -> dict[str, Any]:
    import torch

    records: dict[str, Any] = {
        "linux_cpu": {
            "status": "passed",
            "host_platform": platform.platform(),
            "python": sys.version.split()[0],
            "cpu_count": os.cpu_count(),
            "torch": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "evidence": "real stage:all VAE/Morgan/MFP × RF and sequential CPU feature timing logs",
        },
        "windows_cpu": {
            "status": "not_verified",
            "reason": "No Windows execution host was available in this task; Linux path tests are not a Windows smoke result.",
            "handoff_config": "configs/chemical_vae/step31_9_zinc_v5_windows_cpu_all.yaml",
            "handoff_procedure": "project-docs/chemical-vae-platform-handoff.md",
        },
        "gpu": {
            "status": "not_run",
            "reason": "The frozen comparison configured chemical_vae.device=cpu; no GPU claim is made.",
        },
        "zinc_properties": {
            "status": "not_supported",
            "reason": "Only the C03-verified ZINC/v5 encoder is selectable; zinc_properties has no independent verification asset in this scope.",
        },
    }
    c11_path = repository_root / "derived" / "chemical_vae" / "step31_9_gpu_smoke" / "verification.json"
    if not c11_path.is_file():
        return records
    try:
        c11 = json.loads(c11_path.read_text(encoding="utf-8"))
        if c11.get("schema_version") != "chemical_vae_step31_9_c11/v1":
            raise ComparisonEvidenceError("C11 verification schema is unexpected")
        gpu = c11.get("zinc_v5_gpu", {})
        properties = c11.get("zinc_properties_encoder", {})
        if gpu.get("status") != "passed" or properties.get("status") != "passed_encoder_scope":
            raise ComparisonEvidenceError("C11 verification does not contain passed GPU/property-encoder evidence")
    except (OSError, ValueError, json.JSONDecodeError, ComparisonEvidenceError) as exc:
        raise ComparisonEvidenceError(f"C11 verification cannot be used: {exc}") from exc
    records["gpu"] = {
        "status": "passed",
        "reason": "Real cuda:0 ZINC/v5 VAE×RF all-stage smoke completed through the documented headless yonod.py entrypoint.",
        "evidence": str(c11_path.relative_to(repository_root)),
    }
    records["zinc_properties"] = {
        "status": "passed_encoder_scope",
        "reason": "The separately hashed zinc_properties encoder passed direct-HDF5 TensorFlow and independent NumPy parity, then a cuda:0 VAE×RF smoke. Its property head/decoder/training are explicitly out of scope.",
        "evidence": str(c11_path.relative_to(repository_root)),
    }
    return records


def collect(repository_root: Path) -> dict[str, Path]:
    comparison_root = repository_root / "derived" / "chemical_vae" / "step31_7_comparison"
    fixture_path = repository_root / "dataset" / "benchmark_smoke_fixture.csv"
    fixture = pd.read_csv(fixture_path)
    if len(fixture) != 12:
        raise ComparisonEvidenceError(f"冻结 fixture 行数应为 12，实际为 {len(fixture)}")
    artifacts = _load_artifacts(comparison_root / "comparison_artifacts")
    strict_rows = _strict_common_subset(artifacts, fixture)
    runs = _load_runs(comparison_root / "comparison_outputs")
    summary, fold_rows, shared = _metric_records(artifacts, runs, repository_root)
    timings = _timing_evidence(repository_root / "logs")
    platform_records = _platform_evidence(repository_root)

    summary_path = comparison_root / "metrics_summary.csv"
    folds_path = comparison_root / "paired_fold_metrics.csv"
    subset_path = comparison_root / "strict_common_subset.csv"
    report_path = comparison_root / "comparison_report.json"
    index_path = repository_root / "derived" / "chemical_vae" / "step31_acceptance" / "index.json"
    _atomic_csv(summary_path, summary, list(summary[0]))
    _atomic_csv(folds_path, fold_rows, list(fold_rows[0]))
    _atomic_csv(subset_path, strict_rows, list(strict_rows[0]))

    comparison_log = repository_root / "logs" / "step31_7_comparison_all.log"
    report = {
        "schema_version": "chemical_vae_step31_7_comparison/v1",
        "status": "passed_with_bounded_scope",
        "scope": {
            "fixture": str(fixture_path.relative_to(repository_root)),
            "fixture_sha256": _sha256(fixture_path),
            "selection_rule": "all 12 fixture rows have both selected reactant roles marked encoded by the actual Chemical VAE diagnostics; Morgan and MFP validity masks also cover all 12 rows",
            "n_strict_common_samples": 12,
            "roles": ["reactant_1_smiles", "reactant_2_smiles"],
            "rf": {"n_estimators": 16, "max_features": 1.0, "min_samples_leaf": 1, "n_jobs": 1, "random_state": 42},
        },
        "paired_alignment": {**shared, "status": "passed", "checks": ["same sample_id/repeat/fold/y_true OOF rows", "same dataset/label/split identities", "all static feature masks valid"]},
        "metrics": summary,
        "timing": timings,
        "platform": platform_records,
        "execution": {
            "entrypoint": "yonod.py",
            "comparison_config": "configs/chemical_vae/step31_7_comparison_all.yaml",
            "comparison_log": str(comparison_log.relative_to(repository_root)),
            "comparison_pid_file": "logs/step31_7_comparison_all.pid",
            "headless_command_template": "nohup setsid bash -lc '<activate yonod; printf YAML | python -u yonod.py>' </dev/null > task.log 2>&1 &",
            "timing_method": "Sequential fresh Python processes, POSIX time -p around the public yonod.py stage:features call; warm invocation reused the hash-verified artifact.",
        },
        "limitations": [
            "This is a 12-sample, two-fold engineering smoke. It is not a statistically powered performance comparison or a scientific claim.",
            "The recorded feature timing is end-to-end public-entry wall time, not an isolated encoder kernel benchmark.",
            "Windows CPU did not run and remains not_verified; Linux GPU evidence is recorded separately in the C11 verification.",
        ],
        "outputs": {
            "metrics_summary": str(summary_path.relative_to(repository_root)),
            "paired_fold_metrics": str(folds_path.relative_to(repository_root)),
            "strict_common_subset": str(subset_path.relative_to(repository_root)),
            "per_descriptor_oof_and_fold_files": {row["feature_id"]: {"oof": row["oof_predictions"], "folds": row["fold_metrics"]} for row in summary},
        },
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    _atomic_json(report_path, report)
    _atomic_json(index_path, {
        "schema_version": "chemical_vae_step31_acceptance_index/v4",
        "overall_status": "complete",
        "completion_rule": "The normal completion rule requires actual evidence for every C01-C12 item. On 2026-09-14, the user explicitly authorized closing step 31 while deferring the unavailable Windows CPU execution. This records project completion by user decision; it does not turn the deferred Windows evidence into a passed platform validation.",
        "completion_authorization": {
            "status": "user_authorized_deferred_validation",
            "decision_date": "2026-09-14",
            "deferred_acceptance": ["C11.windows_cpu"],
            "follow_up": "Run the documented Windows CPU handoff and replace the deferred state with real platform evidence when a Windows host is available.",
        },
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "acceptance": {
            "C01": {
                "status": "passed",
                "evidence": [
                    "derived/chemical_vae/step31_1_audit/environment_snapshot.json",
                    "derived/chemical_vae/step31_1_audit/asset_manifest.json",
                    "derived/chemical_vae/step31_1_audit/backend_decision.md",
                ],
                "statement": "Linux environment, ZINC source/converted assets and dependency decision are recorded; runtime remains PyTorch and TensorFlow is isolated as parity evidence.",
            },
            "C02": {
                "status": "passed",
                "evidence": [
                    "derived/chemical_vae/step31_2_coverage/coverage_manifest.json",
                    "derived/chemical_vae/step31_2_coverage/sample_role_diagnostics.csv",
                ],
                "statement": "Role-level coverage, denominators and reaction retention are recorded without claiming preflight as inference success.",
            },
            "C03": {
                "status": "passed",
                "evidence": [
                    "WEIGHTS/chemical_vae/zinc-37e96cd3bc8f9680/v5/conversion_manifest.json",
                    "derived/chemical_vae/step31_3_parity/zinc-37e96cd3bc8f9680-f9202cd6fa2e/parity_protocol.json",
                    "derived/chemical_vae/step31_3_parity/zinc-37e96cd3bc8f9680-f9202cd6fa2e/parity_report.json",
                ],
                "statement": "The verified ZINC/v5 PyTorch state passes frozen direct-HDF5 TensorFlow and independent NumPy reference checks. zinc_properties has separate, non-inherited C11 evidence.",
            },
            "C04": {
                "status": "passed",
                "evidence": [
                    "yonod/descriptors/chemical_vae.py",
                    "tests/test_chemical_vae_descriptor.py",
                ],
                "statement": "The registered static adapter preserves identity text, z_mean output, batch/dedup behavior and failure masks under frozen v5 contracts for explicitly verified encoder sources only.",
            },
            "C05": {
                "status": "passed",
                "evidence": [
                    "tests/test_chemical_vae_descriptor.py",
                    "tests/test_chemical_vae_training_isolation.py",
                ],
                "statement": "Broken VAE assets isolate to their feature candidate, Morgan can continue, and train-only manifest consumption does not import the encoder.",
            },
            "C06": {
                "status": "passed",
                "evidence": [
                    "yonod/pipeline/features.py",
                    "tests/test_chemical_vae_artifact_lineage.py",
                ],
                "statement": "Conversion manifest/state bytes are included before cache lookup; same-path content replacement produces a new feature identity.",
            },
            "C07": {
                "status": "passed",
                "evidence": [
                    "yonod/artifacts/contracts.py",
                    "yonod/artifacts/operations.py",
                    "tests/test_chemical_vae_artifact_lineage.py",
                ],
                "statement": "Full-row diagnostics sidecars are hash checked and remain sample-aligned after supported derivations.",
            },
            "C08": {
                "status": "passed",
                "evidence": [
                    "derived/chemical_vae/step31_6_smoke/verification.json",
                    "tests/test_chemical_vae_training_isolation.py",
                ],
                "statement": "Separate features -> train and all processes produce aligned artifacts/splits/predictions while train reads the manifest only.",
            },
            "C09": {
                "status": "passed",
                "evidence": ["derived/chemical_vae/step31_6_smoke/verification.json"],
                "statement": "Real Chemical VAE × RF and the retained Morgan × RF smoke both completed through yonod.py.",
            },
            "C10": {
                "status": "passed_with_bounded_scope",
                "evidence": [str(report_path.relative_to(repository_root))],
                "statement": "VAE, Morgan and MFP use the same 12 strict-all-role samples, label identity, outer split identity and RF declaration; OOF/fold evidence and sequential cold/warm timing are saved. This is not a powered research comparison.",
            },
            "C11": {
                "status": "deferred_by_user",
                "evidence": [
                    str(report_path.relative_to(repository_root)),
                    "derived/chemical_vae/step31_9_gpu_smoke/verification.json",
                    "project-docs/chemical-vae-platform-handoff.md",
                ],
                "linux_cpu": "passed",
                "windows_cpu": "not_verified",
                "gpu": platform_records["gpu"]["status"],
                "zinc_properties": platform_records["zinc_properties"]["status"],
                "blocker": "No genuine Windows CPU execution host is available. The documented portable YAML and Linux checks are a handoff, not Windows evidence; the user authorized deferring this validation when closing step 31.",
            },
            "C12": {
                "status": "passed",
                "evidence": [
                    "README.md",
                    "example.md",
                    "project-docs/chemical-vae-reference.md",
                    "project-docs/chemical-vae-platform-handoff.md",
                    "configs/chemical_vae/step31_9_zinc_v5_gpu_all.yaml",
                    "configs/chemical_vae/step31_9_zinc_properties_gpu_all.yaml",
                    "configs/chemical_vae/step31_9_zinc_v5_windows_cpu_all.yaml",
                    "requirements.txt",
                    "requirements-chemical-vae-parity.txt",
                    "tests/test_chemical_vae_documentation.py",
                ],
                "statement": "Current YAML examples, asset/runtime contract, parity-only TensorFlow dependency, limitations and the complete acceptance map are cross-checked by a non-modelling documentation test.",
            },
        },
    })
    return {
        "report": report_path,
        "metrics": summary_path,
        "fold_metrics": folds_path,
        "subset": subset_path,
        "acceptance_index": index_path,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=_REPOSITORY_ROOT)
    args = parser.parse_args()
    root = args.repository_root.resolve()
    paths = collect(root)
    print(json.dumps({key: str(value) for key, value in paths.items()}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
