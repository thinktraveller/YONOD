#!/usr/bin/env python3
"""Verify and record the real C11 GPU/extra-encoder evidence for step 31.

This collector never launches a model.  It verifies the artifacts and logs
created by the two dedicated schema-2 ``yonod.py`` jobs, and reads the frozen
``zinc_properties`` direct-HDF5 parity evidence.  The output deliberately
keeps Windows CPU ``not_verified`` until a genuine Windows host supplies the
handoff evidence documented in ``project-docs/chemical-vae-platform-handoff``.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from yonod.artifacts.reader import FeatureArtifact, load_feature_artifact


class EvidenceError(ValueError):
    """Raised when a claimed C11 result is absent or internally inconsistent."""


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    try:
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _exactly_one(root: Path, pattern: str, label: str) -> Path:
    matches = sorted(root.glob(pattern))
    if len(matches) != 1:
        raise EvidenceError(f"{label} expected one {pattern}, found {len(matches)}")
    return matches[0]


def _relative(root: Path, path: Path) -> str:
    return str(path.resolve().relative_to(root.resolve()))


def _completed_log(root: Path, name: str) -> dict[str, str]:
    log = root / "logs" / f"{name}.log"
    pid = root / "logs" / f"{name}.pid"
    if not log.is_file() or not pid.is_file():
        raise EvidenceError(f"missing task log/PID for {name}")
    content = log.read_text(encoding="utf-8")
    for required in ("YAML配置文件", "[all] features=ready", "× rf: complete", "[完成] 建模任务已成功完成!"):
        if required not in content:
            raise EvidenceError(f"{name} log lacks successful yonod.py marker: {required}")
    try:
        pid_value = int(pid.read_text(encoding="utf-8").strip())
    except ValueError as exc:
        raise EvidenceError(f"invalid PID file {pid}") from exc
    return {
        "log": _relative(root, log),
        "pid_file": _relative(root, pid),
        "pid": str(pid_value),
        "completion": "logged_complete; PID was checked after completion and had exited",
    }


def _feature(root: Path, artifact_root: Path, expected_feature_id: str, config_path: Path) -> tuple[FeatureArtifact, dict[str, Any]]:
    manifest_path = _exactly_one(artifact_root, "features/*/manifest.yaml", expected_feature_id)
    artifact = load_feature_artifact(manifest_path)
    if artifact.manifest.get("feature_id") != expected_feature_id:
        raise EvidenceError(f"unexpected feature_id in {manifest_path}")
    if list(artifact.matrix.shape) != [12, 392] or int(artifact.valid_mask.sum()) != 12:
        raise EvidenceError(f"{expected_feature_id} does not have the required 12x392 all-valid artifact")
    rows = artifact.diagnostics.get("rows") if artifact.diagnostics else None
    if not isinstance(rows, list) or len(rows) != 12:
        raise EvidenceError(f"{expected_feature_id} diagnostics sidecar is absent or not full-row")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    params = config["descriptors"][0]["params"]
    if params.get("device") != "cuda:0" or params.get("backend") != "pytorch":
        raise EvidenceError(f"{config_path} is not the required cuda:0 PyTorch configuration")
    return artifact, {
        "config": _relative(root, config_path),
        "feature_manifest": _relative(root, manifest_path),
        "artifact_id": artifact.manifest["artifact_id"],
        "feature_id": expected_feature_id,
        "shape": list(artifact.matrix.shape),
        "valid_rows": int(artifact.valid_mask.sum()),
        "diagnostic_rows": len(rows),
        "device_requested_in_schema": params["device"],
        "model_manifest": params["model_manifest"],
    }


def _run(root: Path, output_root: Path, expected_feature_id: str) -> dict[str, Any]:
    manifest_path = _exactly_one(output_root, "runs/*/run_manifest.yaml", expected_feature_id)
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete" or manifest.get("feature_id") != expected_feature_id:
        raise EvidenceError(f"{manifest_path} is not a complete run for {expected_feature_id}")
    return {
        "run_manifest": _relative(root, manifest_path),
        "run_id": manifest["run_id"],
        "model": manifest["model"],
        "n_samples": manifest["n_samples"],
        "n_folds": manifest["n_folds"],
        "split_identity": manifest["split_identity"],
        "predictions": _relative(root, manifest_path.parent / manifest["outputs"]["predictions"]),
        "fold_metrics": _relative(root, manifest_path.parent / manifest["outputs"]["fold_metrics"]),
    }


def _cuda_audit() -> dict[str, Any]:
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise EvidenceError("GPU smoke artifacts exist but current PyTorch cannot audit CUDA availability")
    records = []
    for index in range(torch.cuda.device_count()):
        properties = torch.cuda.get_device_properties(index)
        records.append({
            "index": index,
            "name": torch.cuda.get_device_name(index),
            "total_memory_bytes": int(properties.total_memory),
        })
    nvidia_smi: dict[str, Any]
    executable = shutil.which("nvidia-smi")
    if executable is None:
        nvidia_smi = {"status": "not_found"}
    else:
        result = subprocess.run(
            [executable, "--query-gpu=index,name,driver_version,memory.total", "--format=csv,noheader"],
            text=True, capture_output=True, check=False,
        )
        nvidia_smi = {
            "status": "ok" if result.returncode == 0 else "failed",
            "command": "nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv,noheader",
            "returncode": result.returncode,
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip(),
        }
    return {
        "torch": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cuda_available": True,
        "device_count": torch.cuda.device_count(),
        "devices": records,
        "nvidia_smi": nvidia_smi,
    }


def _properties_parity(root: Path) -> dict[str, Any]:
    report_path = _exactly_one(
        root / "derived" / "chemical_vae" / "step31_3_parity",
        "zinc_properties-*/parity_report.json",
        "zinc_properties parity report",
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    protocol_path = report_path.parent / str(report.get("protocol_path", ""))
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if report.get("status") != "passed" or report.get("numerical_parity_verified") is not True:
        raise EvidenceError("zinc_properties direct-HDF5 parity did not pass")
    if protocol.get("atol") != 2.0e-5 or protocol.get("rtol") != 2.0e-5:
        raise EvidenceError("zinc_properties parity tolerance differs from the frozen 2e-5 protocol")
    state_path = root / "WEIGHTS" / "chemical_vae" / "zinc_properties-9f923d03c5ce558d" / "v5" / "encoder_state_dict.pt"
    manifest_path = state_path.with_name("conversion_manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_source = "9f923d03c5ce558d1f0cb52d2a2f4e28f080e59ac5a8c81a4365d7b63ba9d1e2"
    if manifest.get("asset", {}).get("source_encoder_sha256") != expected_source:
        raise EvidenceError("zinc_properties conversion manifest source hash is unexpected")
    return {
        "status": "passed_encoder_scope",
        "source_encoder_sha256": expected_source,
        "conversion_manifest": _relative(root, manifest_path),
        "parity_protocol": _relative(root, protocol_path),
        "parity_report": _relative(root, report_path),
        "converter_version": manifest.get("converter_version"),
        "atol": protocol["atol"],
        "rtol": protocol["rtol"],
        "tensorflow_vector_max_abs_error_batch_8": report["candidate_vs_tensorflow_by_batch"]["8"]["max_absolute_error"],
        "tensorflow_layer_max_abs_error": max(
            value["max_absolute_error"] for value in report["candidate_vs_tensorflow_layers"].values()
        ),
        "scope_limit": "z_mean encoder only; zinc_prop_pred.h5/property head, decoder, training and generation remain outside this descriptor boundary",
    }


def collect(root: Path) -> Path:
    zinc_config = root / "configs" / "chemical_vae" / "step31_9_zinc_v5_gpu_all.yaml"
    properties_config = root / "configs" / "chemical_vae" / "step31_9_zinc_properties_gpu_all.yaml"
    zinc, zinc_feature = _feature(
        root,
        root / "derived" / "chemical_vae" / "step31_9_gpu_smoke" / "zinc_v5_artifacts",
        "chemical-vae-zinc-v5-gpu",
        zinc_config,
    )
    properties, properties_feature = _feature(
        root,
        root / "derived" / "chemical_vae" / "step31_9_gpu_smoke" / "zinc_properties_artifacts",
        "chemical-vae-zinc-properties-v5-gpu",
        properties_config,
    )
    cpu_manifest = _exactly_one(
        root / "derived" / "chemical_vae" / "step31_6_smoke" / "all_artifacts",
        "features/*/manifest.yaml",
        "step31_6 CPU feature",
    )
    cpu = load_feature_artifact(cpu_manifest)
    if list(cpu.matrix.shape) != [12, 392] or not np.array_equal(zinc.sample_ids, cpu.sample_ids):
        raise EvidenceError("GPU ZINC artifact is not sample-aligned with the retained CPU smoke artifact")
    if not np.array_equal(zinc.valid_mask, cpu.valid_mask):
        raise EvidenceError("GPU ZINC validity mask differs from the retained CPU smoke artifact")
    max_abs = float(np.abs(zinc.matrix - cpu.matrix).max())

    report = {
        "schema_version": "chemical_vae_step31_9_c11/v1",
        "status": "partial",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "linux_host": {"platform": platform.platform(), "python": sys.version.split()[0]},
        "linux_gpu_audit": _cuda_audit(),
        "zinc_v5_gpu": {
            "status": "passed",
            "entrypoint": "yonod.py headless nohup/setsid job",
            "task": _completed_log(root, "step31_9_zinc_v5_gpu_all"),
            "feature": zinc_feature,
            "run": _run(root, root / "derived" / "chemical_vae" / "step31_9_gpu_smoke" / "zinc_v5_outputs", "chemical-vae-zinc-v5-gpu"),
            "cpu_alignment": {
                "cpu_feature_manifest": _relative(root, cpu_manifest),
                "sample_ids_equal": True,
                "valid_masks_equal": True,
                "max_abs_difference": max_abs,
                "interpretation": "The direct-HDF5 C03 gate is frozen CPU float32 at 2e-5. This post-hoc GPU-vs-CPU execution comparison is recorded, not relabelled as that C03 numerical-parity gate.",
            },
        },
        "zinc_properties_encoder": {
            **_properties_parity(root),
            "gpu_smoke": {
                "entrypoint": "yonod.py headless nohup/setsid job",
                "task": _completed_log(root, "step31_9_zinc_properties_gpu_all"),
                "feature": properties_feature,
                "run": _run(root, root / "derived" / "chemical_vae" / "step31_9_gpu_smoke" / "zinc_properties_outputs", "chemical-vae-zinc-properties-v5-gpu"),
            },
        },
        "windows_cpu": {
            "status": "not_verified",
            "reason": "No genuine Windows execution host is available in this task. Linux checks and the portable YAML are not platform evidence.",
            "handoff_config": "configs/chemical_vae/step31_9_zinc_v5_windows_cpu_all.yaml",
            "handoff_procedure": "project-docs/chemical-vae-platform-handoff.md",
        },
        "c11_conclusion": "partial: Linux CPU/GPU and both separately verified encoder scopes have real evidence; Windows CPU remains not_verified.",
    }
    output = root / "derived" / "chemical_vae" / "step31_9_gpu_smoke" / "verification.json"
    _atomic_json(output, report)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=ROOT)
    args = parser.parse_args()
    path = collect(args.repository_root.resolve())
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
