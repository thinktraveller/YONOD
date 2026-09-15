#!/usr/bin/env python3
"""Convert an audited Chemical VAE encoder and record numerical parity.

This is a step-31.3 tool, not a public feature-generation entry point.  It
persists a CPU PyTorch encoder state dict under ``WEIGHTS/chemical_vae`` and
compares that persisted candidate against both a direct TensorFlow/Keras
execution of the original HDF5 graph and an independently implemented NumPy
forward reference which reads the original Keras-layout HDF5 tensors directly.

The optional TensorFlow/Keras runtime is used only to execute the source graph
for parity; the converted runtime remains PyTorch-only.  Decoder, property
head, training CSV, and latent standardisation are never loaded.  The frozen
parity protocol is intentionally written before numerical comparison and the
script refuses to overwrite an existing parity run directory.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from yonod.descriptors.chemical_vae_backend import (
    ChemicalVaeAssetError,
    converted_payload,
    converted_torch_encoder_forward,
    load_chemical_vae_encoder_asset,
    numpy_encoder_forward,
    smiles_to_right_padded_one_hot,
)


REFERENCE_MODEL_DIR = REPOSITORY_ROOT / "reference-proejct" / "chemical_vae" / "models" / "zinc"
WEIGHTS_ROOT = REPOSITORY_ROOT / "WEIGHTS" / "chemical_vae"
PARITY_ROOT = REPOSITORY_ROOT / "derived" / "chemical_vae" / "step31_3_parity"
# v5 records the direct, executable TensorFlow/Keras reference under the code
# identity that also supplies the HDF5-free runtime loader.  All v1-v4
# historical assets/evidence remain in place; this versioned path never
# overwrites them.
CONVERTER_VERSION = "chemical_vae_torch_encoder/v5"

# Frozen before any candidate/reference run.  Both paths use CPU float32.  The
# absolute tolerance is deliberately wider than a few observed FP32 reduction
# ulps, but narrow enough to expose axis, padding, flatten, BatchNorm or dense
# transpose errors (which produce much larger deviations).
PARITY_PROTOCOL: Dict[str, Any] = {
    "protocol_version": "chemical_vae_step31_3/v1",
    "comparison_dtype": "float32",
    "device": "cpu",
    "inference_mode": True,
    "atol": 2.0e-5,
    "rtol": 2.0e-5,
    "near_zero_reference_threshold": 1.0e-7,
    "required_layers": [
        "input_molecule_smi",
        "encoder_conv0",
        "encoder_norm0",
        "encoder_conv1",
        "encoder_norm1",
        "encoder_conv2",
        "encoder_norm2",
        "flatten_1",
        "encoder_dense0",
        "dropout_1",
        "encoder_dense0_norm",
        "z_mean_sample",
    ],
    "batch_sizes": [1, 2, 3, 8],
    "test_smiles": [
        "CCO",
        "c1ccccc1",
        "O=C=O",
        "C1CC1",
        "N#N",
        "CCO",  # duplicate checks de-duplication is not silently assumed by the encoder.
        "C" * 119,
        "C" * 120,
    ],
    "input_preprocessing": "original Chemical VAE right-padding one-hot; identity SMILES text",
    "output_semantics": "raw z_mean_sample only; no sampling or latent-space standardisation",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _code_identity() -> str:
    digest = hashlib.sha256()
    for path in (
        Path(__file__).resolve(),
        REPOSITORY_ROOT / "yonod" / "descriptors" / "chemical_vae_backend.py",
    ):
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPOSITORY_ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".npz", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_torch_save(path: Path, payload: Mapping[str, Any]) -> None:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - backend emits the detailed error in ordinary use.
        raise ChemicalVaeAssetError("PyTorch is required to persist converted Chemical VAE assets") from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".pt", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(dict(payload), temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _metric(reference: np.ndarray, candidate: np.ndarray, protocol: Mapping[str, Any]) -> Dict[str, Any]:
    if reference.shape != candidate.shape:
        return {
            "passed": False,
            "reason": f"shape mismatch: reference {reference.shape}, candidate {candidate.shape}",
            "reference_shape": list(reference.shape),
            "candidate_shape": list(candidate.shape),
        }
    absolute = np.abs(reference - candidate)
    close = np.isclose(reference, candidate, rtol=float(protocol["rtol"]), atol=float(protocol["atol"]))
    maximum_index = tuple(int(item) for item in np.unravel_index(int(np.argmax(absolute)), absolute.shape))
    reference_at_max = float(reference[maximum_index])
    candidate_at_max = float(candidate[maximum_index])
    near_zero = np.abs(reference) <= float(protocol["near_zero_reference_threshold"])
    nonzero = ~near_zero
    relative = np.zeros_like(absolute, dtype=np.float64)
    relative[nonzero] = absolute[nonzero] / np.abs(reference[nonzero])
    return {
        "passed": bool(close.all()),
        "reference_shape": list(reference.shape),
        "candidate_shape": list(candidate.shape),
        "atol": float(protocol["atol"]),
        "rtol": float(protocol["rtol"]),
        "max_absolute_error": float(absolute.max(initial=0.0)),
        "mean_absolute_error": float(absolute.mean(dtype=np.float64)),
        "max_absolute_error_coordinate": list(maximum_index),
        "reference_value_at_max_error": reference_at_max,
        "candidate_value_at_max_error": candidate_at_max,
        "near_zero_reference_count": int(near_zero.sum()),
        "max_relative_error_nonzero_reference": float(relative[nonzero].max(initial=0.0)),
        "mean_relative_error_nonzero_reference": float(relative[nonzero].mean(dtype=np.float64))
        if bool(nonzero.any())
        else 0.0,
        "failed_coordinate_count": int((~close).sum()),
    }


def _encode_batches(
    forward: Callable[..., Tuple[np.ndarray, Dict[str, np.ndarray]]],
    one_hot: np.ndarray,
    batch_size: int,
    *,
    capture_layers: bool = False,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    outputs = []
    captured: Dict[str, list[np.ndarray]] = {}
    for start in range(0, len(one_hot), batch_size):
        result, layers = forward(one_hot[start : start + batch_size], capture_layers=capture_layers)
        outputs.append(result)
        for name, values in layers.items():
            captured.setdefault(name, []).append(values)
    return np.concatenate(outputs, axis=0), {
        name: np.concatenate(parts, axis=0) for name, parts in captured.items()
    }


def _source_asset(model_dir: Path, *, variant: str) -> Any:
    return load_chemical_vae_encoder_asset(
        model_dir / "zinc_encoder.h5",
        exp_path=model_dir / "exp.json",
        charset_path=model_dir / "zinc.json",
        variant=variant,
    )


def _conversion_paths(asset: Any, weights_root: Path) -> Tuple[Path, Path]:
    version_directory = CONVERTER_VERSION.rsplit("/", maxsplit=1)[-1]
    directory = weights_root / asset.asset_id / version_directory
    return directory / "encoder_state_dict.pt", directory / "conversion_manifest.json"


def _tensorflow_reference_forward(
    one_hot: np.ndarray,
    model_path: Path,
    *,
    required_layers: Sequence[str],
    capture_layers: bool = False,
) -> Tuple[np.ndarray, Dict[str, np.ndarray], Dict[str, Any]]:
    """Run the original HDF5 graph through a modern CPU TensorFlow/Keras runtime.

    This path deliberately loads the source model itself rather than recreating
    architecture/weights.  It is used solely as parity evidence: the public
    descriptor will continue to load the vetted PyTorch state dict only.
    """

    # TensorFlow reads this before import.  Disabling oneDNN makes CPU floating
    # point operation ordering explicit and keeps repeat evidence stable.
    os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise ChemicalVaeAssetError(
            "Step-31.3 requires tensorflow-cpu==2.15.1 for the direct original-HDF5 reference. "
            "Install requirements-chemical-vae-parity.txt before running parity."
        ) from exc
    try:
        tf.config.threading.set_intra_op_parallelism_threads(1)
        tf.config.threading.set_inter_op_parallelism_threads(1)
    except RuntimeError:
        # The process may already be initialized by an embedding caller.  The
        # fixed environment and version are still emitted below for audit.
        pass
    try:
        model = tf.keras.models.load_model(model_path, compile=False)
    except Exception as exc:
        raise ChemicalVaeAssetError(
            f"TensorFlow could not load original Chemical VAE HDF5 encoder {model_path}: {type(exc).__name__}: {exc}"
        ) from exc
    if tuple(model.input_shape[1:]) != tuple(one_hot.shape[1:]):
        raise ChemicalVaeAssetError(
            f"TensorFlow model input {model.input_shape!r} conflicts with parity input {one_hot.shape!r}"
        )
    expected_names = list(required_layers)
    actual_names = [layer.name for layer in model.layers]
    if actual_names != expected_names:
        raise ChemicalVaeAssetError(
            f"TensorFlow-loaded model layer order differs from frozen protocol: {actual_names!r}"
        )
    tensor = tf.convert_to_tensor(np.asarray(one_hot, dtype=np.float32))
    if capture_layers:
        capture_model = tf.keras.Model(
            inputs=model.input,
            outputs=[model.get_layer(name).output for name in expected_names[1:]],
        )
        outputs = capture_model(tensor, training=False)
        layers = {"input_molecule_smi": np.asarray(one_hot, dtype=np.float32)}
        for name, value in zip(expected_names[1:], outputs):
            layers[name] = np.asarray(value.numpy(), dtype=np.float32)
        z_mean = layers["z_mean_sample"]
    else:
        outputs = model(tensor, training=False)
        z_mean = np.asarray(outputs[0].numpy(), dtype=np.float32)
        layers = {}
    environment = {
        "tensorflow_version": tf.__version__,
        "keras_module": "tf.keras",
        "model_load": "tf.keras.models.load_model(original_hdf5, compile=False)",
        "device": "cpu",
        "TF_ENABLE_ONEDNN_OPTS": os.environ.get("TF_ENABLE_ONEDNN_OPTS"),
        "intra_op_threads_requested": 1,
        "inter_op_threads_requested": 1,
    }
    return z_mean, layers, environment


def _ensure_converted_asset(asset: Any, weights_root: Path, code_sha256: str) -> Dict[str, Any]:
    state_path, manifest_path = _conversion_paths(asset, weights_root)
    if state_path.exists() or manifest_path.exists():
        if not state_path.is_file() or not manifest_path.is_file():
            raise ChemicalVaeAssetError(
                f"incomplete existing converted asset directory: {state_path.parent}; refusing to overwrite"
            )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("converter_version") != CONVERTER_VERSION:
            raise ChemicalVaeAssetError(
                f"existing converted asset uses {manifest.get('converter_version')!r}; refusing cross-version overwrite"
            )
        if manifest.get("asset", {}).get("source_encoder_sha256") != asset.source_sha256:
            raise ChemicalVaeAssetError("existing converted asset source hash does not match requested source")
        if manifest.get("converted_state_sha256") != _sha256_file(state_path):
            raise ChemicalVaeAssetError("existing converted state hash does not match its conversion manifest")
        return {"status": "reused", "state_path": state_path, "manifest_path": manifest_path, "manifest": manifest}

    payload = converted_payload(asset, converter_version=CONVERTER_VERSION)
    _atomic_torch_save(state_path, payload)
    manifest = {
        "schema_version": "chemical_vae_conversion_manifest/v1",
        "created_at": _utc_now(),
        "converter_version": CONVERTER_VERSION,
        "converter_code_sha256": code_sha256,
        "asset": asset.manifest_fragment(),
        "converted_state_path": _relative(state_path),
        "converted_state_sha256": _sha256_file(state_path),
        "output_semantics": "raw z_mean_sample only; decoder/property head/standardisation are absent",
        "source_asset_read_only": True,
    }
    _atomic_json(manifest_path, manifest)
    return {"status": "created", "state_path": state_path, "manifest_path": manifest_path, "manifest": manifest}


def run_parity(
    *,
    model_dir: Path = REFERENCE_MODEL_DIR,
    weights_root: Path = WEIGHTS_ROOT,
    parity_root: Path = PARITY_ROOT,
    variant: str = "zinc",
) -> Dict[str, Any]:
    """Convert/reload the real source encoder and write one immutable parity run."""

    asset = _source_asset(model_dir.resolve(), variant=variant)
    code_sha256 = _code_identity()
    conversion = _ensure_converted_asset(asset, weights_root.resolve(), code_sha256)
    one_hot = smiles_to_right_padded_one_hot(PARITY_PROTOCOL["test_smiles"], asset)
    run_id = f"{asset.asset_id}-{code_sha256[:12]}"
    run_dir = parity_root.resolve() / run_id
    if run_dir.exists():
        raise ChemicalVaeAssetError(f"parity run already exists: {run_dir}; refusing to overwrite historical evidence")
    run_dir.mkdir(parents=True)

    protocol = dict(PARITY_PROTOCOL)
    protocol.update(
        {
            "frozen_at": _utc_now(),
            "source_asset": asset.manifest_fragment(),
            "source_exp_path": _relative(model_dir / "exp.json"),
            "source_charset_path": _relative(model_dir / "zinc.json"),
            "converted_state_path": _relative(conversion["state_path"]),
            "converted_state_sha256": _sha256_file(conversion["state_path"]),
            "converter_version": CONVERTER_VERSION,
            "converter_code_sha256": code_sha256,
            "numpy_reference_implementation": {
                "name": "numpy_encoder_forward",
                "independence": (
                    "Direct NumPy channels-last implementation reading raw Keras HDF5 kernels. It never imports "
                    "PyTorch, never reads the converted state dict, and applies Keras Conv1D/Dense layouts directly."
                ),
                "evidence_strength": "independent implementation used as a layer-level cross-check",
            },
            "tensorflow_reference_implementation": {
                "name": "tf.keras.models.load_model",
                "source": "original variant encoder HDF5 only; compile=False; CPU inference mode",
                "required_distribution": "tensorflow-cpu==2.15.1",
                "evidence_strength": (
                    "direct executable reference: modern tf.keras loads and executes the original Keras HDF5 graph; "
                    "it is independent of both the NumPy recreation and the converted PyTorch state dict"
                ),
            },
            "candidate_implementation": {
                "name": "converted_torch_encoder_forward",
                "independence": (
                    "Loads only the persisted PyTorch state dict, uses channels-first Conv1d and transposed kernels/" 
                    "Dense weights, then returns channels-last layer captures for comparison."
                ),
            },
            "input_one_hot_sha256": hashlib.sha256(one_hot.tobytes()).hexdigest(),
        }
    )
    # The protocol must exist before any reference/candidate comparison.  This
    # ensures tolerance cannot be altered after seeing an error value.
    _atomic_json(run_dir / "parity_protocol.json", protocol)
    _atomic_npz(run_dir / "inputs.npz", one_hot=one_hot)

    numpy_reference_by_batch: Dict[int, np.ndarray] = {}
    tensorflow_reference_by_batch: Dict[int, np.ndarray] = {}
    candidate_by_batch: Dict[int, np.ndarray] = {}
    candidate_vs_numpy_by_batch: Dict[str, Any] = {}
    candidate_vs_tensorflow_by_batch: Dict[str, Any] = {}
    numpy_vs_tensorflow_by_batch: Dict[str, Any] = {}
    numpy_reference_layers: Dict[str, np.ndarray] = {}
    tensorflow_reference_layers: Dict[str, np.ndarray] = {}
    candidate_layers: Dict[str, np.ndarray] = {}
    tensorflow_environment: Dict[str, Any] = {}
    primary_batch = max(int(value) for value in protocol["batch_sizes"])
    for batch_size in protocol["batch_sizes"]:
        is_primary = int(batch_size) == primary_batch
        numpy_reference, numpy_maybe_layers = _encode_batches(
            lambda batch, capture_layers: numpy_encoder_forward(batch, asset, capture_layers=capture_layers),
            one_hot,
            int(batch_size),
            capture_layers=is_primary,
        )
        tensorflow_outputs = []
        tensorflow_layer_parts: Dict[str, list[np.ndarray]] = {}
        for start in range(0, len(one_hot), int(batch_size)):
            tensorflow_output, tensorflow_layers, environment = _tensorflow_reference_forward(
                one_hot[start : start + int(batch_size)],
                model_dir / "zinc_encoder.h5",
                required_layers=protocol["required_layers"],
                capture_layers=is_primary,
            )
            tensorflow_outputs.append(tensorflow_output)
            tensorflow_environment = environment
            for name, values in tensorflow_layers.items():
                tensorflow_layer_parts.setdefault(name, []).append(values)
        tensorflow_reference = np.concatenate(tensorflow_outputs, axis=0)
        tensorflow_maybe_layers = {
            name: np.concatenate(parts, axis=0) for name, parts in tensorflow_layer_parts.items()
        }
        candidate, candidate_maybe_layers = _encode_batches(
            lambda batch, capture_layers: converted_torch_encoder_forward(
                batch, conversion["state_path"], asset, capture_layers=capture_layers
            ),
            one_hot,
            int(batch_size),
            capture_layers=is_primary,
        )
        numpy_reference_by_batch[int(batch_size)] = numpy_reference
        tensorflow_reference_by_batch[int(batch_size)] = tensorflow_reference
        candidate_by_batch[int(batch_size)] = candidate
        candidate_vs_numpy_by_batch[str(batch_size)] = _metric(numpy_reference, candidate, protocol)
        candidate_vs_tensorflow_by_batch[str(batch_size)] = _metric(tensorflow_reference, candidate, protocol)
        numpy_vs_tensorflow_by_batch[str(batch_size)] = _metric(numpy_reference, tensorflow_reference, protocol)
        if is_primary:
            numpy_reference_layers = numpy_maybe_layers
            tensorflow_reference_layers = tensorflow_maybe_layers
            candidate_layers = candidate_maybe_layers

    numpy_reference_primary = numpy_reference_by_batch[primary_batch]
    tensorflow_reference_primary = tensorflow_reference_by_batch[primary_batch]
    candidate_primary = candidate_by_batch[primary_batch]
    candidate_vs_numpy_layers = {
        name: _metric(numpy_reference_layers[name], candidate_layers[name], protocol)
        for name in protocol["required_layers"]
    }
    candidate_vs_tensorflow_layers = {
        name: _metric(tensorflow_reference_layers[name], candidate_layers[name], protocol)
        for name in protocol["required_layers"]
    }
    numpy_vs_tensorflow_layers = {
        name: _metric(numpy_reference_layers[name], tensorflow_reference_layers[name], protocol)
        for name in protocol["required_layers"]
    }
    numpy_reference_batch_invariance = {
        str(batch_size): _metric(numpy_reference_primary, values, protocol)
        for batch_size, values in numpy_reference_by_batch.items()
    }
    tensorflow_reference_batch_invariance = {
        str(batch_size): _metric(tensorflow_reference_primary, values, protocol)
        for batch_size, values in tensorflow_reference_by_batch.items()
    }
    candidate_batch_invariance = {
        str(batch_size): _metric(candidate_primary, values, protocol)
        for batch_size, values in candidate_by_batch.items()
    }
    metric_groups = (
        candidate_vs_numpy_by_batch,
        candidate_vs_tensorflow_by_batch,
        numpy_vs_tensorflow_by_batch,
        candidate_vs_numpy_layers,
        candidate_vs_tensorflow_layers,
        numpy_vs_tensorflow_layers,
        numpy_reference_batch_invariance,
        tensorflow_reference_batch_invariance,
        candidate_batch_invariance,
    )
    all_passed = all(item["passed"] for group in metric_groups for item in group.values())

    _atomic_npz(
        run_dir / "numpy_reference_vectors.npz",
        **{f"batch_{batch_size}": values for batch_size, values in numpy_reference_by_batch.items()},
    )
    _atomic_npz(
        run_dir / "tensorflow_reference_vectors.npz",
        **{f"batch_{batch_size}": values for batch_size, values in tensorflow_reference_by_batch.items()},
    )
    _atomic_npz(
        run_dir / "candidate_vectors.npz",
        **{f"batch_{batch_size}": values for batch_size, values in candidate_by_batch.items()},
    )
    _atomic_npz(run_dir / "numpy_reference_layers.npz", **numpy_reference_layers)
    _atomic_npz(run_dir / "tensorflow_reference_layers.npz", **tensorflow_reference_layers)
    _atomic_npz(run_dir / "candidate_layers.npz", **candidate_layers)
    report = {
        "schema_version": "chemical_vae_parity_report/v1",
        "completed_at": _utc_now(),
        "status": "passed" if all_passed else "failed",
        "numerical_parity_verified": bool(all_passed),
        "protocol_path": "parity_protocol.json",
        "numpy_reference_vectors_path": "numpy_reference_vectors.npz",
        "tensorflow_reference_vectors_path": "tensorflow_reference_vectors.npz",
        "candidate_vectors_path": "candidate_vectors.npz",
        "numpy_reference_layers_path": "numpy_reference_layers.npz",
        "tensorflow_reference_layers_path": "tensorflow_reference_layers.npz",
        "candidate_layers_path": "candidate_layers.npz",
        "tensorflow_reference_environment": tensorflow_environment,
        "candidate_vs_numpy_by_batch": candidate_vs_numpy_by_batch,
        "candidate_vs_tensorflow_by_batch": candidate_vs_tensorflow_by_batch,
        "numpy_vs_tensorflow_by_batch": numpy_vs_tensorflow_by_batch,
        "candidate_vs_numpy_layers": candidate_vs_numpy_layers,
        "candidate_vs_tensorflow_layers": candidate_vs_tensorflow_layers,
        "numpy_vs_tensorflow_layers": numpy_vs_tensorflow_layers,
        "numpy_reference_batch_invariance": numpy_reference_batch_invariance,
        "tensorflow_reference_batch_invariance": tensorflow_reference_batch_invariance,
        "candidate_batch_invariance": candidate_batch_invariance,
        "acceptance_interpretation": (
            "Passed against a direct TensorFlow/Keras execution of the original HDF5 encoder and a separately "
            "implemented NumPy layer-level reference under the pre-frozen CPU float32 protocol."
        ),
    }
    _atomic_json(run_dir / "parity_report.json", report)
    return {
        "run_dir": run_dir,
        "conversion": conversion,
        "report": report,
        "asset": asset,
    }


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=REFERENCE_MODEL_DIR)
    parser.add_argument("--weights-root", type=Path, default=WEIGHTS_ROOT)
    parser.add_argument("--parity-root", type=Path, default=PARITY_ROOT)
    parser.add_argument("--variant", choices=("zinc", "zinc_properties"), default="zinc")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    try:
        result = run_parity(
            model_dir=args.model_dir,
            weights_root=args.weights_root,
            parity_root=args.parity_root,
            variant=args.variant,
        )
    except ChemicalVaeAssetError as exc:
        print(f"Chemical VAE step-31.3 failed: {exc}", file=sys.stderr)
        return 2
    report = result["report"]
    print(
        json.dumps(
            {
                "status": report["status"],
                "numerical_parity_verified": report["numerical_parity_verified"],
                "run_dir": str(result["run_dir"]),
                "converted_state": str(result["conversion"]["state_path"]),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if report["numerical_parity_verified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
