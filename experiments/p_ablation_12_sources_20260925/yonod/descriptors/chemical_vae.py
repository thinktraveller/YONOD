"""Frozen Chemical VAE ``z_mean`` descriptor backed by a verified PyTorch state.

The adapter accepts only v5 conversion manifests for separately verified
``zinc`` and ``zinc_properties`` encoders.  It loads no HDF5/TensorFlow
component and returns the original 196-dimensional ``z_mean_sample`` output
without normalisation or sampling; the properties-predictor head is excluded.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, List, Mapping, Tuple

import numpy as np

from .base import BaseDescriptor
from .chemical_vae_backend import (
    ChemicalVaeAssetError,
    load_converted_torch_encoder_runtime,
    smiles_to_right_padded_one_hot,
)


class ChemicalVaeDescriptor(BaseDescriptor):
    """Batch verified Chemical VAE encoder inference from a checked conversion."""

    name = "chemical_vae"
    output_dim = 196

    def __init__(
        self,
        *,
        model_manifest: Path | str,
        backend: str = "pytorch",
        device: str = "cpu",
        batch_size: int = 64,
        input_preprocessing: str = "identity",
        output: str = "z_mean_sample",
    ) -> None:
        if backend != "pytorch":
            raise ChemicalVaeAssetError("chemical_vae.backend 当前只支持 'pytorch'")
        if input_preprocessing != "identity":
            raise ChemicalVaeAssetError(
                "chemical_vae.input_preprocessing 当前固定为 'identity'；不拆盐、不 canonicalize、不截断"
            )
        if output != "z_mean_sample":
            raise ChemicalVaeAssetError("chemical_vae.output 当前只支持原始 'z_mean_sample'")
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size <= 0:
            raise ChemicalVaeAssetError("chemical_vae.batch_size 必须是正整数")
        self.manifest_path = Path(model_manifest).expanduser().resolve()
        if not self.manifest_path.is_file():
            raise ChemicalVaeAssetError(f"chemical_vae model_manifest 不存在：{self.manifest_path}")
        try:
            manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ChemicalVaeAssetError(f"无法读取 chemical_vae conversion manifest：{exc}") from exc
        if not isinstance(manifest, Mapping) or manifest.get("schema_version") != "chemical_vae_conversion_manifest/v1":
            raise ChemicalVaeAssetError("不支持的 chemical_vae conversion manifest schema")
        if manifest.get("converter_version") != "chemical_vae_torch_encoder/v5":
            raise ChemicalVaeAssetError(
                "chemical_vae 只接受已由直接 HDF5 参考验证的 chemical_vae_torch_encoder/v5 资产"
            )
        asset = manifest.get("asset")
        if not isinstance(asset, Mapping) or not isinstance(asset.get("source_encoder_sha256"), str):
            raise ChemicalVaeAssetError("chemical_vae conversion manifest 缺少 source asset identity")
        # v4's manifest stores a repository-relative audit path, but runtime
        # ownership is the adjacent state file so copied asset directories stay
        # self-contained.  A non-adjacent declared path is intentionally not
        # followed here.
        state_path = self.manifest_path.parent / "encoder_state_dict.pt"
        if not state_path.is_file():
            raise ChemicalVaeAssetError(f"chemical_vae converted state 不存在：{state_path}")
        expected_state_sha = manifest.get("converted_state_sha256")
        if not isinstance(expected_state_sha, str) or self._sha256_file(state_path) != expected_state_sha:
            raise ChemicalVaeAssetError("chemical_vae converted state hash 与 conversion manifest 不一致")
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - ordinary environment has PyTorch.
            raise ChemicalVaeAssetError("chemical_vae backend='pytorch' 需要安装 PyTorch") from exc
        requested_device = str(device)
        try:
            torch_device = torch.device(requested_device)
        except (TypeError, RuntimeError) as exc:
            raise ChemicalVaeAssetError(f"chemical_vae.device 无效：{requested_device!r}") from exc
        if torch_device.type == "cuda" and not torch.cuda.is_available():
            raise ChemicalVaeAssetError("chemical_vae 请求 CUDA，但当前 PyTorch 没有可用 CUDA")
        model, runtime = load_converted_torch_encoder_runtime(
            state_path,
            expected_source_sha256=asset["source_encoder_sha256"],
        )
        self._torch = torch
        self._device = torch_device
        self._model = model.to(torch_device).eval()
        self._runtime = runtime
        self.output_dim = runtime.hidden_dim
        self.batch_size = batch_size
        self.backend = backend
        self.device = str(torch_device)
        self.input_preprocessing = input_preprocessing
        self.output = output
        # Per-call, full-row diagnostics are consumed by the step-31.5
        # feature publisher.  They are not a success signal and are reset for
        # every invocation to avoid leaking one column's records into another.
        self.last_diagnostics: list[dict[str, Any]] = []

    @staticmethod
    def _sha256_file(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def featurize(self, smiles_list: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """Encode unique valid identity-SMILES in batches and restore row order."""

        features, mask = self._empty_outputs(len(smiles_list), dtype=np.float32)
        diagnostics: list[dict[str, Any]] = []
        charset = set(self._runtime.charset)
        for index, value in enumerate(smiles_list):
            text = value if isinstance(value, str) else ""
            record: dict[str, Any] = {
                "input_index": index,
                "input_text": text,
                "preprocessed_text": text,
                "length": len(text),
                "status": "pending",
                "reason": None,
                "unknown_characters": [],
            }
            if not text:
                record.update(status="failed", reason="missing_or_empty_input")
            elif len(text) > self._runtime.max_len:
                record.update(status="failed", reason="length_exceeds_max_len")
            else:
                unknown = sorted(set(character for character in text if character not in charset))
                if unknown:
                    record.update(status="failed", reason="unsupported_character", unknown_characters=unknown)
            diagnostics.append(record)
        self.last_diagnostics = diagnostics
        if not smiles_list:
            return features, mask
        # RDKit is only a validity precheck.  The text subsequently submitted
        # to Chemical VAE remains byte-for-byte identical to the caller's
        # string, including its whitespace and separators.
        from rdkit import Chem

        accepted: dict[str, list[int]] = {}
        for index, value in enumerate(smiles_list):
            record = diagnostics[index]
            if record["status"] != "pending" or not isinstance(value, str):
                continue
            if Chem.MolFromSmiles(value) is None:
                record.update(status="failed", reason="rdkit_invalid_smiles")
                continue
            accepted.setdefault(value, []).append(index)
        unique_smiles = list(accepted)
        for start in range(0, len(unique_smiles), self.batch_size):
            batch_smiles = unique_smiles[start : start + self.batch_size]
            try:
                one_hot = smiles_to_right_padded_one_hot(batch_smiles, self._runtime)
                tensor = self._torch.from_numpy(one_hot).to(self._device)
                with self._torch.no_grad():
                    vectors = self._model(tensor).detach().cpu().numpy().astype(np.float32, copy=False)
            except Exception as exc:
                raise ChemicalVaeAssetError(f"chemical_vae encoder 批量推理失败：{type(exc).__name__}: {exc}") from exc
            if vectors.shape != (len(batch_smiles), self.output_dim) or not np.isfinite(vectors).all():
                raise ChemicalVaeAssetError("chemical_vae encoder 返回非有限值或错误的 z_mean_sample 形状")
            for smile, vector in zip(batch_smiles, vectors):
                for index in accepted[smile]:
                    features[index] = vector
                    mask[index] = True
                    diagnostics[index].update(status="encoded", reason=None)
        return features, mask
