"""MolMetaLM Embedding descriptor (768-d, masked mean-pool over last hidden state).

Loads the pretrained MolMetaLM checkpoint at ``WEIGHTS/MolMetaLM-base/`` lazily
on the first ``featurize`` call. For each SMILES the encoder runs once and the
sequence representation is reduced to a fixed 768-d vector by attention-mask
weighted mean-pooling.

Implementation notes:
  * Lazy load -- importing this module does NOT load the 500MB checkpoint.
  * Auto-select GPU when available; fall back to CPU.
  * Batched forward (default ``batch_size=32``); halves on
    ``torch.cuda.OutOfMemoryError`` and retries.
  * Native ``AutoModel`` loads the pinned Llama encoder without an LM head.
  * All assets are validated before loading; only local safetensors are allowed.
  * Author character-spacing and the fast tokenizer preserve the trained input.
    Unsupported characters/UNK fail per row; explicit max_length truncation stays.
  * Failed tokenization or non-mol SMILES -> zero row, mask=False.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Mapping, Optional, Tuple

import numpy as np

from .base import BaseDescriptor
from .molmetalm_assets import (
    INPUT_PREPROCESSING_POLICY, MolMetaLMAssetError,
    character_spaced_smiles, validated_asset_identity,
)

# Default checkpoint location -- matches YONOD §3.5 path convention.
DEFAULT_WEIGHT_PATH = str(
    Path(__file__).resolve().parents[2] / "WEIGHTS" / "MolMetaLM-base"
)


class MolMetaLMDescriptor(BaseDescriptor):
    """768-d pretrained-LM embedding via attention-masked mean pooling."""

    name = "molmetalm"
    output_dim = 768
    input_preprocessing_policy = INPUT_PREPROCESSING_POLICY

    def __init__(
        self,
        weight_path: str = DEFAULT_WEIGHT_PATH,
        device: Optional[str] = None,
        batch_size: int = 32,
        max_length: int = 512,
        asset_sha256: Optional[Mapping[str, str]] = None,
    ) -> None:
        self.weight_path = weight_path
        self.batch_size = batch_size
        self.max_length = max_length
        self.device = device  # resolved lazily to avoid importing torch at module load
        self._tokenizer = None
        self._model = None
        # Explicit pins are a caller trust decision, not a checkpoint-supplied
        # manifest. Copy them so subsequent mutation of the input cannot alter it.
        self.asset_sha256 = None if asset_sha256 is None else dict(asset_sha256)

    # --- Lazy loaders -----------------------------------------------------

    def asset_identity(self) -> dict:
        """Validate assets without importing model libraries or loading weights."""
        return validated_asset_identity(self.weight_path, self.asset_sha256)

    def _resolve_device(self) -> str:
        if self.device is not None:
            return self.device
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"

    def _load(self) -> None:
        if self._model is not None:
            return
        # Reject untrusted configuration before any Transformers Auto* API.
        # Validation is repeated here even if a cache caller already checked it.
        self.asset_identity()
        import sys

        import torch
        from transformers import AutoModel, AutoTokenizer

        self.device = self._resolve_device()
        local_path = str(Path(self.weight_path).expanduser().resolve(strict=True))
        tokenizer = AutoTokenizer.from_pretrained(
            local_path, local_files_only=True, trust_remote_code=False, use_fast=True,
        )
        if (
            not getattr(tokenizer, "is_fast", False)
            or getattr(tokenizer, "unk_token_id", None) != 3
            or tokenizer.num_special_tokens_to_add(pair=False) != 0
        ):
            raise MolMetaLMAssetError("MolMetaLM requires its approved fast tokenizer without special-token insertion")
        # LlamaConfig is natively supported by AutoModel. Do not retry a failed
        # security/format/config check through a different model loader.
        model = AutoModel.from_pretrained(
            local_path, local_files_only=True, trust_remote_code=False,
            use_safetensors=True,
        )
        if (
            getattr(model.config, "model_type", None) != "llama"
            or getattr(model.config, "hidden_size", None) != self.output_dim
        ):
            raise MolMetaLMAssetError("Loaded MolMetaLM model does not match its approved Llama/768 profile")
        model = model.to(self.device).eval()
        # Publish loaded state only after the complete operation succeeds.
        self._tokenizer, self._model, self._torch = tokenizer, model, torch
        print(
            "[MolMetaLMDescriptor] loaded via AutoModel, "
            f"class={type(self._model).__name__}, "
            f"hidden_size={self.output_dim}, device={self.device}",
            file=sys.stderr,
        )

    # --- Core ------------------------------------------------------------

    def _prepare_texts(self, batch_smiles: List[str]) -> Tuple[List[int], List[str]]:
        """Reject unsupported rows before truncation; retain valid batch peers."""
        import sys

        indices, texts, truncated = [], [], []
        for index, smiles in enumerate(batch_smiles):
            try:
                text = character_spaced_smiles(smiles)
                pieces = self._tokenizer.tokenize(text, add_special_tokens=False)
                if not pieces or pieces != list(smiles) or self._tokenizer.unk_token in pieces:
                    raise ValueError("characters are unsupported or were normalized/lost by the approved vocabulary")
            except (TypeError, ValueError) as exc:
                print(f"[MolMetaLMDescriptor] rejected batch row {index}: {exc}", file=sys.stderr)
                continue
            indices.append(index)
            texts.append(text)
            if len(pieces) > self.max_length:
                truncated.append(index)
        if truncated:
            print(
                f"[MolMetaLMDescriptor] explicit max_length={self.max_length} truncation for batch rows {truncated}",
                file=sys.stderr,
            )
        return indices, texts

    def _encode_batch(self, batch_smiles: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """Return (float32 embeddings, valid-row mask); no model call for all-bad rows."""
        import sys

        features, valid = self._empty_outputs(len(batch_smiles), dtype=np.float32)
        indices, texts = self._prepare_texts(batch_smiles)
        if not indices:
            return features, valid
        torch = self._torch
        tokens = self._tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            return_tensors="pt",
        )
        # Check the actual batched IDs too, before moving tensors or forwarding.
        ids = tokens["input_ids"].detach().cpu().numpy()
        attention = tokens["attention_mask"].detach().cpu().numpy()
        good = (~np.any(ids == self._tokenizer.unk_token_id, axis=1)) & np.any(attention != 0, axis=1)
        positions = np.flatnonzero(good).tolist()
        rejected = [indices[index] for index in np.flatnonzero(~good)]
        if rejected:
            print(f"[MolMetaLMDescriptor] rejected batch rows with UNK/empty token IDs: {rejected}", file=sys.stderr)
        if not positions:
            return features, valid

        # Llama (and many decoder-only models) doesn't accept token_type_ids.
        # Pass only the keys the model's forward signature actually uses.
        accepted = {"input_ids", "attention_mask", "position_ids"}
        model_inputs = {k: v[positions].to(self.device) for k, v in tokens.items() if k in accepted}

        with torch.no_grad():
            out = self._model(**model_inputs)
        hidden = getattr(out, "last_hidden_state", None)
        if hidden is None:
            # Some models return tuple; take element 0.
            hidden = out[0]

        # Masked mean-pool over the sequence dimension.
        mask = model_inputs["attention_mask"].unsqueeze(-1).to(hidden.dtype)
        summed = (hidden * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1)
        pooled = summed / denom  # (B, D)
        rows = [indices[position] for position in positions]
        features[rows] = pooled.detach().cpu().numpy().astype(np.float32)
        valid[rows] = True
        return features, valid

    def featurize(
        self, smiles_list: List[str]
    ) -> Tuple[np.ndarray, np.ndarray]:
        import sys
        import traceback

        self._load()
        n = len(smiles_list)
        features, mask = self._empty_outputs(n, dtype=np.float32)

        # Filter to valid (non-empty) strings; failed ones stay as zero rows.
        valid_idx = [i for i, s in enumerate(smiles_list) if s]
        if not valid_idx:
            return features, mask

        bs = self.batch_size
        i = 0
        first_error_logged = False
        while i < len(valid_idx):
            sub = valid_idx[i : i + bs]
            batch = [smiles_list[k] for k in sub]
            try:
                vecs, batch_mask = self._encode_batch(batch)
                for j, k in enumerate(sub):
                    if batch_mask[j]:
                        features[k] = vecs[j]
                        mask[k] = True
                i += bs
            except Exception as e:
                # CUDA OOM: halve batch and retry; otherwise mark rows failed and skip.
                if (
                    self.device == "cuda"
                    and "out of memory" in str(e).lower()
                    and bs > 1
                ):
                    self._torch.cuda.empty_cache()
                    bs = max(bs // 2, 1)
                    continue
                if not first_error_logged:
                    print(
                        f"[MolMetaLMDescriptor] batch failed (first occurrence). "
                        f"batch_size={len(sub)} sample[0]={batch[0]!r}",
                        file=sys.stderr,
                    )
                    traceback.print_exc()
                    first_error_logged = True
                # Non-OOM failure: leave this batch zeroed, mask False, move on.
                i += len(sub)

        return features, mask
