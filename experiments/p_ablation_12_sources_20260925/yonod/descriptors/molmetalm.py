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
  * ``AutoModel`` is preferred (no LM head), with ``AutoModelForCausalLM``
    as the fallback for checkpoints whose config defaults to causal LM.
  * Failed tokenization or non-mol SMILES -> zero row, mask=False.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from .base import BaseDescriptor

# Default checkpoint location -- matches YONOD §3.5 path convention.
DEFAULT_WEIGHT_PATH = str(
    Path(__file__).resolve().parents[2] / "WEIGHTS" / "MolMetaLM-base"
)


class MolMetaLMDescriptor(BaseDescriptor):
    """768-d pretrained-LM embedding via attention-masked mean pooling."""

    name = "molmetalm"
    output_dim = 768

    def __init__(
        self,
        weight_path: str = DEFAULT_WEIGHT_PATH,
        device: Optional[str] = None,
        batch_size: int = 32,
        max_length: int = 512,
    ) -> None:
        self.weight_path = weight_path
        self.batch_size = batch_size
        self.max_length = max_length
        self.device = device  # resolved lazily to avoid importing torch at module load
        self._tokenizer = None
        self._model = None

    # --- Lazy loaders -----------------------------------------------------

    def _resolve_device(self) -> str:
        if self.device is not None:
            return self.device
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"

    def _load(self) -> None:
        if self._model is not None:
            return
        import sys

        import torch
        from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

        self.device = self._resolve_device()
        self._tokenizer = AutoTokenizer.from_pretrained(self.weight_path)
        try:
            model = AutoModel.from_pretrained(self.weight_path)
            loaded_via = "AutoModel"
        except Exception as e:
            print(
                f"[MolMetaLMDescriptor] AutoModel.from_pretrained failed ({e}); "
                "falling back to AutoModelForCausalLM and unwrapping.",
                file=sys.stderr,
            )
            lm = AutoModelForCausalLM.from_pretrained(self.weight_path)
            model = getattr(lm, "model", None) or getattr(lm, "base_model", lm)
            loaded_via = f"AutoModelForCausalLM -> {type(model).__name__}"
        self._model = model.to(self.device).eval()
        self._torch = torch

        # Probe hidden size and overwrite output_dim if config disagrees.
        hidden_size = getattr(self._model.config, "hidden_size", None)
        if hidden_size and hidden_size != self.output_dim:
            self.output_dim = int(hidden_size)
        print(
            f"[MolMetaLMDescriptor] loaded via {loaded_via}, "
            f"class={type(self._model).__name__}, "
            f"hidden_size={self.output_dim}, device={self.device}",
            file=sys.stderr,
        )

    # --- Core ------------------------------------------------------------

    def _encode_batch(self, batch_smiles: List[str]) -> np.ndarray:
        """Run one forward pass; return (B, D) ndarray (CPU, float32)."""
        torch = self._torch
        tokens = self._tokenizer(
            batch_smiles,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(self.device)

        # Llama (and many decoder-only models) doesn't accept token_type_ids.
        # Pass only the keys the model's forward signature actually uses.
        accepted = {"input_ids", "attention_mask", "position_ids"}
        model_inputs = {k: v for k, v in tokens.items() if k in accepted}

        with torch.no_grad():
            out = self._model(**model_inputs)
        hidden = getattr(out, "last_hidden_state", None)
        if hidden is None:
            # Some models return tuple; take element 0.
            hidden = out[0]

        # Masked mean-pool over the sequence dimension.
        mask = tokens["attention_mask"].unsqueeze(-1).to(hidden.dtype)
        summed = (hidden * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1)
        pooled = summed / denom  # (B, D)
        return pooled.detach().cpu().numpy().astype(np.float32)

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
                vecs = self._encode_batch(batch)
                for j, k in enumerate(sub):
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
