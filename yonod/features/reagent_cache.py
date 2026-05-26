"""Reagent SMILES normalization + descriptor cache.

The amide-condensation CSV stores reagent SMILES directly in the four
``*_id`` columns (already mapped from id codes). This module:

  1. Normalizes raw cell values:
       - ``"(无)"`` -> ``None`` (no reagent in that role; zero vector downstream)
       - ``","`` (multi-fragment separator) -> ``"."`` (RDKit standard)
       - otherwise: passthrough.
  2. Collects unique reagent SMILES from the dataframe (across 4 columns).
  3. Precomputes descriptor vectors for those unique SMILES, persists the
     result to ``cache/reagent_feats_{descriptor.name}.pkl``, and exposes
     ``get_reagent_feat()`` for O(1) lookup during feature assembly.

This avoids re-running the descriptor model on the same reagent across
all 47015 reactions; total unique reagents ≈ 58 + 1 (None).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Set

import joblib
import numpy as np
import pandas as pd

from ..descriptors.base import BaseDescriptor


REAGENT_COLS = ["activation_id", "additive_id", "base_id", "solvent_id"]
NA_TOKEN = "(无)"

# YONOD/yonod_yield/features/reagent_cache.py -> parents[2] = YONOD/
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_CACHE_DIR = _PROJECT_ROOT / "cache"


def normalize_reagent_smiles(s: Optional[str]) -> Optional[str]:
    """Normalize a raw reagent cell.

    Returns ``None`` for ``(无)`` (and NaN); replaces ``,`` with ``.`` so RDKit
    parses multi-fragment SMILES (salts / counterions) correctly.
    """
    if s is None:
        return None
    if not isinstance(s, str):
        # pandas may give float('nan') for missing cells
        if isinstance(s, float) and np.isnan(s):
            return None
        s = str(s)
    if s == NA_TOKEN:
        return None
    return s.replace(",", ".")


def collect_unique_reagents(df: pd.DataFrame) -> Set[Optional[str]]:
    """Collect the set of normalized unique reagent SMILES across 4 columns."""
    unique: Set[Optional[str]] = set()
    for col in REAGENT_COLS:
        if col not in df.columns:
            raise KeyError(f"Expected reagent column {col!r} in DataFrame")
        for v in df[col].unique():
            unique.add(normalize_reagent_smiles(v))
    return unique


def _default_cache_path(descriptor_name: str) -> Path:
    return _DEFAULT_CACHE_DIR / f"reagent_feats_{descriptor_name}.pkl"


def build_reagent_feat_cache(
    descriptor: BaseDescriptor,
    df: pd.DataFrame,
    cache_path: Optional[Path] = None,
    force_rebuild: bool = False,
) -> Dict[Optional[str], np.ndarray]:
    """Build (or load) ``dict[normalized_smiles | None -> ndarray(d,)]``.

    On disk hit (and ``force_rebuild=False``) the pkl is loaded directly.
    Otherwise unique reagents are extracted from ``df``, fed to
    ``descriptor.featurize()`` in one batch, and the resulting cache is
    persisted to ``cache_path`` (default: ``YONOD/cache/reagent_feats_{name}.pkl``).

    The ``None`` key always maps to a zero vector (``(无)`` reagent slots).
    """
    if cache_path is None:
        cache_path = _default_cache_path(descriptor.name)
    cache_path = Path(cache_path)

    if cache_path.exists() and not force_rebuild:
        cache: Dict[Optional[str], np.ndarray] = joblib.load(cache_path)
        return cache

    unique = collect_unique_reagents(df)
    non_null = sorted(s for s in unique if s is not None)

    if non_null:
        features, mask = descriptor.featurize(non_null)
        failed = [non_null[i] for i, ok in enumerate(mask) if not ok]
        if failed:
            raise RuntimeError(
                f"{len(failed)} reagent SMILES failed to featurize via "
                f"{descriptor.name}: {failed[:3]}..."
            )
        cache = {smi: features[i] for i, smi in enumerate(non_null)}
    else:
        cache = {}

    # Always ensure a None -> zeros entry exists (covers '(无)' lookups even if
    # this df happens to contain no '(无)' rows).
    cache[None] = np.zeros(descriptor.output_dim, dtype=np.float32)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(cache, cache_path)
    return cache


def get_reagent_feat(
    raw_smiles: Optional[str],
    cache: Dict[Optional[str], np.ndarray],
) -> np.ndarray:
    """Look up the descriptor vector for one raw reagent cell.

    ``raw_smiles`` is the unmodified CSV cell (e.g. ``"(无)"`` or
    ``"CCN=C=NCCCN(C)C,Cl"``); normalization is applied internally.
    Raises ``KeyError`` if the normalized key is missing from ``cache`` --
    this signals a bug (the cache should have been built from the same df).
    """
    key = normalize_reagent_smiles(raw_smiles)
    if key not in cache:
        raise KeyError(
            f"Reagent SMILES not in cache: raw={raw_smiles!r}, "
            f"normalized={key!r}. Rebuild the cache with the current df."
        )
    return cache[key]
