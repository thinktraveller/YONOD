"""Write a small, synthetic ablation2 contract fixture without fitting models."""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "result" / "ablation2_amide_micro_v1" / "data" / "standardized_population.csv"


def _hash(*values: str) -> str:
    return hashlib.sha256(json.dumps(values, separators=(",", ":")).encode("utf-8")).hexdigest()[:16]


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    try:
        frame.to_csv(temporary, index=False, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def main() -> int:
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite fixture: {OUTPUT}")
    components = [
        # A, B, P, activation, additive, base, yield.  The first two rows are
        # an intentionally duplicated complete input with divergent labels.
        ("CN", "CC(=O)O", "CC(=O)NC", "CCN=C=NCC", "", "N", 0.42),
        ("CN", "CC(=O)O", "CC(=O)NC", "CCN=C=NCC", "", "N", 0.58),
        ("CN", "CC(=O)O", "CC(=O)NC", "CC(C)N=C=NC(C)C", "O", "N", 0.51),
        ("CN", "CCC(=O)O", "CCC(=O)NC", "", "", "N", 0.64),
        ("CCN", "CC(=O)O", "CC(=O)NCC", "CCN=C=NCC", "", "N", 0.31),
        ("CCN", "CC(=O)O", "CC(=O)NCC", "CC(C)N=C=NC(C)C", "O", "N", 0.37),
        ("CCN", "CCC(=O)O", "CCC(=O)NCC", "", "", "N", 0.72),
        ("CCN", "CCC(=O)O", "CCC(=O)NCC", "CCN=C=NCC", "O", "N", 0.67),
    ]
    rows = []
    for index, (amine, acid, product, activation, additive, base, target) in enumerate(components, start=1):
        full = (amine, acid, product, activation, additive, base, "CN(C)C=O")
        rows.append({
            "sample_id": f"ablation2-micro:{index:02d}", "source_row_id": f"synthetic-{index:02d}",
            "yield": target, "population_status": "included",
            "amine_smiles": amine, "acid_smiles": acid, "product_smiles": product,
            "activation_smiles": activation, "additive_smiles": additive, "base_smiles": base,
            "solvent_smiles": "CN(C)C=O",
            "amine_group_id": "amine-" + _hash(amine),
            "acid_group_id": "acid-" + _hash(acid),
            "substrate_pair_group_id": "pair-" + _hash(amine, acid),
            "repeat_group_id": "full-input-" + _hash(*full),
            "fixture_note": "synthetic contract fixture; never a formal research population",
        })
    frame = pd.DataFrame.from_records(rows)
    _atomic_csv(frame, OUTPUT)
    print(f"wrote {len(frame)} rows to {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
