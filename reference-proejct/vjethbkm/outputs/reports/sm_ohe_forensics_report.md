# SM/OHE forensic report

Official SM/OHE y_true and fold_id exactly match KFold over Data/HTE_datasets/SM/SM.csv, so row order, labels, and split generation are ruled out as causes. The 2SM.csv file only adds catalyst_smiles while preserving all shared columns and Yield. Neither a float64 OHE variant nor a 2SM+catalyst variant reproduces the official SM/OHE summary. The supported conclusion is that the bundled Suzuki OHE RF result is a stale or differently generated artifact whose missing generation details are not recoverable from the included README/config/scripts alone.

## Key evidence

- SM.csv shape: `5760x6`; sha256 `111da4e67305f0e00ae7551a0252b78aee4438d434dd5f20e80917e2d0ab596a`.
- 2SM.csv shape: `5760x7`; extra columns: `catalyst_smiles`.
- Shared columns and Yield equal between SM.csv and 2SM.csv: `True`.
- Official y_true sequence equals generated KFold sequence: `True`.
- Official fold_id equals generated KFold sequence: `True`.

## Variant diffs vs official test summary

```json
{
  "float64_variant": {
    "mae": 1.522978020038491,
    "rmse": 1.8934656830144991,
    "r2": 0.058183486243709504,
    "kendall_tau": 0.04120544744249832
  },
  "two_sm_catalyst_variant": {
    "mae": 1.4937319043306747,
    "rmse": 1.8765107948321589,
    "r2": 0.0577006568410644,
    "kendall_tau": 0.04069945928011964
  }
}
```
