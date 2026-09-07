# SL1 MFP + RF independent paper-native rerun

This independently regenerates MFP features from the official SL1 CSV using unmodified bundled `Gen_MFP.py`, then mirrors the RF-only branch of bundled `Train_descriptors.py`. It does not train from the official MFP NPZ and does not use or modify YONOD.

## Provenance and environment

- Official CSV SHA256: `3fc0481e46d69bf8657b739a3872ba4c77682b68fd8b3af64fac6feaefc42d6a`
- Generated MFP vs bundled NPZ: X exact `True`, y exact `True`, component columns exact `True`.
- Actual versions: `{"numpy": "1.26.4", "pandas": "3.0.5", "python": "3.11.15", "rdkit": "2026.3.4", "rdkit_runtime": "2026.03.4", "scikit-learn": "1.4.0", "scipy": "1.17.1"}`.
- Paper SI reports RDKit 2025.03.6 and scikit-learn 1.6.1. This run records version drift and cannot claim bit-identical reproduction on a differing stack.
- The unmodified upstream trainer imports unavailable LightGBM before selecting RF. The wrapper preserves its RF parameters, KFold schedule, random seeds and fold metrics without editing upstream files.

## 5 × 5 result comparison

| Metric | Table S3 mean ± std | Official JSON mean ± std | Independent native mean ± std | Native − official mean | Native − official std |
| --- | --- | --- | --- | ---: | ---: |
| mae | 18.0 ± 1.8 | 18.0433812538 ± 1.84066736104 | 18.0433812538 ± 1.84066736104 | 0 | -6.66133814775e-16 |
| rmse | 35.1 ± 5.1 | 35.1229956765 ± 5.08988909135 | 35.1229956765 ± 5.08988909135 | 0 | 8.881784197e-16 |
| r2 | 0.77 ± 0.05 | 0.768194878222 ± 0.052291947803 | 0.768194878222 ± 0.052291947803 | 0 | 0 |
| kendall_tau | 0.51 ± 0.04 | 0.509515292576 ± 0.0357339404448 | 0.509515292576 ± 0.0357339404448 | 0 | 0 |

## Scope boundary

This increment covers SL1/MFP/RF only. It does not validate OHE, BH1, BH2 or SM; in particular, the SM 4620-versus-5760 conflict is neither resolved nor silently selected.
