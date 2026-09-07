# Four-dataset MFP + RF independent paper-native rerun

Each MFP is regenerated from the paper's official CSV using unmodified `Gen_MFP.py`; no bundled NPZ is used for training. The RF-only wrapper mirrors the parameters, seeds, KFold schedule and metrics of `Train_descriptors.py`, whose unmodified module cannot import locally because it imports unavailable LightGBM before RF selection. YONOD is not used or modified.

- SI versions: RDKit 2025.03.6, scikit-learn 1.6.1. Actual versions are recorded in the JSON, so this differing-stack result is not claimed to be bit-identical across environments.
- SM uniquely passes the bundled `--skip_rows_with_missing_values` flag: the original 5760 CSV rows become the SI-described 4620 complete-component rows. This is an explicit MFP population choice, not an OHE result.

| Dataset | Metric | Official JSON mean ± std | Independent native mean ± std | Δ mean | Δ std | MFP X/y/columns exact |
| --- | --- | --- | --- | ---: | ---: | --- |
| BH1 | mae | 4.53153549235 ± 0.214447935482 | 4.53153549235 ± 0.214447935482 | 0 | 5.55111512313e-17 | True/True/True |
| BH1 | rmse | 6.85195562732 ± 0.475351569628 | 6.85195562732 ± 0.475351569628 | 0 | 5.55111512313e-17 | True/True/True |
| BH1 | r2 | 0.936455587292 ± 0.00956762581861 | 0.936455587292 ± 0.00956762581861 | 0 | 0 | True/True/True |
| BH1 | kendall_tau | 0.848425971509 ± 0.00893344311275 | 0.848425971509 ± 0.00893344311275 | 0 | 0 | True/True/True |
| BH2 | mae | 12.0560006864 ± 0.52733588712 | 12.0560006864 ± 0.52733588712 | 0 | 0 | True/True/True |
| BH2 | rmse | 17.8925433355 ± 0.835834377258 | 17.8925433355 ± 0.835834377258 | 0 | 0 | True/True/True |
| BH2 | r2 | 0.725046190983 ± 0.0247269447555 | 0.725046190983 ± 0.0247269447555 | 0 | 0 | True/True/True |
| BH2 | kendall_tau | 0.684072586806 ± 0.0127806133304 | 0.684072586806 ± 0.0127806133304 | 0 | 0 | True/True/True |
| SL1 | mae | 18.0433812538 ± 1.84066736104 | 18.0433812538 ± 1.84066736104 | -3.5527136788e-15 | -6.66133814775e-16 | True/True/True |
| SL1 | rmse | 35.1229956765 ± 5.08988909135 | 35.1229956765 ± 5.08988909135 | -7.1054273576e-15 | -8.881784197e-16 | True/True/True |
| SL1 | r2 | 0.768194878222 ± 0.052291947803 | 0.768194878222 ± 0.052291947803 | 0 | 1.38777878078e-17 | True/True/True |
| SL1 | kendall_tau | 0.509515292576 ± 0.0357339404448 | 0.509515292576 ± 0.0357339404448 | 0 | 0 | True/True/True |
| SM | mae | 7.39894562131 ± 0.219973943229 | 7.39894562131 ± 0.219973943229 | 0 | 1.11022302463e-16 | True/True/True |
| SM | rmse | 11.0085774055 ± 0.322999087671 | 11.0085774055 ± 0.322999087671 | 0 | 0 | True/True/True |
| SM | r2 | 0.852799480798 ± 0.00865236832922 | 0.852799480798 ± 0.00865236832922 | 0 | 0 | True/True/True |
| SM | kendall_tau | 0.761511588057 ± 0.00836490121226 | 0.761511588057 ± 0.00836490121226 | 0 | 0 | True/True/True |
