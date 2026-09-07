# VJETHBKM MFP/OHE + RF paper-native reproduction acceptance

**Result: 32/32 metrics pass strict 1e-12 numeric comparison.**

This package reproduces the paper's four MFP and four OHE RF 5×5 results only from bundled `yieldsmarter` CSVs, feature/training logic and official artifacts. It does not use or modify YONOD. The unmodified upstream trainers cannot import locally because they import unavailable LightGBM before selecting RF; narrow RF-only wrappers preserve the upstream RF parameters, seeds, folds, metrics and feature logic without editing source files.

## Population and protocol evidence

- MFP: each official CSV was passed to unmodified `Gen_MFP.py`; every MFP X/y/component-column artifact is exact. SM uses the original generator's `--skip_rows_with_missing_values`, transforming 5760 raw rows to 4620 complete-component rows.
- OHE: each fold fits OHE on train rows only, zeros missing-component blocks and ignores unseen values. Official OOF targets and fold IDs are exact; predictions differ only by at most 1.71e-13, below the explicit 1e-12 acceptance tolerance.
- Environment caveat: SI reports RDKit 2025.03.6 and scikit-learn 1.6.1; actual versions are in the JSON. The exact numerical result is evidence of a compatible rerun, not a claim that the package pins the paper's original environment.

## Metric acceptance

| Dataset | Descriptor | Metric | Population | Table S3 | Official JSON mean ± std | Native mean ± std | Δ mean | Δ std | Status |
| --- | --- | --- | ---: | --- | --- | --- | ---: | ---: | --- |
| BH1 | MFP | mae | 3955 | 4.5 ± 0.2 | 4.53153549235 ± 0.214447935482 | 4.53153549235 ± 0.214447935482 | 0 | 5.55111512313e-17 | pass |
| BH1 | MFP | rmse | 3955 | 6.9 ± 0.5 | 6.85195562732 ± 0.475351569628 | 6.85195562732 ± 0.475351569628 | 0 | 5.55111512313e-17 | pass |
| BH1 | MFP | r2 | 3955 | 0.94 ± 0.01 | 0.936455587292 ± 0.00956762581861 | 0.936455587292 ± 0.00956762581861 | 0 | 0 | pass |
| BH1 | MFP | kendall_tau | 3955 | 0.85 ± 0.01 | 0.848425971509 ± 0.00893344311275 | 0.848425971509 ± 0.00893344311275 | 0 | 0 | pass |
| BH2 | MFP | mae | 3359 | 12.1 ± 0.5 | 12.0560006864 ± 0.52733588712 | 12.0560006864 ± 0.52733588712 | 0 | 0 | pass |
| BH2 | MFP | rmse | 3359 | 17.9 ± 0.8 | 17.8925433355 ± 0.835834377258 | 17.8925433355 ± 0.835834377258 | 0 | 0 | pass |
| BH2 | MFP | r2 | 3359 | 0.73 ± 0.02 | 0.725046190983 ± 0.0247269447555 | 0.725046190983 ± 0.0247269447555 | 0 | 0 | pass |
| BH2 | MFP | kendall_tau | 3359 | 0.68 ± 0.01 | 0.684072586806 ± 0.0127806133304 | 0.684072586806 ± 0.0127806133304 | 0 | 0 | pass |
| SL1 | MFP | mae | 1150 | 18.0 ± 1.8 | 18.0433812538 ± 1.84066736104 | 18.0433812538 ± 1.84066736104 | -3.5527136788e-15 | -6.66133814775e-16 | pass |
| SL1 | MFP | rmse | 1150 | 35.1 ± 5.1 | 35.1229956765 ± 5.08988909135 | 35.1229956765 ± 5.08988909135 | -7.1054273576e-15 | -8.881784197e-16 | pass |
| SL1 | MFP | r2 | 1150 | 0.77 ± 0.05 | 0.768194878222 ± 0.052291947803 | 0.768194878222 ± 0.052291947803 | 0 | 1.38777878078e-17 | pass |
| SL1 | MFP | kendall_tau | 1150 | 0.51 ± 0.04 | 0.509515292576 ± 0.0357339404448 | 0.509515292576 ± 0.0357339404448 | 0 | 0 | pass |
| SM | MFP | mae | 4620 | 7.4 ± 0.2 | 7.39894562131 ± 0.219973943229 | 7.39894562131 ± 0.219973943229 | 0 | 1.11022302463e-16 | pass |
| SM | MFP | rmse | 4620 | 11.0 ± 0.3 | 11.0085774055 ± 0.322999087671 | 11.0085774055 ± 0.322999087671 | 0 | 0 | pass |
| SM | MFP | r2 | 4620 | 0.85 ± 0.01 | 0.852799480798 ± 0.00865236832922 | 0.852799480798 ± 0.00865236832922 | 0 | 0 | pass |
| SM | MFP | kendall_tau | 4620 | 0.76 ± 0.01 | 0.761511588057 ± 0.00836490121226 | 0.761511588057 ± 0.00836490121226 | 0 | 0 | pass |
| BH1 | OHE | mae | 3955 | 6.4 ± 0.3 | 6.43266540035 ± 0.294522282824 | 6.43266540035 ± 0.294522282824 | 0 | 5.55111512313e-17 | pass |
| BH1 | OHE | rmse | 3955 | 9.3 ± 0.5 | 9.30357961068 ± 0.474888622809 | 9.30357961068 ± 0.474888622809 | 0 | -2.22044604925e-16 | pass |
| BH1 | OHE | r2 | 3955 | 0.88 ± 0.01 | 0.883127662199 ± 0.0132260174298 | 0.883127662199 ± 0.0132260174298 | 0 | 0 | pass |
| BH1 | OHE | kendall_tau | 3955 | 0.8 ± 0.01 | 0.795211038939 ± 0.0109041461379 | 0.795211038939 ± 0.0109041461379 | 0 | 0 | pass |
| BH2 | OHE | mae | 3359 | 15.3 ± 0.6 | 15.2512853229 ± 0.564305490037 | 15.2512853229 ± 0.564305490037 | 0 | 0 | pass |
| BH2 | OHE | rmse | 3359 | 21.9 ± 0.9 | 21.8544681283 ± 0.857834050618 | 21.8544681283 ± 0.857834050618 | 0 | 0 | pass |
| BH2 | OHE | r2 | 3359 | 0.59 ± 0.03 | 0.589908137569 ± 0.0323653137969 | 0.589908137569 ± 0.0323653137969 | 0 | 0 | pass |
| BH2 | OHE | kendall_tau | 3359 | 0.61 ± 0.02 | 0.614065273409 ± 0.0171160235533 | 0.614065273409 ± 0.0171160235533 | 0 | 0 | pass |
| SL1 | OHE | mae | 1150 | 22.0 ± 2.8 | 21.9742464301 ± 2.79454839636 | 21.9742464301 ± 2.79454839636 | 0 | 0 | pass |
| SL1 | OHE | rmse | 1150 | 45.9 ± 7.6 | 45.8611039514 ± 7.60491366675 | 45.8611039514 ± 7.60491366675 | 0 | 0 | pass |
| SL1 | OHE | r2 | 1150 | 0.61 ± 0.1 | 0.605828343629 ± 0.0970297253604 | 0.605828343629 ± 0.0970297253604 | -1.11022302463e-16 | -1.38777878078e-17 | pass |
| SL1 | OHE | kendall_tau | 1150 | 0.48 ± 0.04 | 0.481705158224 ± 0.0387478622445 | 0.481705158224 ± 0.0387478622445 | 0 | 0 | pass |
| SM | OHE | mae | 5760 | 9.4 ± 0.2 | 9.36293340915 ± 0.244897680837 | 9.36293340915 ± 0.244897680837 | 0 | -2.77555756156e-17 | pass |
| SM | OHE | rmse | 5760 | 13.1 ± 0.3 | 13.0592879892 ± 0.320226246043 | 13.0592879892 ± 0.320226246043 | 0 | 0 | pass |
| SM | OHE | r2 | 5760 | 0.78 ± 0.01 | 0.783424395583 ± 0.00971820019502 | 0.783424395583 ± 0.00971820019502 | 0 | 0 | pass |
| SM | OHE | kendall_tau | 5760 | 0.71 ± 0.01 | 0.710176550158 ± 0.00895318685044 | 0.710176550158 ± 0.00895318685044 | 0 | 0 | pass |

## Scope boundary

This accepts Table S3's MFP/OHE + RF subset, not other descriptors, non-RF models, Figure 5's five-descriptor interval comparison, BH2 external validation, reweighting or generalization experiments. SM still has distinct MFP=4620 and OHE=5760 population definitions; the original native paths now make both operationally reproducible, while the paper text/artifact provenance distinction remains recorded rather than silently merged.
