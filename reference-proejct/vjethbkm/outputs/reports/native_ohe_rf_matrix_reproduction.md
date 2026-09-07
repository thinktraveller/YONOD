# Four-dataset OHE + RF independent paper-native rerun

This mirrors the unmodified `Train_OHE.py` OHE/RF branch: each fold fits `OneHotEncoder(handle_unknown='ignore')` on the training rows only, zeros whole missing-component blocks, then fits RF with the paper's 500 trees, 0.30 max_features and repeat seeds 1000–1004. The upstream file remains unmodified, but its unconditional unavailable LightGBM import requires this RF-only wrapper. YONOD is not used or modified.

- The report compares all 25 fold metrics with the bundled official JSON and the concatenated OOF arrays with the bundled official OOF artifact.
- SI records RDKit 2025.03.6 and scikit-learn 1.6.1; actual installed versions are in the JSON. Therefore a differing-stack run is not claimed to be bit-identical across environments.

| Dataset | Rows | Metric | Official JSON mean ± std | Independent native mean ± std | Δ mean | Δ std | OOF y/fold/pred exact | max |Δ pred| |
| --- | ---: | --- | --- | --- | ---: | ---: | --- | ---: |
| BH1 | 3955 | mae | 6.43266540035 ± 0.294522282824 | 6.43266540035 ± 0.294522282824 | 0 | 5.55111512313e-17 | True/True/False | 9.94759830064e-14 |
| BH1 | 3955 | rmse | 9.30357961068 ± 0.474888622809 | 9.30357961068 ± 0.474888622809 | 0 | -2.22044604925e-16 | True/True/False | 9.94759830064e-14 |
| BH1 | 3955 | r2 | 0.883127662199 ± 0.0132260174298 | 0.883127662199 ± 0.0132260174298 | 0 | 0 | True/True/False | 9.94759830064e-14 |
| BH1 | 3955 | kendall_tau | 0.795211038939 ± 0.0109041461379 | 0.795211038939 ± 0.0109041461379 | 0 | 0 | True/True/False | 9.94759830064e-14 |
| BH2 | 3359 | mae | 15.2512853229 ± 0.564305490037 | 15.2512853229 ± 0.564305490037 | 0 | 0 | True/True/False | 1.42108547152e-14 |
| BH2 | 3359 | rmse | 21.8544681283 ± 0.857834050618 | 21.8544681283 ± 0.857834050618 | 0 | 0 | True/True/False | 1.42108547152e-14 |
| BH2 | 3359 | r2 | 0.589908137569 ± 0.0323653137969 | 0.589908137569 ± 0.0323653137969 | 0 | 0 | True/True/False | 1.42108547152e-14 |
| BH2 | 3359 | kendall_tau | 0.614065273409 ± 0.0171160235533 | 0.614065273409 ± 0.0171160235533 | 0 | 0 | True/True/False | 1.42108547152e-14 |
| SL1 | 1150 | mae | 21.9742464301 ± 2.79454839636 | 21.9742464301 ± 2.79454839636 | 0 | 0 | True/True/False | 1.70530256582e-13 |
| SL1 | 1150 | rmse | 45.8611039514 ± 7.60491366675 | 45.8611039514 ± 7.60491366675 | 0 | 0 | True/True/False | 1.70530256582e-13 |
| SL1 | 1150 | r2 | 0.605828343629 ± 0.0970297253604 | 0.605828343629 ± 0.0970297253604 | -1.11022302463e-16 | -1.38777878078e-17 | True/True/False | 1.70530256582e-13 |
| SL1 | 1150 | kendall_tau | 0.481705158224 ± 0.0387478622445 | 0.481705158224 ± 0.0387478622445 | 0 | 0 | True/True/False | 1.70530256582e-13 |
| SM | 5760 | mae | 9.36293340915 ± 0.244897680837 | 9.36293340915 ± 0.244897680837 | 0 | -2.77555756156e-17 | True/True/False | 8.52651282912e-14 |
| SM | 5760 | rmse | 13.0592879892 ± 0.320226246043 | 13.0592879892 ± 0.320226246043 | 0 | 0 | True/True/False | 8.52651282912e-14 |
| SM | 5760 | r2 | 0.783424395583 ± 0.00971820019502 | 0.783424395583 ± 0.00971820019502 | 0 | 0 | True/True/False | 8.52651282912e-14 |
| SM | 5760 | kendall_tau | 0.710176550158 ± 0.00895318685044 | 0.710176550158 ± 0.00895318685044 | 0 | 0 | True/True/False | 8.52651282912e-14 |
