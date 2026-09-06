# Final MFP/OHE + RF metrics from official YieldSmarter artifacts

Each mean and standard deviation summarizes 25 fold-level values from 5 repeats × 5 folds. This is an official-artifact summary, not an independent retraining claim. SM/MFP uses 4620 rows, while bundled SM/OHE artifacts use 5760 rows; this remains an audit boundary.

| Dataset | Descriptor | Metric | Mean ± std | Unit | Valid folds | Population | Table S3 display |
| --- | --- | --- | --- | --- | ---: | --- | --- |
| BH1 | MFP | kendall_tau | 0.85 ± 0.01 | Yield (%) | 25 | official CSV: 3955 reactions | True |
| BH1 | MFP | mae | 4.5 ± 0.2 | Yield (%) | 25 | official CSV: 3955 reactions | True |
| BH1 | MFP | r2 | 0.94 ± 0.01 | Yield (%) | 25 | official CSV: 3955 reactions | True |
| BH1 | MFP | rmse | 6.9 ± 0.5 | Yield (%) | 25 | official CSV: 3955 reactions | True |
| BH1 | OHE | kendall_tau | 0.80 ± 0.01 | Yield (%) | 25 | official CSV: 3955 reactions | True |
| BH1 | OHE | mae | 6.4 ± 0.3 | Yield (%) | 25 | official CSV: 3955 reactions | True |
| BH1 | OHE | r2 | 0.88 ± 0.01 | Yield (%) | 25 | official CSV: 3955 reactions | True |
| BH1 | OHE | rmse | 9.3 ± 0.5 | Yield (%) | 25 | official CSV: 3955 reactions | True |
| BH2 | MFP | kendall_tau | 0.68 ± 0.01 | Yield (%) | 25 | official CSV: 3359 reactions | True |
| BH2 | MFP | mae | 12.1 ± 0.5 | Yield (%) | 25 | official CSV: 3359 reactions | True |
| BH2 | MFP | r2 | 0.73 ± 0.02 | Yield (%) | 25 | official CSV: 3359 reactions | True |
| BH2 | MFP | rmse | 17.9 ± 0.8 | Yield (%) | 25 | official CSV: 3359 reactions | True |
| BH2 | OHE | kendall_tau | 0.61 ± 0.02 | Yield (%) | 25 | official CSV: 3359 reactions | True |
| BH2 | OHE | mae | 15.3 ± 0.6 | Yield (%) | 25 | official CSV: 3359 reactions | True |
| BH2 | OHE | r2 | 0.59 ± 0.03 | Yield (%) | 25 | official CSV: 3359 reactions | True |
| BH2 | OHE | rmse | 21.9 ± 0.9 | Yield (%) | 25 | official CSV: 3359 reactions | True |
| SL1 | MFP | kendall_tau | 0.51 ± 0.04 | LC-MS product ratio | 25 | official CSV: 1150 reactions | True |
| SL1 | MFP | mae | 18.0 ± 1.8 | LC-MS product ratio | 25 | official CSV: 1150 reactions | True |
| SL1 | MFP | r2 | 0.77 ± 0.05 | LC-MS product ratio | 25 | official CSV: 1150 reactions | True |
| SL1 | MFP | rmse | 35.1 ± 5.1 | LC-MS product ratio | 25 | official CSV: 1150 reactions | True |
| SL1 | OHE | kendall_tau | 0.48 ± 0.04 | LC-MS product ratio | 25 | official CSV: 1150 reactions | True |
| SL1 | OHE | mae | 22.0 ± 2.8 | LC-MS product ratio | 25 | official CSV: 1150 reactions | True |
| SL1 | OHE | r2 | 0.61 ± 0.10 | LC-MS product ratio | 25 | official CSV: 1150 reactions | True |
| SL1 | OHE | rmse | 45.9 ± 7.6 | LC-MS product ratio | 25 | official CSV: 1150 reactions | True |
| SM | MFP | kendall_tau | 0.76 ± 0.01 | Yield (%) | 25 | SI all-components-specified subset: 4620 reactions | True |
| SM | MFP | mae | 7.4 ± 0.2 | Yield (%) | 25 | SI all-components-specified subset: 4620 reactions | True |
| SM | MFP | r2 | 0.85 ± 0.01 | Yield (%) | 25 | SI all-components-specified subset: 4620 reactions | True |
| SM | MFP | rmse | 11.0 ± 0.3 | Yield (%) | 25 | SI all-components-specified subset: 4620 reactions | True |
| SM | OHE | kendall_tau | 0.71 ± 0.01 | Yield (%) | 25 | official OHE artifact: raw SM.csv, 5760 reactions | True |
| SM | OHE | mae | 9.4 ± 0.2 | Yield (%) | 25 | official OHE artifact: raw SM.csv, 5760 reactions | True |
| SM | OHE | r2 | 0.78 ± 0.01 | Yield (%) | 25 | official OHE artifact: raw SM.csv, 5760 reactions | True |
| SM | OHE | rmse | 13.1 ± 0.3 | Yield (%) | 25 | official OHE artifact: raw SM.csv, 5760 reactions | True |
