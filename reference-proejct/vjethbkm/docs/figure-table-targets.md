# Figure and table targets

Last updated: 2026-08-25

This file maps paper targets to implementation priorities. Numerical paper
values are deliberately marked as pending until the SI/main-paper tables are
read directly from Z-AIHub attachments or the official files.

| target_id | source | dataset | descriptors | model | split/evaluation | metric | priority | paper_value_status | YONOD implementation |
|---|---|---|---|---|---|---|---|---|---|
| main_descriptor_tradeoff | Main article discussion and SI S2/S3 | BH, SM, SLAP | OHE, MFP, PhysChem, DFT, SOAP | RF | repeated CV | MAE, RMSE, R2, Kendall tau | A/B | pending_direct_si_audit | Stage A implements OHE/MFP/PhysChem + RF; DFT/SOAP remain registered extensions. |
| main_component_generalization | Main article discussion | public HTE datasets | low-cost and high-cost descriptors | RF | component-wise 0D/1D/2D-like splits | MAE, RMSE, R2, Kendall tau | B | pending_direct_si_audit | Stage B adds component holdout manifests and leakage checks. |
| main_external_validation | Main article discussion | BH1/BH2 | descriptor matrix | RF | external validation | MAE, RMSE, R2, Kendall tau | B | pending_direct_si_audit | Stage B requires official dataset identity and no fit on external holdout. |
| main_yield_distribution | Main article Figure 2 and discussion | HTE and literature/patent contrast | n/a | n/a | descriptive statistics | yield bins | A/C | qualitative_verified_public_page | Stage A records yield bins; Stage C adds imbalance metrics and high-yield recall. |
| si_rf_cv | SI S3 | per official dataset | descriptor matrix | RF | cross-validation | fold metrics | B | pending_direct_si_audit | Consumed by `figure_table_comparison.csv` after data package inspection. |
| si_classical_algorithms | SI S4 | per official dataset | descriptor matrix | RF, Ridge, KNN, LightGBM-like extensions | cross-validation | model metrics | C | pending_direct_si_audit | Registered as extension model matrix. |
| si_reweighting | SI S5 | per official dataset | selected descriptors | RF/other selected models | weighted vs unweighted training | global and binned metrics | C | pending_direct_si_audit | Registered as imbalance/reweighting extension. |

## Acceptance vocabulary

| status | meaning |
|---|---|
| matched | Dataset, split, descriptor, model, and metric definition are aligned, and reproduced values are within the chosen tolerance. |
| trend_only | The qualitative trend matches, but numerical differences remain because data, split, descriptor, dependency, or random seed details differ. |
| not_reproduced | Neither numerical values nor qualitative trends match; requires root-cause analysis. |
| pending_direct_si_audit | The target is known from article/SI structure, but exact values have not yet been extracted from Z-AIHub or official files. |

