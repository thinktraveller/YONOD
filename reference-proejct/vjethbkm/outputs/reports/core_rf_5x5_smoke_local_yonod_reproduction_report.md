# VJETHBKM core_rf_5x5 reproduction report

Run id: `core_rf_5x5_20260825_150126`

This run validates `core_rf_5x5` on dataset `smoke_local_yonod`.
It is not a claim of numerical reproduction of the JACS paper.

## Metric summary

| stage | dataset | descriptor | model | mae_mean | rmse_mean | r2_mean | kendall_tau_mean | feature_dim | folds | train_elapsed_s_mean | feature_elapsed_s_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| core_rf_5x5 | smoke_local_yonod | physchem | rf | 0.332278 | 0.368695 | -86.429822 | 0.12 | 434 | 25 | 0.200433 | 0.111683 |
| core_rf_5x5 | smoke_local_yonod | morgan | rf | 0.355965 | 0.381806 | -75.958207 | 0.12 | 4096 | 25 | 0.209332 | 0.004025 |
| core_rf_5x5 | smoke_local_yonod | ohe | rf | 0.396517 | 0.425189 | -132.313914 | -0.125 | 13 | 25 | 0.203445 | 0.004599 |

## Evidence status

- Paper/SI numerical targets are still marked `pending_direct_si_audit`.
- Official ETH data/code package inspection is pending.
- Outputs were kept under `reference-proejct/vjethbkm/outputs/`.

Table S3 style summary: `D:\大创\YONOD\reference-proejct\vjethbkm\outputs\tables\table_s3_style_smoke_local_yonod_summary.csv`
