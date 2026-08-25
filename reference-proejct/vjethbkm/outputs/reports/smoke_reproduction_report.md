# VJETHBKM smoke reproduction report

Run id: `smoke_20260825_140758`

This smoke run validates the Stage A pipeline shape on a small local YONOD dataset.
It is not a claim of numerical reproduction of the JACS paper.

## Metric summary

| stage | dataset | descriptor | model | mae_mean | rmse_mean | r2_mean | kendall_tau_mean | feature_dim | folds | train_elapsed_s_mean | feature_elapsed_s_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| smoke | smoke_local_yonod | physchem | rf | 0.371915 | 0.422108 | -1.717151 | 0.3 | 434 | 2 | 0.220702 | 0.105642 |
| smoke | smoke_local_yonod | morgan | rf | 0.383658 | 0.42273 | -1.719106 | -0.4 | 4096 | 2 | 0.191018 | 0.003898 |
| smoke | smoke_local_yonod | ohe | rf | 0.433384 | 0.498947 | -2.799861 | 0.136943 | 9 | 2 | 0.186466 | 0.015496 |

## Evidence status

- Paper/SI numerical targets are still marked `pending_direct_si_audit`.
- Official ETH data/code package inspection is pending.
- Outputs were kept under `reference-proejct/vjethbkm/outputs/`.
