# VJETHBKM reproduction status report

MFP and PhysChem RF Table S3 targets are reproduced through official precomputed NPZ features. OHE is a local compatibility rerun; SM/OHE remains a documented unresolved official artifact difference.

## Table S3 coverage

- Rows: `48`
- Matched local rows: `48`
- Unresolved rows: `4`

```csv
descriptor,rows,matched,max_abs_diff
MFP,16,16,0.0
OHE,16,16,1.9006496070565329
PhysChem,16,16,1.7763568394002505e-15
```

## Unresolved scope

- `SM/OHE/kendall_tau`
- `SM/OHE/mae`
- `SM/OHE/r2`
- `SM/OHE/rmse`

## BH2 external 187 summary

```csv
rows,predict_mae_from_columns,ours_mae_from_columns,ae_predict_mean,ae_ours_mean
187,20.240641711229948,18.610533011401078,20.240641711229948,18.610533011422454
```

## Available downstream analyses

- High-yield summary rows: `12`
- Component split schema rows: `25`
- Detailed tables are stored under `outputs/tables/`.
