# Official NPZ RF 5x5 benchmark: SLAP

Run id: `official_npz_rf_5x5_20260825_171526`

This run uses official precomputed `.npz` descriptors from the local yieldsmarter package.
Raw `.npz` files remain ignored and are not copied into git.

```csv
stage,dataset,descriptor,model,mae_mean,rmse_mean,r2_mean,kendall_tau_mean,feature_dim,folds,train_elapsed_s_mean,feature_elapsed_s_mean,source_npz
official_npz_rf_5x5,SLAP,MFP,rf,18.043381,35.122996,0.768195,0.509515,3072,25,1.371923,0.0,yieldsmarter/Results/Compare_Complexity/Bode_2023/MFP/Bode_MFP.npz
official_npz_rf_5x5,SLAP,PhysChem,rf,18.425319,34.687128,0.773573,0.490096,39,25,0.393599,0.0,yieldsmarter/Results/Compare_Complexity/Bode_2023/PhysChem/Bode_PhysChem.npz
```
