# Official NPZ RF 5x5 benchmark: SM

Run id: `official_npz_rf_5x5_20260825_180439`

This run uses official precomputed `.npz` descriptors from the local yieldsmarter package.
Raw `.npz` files remain ignored and are not copied into git.

```csv
stage,dataset,descriptor,model,mae_mean,rmse_mean,r2_mean,kendall_tau_mean,feature_dim,folds,train_elapsed_s_mean,feature_elapsed_s_mean,source_npz
official_npz_rf_5x5,SM,MFP,rf,7.398946,11.008577,0.852799,0.761512,5120,25,11.798798,0.0,yieldsmarter/Results/Compare_Complexity/Suzuki_2018/MFP/Suzuki_MFP.npz
official_npz_rf_5x5,SM,PhysChem,rf,7.37378,10.860896,0.856724,0.76537,20,25,0.441876,0.0,yieldsmarter/Results/Compare_Complexity/Suzuki_2018/PhysChem/Suzuki_PhysChem.npz
```
