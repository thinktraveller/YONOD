# Official NPZ RF 5x5 benchmark: BH

Run id: `official_npz_rf_5x5_20260825_172353`

This run uses official precomputed `.npz` descriptors from the local yieldsmarter package.
Raw `.npz` files remain ignored and are not copied into git.

```csv
stage,dataset,descriptor,model,mae_mean,rmse_mean,r2_mean,kendall_tau_mean,feature_dim,folds,train_elapsed_s_mean,feature_elapsed_s_mean,source_npz
official_npz_rf_5x5,BH,MFP,rf,4.531535,6.851956,0.936456,0.848426,4096,25,8.134272,0.0,yieldsmarter/Results/Compare_Complexity/Doyle_2018/MFP/Doyle_MFP.npz
official_npz_rf_5x5,BH,PhysChem,rf,6.044628,8.750706,0.896515,0.810886,15,25,0.424582,0.0,yieldsmarter/Results/Compare_Complexity/Doyle_2018/PhysChem/Doyle_PhysChem.npz
```
