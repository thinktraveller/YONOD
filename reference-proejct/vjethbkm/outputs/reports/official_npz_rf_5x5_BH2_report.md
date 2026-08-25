# Official NPZ RF 5x5 benchmark: BH2

Run id: `official_npz_rf_5x5_20260825_175846`

This run uses official precomputed `.npz` descriptors from the local yieldsmarter package.
Raw `.npz` files remain ignored and are not copied into git.

```csv
stage,dataset,descriptor,model,mae_mean,rmse_mean,r2_mean,kendall_tau_mean,feature_dim,folds,train_elapsed_s_mean,feature_elapsed_s_mean,source_npz
official_npz_rf_5x5,BH2,MFP,rf,12.056001,17.892543,0.725046,0.684073,5120,25,13.620421,0.0,yieldsmarter/Results/Compare_Complexity/Denmark_2023/MFP/Denmark_MFP.npz
official_npz_rf_5x5,BH2,PhysChem,rf,13.861132,19.480151,0.674294,0.661056,36,25,0.469278,0.0,yieldsmarter/Results/Compare_Complexity/Denmark_2023/PhysChem/Denmark_PhysChem.npz
```
