# combine_mae_summaries.py
import pandas as pd, argparse, os

p = argparse.ArgumentParser()
p.add_argument("dft_csv")   # e.g. .../Per_Product/mae_summary.csv
p.add_argument("mf_csv")    # e.g. .../Per_Product_preds/mae_summary_preds.csv
p.add_argument("--out", default="combined_mae_summary.csv")
a = p.parse_args()

dft = pd.read_csv(a.dft_csv).rename(columns={"n":"n_DFT","MAE":"MAE_DFT"})
mf  = pd.read_csv(a.mf_csv ).rename(columns={"n":"n_MF","MAE":"MAE_MF"})

out = pd.merge(dft, mf, on="Product", how="outer").sort_values("Product")
if os.path.dirname(a.out): os.makedirs(os.path.dirname(a.out), exist_ok=True)
out.to_csv(a.out, index=False)
print(out)

