import pandas as pd
import os
import re
import argparse

def slug(x): return re.sub(r'[^A-Za-z0-9._-]+', '_', str(x))

def main(inp, outdir):
    df = pd.read_csv(inp)
    os.makedirs(outdir, exist_ok=True)
    for prod in sorted(df['Product'].unique()):
        s = slug(prod)
        df_p = df[df['Product'] == prod].copy()

        # 1. full per-product file (keeps Product)
        df_p.to_csv(os.path.join(outdir, f"Product_{s}.csv"), index=False)

        # columns except Product
        cols_no_product = [c for c in df_p.columns if c != 'Product']

        # 2. DFT_Yield -> Yield (drop Product + other yield)
        dft = (df_p[cols_no_product]
               .drop(columns=['True_Yield'], errors='ignore')
               .rename(columns={'DFT_Yield': 'Yield'}))
        dft.to_csv(os.path.join(outdir, f"Product_{s}_DFT_Yield.csv"), index=False)

        # 3. True_Yield -> Yield (drop Product + other yield)
        true = (df_p[cols_no_product]
                .drop(columns=['DFT_Yield'], errors='ignore')
                .rename(columns={'True_Yield': 'Yield'}))
        true.to_csv(os.path.join(outdir, f"Product_{s}_True_Yield.csv"), index=False)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("input_csv")
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()
    outdir = args.outdir or os.path.join(os.path.dirname(args.input_csv), "Per_Product")
    main(args.input_csv, outdir)

