import pandas as pd 
import numpy as np
import argparse
import os
import glob

def main(indir, outcsv):
    """
    Compute per-product mean absolute error (MAE) between experimental and DFT-predicted yields for the BH2 (Denmark-
    out-of-distribution dataset of 187 reactions).

    For each CSV file matching ``Product_*.csv`` in the input directory:
      - Read the experimental (``True_Yield``) and predicted (``DFT_Yield``) yields.
      - Compute the absolute error |True_Yield − DFT_Yield| for each reaction.
      - Aggregate errors by product and report the mean absolute error (MAE)
        along with the number of valid reactions contributing to that MAE.

    An overall MAE is also computed as a reaction-count–weighted average
    across all products.

    Parameters
    ----------
    indir : str
        Path to directory containing ``Product_*.csv`` files (excluding
        ``*_DFT_Yield.csv`` and ``*_True_Yield.csv``).
    outcsv : str
        Output CSV filename for the MAE summary table.
    """
    rows = []
    for f in sorted(glob.glob(os.path.join(indir, "Product_*.csv"))):
        if f.endswith("_DFT_Yield.csv") or f.endswith("_True_Yield.csv"):
            continue
        df = pd.read_csv(f)
        if not {"True_Yield", "DFT_Yield"}.issubset(df.columns):
            continue
        diff = (df["True_Yield"] - df["DFT_Yield"]).abs().dropna()
        if len(diff) == 0:
            continue
        product = os.path.splitext(os.path.basename(f))[0].replace("Product_", "")
        rows.append({"Product": product, "n": int(len(diff)), "MAE": float(diff.mean())})

    out = pd.DataFrame(rows).sort_values("Product")
    out.to_csv(outcsv, index=False)

    overall_mae = (out["MAE"] * out["n"]).sum() / out["n"].sum() if len(out) else np.nan
    print(out.to_string(index=False))
    print(f"\nOverall_MAE_across_all_lines: {overall_mae:.6f}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("indir", help="Directory with Product_*.csv files (NOT the _DFT_Yield/_True_Yield ones)")
    ap.add_argument("--outcsv", default="mae_summary.csv")
    a = ap.parse_args()
    main(a.indir, a.outcsv)

