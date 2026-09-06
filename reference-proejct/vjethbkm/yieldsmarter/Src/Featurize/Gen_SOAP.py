import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
from ase.io import read
from dscribe.descriptors import SOAP

nSOAP = 105

cfg = json.load(open(sys.argv[1]))
df = pd.read_csv(cfg["reaction_csv"])

xyz_dir = Path(cfg.get("xyz_dir", "."))
target = cfg["target_col"]

# map reaction values -> lookup ID -> xyz path
xyz_cols = []
for comp in cfg["components"]:
    name = comp["name"]
    lk = pd.read_csv(comp["lookup_csv"]).rename(
        columns={comp.get("id_col", "ID"): f"{name}_ID"})
    # print('lk: ', lk.columns)
    # print('df1: ', df.columns)
    df = df.merge(
        lk[[comp["lookup_key_col"], f"{name}_ID"]],
        left_on=comp["reaction_col"],
        right_on=comp["lookup_key_col"],
        how="left",
    )
    # print('df2: ', df.columns)
    xyz_col = f"{name}_xyz"
    df[xyz_col] = df[f"{name}_ID"].apply(lambda i: str(xyz_dir / comp[
        "xyz_template"].format(id=int(i))) if pd.notna(i) else None)
    xyz_cols.append(xyz_col)

soap = SOAP(**cfg["soap"])


def featurize(row):
    vecs = []
    for col in xyz_cols:
        fp = row[col]
        if not fp or not Path(fp).exists():
            if cfg.get("skip_rows_with_missing_values", False):
                return None
            else:
                vec = np.zeros(nSOAP, np.float64)
        else:
            try:
                vec = soap.create(read(fp))
                assert len(
                    vec
                ) == nSOAP, f"Expected SOAP vector of length {nSOAP}, got {len(vec)} for file {fp}"
            except Exception:
                import traceback
                traceback.print_exc()
                assert (0)
                return None
        vecs.append(vec)
    return np.concatenate(vecs)


df["X"] = df.apply(featurize, axis=1)
dfv = df.dropna(subset=["X"])
print('Dropped rows with missing features:', len(df) - len(dfv))
X = np.vstack(dfv["X"].values)
y = dfv[target].values.astype(float)

out = Path(
    cfg.get(
        "output_npz",
        Path(cfg["reaction_csv"]).with_suffix("").name + "_soap_features.npz"))
out.parent.mkdir(parents=True, exist_ok=True)
np.savez(out, X=X, y=y)
print(out, X.shape, y.shape)
