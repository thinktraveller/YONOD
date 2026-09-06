import numpy as np
import pandas as pd
import argparse
from rdkit.Chem import MolFromSmiles
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator

# Input parameters
RADIUS = 3
NUM_BITS = 1024
smiles_cache = {}


# Cache unique SMILES
# No need to run all, just unique identities
def smiles_to_mols(smiles_list):
    mols = []
    for s in smiles_list:
        if pd.isna(s) or s == "":
            mol = None
        else:
            mol = smiles_cache.get(s)
            if mol is None:
                mol = MolFromSmiles(str(s))
                smiles_cache[s] = mol
        mols.append(mol)
    return mols


# Fingerprint Count Vectors
def get_morgan_fp(mol, mfpgen):
    if mol is None:
        return np.zeros(NUM_BITS, dtype=int)
    return np.array(mfpgen.GetCountFingerprintAsNumPy(mol))


# Concatenate component vectors of each non-yield column
def featurize_reaction_dataset(df, smiles_columns, yield_column):
    mfpgen = GetMorganGenerator(radius=RADIUS, fpSize=NUM_BITS)
    all_mol_lists = [smiles_to_mols(df[col]) for col in smiles_columns]
    X = []
    for i in range(len(df)):
        fps = [get_morgan_fp(mols[i], mfpgen) for mols in all_mol_lists]
        X.append(np.concatenate(fps))
    return np.array(X), df[yield_column].to_numpy()


def process_and_save_dataset(input_file,
                             output_file,
                             skip_rows_with_missing_values=False):
    df = pd.read_csv(input_file)
    if "Yield" not in df.columns:
        raise ValueError(f"'Yield' column not found in {input_file}.")
    yield_col = "Yield"
    smiles_cols = [c for c in df.columns if c != yield_col]
    if skip_rows_with_missing_values:
        initial_len = len(df)
        df = df.dropna(subset=smiles_cols + [yield_col])
        print(f"Dropped {initial_len - len(df)} rows with missing values.")
    if not smiles_cols:
        raise ValueError("No SMILES columns detected.")
    X, y = featurize_reaction_dataset(df, smiles_cols, yield_col)
    np.savez(output_file,
             X=X,
             y=y,
             smiles_columns=np.array(smiles_cols),
             yield_column=yield_col)
    print(
        f"Output file Saved as {output_file} with X Shape: X={X.shape},y Shape: y={y.shape}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file",
                        help="CSV file with SMILES + Yield columns")
    parser.add_argument("output_file", help="Output .npz file")
    parser.add_argument("--skip_rows_with_missing_values",
                        action="store_true",
                        default=False,
                        help="Skip rows with missing SMILES or Yield values")
    args = parser.parse_args()
    process_and_save_dataset(args.input_file, args.output_file,
                             args.skip_rows_with_missing_values)
