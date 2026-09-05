import io
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

try:
    import umap
    _HAS_UMAP = True
except Exception:
    _HAS_UMAP = False

try:
    import plotly.express as px
    _HAS_PLOTLY = True
except Exception:
    _HAS_PLOTLY = False

try:
    import seaborn as sns
    import matplotlib.pyplot as plt
    _HAS_SEABORN = True
except Exception:
    _HAS_SEABORN = False


# RDKit (SMILES featurization)
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, MACCSkeys, Descriptors
    _HAS_RDKIT = True
except Exception:
    _HAS_RDKIT = False


# -----------------------------
# App config
# -----------------------------
st.set_page_config(page_title="ChemSpace Builder", layout="wide")

if not _HAS_PLOTLY:
    st.error("This app requires plotly. Please install it in your environment.")
    st.stop()

if not _HAS_SEABORN:
    st.error("This app requires seaborn and matplotlib for academic-style plotting. Please install seaborn and matplotlib.")
    st.stop()


# -----------------------------
# Session state
# -----------------------------
PAGE_NAMES = ["Data", "Features", "Embedding", "Plot & Export"]

if "page" not in st.session_state:
    st.session_state.page = 0

if "datasets" not in st.session_state:
    # list of {name: str, df: pd.DataFrame}
    st.session_state.datasets: List[Dict[str, object]] = []

if "role_map" not in st.session_state:
    # role -> list[column]
    st.session_state.role_map = {
        "Substrate 1": [],
        "Substrate 2": [],
        "Substrate 3": [],
        "Product": [],
        "Other": [],
    }

if "build_mode" not in st.session_state:
    st.session_state.build_mode = "Whole reaction"

if "selected_rolecols" not in st.session_state:
    st.session_state.selected_rolecols = []  # list of "Role :: Column"


# -----------------------------
# Utilities
# -----------------------------

def _nav_controls():
    """Top minimal nav + bottom prev/next."""
    top = st.container()
    with top:
        cols = st.columns([6, 1.5, 1.5])
        with cols[0]:
            sel = st.radio(
                "Navigation",
                PAGE_NAMES,
                index=int(st.session_state.page),
                horizontal=True,
                label_visibility="collapsed",
            )
            st.session_state.page = PAGE_NAMES.index(sel)
        with cols[1]:
            if st.button("← Prev", use_container_width=True, disabled=st.session_state.page == 0, key="prev_top"):
                st.session_state.page = max(0, st.session_state.page - 1)
                st.rerun()
        with cols[2]:
            if st.button("Next →", use_container_width=True, disabled=st.session_state.page == len(PAGE_NAMES) - 1, key="next_top"):
                st.session_state.page = min(len(PAGE_NAMES) - 1, st.session_state.page + 1)
                st.rerun()


def _bottom_next_prev():
    st.markdown("---")
    cols = st.columns([6, 2, 2])
    with cols[0]:
        st.caption(f"Step {st.session_state.page + 1} / {len(PAGE_NAMES)} · {PAGE_NAMES[st.session_state.page]}")
    with cols[1]:
        if st.button("← Prev", use_container_width=True, disabled=st.session_state.page == 0, key="prev_bottom"):
            st.session_state.page = max(0, st.session_state.page - 1)
            st.rerun()
    with cols[2]:
        # IMPORTANT: Next should always work even if Data step isn't complete.
        if st.button("Next →", use_container_width=True, disabled=st.session_state.page == len(PAGE_NAMES) - 1, key="next_bottom"):
            st.session_state.page = min(len(PAGE_NAMES) - 1, st.session_state.page + 1)
            st.rerun()


def _all_columns(datasets: List[Dict[str, object]]) -> List[str]:
    cols = set()
    for d in datasets:
        df = d["df"]
        cols.update(df.columns.tolist())
    return sorted(cols)


def _rolecol_str(role: str, col: str) -> str:
    return f"{role} :: {col}"


def parse_rolecol(s: str) -> Tuple[str, str]:
    # Robust parsing even if column contains '::'
    if " :: " not in s:
        raise ValueError(f"Invalid role-column string: {s}")
    role, col = s.split(" :: ", 1)
    return role.strip(), col.strip()


@st.cache_data(show_spinner=False)
def _read_csv(file_name: str, file_bytes: bytes) -> pd.DataFrame:
    # Let pandas infer encoding
    return pd.read_csv(io.BytesIO(file_bytes))


def _clean_smiles_series(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().replace({"": np.nan, "nan": np.nan, "None": np.nan})


@st.cache_data(show_spinner=False)
def featurize_smiles(
    smiles_list: List[str],
    feature_mode: str,
    morgan_bits: int,
    morgan_radius: int,
    include_maccs: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (X, valid_mask).

    feature_mode options (UI):
      - MorganFP
      - RDKit descriptors
      - RDKit + MorganFP
      - RDKit + MorganFP + MACCS
    """
    if not _HAS_RDKIT:
        raise RuntimeError("RDKit is not available in this environment.")

    mode = str(feature_mode)

    use_desc = ("RDKit" in mode)  # RDKit descriptors
    use_morgan = ("MorganFP" in mode)
    use_maccs = ("MACCS" in mode) and bool(include_maccs)

    desc_list = Descriptors._descList if use_desc else []
    desc_dim = len(desc_list)
    fp_dim = int(morgan_bits) if use_morgan else 0
    maccs_dim = 167 if use_maccs else 0

    n = len(smiles_list)
    X = np.zeros((n, desc_dim + fp_dim + maccs_dim), dtype=np.float32)
    valid = np.zeros(n, dtype=bool)

    for i, smi in enumerate(smiles_list):
        if smi is None or (isinstance(smi, float) and np.isnan(smi)):
            continue
        smi = str(smi).strip()
        if not smi:
            continue
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue

        valid[i] = True
        offset = 0

        if use_desc:
            for j, (_name, fn) in enumerate(desc_list):
                try:
                    X[i, offset + j] = float(fn(mol))
                except Exception:
                    X[i, offset + j] = np.nan
            offset += desc_dim

        if use_morgan:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, int(morgan_radius), nBits=int(morgan_bits))
            arr = np.zeros((fp_dim,), dtype=np.int8)
            Chem.DataStructs.ConvertToNumpyArray(fp, arr)
            X[i, offset : offset + fp_dim] = arr
            offset += fp_dim

        if use_maccs:
            m = MACCSkeys.GenMACCSKeys(mol)
            marr = np.zeros((167,), dtype=np.int8)
            Chem.DataStructs.ConvertToNumpyArray(m, marr)
            X[i, offset : offset + 167] = marr

    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X, valid

@st.cache_data(show_spinner=False)
def embed_2d(
    X: np.ndarray,
    method: str,
    seed: int,
    umap_n_neighbors: int,
    umap_min_dist: float,
    tsne_n_components: int,
    tsne_perplexity: int,
    tsne_early_exaggeration: float,
    tsne_learning_rate: float,
    # tsne_n_iter: int,
    tsne_init: str,
    tsne_metric: str,
    pca_whiten: bool,
    pca_svd_solver: str,
) -> np.ndarray:
    Xs = StandardScaler().fit_transform(X)

    if method == "UMAP":
        if not _HAS_UMAP:
            raise RuntimeError("UMAP is not installed. Install umap-learn.")
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=int(umap_n_neighbors),
            min_dist=float(umap_min_dist),
            metric="cosine",
            random_state=int(seed),
        )
        return reducer.fit_transform(Xs)

    if method == "t-SNE":
        tsne = TSNE(
            n_components=int(tsne_n_components),
            perplexity=int(tsne_perplexity),
            early_exaggeration=float(tsne_early_exaggeration),
            learning_rate=float(tsne_learning_rate),
            # n_iter=int(tsne_n_iter),
            init=str(tsne_init),
            metric=str(tsne_metric),
            random_state=int(seed),
        )
        return tsne.fit_transform(Xs)

    # PCA fallback
    pca = PCA(
        n_components=2,
        whiten=bool(pca_whiten),
        svd_solver=str(pca_svd_solver),
        random_state=int(seed),
    )
    return pca.fit_transform(Xs)


def build_points_from_roles(
    datasets: List[Dict[str, object]],
    role_map: Dict[str, List[str]],
    selected_rolecols: List[str],
    sample_per_dataset: int,
    seed: int,
) -> pd.DataFrame:
    """Return long table of molecules to embed: one row = one molecule instance."""

    rng = np.random.default_rng(int(seed))
    frames = []

    for d in datasets:
        name = str(d["name"])
        df: pd.DataFrame = d["df"]
        if df is None or df.empty:
            continue

        # optional sampling at row-level (keeps reaction context)
        if sample_per_dataset and len(df) > sample_per_dataset:
            idx = rng.choice(len(df), size=int(sample_per_dataset), replace=False)
            df_use = df.iloc[idx].copy()
        else:
            df_use = df.copy()

        for rc in selected_rolecols:
            role, col = parse_rolecol(rc)
            if col not in df_use.columns:
                continue
            s = _clean_smiles_series(df_use[col])
            tmp = pd.DataFrame({
                "dataset": name,
                "dataset_source": str(d.get("source_name", name)),
                "role": role,
                "column": col,
                "smiles": s,
                "row_index": df_use.index.astype(int),
            })
            frames.append(tmp)

    if not frames:
        return pd.DataFrame(columns=["dataset", "role", "column", "smiles", "row_index"])

    out = pd.concat(frames, ignore_index=True)
    # drop missing smiles
    out = out.dropna(subset=["smiles"]).reset_index(drop=True)
    return out


def build_reaction_points_parity(
    datasets: List[Dict[str, object]],
    role_map: Dict[str, List[str]],
) -> Tuple[pd.DataFrame, List[str]]:
    """Notebook-parity: one row = one reaction.

    We build a reaction-level table using exactly three roles in a fixed order:
    Substrate 1, Substrate 2, Product (matching draw_fig.ipynb semantics).

    Returns:
        df_react: columns [dataset, dataset_source, row_index, smiles_0, smiles_1, smiles_2]
        smiles_cols: the underlying column names used (length=3)
    """
    roles_order = ["Substrate 1", "Substrate 2", "Product"]
    picked_cols = []
    for r in roles_order:
        cols = role_map.get(r, []) or []
        if len(cols) < 1:
            return pd.DataFrame(), []
        picked_cols.append(str(cols[0]))

    frames = []
    for d in datasets:
        ds_name = str(d.get("name", "dataset"))
        ds_src = str(d.get("source_name", ds_name))
        df: pd.DataFrame = d.get("df")
        if df is None or df.empty:
            continue
        missing = [c for c in picked_cols if c not in df.columns]
        if missing:
            continue
        tmp = df[picked_cols].copy()
        for c in picked_cols:
            tmp[c] = _clean_smiles_series(tmp[c])
        tmp["dataset"] = ds_name
        tmp["dataset_source"] = ds_src
        tmp["row_index"] = tmp.index.astype(int)
        frames.append(tmp)

    if not frames:
        return pd.DataFrame(), []

    df_all = pd.concat(frames, ignore_index=True)
    df_all = df_all.rename(columns={picked_cols[0]: "smiles_0", picked_cols[1]: "smiles_1", picked_cols[2]: "smiles_2"})
    return df_all.reset_index(drop=True), picked_cols


def featurize_reaction_parity(
    df_react: pd.DataFrame,
    morgan_bits: int = 2048,
    morgan_radius: int = 2,
) -> Tuple[np.ndarray, np.ndarray]:
    """Notebook-parity features: RDKit Properties + Morgan + MACCS, concatenated for 3 molecules.

    Invalid policy matches draw_fig.ipynb: drop any reaction row where ANY of the three SMILES is invalid.

    Returns:
        X: (n_reactions, 3*(n_props + morgan_bits + 167))
        keep_mask: boolean mask over df_react rows that were kept
    """
    if not _HAS_RDKIT:
        raise RuntimeError("RDKit not installed")

    from rdkit import Chem
    from rdkit.Chem import AllChem, MACCSkeys, rdMolDescriptors
    from rdkit import DataStructs

    prop_calc = rdMolDescriptors.Properties()
    n_props = len(list(prop_calc.GetPropertyNames()))

    def mol_from_smiles(s):
        if s is None or (isinstance(s, float) and np.isnan(s)):
            return None
        s = str(s).strip()
        if not s:
            return None
        return Chem.MolFromSmiles(s)

    def rdkit_props(m):
        return np.asarray(list(prop_calc.ComputeProperties(m)), dtype=np.float32)

    def morgan_bits_arr(m):
        fp = AllChem.GetMorganFingerprintAsBitVect(m, int(morgan_radius), nBits=int(morgan_bits))
        arr = np.zeros((int(morgan_bits),), dtype=np.float32)
        DataStructs.ConvertToNumpyArray(fp, arr)
        return arr

    def maccs_bits_arr(m):
        fp = MACCSkeys.GenMACCSKeys(m)  # 167
        arr = np.zeros((167,), dtype=np.float32)
        DataStructs.ConvertToNumpyArray(fp, arr)
        return arr

    mol0 = [mol_from_smiles(x) for x in df_react["smiles_0"].tolist()]
    mol1 = [mol_from_smiles(x) for x in df_react["smiles_1"].tolist()]
    mol2 = [mol_from_smiles(x) for x in df_react["smiles_2"].tolist()]
    keep = np.array([(m0 is not None) and (m1 is not None) and (m2 is not None) for m0, m1, m2 in zip(mol0, mol1, mol2)], dtype=bool)

    mol0 = [m for m, k in zip(mol0, keep) if k]
    mol1 = [m for m, k in zip(mol1, keep) if k]
    mol2 = [m for m, k in zip(mol2, keep) if k]

    if len(mol0) == 0:
        return np.zeros((0, 3 * (n_props + int(morgan_bits) + 167)), dtype=np.float32), keep

    def feat_block(mols):
        feats = []
        for m in mols:
            v = np.concatenate([rdkit_props(m), morgan_bits_arr(m), maccs_bits_arr(m)], axis=0)
            feats.append(v)
        return np.vstack(feats).astype(np.float32)

    X0 = feat_block(mol0)
    X1 = feat_block(mol1)
    X2 = feat_block(mol2)
    X = np.hstack([X0, X1, X2]).astype(np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X, keep


# -----------------------------
# UI
# -----------------------------
_nav_controls()

st.title("ChemSpace Builder")
st.caption("Upload reaction tables, map columns to roles, then build a 2D chemical space with RDKit + UMAP/t-SNE/PCA.")

page = int(st.session_state.page)

datasets: List[Dict[str, object]] = st.session_state.datasets
all_cols = _all_columns(datasets) if datasets else []

# -----------------------------
# Page 1: Data
# -----------------------------
if page == 0:
    st.subheader("1) Data")

    uploaded = st.file_uploader(
        "Upload one or more CSV files",
        type=["csv"],
        accept_multiple_files=True,
    )

    c1, c2 = st.columns([1, 3])
    with c1:
        clear = st.button("Clear loaded datasets")
    if clear:
        st.session_state.datasets = []
        datasets = st.session_state.datasets
        st.success("Cleared.")

    if uploaded:
        new = []
        for f in uploaded:
            try:
                df = _read_csv(f.name, f.getvalue())
                new.append({"name": f.name, "source_name": f.name, "df": df})
            except Exception as e:
                st.warning(f"Failed to read {f.name}: {e}")

        if new:
            st.session_state.datasets = new
            datasets = st.session_state.datasets
            st.success(f"Loaded {len(datasets)} dataset(s).")

    if not datasets:
        st.info("No datasets loaded yet. You can still click Next to configure settings, but plotting requires data.")
    else:
        st.write("Preview")
        for i, d in enumerate(datasets):
            df = d["df"]
            # Allow user-defined dataset display name (kept across steps)
            new_name = st.text_input(
                "Dataset name",
                value=str(d.get("name", "")),
                key=f"dataset_display_name_{i}",
                help="This name is used in hover / tables. The original filename is kept internally.",
            )
            d["name"] = new_name.strip() if str(new_name).strip() else str(d.get("name", ""))

            # Show original filename (for reproducibility / auto-style mapping)
            if d.get("source_name") and str(d.get("source_name")) != str(d.get("name")):
                st.caption(f"Source file: {d.get('source_name')}")

            st.markdown(f"**{d['name']}** · {df.shape[0]} rows × {df.shape[1]} cols")
            st.dataframe(df.head(8), use_container_width=True)

    _bottom_next_prev()


# -----------------------------
# Page 2: Features
# -----------------------------
elif page == 1:
    st.subheader("2) Features")

    if not datasets:
        st.warning("No datasets loaded. Go back to Data and upload CSV files.")
        all_cols = []

    # Role mapping
    st.markdown("### Map columns to roles")
    st.caption("Role mapping is global (applies to every uploaded dataset). Missing columns in a dataset are skipped.")

    role_map: Dict[str, List[str]] = st.session_state.role_map

    roles = ["Substrate 1", "Substrate 2", "Substrate 3", "Product", "Other"]
    for r in roles:
        default = role_map.get(r, [])
        label = f"{r} column(s)"
        if r == "Substrate 3":
            label = "Substrate 3 (optional) column(s)"
        role_map[r] = st.multiselect(
            label,
            options=all_cols,
            default=[c for c in default if c in all_cols],
            key=f"role_{r}",
        )

    st.session_state.role_map = role_map

    # Build chemical space from...
    st.markdown("### Build chemical space from")
    st.caption(
        "This controls *which mapped columns* are included when building the chemical space. "
        "**Whole reaction** includes *all* mapped role-columns (Substrates/Product/Other). "
        "**Select role-columns** lets you include only a subset (e.g., Product-only)."
    )
    build_mode = st.radio(
        "Choose how to include mapped columns",
        options=["Whole reaction", "Select role-columns"],
        index=0 if st.session_state.build_mode == "Whole reaction" else 1,
        horizontal=True,
    )
    st.session_state.build_mode = build_mode

    rolecols: List[str] = []
    for r, cols in role_map.items():
        for c in cols:
            rolecols.append(_rolecol_str(r, c))

    rolecols = sorted(rolecols)

    if build_mode == "Whole reaction":
        st.session_state.selected_rolecols = rolecols
        st.info(
            f"Whole reaction selected: {len(rolecols)} role-column(s) will be included (all mapped columns)."
        )
        st.code("\n".join(rolecols) if rolecols else "(no mapped columns yet)")
    else:
        sel = st.multiselect(
            "Select role-column(s) to include",
            options=rolecols,
            default=[s for s in st.session_state.selected_rolecols if s in rolecols],
        )
        st.session_state.selected_rolecols = sel
        st.info(
            "Only the selected role-columns will be included. Useful for focusing on a single component "
            "(e.g., Product-only space or Substrate-only space)."
        )

    # Featurization params
    st.markdown("### RDKit featurization")

    feature_modes = [
        "MorganFP",
        "RDKit descriptors",
        "RDKit + MorganFP",
        "RDKit + MorganFP + MACCS",
    ]
    prev_fmode = st.session_state.get("feature_mode", "MorganFP")
    try:
        fmode_index = feature_modes.index(prev_fmode)
    except Exception:
        fmode_index = 0
    feature_mode = st.selectbox(
        "Feature mode",
        feature_modes,
        index=fmode_index,
        help="Choose how molecules are featurized before embedding. "
             "MorganFP is fast; RDKit descriptors adds physicochemical descriptors; "
             "combined modes concatenate features.",
    )
    st.session_state.feature_mode = feature_mode

    if not _HAS_RDKIT:
        st.error("RDKit not found. Install via conda-forge: `conda install -c conda-forge rdkit`")

    colA, colB, colC = st.columns(3)
    with colA:
        morgan_bits = st.selectbox("Morgan bits", [512, 1024, 2048, 4096], index=2)
    with colB:
        morgan_radius = st.selectbox("Morgan radius", [1, 2, 3], index=1)
    with colC:
        include_maccs = st.checkbox("Include MACCS", value=False)

    st.session_state.morgan_bits = int(morgan_bits)
    st.session_state.morgan_radius = int(morgan_radius)
    st.session_state.include_maccs = bool(include_maccs)

    _bottom_next_prev()


# -----------------------------
# Page 3: Embedding
# -----------------------------
elif page == 2:
    st.subheader("3) Embedding")

    with st.expander("Settings", expanded=True):
        if st.button("Back to default settings", key="reset_embedding_defaults"):
            # Embedding defaults
            st.session_state.embed_method = "UMAP"
            st.session_state.seed = 42
            st.session_state.umap_n_neighbors = 30
            st.session_state.umap_min_dist = 0.1
            st.session_state.tsne_n_components = 3
            st.session_state.tsne_perplexity = 50
            st.session_state.tsne_early_exaggeration = 12.0
            st.session_state.tsne_lr = 10.0
            st.session_state.tsne_init = "pca"
            st.session_state.tsne_metric = "euclidean"
            st.session_state.pca_whiten = False
            st.session_state.pca_svd_solver = "auto"
            st.rerun()
        method_choices = ["UMAP", "t-SNE", "PCA"]
        prev_method = st.session_state.get("embed_method", "UMAP")
        try:
            method_index = method_choices.index(prev_method)
        except Exception:
            method_index = 0
        method = st.selectbox("Method", method_choices, index=method_index)

        prev_seed = int(st.session_state.get("seed", 42))
        seed = st.number_input("Random seed", min_value=0, max_value=10_000, value=prev_seed, step=1)
        st.session_state.embed_method = method
        st.session_state.seed = int(seed)


        if method == "UMAP":
            if not _HAS_UMAP:
                st.error("UMAP not installed. Install `umap-learn`.")
            u1, u2 = st.columns(2)
            with u1:
                umap_n_neighbors = st.slider("UMAP n_neighbors", 5, 200, int(st.session_state.get("umap_n_neighbors", 30)))
            with u2:
                umap_min_dist = st.slider("UMAP min_dist", 0.0, 1.0, float(st.session_state.get("umap_min_dist", 0.1)))
            st.session_state.umap_n_neighbors = int(umap_n_neighbors)
            st.session_state.umap_min_dist = float(umap_min_dist)

        if method == "t-SNE":
            # Minimal UI extension: expose key TSNE parameters via sliders/selectors
            c1, c2, c3 = st.columns(3)
            with c1:
                tsne_n_components = st.selectbox("t-SNE n_components", [2, 3], index=([2,3].index(int(st.session_state.get("tsne_n_components", 3))) if int(st.session_state.get("tsne_n_components", 3)) in [2,3] else 1))
            with c2:
                tsne_perplexity = st.slider("t-SNE perplexity", 5, 100, int(st.session_state.get("tsne_perplexity", 50)))
            with c3:
                tsne_early_exag = st.slider("t-SNE early_exaggeration", 1.0, 50.0, float(st.session_state.get("tsne_early_exaggeration", 12.0)))

            c4, c5, c6 = st.columns(3)
            with c4:
                tsne_lr = st.slider("t-SNE learning_rate", 1.0, 1000.0, float(st.session_state.get("tsne_lr", 10.0)))
            # with c5:
                # tsne_n_iter = st.slider("t-SNE n_iter", 250, 5000, 1000, step=50)
            with c6:
                tsne_init = st.selectbox("t-SNE init", ["pca", "random"], index=(["pca","random"].index(str(st.session_state.get("tsne_init","pca"))) if str(st.session_state.get("tsne_init","pca")) in ["pca","random"] else 0))

            tsne_metric = st.selectbox("t-SNE metric", ["euclidean", "cosine"], index=(["euclidean","cosine"].index(str(st.session_state.get("tsne_metric","euclidean"))) if str(st.session_state.get("tsne_metric","euclidean")) in ["euclidean","cosine"] else 0))

            st.session_state.tsne_n_components = int(tsne_n_components)
            st.session_state.tsne_perplexity = int(tsne_perplexity)
            st.session_state.tsne_early_exaggeration = float(tsne_early_exag)
            st.session_state.tsne_lr = float(tsne_lr)
            # st.session_state.tsne_n_iter = int(tsne_n_iter)
            st.session_state.tsne_init = str(tsne_init)
            st.session_state.tsne_metric = str(tsne_metric)


        if method == "PCA":
            p1, p2 = st.columns(2)
            with p1:
                pca_whiten = st.checkbox(
                    "PCA whiten",
                    value=bool(st.session_state.get("pca_whiten", False)),
                    help="If enabled, components are scaled to unit variance (sometimes helps when feature scales differ).",
                )
            with p2:
                pca_solvers = ["auto", "full", "arpack", "randomized"]
                prev_solver = str(st.session_state.get("pca_svd_solver", "auto"))
                try:
                    solver_index = pca_solvers.index(prev_solver)
                except Exception:
                    solver_index = 0
                pca_svd_solver = st.selectbox(
                    "PCA svd_solver",
                    pca_solvers,
                    index=solver_index,
                    help="Advanced PCA solver choice. 'auto' is usually fine; 'randomized' can be faster for high-dim.",
                )
            st.session_state.pca_whiten = bool(pca_whiten)
            st.session_state.pca_svd_solver = str(pca_svd_solver)

    _bottom_next_prev()



# -----------------------------
# Page 4: Plot & Export
# -----------------------------
elif page == 3:
    st.subheader("4) Plot & Export")

    datasets = st.session_state.datasets
    role_map = st.session_state.role_map
    selected_rolecols = st.session_state.selected_rolecols

    if not datasets:
        st.warning("No datasets loaded. Go back to Data.")
        _bottom_next_prev()
        st.stop()

    if not selected_rolecols:
        st.warning("No role-columns selected. Go back to Features and map/select columns.")
        _bottom_next_prev()
        st.stop()

    # Combined settings (plot + academic styling) BEFORE compute
    with st.expander("Figure settings (plot + academic style)", expanded=True):
        st.caption("All controls below affect *drawing* (unless otherwise noted). Configure them first, then click Build & plot.")

        if st.button("Back to default settings", key="reset_plot_defaults"):
            # Plot defaults
            st.session_state.plot_point_size = 7
            st.session_state.plot_opacity = 0.85
            st.session_state.fig_width = 6.5
            st.session_state.fig_height = 6.0
            st.session_state.font_scale = 1.1
            st.session_state.legend_pos = "Inside: lower left"
            st.session_state.legend_fontsize = 9
            st.session_state.panel_title = ""
            st.session_state.xlabel = "t-SNE-1"
            st.session_state.ylabel = "t-SNE-2"
            st.session_state.axis_linewidth = 1.5
            st.session_state.manual_axes = False
            st.session_state.pop("x_range", None)
            st.session_state.pop("y_range", None)
            # Reset per-dataset markers (matches UI order)
            _default_cycle = ["o", "*", "X", "^", "s", "D", "P", "v", "+", "x"]
            for _i, _d in enumerate(st.session_state.get("datasets", [])):
                st.session_state[f"marker_ds_{_i}"] = _default_cycle[_i % len(_default_cycle)]
            st.rerun()


        # --- Plot basics
        c0a, c0b, c0c, c0d = st.columns([1.2, 1.0, 1.0, 1.2])
        with c0a:
            color_by = st.selectbox("Color by", ["dataset", "role"], index=0, help="Kept for compatibility; academic plot uses the fixed 4-set styling.")
        with c0b:
            point_size = st.slider("Point size", 3, 18, int(st.session_state.get("plot_point_size", 7)), key="plot_point_size")
        with c0c:
            point_opacity = st.slider("Opacity", 0.1, 1.0, float(st.session_state.get("plot_opacity", 0.85)), key="plot_opacity")
        with c0d:
            show_invalid = st.checkbox("Keep invalid SMILES rows (dropped for embedding)", value=False)

        st.markdown("#### Academic figure style")
        cA, cB, cC, cD = st.columns(4)
        with cA:
            fig_width = st.slider("Figure width (inch)", 3.0, 10.0, float(st.session_state.get("fig_width", 6.5)), 0.1, key="fig_width")
        with cB:
            fig_height = st.slider("Figure height (inch)", 3.0, 10.0, float(st.session_state.get("fig_height", 6.0)), 0.1, key="fig_height")
        with cC:
            font_scale = st.slider("Font scale", 0.6, 1.6, float(st.session_state.get("font_scale", 1.1)), 0.05, key="font_scale")
        with cD:
            legend_pos = st.selectbox(
                "Legend placement",
                [

                    "Inside: lower left",
                    "Inside: upper left",
                    "Inside: lower right",
                    "Inside: upper right",
                    "Outside: right",
                    "Outside: bottom",
                                ],
                index=( [
                    "Inside: lower left",
                    "Inside: upper left",
                    "Inside: lower right",
                    "Inside: upper right",
                    "Outside: right",
                    "Outside: bottom",
                ].index(str(st.session_state.get("legend_pos", "Inside: lower left")))
                if str(st.session_state.get("legend_pos", "Inside: lower left")) in [
                    "Inside: lower left", "Inside: upper left", "Inside: lower right", "Inside: upper right", "Outside: right", "Outside: bottom"
                ] else 0),
                key="legend_pos",
            )
            legend_fontsize = st.slider("Legend font size", 6, 20, int(st.session_state.get("legend_fontsize", 9)), 1, key="legend_fontsize")

        cE, cF, cG, cH = st.columns(4)
        with cE:
            panel_title = st.text_input("Panel title (optional)", value=str(st.session_state.get("panel_title", "")), key="panel_title")
        with cF:
            xlabel = st.text_input("X label", value=str(st.session_state.get("xlabel", "t-SNE-1")), key="xlabel")
        with cG:
            ylabel = st.text_input("Y label", value=str(st.session_state.get("ylabel", "t-SNE-2")), key="ylabel")
        with cH:
            axis_linewidth = st.slider("Axis box linewidth", 0.5, 3.0, float(st.session_state.get("axis_linewidth", 1.5)), 0.1, key="axis_linewidth")

        st.markdown("#### Marker styles (by dataset)")
        marker_options = ["o", "s", "^", "v", "D", "P", "X", "*", "+", "x"]
        _default_cycle = ["o", "*", "X", "^", "s", "D", "P", "v", "+", "x"]
        dataset_names_for_markers = [str(d.get("name", f"dataset_{i+1}")) for i, d in enumerate(datasets)]
        marker_by_dataset = {}
        for _start in range(0, len(dataset_names_for_markers), 4):
            _cols = st.columns(min(4, len(dataset_names_for_markers) - _start))
            for _j, _col in enumerate(_cols):
                _idx = _start + _j
                _ds = dataset_names_for_markers[_idx]
                _def = _default_cycle[_idx % len(_default_cycle)]
                with _col:
                    _val = st.selectbox(
                        _ds,
                        marker_options,
                        index=(marker_options.index(str(st.session_state.get(f"marker_ds_{_idx}", _def)))
                               if str(st.session_state.get(f"marker_ds_{_idx}", _def)) in marker_options
                               else marker_options.index(_def)),
                        key=f"marker_ds_{_idx}",
                    )
                marker_by_dataset[_ds] = _val
        st.session_state.marker_by_dataset = marker_by_dataset

        st.markdown("#### Axis range (manual)")
        if st.session_state.get("cached_points") is None:
            st.caption("Axis sliders will appear after you compute an embedding once.")
        else:
            _pts = st.session_state.get("cached_points").copy()
            if "x" in _pts.columns and "y" in _pts.columns and len(_pts) > 0:
                x_min = float(np.nanmin(_pts["x"].to_numpy()))
                x_max = float(np.nanmax(_pts["x"].to_numpy()))
                y_min = float(np.nanmin(_pts["y"].to_numpy()))
                y_max = float(np.nanmax(_pts["y"].to_numpy()))
                # safe padding
                x_pad = (x_max - x_min) * 0.15 if (x_max > x_min) else 1.0
                y_pad = (y_max - y_min) * 0.15 if (y_max > y_min) else 1.0
                x_lo, x_hi = x_min - x_pad, x_max + x_pad
                y_lo, y_hi = y_min - y_pad, y_max + y_pad

                manual_axes = st.checkbox("Manually set axis limits", value=bool(st.session_state.get("manual_axes", False)))
                st.session_state.manual_axes = bool(manual_axes)
                if manual_axes:
                    xr = st.slider("X range", min_value=float(x_lo), max_value=float(x_hi), value=tuple(st.session_state.get("x_range", (x_min, x_max))))
                    yr = st.slider("Y range", min_value=float(y_lo), max_value=float(y_hi), value=tuple(st.session_state.get("y_range", (y_min, y_max))))
                    st.session_state.x_range = tuple(float(v) for v in xr)
                    st.session_state.y_range = tuple(float(v) for v in yr)

    # Compute (embedding + points) — only runs when clicking the button
    compute = st.button("Build & plot chemical space", type="primary")

    if "cached_points" not in st.session_state:
        st.session_state.cached_points = None

    if compute:
        seed = int(st.session_state.get("seed", 42))

        feature_mode = str(st.session_state.get("feature_mode", "MorganFP"))
        morgan_bits = int(st.session_state.get("morgan_bits", 2048))
        morgan_radius = int(st.session_state.get("morgan_radius", 2))
        include_maccs = bool(st.session_state.get("include_maccs", False))

        method = st.session_state.get("embed_method", "UMAP")
        umap_n_neighbors = int(st.session_state.get("umap_n_neighbors", 30))
        umap_min_dist = float(st.session_state.get("umap_min_dist", 0.1))

        tsne_n_components = int(st.session_state.get("tsne_n_components", 3))
        tsne_perplexity = int(st.session_state.get("tsne_perplexity", 50))
        tsne_early_exaggeration = float(st.session_state.get("tsne_early_exaggeration", 12.0))
        tsne_lr = float(st.session_state.get("tsne_lr", 10.0))
        tsne_init = str(st.session_state.get("tsne_init", "pca"))
        tsne_metric = str(st.session_state.get("tsne_metric", "euclidean"))

        pca_whiten = bool(st.session_state.get("pca_whiten", False))
        pca_svd_solver = str(st.session_state.get("pca_svd_solver", "auto"))

        with st.status("Computing…", expanded=True) as status:
            status.update(label="Collect molecules from selected role-columns")

            # --- draw_fig.ipynb parity mode (minimal change; only activates under exact conditions) ---
            build_mode = str(st.session_state.get("build_mode", "Whole reaction"))
            parity_mode = (
                build_mode == "Whole reaction"
                and method == "t-SNE"
                and feature_mode == "RDKit + MorganFP + MACCS"
                and bool(include_maccs)
                and int(morgan_bits) == 2048
                and int(morgan_radius) == 2
            )

            if parity_mode:
                status.update(label="Notebook-parity: build reaction-level table (Substrate 1 + Substrate 2 + Product)")
                df_react, picked_cols = build_reaction_points_parity(datasets, role_map)
                if df_react.empty or len(picked_cols) != 3:
                    st.error(
                        "Notebook-parity mode requires role mapping for Substrate 1, Substrate 2, and Product (one column each), "
                        "and those columns must exist in the uploaded CSV(s)."
                    )
                    status.update(label="Failed", state="error")
                    _bottom_next_prev()
                    st.stop()

                status.update(label="Notebook-parity: featurizing reactions (RDKit Properties + Morgan2048 + MACCS)")
                X, keep_mask = featurize_reaction_parity(df_react, morgan_bits=2048, morgan_radius=2)
                dfk = df_react.loc[keep_mask].reset_index(drop=True)

                if len(dfk) < 5:
                    st.error("Too few valid reactions to embed. (Need at least ~5)\nNote: parity mode drops any row where ANY of the 3 SMILES is invalid.")
                    status.update(label="Failed", state="error")
                    _bottom_next_prev()
                    st.stop()

                status.update(label=f"Embedding to 2D via {method} (cached)")
                emb = embed_2d(
                    X,
                    method=method,
                    seed=seed,
                    umap_n_neighbors=umap_n_neighbors,
                    umap_min_dist=umap_min_dist,
                    tsne_n_components=tsne_n_components,
                    tsne_perplexity=tsne_perplexity,
                    tsne_early_exaggeration=tsne_early_exaggeration,
                    tsne_learning_rate=tsne_lr,
                    tsne_init=tsne_init,
                    tsne_metric=tsne_metric,
                    pca_whiten=pca_whiten,
                    pca_svd_solver=pca_svd_solver,
                )
                if emb.shape[1] > 2:
                    emb = emb[:, :2]

                points = pd.DataFrame({
                    "dataset": dfk["dataset"].astype(str).values,
                    "dataset_source": dfk.get("dataset_source", dfk["dataset"]).astype(str).values,
                    "role": "reaction",
                    "column": " | ".join(picked_cols),
                    "smiles": (
                        dfk["smiles_0"].astype(str)
                        + " | " + dfk["smiles_1"].astype(str)
                        + " | " + dfk["smiles_2"].astype(str)
                    ).values,
                    "row_index": dfk["row_index"].astype(int).values,
                })
                points["valid_smiles"] = True
                points["x"] = emb[:, 0]
                points["y"] = emb[:, 1]
                st.session_state._parity_draw_fig_active = True
                status.update(label="Ready", state="complete")

            else:
                st.session_state._parity_draw_fig_active = False
                points = build_points_from_roles(
                    datasets,
                    role_map,
                    selected_rolecols,
                    sample_per_dataset=0,
                    seed=seed,
                )
                if points.empty:
                    st.error("No molecules found. Check your selected columns contain SMILES.")
                    status.update(label="Failed", state="error")
                    _bottom_next_prev()
                    st.stop()

                status.update(label=f"Featurizing {len(points)} SMILES (cached)")
                X, valid = featurize_smiles(
                    points["smiles"].tolist(),
                    feature_mode=feature_mode,
                    morgan_bits=morgan_bits,
                    morgan_radius=morgan_radius,
                    include_maccs=include_maccs,
                )
                points["valid_smiles"] = valid

                if not show_invalid:
                    points = points.loc[points["valid_smiles"]].reset_index(drop=True)
                    X = X[valid]

                if len(points) < 5:
                    st.error("Too few valid molecules to embed. (Need at least ~5)")
                    status.update(label="Failed", state="error")
                    _bottom_next_prev()
                    st.stop()

                status.update(label=f"Embedding to 2D via {method} (cached)")
                emb = embed_2d(
                    X,
                    method=method,
                    seed=seed,
                    umap_n_neighbors=umap_n_neighbors,
                    umap_min_dist=umap_min_dist,
                    tsne_n_components=tsne_n_components,
                    tsne_perplexity=tsne_perplexity,
                    tsne_early_exaggeration=tsne_early_exaggeration,
                    tsne_learning_rate=tsne_lr,
                    tsne_init=tsne_init,
                    tsne_metric=tsne_metric,
                    pca_whiten=pca_whiten,
                    pca_svd_solver=pca_svd_solver,
                )

                if emb.shape[1] > 2:
                    emb = emb[:, :2]

                points["x"] = emb[:, 0]
                points["y"] = emb[:, 1]
                status.update(label="Ready", state="complete")

            st.session_state.cached_points = points

    # Draw (reactive): uses cached_points; changing figure controls will update without recomputing embedding
    points_cached = st.session_state.get("cached_points")
    if points_cached is None:
        st.info("Click **Build & plot chemical space** to compute an embedding. After that, figure controls will update live.")
        _bottom_next_prev()
        st.stop()

    points = points_cached.copy()

    # Group by uploaded dataset (dynamic): legend/markers adjust with number of CSVs
    points_plot = points.copy()
    ds_order = [str(d.get("name", f"dataset_{i+1}")) for i, d in enumerate(datasets)]
    ds_to_idx = {str(d.get("name", f"dataset_{i+1}")): i for i, d in enumerate(datasets)}
    ds_present = [ds for ds in ds_order if ds in set(points_plot["dataset"].astype(str))]
    if not ds_present:
        ds_present = sorted(points_plot["dataset"].astype(str).unique().tolist())

    # Seaborn styling (as in notebook)
    sns.set_style("white")
    sns.set_context("paper", font_scale=float(font_scale))

    fig, ax = plt.subplots(figsize=(float(fig_width), float(fig_height)))

    # Assign colors per dataset (stable, academic palette)

    # If we are in draw_fig.ipynb parity mode AND dataset names match the notebook's 4 sets,
    # draw with the same markers/colors as the notebook (sizes/alpha per set).
    NOTEBOOK_STYLE_MAP = {
        "modeling": {"label": "modeling set", "marker": "o", "color": "#8da0cb", "size": 40, "alpha": 0.85},
        "component-disjoint": {"label": "component-disjoint validation set", "marker": "*", "color": "#e78ac3", "size": 120, "alpha": 0.90},
        "class-shift": {"label": "class-shift validation set", "marker": "X", "color": "#00c853", "size": 65, "alpha": 0.90},
        "prospective": {"label": "prospective synthesis set", "marker": "^", "color": "#cc00ff", "size": 85, "alpha": 0.95},
    }

    from matplotlib.colors import to_hex
    _parity_active = bool(st.session_state.get('_parity_draw_fig_active', False))
    _canonical = [k for k in ['modeling','component-disjoint','class-shift','prospective'] if k in set(ds_present)]

    if _parity_active and len(_canonical) == 4 and set(ds_present) == set(_canonical):
        ds_colors = {k: NOTEBOOK_STYLE_MAP[k]['color'] for k in ds_present}
    else:
        _palette = sns.color_palette("Set2", n_colors=max(1, len(ds_present)))
        ds_colors = {ds: to_hex(_palette[i % len(_palette)]) for i, ds in enumerate(ds_present)}

    # Markers from UI (per dataset)
    _default_cycle = ["o", "*", "X", "^", "s", "D", "P", "v", "+", "x"]
    s = float(point_size) ** 2
    a = float(point_opacity)

    for ds in ds_present:
        sub = points_plot.loc[points_plot["dataset"].astype(str) == ds]
        if sub.empty:
            continue
        idx = ds_to_idx.get(ds, 0)
        _parity_active = bool(st.session_state.get('_parity_draw_fig_active', False))
        if _parity_active and ds in NOTEBOOK_STYLE_MAP:
            m = NOTEBOOK_STYLE_MAP[ds]['marker']
            _s_local = NOTEBOOK_STYLE_MAP[ds]['size']
            _a_local = NOTEBOOK_STYLE_MAP[ds]['alpha']
            _label = NOTEBOOK_STYLE_MAP[ds]['label']
        else:
            m = str(st.session_state.get(f"marker_ds_{idx}", _default_cycle[idx % len(_default_cycle)]))
            _s_local = s
            _a_local = a
            _label = str(ds)
        ax.scatter(
            sub["x"].to_numpy(),
            sub["y"].to_numpy(),
            s=_s_local,
            c=ds_colors.get(ds, "#4C72B0"),
            marker=m,
            alpha=_a_local,
            edgecolors="none",
            label=_label,
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if str(panel_title).strip():
        ax.set_title(panel_title, loc="left")

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(float(axis_linewidth))

    ax.grid(False)

    # Manual axis range
    if bool(st.session_state.get("manual_axes", False)):
        xr = st.session_state.get("x_range", None)
        yr = st.session_state.get("y_range", None)
        if xr is not None and len(xr) == 2:
            ax.set_xlim(float(xr[0]), float(xr[1]))
        if yr is not None and len(yr) == 2:
            ax.set_ylim(float(yr[0]), float(yr[1]))

    if str(legend_pos).startswith("Inside:"):
        _loc = str(legend_pos).split("Inside:", 1)[1].strip()
        ax.legend(loc=_loc, frameon=False, fontsize=int(legend_fontsize))
    elif str(legend_pos) == "Outside: right":
        ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=int(legend_fontsize))
    else:  # Outside: bottom
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, frameon=False, fontsize=int(legend_fontsize))

    st.pyplot(fig, use_container_width=True)

    # Figure export (PNG + SVG) — uses CURRENT styling
    st.markdown("### Export figure")
    buf_png = io.BytesIO()
    fig.savefig(buf_png, format="png", dpi=300, bbox_inches="tight")
    buf_png.seek(0)
    st.download_button("Download figure (PNG, 300 dpi)", data=buf_png.getvalue(), file_name="chemspace_plot.png", mime="image/png")

    buf_svg = io.BytesIO()
    fig.savefig(buf_svg, format="svg", bbox_inches="tight")
    buf_svg.seek(0)
    st.download_button("Download figure (SVG)", data=buf_svg.getvalue(), file_name="chemspace_plot.svg", mime="image/svg+xml")

    plt.close(fig)

    # Export embedding table
    st.markdown("### Export")
    out_csv = points.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download embedding CSV",
        data=out_csv,
        file_name="chemspace_embedding.csv",
        mime="text/csv",
    )

    _bottom_next_prev()
