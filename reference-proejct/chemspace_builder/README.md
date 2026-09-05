# ChemSpace Builder — Streamlit app for chemical-space visualization

This project is a **Streamlit GUI** to build a chemical space from one or more CSV files (SMILES columns), compute features (e.g., Morgan fingerprints / RDKit features), embed molecules to **2D** (PCA / UMAP / t‑SNE), and export **publication-ready (academic style)** figures plus the embedding table.

## Quickstart (conda)

```bash
# Only if you have a conda environment named chemspace
conda remove -n chemspace --all -y

# Install chemspace env from .yml
conda env create -f environment.yml

# Run the project
conda activate chemspace
streamlit run app.py
```

---

## What the app does

The UI is a 4-step wizard with **Prev / Next** navigation:

1) **Data**
- Upload one or more CSV files.
- Optionally rename each dataset (the custom name is used in legend and exports).

2) **Features**
- Map CSV columns to reaction roles (e.g., Substrate 1/2/3 and Product; Substrate 3 is optional).
- Choose how to build chemical space:
  - **Whole reaction**: use all mapped roles together
  - **Select role-columns**: pick only specific role-columns to include
- Choose feature mode and its parameters (e.g., MorganFP bits/radius; optional MACCS).

3) **Embedding**
- Choose embedding method: **PCA**, **UMAP**, or **t‑SNE**.
- Tune embedding parameters from the UI.
- Use **Back to default** to reset embedding settings.

4) **Plot & Export**
- Plot settings and academic figure styling are configured together.
- Customize:
  - point size / opacity
  - legend location (inside corners or outside) and legend font size
  - marker style (shape) per dataset
  - manual axis ranges (x/y limits)
- Export:
  - `chemspace_plot.png` (300 dpi)
  - `chemspace_plot.svg`
  - `chemspace_embedding.csv`

---

## Input CSV requirements

- Any CSV schema is OK as long as the columns you map contain valid **SMILES**.
- Other columns (e.g., yield, predicted values, IDs) are carried through and can be shown in hover/export.

---

## Troubleshooting

- **RDKit import errors**: install RDKit from conda-forge into `chemspace`.
- **UMAP not available**: `conda install -c conda-forge umap-learn`.
- **Invalid SMILES rows**: enable dropping invalid SMILES in the UI (if you see that option) or clean the CSV.

## Learn-by-coding example (optional)

If you prefer a step-by-step, code-first walkthrough (instead of the no-code GUI), see **`draw_fig.ipynb`**. It demonstrates how to reproduce a **t-SNE chemical-space figure** from the exported embedding/table data using a notebook workflow, and is intended as an educational example for learning the plotting procedure end-to-end.
