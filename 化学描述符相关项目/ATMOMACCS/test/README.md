# ATMOMACCS-KRR

ATMOMACCS-KRR is a research project exploring **molecular descriptor construction** and **Kernel Ridge Regression (KRR)** modeling for predicting thermodynamic and physicochemical properties of atmospheric organic molecules.

The project introduces the **ATMOMACCS** descriptors (versions 1–5), which extend the traditional MACCS fingerprint to better represent structural features relevant in atmospheric chemistry.

---

## Repository Structure

### Main Script
- [`main.py`](main.py)  
  End-to-end pipeline: generates descriptors, trains a KRR model, and produces predictions.

### Descriptor Construction
- **ATMOMACCS v1–v4:**  
  [`ATMOMACCS.py`](ATMOMACCS.py)  
  Implements versions v1-v4 of ATMOMACCS descriptors.

- **ATMOMACCS v5:**  
  [`ATMOMACCS_no_binary.py`](ATMOMACCS_no_binary.py)  
  Implements version v5 of ATMOMACCS.

### Descriptor Generation Scripts
- [`generate_ATMOMACCS.py`](generate_ATMOMACCS.py)  
  Generate ATMOMACCS descriptors for a dataset.

- [`generate_MACCS.py`](generate_MACCS.py)  
  Generate standard MACCS keys as baseline descriptors.

- [`generate_optimal_topfp.py`](generate_optimal_topfp.py)  
  Generate optimized topological fingerprints.

- [`generate_topological.py`](../../ATMOMACCS/src/model/generate_topological.py)  
  Generate topological fingerprints.

### Results Processing
- [`process_results.py`](process_results.py)  
  Aggregates results across multiple random seeds and produces **mean** and **standard deviation** CSV summaries for each descriptor/target combination.

---

## Getting Started

Example run with `main.py`:

```bash
python src/main.py -v 4 -d data/Wang -t log_p_sat.txt -ds Wang -s 2435
```

This will:

Generate descriptors (ATMOMACCS v4 in this case).

Train a KRR model on the chosen dataset and target.

Save predictions and performance metrics to the results folder.

## Requirements

This project was developed and tested with **Python 3.12.0**.

Dependencies:
```txt
matplotlib==3.8.0
pandas==2.1.1
rdkit==2023.9.1
scikit-learn==1.3.2
scipy==1.11.3
```
---
## Licensing and Data Sources

This repository redistributes and processes several benchmark datasets. Each dataset retains its **original license** as specified by the authors and publishers. Please ensure you comply with the relevant licenses when using these datasets.

---

### Ferraz-Caetano
- **License:** MIT License  
- **Original data file:** `VOC-Database.csv`  
- **Source:** [VOC-EnthVapML Database](https://github.com/jfcaetano/VOC-EnthVapML/tree/main/Database)  
- **Publication:**  
  José Ferraz-Caetano, Filipe Teixeira, M. Natália D.S. Cordeiro,  
  *Data-driven, explainable machine learning model for predicting volatile organic compounds’ standard vaporization enthalpy*,  
  *Chemosphere*, Volume 359, 142257 (2024).  
  [https://doi.org/10.1016/j.chemosphere.2024.142257](https://doi.org/10.1016/j.chemosphere.2024.142257)

---

### GeckoQ
- **License:** Creative Commons Attribution 4.0 International (CC BY 4.0)  
- **Original data file:** `Dataframe.csv`  
- **Source:** [Dataset DOI](https://doi.org/10.23729/022475cc-e527-41a9-bbc0-0113923cf04c)  
- **Publication:**  
  Vitus Besel, Milica Todorović, Theo Kurtén, Patrick Rinke & Hanna Vehkamäki,  
  *Atomic structures, conformers and thermodynamic properties of 32k atmospheric molecules*,  
  *Scientific Data* 10, 450 (2023).  
  [https://doi.org/10.1038/s41597-023-02366-x](https://doi.org/10.1038/s41597-023-02366-x)

---

### Wang
- **License:** Creative Commons Attribution 3.0 Unported (CC BY 3.0)  
- **Original data file:** Supplementary data (`acp-17-7529-2017-supplement.zip`)  
- **Source:** [ACP Supplement](https://acp.copernicus.org/articles/17/7529/2017/acp-17-7529-2017-supplement.zip)  
- **Publication:**  
  Chen Wang, Tiange Yuan, Stephen A. Wood, Kai-Uwe Goss, Jingyi Li, Qi Ying, and Frank Wania,  
  *Uncertain Henry's law constants compromise equilibrium partitioning calculations of atmospheric oxidation products*,  
  *Atmos. Chem. Phys.*, 17, 7529–7540 (2017).  
  [https://doi.org/10.5194/acp-17-7529-2017](https://doi.org/10.5194/acp-17-7529-2017)  

- **Modifications:** Documented in `data_preprocessing.ipynb` (formatting and restructuring of files).

---

### Li
- **License:** Creative Commons Attribution 4.0 International (CC BY 4.0)  
- **Original data file:** `Li et al. OA viscosity_Table S2.xls`  
- **Source:** [ACP Supplement](https://doi.org/10.5194/acp-20-8103-2020-supplement)  
- **Publication:**  
  Ying Li, Douglas A. Day, Harald Stark, Jose L. Jimenez, and Manabu Shiraiwa,  
  *Predictions of the glass transition temperature and viscosity of organic aerosols from volatility distributions*,  
  *Atmos. Chem. Phys.*, 20, 8103–8122 (2020).  
  [https://doi.org/10.5194/acp-20-8103-2020](https://doi.org/10.5194/acp-20-8103-2020)  

- **Modifications:** Documented in `CAS_to_smiles.ipynb` (formatting and restructuring of files).
