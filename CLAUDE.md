# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

YONOD (You Only Need Outstanding Descriptors) is an organic synthesis reaction yield prediction platform. It systematically compares **7 molecular descriptors × 4 ML models** for predicting reaction outcomes using SMILES as unified input.

## Common Commands

### Environment Setup
```bash
conda activate yonod
# Or create new: conda create -n yonod python=3.9 -y

# Install PyTorch with CUDA 12.1 first
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 \
    --index-url https://download.pytorch.org/whl/cu121

pip install -r requirements.txt
pip install "autogluon.tabular[lightgbm,catboost]==1.1.1"
```

### Running the Project

YONOD 采用双入口设计，支持三种使用场景：

#### 场景A: 完整向导流程（首次使用推荐）
```bash
python yonod.py
# → 8步向导收集配置 → 自动调用 main.py 进行建模
```

#### 场景B: 使用配置文件建模
```bash
python main.py --config project-folder/yonod_config.json
```

#### 场景C: 传统CLI模式（向后兼容）
```bash
python main.py \
    --csv dataset/amide-coupling.csv \
    --label-col yield \
    --smiles-cols sub_1_smiles sub_2_smiles activation_id additive_id base_id solvent_id \
    --descriptors morgan maccs \
    --models xgb rf

# Quick test (10 rows)
python main.py \
    --csv dataset/test-amide-coupling.csv \
    --label-col yield \
    --smiles-cols sub_1_smiles sub_2_smiles activation_id additive_id base_id solvent_id \
    --descriptors morgan --models xgb
```

### Running Tests
```bash
pytest tests/ -v                          # All tests
pytest tests/test_step1.py -v             # Single test file
pytest tests/test_step1.py::test_load_valid_csv -v  # Single test
```

## Architecture

### Entry Points (双入口设计)
- **`yonod.py`**: 数据准备入口（8步交互式向导）
  - 完成后生成 `yonod_config.json` 配置文件
  - 询问用户是否自动调用 `main.py` 进行建模
- **`main.py`**: 建模入口（纯CLI模式）
  - 支持 `--config` 读取配置文件
  - 支持传统CLI参数（向后兼容）

### Core Package: `yonod/`

```
yonod/
├── descriptors/          # 7 molecular descriptor implementations
│   ├── base.py           # BaseDescriptor interface + split_multi_smiles()
│   ├── morgan.py         # Morgan ECFP4 (1024d)
│   ├── atmomaccs.py      # MACCS keys (166d)
│   ├── fisd.py           # GCN embedding (50d, needs weights)
│   ├── molmetalm.py      # Llama embedding (768d, needs weights)
│   ├── maf.py            # Multi-molecule Addition Fingerprint (128d)
│   ├── rdkit2d.py        # RDKit 2D descriptors (~200d)
│   └── drfp_desc.py      # DRFP (2048d)
│
├── models/               # 4 ML model adapters
│   ├── xgb_model.py      # XGBoost (GPU hist, 300 trees)
│   ├── rf_model.py       # Random Forest (300 trees)
│   ├── svm_model.py      # SVR (RBF + PCA + subsampling)
│   └── autogluon_model.py # AutoGluon (medium_quality)
│
├── features/             # Feature engineering
│   ├── reagent_cache.py  # Reagent descriptor caching
│   └── reaction_featurizer.py  # Reaction-level feature concatenation
│
├── universal/            # Data processing utilities
│   ├── csv_loader.py     # Auto column role detection
│   ├── feature_builder.py # Feature matrix construction
│   └── report.py         # HTML/Markdown report generation
│
└── evaluate.py           # Model registry + cross-validation
```

### Data Flow
```
CSV (multiple SMILES cols + label)
  → Column role detection (csv_loader)
  → SMILES normalization ("," → ".", "(无)" → None)
  → Descriptor computation (with reagent caching)
  → Feature concatenation (6 molecules × desc_dim)
  → K-Fold CV training (StandardScaler per fold)
  → Metrics + scatter plots + HTML report
```

### Output Structure
```
project-folder/                        # 向导生成的项目文件夹
├── {project}_yonod_config.json        # 配置文件（供 main.py 读取）
├── {project}_column_mapping.csv       # 列映射表
├── {project}_normalized_dataset.csv   # 规范化数据集
├── {project}_invalid_report.md        # 非法输入报告（如有）
└── metrics_summary.csv                # 建模结果指标
    report.html                        # 可视化报告
    pictures/                          # 散点图
```

### 配置文件格式 (yonod_config.json)
```json
{
  "version": "1.0",
  "project_name": "amide_coupling_test",
  "dataset_path": "path/to/normalized_dataset.csv",
  "column_mapping_path": "path/to/column_mapping.csv",
  "descriptors": [
    {"descriptor": "morgan", "mode": "concat", "columns": ["reactant-1", "product-1"]}
  ],
  "models": ["XGBoost", "Random Forest"],
  "metadata": {"repo_url": "", "doi": "", "notes": ""},
  "report_formats": ["HTML", "Markdown"],
  "column_roles": {
    "label": "yield",
    "reactants": ["reactant-1"],
    "products": ["product-1"],
    "others": ["catalyst"],
    "conditions": ["temperature"]
  }
}
```

## Key Design Decisions

### Descriptor Interface
All descriptors implement `BaseDescriptor.featurize(smiles_list) -> (features, mask)`:
- `features`: shape (n, dim), failed rows filled with zeros
- `mask`: boolean array, True = successful parse

### Data Leakage Prevention
- StandardScaler fitted **only on training fold** within K-Fold loop
- Numeric columns scaled per-fold, not globally
- AutoGluon uses internal holdout (cannot do strict K-Fold)

### SVM Optimization
- Auto PCA reduction when dim > 512
- Training subsampling (default 8000) to handle O(n²d) complexity
- Test set remains complete (no leakage)

### SMILES Normalization
```python
# Multi-component handling
smiles.replace(",", ".")  # Convert to RDKit standard
"(无)" → None             # Maps to zero vector
```

## Model Hyperparameters

| Model | Key Parameters |
|-------|----------------|
| XGBoost | n_estimators=300, lr=0.05, max_depth=6, tree_method='hist' |
| RF | n_estimators=300, n_jobs=-1 |
| SVM | kernel='rbf', C=1.0, subsample=8000, auto-PCA if dim>512 |
| AutoGluon | time_limit=300s, presets='medium_quality' |

## Weight Files

Required for FISD and MolMetaLM descriptors:
- **MolMetaLM**: `WEIGHTS/MolMetaLM-base/` (download from HuggingFace)
- **FISD**: `WEIGHTS/FISD/` (download from GitHub)

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `FileNotFoundError: WEIGHTS/FISD/` | Download FISD weights from GitHub |
| `torch.cuda.OutOfMemoryError` | Reduce batch_size or use CPU |
| SVM running >10 min | Use `--svm-subsample 8000` |
| Chinese path encoding error | Copy dataset to English path |
