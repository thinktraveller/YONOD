# YONOD — 有机合成反应预测平台

> **Y**our **O**ne-stop **N**otebook **O**f **D**escriptors
> 以 SMILES 为统一输入，比较 4 类分子描述符 × 4 种机器学习算法在多类有机合成任务上的表现。
>
> **Track A**：酰胺缩合反应**产率**预测（47015 条）
> **Track B**：镍催化不对称交叉偶联**对映选择性 ΔΔG** 预测（6590 条）

[![Python](https://img.shields.io/badge/Python-3.9-blue.svg)](https://www.python.org)
[![CUDA](https://img.shields.io/badge/CUDA-12.1-green.svg)](https://developer.nvidia.com/cuda-12-1-0-download-archive)
[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](LICENSE)

---

## 项目简介

本项目以 **SMILES 字符串作为统一输入**，对不同有机合成场景系统比较多种描述符与 ML 模型的组合，所有任务均使用 5 折交叉验证。

### Track A — 酰胺缩合产率预测

47015 条酰胺缩合反应，比较 **16 种（4 × 4）** 描述符 × 模型组合：

|                | XGBoost | Random Forest | SVM (RBF) | AutoGluon |
|---|---|---|---|---|
| **Morgan ECFP4**   | ✓ | ✓ | ✓ | ✓ |
| **ATMOMACCS (MACCS)** | ✓ | ✓ | ✓ | ✓ |
| **FISD (GNN embedding)** | ✓ | ✓ | ✓ | ✓ |
| **MolMetaLM (Llama embedding)** | ✓ | ✓ | ✓ | ✓ |

反应级特征：6 个分子（2 底物 + 4 试剂）描述符拼接，总维度 = 6 × 单分子维度。

### Track B — 镍催化对映选择性预测

6590 条 Ni 催化不对称交叉偶联反应，预测 **ΔΔG（kcal/mol）**，比较 **8 种（2 × 4）** 组合：

|                | XGBoost | Random Forest | SVM (RBF) | AutoGluon |
|---|---|---|---|---|
| **Morgan ECFP4**   | ✓ | ✓ | ✓ | ✓ |
| **ATMOMACCS (MACCS)** | ✓ | ✓ | ✓ | ✓ |

反应级特征：配体 SMILES + 产物底物 SMILES 描述符拼接，再附加温度标量（共 2×单分子维度 + 1 维）。

详细设计文档见 [YONOD项目构建计划书.md](YONOD项目构建计划书.md)；
迁移到云端/Linux/GitHub 的指南见 [MIGRATION.md](MIGRATION.md)。

---

## 主要结果

### Track A — 酰胺缩合产率（R²，5 折 CV）

| Descriptor | XGB | RF | SVM | AutoGluon |
|---|---|---|---|---|
| Morgan ECFP4 | 0.732 | — | 0.703 | **0.874** |
| ATMOMACCS    | 0.740 | — | 0.685 | **0.866** |
| FISD         | 0.771 | — | 0.635 | **0.861** |
| MolMetaLM    | 0.696 | — | 0.581 | 0.700 |

**关键发现**：AutoGluon 在 3/4 描述符上最优；FISD 在 XGB 列最优（0.771），跨域迁移有效；MolMetaLM 未经 fine-tune 效果偏弱。

### Track B — 镍催化对映选择性（R²，5 折 CV）

| Descriptor | XGB | RF | SVM | AutoGluon |
|---|---|---|---|---|
| Morgan ECFP4 | 0.827 | **0.835** | 0.787 | — |
| ATMOMACCS    | 0.799 | 0.816 | 0.779 | — |

附加指标（ee MAE）：Morgan × RF = **6.94%**，Morgan × XGB = 7.29%。AutoGluon 因特征矩阵含 NaN（AutoGluon 内部模型限制）未能完成，待修复。

完整指标 + 散点图见运行后生成的 `results/report.html` 和 `results/ecc_results/metrics_summary.csv`。

---

## 快速开始

### 环境准备

```bash
# 1) 创建 conda 环境
conda create -n yonod-yield python=3.9 -y
conda activate yonod-yield

# 2) 安装 PyTorch (CUDA 12.1)
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 \
    --index-url https://download.pytorch.org/whl/cu121

# 3) 安装其余依赖
pip install -r requirements.txt
pip install "autogluon.tabular[lightgbm,catboost]==1.1.1"

# 4) 验证 GPU 可用
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

Windows 用户离线安装 PyTorch（无网环境）参见 [YONOD项目构建计划书.md §3.3](YONOD项目构建计划书.md)。

### 下载外部资产

仓库**不包含**以下文件（体积大且非源码），需要单独获取：

| 资产 | 是否随仓库提供 | 默认放置路径 | 获取方式 |
|---|---|---|---|
| 数据集 `酰胺缩合数据集.csv` | ✅ 仓库已含（公开数据） | `数据集/` | 直接 `git clone` 即获得，详见 [数据集来源](#数据集来源) |
| ECC 数据集 `镍催化偶联数据集/Raw_Dataset.csv` | ✅ 仓库已含 | `数据集/镍催化偶联数据集/` | 直接 `git clone` 即获得，详见 [ECC 数据集](#ecc-数据集) |
| FISD 模型权重（3 个 .pth） | ✅ 仓库已含 | `WEIGHTS/FISD/` | 直接 `git clone` 即获得，详见 [FISD 权重](#fisd-权重) |
| MolMetaLM 权重 (~500 MB) | ❌ 需单独下载 | `WEIGHTS/MolMetaLM-base/` | 详见 [MolMetaLM 权重](#molmetalm-权重) |

### 烟测

```bash
# Track A（酰胺缩合产率）
python test_run_yield.py

# Track B（镍催化对映选择性）
python run_ecc_prediction.py --desc morgan --model xgb --nrows 500
```

Track A 烟测预期：30 秒内完成，输出 `[PASS]`，`results/` 下产生 `metrics_summary.csv`（1 行）和 `scatter_morgan_xgb.png`。

### Track A 完整 4×4 grid

```bash
python run_yield_prediction.py \
    --svm-subsample 8000 \
    --rf-verbose 1 \
    --heartbeat 60
```

GPU 单卡 + 16 核 CPU + 32 GB RAM 上预计 30~60 分钟。完成后查看 `results/report.html`。

### Track B 完整 2×4 grid

```bash
python run_ecc_prediction.py --desc morgan atmomaccs --model xgb rf svm autogluon
```

预计 30~60 分钟（主要耗时在 RF）。结果见 `results/ecc_results/metrics_summary.csv`。

**本地 Windows 慢机用户**：建议分两阶段跑，详见 [MIGRATION.md §一](MIGRATION.md)。

---

## 外部资产说明

### 数据集来源

`数据集/酰胺缩合反应数据集.csv` 衍生自公开发表的反应数据库（USPTO / Reaxys 风格的酰胺缩合反应集合），
所有反应均已完成 id → SMILES 映射；产率取实验报告值并归一化到 [0, 1]。
本仓库随源码一并提供。

如您将该数据集用于发表，请同时引用原始数据来源（具体文献信息见 `数据集/试剂编号-SMILES对应表.md` 与项目计划书）。

数据集 schema（CSV 列说明）：

| 列名 | 类型 | 说明 |
|---|---|---|
| `row_id` | int | 行号 |
| `sub_1_smiles` | str | 底物 1 的 SMILES |
| `sub_2_smiles` | str | 底物 2 的 SMILES |
| `product_smiles` | str | 产物 SMILES（**不参与训练**，仅供参考） |
| `activation_id` | str | 激活剂 SMILES（已映射；`(无)` 表示无） |
| `additive_id` | str | 添加剂 SMILES |
| `base_id` | str | 碱 SMILES |
| `solvent_id` | str | 溶剂 SMILES |
| `yield` | float [0, 1] | 实验产率（回归目标） |

注：`*_id` 列名虽叫 `id`，但内容已是 SMILES 字符串；含 `,` 的为盐型/复合物，代码会自动转 `.`。

### ECC 数据集

`数据集/镍催化偶联数据集/Raw_Dataset.csv` 来自公开论文：

> *AI-Driven Development of Nickel-Catalyzed Enantioselective Cross-Coupling Reactions*

数据集 schema（核心列）：

| 列名 | 类型 | 说明 |
|---|---|---|
| `Ligand_SMILES` | str | 催化剂配体 SMILES |
| `Product_SMILES` | str | 产物底物 SMILES |
| `Temperature` | float (°C) | 反应温度 |
| `△△G (Kcal/mol)` | float | **回归目标**，两对映体过渡态自由能差 |
| `ee (%)` | float | 对映体过量值（与 ΔΔG 等价，`ee = tanh(ΔΔG/2RT)×100`） |

使用该数据集时请引用原论文（见 `数据集/镍催化偶联数据集/文献/`）。

---

### MolMetaLM 权重

来自 HuggingFace：[wudejian789/MolMetaLM-base](https://huggingface.co/wudejian789/MolMetaLM-base)

```bash
# 方式 A：用 huggingface_hub CLI
pip install -U "huggingface_hub>=0.34"
# 国内用户加镜像：export HF_ENDPOINT=https://hf-mirror.com
hf download wudejian789/MolMetaLM-base --local-dir WEIGHTS/MolMetaLM-base

# 方式 B：Python 脚本
python -c "
from huggingface_hub import snapshot_download
snapshot_download('wudejian789/MolMetaLM-base', local_dir='WEIGHTS/MolMetaLM-base')
"
```

文件总大小约 500 MB。下载完成后 `WEIGHTS/MolMetaLM-base/` 应包含：
`config.json`、`model.safetensors`、`tokenizer.json`、`tokenizer_config.json`、`vocab.txt` 等。

### FISD 权重

FISD 描述符依赖 3 个在 QM9 上预训练的 GCN 权重文件（合计约 19 MB），**已随仓库一同提供**：

```
WEIGHTS/FISD/
├── qm_9_mse_model.pth     (~7.6 MB) — MSE 监督的 GCN
├── qm_9_cos_model.pth     (~7.6 MB) — Cosine 监督的 GCN
└── qm_9_2in1_model.pth    (~3.5 MB) — 拼接两路的 TwoInOne MLP
```

`git clone` 后即可直接使用，无需额外下载。`fisd.py` 默认从此目录加载，亦支持 `model_dir=` 参数覆盖。

> 注：上游 FISD 项目在 `化学描述符相关项目/FISD/model/` 中还有 `VS2Net_*.pth` / `dude_82_*.pth` 等其他模型的权重，本项目不使用，已在 `.gitignore` 中排除以避免和上游 git 历史混淆。

### ATMOMACCS 源码

本项目的 ATMOMACCS 描述符**仅调用 RDKit 的标准 MACCS keys**（167 位去除占位 bit 0 后为 166 维），未引入上游 ATMOMACCS 项目源码，因此**无需下载**任何外部资产。

如您在论文中引用本项目的 ATMOMACCS 实验结果，请同时致谢上游：

- **代码与权重**：[Zenodo / 18669279](https://zenodo.org/records/18669279)
- **文献**：[J. Chem. Phys. — DOI:10.1063/5.0308548](https://doi.org/10.1063/5.0308548)

---

## 算法实现对照表

> 为保持仓库自包含（`git clone` 后即可直接 `python run_yield_prediction.py`），
> FISD 等需要自行实现的算法已**直接 inline 到 `yonod_yield/` 中**，未把上游源码 vendor 进本仓库。
> 下表给出 inline 代码 ↔ 上游来源 的逐行映射，便于审计或基于新数据重训练。

| 算法 | inline 实现位置 | 上游来源 / 外部库 |
|---|---|---|
| Morgan ECFP4 指纹 | [morgan.py:32-50](yonod_yield/descriptors/morgan.py#L32-L50) | RDKit `rdkit.Chem.AllChem.GetMorganFingerprintAsBitVect`（仅作 API 调用） |
| MACCS keys（166 维） | [atmomaccs.py:30-44](yonod_yield/descriptors/atmomaccs.py#L30-L44) | RDKit `rdkit.Chem.rdMolDescriptors.GetMACCSKeysFingerprint`；与上游 ATMOMACCS 的 `generate_MACCS.py` 完全等价（去掉占位 bit 0）。引用：[Zenodo 18669279](https://zenodo.org/records/18669279) / [DOI:10.1063/5.0308548](https://doi.org/10.1063/5.0308548) |
| FISD 双 GCN 嵌入（架构） | [fisd.py:51-95](yonod_yield/descriptors/fisd.py#L51-L95) (`_GNN`、`_TwoInOne` 类) | 与上游 FISD `code/MLMS_mse.ipynb` / `MLMS_cos.ipynb` / `MLMS_2IN1.ipynb` 完全一致 |
| FISD 原子特征（45 维） | [fisd.py:99-132](yonod_yield/descriptors/fisd.py#L99-L132) (`_atom_features`) | 与上游 `test_MLMS/reproduce_mlms.py::get_atom_features` 一致 |
| FISD 图构建 | [fisd.py:134-150](yonod_yield/descriptors/fisd.py#L134-L150) (`_smiles_to_graph`) | 同上游；含单原子分子自环兜底 |
| FISD 前向 + 池化 | [fisd.py:153-216](yonod_yield/descriptors/fisd.py#L153-L216) (`FISDDescriptor`) | 加载 3 个 `qm_9_*.pth`（`WEIGHTS/FISD/`），双 GCN → concat → TwoInOne → 50 维 |
| MolMetaLM 嵌入 | [molmetalm.py:80-149](yonod_yield/descriptors/molmetalm.py#L80-L149) | HuggingFace [`wudejian789/MolMetaLM-base`](https://huggingface.co/wudejian789/MolMetaLM-base)；用 `AutoModel`（带 CausalLM 兜底）+ attention-masked mean-pool 取 768 维 |
| 反应级 6 分子特征拼接 | [reaction_featurizer.py:33-57](yonod_yield/features/reaction_featurizer.py#L33-L57) | 本项目原创设计（v0.3 §2.3），含 `(无)` 零向量 + `,` → `.` 预处理 |
| 试剂 SMILES 缓存 | [reagent_cache.py](yonod_yield/features/reagent_cache.py) | 本项目原创；首次运行落 `cache/reagent_feats_<desc>.pkl` |
| XGBoost 回归适配器 | [xgb_model.py](yonod_yield/models/xgb_model.py) | `xgboost.XGBRegressor`（`tree_method='hist', device='cuda'`） |
| RandomForest 适配器 | [rf_model.py](yonod_yield/models/rf_model.py) | `sklearn.ensemble.RandomForestRegressor`（300 树，n_jobs=-1） |
| SVR + 自动 PCA + 子采样 | [svm_model.py](yonod_yield/models/svm_model.py) | `sklearn.svm.SVR(kernel='rbf')` + `PCA(n=256)`（输入 >512 维触发）+ 每折随机抽 8000 行训练 |
| AutoGluon 适配器 | [autogluon_model.py](yonod_yield/models/autogluon_model.py) | `autogluon.tabular.TabularPredictor`（medium_quality preset，80/20 holdout） |

**为什么不把上游源码 vendor 进本仓库？**
- 上游各项目（ATMOMACCS / FISD / MolMetaLM）有各自的 LICENSE，逐项重发分布需要审查每份协议；
- inline 实现反而让代码自包含，`git clone` 后即跑，无需 submodule 操作或单独 clone；
- 上游 URL 在本文档完整列出，审计者可逐行对照原仓库的对应文件。

---

## 仓库结构

```
YONOD/
├── README.md                       # 本文件
├── MIGRATION.md                    # 云端/Linux/GitHub 迁移指南
├── CHANGELOG.md                    # 版本变更记录
├── YONOD项目构建计划书.md          # 完整设计文档（含决策记录、Q&A）
├── requirements.txt                # Python 依赖（不含 torch / autogluon，见快速开始）
├── .gitignore
├── LICENSE
│
├── yonod_yield/                    # 主代码包
│   ├── descriptors/                # 描述符实现（base / morgan / atmomaccs / fisd / molmetalm）
│   ├── features/
│   │   ├── dataset.py              # Track A 数据加载（6 列 SMILES + yield）
│   │   ├── ecc_dataset.py          # Track B 数据加载（配体 + 产物 SMILES + ΔΔG）
│   │   ├── reaction_featurizer.py  # 反应级特征拼接器
│   │   └── reagent_cache.py        # 试剂描述符缓存（加速重复计算）
│   ├── metrics/
│   │   └── ee_metrics.py           # ΔΔG → ee% 换算（Track B 专用）
│   ├── models/                     # 4 个 ML 模型适配器（xgb / rf / svm / autogluon）
│   ├── evaluate.py                 # 注册表 + evaluate_one() 入口
│   └── plot.py                     # 散点图生成
│
├── run_yield_prediction.py         # Track A 入口：酰胺缩合产率 4×4 grid
├── run_ecc_prediction.py           # Track B 入口：ECC 对映选择性 2×4 grid
├── generate_report.py              # 生成自包含 HTML 报告（Track A）
├── test_run_yield.py               # Track A 烟测脚本
├── test_ecc_load.py                # Track B 数据加载烟测
├── verify_morgan_rf.py             # RF 性能诊断脚本
│
├── 数据集/
│   ├── 酰胺缩合数据集.csv          # Track A 数据（公开来源，仓库已含）
│   └── 镍催化偶联数据集/           # Track B 数据
│       ├── Raw_Dataset.csv         # 主文件（含 SMILES + ΔΔG，6590 条）
│       ├── 预计算描述符/            # DFT/RDKit 预计算特征（供基线对比）
│       ├── 文献/                   # 原论文 PDF + Markdown
│       └── README.md               # 列名说明
│
├── WEIGHTS/
│   ├── FISD/                       # FISD 3 个预训练 .pth（仓库已含）
│   └── MolMetaLM-base/             # MolMetaLM 权重（需自行下载）
├── 化学描述符相关项目/             # 第三方项目源码（不入库；见 .gitignore）
├── cache/                          # 运行时缓存（自动生成）
└── results/
    ├── metrics_summary.csv         # Track A 汇总指标
    ├── report.html                 # Track A HTML 报告
    ├── scatter_*.png               # Track A 散点图
    └── ecc_results/                # Track B 所有输出
        ├── metrics_summary.csv
        └── scatter_*.png
```

---

## 命令行参数速查

### Track A — `run_yield_prediction.py`

| 参数 | 默认 | 用途 |
|---|---|---|
| `--csv PATH` | `数据集/酰胺缩合数据集.csv` | 数据集路径 |
| `--desc {morgan,atmomaccs,fisd,molmetalm}` | 全部 4 个 | 选择描述符（可多选） |
| `--model {xgb,rf,svm,autogluon}` | 全部 4 个 | 选择模型（可多选） |
| `--nrows N` | 全量 | 限制读取行数（调试时用 500） |
| `--cv K` | 5 | K 折交叉验证 |
| `--svm-subsample N` | 8000 | SVM 每折训练子采样（RBF SVR O(n²) 复杂度限制） |
| `--rf-verbose 0/1/2` | 0 | RF sklearn 详细程度 |
| `--rf-n-estimators N` | 300 | RF 树数 |
| `--heartbeat SEC` | 30 | 长任务心跳间隔（0=关） |
| `--log-file PATH` | `results/run_<ts>.log` | 日志文件 |
| `--append` | False | 追加合并入已有 metrics_summary.csv |
| `--skip-plots` | False | 跳过散点图生成 |
| `--skip-report` | False | 跳过最后的 HTML 报告生成 |

### Track B — `run_ecc_prediction.py`

| 参数 | 默认 | 用途 |
|---|---|---|
| `--xlsx PATH` | `数据集/镍催化偶联数据集/Raw_Dataset.csv` | ECC 数据集路径 |
| `--desc {morgan,atmomaccs}` | 全部 2 个 | 选择描述符（可多选） |
| `--model {xgb,rf,svm,autogluon}` | 全部 4 个 | 选择模型（可多选） |
| `--nrows N` | 全量 | 限制读取行数（调试时用 500） |
| `--cv K` | 5 | K 折交叉验证 |
| `--svm-subsample N` | 4000 | SVM 每折训练子采样 |
| `--heartbeat SEC` | 30 | 长任务心跳间隔 |
| `--results-dir PATH` | `results/ecc_results` | 输出目录 |
| `--append` | False | 追加合并入已有 metrics_summary.csv |
| `--skip-plots` | False | 跳过散点图生成 |

---

## 常见提示与警告

运行时出现以下提示均属正常，**不影响结果正确性**，可直接忽略。

### AutoGluon 相关

| 提示信息 | 原因 | 是否需要处理 |
|---|---|---|
| `[warning] AutoGluon + numeric cols: 使用全局 StandardScaler（非折内归一化）` | AutoGluon 内部使用 holdout 划分，无法嵌入 KFold 循环，数值辅助列（如温度）只能全局 z-score | 已知局限，不影响指标输出；若需严格无泄露评估请不使用 AutoGluon |
| `Warning: path already exists! This predictor may overwrite an existing predictor!` | 同一次 grid 运行中多个 (描述符 × 模型) 组合复用了同一 temp 目录，后续组合覆盖前面的模型文件 | 无需处理；本项目只取指标，不持久化 AutoGluon 模型 |
| `pkg_resources is deprecated as an API` | AutoGluon 内部某处仍依赖已废弃的 `pkg_resources`，属 AutoGluon 自身代码问题 | 无需处理；升级 AutoGluon 版本可消除 |
| `No valid features to train KNeighborsUnif / KNeighborsDist... Skipping` | Morgan / MACCS 指纹是高维稀疏向量，KNN 距离度量在高维空间失效，AutoGluon 自动跳过 KNN 子模型 | 无需处理；AutoGluon 会用其他子模型（GBM / XGB / RF 等）继续训练 |
| `NeuralNetFastAI failed (ImportError)... Skipping` | 当前环境未安装 `fastai`，AutoGluon 尝试加载失败后跳过该子模型 | 无需处理（见下方说明） |

### 关于 fastai 安装失败

`fastai 1.1.1` 依赖 `spacy`，而 `spacy` 的最新版要求 `thinc >= 8.3.12`（仅支持 Python ≥ 3.10），与本项目的 **Python 3.9** 环境冲突，无法安装。

**建议直接跳过 fastai**：FastAI 只是 AutoGluon 16 个内置子模型之一，其余主力模型（LightGBM、XGBoost、RandomForest、ExtraTrees、CatBoost 等）均正常运行。对分子描述符（高维稀疏指纹）任务，树模型通常优于 FastAI tabular，安装 fastai 不会带来明显的 R² 提升。

---

## 引用

如需引用本仓库代码，可使用如下 BibTeX（请按需补充作者与年份）：

```bibtex
@misc{yonod2026,
  title  = {YONOD: Descriptor-Model Comparison for Amide Condensation Yield Prediction},
  author = {thinktraveller},
  year   = {2026},
  howpublished = {\url{https://github.com/thinktraveller/YONOD}}
}
```

如使用了 ATMOMACCS 结果，请同时引用：

```bibtex
@article{atmomaccs,
  title   = {ATMOMACCS},
  journal = {J. Chem. Phys.},
  doi     = {10.1063/5.0308548},
  url     = {https://zenodo.org/records/18669279}
}
```

---

## License

本项目代码与文档采用 **[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/)** 协议：

- ✅ **允许**：自由分享、改编、二次创作；
- 📌 **要求**：署名（必须保留 `https://github.com/thinktraveller/YONOD` 作为来源）；
- ❌ **限制**：**不得用于商业目的**。

完整协议见 [LICENSE](LICENSE) 文件。

依赖与第三方资产遵循各自的协议：

| 资产 | 协议 |
|---|---|
| RDKit | BSD 3-Clause |
| scikit-learn / PyTorch / PyTorch Geometric | BSD-style |
| XGBoost / AutoGluon | Apache 2.0 |
| MolMetaLM 权重 | 见 [上游仓库](https://huggingface.co/wudejian789/MolMetaLM-base) |
| FISD 预训练权重 | 见上游 FISD 项目（仓库内仅含运行必需的 3 个 `.pth`） |
| ATMOMACCS 引用 | [J. Chem. Phys. DOI:10.1063/5.0308548](https://doi.org/10.1063/5.0308548) |

---

## 致谢

- 公开反应数据社区（USPTO / Reaxys 等）；
- MolMetaLM 作者 [@wudejian789](https://huggingface.co/wudejian789)；
- FISD 与 ATMOMACCS 上游项目作者。
