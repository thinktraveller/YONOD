# YONOD — 有机合成反应预测平台

> **Y**ou **O**nly **N**eed **O**utstanding **D**escriptors
> 以 SMILES 为统一输入，比较 4 类分子描述符 × 4 种机器学习算法在多类有机合成任务上的表现。

[![Python](https://img.shields.io/badge/Python-3.9-blue.svg)](https://www.python.org) [![CUDA](https://img.shields.io/badge/CUDA-12.1-green.svg)](https://developer.nvidia.com/cuda-12-1-0-download-archive) [![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](LICENSE)

---

## 目录

- [项目简介](#项目简介)
- [新手快速上手](#新手快速上手)
- [主要结果](#主要结果)
- [仓库结构与文件说明](#仓库结构与文件说明)
- [外部资产说明](#外部资产说明)
- [算法实现对照表](#算法实现对照表)
- [命令行参数速查](#命令行参数速查)
- [常见提示与警告](#常见提示与警告)
- [引用](#引用)
- [License](#license)

---

## 项目简介

本项目以 **SMILES 字符串作为统一输入**，对不同有机合成场景系统比较多种描述符与 ML 模型的组合，所有任务均使用 5 折交叉验证。

|                | XGBoost | Random Forest | SVM (RBF) | AutoGluon |
|---|---|---|---|---|
| **Morgan ECFP4**   | ✓ | ✓ | ✓ | ✓ |
| **ATMOMACCS (MACCS)** | ✓ | ✓ | ✓ | ✓ |
| **FISD (GNN embedding)** | ✓ | ✓ | ✓ | ✓ |
| **MolMetaLM (Llama embedding)** | ✓ | ✓ | ✓ | ✓ |

详细设计文档见 [YONOD项目构建计划书.md](YONOD项目构建计划书.md)；
迁移到云端/Linux/GitHub 的指南见 [MIGRATION.md](MIGRATION.md)。

---

## 新手快速上手

> 如果你是第一次接触本项目，请按本节顺序执行，每一步都有说明。

### 第一步：了解前置知识

本项目需要以下基础：
- 会使用命令行（Windows PowerShell 或 conda 终端）
- 知道什么是 conda / pip（Python 包管理工具）
- 了解 SMILES 字符串（化学分子的文本表示）

不需要提前了解 XGBoost、AutoGluon 或 GNN，运行脚本会自动调用。

### 第二步：克隆仓库

```bash
git clone https://github.com/thinktraveller/YONOD.git
cd YONOD
```

克隆后你会得到代码和酰胺缩合样本数据集（`dataset/amide-coupling.csv`，MIT 协议）。**FISD 权重和 MolMetaLM 权重需要单独获取**，见 [外部资产说明](#外部资产说明)。

### 第三步：创建 conda 环境

```bash
# 创建专用 Python 3.9 环境（名称可自定义）
conda create -n yonod python=3.9 -y
conda activate yonod
```

> 为什么用 conda 而不是 pip venv？因为后续安装 PyTorch 时需要指定 CUDA 版本，conda 能更好地管理这类依赖。

### 第四步：安装 PyTorch（含 GPU 支持）

> 如果你的电脑**没有 NVIDIA GPU**，把 `cu121` 改为 `cpu` 即可，但 FISD 和 MolMetaLM 会慢很多。

```bash
# 有 GPU（CUDA 12.1）
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 \
    --index-url https://download.pytorch.org/whl/cu121

# 无 GPU / CPU 模式
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 \
    --index-url https://download.pytorch.org/whl/cpu
```

验证安装：

```bash
python -c "import torch; print('CUDA可用:', torch.cuda.is_available())"
```

### 第五步：安装其余依赖

```bash
pip install -r requirements.txt
pip install "autogluon.tabular[lightgbm,catboost]==1.1.1"
```

> AutoGluon 较大（约 1-2 GB），安装时间较长，请耐心等待。

### 第六步：准备数据集

**作为示例，酰胺缩合数据集已随仓库提供**（`dataset/amide-coupling.csv`，47015 条，**该数据集的使用遵循 MIT 协议**），克隆后可直接使用。

```
YONOD/
└── dataset/
    ├── amide-coupling.csv              ← 完整数据集（47015 条，✅ 仓库已含）
    ├── test-amide-coupling.csv         ← 10 行样本（✅ 仓库已含，快速调试用）
```

### 第七步：快速验证

运行以下命令启动交互向导（共 8 步，约 10 秒完成）：

```bash
python yonod.py
```

以下是使用 10 行样本数据集 `dataset/test-amide-coupling.csv` 的完整交互示例，`>` 后为你的输入：

```
============================================================
  YONOD - Your One-stop Notebook Of Descriptors
  通用交互向导  v3
============================================================

[1/8] 输入 CSV 文件路径（可直接拖拽文件到终端）
  路径: > dataset/test-amide-coupling.csv
  正在读取 CSV ...
  已读取 10 行 × 9 列

  列序号  列名
  ------  ----
  A( 1)   row_id
  B( 2)   sub_1_smiles
  C( 3)   sub_2_smiles
  D( 4)   product_smiles
  E( 5)   activation
  F( 6)   additive
  G( 7)   base
  H( 8)   solvent
  I( 9)   yield

[2/8] 标签列（预测目标，如 yield / ee / ddG）
      支持：列名  /  字母（如 B）  /  序号（如 2）
  标签列: > yield
  → 标签列确认：'yield'
  [验证] 标签列 'yield' 全量数值验证通过（共 10 行）。

[3/8] SMILES 列（分子结构列，至少指定一列）
      支持：列名 / 字母 / 序号，多列用空格分隔
  SMILES 列（空格分隔多列）: > B C E F G H
  → SMILES 列确认：['sub_1_smiles', 'sub_2_smiles', 'activation', 'additive', 'base', 'solvent']
  [验证] sub_1_smiles 全量验证通过。
  [验证] sub_2_smiles 全量验证通过。
  [验证] activation 全量验证通过。
  [验证] additive 全量验证通过。
  [验证] base 全量验证通过。
  [验证] solvent 全量验证通过。

[4/8] 数值辅助列（温度/压力等，可选）
      支持：列名 / 字母 / 序号，多列用空格分隔
  数值辅助列（空格分隔多列）[留空跳过]: >（直接回车）
  → 数值辅助列确认：（无）

[5/8] 任务名称（用于输出目录和报告标题）
      规则：仅允许英文字母、数字、下划线、连字符，如 amide_coupling
  任务名称: > smoke-test
  → 任务名称确认：'smoke-test'

[6/8] 输出目录（默认：<项目目录>/result/smoke-test）
  输出目录 [默认: <项目目录>/result/smoke-test]: >（直接回车）
  → 输出目录确认：'<项目目录>/result/smoke-test'

[7/8] 描述符选择（可选）
      可选值: morgan  maccs  fisd  molmetalm
  描述符（空格分隔，留空=全选）[留空跳过]: > morgan
  → 描述符确认：['morgan']

[8/8] 模型选择（可选）
      可选值: xgb  rf  svm  autogluon
  模型（空格分隔，留空=全选）[留空跳过]: > xgb
  → 模型确认：['xgb']

============================================================
  即将执行（等效命令）:
  python yonod.py --csv "dataset/test-amide-coupling.csv" --label-col yield
    --smiles-cols sub_1_smiles sub_2_smiles activation additive base solvent
    --task-name smoke-test --output-dir "<项目目录>/result/smoke-test"
    --descriptors morgan --models xgb
============================================================

按 Enter 确认执行，Ctrl+C 取消... >（直接回车）

[init] 任务：smoke-test
[load] n_rows=10  SMILES列=[...]  标签列='yield'
[desc] 计算描述符: morgan ...
[eval] 开始: morgan x xgb (1/1)
[done] morgan x xgb  R²=0.XXXX  RMSE=0.XXXX  t=X.Xs
[save] 指标已保存: <项目目录>/result/smoke-test/metrics_summary.csv
[report] HTML 报告已生成: <项目目录>/result/smoke-test/report.html
[done] 全部完成。
```

最后会输出`result/smoke-test/metrics_summary.csv` ，文件中包含 R² 指标。

### 第八步：完整运行

再次运行 `python yonod.py` 启动向导，使用完整数据集和全量 4×4 建模（约 60 ~ 120 分钟，建议 GPU）。

结果输出将默认到 `results/<task-name>/metrics_summary.csv`，HTML 报告见同目录。

> **Windows 用户**：如遇中文路径问题，可将数据集复制到纯英文路径再指定 `--csv`。

#### 控制台输出示例

```
[init] 任务：<task-name>
[load] n_rows=N  SMILES列=[M 列]  数值列=无  标签列='yield'
[grid] 描述符=['morgan', 'maccs', 'fisd', 'molmetalm']  模型=['xgb', 'rf', 'svm', 'autogluon']  cv=5

[desc] 计算描述符: morgan ...
[eval] 开始: morgan x xgb (1/16)
[done] morgan x xgb (1/16)  R²=X.XXXX  RMSE=X.XXXX  t=X.Xs
...
[done] molmetalm x autogluon (16/16)  R²=X.XXXX  RMSE=X.XXXX  t=X.Xs

[save] 指标已保存: result/<task-name>/metrics_summary.csv
[report] HTML 报告已生成: result/<task-name>/report.html
[done] 全部完成。
```

---

## 仓库结构与文件说明

> 本节说明 `git clone` 后你会得到的每个文件/目录的作用。
> 数据集（`数据集/`）和第三方源码（`化学描述符相关项目/`）不随仓库提供，见 [外部资产说明](#外部资产说明)。

```
YONOD/
│
├── yonod.py                        ★ 统一入口（向导 + CLI 合并）
│                                     - 不带参数：启动交互向导，引导你选择数据集
│                                     - 带 --csv 等参数：直接进入 CLI pipeline
│
├── yonod/                          ★ 核心代码包
│   │
│   ├── descriptors/                  分子描述符实现（4 种）
│   │   ├── base.py                     描述符基类（接口定义）
│   │   ├── morgan.py                   Morgan ECFP4 指纹（RDKit 调用）
│   │   ├── atmomaccs.py                MACCS keys 166 维（RDKit 调用）
│   │   ├── fisd.py                     FISD 双 GCN 嵌入（加载 WEIGHTS/FISD/*.pth）
│   │   └── molmetalm.py                MolMetaLM Llama 嵌入（加载 WEIGHTS/MolMetaLM-base/）
│   │
│   ├── features/                     反应级特征工程
│   │   ├── dataset.py                  酰胺缩合数据加载（6 列 SMILES + yield）
│   │   ├── ecc_dataset.py              ECC 数据加载（配体 + 产物 SMILES + ΔΔG）
│   │   ├── reaction_featurizer.py      反应级特征拼接器（6 分子描述符 concat）
│   │   └── reagent_cache.py            试剂描述符缓存（加速重复计算，落 cache/ 目录）
│   │
│   ├── universal/                    通用数据处理（不依赖具体反应类型）
│   │   ├── csv_loader.py               自动探测 SMILES 列、标签列（支持任意 CSV）
│   │   ├── feature_builder.py          通用特征矩阵构建（SMILES + 数值辅助列拼接）
│   │   └── report.py                   HTML 报告生成器
│   │
│   ├── metrics/
│   │   └── ee_metrics.py               ΔΔG → ee% 换算
│   │
│   ├── models/                       4 个 ML 模型适配器
│   │   ├── xgb_model.py                XGBoost（GPU hist 模式）
│   │   ├── rf_model.py                 Random Forest（300 树，多核）
│   │   ├── svm_model.py                SVR（RBF 核 + PCA 降维 + 子采样）
│   │   └── autogluon_model.py          AutoGluon（medium_quality preset）
│   │
│   ├── evaluate.py                   模型注册表 + evaluate_one() 入口
│   └── plot.py                       预测 vs 真值散点图生成
│
│
├── WEIGHTS/                          模型权重目录
│   ├── FISD/                         ✗ 需单独获取（约 19 MB，无明确许可证，见外部资产说明）
│   │   ├── qm_9_mse_model.pth          GCN（MSE 监督），~7.6 MB
│   │   ├── qm_9_cos_model.pth          GCN（Cosine 监督），~7.6 MB
│   │   └── qm_9_2in1_model.pth         TwoInOne MLP，~3.5 MB
│   └── MolMetaLM-base/               ✗ 需单独下载（约 500 MB，见外部资产说明）
│
├── results/                          运行输出（自动生成，不入库）
│   └── <task-name>/
│       ├── metrics_summary.csv         所有 (描述符, 模型) 组合的 R²/RMSE/MAE
│       ├── report.html                 可视化报告（含散点图）
│       └── run_<timestamp>.log         运行日志镜像
│
├── cache/                            运行时缓存（自动生成，不入库）
│                                     存放试剂描述符的 .pkl 缓存，加速重复运行
│
├── dataset/                          数据集目录
│   ├── amide-coupling.csv            ★ 酰胺缩合完整数据集（47015 条，MIT 协议，仓库已含）
│   └── test-amide-coupling.csv       ★ 酰胺缩合 10 行样本（MIT 协议，仓库已含，调试用）
│
├── 化学描述符相关项目/               ✗ 不随仓库提供（第三方源码，仅供本地参考）
│
├── README.md                         本文件
├── MIGRATION.md                      云端/Linux/GitHub 迁移指南
├── CHANGELOG.md                      版本变更记录
├── YONOD项目构建计划书.md            完整设计文档（含决策记录、Q&A）
├── requirements.txt                  Python 依赖（不含 torch / autogluon，见快速开始）
├── .gitignore
└── LICENSE
```

---

## 外部资产说明

仓库**不包含**以下文件（体积大或含第三方 LICENSE），需要单独获取：

| 资产 | 默认放置路径 | 获取方式 |
|---|---|---|
| FISD 模型权重（3 个 .pth，~19 MB） | `WEIGHTS/FISD/` | 见下方 [FISD 权重](#fisd-权重) |
| MolMetaLM 权重 (~500 MB) | `WEIGHTS/MolMetaLM-base/` | 见下方 [MolMetaLM 权重](#molmetalm-权重) |

### 酰胺缩合数据集

`dataset/amide-coupling.csv` 来自 [aichemeco/amide_coupling](https://github.com/aichemeco/amide_coupling/tree/main)（MIT 协议），47015 条酰胺缩合反应，产率归一化到 [0, 1]。`dataset/test-amide-coupling.csv` 为其中 10 行子集，用于快速调试。

如您将该数据集用于发表，请引用原始论文：

> Dai *et al.*, *Chem. Sci.*, 2025. DOI: [10.1039/D5SC03364K](https://doi.org/10.1039/D5SC03364K)

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

---

### MolMetaLM 权重

- **HuggingFace**：[wudejian789/MolMetaLM-base](https://huggingface.co/wudejian789/MolMetaLM-base)
- **GitHub**：[CSUBioGroup/MolMetaLM](https://github.com/CSUBioGroup/MolMetaLM)
- **论文**：DOI [10.48550/arXiv.2411.15500](https://doi.org/10.48550/arXiv.2411.15500)

```bash
# 从 huggingface_hub CLI 下载权重
pip install -U "huggingface_hub>=0.34"
hf download wudejian789/MolMetaLM-base --local-dir WEIGHTS/MolMetaLM-base
```

文件总大小约 500 MB。下载完成后 `WEIGHTS/MolMetaLM-base/` 应包含：
`config.json`、`model.safetensors`、`tokenizer.json`、`tokenizer_config.json`、`vocab.txt` 等。

### FISD 权重

- **论文**：DOI [10.1039/D5SC00451A](https://doi.org/10.1039/D5SC00451A)

FISD 描述符依赖 3 个在 QM9 上预训练的 GCN 权重文件（合计约 19 MB）。需从 [项目的github仓库](https://github.com/KeantChen/FISD/tree/main/model) 单独获取后放置到 `WEIGHTS/FISD/`：

```
WEIGHTS/FISD/           ← 需手动创建此目录并放入以下文件
├── qm_9_mse_model.pth     (~7.6 MB) — MSE 监督的 GCN
├── qm_9_cos_model.pth     (~7.6 MB) — Cosine 监督的 GCN
└── qm_9_2in1_model.pth    (~3.5 MB) — 拼接两路的 TwoInOne MLP
```

`fisd.py` 默认从 `WEIGHTS/FISD/` 加载，亦支持 `model_dir=` 参数覆盖路径。若文件缺失，运行时会报 `FileNotFoundError`。

### ATMOMACCS 说明

- **代码与权重**：[Zenodo / 18669279](https://zenodo.org/records/18669279)
- **文献**：[J. Chem. Phys. — DOI:10.1063/5.0308548](https://doi.org/10.1063/5.0308548)
- **许可证**：上游 ATMOMACCS 项目采用 **[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)** 协议，请遵守该协议及 Zenodo 存档页面上各文件附带的许可条款。

本项目的 ATMOMACCS 描述符**仅调用 RDKit 的标准 MACCS keys**（167 位去除占位 bit 0 后为 166 维），**无需下载**任何外部资产。

如您在论文中引用本项目的 ATMOMACCS 实验结果，请同时引用上游论文（[DOI:10.1063/5.0308548](https://doi.org/10.1063/5.0308548)）及 Zenodo 存档（[Zenodo 18669279](https://zenodo.org/records/18669279)）。

---

## 算法实现对照表

> 为保持仓库自包含（`git clone` 后即可直接 `python yonod.py`），
> FISD 等需要自行实现的算法已**直接 inline 到 `yonod/` 中**，未把上游源码 vendor 进本仓库。
> 下表给出 inline 代码 ↔ 上游来源 的逐行映射，便于审计或基于新数据重训练。

| 算法 | inline 实现位置 | 上游来源 / 外部库 |
|---|---|---|
| Morgan ECFP4 指纹 | [morgan.py:32-50](yonod/descriptors/morgan.py#L32-L50) | RDKit `rdkit.Chem.AllChem.GetMorganFingerprintAsBitVect`（仅作 API 调用） |
| MACCS keys（166 维） | [atmomaccs.py:30-44](yonod/descriptors/atmomaccs.py#L30-L44) | RDKit `rdkit.Chem.rdMolDescriptors.GetMACCSKeysFingerprint`；与上游 ATMOMACCS 的 `generate_MACCS.py` 完全等价（去掉占位 bit 0）。引用：[Zenodo 18669279](https://zenodo.org/records/18669279) / [DOI:10.1063/5.0308548](https://doi.org/10.1063/5.0308548) |
| FISD 双 GCN 嵌入（架构） | [fisd.py:51-95](yonod/descriptors/fisd.py#L51-L95) (`_GNN`、`_TwoInOne` 类) | 与上游 FISD `code/MLMS_mse.ipynb` / `MLMS_cos.ipynb` / `MLMS_2IN1.ipynb` 完全一致；上游：[KeantChen/FISD](https://github.com/KeantChen/FISD)；论文：[DOI:10.1039/D5SC00451A](https://doi.org/10.1039/D5SC00451A) |
| FISD 原子特征（45 维） | [fisd.py:99-132](yonod/descriptors/fisd.py#L99-L132) (`_atom_features`) | 与上游 `test_MLMS/reproduce_mlms.py::get_atom_features` 一致 |
| FISD 图构建 | [fisd.py:134-150](yonod/descriptors/fisd.py#L134-L150) (`_smiles_to_graph`) | 同上游；含单原子分子自环兜底 |
| FISD 前向 + 池化 | [fisd.py:153-216](yonod/descriptors/fisd.py#L153-L216) (`FISDDescriptor`) | 加载 3 个 `qm_9_*.pth`（`WEIGHTS/FISD/`），双 GCN → concat → TwoInOne → 50 维 |
| MolMetaLM 嵌入 | [molmetalm.py:80-149](yonod/descriptors/molmetalm.py#L80-L149) | HuggingFace [`wudejian789/MolMetaLM-base`](https://huggingface.co/wudejian789/MolMetaLM-base)；GitHub：[CSUBioGroup/MolMetaLM](https://github.com/CSUBioGroup/MolMetaLM)；论文：[DOI:10.48550/arXiv.2411.15500](https://doi.org/10.48550/arXiv.2411.15500)；用 `AutoModel`（带 CausalLM 兜底）+ attention-masked mean-pool 取 768 维 |
| 通用 CSV 自动探测 | [csv_loader.py](yonod/universal/csv_loader.py) | 本项目原创；用 RDKit 解析率 > threshold 判定 SMILES 列 |
| 通用特征矩阵构建 | [feature_builder.py](yonod/universal/feature_builder.py) | 本项目原创；SMILES 描述符 + 数值辅助列拼接 |
| 反应级 6 分子特征拼接 | [reaction_featurizer.py:33-57](yonod/features/reaction_featurizer.py#L33-L57) | 本项目原创设计（v0.3 §2.3），含 `(无)` 零向量 + `,` → `.` 预处理 |
| 试剂 SMILES 缓存 | [reagent_cache.py](yonod/features/reagent_cache.py) | 本项目原创；首次运行落 `cache/reagent_feats_<desc>.pkl` |
| XGBoost 回归适配器 | [xgb_model.py](yonod/models/xgb_model.py) | `xgboost.XGBRegressor`（`tree_method='hist', device='cuda'`） |
| RandomForest 适配器 | [rf_model.py](yonod/models/rf_model.py) | `sklearn.ensemble.RandomForestRegressor`（300 树，n_jobs=-1） |
| SVR + 自动 PCA + 子采样 | [svm_model.py](yonod/models/svm_model.py) | `sklearn.svm.SVR(kernel='rbf')` + `PCA(n=256)`（输入 >512 维触发）+ 每折随机抽样训练 |
| AutoGluon 适配器 | [autogluon_model.py](yonod/models/autogluon_model.py) | `autogluon.tabular.TabularPredictor`（medium_quality preset，80/20 holdout） |

---

## 命令行参数速查

### 统一入口 — `yonod.py`

`yonod.py` 同时支持**交互向导**和**CLI 模式**：

- **不带参数**：启动交互向导，逐步引导选择数据集、SMILES 列、标签列、描述符和模型
- **带 `--csv`**：直接进入 CLI pipeline，跳过向导

| 参数 | 默认 | 用途 |
|---|---|---|
| `--csv PATH` | 无（向导询问） | 数据集 CSV 路径 |
| `--label-col COL` | 自动探测最后一列浮点 | 回归目标列名 |
| `--smiles-cols COL [COL ...]` | 自动探测 | SMILES 列名（可多列） |
| `--numeric-cols COL [COL ...]` | 无 | 数值辅助列（如温度），会拼入特征 |
| `--task-name NAME` | CSV 文件名（无后缀） | 结果子目录名（`results/<NAME>/`） |
| `--descriptors {morgan,maccs,fisd,molmetalm}` | 全部 4 个 | 选择描述符（可多选） |
| `--models {xgb,rf,svm,autogluon}` | 全部 4 个 | 选择模型（可多选） |
| `--nrows N` | 全量 | 限制读取行数（调试时用 500） |
| `--cv K` | 5 | K 折交叉验证 |
| `--svm-subsample N` | 8000 | SVM 每折训练子采样（RBF SVR O(n²) 复杂度限制） |
| `--heartbeat SEC` | 30 | 长任务心跳间隔（0=关） |
| `--skip-plots` | False | 跳过散点图生成 |
| `--skip-report` | False | 跳过最后的 HTML 报告生成 |
| `--append` | False | 追加合并入已有 metrics_summary.csv |

**示例**：

```bash
# 酰胺缩合完整运行（全量 4×4，跳过 MolMetaLM）
python yonod.py \
    --csv dataset/amide-coupling.csv \
    --label-col yield \
    --smiles-cols sub_1_smiles sub_2_smiles activation_id additive_id base_id solvent_id \
    --descriptors morgan maccs fisd \
    --task-name amide-no-LLM

# 调试模式（500 行 + 只跑 morgan×rf）
python yonod.py \
    --csv dataset/amide-coupling.csv \
    --label-col yield \
    --smiles-cols sub_1_smiles sub_2_smiles activation_id additive_id base_id solvent_id \
    --nrows 500 --descriptors morgan --models rf --task-name debug
```

---

## 常见提示与警告

运行时出现以下提示均属正常，**不影响结果正确性**，可直接忽略。

### AutoGluon 相关

| 提示信息 | 原因 | 是否需要处理 |
|---|---|---|
| `[warning] AutoGluon + numeric cols: 使用全局 StandardScaler（非折内归一化）` | AutoGluon 内部使用 holdout 划分，无法嵌入 KFold 循环，数值辅助列（如温度）只能全局 z-score | 已知局限，不影响指标输出；若需严格无泄露评估请不使用 AutoGluon |
| `Warning: path already exists! This predictor may overwrite an existing predictor!` | 同一次 grid 运行中多个 (描述符 × 模型) 组合复用了同一 temp 目录 | 无需处理；本项目只取指标，不持久化 AutoGluon 模型 |
| `pkg_resources is deprecated as an API` | AutoGluon 内部某处仍依赖已废弃的 `pkg_resources` | 无需处理；升级 AutoGluon 版本可消除 |
| `No valid features to train KNeighborsUnif / KNeighborsDist... Skipping` | Morgan / MACCS 指纹是高维稀疏向量，KNN 距离度量在高维空间失效 | 无需处理；AutoGluon 会用其他子模型继续训练 |
| `NeuralNetFastAI failed (ImportError)... Skipping` | 当前环境未安装 `fastai` | 无需处理（见下方说明） |

### 关于 fastai 安装失败

`fastai 1.1.1` 依赖 `spacy`，而 `spacy` 的最新版要求 `thinc >= 8.3.12`（仅支持 Python ≥ 3.10），与本项目的 **Python 3.9** 环境冲突，无法安装。

**建议直接跳过 fastai**：FastAI 只是 AutoGluon 16 个内置子模型之一，其余主力模型（LightGBM、XGBoost、RandomForest、ExtraTrees、CatBoost 等）均正常运行。对分子描述符（高维稀疏指纹）任务，树模型通常优于 FastAI tabular，安装 fastai 不会带来明显的 R² 提升。

## License

本项目代码与文档采用 **[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/)** 协议：

- ✅ **允许**：自由分享、改编、二次创作；
- 📌 **要求**：署名（必须保留 `https://github.com/thinktraveller/YONOD` 作为来源）；
- ❌ **限制**：**不得用于商业目的**。

完整协议见 [LICENSE](LICENSE) 文件。

