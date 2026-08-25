# YONOD：酰胺缩合反应产率预测专题 构建计划书

> 平台名称：**YONOD**（Your One-stop Notebook Of Descriptors）
> 专题方向：**酰胺缩合反应产率预测（Amide Bond Formation Yield Prediction）**
> 最终交付物：一个可直接执行的 Python 脚本 `run_yield_prediction.py`
> 文档版本：**v0.3**（2026-05-14）
>
> **变更记录**
> - **v0.3**（本版本）：项目方向从"通用 SMILES + target 平台"收窄为"酰胺缩合反应产率预测专题"。主要变更：
>   1. 任务定义改为回归预测酰胺缩合 yield（0~1）；
>   2. 环境改为 **Python 3.9 + CUDA 12.1 + torch 2.1.2**（适配离线 whl 轮子）；
>   3. 交付物从 Streamlit web app 改为单一可执行脚本 `run_yield_prediction.py`；
>   4. 输入特征改为"反应级特征拼接"：两个底物 + 4 个试剂全部通过 SMILES 描述符化（见 §6.7）；
>   5. 移除 Streamlit、SHAP、zip 打包、GitHub Actions、子进程跨 venv 等过度工程化组件；
>   6. 修正所有路径，项目根为 `YONOD/`，权重已存在于 `WEIGHTS/MolMetaLM-base/`；
>   7. 每个任务补充了可直接粘贴执行的 PowerShell 命令；
>   8. AutoGluon 保留但简化（无异步、无前端调度）；
>   9. **试剂 SMILES 化**：4 个试剂列改为通过"试剂编号-SMILES 对应表"映射后再描述符化，`(无)` 用零向量填充，反应级特征 = 6 个分子描述符拼接；
>   10. **DFT 暂缓**：从主流程剔除 DFT 描述符，描述符总数从 5 降为 **4**，结果矩阵由 5×4=20 变为 **4×4=16**，原 T2.D1~T2.D4 任务删除。
>   11. **包管理改用 conda**：环境约定 `yonod`，T0.2 命令从 `py -3.9 -m venv` 改为 `conda create -n yonod python=3.9`；所有 `Activate.ps1` 调用替换为 `conda activate yonod`；离线 torch whl 仍通过 pip 装（conda 无同版本通道）。理由见 §九 Q11。
>   12. **数据集已预先完成 id → SMILES 映射**（2026-05-15 实测确认）：CSV 中 `activation_id` / `additive_id` / `base_id` / `solvent_id` 4 列直接存储 SMILES 字符串（含 `,` 分隔的盐/复合物形式），**不再需要解析"试剂编号-SMILES对应表.md"构建字典**。T0.4 由"解析映射表"改为"提取唯一试剂 SMILES 集 + 预计算描述符缓存"；T1.3 从"按 id 查字典"改为"直接对 SMILES 列调用描述符（含 `(无)` → 零向量、`,` → `.` 多片段合并）"。详见 §九 Q12。
> - **v0.3.2**（2026-05-15，全量执行配套）：
>   13. **全量 4×4 grid 改为分两阶段执行**（方案 C）：先跑 12 组非 RF 组合，再跑 4 个 RF 组合，用 `--append` 合并入同一 `metrics_summary.csv`。理由是 RF 在 Morgan/MolMetaLM 高维特征上单折 7~8 分钟、5 折 ≈ 38 分钟，本地一次跑完时间窗太长。详见新增 T4.4 + Q13。
>   14. **主脚本新增 7 个 CLI 参数**：`--log-file` / `--heartbeat` / `--rf-verbose` / `--rf-n-jobs` / `--rf-n-estimators` / `--rf-max-depth` / `--append`，满足长任务可观测性 + 分阶段合并需求。
>   15. **新增辅助脚本** `verify_morgan_rf.py`：独立诊断 RF 在全量数据上的单折耗时，verbose=2 显示每棵树进度，找出 RF 慢/卡的根因。
>   16. **云端迁移准备**：新增 §五 T4.5 列出迁移清单（数据 + 代码 + 权重路径）和云端 conda 环境搭建命令，为后续高配云服务器一次跑完全量 4×4 做准备。
>   17. **SVM 加子采样训练**（2026-05-15 实测后修订）：实测 Morgan × SVM 在全量 47015 行上单折 > 75 分钟仍未出结果，原因是 RBF SVR 的 O(n²d) 复杂度在 n > 10k 时无法承受。修复方案：SVR 适配器新增 `subsample_n` 参数，主脚本默认 `--svm-subsample 8000`；每折训练前从 train fold 随机抽 8000 行，测试集不变。预计 SVM 单折时间从 75+ 分钟降至 1~3 分钟。详见 Q14。
> - v0.2：HSPOC 暂缓；新增 Morgan 1024 维指纹；补充 MolMetaLM 权重下载指引。
> - v0.1：初始版本，5 描述符 × 4 算法总体规划。

---

## 一、项目概述

### 1.1 项目目标

**输入**：酰胺缩合反应数据集 CSV，每行包含：
- `sub_1_smiles`、`sub_2_smiles`（两个底物的 SMILES）
- `activation_id`、`additive_id`、`base_id`、`solvent_id`（试剂编号，可能为"(无)"）
- `yield`（实验产率，0~1 浮点数，回归目标）

**处理**：
1. 对 sub_1、sub_2 及 4 个试剂列（activation / additive / base / solvent，**已是 SMILES，无需映射**）各自生成 4 类描述符向量，6 个向量拼接为反应级特征；`(无)` 用零向量填充，`,` 分隔的多片段 SMILES 先转换为 RDKit 标准的 `.` 连接后整体描述符化；样本不丢弃；
2. 用 4 种 ML 模型（XGBoost / SVM / RF / AutoGluon）在 5 折 CV 下训练和评估；
3. 输出 R²、RMSE、MAE 评估表 + 预测 vs 真实散点图 PNG。

**输出**：
- 控制台打印评估指标表（4 描述符 × 4 模型 = 最多 16 行）
- `results/scatter_<desc>_<model>.png`：每个组合的散点图
- `results/metrics_summary.csv`：汇总所有模型指标

### 1.1.1 描述符清单（v0.3，针对全部 6 个反应组分）

| # | 描述符 | 用途 | 维度（单分子） | 反应特征维度（×6 分子） |
|---|---|---|---|---|
| 1 | **Morgan ECFP4** | RDKit 直接调用 | 1024 bit | 6144 |
| 2 | **ATMOMACCS** | `化学描述符相关项目/ATMOMACCS/` | ~167 | ~1002 |
| 3 | **FISD** | `化学描述符相关项目/FISD/` | 50 | 300 |
| 4 | **MolMetaLM Embedding** | `WEIGHTS/MolMetaLM-base/`（已下载） | 768 | 4608 |

> DFT 描述符已从 v0.3 主流程移除，原因见 §6.6。

**试剂特征**：CSV 的 4 个试剂列已直接为 SMILES（无需映射），与底物采用相同描述符方法生成向量；`(无)` 对应零向量（维度与其他试剂描述符相同）；含 `,` 的多片段 SMILES 在解析前转 `.` 连接。最终反应特征 = 6 个分子向量拼接（见 §2.3）。

### 1.2 预期成果

| 形态 | 内容 |
|---|---|
| 可执行脚本 | `run_yield_prediction.py`，命令行参数指定 CSV 路径 |
| 评估报告 | `results/metrics_summary.csv` + 散点图 PNG |
| 环境文件 | `requirements_py39.txt`（含离线 torch whl 安装说明） |

### 1.3 项目定位

本专题是大创项目的**核心研究内容**。研究意义：
- 酰胺键形成反应（peptide bond coupling）是制药工业最常见反应之一；
- 产率预测模型可辅助实验设计，减少试错成本；
- 4 类描述符对比实验具有学术创新价值（比较经典指纹 vs 深度学习嵌入在反应预测中的性能）。

---

## 二、可行性分析

### 2.1 描述符接入情况

| 描述符 | 可行性 | 反应级拼接维度（×6） | 状态 |
|---|---|---|---|
| Morgan ECFP4 | 完全可行，零外部依赖 | 6144 | v0.3 主流程 |
| ATMOMACCS | 可行，需读现有代码 | ~1002 | v0.3 主流程 |
| FISD | 可行，需 torch 环境 | 300 | v0.3 主流程 |
| MolMetaLM | 可行，权重已就绪 | 4608 | v0.3 主流程 |
| DFT | 需 Gaussian log，本数据集不可用 | — | 暂缓，见 §6.6 |

### 2.2 数据集情况

| 项目 | 值 |
|---|---|
| 文件路径 | `C:\Users\joyjo\Desktop\其他大学资料\大创\YONOD\数据集\酰胺缩合反应数据集.csv` |
| 样本量 | 47015 条反应 |
| 列结构 | `row_id, sub_1_smiles, sub_2_smiles, product_smiles, activation_id, additive_id, base_id, solvent_id, yield` |
| **试剂列实际内容** | **`*_id` 列已直接存储 SMILES 字符串**（CSV 已完成 id → SMILES 映射），无需再读取"试剂编号-SMILES对应表.md"做转换；该表仅作人类可读的索引参考 |
| 多片段格式 | 部分试剂为盐型/复合物，用 `,` 分隔多段 SMILES（如 `CCN=C=NCCCN(C)C,Cl`、`[B-](F)(F)(F)F,CN(C)C(=[N+](C)C)ON1C(=O)CCC1=O`），描述符化前先把 `,` 替换为 `.`（RDKit 多片段标准分隔符）整体解析 |
| 特殊值 | 试剂列可能为 `(无)`（如不加添加剂或不加碱的反应条件），此时描述符用零向量填充，样本不丢弃 |
| 目标值 | `yield` 为 0~1 浮点数，连续回归任务 |

### 2.3 反应级特征构造方案

```
reaction_feature = concat([
    desc(sub_1_smiles),                    # shape (d,)
    desc(sub_2_smiles),                    # shape (d,)
    desc(normalize(activation_id)),        # 已是 SMILES，仅做 (,→.) 与 (无)→zeros 预处理
    desc(normalize(additive_id)),
    desc(normalize(base_id)),
    desc(normalize(solvent_id)),
])
# 总维度 = 6 × d_descriptor
```

**预处理规则 `normalize(s)`（数据加载阶段统一应用于 4 个试剂列）**：
1. 若 `s == "(无)"` → 返回特殊标记 `None`（描述符层把它映射为零向量，不调用模型）；
2. 若 `s` 含 `,` → 将所有 `,` 替换为 `.`（RDKit 多片段标准分隔符，例如 `CCN=C=NCCCN(C)C,Cl` → `CCN=C=NCCCN(C)C.Cl`）；
3. 其余 SMILES 原样传给描述符。

**唯一试剂去重 + 缓存**：尽管 CSV 共 47015 行，但 4 个试剂列的唯一 SMILES 数量很小（粗估 < 70 种）；T0.4 提取这些唯一值统一描述符化一次后缓存到 `cache/reagent_feats_{desc_name}.pkl`，后续按 SMILES 字符串作 key 查表。`(无)` 对应缓存中的零向量。

**反应级维度汇总**：

| 描述符 | 单分子维度 d | 反应级维度（×6） |
|---|---|---|
| Morgan ECFP4 | 1024 | 6144 |
| ATMOMACCS | ~167 | ~1002 |
| FISD | 50 | 300 |
| MolMetaLM | 768 | 4608 |

**SVM PCA 阈值调整**：原方案 >256 维触发 PCA 至 128 维已不适用（最小拼接维度 300）。v0.3 调整为：**输入维度 > 512 时，自动 PCA 至 256 维**。

### 2.4 资源需求估算

| 部件 | 需求 |
|---|---|
| CPU | 4 核+，RAM 16 GB+ |
| GPU | 已有，用于 MolMetaLM 推理加速（CUDA 12.1） |
| 磁盘 | MolMetaLM 权重已占用约 500MB，额外需 2GB |
| 47015 条特征化时间 | Morgan：~2 分钟；MolMetaLM（GPU）：~20 分钟（6 列 × 约 3 分钟） |

### 2.5 风险与应对

| 风险 | 概率 | 应对 |
|---|---|---|
| ATMOMACCS 接口路径不对 | 中 | 先运行原项目的 notebook，确认入口函数后再接入 |
| FISD 的 .pth 文件路径问题 | 中 | 在描述符适配器中将路径做为参数传入，不硬编码 |
| MolMetaLM 47015 条推理 OOM | 低 | batch_size 从 32 开始，OOM 时自动减半 |
| 试剂 SMILES 映射缺失 | 低 | 映射表已覆盖全部 R/A/B/S 类，读表时对缺失 key 抛警告后填零向量 |
| AutoGluon 在 Windows 上多进程问题 | 中 | 用 `num_cpus=1` + `time_limit=300` 规避 |

---

## 三、环境配置

### 3.1 硬件要求

- CPU：4 核以上
- RAM：16 GB 以上
- GPU：已有，CUDA 12.1（对应离线 whl）
- 磁盘：额外 5 GB 空闲（权重已下载）

### 3.2 软件环境

- OS：Windows 11（当前开发环境）
- **Python 3.9**（严格要求，因为离线 torch whl 是 cp39 版本）
- 包管理：**conda**（Miniconda / Anaconda 均可），环境名约定 `yonod`
- CUDA：12.1

### 3.3 离线 torch 轮子位置

```text
H:\AI模型及torch包\torch_wheels\
├── torch-2.1.2+cu121-cp39-cp39-win_amd64.whl
├── torchaudio-2.1.2+cu121-cp39-cp39-win_amd64.whl
└── torchvision-0.16.2+cu121-cp39-cp39-win_amd64.whl
```

### 3.4 依赖清单（核心）

```text
# 注意：torch 系列必须先从本地 whl 离线安装，不能用 pip install torch（会装 CPU 版）

# 描述符与化学信息学
rdkit==2023.9.5
transformers==4.40.0
networkx==3.2.1

# ML
scikit-learn==1.4.0
xgboost==2.0.3
autogluon.tabular==1.1.1

# 数据处理与可视化
pandas==2.0.3
numpy==1.26.4
matplotlib==3.8.4

# 工具
joblib==1.3.2
pyyaml==6.0.1
tqdm==4.66.2
```

### 3.5 MolMetaLM 权重说明

> **权重已存在于 `WEIGHTS/MolMetaLM-base/`，无需重新下载。** 如需重新下载，请按以下步骤操作。

**权重文件清单**（已确认存在）：

```text
YONOD/WEIGHTS/MolMetaLM-base/
├── README.md
├── added_tokens.json
├── config.json
├── generation_config.json
├── model.safetensors
├── special_tokens_map.json
├── tokenizer.json
├── tokenizer_config.json
└── vocab.txt
```

**重新下载方法（仅供参考）**：

```powershell
# 设置国内镜像（国内用户）
$env:HF_ENDPOINT = "https://hf-mirror.com"
pip install -U "huggingface_hub>=0.34"

# 下载到正确位置
hf download wudejian789/MolMetaLM-base `
    --local-dir "C:\Users\joyjo\Desktop\其他大学资料\大创\YONOD\WEIGHTS\MolMetaLM-base"
```

**验证权重可用性**：

```python
from transformers import AutoTokenizer, AutoModel
import torch

weight_path = r"C:\Users\joyjo\Desktop\其他大学资料\大创\YONOD\WEIGHTS\MolMetaLM-base"
tok = AutoTokenizer.from_pretrained(weight_path)
mdl = AutoModel.from_pretrained(weight_path)
print("权重加载成功，词表大小 =", tok.vocab_size)
```

---

## 四、目录结构（v0.3 修正版）

```text
YONOD/                                    # 项目根目录
├── YONOD项目构建计划书.md                # 本文档
├── 化学描述符相关项目/                   # 既有项目（保持只读）
│   ├── ATMOMACCS/
│   ├── DFTDescriptorPipeline/
│   ├── FISD/
│   ├── HSPOC/
│   └── MolMetaLM/
├── 数据集/
│   ├── 酰胺缩合反应数据集.csv            # 47015 条反应数据
│   └── 试剂编号-SMILES对应表.md          # R/A/B/S id → SMILES 映射
├── WEIGHTS/
│   └── MolMetaLM-base/                   # ★ 权重已就绪
│       ├── config.json
│       ├── model.safetensors
│       ├── tokenizer.json
│       └── ...
├── yonod_yield/                          # ★ v0.3 新建的主代码目录
│   ├── descriptors/
│   │   ├── base.py
│   │   ├── morgan.py
│   │   ├── atmomaccs.py
│   │   ├── fisd.py
│   │   └── molmetalm.py
│   ├── models/
│   │   ├── xgb_model.py
│   │   ├── svm_model.py
│   │   ├── rf_model.py
│   │   └── autogluon_model.py
│   ├── features/
│   │   ├── reagent_cache.py              # 提取唯一试剂 SMILES + 描述符缓存（含 (无)/逗号 预处理）
│   │   └── reaction_featurizer.py        # 组合反应级特征（6 分子拼接）
│   ├── evaluate.py                       # CV 评估 + 指标计算
│   ├── plot.py                           # 散点图生成
│   └── utils.py                          # 数据读取、路径工具
├── run_yield_prediction.py               # ★ 最终可执行入口脚本
├── requirements_py39.txt                 # 依赖清单（含 whl 安装说明）
├── cache/                                # 描述符特征缓存（.npz）
└── results/                              # 输出目录（散点图 + metrics.csv）
```

---

## 五、实施步骤（任务清单）

> 每个任务含：操作要点 · 产出物 · 验收标准 · **可直接执行的 PowerShell 命令**。

---

### 阶段 0：环境搭建与数据探索（第 1 周前半）

**目标**：建好 Python 3.9 conda 环境，装好离线 torch，读通数据集，建好目录骨架。

---

#### T0.1 创建项目代码目录骨架

**操作要点**：在 `YONOD/` 下新建 `yonod_yield/` 及其子目录、`results/`、`cache/`。

**产出物**：目录结构骨架。

**验收标准**：`ls YONOD/yonod_yield/` 能看到 descriptors、models、features 三个子目录。

**PowerShell 命令**：

```powershell
$root = "C:\Users\joyjo\Desktop\其他大学资料\大创\YONOD"
New-Item -ItemType Directory -Force "$root\yonod_yield\descriptors"
New-Item -ItemType Directory -Force "$root\yonod_yield\models"
New-Item -ItemType Directory -Force "$root\yonod_yield\features"
New-Item -ItemType Directory -Force "$root\results"
New-Item -ItemType Directory -Force "$root\cache"
# 创建空的 __init__.py
"" | Out-File -FilePath "$root\yonod_yield\__init__.py" -Encoding utf8
"" | Out-File -FilePath "$root\yonod_yield\descriptors\__init__.py" -Encoding utf8
"" | Out-File -FilePath "$root\yonod_yield\models\__init__.py" -Encoding utf8
"" | Out-File -FilePath "$root\yonod_yield\features\__init__.py" -Encoding utf8
Write-Host "目录骨架创建完成"
```

---

#### T0.2 创建 conda 环境并安装依赖

**操作要点**：
1. 确认已安装 Miniconda 或 Anaconda（检查 `conda --version`）；
2. 用 conda 创建 Python 3.9 环境 `yonod`；
3. 在环境内先用 pip 离线安装 torch 三件套（顺序：torch → torchvision → torchaudio）；
4. 再用 pip 安装其余依赖（rdkit 也直接走 pip，与 transformers/sklearn 版本对齐更稳）。

**为什么 conda 内仍用 pip 装包？**
- 离线 torch whl 是 pip 格式，conda 没有同版本通道；
- transformers / xgboost / autogluon 等在 PyPI 上更新更及时；
- conda 主要作用是**隔离 Python 解释器**和后续可能引入的非 Python 二进制依赖（如未来要装 cudatoolkit、openbabel 时切换更方便）。

**产出物**：conda 环境 `yonod`，依赖全部就绪。

**验收标准**：`python -c "import torch; print(torch.cuda.is_available())"` 输出 `True`。

**PowerShell 命令**：

```powershell
$root = "C:\Users\joyjo\Desktop\其他大学资料\大创\YONOD"
$whl  = "H:\AI模型及torch包\torch_wheels"

# 步骤 1：确认 conda 可用
conda --version

# 步骤 2：创建并激活环境（约定环境名 yonod）
conda create -n yonod python=3.9 -y
conda activate yonod

# 步骤 3：升级 pip
python -m pip install --upgrade pip

# 步骤 4：离线安装 torch（必须先装 torch 本体）
pip install "$whl\torch-2.1.2+cu121-cp39-cp39-win_amd64.whl"
pip install "$whl\torchvision-0.16.2+cu121-cp39-cp39-win_amd64.whl"
pip install "$whl\torchaudio-2.1.2+cu121-cp39-cp39-win_amd64.whl"

# 步骤 5：安装其余依赖
pip install rdkit==2023.9.5 transformers==4.40.0 networkx==3.2.1
pip install scikit-learn==1.4.0 xgboost==2.0.3
pip install pandas==2.0.3 numpy==1.26.4 matplotlib==3.8.4
pip install joblib==1.3.2 pyyaml==6.0.1 tqdm==4.66.2

# 验证
python -c "import torch; print('CUDA 可用:', torch.cuda.is_available()); print('版本:', torch.__version__)"
```

**常见问题**：
- `conda` 命令找不到：先安装 Miniconda（[https://docs.conda.io/en/latest/miniconda.html](https://docs.conda.io/en/latest/miniconda.html)），或使用 Anaconda Prompt / 在 PowerShell 执行一次 `conda init powershell` 后重启终端。
- `conda activate` 在 PowerShell 报错"CommandNotFoundError"：同上，需要 `conda init powershell` 后重启。
- torch 安装报错"不是有效的 wheel"：检查 whl 文件名是否完整，路径是否含空格（用引号包裹）。
- 若希望用 conda 直接装 rdkit（不走 pip），可改为：`conda install -c conda-forge rdkit=2023.9.5 -y`。

---

#### T0.3 数据探索：读取 CSV 并统计各列情况

**操作要点**：用 pandas 读 CSV，统计 `activation_id`、`additive_id`、`base_id`、`solvent_id` 的唯一值数量（确认与映射表行数一致）；检查 yield 分布；确认 `(无)` 出现的列和频次。

**产出物**：控制台打印统计摘要。

**验收标准**：打印出 4 个 id 列的 value_counts（前 10）和 yield 的 mean/std；确认每列 id 均能在映射表中找到对应。

**PowerShell 命令**：

```powershell
$root = "C:\Users\joyjo\Desktop\其他大学资料\大创\YONOD"
conda activate yonod
python -c "
import pandas as pd
df = pd.read_csv(r'$root\数据集\酰胺缩合反应数据集.csv')
print('形状:', df.shape)
for col in ['activation_id','additive_id','base_id','solvent_id']:
    vc = df[col].value_counts()
    print(f'--- {col} (共 {df[col].nunique()} 种) ---')
    print(vc.head(5))
print('yield 统计:')
print(df['yield'].describe())
"
```

---

#### T0.4 提取唯一试剂 SMILES 集 + 构建描述符缓存

> **背景说明（v0.3.1 修订）**：实测确认 CSV 中 `activation_id` / `additive_id` / `base_id` / `solvent_id` 4 列已直接存储 SMILES 字符串（含 `,` 分隔的盐/复合物形式），无需读取"试剂编号-SMILES对应表.md"做映射。本任务由原"解析映射表"改为"从 CSV 直接收集唯一 SMILES 并预计算描述符"。

**操作要点**：
- 在 `yonod_yield/features/reagent_cache.py` 实现：
  1. `normalize_reagent_smiles(s: str) -> str | None`：`(无)` → `None`；含 `,` → 替换为 `.`；其余原样返回；
  2. `collect_unique_reagents(df) -> set[str | None]`：遍历 4 个试剂列，应用 `normalize_*` 后去重；
  3. `build_reagent_feat_cache(descriptor, df) -> dict[str | None, np.ndarray]`：对所有非 None 唯一 SMILES 调用 `descriptor.featurize()`，结果存字典；`None` 对应 `np.zeros(d)`；持久化到 `cache/reagent_feats_{desc_name}.pkl`。
- 后续 `reaction_featurizer` 按归一化后的 SMILES 字符串作 key 直接查表，**绝不重复调用描述符模型**。

**产出物**：`yonod_yield/features/reagent_cache.py`。

**验收标准**：
- 函数对 47015 行 CSV 应能在 < 5 秒内完成全部唯一试剂的描述符化（Morgan / 简单描述符；MolMetaLM 因加载模型会更慢但 < 60 秒）；
- `(无)` 映射为全零向量；
- 含 `,` 的 SMILES 能被 RDKit 正常解析（无 `MolFromSmiles` 返回 None）。

**PowerShell 命令**（验证当前 CSV 中试剂列的唯一值与规范化）：

```powershell
$root = "C:\Users\joyjo\Desktop\其他大学资料\大创\YONOD"
conda activate yonod
python -c "
import pandas as pd
csv = r'$root\数据集\酰胺缩合反应数据集.csv'
df = pd.read_csv(csv)
cols = ['activation_id','additive_id','base_id','solvent_id']

def normalize(s):
    if s == '(无)': return None
    return s.replace(',', '.')

unique = set()
for c in cols:
    for v in df[c].unique():
        unique.add(normalize(v))
print('唯一试剂 SMILES（含 None） 共:', len(unique))
print('其中 None（即 (无)） 数量:', sum(1 for v in unique if v is None))
# 抽样几条
sample = [v for v in unique if v is not None][:5]
for s in sample:
    print(' sample:', s)

# 用 RDKit 验证是否都能解析
from rdkit import Chem
bad = [s for s in unique if s is not None and Chem.MolFromSmiles(s) is None]
print('无法被 RDKit 解析的 SMILES 数量:', len(bad))
for s in bad[:5]:
    print('  bad:', s)
"
```

---

### 阶段 1：反应级特征工程（第 1 周后半）

**目标**：完成 Morgan 描述符 + 试剂 SMILES 描述符化的反应级特征构造（6 分子拼接），建立统一接口。

---

#### T1.1 实现 BaseDescriptor 抽象

**操作要点**：每个描述符封装为 `BaseDescriptor` 子类，统一接口 `featurize(smiles_list) -> (np.ndarray, mask)`。mask 为布尔数组，标记特征化成功的样本（失败的用零向量填充）。

**产出物**：`yonod_yield/descriptors/base.py`。

**验收标准**：文件语法无误，接口有 docstring。

**PowerShell 命令**：

```powershell
# 验证文件语法
$root = "C:\Users\joyjo\Desktop\其他大学资料\大创\YONOD"
conda activate yonod
python -m py_compile "$root\yonod_yield\descriptors\base.py"; if ($?) { Write-Host "语法检查通过" }
```

---

#### T1.2 实现 Morgan ECFP4 描述符

**操作要点**：
- 调用 `rdkit.Chem.AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024)`
- 输出 `(n, 1024)` uint8 数组
- 非法 SMILES → mask 置 False，对应行填 0

**产出物**：`yonod_yield/descriptors/morgan.py`。

**验收标准**：对数据集前 100 行的 sub_1_smiles 能输出 shape=(合法数量, 1024)，mask 无异常。

**PowerShell 命令**：

```powershell
$root = "C:\Users\joyjo\Desktop\其他大学资料\大创\YONOD"
conda activate yonod
python -c "
import sys
sys.path.insert(0, r'$root')
import pandas as pd, numpy as np
from yonod_yield.descriptors.morgan import MorganDescriptor
df = pd.read_csv(r'$root\数据集\酰胺缩合反应数据集.csv', nrows=100)
desc = MorganDescriptor()
feats, mask = desc.featurize(df['sub_1_smiles'].tolist())
print('shape:', feats.shape, 'mask sum:', mask.sum())
"
```

---

#### T1.3 实现试剂描述符缓存与查表接口

**操作要点**：
- 在 T0.4 创建的 `reagent_cache.py` 中追加：
  - 缓存持久化：将 `dict[str | None, np.ndarray]` 用 `joblib.dump` 存到 `cache/reagent_feats_{desc_name}.pkl`；
  - 启动时若 pkl 已存在则 `joblib.load` 跳过重建；
  - `get_reagent_feat(reagent_smiles_raw: str, cache: dict, d: int) -> np.ndarray`：内部先 `normalize_reagent_smiles()`，然后查 cache；若缺失（不该出现，因为 cache 已覆盖全部唯一值）则抛 KeyError 而不是静默零向量，方便发现 bug。
- 对于同一 SMILES 的试剂直接从字典查找，**绝不重复调用描述符模型**。

**产出物**：`yonod_yield/features/reagent_cache.py`（T0.4 已创建文件骨架，本任务补充缓存持久化与查表接口）。

**验收标准**：对 47015 条数据，4 个试剂列全部查字典后输出 shape 为 `(47015, 4 × d)`；第二次运行命中磁盘缓存，构建耗时 < 1 秒。

**PowerShell 命令**：

```powershell
$root = "C:\Users\joyjo\Desktop\其他大学资料\大创\YONOD"
conda activate yonod
python -c "
import sys, pandas as pd, numpy as np, joblib, time
sys.path.insert(0, r'$root')
from yonod_yield.features.reagent_cache import build_reagent_feat_cache, get_reagent_feat
from yonod_yield.descriptors.morgan import MorganDescriptor

csv = r'$root\数据集\酰胺缩合反应数据集.csv'
df = pd.read_csv(csv)
t0 = time.time()
cache = build_reagent_feat_cache(MorganDescriptor(), df)
print(f'缓存构建完成，共 {len(cache)} 种唯一试剂（含 None），耗时 {time.time()-t0:.1f}s')
print('None (即 (无)) 向量 sum:', cache[None].sum(), '（预期为 0）')
# 抽样一条非空
nonnull = [k for k in cache if k is not None][0]
print(f'sample {nonnull[:40]}... 向量 sum:', cache[nonnull].sum(), '（预期非 0）')
joblib.dump(cache, r'$root\cache\reagent_feats_morgan.pkl')
print('试剂描述符缓存已保存')
"
```

---

#### T1.4 实现反应级特征组合器

**操作要点**：
- `ReactionFeaturizer(descriptor)` 初始化只需描述符对象，不再接收映射表路径；
- `transform(df)` 流程：
  1. 调用 `build_reagent_feat_cache(descriptor, df)` 构建（或从磁盘加载）试剂描述符缓存；
  2. 对 sub_1、sub_2 列调用 `descriptor.featurize()`（这两列总是有效 SMILES，不需要 normalize）；
  3. 对 4 个试剂列逐行 `get_reagent_feat(raw_smiles, cache, d)` 查表；
  4. `np.concatenate` 拼接 6 个向量为 `(d * 6,)`；
- mask 仅由 sub_1、sub_2 的 featurize 成败决定（试剂列已被缓存覆盖，不影响 mask）。

**产出物**：`yonod_yield/features/reaction_featurizer.py`。

**验收标准**：对 100 条数据，Morgan 方案输出 shape 为 `(≤100, 6144)`；FISD 为 `(≤100, 300)`；mask 合理。

**PowerShell 命令**：

```powershell
$root = "C:\Users\joyjo\Desktop\其他大学资料\大创\YONOD"
conda activate yonod
python -c "
import sys, pandas as pd
sys.path.insert(0, r'$root')
from yonod_yield.features.reaction_featurizer import ReactionFeaturizer
from yonod_yield.descriptors.morgan import MorganDescriptor
df = pd.read_csv(r'$root\数据集\酰胺缩合反应数据集.csv', nrows=100)
feat = ReactionFeaturizer(descriptor=MorganDescriptor())
X, mask = feat.transform(df)
print('反应特征 shape:', X.shape, '（预期: (<=100, 6144)）')
print('mask sum:', mask.sum())
"
```

---

### 阶段 2：ML 模型接入（第 2 周）

**目标**：4 种 ML 模型均可在反应级特征上完成 5 折 CV 训练，输出 R²/RMSE/MAE。

---

#### T2.1 实现 XGBoost 回归器适配器

**操作要点**：包装 `XGBRegressor`（`n_estimators=300, learning_rate=0.05, tree_method='hist', device='cuda'`），`cross_val_score` 5 折，返回 `{R2_mean, R2_std, RMSE_mean, MAE_mean, train_time}`。

**产出物**：`yonod_yield/models/xgb_model.py`。

**验收标准**：在数据集前 500 条 + Morgan 特征上，5 折 R² 均值 > 0.3（产率预测任务的合理下界）。

**PowerShell 命令**：

```powershell
$root = "C:\Users\joyjo\Desktop\其他大学资料\大创\YONOD"
conda activate yonod
python -c "
import sys, pandas as pd, numpy as np
sys.path.insert(0, r'$root')
from yonod_yield.features.reaction_featurizer import ReactionFeaturizer
from yonod_yield.descriptors.morgan import MorganDescriptor
from yonod_yield.models.xgb_model import XGBYieldModel
df = pd.read_csv(r'$root\数据集\酰胺缩合反应数据集.csv', nrows=500)
feat = ReactionFeaturizer(MorganDescriptor())
X, mask = feat.transform(df)
y = df['yield'].values[mask]
model = XGBYieldModel()
metrics = model.cross_validate(X, y, cv=5)
print(metrics)
"
```

---

#### T2.2 实现 Random Forest 适配器

**操作要点**：包装 `RandomForestRegressor(n_estimators=300, n_jobs=-1, random_state=42)`，接口与 T2.1 相同。

**产出物**：`yonod_yield/models/rf_model.py`。

**验收标准**：同 T2.1，500 条 Morgan 特征上 R² > 0.2。

**PowerShell 命令**：

```powershell
# 将上方命令中 XGBYieldModel 替换为 RFYieldModel 即可，路径相同
$root = "C:\Users\joyjo\Desktop\其他大学资料\大创\YONOD"
conda activate yonod
python -c "
import sys, pandas as pd
sys.path.insert(0, r'$root')
from yonod_yield.features.reaction_featurizer import ReactionFeaturizer
from yonod_yield.descriptors.morgan import MorganDescriptor
from yonod_yield.models.rf_model import RFYieldModel
df = pd.read_csv(r'$root\数据集\酰胺缩合反应数据集.csv', nrows=500)
X, mask = ReactionFeaturizer(MorganDescriptor()).transform(df)
y = df['yield'].values[mask]
print(RFYieldModel().cross_validate(X, y, cv=5))
"
```

---

#### T2.3 实现 SVM 适配器

**操作要点**：
- 包装 `SVR(kernel='rbf', C=1.0)`
- **当输入维度 > 512 时，自动 PCA 至 256 维**（参数 `auto_pca=True`，阈值从原来 256→512，目标维度从 128→256），适配试剂描述符化后的最小拼接维度（FISD 300 维无需 PCA，Morgan 6144 / MolMetaLM 4608 / ATMOMACCS ~1002 均需 PCA）
- 接口统一

**产出物**：`yonod_yield/models/svm_model.py`。

**验收标准**：6144 维 Morgan 特征输入时，自动降维到 256，500 条数据 5 折 CV 在 5 分钟内完成。

**PowerShell 命令**：

```powershell
$root = "C:\Users\joyjo\Desktop\其他大学资料\大创\YONOD"
conda activate yonod
python -c "
import sys, pandas as pd
sys.path.insert(0, r'$root')
from yonod_yield.features.reaction_featurizer import ReactionFeaturizer
from yonod_yield.descriptors.morgan import MorganDescriptor
from yonod_yield.models.svm_model import SVMYieldModel
df = pd.read_csv(r'$root\数据集\酰胺缩合反应数据集.csv', nrows=500)
X, mask = ReactionFeaturizer(MorganDescriptor()).transform(df)
y = df['yield'].values[mask]
print('输入维度:', X.shape[1], '（预期 6144，将自动 PCA 至 256）')
print(SVMYieldModel(auto_pca=True).cross_validate(X, y, cv=5))
"
```

---

#### T2.4 实现 AutoGluon 适配器

**操作要点**：
- `pip install autogluon.tabular==1.1.1`（需在 conda 环境 `yonod` 激活后单独运行，体积大）
- 包装 `TabularPredictor(label='yield', problem_type='regression')`
- 设置 `presets='medium_quality'`、`time_limit=300`、`num_cpus=1`（Windows 兼容）
- 用 `holdout_frac=0.2` 代替 K 折（AutoGluon 内置 CV 策略）

**产出物**：`yonod_yield/models/autogluon_model.py`。

**验收标准**：500 条数据 5 分钟内出结果，无 multiprocessing 相关报错。

**PowerShell 命令**：

```powershell
$root = "C:\Users\joyjo\Desktop\其他大学资料\大创\YONOD"
conda activate yonod
# 先安装 AutoGluon（首次较慢，需联网）
pip install autogluon.tabular==1.1.1

# 验证（用小数据集冒烟测试）
python -c "
import sys, pandas as pd
sys.path.insert(0, r'$root')
from yonod_yield.features.reaction_featurizer import ReactionFeaturizer
from yonod_yield.descriptors.morgan import MorganDescriptor
from yonod_yield.models.autogluon_model import AutoGluonYieldModel
df = pd.read_csv(r'$root\数据集\酰胺缩合反应数据集.csv', nrows=300)
X, mask = ReactionFeaturizer(MorganDescriptor()).transform(df)
y = df['yield'].values[mask]
print(AutoGluonYieldModel(time_limit=120).fit_evaluate(X, y))
"
```

---

#### T2.5 实现统一评估模块

**操作要点**：
- 函数 `evaluate_cv(X, y, model, cv=5)` → 返回 R²/RMSE/MAE 的均值和标准差
- 使用 `sklearn.model_selection.KFold`，每折 fit 后 predict，汇总

**产出物**：`yonod_yield/evaluate.py`。

**验收标准**：传入任意 (X, y, model) 能返回完整 metrics dict，含 `r2_mean`、`r2_std`、`rmse_mean`、`mae_mean`、`train_time_s`。

**PowerShell 命令**：

```powershell
$root = "C:\Users\joyjo\Desktop\其他大学资料\大创\YONOD"
conda activate yonod
python -m py_compile "$root\yonod_yield\evaluate.py"; if ($?) { Write-Host "语法检查通过" }
```

---

### 阶段 3：其余描述符接入（第 3-4 周）

**目标**：将描述符从 1 个（Morgan）扩展到 4 个（加 ATMOMACCS / FISD / MolMetaLM），使结果矩阵达到 4 描述符 × 4 模型 = 16 单元。

> **执行顺序调整（2026-05-15）**：原顺序按字母 A→B→C 推进；实际执行改为 **C→A→B**（MolMetaLM 优先 → ATMOMACCS → FISD）。理由：MolMetaLM 权重已就绪、用标准 HuggingFace 接口、零代码阅读成本，可最快拿到第二行结果矩阵；ATMOMACCS / FISD 需要阅读外部仓库代码后再接入。任务 ID 不变，仅 §五 子节顺序调整。

---

#### T3.C1 接入 MolMetaLM 描述符（**v0.3.1 优先执行**）

**操作要点**：
- 权重路径：`C:\Users\joyjo\Desktop\其他大学资料\大创\YONOD\WEIGHTS\MolMetaLM-base`
- 加载 `AutoTokenizer` + `AutoModel`
- 对每个 SMILES：tokenize → forward → `last_hidden_state` 按 `attention_mask` 做加权 mean-pool → 768 维向量
- `torch.cuda.is_available()` 优先 GPU；batch_size=32，OOM 时减半重试
- 懒加载：模块 import 时不加载权重，首次 `featurize` 调用才加载

**产出物**：`yonod_yield/descriptors/molmetalm.py`。

**验收标准**：单条 SMILES 输出 shape=(768,)；100 条数据在 GPU 上 < 60 秒。

**PowerShell 命令**：

```powershell
$root = "C:\Users\joyjo\Desktop\其他大学资料\大创\YONOD"
conda activate yonod
python -c "
import sys, torch
sys.path.insert(0, r'$root')
from yonod_yield.descriptors.molmetalm import MolMetaLMDescriptor
import pandas as pd
df = pd.read_csv(r'$root\数据集\酰胺缩合反应数据集.csv', nrows=100)
desc = MolMetaLMDescriptor(
    weight_path=r'$root\WEIGHTS\MolMetaLM-base',
    device='cuda' if torch.cuda.is_available() else 'cpu'
)
feats, mask = desc.featurize(df['sub_1_smiles'].tolist())
print('shape:', feats.shape, 'device:', 'GPU' if torch.cuda.is_available() else 'CPU')
print('mask sum:', mask.sum())
"
```

---

#### T3.A1 确认 ATMOMACCS 入口函数

**操作要点**：阅读 `化学描述符相关项目/ATMOMACCS/` 下的源代码，定位生成 MACCS（167 维）的入口函数路径与调用方式。

**产出物**：读码笔记（记录在 Q&A 章节）。

**验收标准**：写出调用伪代码：`from ATMOMACCS.xxx import yyy; yyy(smiles) -> ndarray`。

**PowerShell 命令**：

```powershell
# 查看 ATMOMACCS 目录结构
Get-ChildItem -Recurse "C:\Users\joyjo\Desktop\其他大学资料\大创\YONOD\化学描述符相关项目\ATMOMACCS" -Filter "*.py" | Select-Object FullName
```

---

#### T3.A2 实现 ATMOMACCS 描述符适配器

**操作要点**：
- 通过 `sys.path.insert` 导入 ATMOMACCS 源码
- 默认输出 167 维 MACCS 向量
- 失败样本进 mask

**产出物**：`yonod_yield/descriptors/atmomaccs.py`。

**验收标准**：对数据集前 100 条 sub_1_smiles 输出 `(合法数, 167)` 数组。

**PowerShell 命令**：

```powershell
$root = "C:\Users\joyjo\Desktop\其他大学资料\大创\YONOD"
conda activate yonod
python -c "
import sys
sys.path.insert(0, r'$root')
import pandas as pd
from yonod_yield.descriptors.atmomaccs import ATMOMACCSDescriptor
df = pd.read_csv(r'$root\数据集\酰胺缩合反应数据集.csv', nrows=100)
feats, mask = ATMOMACCSDescriptor().featurize(df['sub_1_smiles'].tolist())
print('shape:', feats.shape, 'success:', mask.sum())
"
```

---

#### T3.B1 接入 FISD 描述符

**操作要点**：
- 阅读 `化学描述符相关项目/FISD/code/` 中的推理代码，定位 3 个 `.pth` 文件路径
- 实现 `yonod_yield/descriptors/fisd.py`：SMILES → Graph → MLMS → 50 维向量
- 路径通过参数传入，不硬编码

**产出物**：`yonod_yield/descriptors/fisd.py`。

**验收标准**：100 条 SMILES 输出 `(合法数, 50)` 数组，批推理 < 60 秒（CPU）。

**PowerShell 命令**：

```powershell
# 先查看 FISD 目录结构
Get-ChildItem -Recurse "C:\Users\joyjo\Desktop\其他大学资料\大创\YONOD\化学描述符相关项目\FISD" -Filter "*.py" | Select-Object FullName
Get-ChildItem -Recurse "C:\Users\joyjo\Desktop\其他大学资料\大创\YONOD\化学描述符相关项目\FISD" -Filter "*.pth" | Select-Object FullName
```

---


---

### 阶段 4：主脚本与结果输出（第 5 周）

**目标**：实现 `run_yield_prediction.py`，将前三阶段的所有组件串联为一次可执行的端到端流程。

---

#### T4.1 实现散点图生成模块

**操作要点**：
- 用 matplotlib 画"预测值 vs 真实值"散点图
- 标注 R²、RMSE
- 保存为 `results/scatter_{desc_name}_{model_name}.png`

**产出物**：`yonod_yield/plot.py`。

**验收标准**：给定 y_true 和 y_pred，能生成 PNG，坐标轴有标签，对角线参考线清晰。

**PowerShell 命令**：

```powershell
$root = "C:\Users\joyjo\Desktop\其他大学资料\大创\YONOD"
conda activate yonod
python -c "
import sys, numpy as np
sys.path.insert(0, r'$root')
from yonod_yield.plot import plot_scatter
np.random.seed(0)
y_true = np.random.rand(200)
y_pred = y_true + np.random.randn(200) * 0.1
plot_scatter(y_true, y_pred, desc='Morgan', model='XGB', save_dir=r'$root\results')
print('散点图已保存至 results/')
"
```

---

#### T4.2 实现主脚本 run_yield_prediction.py

**操作要点**：
- 接受命令行参数：`--csv`（数据集路径，必选）、`--desc`（描述符名称列表，默认 all）、`--model`（模型名称列表，默认 all）、`--nrows`（读取行数，默认全量，调试时用 500）、`--cv`（折数，默认 5）
- 双循环：for each desc × for each model → 构造特征 → CV 评估 → 保存散点图
- 最终将所有 metrics 汇总为 `results/metrics_summary.csv`
- 打印格式化表格到控制台

**产出物**：`run_yield_prediction.py`。

**验收标准**：以下命令能在 30 分钟内（全描述符）或 5 分钟内（仅 Morgan）完成并输出结果。

**PowerShell 命令**（完整运行）：

```powershell
$root = "C:\Users\joyjo\Desktop\其他大学资料\大创\YONOD"
conda activate yonod

# 快速冒烟测试（仅 Morgan × XGB，500 行数据）
python "$root\run_yield_prediction.py" `
    --csv "$root\数据集\酰胺缩合反应数据集.csv" `
    --desc morgan `
    --model xgb `
    --nrows 500 `
    --cv 5

# 完整运行（全量数据，所有描述符和模型）
python "$root\run_yield_prediction.py" `
    --csv "$root\数据集\酰胺缩合反应数据集.csv" `
    --desc morgan atmomaccs fisd molmetalm `
    --model xgb rf svm autogluon `
    --cv 5
```

---

#### T4.3 验收：检查输出文件

**操作要点**：确认 `results/` 目录下有散点图 PNG 和 `metrics_summary.csv`；打开 CSV 检查指标是否合理（R² 在 0 ~ 1 范围，RMSE 在 0 ~ 0.5 范围）。

**PowerShell 命令**：

```powershell
$root = "C:\Users\joyjo\Desktop\其他大学资料\大创\YONOD"
Get-ChildItem "$root\results" | Select-Object Name, Length, LastWriteTime
# 查看 metrics 内容
conda activate yonod
python -c "
import pandas as pd
df = pd.read_csv(r'$root\results\metrics_summary.csv')
print(df.to_string())
"
```

---

#### T4.4 全量 4×4 grid 分两阶段执行（v0.3.2 新增，2026-05-15）

**背景**：T4.2 单条命令一次跑完 4×4=16 组合在本地 Win11 上耗时过长，主要瓶颈是 Random Forest 在 Morgan/MolMetaLM 高维特征上的训练时间（详见 §九 Q13 实测：单折 7.6 分钟，5 折 ≈ 38 分钟，全部 RF 4 组合合计 ~77 分钟）。Win11 4 核 + 16 GB RAM 单机跑全量预计 2~4 小时，且任一中间环节崩溃都要从头重跑。

**v0.3.2 应对方案：分两阶段 + `--append` 合并 + 文件日志**：
1. **阶段 A**：先跑除 RF 之外的 12 组（XGB / SVM / AutoGluon × 4 描述符），约 1~2 小时；
2. **阶段 B**：单独跑 4 个 RF 组合（用 `--append` 合并入同一 `metrics_summary.csv`），约 1~1.5 小时；
3. 每阶段独立的 `results/run_<时间戳>.log` 文件落盘，便于事后排查；
4. 所有长时任务有 30~60 秒心跳打印，避免误判"卡死"。

**新增的主脚本 CLI 参数**：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--log-file PATH` | `results/run_<ts>.log` | 镜像 stdout 到日志文件，每行带 HH:MM:SS 时间戳 |
| `--heartbeat SEC` | 30 | 长任务每 SEC 秒打印一次"still running, elapsed Xs"；0 关闭 |
| `--rf-verbose 0/1/2` | 0 | RF 子任务的 sklearn verbose 等级（1 适合长跑，2 刷屏） |
| `--rf-n-jobs N` | -1 | RF 并行 worker 数；遇 joblib 卡死可改 1 |
| `--rf-n-estimators N` | 300 | RF 树数；100 可以 3× 加速，R² 损失通常 ≤ 3 个点 |
| `--rf-max-depth N` | None | RF 最大深度；30 可在高维上半量化加速 |
| `--append` | False | 不覆盖已有 `metrics_summary.csv`，按 (desc, model) 去重后合并 |
| `--svm-subsample N` | 8000 | SVM 每折随机抽 N 行训练（RBF SVR 在 n>10k 时不可行）；传 0 关闭 |

**PowerShell 命令（v0.3.2 推荐 = 方案 C）**：

```powershell
$root = "C:\Users\joyjo\Desktop\其他大学资料\大创\YONOD"
conda activate yonod

# === 阶段 A：12 组（约 1~2 小时） ===
python "$root\run_yield_prediction.py" `
    --model xgb svm autogluon `
    --svm-subsample 8000 `
    --heartbeat 60 `
    --log-file "$root\results\run_no_rf.log"

# === 阶段 B：4 个 RF 组合（约 1~1.5 小时，可在 A 完成后单独启动） ===
python "$root\run_yield_prediction.py" `
    --model rf `
    --rf-verbose 1 `
    --heartbeat 60 `
    --append `
    --log-file "$root\results\run_rf_only.log"
```

**验收标准**：
- `results/metrics_summary.csv` 共 16 行（4 描述符 × 4 模型）
- `results/scatter_*.png` 共 16 张
- 两个独立日志文件落盘 `run_no_rf.log` / `run_rf_only.log`

---

#### T4.5 云端迁移准备（v0.3.2 新增，可选）

**动机**：本地全量耗时 2~4 小时，且 RF 高维任务对内存压力大；云端高配机（多核 + 大内存 + 快 SSD）单机跑可压缩到 30 分钟以内，且不占本地工作时间。

**迁移清单**（按上传体积排序）：

| 资产 | 路径 | 体积 | 是否必需 |
|---|---|---|---|
| 数据集 CSV | `数据集/酰胺缩合反应数据集.csv` | ~10 MB | 必需 |
| 主代码包 | `yonod_yield/` | < 1 MB | 必需 |
| 入口脚本 | `run_yield_prediction.py` + `verify_morgan_rf.py` + `test_run_yield.py` | < 50 KB | 必需 |
| MolMetaLM 权重 | `WEIGHTS/MolMetaLM-base/` | ~500 MB | 仅用 MolMetaLM 才需 |
| FISD 模型 | `化学描述符相关项目/FISD/model/` 中 3 个 `qm_9_*.pth` | ~100 MB | 仅用 FISD 才需 |
| ATMOMACCS 源码 | `化学描述符相关项目/ATMOMACCS/` | < 1 MB | 当前实现已不依赖（用 RDKit 直调），可不传 |
| 试剂描述符缓存 | `cache/reagent_feats_*.pkl` | < 5 MB | 可选；不传则首次运行重建 |

**云端环境搭建**（无离线 whl 时的标准流程）：

```bash
# 1. 创建 conda 环境
conda create -n yonod python=3.9 -y
conda activate yonod

# 2. 装 torch（云端有网，直接走官方 cu121 通道）
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 \
    --index-url https://download.pytorch.org/whl/cu121

# 3. 装其余依赖（与本地一致）
pip install rdkit==2023.9.5 transformers==4.40.0 networkx==3.2.1 \
    scikit-learn==1.4.0 xgboost==2.0.3 \
    pandas==2.0.3 numpy==1.26.4 matplotlib==3.8.4 \
    joblib==1.3.2 pyyaml==6.0.1 tqdm==4.66.2
pip install torch_geometric==2.5.3
pip install "autogluon.tabular[lightgbm,catboost]==1.1.1"

# 4. 验证 GPU 可用
python -c "import torch; print('CUDA:', torch.cuda.is_available(), 'devices:', torch.cuda.device_count())"
```

**云端运行（一次跑完 4×4，无需分阶段）**：

```bash
cd /path/to/YONOD
python run_yield_prediction.py --heartbeat 120 \
    --log-file results/run_cloud_full.log
```

云端高配机预计 30~60 分钟跑完全量 16 组合。

**结果回传**：只需打包 `results/` 目录（含 16 张 PNG + metrics_summary.csv + 2 个 log），通常 < 5 MB。

**关键提醒**：
- **路径硬编码风险**：本地 `verify_morgan_rf.py` 等脚本含 Windows 风格的硬编码 `C:\Users\joyjo\...`，云端 Linux 上跑前需要检查并改为相对路径或环境变量；
- **CSV 文件名含中文**：上传到 Linux 时文件名编码注意保 UTF-8；某些 SCP 客户端会乱码；
- **MolMetaLM 权重**：必传 `model.safetensors` + `tokenizer*` + `config.json`，README 之类可省；
- **FISD 三个 .pth**：路径写死在 `fisd.py` 的 `DEFAULT_MODEL_DIR`，云端目录布局保持一致即可，否则传 `model_dir=` 参数。

---

### 阶段 5：缓存优化与稳定性（第 6 周，可选）

**目标**：避免每次运行都重新计算描述符（MolMetaLM 47015 条约 20 分钟），加入 npz 缓存。

---

#### T5.1 描述符特征缓存

**操作要点**：
- 以 `hash(sorted(smiles_list)) + desc_name` 为 key，存 `cache/{key}.npz`
- 第二次运行同数据集 + 同描述符 → 直接加载，跳过计算
- 缓存文件加入 `.gitignore`

**产出物**：`yonod_yield/utils.py` 中的缓存函数，集成到各描述符适配器。

**验收标准**：第二次运行 Morgan 描述符时，控制台打印"命中缓存，跳过特征化"，时间 < 5 秒。

**PowerShell 命令**：

```powershell
# 查看缓存目录大小
$root = "C:\Users\joyjo\Desktop\其他大学资料\大创\YONOD"
Get-ChildItem "$root\cache" | Measure-Object -Property Length -Sum | Select-Object @{Name="总大小MB";Expression={[math]::Round($_.Sum/1MB,2)}}
```

---

## 六、关键设计决策记录

### 6.1 为什么不做 Streamlit web app？

用户确认最终交付物是可执行 Python 脚本，不需要 web UI。脚本形态：
- 更易于调试和修改；
- 适合大创项目汇报时演示（直接在终端运行）；
- 避免 Streamlit 对 Windows 多进程的额外限制。

### 6.2 为什么试剂采用 SMILES 描述符化而非 one-hot？

详见 §6.7 完整论述。核心理由：one-hot 丢失化学结构信息；试剂总数仅 62 种，缓存后几乎无额外计算开销；与底物统一表征方式，便于消融实验对比。

### 6.3 为什么 Python 3.9 而不是 3.11？

用户提供的离线 CUDA torch 轮子是 `cp39`（Python 3.9 专属），在其他版本 Python 上无法安装。3.9 对 rdkit、transformers、sklearn 的兼容性均无问题。

### 6.4 MolMetaLM 权重路径约定

权重统一存放于 `YONOD/WEIGHTS/MolMetaLM-base/`，代码中使用相对于项目根的路径或绝对路径（通过 `--weight-path` 参数传入），不硬编码。

### 6.5 为什么 HSPOC 暂缓？

HSPOC 是一个基于溶剂化自由能的描述符，计算流程依赖 COSMO-RS 软件（TURBOMOLE/ORCA），本项目暂无相关许可和计算资源，故 v0.3 不纳入主流程。

### 6.6 为什么 v0.3 暂缓 DFT？

原因与 HSPOC 性质相同——资源与数据双重缺失：
1. 本数据集 47015 条反应**没有配套 Gaussian log 文件**，DFT 描述符的输入材料不存在；
2. 即使有 SMILES，对 47015 个底物从头做量子化学计算（每个分子需数小时 CPU），总计算量极为庞大，完全不符合"演示脚本"的轻量交付定位；
3. DFT 描述符维度仅约 15，信息量有限，不如 Morgan/MolMetaLM 丰富，性价比不高。

结论：v0.3 从主流程完全剔除 DFT，描述符总数由 5 降为 **4**，结果矩阵由 20 变为 **16**。后续如有 Gaussian log 数据，可参考原 T2.D 接口规范重新接入。

### 6.7 为什么试剂列选择 SMILES 描述符化而非 one-hot？

v0.2 草案曾考虑 one-hot，v0.3 正式改为 SMILES 描述符化，原因如下：
1. **one-hot 丢失化学结构信息**：激活剂 R2（DIC）和 R9（DCC）结构相似，one-hot 编码它们为完全正交的向量，无法让模型学到"两者结构相近、效果相近"的归纳偏置；SMILES 描述符则能保留这一信息。
2. **试剂数量有限，描述符化几乎不增加计算开销**：全部 62 种试剂一次性缓存，后续每条反应只需字典查找（O(1)），边际计算成本为零。
3. **与底物表征一致，便于消融实验**：底物和试剂使用同一套描述符，消融时可以直接对比"有底物/无底物/有试剂/无试剂"对模型性能的贡献，实验设计更清晰。
4. **`(无)` 自然处理**：零向量是结构信息为"空"的自然表达，比 one-hot 中单独设一个"(无)"类别更具语义一致性。

---

## 七、里程碑总览

| 周次 | 里程碑 | 可演示物 |
|---|---|---|
| W1 前半 | 环境搭建 + 数据探索 + 试剂映射表解析 | torch CUDA 可用，数据集读取成功，试剂缓存字典构建完毕 |
| W1 后半 | Morgan × 6 分子拼接 → 反应特征 | shape=(N, 6144)，mask 无异常 |
| W2 | 4 种 ML 模型全接入 | Morgan × 4 模型的 R²/RMSE 评估表（4 行） |
| W3-4 | ATMOMACCS + FISD + MolMetaLM 接入 | 4 描述符 × 4 模型 = 16 个指标 |
| W5 | 主脚本 `run_yield_prediction.py` 完成 | 一条命令跑出全部 16 组结果 + 散点图 PNG |
| W6（可选） | 缓存优化 | 第二次运行秒级返回 |

---

## 八、任务总览

| 阶段 | 任务区间 | 数量 | 关键交付物 |
|---|---|---|---|
| 0 | T0.1 – T0.4 | 4 | 环境、目录、数据探索、试剂映射表 + 描述符缓存字典 |
| 1 | T1.1 – T1.4 | 4 | Morgan + 试剂 SMILES 描述符化 + 6 分子拼接特征组合器 |
| 2 | T2.1 – T2.5 | 5 | 4 种 ML 模型 + 评估模块 |
| 3 | T3.A1 – T3.C1 | 4 | 3 种额外描述符全接入（ATMOMACCS / FISD / MolMetaLM） |
| 4 | T4.1 – T4.5 | 5 | 散点图 + 主脚本 + 验收 + 分阶段执行 + 云端迁移 |
| 5（可选） | T5.1 | 1 | 描述符缓存 |
| **合计** | | **23** | `run_yield_prediction.py` 完整交付 + 云端就绪 |

---

## 九、Q&A 记录

### Q1（2026-05-14）：每个描述符张量维度不同怎么处理？
**答**：见 §2.2。接口层统一为 `featurize() -> (ndarray, mask)`，每个描述符走独立 pipeline（独立 scaler、独立超参），结果在评估表横向比较。维度差异在 ML 端不是问题，SVM 高维需 auto PCA 降本。

### Q2（2026-05-14）：5 个描述符都能仅凭 SMILES 跑通吗？
**答**（v0.3 修订）：Morgan / ATMOMACCS / FISD / MolMetaLM 均可仅凭 SMILES 跑通。DFT 需 Gaussian log，本数据集无法使用（v0.3 标 N/A）。

### Q3（2026-05-14）：MolMetaLM 权重去哪下载？
**答**：权重已存在于 `YONOD/WEIGHTS/MolMetaLM-base/`，无需下载。如需重新下载，见 §3.5。

### Q4（2026-05-14）：为什么新增 Morgan 1024 维指纹？
**答**：业界基线必备 + 零依赖 + 与其他描述符互补，让结果矩阵更有信息量。

### Q5（2026-05-14）：项目方向最终确定为？
**答**：限定为**酰胺缩合反应产率预测（回归）**，数据集已有 47015 条反应数据，目标值为 yield（0~1）。

### Q6（2026-05-14）：Python 和 torch 版本为何从 3.11 + 2.3.1 改为 3.9 + 2.1.2？
**答**：用户提供的离线 CUDA torch 轮子为 `cp39`（Python 3.9 专属），强制使用 Python 3.9 和 torch 2.1.2+cu121。三个轮子文件位于 `H:\AI模型及torch包\torch_wheels\`。

### Q7（2026-05-14）：交付物为何从 Streamlit 改为可执行脚本？
**答**：用户确认本轮只需 API 调用演示，不需要 web UI。可执行脚本更适合调试和大创汇报展示。

### Q8（2026-05-14）：目录结构问题？
**答**：v0.2 中错误写成了根目录为 `描述符建模/`，实际项目根是 `YONOD/`。MolMetaLM 权重已下载至 `YONOD/WEIGHTS/MolMetaLM-base/`，v0.3 已全部修正。

### Q9（2026-05-14）：试剂列（activation_id 等）如何处理？不用 one-hot 吗？
**答**：v0.3 正式采用 **SMILES 描述符化**方案，不再使用 one-hot。每条反应特征 = 6 个分子描述符向量的拼接（sub_1、sub_2、激活剂、添加剂、碱、溶剂），总维度 = 6 × d。`(无)` 填零向量，样本不丢弃。具体原理见 §6.7。映射表使用方式见 Q12（v0.3.1 修订）。

### Q10（2026-05-14）：为什么不做 DFT 描述符？
**答**：两个根本原因。第一，本数据集 47015 条反应**没有任何配套的 Gaussian log 文件**，DFT 描述符的原始输入材料根本不存在；第二，即使临时补算，对每个底物做 DFT 单点计算（B3LYP/6-31G* 级别）通常需要数小时，47015 条全部算完是巨大算力负担，不符合本项目"演示脚本"的轻量定位。与 HSPOC 处理方式一致，v0.3 完全移除 DFT，结果矩阵由 5×4=20 缩减为 **4×4=16**。详见 §6.6。

### Q11（2026-05-14）：为什么从 venv 改为 conda？
**答**：venv 是 Python 标准库自带、零额外安装、足够轻量；conda 的优势在化学/AI 项目场景下更明显：① RDKit 在 conda-forge 上的二进制最为稳定可靠（虽然 PyPI 的 `rdkit==2023.9.5` 也已可用）；② 后续若需要 cudatoolkit、openbabel、psi4 等非 Python 二进制依赖，conda 的安装体验明显优于 venv + pip；③ `conda env list` 在多项目场景下管理更直观。本项目 v0.3 正式约定环境名 `yonod`。注意：离线 torch whl 仍通过 pip 安装（conda 没有同版本通道），二者不冲突。

### Q12（2026-05-15）：还需要解析"试剂编号-SMILES对应表.md"吗？
**答**：**不需要**。T0.3 数据探索阶段实测确认，CSV 中的 `activation_id` / `additive_id` / `base_id` / `solvent_id` 4 列已经直接存储 SMILES 字符串（例如 `activation_id = "CC(C)N=C=NC(C)C"` 是 DIC 的 SMILES，`additive_id = "C1=CC2=C(N=C1)N(N=N2)O"` 是 HOAt 的 SMILES），CSV 在分发前已完成 id → SMILES 映射。"试剂编号-SMILES对应表.md" 仅作人类可读的索引参考，**程序运行时不需要读取**。

**两个新增预处理细节**：
1. **逗号分隔的多片段 SMILES**：部分试剂为盐型/复合物，CSV 中用 `,` 连接多段（如 EDC·HCl 写作 `CCN=C=NCCCN(C)C,Cl`）。RDKit 标准的多片段分隔符是 `.`，因此预处理时统一将 `,` 替换为 `.` 后再调用 `MolFromSmiles`。
2. **`(无)` 值**：仍按原方案填零向量、不丢弃样本。

**影响范围**：T0.4 由"解析映射表"改为"从 CSV 抽唯一试剂 SMILES 并预计算描述符"；T1.3、T1.4 接口去除 `reagent_map_path` 参数；`features/reagent_smiles_map.py` 重命名为 `reagent_cache.py`。

### Q14（2026-05-15）：Morgan × SVM 跑了 75 分钟还没出结果，怎么办？
**答**：**RBF SVR 在 n > 10000 时复杂度爆炸，是算法天花板，不是 bug**。

| 检查项 | 数值 |
|---|---|
| 训练样本（每折） | 37612 |
| 特征维度（PCA 后） | 256 |
| 单折复杂度 | O(37612² × 256) ≈ 3.6 × 10¹¹ ops |
| 实测单折时间 | > 75 分钟（4441s 仍未完成） |
| 5 折预估 | 6~10 小时 |
| 4×4 grid 中 4 个 SVM 组合预估 | 24~40 小时 |

**修复**：v0.3.2 给 `SVMYieldModel.cross_validate` 加 `subsample_n` 参数。每折在拟合前从 train fold 随机抽样 N 行（默认 8000），test fold 不动。这样：
- SVM 单折训练时间从 75+ 分钟降至 1~3 分钟（n=8000 时 O 复杂度下降 ~22×）
- 测试集仍是真实的 ~9400 行，R²/RMSE 的统计意义不变
- 主脚本默认 `--svm-subsample 8000`，传 0 关闭

**实践依据**：scikit-learn 官方文档明确说明 SVR/SVC 复杂度在 n > 10k 时不实用；推荐方案就是子采样或换 LinearSVR / Nystroem 近似。本项目选最简单的子采样（保留 RBF 非线性核），R² 损失通常 < 5 个百分点。

**复跑命令**（阶段 A 用新版主脚本）：

```powershell
python "$root\run_yield_prediction.py" `
    --model xgb svm autogluon `
    --svm-subsample 8000 `
    --heartbeat 60 `
    --log-file "$root\results\run_no_rf.log"
```

---

### Q13（2026-05-15）：T4.2 一次跑完 4×4 时为什么 Morgan + RF 看起来卡死了？
**答**：**没卡死，只是慢**。`verify_morgan_rf.py` 全量诊断结果：

| 检查项 | 数值 |
|---|---|
| 数据形状 | (47015, 6144)，1.15 GB float32 |
| RF 配置 | n_estimators=300, max_depth=None, n_jobs=-1（16 线程） |
| 单折 fit 耗时 | **458 秒（7.6 分钟）** |
| 单折 predict 耗时 | 0.6 秒 |
| **5 折 CV 全量** | **≈ 38 分钟** |
| 单折 R² | 0.854（80/20 holdout） |

**根因**：RF 树构建复杂度 O(n_trees × n_samples × log n × √n_features)，6144 维 + 37k 训练样本 + 300 树 + 16 线程，单折 7.6 分钟是合理预期，不是死锁。之前 RF 默认 `verbose=0` 完全静默才让人误判。

**v0.3.2 应对**：
1. 给 RF 适配器加 `verbose` 参数（详见 §五 T4.4）；
2. 主脚本加心跳线程，每 30~60 秒打印一次 "still running, elapsed Xs"；
3. 主脚本加文件日志，所有进度落盘 `results/run_<时间戳>.log` 便于事后查；
4. 全量 4×4 改为分两阶段（方案 C），用 `--append` 合并 CSV；
5. 新增 `verify_morgan_rf.py` 作为长期诊断工具，未来任何 RF 慢/卡都用它复现。

**意外正面发现**：500 行 baseline R²=0.375，全量 47015 行 RF 单折 R²=0.854 —— 全量数据上模型能力被严重低估。后续所有 16 组合的全量结果可能都比 500 行的预估高 30~50 个百分点。

---

## 十、下一步行动建议

**进度（2026-05-15 晚）**：阶段 0~3 全部完成；阶段 4 的 T4.1（散点图）+ T4.2（主脚本，含心跳/日志/--append 等 7 个新增 CLI 参数）+ T4.3（小数据验收）已完成；T4.4（全量分阶段执行规范）+ T4.5（云端迁移准备）已规划入文档。

**4 描述符 × XGB 在 500 行的 baseline**（已实测）：

| desc | R² | feature_dim | device | 备注 |
|---|---|---|---|---|
| FISD | **0.417** | 300 | cuda | 5 折最稳（std=0.053） |
| Morgan | 0.375 | 6144 | cuda | — |
| ATMOMACCS | 0.363 | 996 | cuda | 训练最快 7s |
| MolMetaLM | 0.308 | 4608 | cuda | 嵌入需要更多数据 |

**全量 47015 行单折 RF 结果**（验证规模效应）：Morgan R²=0.854，比 500 行高 +47 个点。

1. **本地下一步**（方案 C，分两阶段）：
   - 阶段 A：`python run_yield_prediction.py --model xgb svm autogluon --heartbeat 60 --log-file results/run_no_rf.log`（约 1~2 小时）
   - 阶段 B：`python run_yield_prediction.py --model rf --rf-verbose 1 --heartbeat 60 --append --log-file results/run_rf_only.log`（约 1~1.5 小时）
2. **云端迁移**（推荐）：参考 §五 T4.5 的迁移清单上传，云端单条命令一次跑完，预计 30~60 分钟。
3. **结果交付**：`results/metrics_summary.csv`（16 行）+ 16 张散点图 + 2 个 log。
4. **后续可选**：T5.1 描述符特征缓存（避免每次跑都重新算 MolMetaLM 47015 行 forward）；超参数网格搜索；消融实验（移除某一类分子的特征看 R² 变化）。

---

## 十一、新数据集调研记录

> 调研日期：2026-05-25
> 数据集来源：`数据集/Enantioselective-Cross-Coupling-Prediction/`
> 论文标题：AI-Driven Development of Nickel-Catalyzed Enantioselective Cross-Coupling Reactions

---

### A. 这个数据集是做什么的？

**研究问题**：预测镍催化不对称交叉偶联反应（Ni-catalyzed Enantioselective Cross-Coupling）的**对映选择性**，核心指标为产物的对映体过量值（ee, enantiomeric excess）及其对应的自由能差（ΔΔG‡）。

**预测目标**：
- 主要目标：`∆∆G (Kcal/mol)`，即两个对映体过渡态的自由能差（单位 kcal/mol），连续值回归任务
- 辅助目标：`ee (%)`，与 ΔΔG 有严格的热力学换算关系（`ee = tanh(ΔΔG / (2RT)) × 100%`），二者互为单调变换，不是独立目标
- 预测性质：对映选择性（手性选择），**不是产率**

**数据规模**：
- 原始数据集（Raw_Dataset）：**6590 条**反应记录，14 列
- 特征数据集（Data_AutoGluon_DFT）：6590 行 × 94 列（93 个 DFT 描述符特征 + 温度 + 1 个目标列）
- 特征数据集（Data_AutoGluon_RDKIT）：6590 行 × 241 列（239 个 RDKit 描述符特征 + 温度 + 1 个目标列）

**反应体系组成**：
- 催化剂配体（Ligand）：380 种不同配体，提供 `Ligand_SMILES` 列
- 产物（Product）：3385 种不同底物组合，提供 `Product_SMILES` 列
- 反应类型（R_Type）：9 类，包括 NiH、C(sp3)-C(sp2)、C(sp3)-C(sp3)、3-Component 等
- 数据来源（Data_Type）：高通量实验（HTE）+ 文献
- 温度（Temp, K）：作为数值特征列入模型

**特征描述符类型**：
- DFT 描述符（93 维）：量子化学计算得到的几何特征（体积 Volume、表面积、半径 Radius）、电子结构特征（Mulliken 电荷、硬度 Hardness、亲核/亲电指数、偶极矩、HOMO/LUMO 能级、能隙等），**不包含 SMILES**
- RDKit 描述符（239 维）：RDKit 计算的物理化学描述符（logP、ASA、TPSA、分子量、旋转键数、氢键受/供体数、MQN 等），**不包含 SMILES**
- 注意：特征均为**配体和产物的描述符**，已预先计算并展平为表格，DFT 和 RDKit 版本均不再保留原始 SMILES 列

**是否有 SMILES**：Raw_Dataset.csv 有 `Ligand_SMILES` 和 `Product_SMILES`；预计算特征文件（DFT/RDKIT 版本）已不保留 SMILES。

---

### B. 与 YONOD 现有项目的异同

**相似之处**：

| 维度 | YONOD（酰胺缩合） | ECC（不对称交叉偶联） |
|---|---|---|
| 任务类型 | 回归 | 回归 |
| 反应类别 | 有机合成反应 | 有机合成反应 |
| 预测目标数值范围 | yield 0~1 | ΔΔG 0~4.8 kcal/mol；ee -100~0% |
| ML 框架 | AutoGluon 等 | AutoGluon（原项目使用） |
| 数据集大小 | 47015 | 6590 |
| 分子信息输入 | SMILES → 描述符 | SMILES → 描述符（DFT/RDKit，已预计算） |

**本质差异**：

| 维度 | YONOD（酰胺缩合） | ECC（不对称交叉偶联） | 影响 |
|---|---|---|---|
| **预测目标的物理含义** | 反应**产率**（生成产物的量） | 反应**对映选择性**（手性偏好方向和强度） | 完全不同的化学问题，模型无法直接复用 |
| **特征维度** | 6 分子拼接，6144~4608 维 | 配体 + 产物的 DFT/RDKit 描述符，93~239 维 | 特征工程方案不同 |
| **描述符来源** | 实时从 SMILES 计算（无需 DFT） | DFT 描述符需要量子化学预算，成本极高 | DFT 版本在 YONOD 框架中无法直接生成 |
| **反应组分数量** | 6 个（sub_1、sub_2、激活剂、添加剂、碱、溶剂） | 2 个（配体、产物） | 反应级特征拼接方式不同 |
| **数据规模** | 47015（约 7× 大） | 6590 | 规模效应不同，模型选择策略有差异 |
| **催化机制** | 氨基酸/肽偶联（非金属催化） | Ni 催化不对称催化 | 特征重要性差异极大 |
| **数据集分割策略** | 随机 K 折 CV | 原项目使用 Kennard-Stone 算法（最大化训练集化学空间覆盖） | 评估协议不同 |

---

### C. 可行性评估：能否将其纳入 YONOD 项目？

**结论：不建议强行"纳入"，建议作为独立扩展专题（ECC-Track）并行开发。**

#### 技术层面：条件可行，但需大量适配

**可行的部分**：
1. ECC 的 Raw_Dataset 提供了 `Ligand_SMILES` 和 `Product_SMILES`，理论上可以用 YONOD 现有的描述符管线（Morgan / ATMOMACCS / FISD / MolMetaLM）对配体和产物进行描述符化，再接 AutoGluon 等 ML 模型，预测 ΔΔG 或 ee
2. 数据规模（6590）在 YONOD 框架下完全可承受
3. 分割策略可以沿用 K 折 CV，也可以尝试 Kennard-Stone（原项目使用）
4. 目标列 `∆∆G (Kcal/mol)` 是连续值，回归任务接口与 YONOD 完全兼容

**需要做的适配工作**：
1. **反应特征构造**：ECC 是 2 分子体系（配体 + 产物），YONOD 是 6 分子体系。`ReactionFeaturizer` 需要新增一个参数化的"分子列表"配置，使其不硬编码 6 列，而是接受任意数量的 SMILES 列名
2. **目标列名更改**：从 `yield` 改为 `∆∆G (Kcal/mol)`（列名含特殊字符，需处理）
3. **评估指标补充**：ΔΔG 预测任务通常还关注 MAE in kcal/mol 尺度，需确认 RMSE 和 R² 的物理意义是否合适；可额外换算出 ee 的 MAE 作为可解释指标
4. **数据规范化差异**：ECC 数据目标值范围约 0~5 kcal/mol（不是 0~1），需重新评估各模型的默认超参数是否仍适用
5. **DFT 特征选项**：若想复用原项目的 DFT 描述符（93 维），需要读取预计算的 `Data_AutoGluon_DFT.csv`，用 `Num` 列与 Raw_Dataset 的行号对应后拼接；这是额外的数据准备工作

**不可行 / 核心障碍**：
1. **DFT 描述符无法为新配体实时生成**：原项目 DFT 描述符需要量子化学软件（Gaussian/ORCA）预算，每个分子数小时计算，**无法纳入 YONOD 的"仅凭 SMILES 实时推理"主流程**。若要复现原论文的最优性能，必须使用预计算的 DFT 表格，将 YONOD 变成"离线表格模式"而非通用 SMILES 管线
2. **化学问题本质不同**：对映选择性预测的学术价值和物理意义与产率预测完全不同，不适合混在同一个评估表中比较
3. **没有溶剂、碱、添加剂列**：ECC 反应条件极为简化（只有配体+底物+温度），这与 YONOD 的 6 分子拼接设计不对应

#### 推荐方案

**方案一（推荐）：ECC 作为独立脚本 `run_ecc_prediction.py`**
- 复用 YONOD 的描述符适配器（Morgan / MolMetaLM 等）
- 新建一个针对 ECC 的入口脚本，读取 Raw_Dataset.csv，对 Ligand_SMILES + Product_SMILES 各自生成描述符，拼接后预测 ΔΔG
- 模型仍使用 XGBoost / RF / AutoGluon
- 评估时同时报告 ΔΔG 的 RMSE/R² 和换算出的 ee MAE
- 这样 YONOD 项目增加了第二个反应体系，学术对比价值更高（两种催化反应、两种预测目标、相同描述符框架）

**方案二：直接复现原论文 AutoGluon + RDKit/DFT 描述符**
- 使用预计算的 `Data_AutoGluon_RDKIT.csv` 或 `Data_AutoGluon_DFT.csv`，以 `∆∆G` 为 label 直接跑 AutoGluon
- 代码极简（参考原项目 `Code/AutoGluon` 文件，约 30 行），无需任何 SMILES 描述符化
- 缺点：不利用 YONOD 的核心技术积累，学术创新性较低

**总结判断**：

| 评估维度 | 结论 |
|---|---|
| 数据集本身质量 | 高质量、结构清晰、有 SMILES + DFT + RDKit 三套特征 |
| 与 YONOD 现有代码的兼容性 | 中等（描述符管线可复用，反应特征构造需小改） |
| 是否适合合并进同一 `run_yield_prediction.py` | 不适合（目标含义不同，混淆评估表） |
| 是否值得做成扩展专题 | 强烈推荐（丰富大创项目内容，体现跨反应体系通用性） |
| 核心障碍 | DFT 描述符无法实时计算；化学问题与产率预测无直接可比性 |

---

## 十二、ECC 不对称偶联专题（ECC-Track）扩展计划

> 本章节依据 §十一 的调研结论，给出将 ECC 数据集以"独立脚本"方式纳入 YONOD 项目的完整实施计划。
> 技术路线与酰胺缩合专题**完全相同**：SMILES 输入 → 描述符特征化 → ML 模型 → 回归评估。

---

### 12.1 专题目标

以 Ni 催化不对称交叉偶联反应的 `Ligand_SMILES` + `Product_SMILES` + `Temp (K)` 为输入，通过 YONOD 现有的 4 类描述符管线（Morgan / ATMOMACCS / FISD / MolMetaLM）结合 4 种 ML 模型（XGBoost / RF / SVM / AutoGluon），对反应的对映选择性自由能差 `ΔΔG (kcal/mol)` 进行回归预测，并额外换算 `ee MAE` 作为可解释评估指标。

---

### 12.2 关键设计决策

#### 12.2.1 反应特征如何构建？

ECC 数据集有 2 个 SMILES 列（`Ligand_SMILES`、`Product_SMILES`）和 1 个标量特征（`Temp (K)`）。构建反应特征的方案如下：

```
reaction_feature = concat([
    desc(Ligand_SMILES),    # shape (d,)
    desc(Product_SMILES),   # shape (d,)
    [Temp_normalized],      # shape (1,)  -- 温度标准化后拼接
])
# 总维度 = 2 × d + 1
```

**原理**：两个 SMILES 各自独立描述符化后拼接，与酰胺缩合专题的"6 分子拼接"思路完全一致，只是从 6 分子简化为 2 分子。温度作为标量附加在向量末尾，而非嵌入到分子描述符中，因为温度是反应条件而非分子性质，二者物理含义不同，分开处理更合理。

**温度归一化**：用训练集的 mean/std 做 z-score 标准化（避免量纲不一致干扰树模型以外的模型），归一化参数在 train fold 上拟合，在 test fold 上 transform，防止数据泄露。

#### 12.2.2 现有 `ReactionFeaturizer` 是否需要改动？

**不需要改动 `ReactionFeaturizer`**，采用更简洁的方案：

当前 `run_yield_prediction.py`（v1.1+）已经支持"2列CSV通用模式"——column 0 = SMILES, column 1 = float label。ECC-Track 的入口脚本 `run_ecc_prediction.py` 只需在调用现有管线前做一步数据预处理：将 `Ligand_SMILES` 和 `Product_SMILES` 拼接为一个"反应 SMILES"列，用 `>>` 分隔（或直接用 `.` 连接后整体描述符化），然后以标准 2 列格式传入。

更好的方案是**新建专用数据加载器** `load_ecc_dataset()`，返回拼接后的特征矩阵（含 Temp 维度），跳过 `load_dataset()` 的 2 列限制。这样对 YONOD 核心代码零侵入。

#### 12.2.3 评估指标

主要指标：R²、RMSE（kcal/mol）、MAE（kcal/mol），与酰胺缩合专题一致。

额外指标：ee MAE（%）。换算公式为：

```
ee = tanh(ΔΔG / (2 × R × T)) × 100%
```

其中 R = 0.001987 kcal/(mol·K)，T 取每个样本的实验温度。预测 ee 和真实 ee 的 MAE 即为 ee MAE。这是原论文的核心评估指标，报告此指标有助于与原文直接对比。

#### 12.2.4 训练/测试划分策略

| 方案 | 说明 | 建议 |
|---|---|---|
| 随机 K 折 CV（k=5） | 与酰胺缩合专题一致，代码零改动 | **推荐（首选）** |
| Kennard-Stone 划分 | 最大化训练集化学空间覆盖，原论文使用此方案 | 可选，作为对比实验 |
| 按反应类型分层 CV | 保证每折含所有 9 类反应类型 | 可选，适合学术发表 |

首选**随机 5 折 CV**，理由是与 YONOD 主专题保持方法论一致，便于横向对比；同时数据量 6590 在 5 折下每折验证集 ~1318 条，统计意义充分。

---

### 12.3 代码改动清单

| # | 操作 | 文件路径 | 改动内容 | 是否影响现有代码 |
|---|---|---|---|---|
| 1 | 新建 | `数据集/Enantioselective-Cross-Coupling-Prediction/Data/csv/ecc_smiles_label.csv` | 预处理脚本生成的 2 列 CSV（`reaction_smiles`, `ddG`），供快速验证 | 否 |
| 2 | 新建 | `yonod_yield/features/ecc_dataset.py` | ECC 专用数据加载器 `load_ecc_dataset()`，返回 `(X, y, feature_names)` | 否 |
| 3 | 新建 | `run_ecc_prediction.py` | ECC 专题入口脚本，复用 `DESCRIPTOR_REGISTRY`、`MODEL_REGISTRY`、`evaluate_one`、`plot_scatter` | 否 |
| 4 | 新建 | `yonod_yield/metrics/ee_metrics.py` | `ddg_to_ee(ddg, T)` 换算函数 + `ee_mae(y_true, y_pred, T)` 指标计算 | 否 |
| 5 | 新建（可选） | `scripts/prepare_ecc_csv.py` | 一次性预处理脚本：读 Raw_Dataset.csv → 清洗 → 保存标准化 CSV | 否 |
| 6 | 修改（小改） | `yonod_yield/evaluate.py` | 在 `evaluate_one` 的返回 dict 中增加可选字段 `ee_mae`（当调用方传入 `compute_ee_mae=True` 和温度数组时才计算），否则默认 None，不影响现有调用 | 影响小，向后兼容 |

**总结**：主要新建 3~4 个文件，对现有代码只有 1 处小改（`evaluate.py` 增加可选字段），**不影响酰胺缩合专题的任何现有功能**。

---

### 12.4 实施步骤

---

#### ECC-T1 数据探索与预处理

**原理说明**：在建模前必须了解数据分布，特别是 ΔΔG 的值域和方向性（正/负代表不同手性偏好），以及 SMILES 的有效性。ECC 数据集的 ΔΔG 范围约为 -5 ~ +5 kcal/mol，正负号有物理含义（不能随意取绝对值）。

**具体操作**：

```powershell
$root = "C:\Users\joyjo\Desktop\其他大学资料\大创\YONOD"
conda activate yonod
python -c "
import pandas as pd, numpy as np
from rdkit import Chem

df = pd.read_csv(r'$root\数据集\Enantioselective-Cross-Coupling-Prediction\Data\csv\Raw_Dataset.csv')
print('形状:', df.shape)
print('列名:', df.columns.tolist())
print()
print('ΔΔG 统计:')
print(df['∆∆G (Kcal/mol)'].describe())
print()
print('温度分布 (K):')
print(df['Temp (K)'].value_counts().head(10))
print()
print('反应类型分布:')
print(df['R_Type'].value_counts())
print()
# SMILES 有效性检验
for col in ['Ligand_SMILES', 'Product_SMILES']:
    bad = [s for s in df[col] if Chem.MolFromSmiles(str(s)) is None]
    print(f'{col}: {len(bad)} 条无效 SMILES（共 {len(df)} 条）')
"
```

**验证方法**：
- ΔΔG 分布直方图显示双峰或近似正态，正负值均有；
- 无效 SMILES 数量为 0 或极少（< 5 条，可直接丢弃）；
- 温度集中在几个离散值（原文为 303.15 K / 253.15 K 等），确认后用于 ee 换算时的 T 值。

**常见问题**：
- `∆∆G (Kcal/mol)` 列名含特殊字符 `∆`，用 `df.columns` 查看精确列名后在代码中引用；
- 如果读取 CSV 时出现编码问题，添加 `encoding='utf-8-sig'` 参数。

---

#### ECC-T2 生成反应特征向量（复用 YONOD 描述符管线）

**原理说明**：对 `Ligand_SMILES` 和 `Product_SMILES` 各自调用描述符模型，生成两个向量后与温度标量拼接。这与酰胺缩合专题的"6分子拼接"完全相同，只是分子数量从 6 变为 2。温度作为第 3 个特征维度（1维），其标准化参数在每个 CV 折的训练集上拟合，防止数据泄露。

**具体操作（`yonod_yield/features/ecc_dataset.py` 核心逻辑）**：

```python
"""ECC 专题数据加载器（新建文件）"""
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from ..descriptors.base import BaseDescriptor

ECC_CSV = (
    Path(__file__).resolve().parents[2]
    / "数据集/Enantioselective-Cross-Coupling-Prediction/Data/csv/Raw_Dataset.csv"
)
DDG_COL = "∆∆G (Kcal/mol)"
TEMP_COL = "Temp (K)"

def load_ecc_dataset(csv_path=None):
    """读取 ECC Raw_Dataset.csv，返回 (ligand_smiles, product_smiles, temp_K, y)。"""
    csv_path = Path(csv_path) if csv_path else ECC_CSV
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    # 丢弃无效 SMILES 行
    from rdkit import Chem
    valid_mask = (
        df["Ligand_SMILES"].apply(lambda s: Chem.MolFromSmiles(str(s)) is not None)
        & df["Product_SMILES"].apply(lambda s: Chem.MolFromSmiles(str(s)) is not None)
        & df[DDG_COL].notna()
    )
    df = df[valid_mask].reset_index(drop=True)
    return (
        df["Ligand_SMILES"].tolist(),
        df["Product_SMILES"].tolist(),
        df[TEMP_COL].to_numpy(dtype=np.float64),
        df[DDG_COL].to_numpy(dtype=np.float64),
    )

def build_ecc_features(descriptor: BaseDescriptor,
                        ligand_smiles, product_smiles, temp_K,
                        temp_scaler=None, fit_scaler=True):
    """
    为 ECC 数据集生成反应级特征矩阵。

    参数：
        descriptor: YONOD 描述符对象（与主专题完全相同的 4 类之一）
        temp_scaler: sklearn StandardScaler；None 时新建；fit_scaler=True 时在当前数据上 fit
    返回：
        X: ndarray (n_valid, 2*d + 1)
        mask: bool ndarray (n_total,)，标记有效行
        temp_scaler: 已 fit 的 scaler（供 test fold 复用）
    """
    feats_lig, mask_lig = descriptor.featurize(ligand_smiles)
    feats_prod, mask_prod = descriptor.featurize(product_smiles)
    mask = mask_lig & mask_prod
    feats_lig = feats_lig[mask]
    feats_prod = feats_prod[mask]
    temp_valid = temp_K[mask].reshape(-1, 1)
    if temp_scaler is None:
        temp_scaler = StandardScaler()
    if fit_scaler:
        temp_valid = temp_scaler.fit_transform(temp_valid)
    else:
        temp_valid = temp_scaler.transform(temp_valid)
    X = np.concatenate([feats_lig, feats_prod, temp_valid], axis=1)
    return X, mask, temp_scaler
```

**验证方法**：

```powershell
$root = "C:\Users\joyjo\Desktop\其他大学资料\大创\YONOD"
conda activate yonod
python -c "
import sys
sys.path.insert(0, r'$root')
from yonod_yield.features.ecc_dataset import load_ecc_dataset, build_ecc_features
from yonod_yield.descriptors.morgan import MorganDescriptor

ligand_sm, product_sm, temp_K, y = load_ecc_dataset()
print(f'有效样本数: {len(y)}  ΔΔG 范围: [{y.min():.3f}, {y.max():.3f}]')

desc = MorganDescriptor()
X, mask, scaler = build_ecc_features(desc, ligand_sm, product_sm, temp_K)
print(f'特征矩阵 shape: {X.shape}  (预期: (n, 2*1024+1) = (n, 2049))')
print(f'最后一列（温度归一化）mean={X[:,-1].mean():.3f} std={X[:,-1].std():.3f}（预期 ~0 和 ~1）')
"
```

**常见问题**：
- 若描述符化时遇到 MolFromSmiles 返回 None，检查是否有含立体化学标注 `@` 的 SMILES 需要 sanitize；
- 温度列若含多个不同值（303.15 / 253.15 等），归一化后会正常分布，不是 bug。

---

#### ECC-T3 复用 YONOD 模型训练/评估循环

**原理说明**：`run_ecc_prediction.py` 直接调用 `yonod_yield.evaluate` 中的 `MODEL_REGISTRY` 和 `evaluate_one` 框架。与酰胺缩合专题的区别只有两点：输入矩阵形状不同（2d+1 而非 6d），以及指标中额外计算 ee MAE。其余 CV 循环、RF/SVM/XGB/AutoGluon 适配逻辑完全复用，零改动。

**具体操作（`run_ecc_prediction.py` 骨架）**：

```python
"""ECC 专题入口脚本：以 SMILES 为输入预测 ΔΔG。

用法：
  # 快速冒烟测试（仅 Morgan × XGB）：
  python run_ecc_prediction.py --desc morgan --model xgb --cv 5

  # 完整 4×4 grid：
  python run_ecc_prediction.py

输出：
  results/Raw_Dataset建模报告/metrics_summary_ecc.csv
  results/Raw_Dataset建模报告/scatter_<desc>_<model>_ecc.png
"""
import argparse, sys, time
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from yonod_yield.evaluate import DESCRIPTOR_REGISTRY, MODEL_REGISTRY, build_model, build_descriptor
from yonod_yield.features.ecc_dataset import load_ecc_dataset, build_ecc_features
from yonod_yield.metrics.ee_metrics import ee_mae as calc_ee_mae
from yonod_yield.plot import plot_scatter
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

ROOT = Path(__file__).resolve().parent
ECC_CSV = ROOT / "数据集/Enantioselective-Cross-Coupling-Prediction/Data/csv/Raw_Dataset.csv"
RESULTS_DIR = ROOT / "results" / "Raw_Dataset建模报告"

def evaluate_ecc_one(desc_name, model_name, ligand_sm, product_sm, temp_K, y, cv=5):
    """单个 (desc, model) 组合的 K 折 CV 评估，返回 metrics dict。"""
    desc = build_descriptor(desc_name)
    model = build_model(model_name)
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)
    r2s, rmses, maes, ee_maes = [], [], [], []
    oof_pred = np.full(len(y), np.nan)

    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(np.arange(len(y)))):
        # 训练集：fit 描述符 + scaler
        X_train, mask_train, scaler = build_ecc_features(
            desc,
            [ligand_sm[i] for i in train_idx],
            [product_sm[i] for i in train_idx],
            temp_K[train_idx], fit_scaler=True
        )
        y_train = y[train_idx][mask_train]

        # 测试集：用训练集 scaler transform
        X_test, mask_test, _ = build_ecc_features(
            desc,
            [ligand_sm[i] for i in test_idx],
            [product_sm[i] for i in test_idx],
            temp_K[test_idx], temp_scaler=scaler, fit_scaler=False
        )
        y_test = y[test_idx][mask_test]
        temp_test = temp_K[test_idx][mask_test]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        r2s.append(r2_score(y_test, y_pred))
        rmses.append(mean_squared_error(y_test, y_pred) ** 0.5)
        maes.append(mean_absolute_error(y_test, y_pred))
        ee_maes.append(calc_ee_mae(y_test, y_pred, temp_test))

        # 记录 OOF 预测（供散点图）
        valid_test_global = test_idx[mask_test]
        oof_pred[valid_test_global] = y_pred

    return {
        "descriptor": desc_name, "model": model_name,
        "r2_mean": np.mean(r2s), "r2_std": np.std(r2s),
        "rmse_mean": np.mean(rmses), "mae_mean": np.mean(maes),
        "ee_mae_mean": np.mean(ee_maes),
        "feature_dim": X_train.shape[1],
        "n_samples": len(y_train),
        "oof_pred": oof_pred,
    }

if __name__ == "__main__":
    # 解析参数、循环 desc×model、保存结果（参考 run_yield_prediction.py 结构）
    pass  # 完整实现见 ECC-T5 验收
```

**验证方法**：

```powershell
$root = "C:\Users\joyjo\Desktop\其他大学资料\大创\YONOD"
conda activate yonod
# 快速冒烟：Morgan × XGB，5 折，全量数据
python "$root\run_ecc_prediction.py" --desc morgan --model xgb --cv 5
# 预期输出：R² > 0.3，RMSE < 0.8 kcal/mol（保守估计；ECC 数据集较小但特征性强）
```

---

#### ECC-T4 结果分析与对比

**原理说明**：将 YONOD 的 SMILES 描述符（Morgan / MolMetaLM 等）结果与原论文使用的 DFT 描述符结果对比，分析差距来源，撰写大创报告的"局限性与展望"部分。

**原论文基准性能**（来自 Raw_Dataset 同配置数据）：

| 描述符类型 | 模型 | R²（原文报告）| 备注 |
|---|---|---|---|
| DFT（93维，需量子化学计算） | AutoGluon | ~0.85~0.92 | 原论文最优结果 |
| RDKit 物理化学描述符（239维） | AutoGluon | ~0.75~0.82 | 原论文次优结果 |
| YONOD SMILES 描述符（本专题目标） | XGB/RF 等 | 待测 | 预期 0.50~0.75 |

**差距来源分析框架**（供报告写作参考）：
1. **DFT 编码了量子化学信息**：轨道能级、电荷分布、HOMO/LUMO 是对映选择性的直接决定因素，而 Morgan 指纹/MolMetaLM 嵌入是拓扑/统计信息，间接编码化学效应；
2. **配体的 3D 构象**：对映选择性强依赖催化剂配体的空间手性环境，2D 拓扑描述符（Morgan/MACCS）对此编码能力弱；MolMetaLM 在大量 SMILES 上预训练，可能隐式学到部分立体化学偏好；
3. **数据集规模**（6590 条）对深度描述符（MolMetaLM 768 维）不够充足：高维嵌入在小数据集上泛化能力受限，可能不如低维的 Morgan 或 ATMOMACCS；
4. **温度效应**：温度仅 1 维，但对 ΔΔG 的影响显著（直接出现在 ee 换算公式中），低维标量编码已足够。

**验证方法**：完成 4×4 结果矩阵后，将 RMSE、R²、ee MAE 填入对比表，与上表对比，并在报告中定量说明差距。

---

### 12.5 预期困难与应对

| 困难 | 严重程度 | 具体表现 | 应对策略 |
|---|---|---|---|
| **数据集偏小（6590 条）** | 高 | RF/MolMetaLM 等高容量模型在训练集 ~5272 条（5折）时可能过拟合；train R² >> test R² | 监控 train/test R² 差值，差 > 0.2 时启用正则化（RF 减少 n_estimators 或加 max_depth 限制；XGB 降低 learning_rate 加 min_child_weight） |
| **特征维度远超样本量（MolMetaLM 768×2+1=1537 >> 5272/6 个不重复配体）** | 中 | 维度诅咒在小数据上加重；SVM PCA 后效果可能更差 | 增加 PCA 到 64~128 维的实验版本对比 |
| **ΔΔG 分布正负均有** | 低 | 不能直接归一化为 0~1；散点图参考线不再是 y=x 而是穿越原点的线 | `plot_scatter` 的 x_label/y_label 正确标注单位（kcal/mol），坐标轴对称于 0 |
| **列名含特殊字符 `∆`** | 低 | pandas 读 CSV 后用 `df['∆∆G (Kcal/mol)']` 可能报 KeyError | 用 `df.columns.tolist()` 打印后 copy 精确列名，或用 `df.iloc[:, 4]` 按位置取 |
| **6590 条 MolMetaLM 推理内存** | 低 | 6590 × 2 分子 × 768 维 ≈ 76 MB float32，在 GPU 上无压力 | batch_size=32 默认即可，OOM 时减半 |
| **ee 换算公式的温度依赖** | 低 | 不同样本温度不同，不能用固定 T | `ee_metrics.py` 中按行逐一换算，不要用均值温度 |

**过拟合风险重点说明**：ECC 数据集共 6590 条、380 种配体、3385 种底物。由于同一配体在多个底物上重复出现，随机 K 折时训练集和测试集会共享相同配体（只是底物不同），导致模型可能记住"这个配体通常给高 ee"而非真正泛化。更严格的评估应按配体做 leave-one-ligand-out CV（LOLO-CV）。建议首先用随机 K 折快速得到结果，然后视时间决定是否追加 LOLO-CV 作为对比。

---

### 12.6 学术价值说明

本扩展对大创项目具有如下具体学术价值：

1. **跨反应体系通用性验证**：
   YONOD 项目的核心主张是"SMILES 描述符管线对有机合成反应预测具有通用性"。酰胺缩合（酰胺键形成）和 Ni 催化不对称偶联是化学上**完全不同的两类反应**，分别代表热力学驱动（产率）和动力学-手性驱动（对映选择性）两种预测任务。若相同的描述符管线在两类反应上均有效，则显著加强了"通用性"论点的说服力。

2. **预测目标的扩展**：
   从"产率（0~1）"到"ΔΔG（kcal/mol）"，展示了 YONOD 框架不局限于某一种物理量，而是可适配多种连续回归目标。这是从"工具复现"到"框架设计"的学术层次提升。

3. **可与原论文直接对比**：
   ECC 数据集的原论文（"AI-Driven Development of Nickel-Catalyzed Enantioselective Cross-Coupling Reactions"）使用了 DFT 描述符和 AutoGluon，并公开了数据。本专题使用相同数据但替换为仅 SMILES 的描述符，可以直接报告"DFT 描述符 vs SMILES 描述符"的性能差距，为"无 DFT 的轻量化预测方案"提供实验依据——这本身就是一个有价值的研究贡献点。

4. **大创汇报材料充实**：
   两个数据集的结果矩阵（2 × 4 × 4 = 32 格）比单数据集（16 格）内容更丰富，展示了项目的"系统性"和"扩展性"，更容易获得答辩评委的认可。

---

### 12.7 工作量估算

| 任务 | 新增代码行数（估计）| 复杂度 |
|---|---|---|
| `yonod_yield/features/ecc_dataset.py` | ~80 行 | 低（参考 dataset.py 结构） |
| `yonod_yield/metrics/ee_metrics.py` | ~30 行 | 低（纯数学换算） |
| `run_ecc_prediction.py` | ~150 行 | 低-中（参考 run_yield_prediction.py 骨架） |
| `yonod_yield/evaluate.py` 小改 | ~10 行 | 极低（增加可选 ee_mae 字段） |
| 数据预处理脚本（可选） | ~30 行 | 低 |
| **合计** | **~300 行** | **整体复杂度：低** |

预计实施时间：**半天到 1 天**（熟悉 dataset.py / evaluate.py 结构后照搬骨架，主要工作是 ECC 数据预处理和验证）。

---

### 12.8 推荐执行顺序

```
ECC-T1（数据探索，30分钟）
    ↓
ECC-T2（ecc_dataset.py + 验证特征矩阵，2小时）
    ↓
ECC-T3（run_ecc_prediction.py 骨架 + 冒烟测试，2小时）
    ↓
ECC-T4（全量 4×4 grid 运行 + 与原论文对比，运行 1~2 小时）
    ↓
补充 ee_metrics.py（30分钟，可在 ECC-T2 后插入）
```

**快速验证命令（完成 ECC-T2 后即可运行）**：

```powershell
$root = "C:\Users\joyjo\Desktop\其他大学资料\大创\YONOD"
conda activate yonod

# 冒烟测试：Morgan × XGB，不超过 5 分钟
python "$root\run_ecc_prediction.py" --desc morgan --model xgb --cv 5

# 完整 4×4 grid（预计 30~60 分钟，ECC 数据集小）
python "$root\run_ecc_prediction.py" --heartbeat 30
```

---

## 第十三节：通用化重构方案分析

> 记录日期：2026-05-25
> 背景：Track A（酰胺缩合产率预测）和 Track B（Ni 催化对映选择性偶联 ΔΔG 预测）均已跑通。用户提出 5 条通用化重构方案，本节对每条方案进行逐条评价，并给出综合建议。

---

### 13.1 方案逐条评价

---

#### 方案 1：数据集格式限定为 CSV + 仅接受 SMILES 列，排除温度/压力等数值变量

**优点**

- 格式约束清晰，降低了数据接入门槛。研究者只需提供一个普通 CSV，不需要了解任何配置文件结构。
- "仅 SMILES"的限制使特征工程管线（描述符化 → 拼接）保持高度统一，不需要为每种数值变量设计不同的归一化策略。
- 对于绝大多数分子性质预测任务（产率、溶解度、logP 等），输入确实只需要 SMILES，该限制不构成障碍。

**潜在问题**

这是 5 条方案中**问题最突出**的一条，核心矛盾在于：

1. **Track B（ECC 专题）直接违背此限制**。ECC 的温度（`Temp (K)`）不是噪音列，而是 ΔΔG 预测中物理上不可或缺的特征——ΔΔG 与对映体过量 ee 的换算公式 `ee = tanh(ΔΔG / 2RT) × 100%` 明确依赖温度 T。温度不同时，同一底物-配体组合的 ee 值会显著不同。如果强行丢弃温度列，Track B 的预测精度会受到实质性影响，且这种影响无法通过换更好的描述符来弥补。

2. **数值型反应条件在有机合成中极为普遍**。除温度外，压力、pH、反应时间、催化剂用量（mol%）等数值条件在许多高通量实验（HTE）数据集中都是重要特征。强行排除会使"通用平台"在实际应用中很快碰壁。

3. **混合输入（SMILES + 数值）在技术上并不困难**。数值列只需在特征矩阵末尾拼接并做 z-score 标准化即可，Track B 的现有 `build_ecc_features()` 实现已经验证了这一做法的可行性。

**改进建议**

将方案 1 改为：**CSV 格式，最后一列为数值标签，其余列分为两类：SMILES 列和数值列，通过列头类型自动识别或用户显式指定**。具体实现选项如下：

- **自动识别**：尝试用 RDKit 解析，能解析成 Mol 对象的列视为 SMILES 列，否则尝试转为 float，成功则视为数值列，其余报错。
- **用户指定**：增加可选 CLI 参数 `--smiles-cols col1 col2 ...` 和 `--numeric-cols temp pressure`，用户在歧义情况下显式声明。

这样改动之后，Track A（6 个 SMILES 列）和 Track B（2 个 SMILES 列 + 1 个数值列）均可无缝接入，同时新数据集也不需要单独写脚本。

---

#### 方案 2：CSV 最后一列必须是数值标签，前面所有列必须是 SMILES 字符串

**优点**

- "最后一列为标签"的约定简单直观，零配置。
- 前置全为 SMILES 的假设适合绝大多数分子性质数据集（如 MoleculeNet 系列的 SMILES + 单一 target 格式）。
- 前置合法性验证（列类型检查）可以在管线入口拦截绝大多数格式错误，避免错误在后期描述符化阶段以更晦涩的方式暴发。

**潜在问题**

1. **"最后一列为标签"的约定比"用户指定标签列名"更脆弱**。真实的 CSV 数据集往往还有 `smiles`、`row_id`、`source`、`experiment_date` 等非 SMILES 的元数据列，它们排列顺序不固定。如果用户的 CSV 习惯把 SMILES 放在列尾、标签放在中间，就会被拒绝，体验很差。
2. **"前面所有列必须是 SMILES"的验证逻辑需要容忍率**。现实数据中常见 `NaN`、`invalid_smiles`、`""` 等脏数据。如果验证逻辑要求 100% 合法 SMILES 才能通过，会在数据探索阶段就把用户挡在门外，而实际上这些脏行完全可以在描述符化时通过 `mask` 机制丢弃。
3. **对含元数据列的数据集完全不兼容**（如 `[id, smiles_1, smiles_2, temperature, yield]` 这种常见格式）。

**改进建议**

将约定从"位置固定"改为"名称指定 + 位置兜底"：

- 用户可通过 `--label-col yield`（列名）或 `--label-col -1`（倒数第一列，兜底默认值）指定标签列。
- 用户可通过 `--smiles-cols sub_1 sub_2`（显式）或 `--auto-detect`（程序自动探测）指定 SMILES 列。
- 合法性验证改为"软验证"：打印警告并报告无效 SMILES 数量，但不抛出异常阻止运行，除非无效比例超过 50%（此时很可能是配置错误）。

这样既保留了零配置的简单用例（2 列 CSV，列 0 = SMILES，列 1 = label），也能兼容复杂格式的真实数据集。

---

#### 方案 3：特征构建——将每行所有 SMILES 列各自描述符化后拼接

**优点**

- 这是项目现有架构的直接延伸，逻辑清晰且在 Track A（6 分子拼接，6144 维）上已充分验证。
- 拼接策略对不同分子数量的反应体系（2 分子、3 分子、6 分子）天然兼容，无需修改核心代码。
- 每个分子独立描述符化 + 拼接的方式使得"缺失分子（`(无)` → 零向量）"的处理策略保持一致，不需要特殊逻辑。

**潜在问题**

1. **拼接顺序的语义问题**。对于 SMILES 列数量不固定的数据集，如果描述符化后直接按列顺序拼接，模型就隐含了"第 1 列的特征权重 ≠ 第 2 列的特征权重"这一假设。对于某些反应（如底物 1 和底物 2 可互换的对称反应），这一假设是错误的，会引入顺序偏差。
2. **高维问题随分子数量线性增长**。6 分子 × 1024 维 = 6144 维，对 SVM 等内核方法已经需要 PCA 降维。如果未来接入含 10 个组分的数据集，特征维度将达到 10240 维，训练成本急剧上升。
3. **没有利用分子间相互作用信息**。纯拼接方案假设底物和试剂之间的特征是线性叠加的，而真实的反应结果往往依赖分子间的相互作用（如底物-催化剂的空间匹配）。这是该方案的化学层面的局限，对于反应性预测任务尤为突出。不过这是所有"分子指纹 + 传统 ML"路线的共同局限，不是本次重构引入的新问题，可以在学术报告中作为"局限性"讨论，无需在工程层面解决。

**改进建议**

- 在文档中明确说明"按列顺序拼接"的约定，要求用户在 CSV 列顺序上保证语义一致（如"底物总在试剂前面"）。
- 为对称反应提供一个可选的"对称拼接"选项（将两个底物的描述符取均值或排序后拼接），但设为默认关闭，不强制。
- 可在主脚本加一个 `--max-feature-dim` 参数，当拼接后维度超过阈值时自动触发全局 PCA 降维（现有 SVM 的 auto-PCA 逻辑可复用到其他场景）。

---

#### 方案 4：交互式主脚本（运行后提示用户输入数据集路径、标签列名、任务名称）

**优点**

- 对于完全不熟悉命令行的初学者，交互式提示远比"背诵参数顺序"友好。
- 在演示场景（大创汇报、课堂展示）下，交互式引导看起来更"智能"，能给评委留下深刻印象。
- 强制用户确认标签列名和任务名称，可以有效防止"用错列"或"忘记指定"的常见错误。

**潜在问题**

1. **交互式和 CLI 参数驱动并不互斥，但各自有明确适用场景**。CLI 参数驱动（现有的 `argparse` 方案）有三大不可替代的优势：
   - **可脚本化**：可以写 `.ps1` / `.sh` 批处理文件，一键跑完全量 4×4；
   - **可重现**：命令行历史精确记录了每次运行的配置，便于复现实验；
   - **可远程执行**：SSH 远程登录云服务器或通过任务调度系统提交时，交互式输入完全不可用。
2. **交互式脚本在批量运行场景下会阻塞**。如果要对 3 个数据集各跑一次全量评估，交互式脚本需要手动输入 3 次，而 CLI 版本可以在 `.ps1` 文件里写好 3 条命令一次性提交。
3. **已有的 `argparse` 方案已具备 `--help` 文档**，可读性并不差。用户真正的困难不是"不知道有哪些参数"，而是"不知道 CSV 格式要怎么准备"——这是文档问题，不是 CLI vs 交互式的问题。

**改进建议**

不要在交互式和 CLI 之间二选一，而是**在 CLI 基础上增加一个"向导模式"**：

```
python run_yonod.py --wizard
```

当 `--wizard` 被指定时，脚本进入交互式引导，依次询问数据集路径、标签列、任务名称等，最终**生成并打印出等效的 CLI 命令**（而不是直接运行），让用户学会下次如何直接用 CLI 参数。这样既照顾了初学者，又不牺牲高级用户的批处理能力，还有教学效果。

---

#### 方案 5：可读性强的报告——4×4 结果表格 + 最佳组合推荐 + 化学家可理解的指标解释

**优点**

- 这是 5 条方案中**最无争议的一条**，方向完全正确。
- 当前 `generate_report.py` 已经有 HTML 报告生成逻辑，本方案是对其内容质量的提升，工程量可控。
- 在大创汇报和论文写作中，"让评委不需要懂机器学习也能看懂结果"是差异化竞争力。

**潜在问题**

1. **现有报告的指标解释可能仍过于技术化**。`R²_std = 0.032` 这样的数字对化学背景的读者几乎没有直觉意义。
2. **最佳组合推荐需要明确排名标准**。如果 A 组合 R² 最高但 RMSE 也最高，B 组合各指标均衡但没有任何单项最优，谁是"最佳"？需要一个透明的评分公式，而不是黑盒排名。
3. **Track A 和 Track B 的指标量纲不同**（yield 是无单位的 0~1 数值，ΔΔG 是 kcal/mol），单个报告模板无法直接复用于两个 Track，需要参数化。

**改进建议**

报告应包含以下内容，按化学家视角组织：

**指标的化学语言解释模板**（建议直接写入报告的文字说明区）：

| 指标 | 推荐解释语言 |
|---|---|
| R² | "模型解释了实验数据方差的 XX%。R²=0.85 意味着模型预测值与实验值之间的差异，有 85% 可以用输入的分子结构信息来解释。" |
| RMSE | "模型的典型误差（均方根误差）为 X 个单位。对于产率预测，RMSE=0.08 意味着预测产率与真实产率平均相差约 8 个百分点。" |
| MAE | "模型的平均绝对误差为 X 个单位，即大多数预测与真实值相差不超过这个数。" |
| R²_std（5折标准差） | "模型在 5 次独立验证中 R² 的波动幅度。std=0.03 说明模型表现稳定；std>0.1 说明模型对数据分割较敏感，可能存在过拟合风险。" |
| ee MAE（ECC Track 专属）| "模型预测的对映体过量值与实验值平均相差 X%。对于对映选择性 >90% ee 的反应，X<5% 通常被认为实用。" |

**最佳组合推荐的量化标准**（建议采用加权综合得分）：

```
综合得分 = 0.5 × (R²_mean 排名) + 0.3 × (RMSE 排名，越低越好) + 0.2 × (1 - R²_std 排名)
```

报告中应明确写出排名标准，而不仅仅给出推荐结论。

---

### 13.2 综合建议：通用化接口的推荐设计

---

#### 13.2.1 设计目标的双重约束

通用化重构面临两个不能回退的硬约束：

1. **Track A 零回退**：`run_yield_prediction.py` 当前的 CLI 接口、`--append` 分阶段执行、心跳日志等功能必须继续工作，不能因通用化改造而破坏。
2. **Track B 零回退**：`run_ecc_prediction.py` 的温度特征、ee MAE 指标必须继续工作。

同时，通用化的目标是：**新增一个第三个入口 `run_yonod.py`，它能接受任意符合规范的 CSV，自动推断或接受用户指定的列角色，走相同的描述符 + ML 评估管线，输出格式统一的报告。Track A 和 Track B 保持独立入口，通用入口是平行的第三条路，而非替代前两者**。

---

#### 13.2.2 推荐的 CSV 规范（替代方案 1 + 2 的修订版）

```
规范 v1：
- 文件格式：UTF-8 编码的 CSV，有表头行
- 列角色分类：
  (a) SMILES 列：内容为 SMILES 字符串，可有多列
  (b) 数值辅助列：内容为浮点数（如温度、压力），可有多列，也可没有
  (c) 标签列：预测目标，必须是浮点数，有且仅有一列
  (d) 忽略列：row_id、字符串类型的分类变量等，自动跳过
- 列角色声明方式：
  - 优先级 1：用户通过 --smiles-cols 和 --label-col 显式声明
  - 优先级 2：自动探测（SMILES 列：RDKit 解析成功率>50%；标签列：最后一列浮点数）
- 合法性验证：
  - 软验证：打印每列的 SMILES 有效率、NaN 比例，不阻断运行
  - 硬验证：标签列有效值数量 < 10 时报错退出（样本太少）
```

---

#### 13.2.3 推荐的架构方案

```
YONOD/
├── run_yield_prediction.py   # Track A 专用，保持不变
├── run_ecc_prediction.py     # Track B 专用，保持不变
├── run_yonod.py              # [新建] 通用入口，接受任意合规 CSV
└── yonod_yield/
    ├── universal/            # [新建子包] 通用化专属代码
    │   ├── csv_loader.py     # 通用 CSV 加载器（含列角色探测逻辑）
    │   ├── feature_builder.py  # 通用特征构建（SMILES 拼接 + 数值列归一化）
    │   └── report.py         # 通用报告生成（含化学语言指标解释）
    ├── descriptors/          # 保持不变（Track A/B/通用共享）
    ├── models/               # 保持不变
    └── evaluate.py           # 保持不变（通用评估循环可在此扩展）
```

核心原则：**通用化代码完全放在新子包 `yonod_yield/universal/` 中，对现有代码零侵入**。Track A 和 B 的现有脚本和子模块完全不修改。

---

#### 13.2.4 `run_yonod.py` 的接口设计

```
# 最简用法（2列 CSV，自动探测）：
python run_yonod.py --csv my_data.csv

# 多 SMILES 列 + 指定标签：
python run_yonod.py --csv ecc.csv --smiles-cols Ligand_SMILES Product_SMILES --label-col "∆∆G"

# 含数值辅助列：
python run_yonod.py --csv ecc.csv --smiles-cols Ligand_SMILES Product_SMILES --numeric-cols "Temp (K)" --label-col "∆∆G"

# 向导模式（交互式引导，适合演示和初学者）：
python run_yonod.py --wizard

# 指定任务名称（用于报告标题）：
python run_yonod.py --csv my_data.csv --task-name "溶解度预测"
```

---

#### 13.2.5 Track B 兼容性专项说明

Track B（ECC 数据集）的温度特征是本次讨论的核心争议点。推荐的处理方式是：

- 通用入口 `run_yonod.py` 通过 `--numeric-cols "Temp (K)"` 接受温度列，描述符化后将其 z-score 标准化，**在每个 CV 折的训练集上 fit 标准化参数，在测试集上 transform**（防止数据泄露），然后拼接到 SMILES 描述符特征矩阵末尾。
- 这与 Track B 现有的 `build_ecc_features()` 实现逻辑完全一致，已经在 ECC 数据集上验证可行。
- 不需要为温度设计任何特殊处理逻辑，它就是一个普通的数值辅助列。

---

#### 13.2.6 实施优先级建议

考虑到大创项目的时间窗口，通用化重构建议分三个优先级执行：

| 优先级 | 任务 | 工作量估计 | 价值 |
|---|---|---|---|
| P0（立即做）| 完善报告中的化学语言指标解释（方案 5）| 约 2 小时（仅改 HTML 模板文字）| 大创汇报直接受益，零风险 |
| P1（本周）| 新建 `csv_loader.py` + 最简版 `run_yonod.py`（支持任意 SMILES 列数 + 可选数值列）| 约 1 天 | 显著提升项目通用性 |
| P2（可选）| 增加 `--wizard` 交互向导模式 | 约 半天 | 演示价值高，但功能性无增益 |

Track A 和 B 的现有功能在整个重构过程中保持不变，P1 完成后可立即验证通用入口能否接受酰胺缩合 CSV（2 个 SMILES 列）和 ECC CSV（2 个 SMILES + 温度）。

---

### 13.3 小结

5 条方案的整体方向是正确的，主要问题集中在方案 1（温度限制过严，与 Track B 存在直接矛盾）和方案 2（标签列位置固定太脆弱）。修订后的建议是：**以"列角色声明"替代"列位置约定"，以"软验证"替代"硬拒绝"，以"CLI + 可选向导模式"替代"纯交互式"，以"通用入口并行"替代"合并改写现有脚本"**。如此既获得了通用性，又保护了已经验证可用的 Track A 和 Track B 代码资产。

---

## 十四、通用化重构——确认方案与实施计划

> 文档版本：v0.4（2026-05-25）
> 本节是第十三节讨论的最终确认版，包含用户逐条反馈后的技术收敛结论、伪代码规范、CLI 参数规范、BAT 交互流程以及文件变更清单。

---

### 14.1 五条方案最终确认状态

| # | 方案 | 状态 | 关键修订点 |
|---|---|---|---|
| 方案 1 | 数值辅助列归一化 | **确认，有重要细化** | 仅对数值辅助列做 z-score；SMILES 描述符矩阵原样保留；必须在每个 CV fold 内 fit scaler（见 §14.2） |
| 方案 2 | 列角色声明方式 | **确认，改为显式 CLI 优先** | `--label-col` 和 `--smiles-cols` 显式指定；不提供时走自动探测（见 §14.3） |
| 方案 3 | 多 SMILES 列拼接 | **无异议，直接确认** | 所有 SMILES 列独立描述符化后横向 hstack，与 Track A 逻辑一致 |
| 方案 4 | 入口与交互方式 | **确认，改为 run_yonod.py + yonod.bat** | 先建 `run_yonod.py`（CLI），再建 `yonod.bat`（交互式拼接命令）（见 §14.4） |
| 方案 5 | 报告格式 | **无异议，直接确认** | 4×4 结果表 + 化学语言指标解释 + 加权排名推荐，实现于 `report.py`（见 §14.5） |

---

### 14.2 数值列归一化：技术分析与确认方案

#### 14.2.1 当前做法的隐患定量分析

Track B 将温度（233～353 K）直接裸拼接在描述符后，未做任何归一化。以下是各描述符的量纲对比：

| 描述符 | 典型分量值域 | 代表均值 | 273K / 代表均值 |
|---|---|---|---|
| Morgan ECFP4 | {0, 1}，bit 激活率约 1%～5% | ≈ 0.03 | **≈ 9100 倍** |
| MACCS（ATMOMACCS） | {0, 1}，333 维 | ≈ 0.05 | **≈ 5460 倍** |
| FISD | L2 归一化，约 0.01～0.1 | ≈ 0.05 | **≈ 5460 倍** |
| MolMetaLM | L2 归一化，约 0.01～0.1 | ≈ 0.05 | **≈ 5460 倍** |

**实际危害**：
- 树模型（XGBoost、RF）做节点分裂时，温度这一维度的信息增益会完全压倒所有分子描述符维度。模型实际上只在"温度高低"上做判断，分子结构信息几乎无法被利用。
- SVM RBF 核对欧氏距离极度敏感。两个样本之间的距离由 `||x_i - x_j||²` 决定。若温度差为 20K，则 `(20)² = 400`，而 FISD 全维度距离估计仅约 `0.05² × 50 ≈ 0.125`。即 **温度单维度的距离贡献是整个分子描述符向量的 3200 倍**，核函数将由温度差单独决定。

#### 14.2.2 五种候选方案对比

| 方案 | 做法 | 对位指纹的影响 | 对 L2 归一化向量的影响 | 对温度的效果 | 结论 |
|---|---|---|---|---|---|
| A：仅温度 z-score | 对温度列单独标准化 | 无影响 | 无影响 | 消除量纲，均值 0 方差 1 | 正确，是方案 E 的核心部分 |
| B：温度 min-max [0,1] | 线性缩放到 0～1 | 无影响 | 无影响 | 与位指纹值域对齐 | 可用，但对训练集范围外的温度无法外推 |
| C：温度 min-max 到描述符值域 | 需先测量描述符值域 | 无影响 | 无影响 | 理论上最完美 | 实现繁琐，且描述符"值域"语义不清 |
| D：拼接后全局 z-score | 对整个特征向量做 z-score | **破坏稀疏性**：0 变成非零 | **破坏 L2 球面几何** | 消除量纲 | **不推荐**，副作用远大于收益 |
| E：自适应（位指纹不动，数值列 z-score） | 只对数值辅助列做 z-score | 完全不动 | 完全不动 | 消除量纲 | **推荐** |

**补充说明（FISD/MolMetaLM 为何不需要额外 z-score）**：这两类描述符已经过 L2 归一化（每个向量的模为 1），各分量的量级在 0.01～0.1 之间，相互之间已经对齐，不存在量纲不一致问题。如果强行对它们再做 z-score，会破坏向量在 L2 球面上的几何关系，可能损害 SVM RBF 核的相似度计算。

#### 14.2.3 确认方案：仅对数值辅助列做 z-score

**规则**：
1. SMILES 描述符矩阵（`X_smiles`，无论是 Morgan/MACCS/FISD/MolMetaLM）：原样保留，不做任何归一化。
2. 数值辅助列矩阵（`X_numeric`，如温度）：在每个 CV fold 的训练集上 fit `StandardScaler`，对训练集和测试集分别 transform。
3. 最终特征矩阵：`X = hstack([X_smiles, X_numeric_scaled])`，数值列在末尾。
4. 若没有数值辅助列（`--numeric-cols` 未指定），则 `X = X_smiles`，无需任何 scaler 逻辑。

#### 14.2.4 KFold CV 循环中的正确实现（防数据泄露伪代码）

```python
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import numpy as np

# 前置假设：
# X_smiles : np.ndarray, shape (n, d_desc)    -- 描述符矩阵，原样保留
# X_numeric: np.ndarray, shape (n, k) or None -- 数值辅助列（如温度），k >= 1
# y        : np.ndarray, shape (n,)           -- 标签列
# model    : sklearn-compatible estimator

kf = KFold(n_splits=5, shuffle=True, random_state=42)
metrics_all_folds = []

for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X_smiles)):

    # 1. 切分描述符矩阵
    X_smi_train = X_smiles[train_idx]   # shape (n_train, d_desc)
    X_smi_test  = X_smiles[test_idx]    # shape (n_test,  d_desc)
    y_train     = y[train_idx]
    y_test      = y[test_idx]

    # 2. 数值列：只在训练折上 fit，再分别 transform
    if X_numeric is not None and X_numeric.shape[1] > 0:
        scaler = StandardScaler()                                  # 每折新建，不复用
        X_num_train = scaler.fit_transform(X_numeric[train_idx])  # fit 只看训练折
        X_num_test  = scaler.transform(X_numeric[test_idx])       # 测试折只 transform
        X_train = np.hstack([X_smi_train, X_num_train])
        X_test  = np.hstack([X_smi_test,  X_num_test])
    else:
        X_train = X_smi_train
        X_test  = X_smi_test

    # 3. SVM 子采样（如果启用）：在归一化之后进行
    if model_name == "svm" and subsample_n is not None:
        rng = np.random.default_rng(seed=fold_idx)               # 折级别固定种子
        sub_idx = rng.choice(len(X_train), size=min(subsample_n, len(X_train)), replace=False)
        X_train_fit = X_train[sub_idx]
        y_train_fit = y_train[sub_idx]
    else:
        X_train_fit = X_train
        y_train_fit = y_train

    # 4. 训练和评估
    model.fit(X_train_fit, y_train_fit)
    y_pred = model.predict(X_test)
    metrics_all_folds.append(compute_metrics(y_test, y_pred))

# 注意事项：
# - scaler 在 enumerate 循环内部创建，每折独立，绝不能在循环外 fit
# - test_idx 对应的行从未参与任何 scaler.fit_transform() 调用
# - subsample_n 只影响 model.fit 的训练样本，不影响 scaler 的统计量和测试集评估
```

---

### 14.3 CLI 参数规范

#### 14.3.1 `run_yonod.py` 完整 CLI 参数列表

```
python run_yonod.py [OPTIONS]

必选参数：
  --csv <路径>                  输入数据集 CSV 文件路径

列角色声明（可选，不提供时走自动探测）：
  --label-col <列名>            标签列名（预测目标）
  --smiles-cols <列名> [...]    SMILES 列名，可指定多个，空格分隔
  --numeric-cols <列名> [...]   数值辅助列名，可指定多个，空格分隔（可不填）

任务控制：
  --task-name <名称>            任务名称，用于报告标题（默认：csv 文件名去后缀）
  --descriptors <名称> [...]    选用的描述符，可选 morgan maccs fisd molmetalm（默认：全选）
  --models <名称> [...]         选用的模型，可选 xgboost rf svm autogluon（默认：全选）
  --smiles-threshold <float>    自动探测 SMILES 列时的有效率阈值（默认：0.5）
  --output-dir <路径>           结果输出目录（默认：results/<task-name>/）

兼容 Track A 的参数（透传给内部评估循环）：
  --append                      追加写入 metrics_summary.csv（不覆盖已有结果）
  --log-file <路径>             日志文件路径
  --heartbeat <秒>              心跳打印间隔
  --svm-subsample <N>           SVM 训练子采样数量（默认：8000）
```

#### 14.3.2 不提供 `--smiles-cols` 时的自动探测逻辑

```python
def auto_detect_smiles_cols(df: pd.DataFrame, label_col: str, threshold: float = 0.5) -> list[str]:
    """
    对 df 中除 label_col 之外的每一列，随机抽样 50 行，
    用 RDKit 解析，有效率 > threshold 则认定为 SMILES 列。
    """
    from rdkit import Chem
    import random

    candidate_cols = [c for c in df.columns if c != label_col]
    smiles_cols = []

    for col in candidate_cols:
        non_null_vals = df[col].dropna().astype(str).tolist()
        if len(non_null_vals) == 0:
            continue
        sample = random.sample(non_null_vals, min(50, len(non_null_vals)))
        valid_count = sum(1 for s in sample if Chem.MolFromSmiles(s) is not None)
        valid_rate = valid_count / len(sample)
        if valid_rate > threshold:
            smiles_cols.append(col)
            print(f"[自动探测] 列 '{col}' 有效 SMILES 率 {valid_rate:.1%}，纳入 SMILES 列")
        else:
            print(f"[自动探测] 列 '{col}' 有效 SMILES 率 {valid_rate:.1%}，跳过")

    if not smiles_cols:
        raise ValueError(
            "自动探测未找到任何 SMILES 列（有效率均低于阈值）。"
            "请用 --smiles-cols 手动指定，或调低 --smiles-threshold。"
        )
    return smiles_cols
```

不提供 `--label-col` 时的自动推断逻辑：取 DataFrame 中最后一列且 `dtype` 为 float64/float32 的列，并打印警告"自动推断标签列为 [列名]，如有误请用 --label-col 显式指定"。若最后一列不是浮点型，报错退出。

---

### 14.4 BAT 交互流程与脚本

#### 14.4.1 可行性确认

方案完全可行：`.bat` 文件通过 `set /p` 收集用户输入，拼接为 `python run_yonod.py ...` 命令，再调用 `conda activate` 后执行。唯一需要注意的是 conda 的激活方式：在 bat 中必须用 `call conda activate` 而非直接 `conda activate`，否则 conda 激活只影响子进程而不影响当前 bat 进程。

#### 14.4.2 交互步骤说明

| 步骤 | 提示内容 | 备注 |
|---|---|---|
| 1 | 请输入数据集 CSV 路径（可直接拖拽文件到窗口） | 支持带空格路径，自动加引号 |
| 2 | 请输入标签列名（如 yield、ee、delta_G） | 必填 |
| 3 | 请输入 SMILES 列名，多列用空格分隔（留空则自动探测） | 可选 |
| 4 | 请输入数值辅助列名，多列用空格分隔（无则留空） | 可选，留空表示无数值列 |
| 5 | 请输入任务名称（用于报告标题，如 酰胺缩合） | 可选，留空使用 CSV 文件名 |
| 6 | 选择描述符：morgan maccs fisd molmetalm（多选空格分隔，留空=全选） | 可选 |
| 7 | 选择模型：xgboost rf svm autogluon（多选空格分隔，留空=全选） | 可选 |
| 8 | 打印将要执行的完整命令（供用户核对） | 执行前确认 |
| 9 | 执行命令，执行完毕后 pause | 等待用户看到结果后再关闭窗口 |

#### 14.4.3 `yonod.bat` 完整脚本

```bat
@echo off
chcp 65001 > nul
title YONOD 通用化入口

echo.
echo =====================================================
echo   YONOD - Your One-stop Notebook Of Descriptors
echo   通用化入口向导
echo =====================================================
echo.

:: 步骤 1：CSV 路径
set /p CSV_PATH="[1/7] 请输入数据集 CSV 路径 (可拖拽文件): "
:: 去掉拖拽时可能带入的首尾引号
set CSV_PATH=%CSV_PATH:"=%

:: 步骤 2：标签列名（必填）
set /p LABEL_COL="[2/7] 请输入标签列名 (如 yield、ee、delta_G): "

:: 步骤 3：SMILES 列名（可选）
set /p SMILES_COLS="[3/7] SMILES 列名，多列空格分隔 (留空=自动探测): "

:: 步骤 4：数值辅助列名（可选）
set /p NUMERIC_COLS="[4/7] 数值辅助列名，多列空格分隔 (无则留空，如温度): "

:: 步骤 5：任务名称（可选）
set /p TASK_NAME="[5/7] 任务名称，用于报告标题 (留空=CSV 文件名): "

:: 步骤 6：描述符选择（可选）
echo [6/7] 可用描述符: morgan  maccs  fisd  molmetalm
set /p DESCS="        多选空格分隔，留空=全选: "

:: 步骤 7：模型选择（可选）
echo [7/7] 可用模型: xgboost  rf  svm  autogluon
set /p MODELS="        多选空格分隔，留空=全选: "

:: 拼接命令
set CMD=python run_yonod.py --csv "%CSV_PATH%" --label-col "%LABEL_COL%"

if not "%SMILES_COLS%"=="" (
    set CMD=%CMD% --smiles-cols %SMILES_COLS%
)
if not "%NUMERIC_COLS%"=="" (
    set CMD=%CMD% --numeric-cols %NUMERIC_COLS%
)
if not "%TASK_NAME%"=="" (
    set CMD=%CMD% --task-name "%TASK_NAME%"
)
if not "%DESCS%"=="" (
    set CMD=%CMD% --descriptors %DESCS%
)
if not "%MODELS%"=="" (
    set CMD=%CMD% --models %MODELS%
)

echo.
echo =====================================================
echo   将要执行的命令：
echo   %CMD%
echo =====================================================
echo.
pause

:: 激活 conda 环境并执行
call conda activate yonod
if %ERRORLEVEL% NEQ 0 (
    echo [错误] conda activate yonod 失败，请确认环境名称正确。
    pause
    exit /b 1
)

%CMD%

echo.
echo [完成] 结果已保存，按任意键关闭窗口。
pause
```

**已知局限**：bat 的 `set /p` 对含有特殊字符（`&`、`|`、`>`、`<`）的列名处理不佳。如果列名包含这些字符，建议直接在命令行使用 `python run_yonod.py` 手动传参，不使用 bat 向导。

---

### 14.5 报告格式（方案 5 细化）

实现位置：`yonod_yield/universal/report.py`

#### 14.5.1 报告结构

```
1. 任务信息头：任务名称、CSV 路径、样本量、SMILES 列数、数值辅助列情况
2. 4×4 结果表格（描述符 × 模型，单元格显示 R²/RMSE/MAE）
3. 化学语言指标解释（固定文字段落）
4. 加权排名推荐（前三名组合 + 推荐理由）
5. 散点图 PNG 路径索引
```

#### 14.5.2 化学语言指标解释（固定文字，写入报告）

```
R²（决定系数）：取值 0～1，越接近 1 表示模型对产率变化的解释能力越强。
  - R² > 0.85：模型具有较强的预测可靠性，可用于辅助实验设计
  - R² 0.7～0.85：中等预测能力，趋势判断可参考，具体数值需谨慎
  - R² < 0.7：模型对该描述符/模型组合的拟合效果较弱

RMSE（均方根误差）：与产率的量纲相同（本项目 yield 为 0~1 浮点）。
  - RMSE < 0.05：平均预测误差约 5 个百分点，接近实验重复性误差范围
  - RMSE 0.05～0.10：中等误差，可区分高产率和低产率区间
  - RMSE > 0.10：误差较大，不建议用于定量预测

MAE（平均绝对误差）：比 RMSE 对异常值更鲁棒，反映典型单样本的预测偏差。
  综合排名权重建议：R²×0.5 + (1-RMSE/max_RMSE)×0.3 + (1-MAE/max_MAE)×0.2
```

#### 14.5.3 加权排名逻辑伪代码

```python
def rank_combinations(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """
    metrics_df 列：desc_name, model_name, r2, rmse, mae
    返回按加权分排序的结果，附推荐理由。
    """
    df = metrics_df.copy()
    max_rmse = df["rmse"].max()
    max_mae  = df["mae"].max()

    df["score"] = (
        df["r2"]                             * 0.5 +
        (1 - df["rmse"] / max_rmse)          * 0.3 +
        (1 - df["mae"]  / max_mae)           * 0.2
    )

    df = df.sort_values("score", ascending=False).reset_index(drop=True)
    df["rank"] = df.index + 1

    # 自动生成推荐理由（仅对前三名）
    for i in range(min(3, len(df))):
        row = df.iloc[i]
        reason_parts = []
        if row["r2"] > 0.85:
            reason_parts.append(f"R²={row['r2']:.3f} 解释能力强")
        if row["rmse"] < 0.05:
            reason_parts.append(f"RMSE={row['rmse']:.4f} 误差接近实验重复性")
        if not reason_parts:
            reason_parts.append(f"综合评分 {row['score']:.3f} 排名靠前")
        df.at[i, "reason"] = "；".join(reason_parts)

    return df[["rank", "desc_name", "model_name", "r2", "rmse", "mae", "score", "reason"]]
```

---

### 14.6 文件新增与修改清单

#### 14.6.1 新增文件

| 文件路径 | 用途 | 依赖 |
|---|---|---|
| `run_yonod.py` | 通用化 CLI 入口（主脚本） | `yonod_yield/universal/` |
| `yonod.bat` | Windows 交互式向导，双击执行 | `run_yonod.py`，conda 环境 `yonod` |
| `yonod_yield/universal/__init__.py` | 子包初始化 | 无 |
| `yonod_yield/universal/csv_loader.py` | CSV 加载 + 列角色探测 | pandas，rdkit |
| `yonod_yield/universal/feature_builder.py` | 通用特征构建（多 SMILES 拼接 + 数值列 scaler）| 现有描述符适配器 |
| `yonod_yield/universal/report.py` | 结果报告生成（4×4 表格 + 指标解释 + 排名） | pandas，matplotlib |

#### 14.6.2 现有文件（保持不变）

以下文件在本次重构中**零修改**，确保 Track A 和 Track B 现有功能不受影响：

| 文件 | 说明 |
|---|---|
| `run_yield_prediction.py` | Track A 专用入口，不动 |
| `run_ecc_prediction.py` | Track B 专用入口，不动 |
| `yonod_yield/features/dataset.py` | Track A 描述符化逻辑，不动 |
| `yonod_yield/features/ecc_dataset.py` | Track B 描述符化逻辑，不动 |
| `yonod_yield/models/` | 所有模型适配器，不动 |
| `yonod_yield/evaluate.py` | 评估循环，不动（通用入口可直接复用） |
| `yonod_yield/plot.py` | 绘图，不动 |

#### 14.6.3 实施顺序建议

```
第1步（约 2 小时）：新建 yonod_yield/universal/ 子包
  - __init__.py（空）
  - csv_loader.py：auto_detect_smiles_cols() + load_csv_with_roles()
  - 验证：单元测试 csv_loader 对酰胺缩合 CSV 和 ECC CSV 的列探测结果

第2步（约 3 小时）：新建 feature_builder.py
  - build_universal_features(smiles_cols, numeric_cols, df, desc_name) -> X_smiles, X_numeric
  - 内部复用现有 4 个描述符适配器
  - 验证：对两条 SMILES 手动计算，维度是否为 6×d

第3步（约 2 小时）：新建 run_yonod.py
  - 解析 CLI 参数（argparse）
  - 调用 csv_loader → feature_builder → evaluate（复用现有）→ report
  - 验证：python run_yonod.py --csv 数据集/酰胺缩合反应数据集.csv --label-col yield

第4步（约 1 小时）：新建 yonod.bat
  - 按 §14.4.3 模板实现
  - 验证：双击 bat，按提示输入酰胺缩合数据集的信息，能正常执行

第5步（约 2 小时）：新建 report.py
  - 实现 4×4 表格输出（HTML + CSV 两种格式）
  - 实现 rank_combinations() 排名函数
  - 验证：mock 一个 metrics_df 调用 report，检查输出文字是否包含正确的指标解释
```

---

### 14.7 关键技术决策汇总（供后续实现时参考）

| 决策点 | 确认结论 |
|---|---|
| 数值列归一化时机 | 在 KFold 循环内，`train_idx` 切分之后，`model.fit` 之前 |
| StandardScaler 实例复用 | 禁止复用，每个 fold 必须新建 `StandardScaler()` |
| SMILES 描述符是否归一化 | 否，Morgan/MACCS/FISD/MolMetaLM 原样保留 |
| FISD/MolMetaLM 是否额外 z-score | 否，L2 归一化已足够，强行 z-score 会破坏球面几何 |
| SVM 子采样与 scaler 的顺序 | 先 scaler.fit_transform，再 subsample（subsample 不影响统计量） |
| 标签列自动推断策略 | 最后一列浮点型；推断成功后打印警告，不静默执行 |
| SMILES 列自动探测阈值 | 默认 0.5，可通过 `--smiles-threshold` 调整 |
| bat 特殊字符局限 | 列名含 `&`/`|`/`>`/`<` 时不用 bat，直接命令行传参 |
| Track A/B 修改量 | 零，通用化代码完全在新子包 `yonod_yield/universal/` 中 |

---

## 十五、项目迁移指南

> 把 YONOD 从本地 Windows 11 迁到云端 Linux 服务器，并维护 GitHub 仓库。
> 写给：项目作者本人，假设了解 Python，对 Linux/Git 有基础认识。
> 适用版本：v1.2.0（2026-05-27）

本节分三部分：①云端硬件选型与传输；②Linux 适配要点；③GitHub 维护。

---

### 15.1 迁移到云端服务器

#### 15.1.1 为什么要迁

| 痛点 | 本地 (Win11, 4 核, 16GB RAM) | 云端 (推荐配置) |
|---|---|---|
| RF 在 Morgan/MolMetaLM 全量上 | 单折 7~8 分钟，5 折 38 分钟 | 16 核 + 32GB 单折预计 2~3 分钟 |
| 4×4 完整 grid 总耗时 | 2~4 小时 | 30~60 分钟 |
| 需要熬夜守着脚本 | 是 | 否（提交后断开 SSH 也行） |
| GPU 利用率 | MolMetaLM 阶段 80%、RF 阶段 0% | 同左（除非用 cuML） |

#### 15.1.2 推荐配置

| 资源 | 最低 | 推荐 | 说明 |
|---|---|---|---|
| **CPU** | 8 核 | **16~32 核** | RF 训练是 CPU 瓶颈，核越多 sklearn `n_jobs=-1` 越快 |
| **RAM** | 16 GB | **32 GB** | Morgan 全量 X ≈ 1.15 GB；RF 中间状态再吃几 GB；AutoGluon stack 时还会临时分配 |
| **GPU** | 6 GB 显存 | RTX 3060 / T4 / A10 (8~24 GB) | 仅 MolMetaLM forward 需要；不强求高端卡 |
| **磁盘** | 10 GB | 20 GB | 数据 + 权重 + 缓存 + 结果总计 < 2 GB；预留 swap 余量 |
| **OS** | Ubuntu 22.04 LTS | 同 | 与本项目 Python 3.9 + CUDA 12.1 兼容 |

#### 15.1.3 国内云服务商速查

| 平台 | 适配场景 | 备注 |
|---|---|---|
| **AutoDL** | 学生 / 短时任务首选 | 按小时计费、GPU 列表透明、网盘传文件方便、有 PyTorch 2.1+CUDA 12.1 现成镜像 |
| **阿里云 ECS GPU** | 长期/生产 | 配置灵活，但起步价比 AutoDL 高 |
| **腾讯云 GN7** | 同上 | 同上 |
| **Lambda Labs / Vast.ai** | 海外 | 国内访问慢，需翻墙；价格便宜 |

**对本项目**：AutoDL 一个 RTX 3090 + 16 核 + 32GB 实例（每小时 ~2 元）跑完一次完整 4×4 grid 约 1 小时，成本 2~3 元，性价比最高。

#### 15.1.4 需要上传的资产清单

| 资产 | 本地路径 | 体积 | 必需性 | 备注 |
|---|---|---|---|---|
| 数据集（完整） | `dataset/amide-coupling.csv` | ~10 MB | ✅ 必需 | UTF-8 编码；MIT 协议 |
| 数据集（样本） | `dataset/test-amide-coupling.csv` | < 1 KB | 可选 | 10 行，用于烟测 |
| 核心代码包 | `yonod/` | < 1 MB | ✅ 必需 | 含 4 个 descriptor + 4 个 model + universal + features |
| 统一入口脚本 | `yonod.py` | < 100 KB | ✅ 必需 | 交互向导 + CLI 合并入口 |
| MolMetaLM 权重 | `WEIGHTS/MolMetaLM-base/` | ~500 MB | ⚠️ 用 MolMetaLM 才需 | 可跳过该描述符则不传 |
| FISD 权重 | `WEIGHTS/FISD/` | ~19 MB | ⚠️ 用 FISD 才需 | 3 个 `.pth` 文件 |
| 试剂缓存 | `cache/reagent_feats_*.pkl` | < 5 MB | 可选 | 不传则首次运行重建（多 5~10 秒） |
| 第三方源码 | `化学描述符相关项目/` | 任意 | ❌ 不需要 | 当前实现已内联到 `yonod/`，无需上游源码 |

**总传输量**：~520 MB（含全部权重）或 ~10 MB（只跑 morgan/maccs）。

> **注意**：`dataset/amide-coupling.csv` 已随 GitHub 仓库提供（MIT 协议），
> 云端直接 `git clone` 即可获得，无需单独上传。

#### 15.1.5 上传方式

**方法 A：AutoDL 网盘上传（推荐）**

1. AutoDL 控制台 → 数据盘 → 上传文件（支持拖拽，最大 50GB）
2. 实例启动后，文件位于 `/root/autodl-tmp/`

**方法 B：scp 命令**

```bash
# 在本地 Windows 端打开 PowerShell（或 Git Bash）
scp -r "C:\Users\joyjo\Desktop\其他大学资料\大创\YONOD" root@云端IP:/root/
```

**方法 C：rsync（断点续传，大文件首选）**

```bash
# 仅在 Git Bash / WSL 下可用
rsync -avzP \
    --exclude '.git' \
    --exclude 'results' \
    --exclude 'cache' \
    --exclude '化学描述符相关项目' \
    "/c/Users/joyjo/Desktop/其他大学资料/大创/YONOD/" \
    root@云端IP:/root/YONOD/
```

**方法 D：先发 GitHub，云端 git clone**（适合代码部分；大权重仍要单独传）

```bash
git clone https://github.com/thinktraveller/YONOD.git
cd YONOD
# 然后单独 scp 上传 WEIGHTS/ 目录
```

#### 15.1.6 云端环境搭建

> 假定云端是 Ubuntu 22.04，已预装 conda 或 miniconda。AutoDL 镜像通常自带。

```bash
# 1) 创建 Python 3.9 环境
conda create -n yonod python=3.9 -y
conda activate yonod

# 2) 装 torch（云端有网，走官方 CUDA 12.1 通道，无需离线 whl）
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 \
    --index-url https://download.pytorch.org/whl/cu121

# 3) 验证 GPU
python -c "import torch; print('CUDA:', torch.cuda.is_available(), '| device:', torch.cuda.get_device_name(0))"
# 期望: CUDA: True | device: NVIDIA GeForce RTX 3090 (或同级)

# 4) 化学/ML 主依赖
pip install -r requirements.txt

# 5) AutoGluon（含 lightgbm/catboost；体积较大，耐心等待）
pip install "autogluon.tabular[lightgbm,catboost]==1.1.1"

# 6) 修复 molmetalm.py 的 Windows 默认路径（见 §15.2.1）
sed -i 's|DEFAULT_WEIGHT_PATH = .*|DEFAULT_WEIGHT_PATH = "/root/YONOD/WEIGHTS/MolMetaLM-base"|' \
    yonod/descriptors/molmetalm.py

# 7) 烟测（10 行，秒级完成）
cd /root/YONOD
python yonod.py \
    --csv dataset/test-amide-coupling.csv \
    --label-col yield \
    --smiles-cols sub_1_smiles sub_2_smiles activation_id additive_id base_id solvent_id \
    --descriptors morgan --models xgb --task-name smoke-test
```

#### 15.1.7 启动全量任务（推荐 tmux 防 SSH 断线）

```bash
# 进入 tmux 会话；断开 SSH 也不会停
tmux new -s yonod

# 在 tmux 里启动（向导模式，全量 4×4 grid）
cd /root/YONOD
conda activate yonod
python yonod.py \
    --csv dataset/amide-coupling.csv \
    --label-col yield \
    --smiles-cols sub_1_smiles sub_2_smiles activation_id additive_id base_id solvent_id \
    --task-name amide-full \
    --heartbeat 60

# 按 Ctrl+B 然后 D 脱离 tmux（保留运行）
# 重新进入: tmux attach -t yonod
```

完成后预计 30~60 分钟（视实例配置），`results/amide-full/` 下会有散点图、metrics_summary.csv、log、report.html。

跳过 MolMetaLM（无权重时）：

```bash
python yonod.py \
    --csv dataset/amide-coupling.csv \
    --label-col yield \
    --smiles-cols sub_1_smiles sub_2_smiles activation_id additive_id base_id solvent_id \
    --descriptors morgan maccs fisd \
    --task-name amide-no-llm \
    --heartbeat 60
```

#### 15.1.8 结果回传

```bash
# 云端打包
cd /root/YONOD/results/amide-full
tar czf yonod_results.tar.gz \
    metrics_summary.csv \
    report.html \
    pictures/ \
    *.log

# 本地下载（在 PowerShell 里）
scp root@云端IP:/root/YONOD/results/amide-full/yonod_results.tar.gz "C:\Users\joyjo\Desktop\"
```

或在 AutoDL 上直接 web 下载。

#### 15.1.9 进阶：GPU 加速 RF（cuML）

> 选做，能让 RF 训练时间从 30 分钟降到 < 5 分钟。

```bash
# 仅 Linux + NVIDIA GPU，且 driver 与 CUDA 12.x 匹配
conda install -c rapidsai -c conda-forge -c nvidia cuml=24.04 python=3.9 cuda-version=12.1
```

然后写一个 `yonod/models/cuml_rf_model.py` 替换 `rf_model.py` 的 sklearn 调用为 `cuml.ensemble.RandomForestRegressor`，并在 `yonod/evaluate.py` 的注册表中添加对应 key。**注意 cuml 在 Windows 上不支持，是迁 Linux 的核心收益之一**。本项目当前未集成此优化，作为后续扩展点。

---

### 15.2 Linux 适配要点

#### 15.2.1 已知的 Windows-only 代码点

| 文件 | 问题 | 修复 |
|---|---|---|
| `yonod/descriptors/molmetalm.py:28` | `DEFAULT_WEIGHT_PATH = r"C:\Users\joyjo\..."` **硬编码 Windows 路径** | 见 §15.1.6 第 6 步，用 `sed` 替换为 Linux 绝对路径，或传 `weight_path=` 参数覆盖 |

> `reagent_cache.py` 已用 `Path(__file__).parents[2]` 自动定位，`fisd.py` 默认从 `WEIGHTS/FISD/` 加载，均无跨平台问题。

迁移后快速检查残留 Windows 路径：

```bash
grep -rn "C:" yonod/ yonod.py
# 期望：仅 molmetalm.py:28 一行（修复后应无输出）
```

#### 15.2.2 文件名编码

`dataset/` 目录下的文件名均为英文，不存在跨平台中文路径问题。Ubuntu 默认 UTF-8 环境下直接可用：

```bash
locale | grep -i utf
# 期望看到 LANG=en_US.UTF-8 或 zh_CN.UTF-8
ls dataset/
# 期望：amide-coupling.csv  test-amide-coupling.csv
```

#### 15.2.3 PowerShell ↔ Bash 命令对照

| 操作 | PowerShell (Win) | Bash (Linux) |
|---|---|---|
| 变量赋值 | `$root = "/path"` | `root="/path"` |
| 环境变量 | `$env:HF_TOKEN = "xxx"` | `export HF_TOKEN=xxx` |
| 续行 | `` ` `` (反引号) | `\` (反斜杠) |
| 激活 conda | `conda activate yonod` | 完全一致 |
| 删文件 | `Remove-Item file` | `rm file` |
| 列目录 | `Get-ChildItem` / `ls` | `ls` |
| 路径分隔符 | `\` 或 `/` | 仅 `/` |
| 路径中含空格 | `"..."` 双引号 | 同上 |
| 短路 `&&` | PS 5.1 不支持，要 `; if ($?) {...}` | 原生支持 |

#### 15.2.4 sklearn n_jobs 在 Linux 上的差异

Linux 上 sklearn 用 `fork` 启动 worker（Windows 用 `spawn`），有两个好处：
- worker 启动快 10 倍
- 不需要 pickle 大 X 数组传给 worker（fork 直接继承父进程内存）

所以本地遇到的"Morgan × RF n_jobs=-1 启动延迟"在 Linux 上不会出现。

#### 15.2.5 nohup / tmux / screen

| 工具 | 用途 | 推荐场景 |
|---|---|---|
| `nohup ... &` | 后台运行，标准输出到 nohup.out | 一次性长任务 |
| `tmux` | 会话管理器，可断线重连 | **本项目首选** |
| `screen` | 类似 tmux，更老 | 兼容性最好 |

tmux 速查：

```bash
tmux new -s yonod          # 新建会话
tmux attach -t yonod       # 重连
tmux ls                    # 列所有会话
# 会话内: Ctrl+B 然后 D = 脱离，Ctrl+B 然后 [ = 滚屏
```

---

### 15.3 GitHub 仓库维护

> 仓库已建立：https://github.com/thinktraveller/YONOD

#### 15.3.1 已入库 vs 未入库资产

| 文件类型 | 入库状态 | 理由 |
|---|---|---|
| `yonod/` 源码 | ✅ 已入库 | 核心交付物 |
| `yonod.py` | ✅ 已入库 | 统一入口 |
| `YONOD项目构建计划书.md` | ✅ 已入库 | 设计文档 |
| `.gitignore` / `LICENSE` / `README.md` / `CHANGELOG.md` / `requirements.txt` | ✅ 已入库 | 标配 |
| `dataset/amide-coupling.csv`（47015 条） | ✅ 已入库 | MIT 协议，来自 aichemeco/amide_coupling |
| `dataset/test-amide-coupling.csv`（10 条） | ✅ 已入库 | 同上，用于调试 |
| `MIGRATION.md`（本节原始文件） | ❌ gitignored | 含内部部署细节，不公开分发 |
| `WEIGHTS/MolMetaLM-base/` (~500 MB) | ❌ gitignored | 超 GitHub 单文件 100 MB 限制；从 HuggingFace 单独下载 |
| `WEIGHTS/FISD/` (~19 MB) | ❌ gitignored | 无明确许可证；从上游 KeantChen/FISD 获取 |
| `cache/`, `results/`, `*.log` | ❌ gitignored | 运行产物 |
| `化学描述符相关项目/` | ❌ gitignored | 第三方源码，有各自 LICENSE |
| `__pycache__/`, `.venv/` | ❌ gitignored | Python/conda 产物 |

#### 15.3.2 日常推送流程

```bash
cd "C:\Users\joyjo\Desktop\其他大学资料\大创\YONOD"

# 1) 确认变更
git status
git diff

# 2) 暂存并提交（按语义化版本规范写 message）
git add yonod/ yonod.py README.md CHANGELOG.md requirements.txt .gitignore
git commit -m "feat: <简要描述>"

# 3) 推送到 GitHub
git push origin main
```

#### 15.3.3 提交后检查清单

```bash
# 仓库大小（应 < 50 MB；超过说明误入了大文件）
git count-objects -vH

# 检查是否误入了不该入库的文件
git ls-files | grep -E "WEIGHTS|cache|results|\.pth$|\.safetensors$|MIGRATION\.md"
# 期望: 无输出。任何输出都是漏掉的 .gitignore 规则
```

#### 15.3.4 大文件用 Git LFS（如要把权重也入库）

> 不推荐除非必要。建议把权重链接放在 README 里指向 HuggingFace。

```bash
# 装 git-lfs（一次）
# Win: 从 https://git-lfs.com 下载安装
# Ubuntu: sudo apt install git-lfs

git lfs install
git lfs track "*.safetensors" "*.pth"
git add .gitattributes
git commit -m "chore: track large model weights via LFS"
# 然后正常 git add WEIGHTS/  并 push
```

GitHub 免费 LFS 配额 1 GB/月，超出收费；权重 500 MB 入库会很快用完。**强烈建议不入库**。

#### 15.3.5 安全提醒

| 风险 | 应对 |
|---|---|
| HuggingFace token 写进代码 | 用 `.env` 文件（`.gitignore` 已忽略） |
| 内部部署细节（服务器 IP、路径）泄露 | `MIGRATION.md` 已 gitignore，不入库 |
| 第三方项目源码版权 | `.gitignore` 排除 `化学描述符相关项目/`，README 中链接原仓库 |
| 中间结果含个人路径 | `results/*.log` 已 ignored；若手动入库 log，注意脱敏 |
| commit message 暴露内部信息 | 避免在 message 里贴 IP / 密码 / 实验数据 |

---

### 15.4 整合工作流（推荐顺序）

1. **本地验证**
   - `python yonod.py --csv dataset/test-amide-coupling.csv ...` 烟测通过
2. **代码推送到 GitHub**
   - 按 §15.3.2 流程 push
3. **云端部署**
   - `git clone https://github.com/thinktraveller/YONOD.git`
   - 单独上传 `WEIGHTS/`（AutoDL 网盘或 scp）
   - 按 §15.1.6 装环境，**务必执行第 6 步修复 `molmetalm.py` 路径**
   - 烟测通过后 tmux + 全量 run，~1 小时
4. **回传结果**
   - 打包 `results/` 下载
   - 把 `report.html` 发给老师/团队
5. **结果回写仓库**（选做）
   - 把 `results/<task>/metrics_summary.csv` 入库作为基线快照
   - PNG 散点图体积较大，酌情不入库

---

### 15.5 常见坑（项目历史踩过的）

| 坑 | 触发场景 | 修复 |
|---|---|---|
| PowerShell 不支持 `&&` | 直接抄了 bash 命令 | 用 `; if ($?) { ... }` |
| AutoGluon `fastai` 链上的 spacy 要求 Python 3.10+ | `pip install autogluon[fastai]` 在 py3.9 上 | 只装 `[lightgbm,catboost]`，放弃 fastai |
| RF 默认 `verbose=0` 像卡死 | 全量 + 高维 + 300 树 | 加 `--heartbeat 60`，本项目已支持 |
| RBF SVR 在 n>10k 时 O(n²) 爆炸 | 47015 行单折 75+ 分钟未完 | `--svm-subsample 8000`（默认已设） |
| 试剂列里逗号 `,` 不是 RDKit 多片段分隔符 | `MolFromSmiles("A,B")` 返回 None | `yonod.py` 已自动替换 `,` → `.` |
| MolMetaLM 是 Llama 架构，不接受 `token_type_ids` | tokenizer 默认返回了它 | `molmetalm.py` 已过滤参数白名单 |
| `molmetalm.py` 默认路径为 Windows 绝对路径 | 直接在 Linux 上运行 | §15.1.6 第 6 步 sed 替换，或传 `weight_path=` |
| Win11 sklearn n_jobs=-1 的 pickle 启动延迟 | Win 用 spawn 而非 fork | Linux 上不存在；本地可在代码里临时改 `n_jobs=1` |
| `R^2` 标注渲染成字面量 | matplotlib 默认不解析 caret | 用 mathtext `$R^2$`（本项目已修复） |

---

### 15.6 长期演进建议

| 方向 | 改动量 | 收益 |
|---|---|---|
| 引入 cuML RF | 中（写 `yonod/models/cuml_rf_model.py` + 注册到 evaluate.py） | RF 训练 30 min → 3 min |
| MolMetaLM fine-tune | 大（需 yield 标签 + 训练循环） | MolMetaLM R² 可能从 0.70 提到 0.85+ |
| 描述符特征磁盘缓存 | 小 | 第二次跑同数据 < 5 秒 |
| 加 Linear / Lasso / Ridge 基线 | 小（注册新 model class） | 学术汇报常要"线性模型对照" |
| 加分类任务支持 | 中（pass / fail 二分类） | 适配产率阈值切分场景 |
| 把 `yonod.py` 改成 CLI 包 | 中（pyproject.toml + console_scripts） | `pip install yonod` 后命令行直用 |

---

## 十六、新增描述符构建计划（v1.2.0）

> **项目名称**：YONOD 描述符扩展
> **版本**：v1.2.0
> **创建日期**：2026-06-23
> **更新日期**：2026-06-23

**目标**：

1. **新增列角色分类机制**（核心架构升级）
   - 支持用户指定 SMILES 列的三种角色：反应物/产物/其他参与者
   - 反应类描述符（DRFP）仅使用反应物+产物
   - 分子类描述符（Morgan/RDKit2D 等）使用所有列
   - 提供 CLI 参数和交互向导两种配置方式
   - 向后兼容传统模式

2. **新增两个描述符**
   - **RDKit 2D 描述符**（~200维物化性质）
   - **DRFP 描述符**（2048维差分反应指纹）

---

### 16.1 可行性分析

#### 16.1.1 技术可行性

| 描述符 | 依赖 | 安装难度 | 计算速度 | 论文表现 |
|--------|------|----------|----------|----------|
| RDKit 2D | RDKit（已安装） | ⭐ 无需额外安装 | ~1ms/分子 | R²=0.89, MAE=6.6% |
| DRFP | drfp>=0.3.5 | ⭐⭐ pip安装 | ~5ms/反应 | R²=0.84, MAE=7.8% |

#### 16.1.2 数据集适配性与列角色分类

当前数据集 `dataset/amide-coupling.csv` 结构：
```csv
sub_1_smiles,sub_2_smiles,product_smiles,activation,additive,base,solvent,yield
```

**列角色分类**：
- **反应物列**：sub_1_smiles, sub_2_smiles（发生化学转化的底物）
- **产物列**：product_smiles（反应生成物）
- **其他参与者列**：activation, additive, base, solvent（催化剂、溶剂等）

**描述符适配性**：
- **RDKit 2D**：单分子模式，使用所有 7 列（横向拼接）→ 7×200 = 1400 维
- **DRFP**：反应模式，仅使用反应物+产物列构建反应 SMARTS：
  ```
  sub_1_smiles.sub_2_smiles>>product_smiles
  ```
  忽略 activation/additive/base/solvent → 2048 维

**化学语义正确性**：
- ✅ DRFP 不会将溶剂误当作反应物（避免化学意义错误）
- ✅ Morgan/RDKit2D 仍能利用全部列的信息（最大化特征空间）
- ✅ 用户可根据实验设计灵活调整角色分类

#### 16.1.3 风险评估

| 风险 | 等级 | 应对方案 |
|------|------|----------|
| 用户错误指定列角色 | 中 | 添加预览功能，显示示例反应 SMARTS |
| 交互流程变复杂 | 中 | 提供清晰提示文案 + "跳过分类"选项 |
| 向后兼容性问题 | 低 | 自动检测并降级到传统模式 |
| drfp 安装失败（网络） | 中 | 使用清华镜像或离线安装 |
| RDKit 版本兼容 | 低 | 硬编码描述符列表，避免 API 变化 |
| DRFP 依赖冲突 | 低 | 单独虚拟环境测试 |

---

### 16.2 技术选型

#### 16.2.1 列角色分类机制

**设计哲学**：

化学反应数据集本质上是**异构的**：
- 反应物和产物发生化学键断裂/形成（需要反应类描述符）
- 催化剂、溶剂不参与键变化（但影响反应环境）

传统的"所有列一视同仁"策略在分子类描述符上有效，但在反应类描述符上会导致化学语义错误。

**三分类 vs 二分类**：

| 方案 | 分类方式 | 优点 | 缺点 |
|------|---------|------|------|
| 二分类 | 反应列 / 非反应列 | 简单 | 无法区分反应物和产物 |
| 三分类 | 反应物 / 产物 / 其他 | 语义精确 | 交互稍复杂 |

**选择三分类**的理由：
1. DRFP 算法明确区分反应物和产物（`reactants>>products`）
2. 未来可能需要反应中心识别等需要此区分的算法
3. 复杂度增加有限（交互流程只多一步）

**技术实现**：
- 数据结构：`smiles_roles = {'reactant': [...], 'product': [...], 'other': [...]}`
- 传递方式：从 csv_loader → feature_builder → 描述符
- 向后兼容：未指定角色时，所有列归入 `'other'`（传统模式）

---

#### 16.2.2 RDKit 2D 描述符

**来源**：RDKit `rdkit.Chem.Descriptors` 模块

**描述符类别**（共约200个）：
- 分子量相关：MolWt, HeavyAtomMolWt, ExactMolWt
- 脂溶性：MolLogP, MolMR
- 拓扑极性表面积：TPSA
- 氢键：NumHDonors, NumHAcceptors
- 可旋转键：NumRotatableBonds
- 环系统：RingCount, NumAromaticRings
- 杂原子：NumHeteroatoms, FractionCSP3
- 电荷相关：MaxPartialCharge, MinPartialCharge
- 等等...

**设计决策**：
- 使用 2D-only 描述符（不需要3D构象）
- 硬编码描述符列表（避免RDKit版本差异）
- NaN/Inf 值替换为 0

#### 16.2.3 DRFP 描述符

**来源**：Probst et al. 2022, Digital Discovery

**算法原理**：
```
DRFP = Hash(产物子结构) XOR Hash(反应物子结构)
```
捕捉反应前后的分子结构变化，专门针对化学反应预测设计。

**输入格式**：
```
反应物1.反应物2>>产物
```

**输出维度**：2048位（可配置）

---

### 16.3 开发计划

#### 16.3.1 步骤 1：实现 SMILES 列角色分类机制

##### 目标说明

为支持 DRFP 等反应类描述符，需要让用户指定 SMILES 列的三种角色：
1. **反应物列**（reactants）：参与化学转化的底物分子
2. **产物列**（products）：反应生成的目标分子
3. **其他参与者列**（others）：催化剂、溶剂、添加剂等辅助成分

**设计原则**：
- 对于反应类描述符（如 DRFP），仅使用反应物+产物列
- 对于分子类描述符（Morgan/MACCS/RDKit2D/FISD/MolMetaLM/MAF），所有列横向拼接
- 未来扩展性：其他描述符可能也需要此区分（如反应中心识别等）

---

##### CLI 参数设计

**新增三个参数**：
```bash
python yonod.py \
    --csv dataset/amide-coupling.csv \
    --reactant-cols sub_1_smiles sub_2_smiles \      # 反应物列
    --product-cols product_smiles \                  # 产物列
    --other-cols activation additive base solvent \  # 其他参与者列（可选）
    --label-col yield \
    --descriptors drfp morgan \
    --models xgb
```

**参数验证规则**：
```python
# 1. 三类列不能重叠
assert len(set(reactant_cols) & set(product_cols)) == 0, "反应物列和产物列不能重叠"

# 2. 反应物+产物至少需要2列（构成有意义的反应）
assert len(reactant_cols) + len(product_cols) >= 2, "至少需要1个反应物+1个产物"

# 3. 如果未指定 other_cols，自动推断
# 所有 CSV 中的 SMILES 列 - (reactant_cols + product_cols) → other_cols

# 4. 向后兼容：如果用户未指定角色且未选择 DRFP，使用传统模式
if reactant_cols is None and product_cols is None:
    # 传统模式：所有列视为等价
    all_smiles_cols = detect_smiles_columns(csv_path)
else:
    # 三分类模式
    all_smiles_cols = reactant_cols + product_cols + other_cols
```

---

##### 交互向导流程更新

**当前流程**（v1.1）：
```
1. 选择 CSV 文件
2. 选择标签列
3. 选择 SMILES 列（多选）
4. 选择数值辅助列（可选）
5. 选择描述符
6. 选择模型
7. 运行
```

**新流程**（v1.2，三分类模式）：
```
1. 选择 CSV 文件
2. 选择标签列
3. [新增] 是否启用列角色分类？
   ├─ 是 → 进入步骤 4（三分类流程）
   └─ 否 → 使用传统模式（所有 SMILES 列等价）

4. [三分类流程]
   4.1 选择反应物列（多选，必填）
       显示所有 SMILES 列：
       [a] sub_1_smiles
       [b] sub_2_smiles
       [c] product_smiles
       [d] activation
       [e] base
       [f] solvent

       请输入反应物列（多选，如 ab）：ab ✓

   4.2 选择产物列（多选，必填）
       剩余可选列：
       [c] product_smiles
       [d] activation
       [e] base
       [f] solvent

       请输入产物列（多选，如 c）：c ✓

   4.3 确认其他参与者列
       剩余列将自动归为"其他参与者"：
       - activation
       - base
       - solvent

       是否确认？[Y/n]: Y ✓

5. 选择数值辅助列（可选）
6. 选择描述符
7. 选择模型
8. 运行
```

---

##### 具体操作

**文件 1：`yonod/universal/csv_loader.py`**

扩展返回值结构：

```python
def load_csv_with_roles(
    csv_path: str,
    reactant_cols: Optional[List[str]] = None,
    product_cols: Optional[List[str]] = None,
    other_cols: Optional[List[str]] = None,
    label_col: Optional[str] = None,
    numeric_cols: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """加载 CSV 并标记 SMILES 列角色。

    Returns:
        {
            'df': DataFrame,
            'label_col': str,
            'smiles_roles': {
                'reactant': ['sub_1_smiles', 'sub_2_smiles'],
                'product': ['product_smiles'],
                'other': ['activation', 'base', 'solvent'],
            },
            'all_smiles_cols': List[str],  # 所有 SMILES 列（拼接顺序）
            'numeric_cols': List[str],
        }
    """
```

**文件 2：`yonod/universal/feature_builder.py`**

更新特征构建逻辑：

```python
def build_universal_features(
    df: pd.DataFrame,
    smiles_roles: Dict[str, List[str]],  # 新增参数
    all_smiles_cols: List[str],
    numeric_cols: List[str],
    descriptor_name: str,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """根据描述符类型选择性使用 SMILES 列。"""
    # 1. 根据描述符类型选择使用哪些列
    if descriptor_name == "drfp":
        # DRFP：仅使用反应物+产物
        cols_to_use = smiles_roles['reactant'] + smiles_roles['product']
    else:
        # 其他描述符：使用所有 SMILES 列横向拼接
        cols_to_use = all_smiles_cols
```

**文件 3：`yonod/descriptors/drfp_desc.py`**

新增辅助函数：

```python
def build_reaction_smarts_from_df(
    df: pd.DataFrame,
    reactant_cols: List[str],
    product_cols: List[str],
) -> pd.Series:
    """从 DataFrame 构建反应 SMARTS 列。

    Returns:
        Series of reaction SMARTS strings (format: "R1.R2>>P1.P2")

    Example:
        reactant_cols = ['sub_1_smiles', 'sub_2_smiles']
        product_cols = ['product_smiles']
        → "CCO.CC(=O)O>>CCOC(C)=O"
    """
```

**文件 4：`yonod.py`（CLI 参数）**

```python
# 在 argparse 部分添加
parser.add_argument(
    "--reactant-cols",
    nargs="+",
    help="反应物 SMILES 列名（多个用空格分隔）",
)
parser.add_argument(
    "--product-cols",
    nargs="+",
    help="产物 SMILES 列名（多个用空格分隔）",
)
parser.add_argument(
    "--other-cols",
    nargs="+",
    help="其他参与者 SMILES 列名（可选，未指定则自动推断）",
)
```

---

##### 验证方法

**1. CLI 模式测试**：
```bash
# 测试三分类参数
python yonod.py \
    --csv dataset/amide-coupling.csv \
    --reactant-cols sub_1_smiles sub_2_smiles \
    --product-cols product_smiles \
    --other-cols activation base solvent \
    --label-col yield \
    --descriptors drfp morgan \
    --models xgb \
    --task-name test_role_classification

# 验证输出日志中显示：
# [INFO] 反应物列: sub_1_smiles, sub_2_smiles
# [INFO] 产物列: product_smiles
# [INFO] 其他参与者列: activation, base, solvent
# [INFO] DRFP 描述符使用: sub_1_smiles, sub_2_smiles, product_smiles
# [INFO] Morgan 描述符使用: 全部 6 列
```

**2. 交互向导测试**：
```bash
python yonod.py
# 跟随向导流程，验证：
# ✓ 能正确显示三分类提示
# ✓ 多选功能正常工作
# ✓ 剩余列自动归类
# ✓ 错误输入能正确提示
```

---

#### 16.3.2 步骤 2：实现 RDKit 2D 描述符

**文件**：`yonod/descriptors/rdkit2d.py`

```python
"""RDKit 2D physicochemical descriptors (~200 descriptors).

Computes a comprehensive set of 2D molecular properties without
requiring 3D conformer generation.
"""

from __future__ import annotations

import warnings
from typing import List, Tuple, Optional

import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors

from .base import BaseDescriptor


# 硬编码的 2D 描述符列表（避免 RDKit 版本兼容问题）
RDKIT_2D_DESCRIPTORS = [
    'MaxAbsEStateIndex', 'MaxEStateIndex', 'MinAbsEStateIndex', 'MinEStateIndex',
    'qed', 'SPS', 'MolWt', 'HeavyAtomMolWt', 'ExactMolWt', 'NumValenceElectrons',
    # ... (完整列表约 200 个描述符)
]


class RDKit2DDescriptor(BaseDescriptor):
    """RDKit 2D physicochemical descriptors.

    Computes ~200 2D molecular properties for each molecule.
    Failed SMILES parsing results in zero vectors with mask=False.
    """

    name = "rdkit2d"
    output_dim = len(RDKIT_2D_DESCRIPTORS)
```

**验证命令**：
```bash
python -c "from yonod.descriptors.rdkit2d import RDKit2DDescriptor; d = RDKit2DDescriptor(); print(f'维度: {d.output_dim}')"
```

**预期输出**：`维度: 200`（具体数值取决于列表长度）

---

#### 16.3.3 步骤 3：实现 DRFP 描述符

**依赖安装**：
```bash
# 使用清华镜像
pip install drfp -i https://pypi.tuna.tsinghua.edu.cn/simple

# 或阿里云镜像
pip install drfp -i https://mirrors.aliyun.com/pypi/simple
```

**文件**：`yonod/descriptors/drfp_desc.py`

```python
"""DRFP (Differential Reaction Fingerprint) descriptor.

DRFP captures the structural changes during a chemical reaction by
computing the symmetric difference of reactant and product fingerprints.

Reference:
    Probst, D., Schwaller, P., & Reymond, J. L. (2022).
    Reaction classification and yield prediction using the differential
    reaction fingerprint DRFP. Digital Discovery, 1(2), 91-97.

Input format:
    Reaction SMARTS: "reactant1.reactant2>>product"
    Example: "CCO.CC(=O)O>>CCOC(C)=O"
"""


class DRFPDescriptor(BaseDescriptor):
    """Differential Reaction Fingerprint (DRFP) descriptor.

    Computes reaction fingerprints that capture structural changes
    between reactants and products.

    Attributes:
        name: "drfp"
        output_dim: 2048 (default) or custom n_bits
    """

    name = "drfp"
    output_dim = 2048
```

**验证命令**：
```bash
python -c "
from yonod.descriptors.drfp_desc import DRFPDescriptor, build_reaction_smarts

# 测试反应 SMARTS 构建
rxn = build_reaction_smarts('CCO', 'CC(=O)O', 'CCOC(C)=O')
print(f'反应 SMARTS: {rxn}')

# 测试 DRFP 编码
d = DRFPDescriptor(n_bits=2048)
features, mask = d.featurize([rxn])
print(f'维度: {features.shape}, 成功: {mask[0]}')
"
```

---

#### 16.3.4 步骤 4：注册描述符到系统

**文件**：`yonod/evaluate.py`

在 `DESCRIPTOR_REGISTRY` 字典中添加新描述符：

```python
# 在文件顶部添加导入
from yonod.descriptors.rdkit2d import RDKit2DDescriptor
from yonod.descriptors.drfp_desc import DRFPDescriptor

# 在 DESCRIPTOR_REGISTRY 中添加
DESCRIPTOR_REGISTRY: Dict[str, Type[BaseDescriptor]] = {
    "morgan": MorganDescriptor,
    "atmomaccs": ATMOMACCSDescriptor,
    "fisd": FISDDescriptor,
    "molmetalm": MolMetaLMDescriptor,
    "maf": MAFDescriptor,
    "rdkit2d": RDKit2DDescriptor,   # 新增
    "drfp": DRFPDescriptor,          # 新增
}
```

**文件**：`yonod/descriptors/__init__.py`

更新 `__all__` 列表：

```python
from .rdkit2d import RDKit2DDescriptor      # 新增
from .drfp_desc import DRFPDescriptor       # 新增

__all__ = [
    # ...existing...
    "RDKit2DDescriptor",    # 新增
    "DRFPDescriptor",       # 新增
]
```

---

#### 16.3.5 步骤 5：更新主脚本集成

**文件**：`yonod.py`

更新描述符列表：

```python
# 在文件顶部常量定义处
_DESCRIPTOR_NAMES = ["morgan", "maccs", "fisd", "molmetalm", "maf", "rdkit2d", "drfp"]
```

---

#### 16.3.6 步骤 6：编写单元测试

**文件**：`tests/test_rdkit2d_descriptor.py`

```python
"""Unit tests for RDKit 2D descriptor."""

import numpy as np
import pytest

from yonod.descriptors.rdkit2d import RDKit2DDescriptor, RDKIT_2D_DESCRIPTORS


class TestRDKit2DDescriptor:
    """Test cases for RDKit2DDescriptor."""

    def test_output_dim(self):
        """Test output dimension matches descriptor list."""
        desc = RDKit2DDescriptor()
        assert desc.output_dim == len(RDKIT_2D_DESCRIPTORS)

    def test_valid_smiles(self):
        """Test featurization of valid SMILES."""
        desc = RDKit2DDescriptor()
        smiles = ["CCO", "CC(=O)O", "c1ccccc1"]
        features, mask = desc.featurize(smiles)

        assert features.shape == (3, desc.output_dim)
        assert mask.all()
        assert not np.isnan(features).any()
        assert not np.isinf(features).any()
```

**文件**：`tests/test_drfp_descriptor.py`

```python
"""Unit tests for DRFP descriptor."""

import numpy as np
import pytest

from yonod.descriptors.drfp_desc import (
    DRFPDescriptor,
    build_reaction_smarts,
    _ensure_drfp,
)


class TestBuildReactionSmarts:
    """Test cases for build_reaction_smarts function."""

    def test_two_substrates(self):
        """Test with two substrates."""
        rxn = build_reaction_smarts("CCO", "CC(=O)O", "CCOC(C)=O")
        assert rxn == "CCO.CC(=O)O>>CCOC(C)=O"


@pytest.mark.skipif(not _ensure_drfp(), reason="drfp not installed")
class TestDRFPDescriptor:
    """Test cases for DRFPDescriptor (requires drfp installed)."""

    def test_output_dim(self):
        """Test default output dimension."""
        desc = DRFPDescriptor()
        assert desc.output_dim == 2048
```

**运行测试**：
```bash
pytest tests/test_rdkit2d_descriptor.py tests/test_drfp_descriptor.py -v
```

---

#### 16.3.7 步骤 7：端到端集成测试

##### 测试场景 1：三分类模式 + DRFP

```bash
python yonod.py \
    --csv dataset/amide-coupling.csv \
    --reactant-cols sub_1_smiles sub_2_smiles \
    --product-cols product_smiles \
    --other-cols activation additive base solvent \
    --label-col yield \
    --descriptors drfp \
    --models xgb \
    --task-name test_drfp_role_classification
```

**预期输出**：
```
[INFO] SMILES 列角色分类：
  反应物列: sub_1_smiles, sub_2_smiles
  产物列: product_smiles
  其他参与者列: activation, additive, base, solvent

============================================================
  描述符: drfp
============================================================
  使用列（反应物+产物）: sub_1_smiles, sub_2_smiles, product_smiles
  特征维度: 2048
  有效样本: 470 / 470

[运行模型训练...]
```

---

##### 测试场景 2：传统模式 + RDKit 2D

```bash
# 不指定角色参数，使用传统模式
python yonod.py \
    --csv dataset/amide-coupling.csv \
    --label-col yield \
    --smiles-cols sub_1_smiles sub_2_smiles product_smiles activation \
    --descriptors rdkit2d \
    --models xgb \
    --task-name test_rdkit2d_traditional
```

**预期输出**：
```
[INFO] 使用传统模式（所有 SMILES 列等价）

============================================================
  描述符: rdkit2d
============================================================
  使用列（全部）: sub_1_smiles, sub_2_smiles, product_smiles, activation
  特征维度: 800  (200 × 4列)
  有效样本: 468 / 470
```

---

##### 测试场景 3：混合模式（DRFP + Morgan）

```bash
python yonod.py \
    --csv dataset/amide-coupling.csv \
    --reactant-cols sub_1_smiles sub_2_smiles \
    --product-cols product_smiles \
    --label-col yield \
    --descriptors drfp morgan \
    --models xgb rf \
    --task-name test_mixed_descriptors
```

**预期输出**：
```
============================================================
  描述符: drfp
============================================================
  使用列（反应物+产物）: sub_1_smiles, sub_2_smiles, product_smiles
  特征维度: 2048

============================================================
  描述符: morgan
============================================================
  使用列（全部）: sub_1_smiles, sub_2_smiles, product_smiles, activation, ...
  特征维度: 12288  (2048 × 6列)
```

---

#### 16.3.8 步骤 8：文档更新与 Git 提交

**Git 提交序列**：

```bash
# 1. 提交列角色分类机制
git add yonod/universal/csv_loader.py yonod/universal/feature_builder.py
git commit -m "$(cat <<'EOF'
feat(core): add SMILES column role classification (reactant/product/other)

- 新增三分类参数：--reactant-cols, --product-cols, --other-cols
- 扩展 csv_loader 支持角色映射
- feature_builder 根据描述符类型选择性使用列
- 向后兼容传统模式（未指定角色时）

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"

# 2. 提交 RDKit 2D 描述符
git add yonod/descriptors/rdkit2d.py tests/test_rdkit2d_descriptor.py
git commit -m "$(cat <<'EOF'
feat(descriptors): add RDKit 2D physicochemical descriptors (~200 dim)

- 硬编码 200+ 描述符列表（跨版本兼容）
- NaN/Inf 自动替换为 0
- 支持多组分 SMILES（取第一组分）

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"

# 3. 提交 DRFP 描述符
git add yonod/descriptors/drfp_desc.py tests/test_drfp_descriptor.py
git commit -m "$(cat <<'EOF'
feat(descriptors): add DRFP reaction fingerprint (2048 dim)

- 实现差分反应指纹（Probst et al. 2022）
- 自动从反应物/产物列构建 SMARTS
- 仅使用反应主成分，忽略催化剂/溶剂等
- 延迟导入 drfp（允许可选安装）

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"

# 4. 打标签
git tag -a v1.2.0 -m "$(cat <<'EOF'
Release v1.2.0: SMILES role classification + RDKit 2D + DRFP

新特性：
- SMILES 列角色分类（反应物/产物/其他参与者）
- RDKit 2D 描述符（~200维物化性质）
- DRFP 描述符（2048维差分反应指纹）
- 向后兼容传统模式

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

### 16.4 依赖管理

#### 16.4.1 新增依赖

| 依赖 | 版本 | 用途 | 安装方式 |
|------|------|------|----------|
| drfp | >=0.3.5 | DRFP 描述符 | pip install drfp |

#### 16.4.2 安装命令

```bash
# 使用清华镜像（推荐）
pip install drfp -i https://pypi.tuna.tsinghua.edu.cn/simple

# 使用阿里云镜像
pip install drfp -i https://mirrors.aliyun.com/pypi/simple

# 离线安装（如网络不可用）
pip download drfp -d ./wheels
pip install --no-index --find-links=./wheels drfp
```

#### 16.4.3 更新 requirements.txt

```
# 在 requirements.txt 中添加
drfp>=0.3.5
```

---

### 16.5 风险与注意事项

#### 16.5.1 RDKit 2D 描述符

| 风险 | 应对方案 |
|------|----------|
| 某些描述符返回 NaN/Inf | 已在代码中用 `np.nan_to_num` 处理 |
| RDKit 版本更新导致描述符列表变化 | 使用硬编码列表，不依赖动态获取 |
| 多组分 SMILES 处理 | 只取第一个组分计算 |

#### 16.5.2 DRFP 描述符

| 风险 | 应对方案 |
|------|----------|
| drfp 库安装失败 | 提供多个镜像源 + 离线安装方案 |
| 反应 SMARTS 格式错误 | 提供 `build_reaction_smarts()` 辅助函数 |
| 延迟导入失败 | 使用 `_ensure_drfp()` 检查，给出清晰错误提示 |

#### 16.5.3 性能预估

| 描述符 | 单分子/反应耗时 | 47000条数据预计耗时 |
|--------|----------------|---------------------|
| RDKit 2D | ~1ms | ~1分钟 |
| DRFP | ~5ms | ~4分钟 |

---

### 16.6 Q&A 记录

#### Q16.1: 为什么要引入列角色分类？

**A**:
化学反应数据集通常包含三类分子：
1. **反应物**（发生化学转化的底物）
2. **产物**（转化的结果）
3. **辅助成分**（催化剂、溶剂、添加剂等）

对于**反应类描述符**（如 DRFP），只应编码反应物→产物的变化，而忽略辅助成分。将溶剂、催化剂误当作反应物会导致化学语义错误。

对于**分子类描述符**（Morgan/RDKit2D 等），所有分子都提供有用信息，应全部使用。

列角色分类让系统能根据描述符类型智能选择列，提升模型准确性。

---

#### Q16.2: 如果数据集无法明确区分角色怎么办？

**A**:
三种应对方案：

1. **传统模式（不分类）**
   ```bash
   python yonod.py --csv data.csv --smiles-cols A B C --descriptors morgan
   ```
   不指定角色参数，系统自动降级，所有列被视为等价，横向拼接。

2. **保守分类**
   如果不确定某列角色，归入"其他参与者"：
   ```bash
   --reactant-cols A --product-cols B --other-cols C D E
   ```

3. **跳过 DRFP**
   如果无法区分角色，暂时不使用 DRFP 描述符，只用分子类描述符。

---

#### Q16.3: 为什么 DRFP 不能像 Morgan 一样用所有列？

**A**:
DRFP 的算法公式是：

```
DRFP = Hash(产物) XOR Hash(反应物)
```

这个 XOR 操作捕捉"反应前后的差异"。如果将溶剂、催化剂也包含进来：

```
错误示例：
DRFP = Hash(产物.溶剂.催化剂) XOR Hash(反应物.溶剂.催化剂)
```

由于溶剂和催化剂在反应前后不变，它们的 Hash 会相互抵消（XOR 特性），导致信息丢失。更糟的是，如果溶剂用量不同或顺序变化，会引入噪声。

正确做法是只编码发生化学变化的部分：

```
正确：
DRFP = Hash(产物) XOR Hash(反应物)
```

这就是为什么 DRFP 必须区分列角色。

---

#### Q16.4: 老项目升级到 v1.2.0 后，之前的脚本还能运行吗？

**A**:
完全兼容。如果脚本中未使用新参数（--reactant-cols 等），系统会自动使用传统模式，行为与 v1.1.0 完全一致。

唯一差异：如果选择了 DRFP 描述符但未指定列角色，会报错提示（这是新增的合理性检查）。

---

### 16.7 验收标准

#### 步骤 1：列角色分类机制
- [ ] CLI 参数正确解析（--reactant-cols, --product-cols, --other-cols）
- [ ] 交互向导能正确提示三分类流程
- [ ] 角色验证逻辑正常（无重叠、至少2列等）
- [ ] 向后兼容传统模式（未指定参数时）

#### 步骤 2：RDKit 2D 描述符
- [ ] `RDKit2DDescriptor` 能正确计算约200维描述符
- [ ] NaN/Inf 值被正确替换为 0
- [ ] 多组分 SMILES 只取第一组分

#### 步骤 3：DRFP 描述符
- [ ] `DRFPDescriptor` 能正确计算2048维反应指纹
- [ ] `build_reaction_smarts_from_df()` 能正确构建反应 SMARTS
- [ ] 延迟导入机制正常工作
- [ ] 未安装 drfp 时给出清晰提示

#### 步骤 4-8：系统集成
- [ ] 两个描述符已注册到 `DESCRIPTOR_REGISTRY`
- [ ] 主脚本打印角色信息和列使用情况
- [ ] 单元测试全部通过
- [ ] 端到端测试在样本数据上运行成功
- [ ] Git 提交规范，包含 v1.2.0 标签

---

**计划书编写完成**：2026-06-23
**计划书更新时间**：2026-06-23（新增列角色分类机制）
**预计执行时间**：4-6小时（含列角色分类机制开发）


---

## 十七、数据集输入流程重构构建计划

> **专题名称**: YONOD数据集输入向导(Dataset Input Wizard)
> **重构目标**: 完全放弃原有输入步骤,建立逐列声明式输入流程
> **文档版本**: v1.0 (2026-06-28)
> **添加日期**: 2026-06-29



---

### 17.1 项目概述

#### 17.1.1 重构目标
将原本散乱的数据集输入步骤完全重构为一个**向导式、逐列声明**的交互流程,核心改变:

1. **用户逐列声明列角色**(标签列、反应物SMILES、产物SMILES、其他组分SMILES、条件数值)
2. **每列必须指定名称**(英文+符号,默认值可供快速确认)
3. **合法性检验即时进行**,非法行标记但不中断流程
4. **生成规范数据集**:固定列顺序(reactant → others → condition → product → label)
5. **生成列映射文件**(CSV格式,记录原始列名→角色→新列名的对应关系)
6. **非法值报告**(Markdown格式,可选择输出修复后的数据集到目标文件夹)

#### 17.1.2 核心设计决策
基于可行性分析报告的回复,以下设计决策已确认:

| 决策点 | 最终方案 | 依据 |
|--------|---------|------|
| **配置文件导出功能** | 不采纳 | 不同数据集格式大相径庭,难以复用 |
| **规范数据集保留原始列** | 不采纳 | 需要查看原始格式可直接查看原始数据集 |
| **其他组分列智能命名** | 部分采纳 | 默认依然是原始列名,建议名称仅在输入界面供复制 |
| **一键修复脚本** | 部分采纳 | 必须保留原始数据集,修复后数据集输出到目标文件夹,仅保留"删除行"手段 |
| **列映射文件加载时跳过步骤2** | 强制配置 | 跳过步骤2,但正常对每列进行合法性检查,正常进入步骤3生成报告 |
| **非法值检验策略** | 已标记行仍参与检验 | 维护"已标记行"集合,对其仍检验(仅记录错误,不重复计数) |
| **产物列单SMILES定义** | 严格定义(可在文档中说明宽松定义) | 不含任何分隔符(点号、分号、逗号)的SMILES字符串 |
| **描述符配置粒度** | 逐个调整 | 顺序相关描述符有默认顺序,用户逐个调整 |
| **列名冲突处理** | 强制唯一 | 步骤2维护`used_names`集合,禁止重复 |
| **非法输入报告生成** | 有非法输入才生成 | 如无非法输入,不生成报告 |
| **描述符默认顺序** | reactant-others-product | 所有顺序相关描述符默认此顺序拼接 |
| **全部合法时跳过步骤3** | 跳过 | 如果合法性检查全部通过,则跳过步骤3,不重复生成规范数据集 |

#### 17.1.3 预期成果
| 形态 | 内容 |
|------|------|
| 交互式向导程序 | Python脚本,终端运行,逐步引导用户完成配置 |
| 列映射说明表 | CSV文件,记录`origin_name,role,name`三元组 |
| 规范数据集 | CSV文件,按固定顺序排列,每单元格仅含单个SMILES或单个数值 |
| 非法输入报告 | Markdown文档(可选),详细列出被排除的行及非法值位置 |
| 修复后数据集 | CSV文件(可选),删除非法行后的干净数据集 |

---

### 17.2 可行性分析

#### 17.2.1 技术可行性
| 技术点 | 可行性 | 依据 |
|--------|--------|------|
| SMILES合法性检验 | 完全可行 | RDKit的`Chem.MolFromSmiles()`可直接判断 |
| 数值合法性检验 | 完全可行 | Pandas的`pd.to_numeric(errors='coerce')`可识别非数值 |
| CSV读写 | 完全可行 | Pandas标准功能 |
| 交互式输入 | 完全可行 | Python `input()`配合循环和验证逻辑 |
| SMILES分隔符拆分 | 完全可行 | 正则表达式或`str.split()`配合RDKit验证 |
| Markdown报告生成 | 完全可行 | 字符串格式化+文件写入 |

#### 17.2.2 主要风险与应对
| 风险 | 概率 | 影响 | 应对策略 |
|------|------|------|----------|
| 用户输入列序号错误 | 高 | 中 | 每次输入后回显列名,要求用户确认 |
| 列名称含非法字符 | 中 | 低 | 实时正则验证,拒绝非英文符号 |
| SMILES含多种分隔符混用 | 低 | 中 | 支持`.` ` ` `;` `,`多种分隔符,统一拆分 |
| 规范数据集列数过多(如反应物>20个) | 低 | 低 | 无硬性限制,但在报告中提示数据稀疏性 |
| 用户中途退出 | 中 | 低 | 提供保存进度功能(保存部分配置为JSON) |

#### 17.2.3 工作量估算
| 步骤 | 预计工时 | 难度 |
|------|----------|------|
| 步骤1实现(文件路径输入) | 0.5小时 | 低 |
| 步骤2实现(逐列声明) | 3小时 | 中 |
| 步骤3.1实现(非法值报告) | 2小时 | 中 |
| 步骤3.2实现(列映射表) | 0.5小时 | 低 |
| 步骤3.3实现(规范数据集) | 2小时 | 中 |
| 步骤4-8实现(描述符/模型/其他) | 1小时 | 低(主要是UI展示) |
| 测试与文档 | 2小时 | 低 |
| **总计** | **11小时** | **中等** |

---

### 17.3 环境配置

#### 17.3.1 硬件/系统要求
- CPU: 2核以上
- RAM: 4GB以上
- 磁盘: 100MB可用空间(用于缓存和输出文件)
- OS: Windows 11 / Linux / macOS均可

#### 17.3.2 开发环境搭建
#### Git初始化

```powershell
# 如果项目目录尚未初始化Git仓库
cd "C:\Users\joyjo\Desktop\其他大学资料\大创\YONOD"
git init
```

#### 17.3.3 依赖安装
**最低版本要求:**

- Python >= 3.9
- pandas >= 2.0.0
- rdkit >= 2023.9.0
- numpy >= 1.26.0

**安装命令(Windows PowerShell):**

```powershell
# 激活conda环境(如果使用conda)
conda activate yonod

# 或直接使用pip安装(如果使用venv或系统Python)
pip install pandas>=2.0.0 rdkit>=2023.9.0 numpy>=1.26.0

# 国内用户镜像配置(推荐)
pip install pandas>=2.0.0 rdkit>=2023.9.0 numpy>=1.26.0 -i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn
```

**验证安装:**

```powershell
python -c "import pandas; print('pandas version:', pandas.__version__)"
python -c "from rdkit import Chem; print('RDKit imported successfully')"
python -c "import numpy; print('numpy version:', numpy.__version__)"
```

---

### 17.4 开发计划

### 步骤1: 指定初始数据集、列映射文件、项目名称和项目文件夹位置

#### 目标说明

用户提供四项基本信息:
1. **初始数据集路径**(必选): CSV格式,包含原始反应数据
2. **列映射文件路径**(可选): 如果提供,则跳过步骤2的列角色声明,直接进入合法性检验
3. **项目名称**(必选): 用于生成输出文件的前缀
4. **项目文件夹位置**(必选): 所有输出文件将保存在此文件夹下

如果提供了列映射文件,程序将:
- 跳过步骤2(逐列声明)
- 强制对每一列进行合法性检查
- 正常进入步骤3生成报告(如有非法值)
- 如果合法性检查全部通过,则跳过步骤3(无需重复生成一模一样的规范数据集)

#### 具体操作

**1.1 创建主脚本文件**

创建文件: `YONOD/dataset_input_wizard.py`

核心函数:
```python
def step1_collect_basic_info():
    """
    收集基本信息:初始数据集路径、列映射文件路径(可选)、项目名称、项目文件夹

    Returns:
        dict: {
            'dataset_path': str,
            'mapping_path': str | None,
            'project_name': str,
            'project_folder': str
        }
    """
    print("=" * 60)
    print("步骤1: 指定初始数据集、列映射文件、项目名称和项目文件夹")
    print("=" * 60)

    # 1. 输入初始数据集路径
    while True:
        dataset_path = input("请输入初始数据集路径(CSV格式): ").strip()
        if os.path.exists(dataset_path) and dataset_path.endswith('.csv'):
            print(f"✓ 数据集文件存在: {dataset_path}")
            break
        else:
            print("✗ 文件不存在或不是CSV格式,请重新输入")

    # 2. 可选: 输入列映射文件路径
    mapping_path = input("请输入列映射文件路径(可选,直接回车跳过): ").strip()
    if mapping_path:
        if os.path.exists(mapping_path) and mapping_path.endswith('.csv'):
            print(f"✓ 列映射文件存在: {mapping_path}")
            print("  将跳过步骤2,直接使用此映射进行合法性检验")
        else:
            print("✗ 文件不存在或不是CSV格式,将忽略此输入,进入正常流程")
            mapping_path = None
    else:
        mapping_path = None

    # 3. 输入项目名称
    while True:
        project_name = input("请输入项目名称(仅英文字母、数字、下划线): ").strip()
        if re.match(r'^[a-zA-Z0-9_]+$', project_name):
            print(f"✓ 项目名称: {project_name}")
            break
        else:
            print("✗ 项目名称只能包含英文字母、数字、下划线")

    # 4. 输入项目文件夹位置
    while True:
        project_folder = input("请输入项目文件夹位置(将在此创建输出文件): ").strip()
        if os.path.isdir(project_folder) or not os.path.exists(project_folder):
            os.makedirs(project_folder, exist_ok=True)
            print(f"✓ 项目文件夹: {project_folder}")
            break
        else:
            print("✗ 路径无效或不是文件夹")

    return {
        'dataset_path': dataset_path,
        'mapping_path': mapping_path,
        'project_name': project_name,
        'project_folder': project_folder
    }
```

**1.2 读取数据集并展示基本信息**

```python
def load_and_preview_dataset(dataset_path):
    """
    加载数据集并展示基本信息

    Returns:
        pd.DataFrame: 原始数据集
    """
    df = pd.read_csv(dataset_path)
    print(f"\n数据集基本信息:")
    print(f"  总行数: {len(df)}")
    print(f"  总列数: {len(df.columns)}")
    print(f"\n列序号和列名称:")
    for idx, col in enumerate(df.columns):
        print(f"  [{idx}] {col}")

    return df
```

#### 验证方法

**单元测试:**

创建测试文件: `YONOD/tests/test_step1.py`

```python
import pytest
import os
import pandas as pd
from dataset_input_wizard import step1_collect_basic_info, load_and_preview_dataset

def test_load_valid_csv(tmp_path):
    """测试加载有效的CSV文件"""
    # 创建临时CSV文件
    test_csv = tmp_path / "test.csv"
    df = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})
    df.to_csv(test_csv, index=False)

    # 加载并验证
    loaded_df = load_and_preview_dataset(str(test_csv))
    assert len(loaded_df) == 2
    assert list(loaded_df.columns) == ['col1', 'col2']

def test_project_name_validation():
    """测试项目名称验证逻辑"""
    import re

    valid_names = ['project1', 'my_project', 'Project_123']
    invalid_names = ['项目1', 'my-project', 'project name', '']

    pattern = r'^[a-zA-Z0-9_]+$'

    for name in valid_names:
        assert re.match(pattern, name), f"{name} should be valid"

    for name in invalid_names:
        assert not re.match(pattern, name), f"{name} should be invalid"

# 运行测试
pytest tests/test_step1.py -v
```

**手动验证检查项:**

1. 输入不存在的文件路径 → 提示错误并要求重新输入
2. 输入非CSV文件 → 提示错误并要求重新输入
3. 输入含中文/特殊字符的项目名称 → 提示错误并要求重新输入
4. 输入不存在的项目文件夹 → 自动创建文件夹
5. 加载CSV后正确显示行数、列数、列序号和列名称

#### 风险提示

1. **大文件加载:** 如果数据集超过100MB,Pandas读取可能较慢,考虑添加进度提示
2. **编码问题:** 如果CSV不是UTF-8编码,可能读取失败,需添加编码检测逻辑(`chardet`库)
3. **列名重复:** Pandas会自动处理重复列名(添加`.1` `.2`后缀),但需在步骤2中提醒用户

---

### 步骤2: 逐列声明列角色和名称

#### 目标说明

**如果步骤1中提供了列映射文件,则完全跳过本步骤。**

用户从原始数据集的所有列中,先选择需要的列(备选列列表),然后依次声明每列的角色和名称:

1. **标签列**(label): 必须且只能1列,合法性检验排除非数值和空值
2. **反应物SMILES列**(reactant): 可多列,合法性检验排除非法SMILES和空值
3. **产物SMILES列**(product): 必须且只能1列,每单元格只能包含单个SMILES,合法性检验排除非法SMILES和空值
4. **其他组分SMILES列**(others): 可多列,合法性检验排除非法SMILES但**不排除空值**
5. **条件数值列**(condition): 可多列,合法性检验排除非数值但**不排除空值**

**关键约束:**
- 列名称必须仅使用英文和符号,禁止中文
- 维护`used_names`集合,禁止列名称重复(强制唯一)
- 每声明完一列,该列从备选列列表中删除
- 建议的智能名称(如solvent、catalyst、temperature)仅在输入界面显示供用户复制,默认值仍是原始列名

#### 具体操作

**2.1 选择需要的列**

```python
def step2_select_columns(df):
    """
    从原始数据集中选择需要的列

    Args:
        df: 原始数据集DataFrame

    Returns:
        list: 选中的列索引列表
    """
    print("\n" + "=" * 60)
    print("步骤2.1: 选择需要的列")
    print("=" * 60)
    print("当前所有列:")
    for idx, col in enumerate(df.columns):
        print(f"  [{idx}] {col}")

    print("\n请输入需要的列序号,用逗号分隔(例如: 0,1,3,5)")
    print("或输入'all'选择全部列")

    while True:
        user_input = input("列序号: ").strip()

        if user_input.lower() == 'all':
            selected_indices = list(range(len(df.columns)))
            break

        try:
            selected_indices = [int(x.strip()) for x in user_input.split(',')]
            # 验证索引有效性
            if all(0 <= idx < len(df.columns) for idx in selected_indices):
                break
            else:
                print("✗ 存在无效的列序号,请重新输入")
        except ValueError:
            print("✗ 输入格式错误,请使用逗号分隔的数字")

    selected_columns = [df.columns[idx] for idx in selected_indices]
    print(f"\n✓ 已选择 {len(selected_columns)} 列:")
    for idx, col in zip(selected_indices, selected_columns):
        print(f"  [{idx}] {col}")

    return selected_indices
```

**2.2 声明标签列**

```python
def step2_2_declare_label_column(df, available_columns, used_names):
    """
    声明标签列(label)

    Args:
        df: 原始数据集
        available_columns: 可用列索引列表
        used_names: 已使用的列名称集合

    Returns:
        dict: {'origin_idx': int, 'origin_name': str, 'role': 'label', 'name': str, 'invalid_rows': set}
    """
    print("\n" + "=" * 60)
    print("步骤2.2: 声明标签列(label)")
    print("=" * 60)
    print("可选列:")
    for idx in available_columns:
        print(f"  [{idx}] {df.columns[idx]}")

    # 选择列
    while True:
        try:
            col_idx = int(input("请输入标签列序号(只能选择1列): ").strip())
            if col_idx in available_columns:
                break
            else:
                print("✗ 该列不在可选列表中")
        except ValueError:
            print("✗ 请输入有效的数字")

    origin_name = df.columns[col_idx]

    # 输入列名称
    default_name = 'yield'
    print(f"\n请为此列指定名称(默认: {default_name})")
    print("列名称只能包含英文字母、数字、下划线、连字符")

    while True:
        name = input(f"列名称[{default_name}]: ").strip() or default_name

        # 验证名称格式
        if not re.match(r'^[a-zA-Z0-9_-]+$', name):
            print("✗ 列名称只能包含英文、数字、下划线、连字符")
            continue

        # 验证名称唯一性
        if name in used_names:
            print(f"✗ 列名称'{name}'已被使用,请使用其他名称")
            continue

        break

    # 合法性检验: 排除非数值和空值
    invalid_rows = validate_numeric_column(df, col_idx)

    print(f"\n✓ 标签列配置完成:")
    print(f"  原始列名: {origin_name}")
    print(f"  新列名: {name}")
    print(f"  非法行数: {len(invalid_rows)}")

    used_names.add(name)
    available_columns.remove(col_idx)

    return {
        'origin_idx': col_idx,
        'origin_name': origin_name,
        'role': 'label',
        'name': name,
        'invalid_rows': invalid_rows
    }

def validate_numeric_column(df, col_idx):
    """
    验证数值列,返回非法行索引集合

    非法情况:
    1. 非数值值
    2. 空值(NaN)

    Returns:
        set: 非法行索引集合
    """
    column = df.iloc[:, col_idx]
    invalid_rows = set()

    for idx, value in enumerate(column):
        # 检查空值
        if pd.isna(value):
            invalid_rows.add(idx)
            continue

        # 检查是否为数值
        try:
            float(value)
        except (ValueError, TypeError):
            invalid_rows.add(idx)

    return invalid_rows
```

**2.3 声明反应物SMILES列**

```python
def step2_3_declare_reactant_columns(df, available_columns, used_names):
    """
    声明反应物SMILES列(可多列)

    Returns:
        list[dict]: 每个dict包含 origin_idx, origin_name, role='reactant', name, invalid_rows
    """
    print("\n" + "=" * 60)
    print("步骤2.3: 声明反应物SMILES列")
    print("=" * 60)

    reactant_columns = []

    while True:
        if not available_columns:
            print("可选列已全部声明完毕")
            break

        print("\n当前可选列:")
        for idx in available_columns:
            print(f"  [{idx}] {df.columns[idx]}")

        user_input = input("\n请输入反应物列序号(多列用逗号分隔,直接回车结束): ").strip()

        if not user_input:
            break

        try:
            col_indices = [int(x.strip()) for x in user_input.split(',')]

            # 验证有效性
            if not all(idx in available_columns for idx in col_indices):
                print("✗ 存在无效的列序号")
                continue

            # 为每列声明名称
            for col_idx in col_indices:
                origin_name = df.columns[col_idx]
                default_name = 'reactant'

                print(f"\n为列 [{col_idx}] {origin_name} 指定名称(默认: {default_name})")

                while True:
                    name = input(f"列名称[{default_name}]: ").strip() or default_name

                    if not re.match(r'^[a-zA-Z0-9_-]+$', name):
                        print("✗ 列名称只能包含英文、数字、下划线、连字符")
                        continue

                    if name in used_names:
                        print(f"✗ 列名称'{name}'已被使用")
                        continue

                    break

                # 合法性检验: 排除非法SMILES和空值
                invalid_rows = validate_smiles_column(df, col_idx, allow_empty=False)

                print(f"✓ 非法行数: {len(invalid_rows)}")

                reactant_columns.append({
                    'origin_idx': col_idx,
                    'origin_name': origin_name,
                    'role': 'reactant',
                    'name': name,
                    'invalid_rows': invalid_rows
                })

                used_names.add(name)
                available_columns.remove(col_idx)

        except ValueError:
            print("✗ 输入格式错误")

    print(f"\n✓ 共声明 {len(reactant_columns)} 个反应物列")
    return reactant_columns

def validate_smiles_column(df, col_idx, allow_empty=False):
    """
    验证SMILES列,返回非法行索引集合

    Args:
        df: 数据集
        col_idx: 列索引
        allow_empty: 是否允许空值(others列允许,reactant/product列不允许)

    非法情况:
    1. 非法SMILES(RDKit无法解析)
    2. 空值(如果allow_empty=False)

    Returns:
        set: 非法行索引集合
    """
    from rdkit import Chem

    column = df.iloc[:, col_idx]
    invalid_rows = set()

    for idx, value in enumerate(column):
        # 检查空值
        if pd.isna(value) or str(value).strip() == '':
            if not allow_empty:
                invalid_rows.add(idx)
            continue

        # 将值转为字符串
        smiles_str = str(value).strip()

        # 尝试解析SMILES(可能包含多个分子,用.或空格或;或,分隔)
        # 分隔后逐个验证
        separators = ['.', ' ', ';', ',']
        molecules = [smiles_str]  # 默认当作单个分子

        for sep in separators:
            if sep in smiles_str:
                molecules = [s.strip() for s in smiles_str.split(sep) if s.strip()]
                break

        # 验证每个分子
        for mol_smiles in molecules:
            mol = Chem.MolFromSmiles(mol_smiles)
            if mol is None:
                invalid_rows.add(idx)
                break

    return invalid_rows
```

**2.4 声明产物SMILES列**

```python
def step2_4_declare_product_column(df, available_columns, used_names):
    """
    声明产物SMILES列(必须且只能1列)

    关键要求:
    - 每个单元格只能包含一个独立的SMILES(不含分隔符)

    Returns:
        dict: origin_idx, origin_name, role='product', name, invalid_rows
    """
    print("\n" + "=" * 60)
    print("步骤2.4: 声明产物SMILES列")
    print("=" * 60)
    print("⚠️  产物列要求: 每个单元格只能包含一个独立的SMILES")
    print("    不允许含有分隔符(点号.、分号;、逗号,)")
    print()

    print("可选列:")
    for idx in available_columns:
        print(f"  [{idx}] {df.columns[idx]}")

    # 选择列
    while True:
        try:
            col_idx = int(input("请输入产物列序号(只能选择1列): ").strip())
            if col_idx in available_columns:
                break
            else:
                print("✗ 该列不在可选列表中")
        except ValueError:
            print("✗ 请输入有效的数字")

    origin_name = df.columns[col_idx]
    default_name = 'product'

    print(f"\n请为此列指定名称(默认: {default_name})")

    while True:
        name = input(f"列名称[{default_name}]: ").strip() or default_name

        if not re.match(r'^[a-zA-Z0-9_-]+$', name):
            print("✗ 列名称只能包含英文、数字、下划线、连字符")
            continue

        if name in used_names:
            print(f"✗ 列名称'{name}'已被使用")
            continue

        break

    # 合法性检验: 排除非法SMILES、空值、含分隔符的SMILES
    invalid_rows = validate_product_column(df, col_idx)

    print(f"\n✓ 产物列配置完成:")
    print(f"  原始列名: {origin_name}")
    print(f"  新列名: {name}")
    print(f"  非法行数: {len(invalid_rows)}")

    used_names.add(name)
    available_columns.remove(col_idx)

    return {
        'origin_idx': col_idx,
        'origin_name': origin_name,
        'role': 'product',
        'name': name,
        'invalid_rows': invalid_rows
    }

def validate_product_column(df, col_idx):
    """
    验证产物列,返回非法行索引集合

    产物列特殊要求:
    1. 不能为空
    2. 必须是合法SMILES
    3. 不能含有分隔符(.;, 空格)

    注: 此处采用严格定义,可在文档中说明宽松定义的可能性
    """
    from rdkit import Chem

    column = df.iloc[:, col_idx]
    invalid_rows = set()

    for idx, value in enumerate(column):
        # 检查空值
        if pd.isna(value) or str(value).strip() == '':
            invalid_rows.add(idx)
            continue

        smiles_str = str(value).strip()

        # 检查是否含有分隔符
        if any(sep in smiles_str for sep in ['.', ';', ',', ' ']):
            invalid_rows.add(idx)
            continue

        # 验证SMILES
        mol = Chem.MolFromSmiles(smiles_str)
        if mol is None:
            invalid_rows.add(idx)

    return invalid_rows
```

**2.5 声明其他组分SMILES列**

```python
def step2_5_declare_others_columns(df, available_columns, used_names):
    """
    声明其他组分SMILES列(可多列)

    特点:
    - 允许空值
    - 提供智能命名建议(solvent、catalyst、reagent、base)

    Returns:
        list[dict]: 每个dict包含 origin_idx, origin_name, role='others', name, invalid_rows
    """
    print("\n" + "=" * 60)
    print("步骤2.5: 声明其他组分SMILES列")
    print("=" * 60)
    print("常用名称建议: solvent(溶剂)、catalyst(催化剂)、reagent(试剂)、base(碱)")
    print()

    others_columns = []

    while True:
        if not available_columns:
            print("可选列已全部声明完毕")
            break

        print("\n当前可选列:")
        for idx in available_columns:
            print(f"  [{idx}] {df.columns[idx]}")

        user_input = input("\n请输入其他组分列序号(多列用逗号分隔,直接回车结束): ").strip()

        if not user_input:
            break

        try:
            col_indices = [int(x.strip()) for x in user_input.split(',')]

            if not all(idx in available_columns for idx in col_indices):
                print("✗ 存在无效的列序号")
                continue

            for col_idx in col_indices:
                origin_name = df.columns[col_idx]

                # 智能命名建议
                suggested_name = suggest_others_name(origin_name)
                default_name = origin_name  # 默认依然是原始列名

                print(f"\n为列 [{col_idx}] {origin_name} 指定名称")
                print(f"  默认: {default_name}")
                if suggested_name != default_name:
                    print(f"  建议: {suggested_name} (可复制)")

                while True:
                    name = input(f"列名称[{default_name}]: ").strip() or default_name

                    if not re.match(r'^[a-zA-Z0-9_-]+$', name):
                        print("✗ 列名称只能包含英文、数字、下划线、连字符")
                        continue

                    if name in used_names:
                        print(f"✗ 列名称'{name}'已被使用")
                        continue

                    break

                # 合法性检验: 排除非法SMILES,但允许空值
                invalid_rows = validate_smiles_column(df, col_idx, allow_empty=True)

                print(f"✓ 非法行数: {len(invalid_rows)}")

                others_columns.append({
                    'origin_idx': col_idx,
                    'origin_name': origin_name,
                    'role': 'others',
                    'name': name,
                    'invalid_rows': invalid_rows
                })

                used_names.add(name)
                available_columns.remove(col_idx)

        except ValueError:
            print("✗ 输入格式错误")

    print(f"\n✓ 共声明 {len(others_columns)} 个其他组分列")
    return others_columns

def suggest_others_name(origin_name):
    """
    根据原始列名推荐智能命名

    规则:
    - 包含'溶剂'/'solvent' → 'solvent'
    - 包含'催化'/'catalyst' → 'catalyst'
    - 包含'试剂'/'reagent' → 'reagent'
    - 包含'碱'/'base' → 'base'
    - 否则返回原始列名
    """
    origin_lower = origin_name.lower()

    if '溶剂' in origin_name or 'solvent' in origin_lower:
        return 'solvent'
    elif '催化' in origin_name or 'catalyst' in origin_lower:
        return 'catalyst'
    elif '试剂' in origin_name or 'reagent' in origin_lower:
        return 'reagent'
    elif '碱' in origin_name or 'base' in origin_lower:
        return 'base'
    else:
        return origin_name
```

**2.6 声明条件数值列**

```python
def step2_6_declare_condition_columns(df, available_columns, used_names):
    """
    声明条件数值列(可多列)

    特点:
    - 允许空值
    - 提供智能命名建议(temperature、pressure、time)

    Returns:
        list[dict]: 每个dict包含 origin_idx, origin_name, role='condition', name, invalid_rows
    """
    print("\n" + "=" * 60)
    print("步骤2.6: 声明条件数值列")
    print("=" * 60)
    print("常用名称建议: temperature(温度)、pressure(压力)、time(时间)")
    print()

    condition_columns = []

    while True:
        if not available_columns:
            print("可选列已全部声明完毕")
            break

        print("\n当前可选列:")
        for idx in available_columns:
            print(f"  [{idx}] {df.columns[idx]}")

        user_input = input("\n请输入条件数值列序号(多列用逗号分隔,直接回车结束): ").strip()

        if not user_input:
            break

        try:
            col_indices = [int(x.strip()) for x in user_input.split(',')]

            if not all(idx in available_columns for idx in col_indices):
                print("✗ 存在无效的列序号")
                continue

            for col_idx in col_indices:
                origin_name = df.columns[col_idx]

                # 智能命名建议
                suggested_name = suggest_condition_name(origin_name)
                default_name = origin_name

                print(f"\n为列 [{col_idx}] {origin_name} 指定名称")
                print(f"  默认: {default_name}")
                if suggested_name != default_name:
                    print(f"  建议: {suggested_name} (可复制)")

                while True:
                    name = input(f"列名称[{default_name}]: ").strip() or default_name

                    if not re.match(r'^[a-zA-Z0-9_-]+$', name):
                        print("✗ 列名称只能包含英文、数字、下划线、连字符")
                        continue

                    if name in used_names:
                        print(f"✗ 列名称'{name}'已被使用")
                        continue

                    break

                # 合法性检验: 排除非数值,但允许空值
                invalid_rows = validate_numeric_column_allow_empty(df, col_idx)

                print(f"✓ 非法行数: {len(invalid_rows)}")

                condition_columns.append({
                    'origin_idx': col_idx,
                    'origin_name': origin_name,
                    'role': 'condition',
                    'name': name,
                    'invalid_rows': invalid_rows
                })

                used_names.add(name)
                available_columns.remove(col_idx)

        except ValueError:
            print("✗ 输入格式错误")

    print(f"\n✓ 共声明 {len(condition_columns)} 个条件数值列")
    return condition_columns

def suggest_condition_name(origin_name):
    """根据原始列名推荐智能命名"""
    origin_lower = origin_name.lower()

    if '温度' in origin_name or 'temp' in origin_lower:
        return 'temperature'
    elif '压力' in origin_name or 'pressure' in origin_lower:
        return 'pressure'
    elif '时间' in origin_name or 'time' in origin_lower:
        return 'time'
    else:
        return origin_name

def validate_numeric_column_allow_empty(df, col_idx):
    """
    验证数值列(允许空值),返回非法行索引集合

    非法情况: 非数值值(但空值允许)
    """
    column = df.iloc[:, col_idx]
    invalid_rows = set()

    for idx, value in enumerate(column):
        if pd.isna(value):
            continue  # 空值允许

        try:
            float(value)
        except (ValueError, TypeError):
            invalid_rows.add(idx)

    return invalid_rows
```

#### 验证方法

**单元测试:**

```python
# tests/test_step2.py
def test_validate_numeric_column():
    """测试数值列验证"""
    df = pd.DataFrame({'col': [1.0, 2.5, 'abc', np.nan, 3]})
    invalid = validate_numeric_column(df, 0)
    assert invalid == {2, 3}  # 'abc'和NaN

def test_validate_smiles_column():
    """测试SMILES列验证"""
    df = pd.DataFrame({'col': ['CCO', 'invalid', '', np.nan, 'c1ccccc1']})

    # 不允许空值
    invalid = validate_smiles_column(df, 0, allow_empty=False)
    assert 1 in invalid  # 'invalid'
    assert 2 in invalid  # 空字符串
    assert 3 in invalid  # NaN

    # 允许空值
    invalid = validate_smiles_column(df, 0, allow_empty=True)
    assert 1 in invalid  # 'invalid'
    assert 2 not in invalid  # 空字符串允许
    assert 3 not in invalid  # NaN允许

def test_validate_product_column():
    """测试产物列验证(严格单SMILES)"""
    df = pd.DataFrame({'col': [
        'CCO',           # 合法
        'CCO.CCC',       # 非法: 含点号
        'invalid',       # 非法: 非法SMILES
        '',              # 非法: 空
        'c1ccccc1;CC'    # 非法: 含分号
    ]})
    invalid = validate_product_column(df, 0)
    assert invalid == {1, 2, 3, 4}

def test_column_name_uniqueness():
    """测试列名称唯一性"""
    used_names = {'yield', 'reactant'}

    # 尝试添加重复名称
    new_name = 'yield'
    assert new_name in used_names  # 应该被拒绝

    # 尝试添加新名称
    new_name = 'product'
    assert new_name not in used_names
    used_names.add(new_name)
    assert 'product' in used_names

def test_suggest_others_name():
    """测试智能命名建议"""
    assert suggest_others_name('溶剂类型') == 'solvent'
    assert suggest_others_name('Catalyst_name') == 'catalyst'
    assert suggest_others_name('未知列') == '未知列'
```

**手动验证检查项:**

1. 选择不存在的列序号 → 提示错误
2. 输入含中文的列名称 → 提示错误
3. 输入重复的列名称 → 提示错误并要求重新输入
4. 标签列选择包含非数值的列 → 正确识别非法行
5. 产物列包含多个SMILES(用`.`分隔) → 正确识别为非法行
6. 其他组分列包含空值 → 不被标记为非法行
7. 条件数值列包含空值 → 不被标记为非法行

#### 风险提示

1. **SMILES分隔符多样性:** 目前支持`.` ` ` `;` `,`四种分隔符,但可能存在其他分隔符(如`|` `tab`),需扩展正则
2. **列名称语义冲突:** 用户可能将不同角色的列命名为相似名称(如`reactant_1`和`reactant-1`),虽然格式合法但语义易混淆
3. **大数据集性能:** 对每列逐行验证SMILES,如果数据集超过10万行,可能耗时较长,考虑添加进度条
4. **RDKit警告信息:** `Chem.MolFromSmiles()`对非法SMILES会输出警告信息到stderr,可能干扰用户体验,需捕获并静默

---

### 步骤3: 生成规范数据集与非法输入排除报告

#### 目标说明

完成步骤2的逐列声明后,或从列映射文件加载配置后,执行以下三项任务:

**3.1 生成非法输入排除报告** (如果有非法输入)
- Markdown格式
- 两部分内容:
  1. 哪些行被排除,分别在哪个/哪些步骤出现非法值
  2. 被排除行的详细信息(表格形式,非法值加粗)

**3.2 生成列映射说明表**
- CSV格式,3列: `origin_name`, `role`, `name`
- 记录所有声明的列

**3.3 生成规范数据集**
- CSV格式,跳过所有非法行
- 列顺序: `reactant-1, reactant-2, ..., others-1, others-2, ..., condition-1, ..., product, label`
- 每单元格仅含单个SMILES或单个数值
- SMILES列中的多分子用`.`分隔后拆分到不同列

**特殊处理:**
- 如果合法性检查全部通过(无非法行),则跳过步骤3.1和3.3,因为无需重复生成一模一样的规范数据集

#### 具体操作

**3.1 生成非法输入排除报告**

```python
def step3_1_generate_invalid_report(df, all_column_configs, project_folder, project_name):
    """
    生成非法输入排除报告(Markdown格式)

    Args:
        df: 原始数据集
        all_column_configs: 所有列的配置列表(每个元素是dict,包含invalid_rows)
        project_folder: 项目文件夹路径
        project_name: 项目名称

    Returns:
        str: 报告文件路径(如果生成),否则返回None
    """
    # 汇总所有非法行
    all_invalid_rows = set()
    for config in all_column_configs:
        all_invalid_rows.update(config['invalid_rows'])

    if not all_invalid_rows:
        print("\n✓ 无非法输入,跳过报告生成")
        return None

    print(f"\n生成非法输入排除报告... 共 {len(all_invalid_rows)} 行")

    report_path = os.path.join(project_folder, f"{project_name}_非法输入排除报告.md")

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"# {project_name} 非法输入排除报告\n\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("---\n\n")

        # 第一部分: 汇总信息
        f.write("## 一、排除行汇总\n\n")
        f.write(f"**总计排除行数:** {len(all_invalid_rows)}\n\n")

        # 按行索引排序
        sorted_invalid_rows = sorted(all_invalid_rows)

        # 为每行标注在哪些步骤出现非法值
        row_error_map = {}  # {row_idx: [column_configs]}
        for row_idx in sorted_invalid_rows:
            row_error_map[row_idx] = []
            for config in all_column_configs:
                if row_idx in config['invalid_rows']:
                    row_error_map[row_idx].append(config)

        f.write("| 行号 | 出现非法值的列 |\n")
        f.write("|------|----------------|\n")
        for row_idx in sorted_invalid_rows:
            error_cols = [cfg['origin_name'] for cfg in row_error_map[row_idx]]
            f.write(f"| {row_idx} | {', '.join(error_cols)} |\n")

        f.write("\n---\n\n")

        # 第二部分: 逐行详细信息
        f.write("## 二、排除行详细信息\n\n")

        for row_idx in sorted_invalid_rows:
            f.write(f"### 行 {row_idx}\n\n")

            # 提取该行数据
            row_data = df.iloc[row_idx]

            # 生成表格
            f.write("| 列名 | 值 |\n")
            f.write("|------|----|\n")

            for config in all_column_configs:
                col_name = config['origin_name']
                col_value = row_data[col_name]

                # 如果该列在此行非法,加粗
                if row_idx in config['invalid_rows']:
                    col_value_str = f"**{col_value}**"
                else:
                    col_value_str = str(col_value)

                f.write(f"| {col_name} | {col_value_str} |\n")

            # 标注非法值位置
            error_cols = [cfg['origin_name'] for cfg in row_error_map[row_idx]]
            f.write(f"\n**非法值所在列:** {', '.join(error_cols)}\n\n")
            f.write("---\n\n")

    print(f"✓ 报告已生成: {report_path}")
    return report_path
```

**3.2 生成列映射说明表**

```python
def step3_2_generate_column_mapping(all_column_configs, project_folder, project_name):
    """
    生成列映射说明表(CSV格式)

    Args:
        all_column_configs: 所有列的配置列表
        project_folder: 项目文件夹路径
        project_name: 项目名称

    Returns:
        str: 列映射文件路径
    """
    print("\n生成列映射说明表...")

    mapping_path = os.path.join(project_folder, f"{project_name}_列映射.csv")

    mapping_data = []
    for config in all_column_configs:
        mapping_data.append({
            'origin_name': config['origin_name'],
            'role': config['role'],
            'name': config['name']
        })

    mapping_df = pd.DataFrame(mapping_data)
    mapping_df.to_csv(mapping_path, index=False, encoding='utf-8')

    print(f"✓ 列映射表已生成: {mapping_path}")
    print("\n列映射内容预览:")
    print(mapping_df.to_string(index=False))

    return mapping_path
```

**3.3 生成规范数据集**

```python
def step3_3_generate_normalized_dataset(df, all_column_configs, project_folder, project_name):
    """
    生成规范数据集

    关键步骤:
    1. 跳过所有非法行
    2. 按固定顺序排列列: reactant → others → condition → product → label
    3. 拆分包含多个SMILES的单元格(用.分隔)到多列
    4. 每单元格仅含单个SMILES或单个数值

    Args:
        df: 原始数据集
        all_column_configs: 所有列的配置列表
        project_folder: 项目文件夹路径
        project_name: 项目名称

    Returns:
        str: 规范数据集文件路径
    """
    print("\n生成规范数据集...")

    # 汇总所有非法行
    all_invalid_rows = set()
    for config in all_column_configs:
        all_invalid_rows.update(config['invalid_rows'])

    if all_invalid_rows:
        print(f"  跳过 {len(all_invalid_rows)} 个非法行")

    # 有效行
    valid_row_indices = [i for i in range(len(df)) if i not in all_invalid_rows]

    # 按角色分组列配置
    role_groups = {
        'reactant': [],
        'others': [],
        'condition': [],
        'product': [],
        'label': []
    }

    for config in all_column_configs:
        role_groups[config['role']].append(config)

    # 构建规范数据集的列
    normalized_rows = []

    for row_idx in valid_row_indices:
        row_data = df.iloc[row_idx]
        normalized_row = {}

        # 1. 处理reactant列(可能需要拆分)
        reactant_smiles_list = []
        for config in role_groups['reactant']:
            col_name = config['origin_name']
            value = row_data[col_name]

            if pd.notna(value) and str(value).strip():
                # 拆分多个SMILES
                smiles_parts = split_smiles(str(value))
                reactant_smiles_list.extend(smiles_parts)

        # 确定reactant列数(取所有行中最多的)
        # 这里先简化处理,按当前行的reactant数量填充
        for i, smiles in enumerate(reactant_smiles_list):
            normalized_row[f'reactant-{i+1}'] = smiles

        # 2. 处理others列(同样可能需要拆分)
        for config in role_groups['others']:
            col_name = config['origin_name']
            new_name = config['name']
            value = row_data[col_name]

            if pd.notna(value) and str(value).strip():
                smiles_parts = split_smiles(str(value))
                # 如果拆分成多个,需要创建多列
                if len(smiles_parts) == 1:
                    normalized_row[new_name] = smiles_parts[0]
                else:
                    for i, smiles in enumerate(smiles_parts):
                        normalized_row[f'{new_name}-{i+1}'] = smiles
            else:
                normalized_row[new_name] = ''

        # 3. 处理condition列(数值,直接复制)
        for config in role_groups['condition']:
            col_name = config['origin_name']
            new_name = config['name']
            value = row_data[col_name]
            normalized_row[new_name] = value if pd.notna(value) else ''

        # 4. 处理product列(单SMILES,直接复制)
        for config in role_groups['product']:
            col_name = config['origin_name']
            new_name = config['name']
            value = row_data[col_name]
            normalized_row[new_name] = str(value).strip() if pd.notna(value) else ''

        # 5. 处理label列(数值,直接复制)
        for config in role_groups['label']:
            col_name = config['origin_name']
            new_name = config['name']
            value = row_data[col_name]
            normalized_row[new_name] = value if pd.notna(value) else ''

        normalized_rows.append(normalized_row)

    # 转换为DataFrame
    normalized_df = pd.DataFrame(normalized_rows)

    # 确定最终列顺序(需要重新扫描所有行,确定reactant/others的最大列数)
    # 这里简化处理,直接使用当前列顺序
    # 实际应该先扫描一遍确定每类SMILES的最大数量

    # 保存
    normalized_path = os.path.join(project_folder, f"{project_name}_规范数据集.csv")
    normalized_df.to_csv(normalized_path, index=False, encoding='utf-8')

    print(f"✓ 规范数据集已生成: {normalized_path}")
    print(f"  有效行数: {len(normalized_df)}")
    print(f"  总列数: {len(normalized_df.columns)}")

    return normalized_path

def split_smiles(smiles_str):
    """
    拆分包含多个SMILES的字符串

    支持的分隔符: . 空格 ; ,

    Returns:
        list: SMILES列表
    """
    separators = ['.', ' ', ';', ',']

    for sep in separators:
        if sep in smiles_str:
            return [s.strip() for s in smiles_str.split(sep) if s.strip()]

    # 无分隔符,单个SMILES
    return [smiles_str.strip()]
```

**改进版3.3: 两次扫描确定列数**

```python
def step3_3_generate_normalized_dataset_v2(df, all_column_configs, project_folder, project_name):
    """
    生成规范数据集(改进版: 两次扫描)

    第一次扫描: 确定每类SMILES的最大列数
    第二次扫描: 填充数据
    """
    print("\n生成规范数据集...")

    # 汇总所有非法行
    all_invalid_rows = set()
    for config in all_column_configs:
        all_invalid_rows.update(config['invalid_rows'])

    valid_row_indices = [i for i in range(len(df)) if i not in all_invalid_rows]

    # 按角色分组
    role_groups = {
        'reactant': [],
        'others': [],
        'condition': [],
        'product': [],
        'label': []
    }

    for config in all_column_configs:
        role_groups[config['role']].append(config)

    # 第一次扫描: 确定最大列数
    max_reactant_count = 0
    max_others_count = {}  # {others_name: max_count}

    for row_idx in valid_row_indices:
        row_data = df.iloc[row_idx]

        # 统计reactant
        reactant_count = 0
        for config in role_groups['reactant']:
            col_name = config['origin_name']
            value = row_data[col_name]
            if pd.notna(value) and str(value).strip():
                smiles_parts = split_smiles(str(value))
                reactant_count += len(smiles_parts)
        max_reactant_count = max(max_reactant_count, reactant_count)

        # 统计others
        for config in role_groups['others']:
            col_name = config['origin_name']
            new_name = config['name']
            value = row_data[col_name]
            if pd.notna(value) and str(value).strip():
                smiles_parts = split_smiles(str(value))
                if new_name not in max_others_count:
                    max_others_count[new_name] = 0
                max_others_count[new_name] = max(max_others_count[new_name], len(smiles_parts))

    print(f"  最大reactant数量: {max_reactant_count}")
    for name, count in max_others_count.items():
        print(f"  最大{name}数量: {count}")

    # 构建最终列名列表(按顺序)
    final_columns = []

    # reactant列
    for i in range(max_reactant_count):
        final_columns.append(f'reactant-{i+1}')

    # others列
    for config in role_groups['others']:
        name = config['name']
        if name in max_others_count:
            count = max_others_count[name]
            if count == 1:
                final_columns.append(name)
            else:
                for i in range(count):
                    final_columns.append(f'{name}-{i+1}')

    # condition列
    for config in role_groups['condition']:
        final_columns.append(config['name'])

    # product列
    for config in role_groups['product']:
        final_columns.append(config['name'])

    # label列
    for config in role_groups['label']:
        final_columns.append(config['name'])

    # 第二次扫描: 填充数据
    normalized_rows = []

    for row_idx in valid_row_indices:
        row_data = df.iloc[row_idx]
        normalized_row = {col: '' for col in final_columns}  # 初始化为空字符串

        # 填充reactant
        reactant_smiles_list = []
        for config in role_groups['reactant']:
            col_name = config['origin_name']
            value = row_data[col_name]
            if pd.notna(value) and str(value).strip():
                smiles_parts = split_smiles(str(value))
                reactant_smiles_list.extend(smiles_parts)

        for i, smiles in enumerate(reactant_smiles_list):
            if i < max_reactant_count:
                normalized_row[f'reactant-{i+1}'] = smiles

        # 填充others
        for config in role_groups['others']:
            col_name = config['origin_name']
            new_name = config['name']
            value = row_data[col_name]

            if pd.notna(value) and str(value).strip():
                smiles_parts = split_smiles(str(value))
                if len(smiles_parts) == 1:
                    normalized_row[new_name] = smiles_parts[0]
                else:
                    for i, smiles in enumerate(smiles_parts):
                        normalized_row[f'{new_name}-{i+1}'] = smiles

        # 填充condition
        for config in role_groups['condition']:
            col_name = config['origin_name']
            new_name = config['name']
            value = row_data[col_name]
            normalized_row[new_name] = value if pd.notna(value) else ''

        # 填充product
        for config in role_groups['product']:
            col_name = config['origin_name']
            new_name = config['name']
            value = row_data[col_name]
            normalized_row[new_name] = str(value).strip() if pd.notna(value) else ''

        # 填充label
        for config in role_groups['label']:
            col_name = config['origin_name']
            new_name = config['name']
            value = row_data[col_name]
            normalized_row[new_name] = value if pd.notna(value) else ''

        normalized_rows.append(normalized_row)

    # 转换为DataFrame(按final_columns顺序)
    normalized_df = pd.DataFrame(normalized_rows, columns=final_columns)

    # 保存
    normalized_path = os.path.join(project_folder, f"{project_name}_规范数据集.csv")
    normalized_df.to_csv(normalized_path, index=False, encoding='utf-8')

    print(f"✓ 规范数据集已生成: {normalized_path}")
    print(f"  有效行数: {len(normalized_df)}")
    print(f"  总列数: {len(normalized_df.columns)}")
    print("\n列名预览:")
    print(f"  {', '.join(final_columns[:10])}{'...' if len(final_columns) > 10 else ''}")

    return normalized_path
```

#### 验证方法

**单元测试:**

```python
# tests/test_step3.py
def test_split_smiles():
    """测试SMILES拆分"""
    assert split_smiles('CCO.CCC') == ['CCO', 'CCC']
    assert split_smiles('CCO;CCC;CCCC') == ['CCO', 'CCC', 'CCCC']
    assert split_smiles('CCO CCC') == ['CCO', 'CCC']
    assert split_smiles('CCO,CCC') == ['CCO', 'CCC']
    assert split_smiles('CCO') == ['CCO']

def test_normalized_dataset_structure():
    """测试规范数据集结构"""
    # 模拟数据
    df = pd.DataFrame({
        'r1': ['CCO.CCC', 'CCCC'],
        'r2': ['CC', 'CCC'],
        'p': ['CCCCC', 'CCCCCC'],
        'y': [0.85, 0.90]
    })

    all_column_configs = [
        {'origin_name': 'r1', 'origin_idx': 0, 'role': 'reactant', 'name': 'reactant', 'invalid_rows': set()},
        {'origin_name': 'r2', 'origin_idx': 1, 'role': 'reactant', 'name': 'reactant', 'invalid_rows': set()},
        {'origin_name': 'p', 'origin_idx': 2, 'role': 'product', 'name': 'product', 'invalid_rows': set()},
        {'origin_name': 'y', 'origin_idx': 3, 'role': 'label', 'name': 'yield', 'invalid_rows': set()},
    ]

    # 生成规范数据集(使用临时文件夹)
    with tempfile.TemporaryDirectory() as tmpdir:
        normalized_path = step3_3_generate_normalized_dataset_v2(
            df, all_column_configs, tmpdir, 'test_project'
        )

        # 读取并验证
        normalized_df = pd.read_csv(normalized_path)

        # 第一行应该有3个reactant(CCO, CCC, CC)
        assert normalized_df.loc[0, 'reactant-1'] == 'CCO'
        assert normalized_df.loc[0, 'reactant-2'] == 'CCC'
        assert normalized_df.loc[0, 'reactant-3'] == 'CC'

        # 第二行应该有2个reactant(CCCC, CCC)
        assert normalized_df.loc[1, 'reactant-1'] == 'CCCC'
        assert normalized_df.loc[1, 'reactant-2'] == 'CCC'
        assert normalized_df.loc[1, 'reactant-3'] == ''  # 空

        # product和yield列应该存在
        assert 'product' in normalized_df.columns
        assert 'yield' in normalized_df.columns
```

**手动验证检查项:**

1. 非法输入报告正确列出所有非法行
2. 非法行的表格中,非法值被加粗显示
3. 列映射表正确记录所有列的`origin_name`, `role`, `name`
4. 规范数据集正确跳过所有非法行
5. 规范数据集中,含多个SMILES的单元格被正确拆分到多列
6. 规范数据集列顺序符合规范: reactant → others → condition → product → label
7. 如果无非法行,步骤3.1和3.3被跳过

#### 风险提示

1. **SMILES拆分歧义:** 某些SMILES本身含有`.`(如离子对),可能被错误拆分,需要RDKit二次验证
2. **列数过多:** 如果某行的reactant数量超过50个,会导致规范数据集极度稀疏,需在报告中提示用户
3. **内存消耗:** 第一次扫描需要遍历所有有效行两次,如果数据集超过100万行,可能内存压力大
4. **文件编码:** Markdown报告中可能包含特殊字符,需确保使用UTF-8编码保存

---

### 步骤4: 指定描述符

#### 目标说明

用户选择用于建模的描述符,并配置每个描述符的嵌入方式:

- **横向拼接**(morgan、atmomaccs、rdkit2d、fisd、molmetalm): 默认reactant-others-product顺序,用户可调整
- **逐点加和**(maf): 不存在拼接顺序,用户可选择参与嵌入的列
- **固定反应模式**(DRFP): 固定reactant→product,用户可选择将某些others列加入reactant

**补充说明:**
- 所有顺序相关的描述符都有默认顺序(reactant-others-product)
- others下可能有多个标签,默认以用户输入的顺序拼接,用户可单独调整

#### 具体操作

**4.1 展示描述符列表**

```python
def step4_select_descriptors():
    """
    展示所有可用描述符,让用户选择

    Returns:
        list: 选中的描述符名称列表
    """
    print("\n" + "=" * 60)
    print("步骤4: 指定描述符")
    print("=" * 60)

    descriptors = {
        'morgan': '横向拼接,可编辑参与列和拼接顺序',
        'atmomaccs': '横向拼接,可编辑参与列和拼接顺序',
        'rdkit2d': '横向拼接,可编辑参与列和拼接顺序',
        'fisd': '横向拼接,可编辑参与列和拼接顺序',
        'molmetalm': '横向拼接,可编辑参与列和拼接顺序',
        'maf': '逐点加和,可编辑参与列',
        'drfp': '固定反应模式(reactant→product),可编辑额外加入反应物的others列'
    }

    print("\n可用描述符:")
    for idx, (name, desc) in enumerate(descriptors.items(), 1):
        print(f"  [{idx}] {name}: {desc}")

    print("\n请选择描述符(输入序号,用逗号分隔,如: 1,2,5)")
    print("或输入'all'选择全部描述符")

    while True:
        user_input = input("描述符序号: ").strip()

        if user_input.lower() == 'all':
            selected = list(descriptors.keys())
            break

        try:
            indices = [int(x.strip()) for x in user_input.split(',')]
            if all(1 <= idx <= len(descriptors) for idx in indices):
                selected = [list(descriptors.keys())[i-1] for i in indices]
                break
            else:
                print("✗ 存在无效的序号")
        except ValueError:
            print("✗ 输入格式错误")

    print(f"\n✓ 已选择 {len(selected)} 个描述符: {', '.join(selected)}")
    return selected
```

**4.2 配置描述符嵌入方式**

```python
def step4_configure_descriptor(descriptor_name, all_column_configs):
    """
    配置单个描述符的嵌入方式

    Args:
        descriptor_name: 描述符名称
        all_column_configs: 所有列的配置列表(用于展示可选列)

    Returns:
        dict: {
            'descriptor': str,
            'mode': 'concat' | 'sum' | 'reaction',
            'columns': list,  # 参与嵌入的列(按顺序)
            'extra_reactants': list  # 仅DRFP使用,额外加入反应物的others列
        }
    """
    print(f"\n配置描述符: {descriptor_name}")

    # 获取所有SMILES列(reactant, others, product)
    smiles_columns = [
        cfg for cfg in all_column_configs
        if cfg['role'] in ['reactant', 'others', 'product']
    ]

    print("\n当前所有SMILES列:")
    for idx, cfg in enumerate(smiles_columns):
        print(f"  [{idx}] {cfg['name']} (角色: {cfg['role']})")

    if descriptor_name in ['morgan', 'atmomaccs', 'rdkit2d', 'fisd', 'molmetalm']:
        # 横向拼接模式
        print("\n该描述符为横向拼接模式")
        print("默认顺序: reactant → others → product")
        print("您可以:")
        print("  1. 使用默认顺序")
        print("  2. 自定义参与列和顺序(输入列序号,用逗号分隔)")

        choice = input("选择(1/2): ").strip()

        if choice == '1':
            # 使用默认顺序
            columns = []
            # 按角色顺序添加
            for role in ['reactant', 'others', 'product']:
                for cfg in all_column_configs:
                    if cfg['role'] == role:
                        columns.append(cfg['name'])
        else:
            # 自定义顺序
            print("\n请输入列序号(用逗号分隔,顺序即为拼接顺序):")
            while True:
                user_input = input("列序号: ").strip()
                try:
                    indices = [int(x.strip()) for x in user_input.split(',')]
                    if all(0 <= idx < len(smiles_columns) for idx in indices):
                        columns = [smiles_columns[i]['name'] for i in indices]
                        break
                    else:
                        print("✗ 存在无效的序号")
                except ValueError:
                    print("✗ 输入格式错误")

        print(f"✓ 拼接顺序: {' → '.join(columns)}")

        return {
            'descriptor': descriptor_name,
            'mode': 'concat',
            'columns': columns
        }

    elif descriptor_name == 'maf':
        # 逐点加和模式
        print("\n该描述符为逐点加和模式,不存在拼接顺序")
        print("请选择参与嵌入的列(输入列序号,用逗号分隔):")

        while True:
            user_input = input("列序号: ").strip()
            try:
                indices = [int(x.strip()) for x in user_input.split(',')]
                if all(0 <= idx < len(smiles_columns) for idx in indices):
                    columns = [smiles_columns[i]['name'] for i in indices]
                    break
                else:
                    print("✗ 存在无效的序号")
            except ValueError:
                print("✗ 输入格式错误")

        print(f"✓ 参与加和的列: {', '.join(columns)}")

        return {
            'descriptor': descriptor_name,
            'mode': 'sum',
            'columns': columns
        }

    elif descriptor_name == 'drfp':
        # 固定反应模式
        print("\n该描述符为固定反应模式: reactant → product")
        print("您可以选择将某些others列加入到reactant中(如催化剂)")

        # 列出所有others列
        others_columns = [cfg for cfg in all_column_configs if cfg['role'] == 'others']

        if not others_columns:
            print("  无可选的others列")
            extra_reactants = []
        else:
            print("\n可选的others列:")
            for idx, cfg in enumerate(others_columns):
                print(f"  [{idx}] {cfg['name']}")

            user_input = input("请输入要加入reactant的others列序号(用逗号分隔,直接回车跳过): ").strip()

            if user_input:
                try:
                    indices = [int(x.strip()) for x in user_input.split(',')]
                    extra_reactants = [others_columns[i]['name'] for i in indices if 0 <= i < len(others_columns)]
                except ValueError:
                    print("✗ 输入格式错误,跳过")
                    extra_reactants = []
            else:
                extra_reactants = []

        if extra_reactants:
            print(f"✓ 额外加入反应物的列: {', '.join(extra_reactants)}")
        else:
            print("✓ 使用默认reactant列")

        return {
            'descriptor': descriptor_name,
            'mode': 'reaction',
            'extra_reactants': extra_reactants
        }
```

#### 验证方法

**手动验证检查项:**

1. 选择描述符时输入无效序号 → 提示错误
2. 配置横向拼接描述符时,自定义顺序正确记录
3. 配置DRFP时,选择额外others列正确记录
4. 配置结果正确保存到配置文件

#### 风险提示

1. **用户误操作:** 用户可能不理解"拼接顺序"的含义,需要在界面上给出示例
2. **DRFP额外reactant语义:** 催化剂加入reactant在某些反应类型中不合适,需提醒用户根据实际情况选择

---

### 步骤5-8: 其他配置项(简化处理)

#### 步骤5: 指定建模模型

```python
def step5_select_models():
    """
    选择建模模型

    Returns:
        list: 选中的模型名称列表
    """
    print("\n" + "=" * 60)
    print("步骤5: 指定建模模型")
    print("=" * 60)

    models = ['XGBoost', 'Random Forest', 'SVM', 'AutoGluon', 'Neural Network']

    print("\n可用模型:")
    for idx, model in enumerate(models, 1):
        print(f"  [{idx}] {model}")

    print("\n请选择模型(输入序号,用逗号分隔,如: 1,2,4)")
    print("或输入'all'选择全部模型")

    while True:
        user_input = input("模型序号: ").strip()

        if user_input.lower() == 'all':
            selected = models
            break

        try:
            indices = [int(x.strip()) for x in user_input.split(',')]
            if all(1 <= idx <= len(models) for idx in indices):
                selected = [models[i-1] for i in indices]
                break
            else:
                print("✗ 存在无效的序号")
        except ValueError:
            print("✗ 输入格式错误")

    print(f"\n✓ 已选择 {len(selected)} 个模型: {', '.join(selected)}")
    return selected
```

#### 步骤6: 补充数据集信息

```python
def step6_dataset_metadata():
    """
    收集数据集元信息

    Returns:
        dict: {
            'repo_url': str,
            'doi': str,
            'notes': str
        }
    """
    print("\n" + "=" * 60)
    print("步骤6: 补充数据集信息")
    print("=" * 60)

    repo_url = input("项目地址(可选): ").strip()
    doi = input("文献DOI(可选): ").strip()
    notes = input("备注(可选): ").strip()

    metadata = {
        'repo_url': repo_url,
        'doi': doi,
        'notes': notes
    }

    print("\n✓ 元信息已记录")
    return metadata
```

#### 步骤7: 选择报告输出格式

```python
def step7_select_report_format():
    """
    选择报告输出格式

    Returns:
        list: 选中的格式列表
    """
    print("\n" + "=" * 60)
    print("步骤7: 选择报告输出格式")
    print("=" * 60)

    formats = ['Markdown', 'HTML', 'PDF', 'JSON']

    print("\n可用格式:")
    for idx, fmt in enumerate(formats, 1):
        print(f"  [{idx}] {fmt}")

    print("\n请选择输出格式(输入序号,用逗号分隔,如: 1,3)")

    while True:
        user_input = input("格式序号: ").strip()

        try:
            indices = [int(x.strip()) for x in user_input.split(',')]
            if all(1 <= idx <= len(formats) for idx in indices):
                selected = [formats[i-1] for i in indices]
                break
            else:
                print("✗ 存在无效的序号")
        except ValueError:
            print("✗ 输入格式错误")

    print(f"\n✓ 已选择输出格式: {', '.join(selected)}")
    return selected
```

#### 步骤8: 确认命令行并生成修复后数据集

```python
def step8_confirm_and_generate():
    """
    展示最终配置,生成修复后数据集(可选)
    """
    print("\n" + "=" * 60)
    print("步骤8: 确认配置并生成修复后数据集")
    print("=" * 60)

    print("\n是否生成修复后数据集(删除非法行)?")
    print("  1. 是,生成修复后数据集")
    print("  2. 否,仅保留原始数据集")

    choice = input("选择(1/2): ").strip()

    if choice == '1':
        print("\n✓ 将生成修复后数据集(删除非法行)")
        generate_fixed = True
    else:
        print("\n✓ 不生成修复后数据集")
        generate_fixed = False

    return generate_fixed

def generate_fixed_dataset(df, all_invalid_rows, project_folder, project_name):
    """
    生成修复后数据集(删除非法行)

    Args:
        df: 原始数据集
        all_invalid_rows: 所有非法行的索引集合
        project_folder: 项目文件夹
        project_name: 项目名称

    Returns:
        str: 修复后数据集文件路径
    """
    print("\n生成修复后数据集...")

    valid_row_indices = [i for i in range(len(df)) if i not in all_invalid_rows]
    fixed_df = df.iloc[valid_row_indices]

    fixed_path = os.path.join(project_folder, f"{project_name}_修复后数据集.csv")
    fixed_df.to_csv(fixed_path, index=False, encoding='utf-8')

    print(f"✓ 修复后数据集已生成: {fixed_path}")
    print(f"  原始行数: {len(df)}")
    print(f"  删除行数: {len(all_invalid_rows)}")
    print(f"  剩余行数: {len(fixed_df)}")

    return fixed_path
```

#### 验证方法

**手动验证检查项:**

1. 选择模型时输入无效序号 → 提示错误
2. 数据集元信息正确保存
3. 报告格式选择正确记录
4. 修复后数据集正确删除所有非法行

---

### 17.5 Q&A 记录

### 步骤1: 指定初始数据集、列映射文件、项目名称和项目文件夹位置

**Q: 如果提供了列映射文件,还需要手动声明列吗?**
**A:** 不需要。如果提供了列映射文件,程序将跳过步骤2(逐列声明),直接使用映射文件中的配置进行合法性检验,并正常进入步骤3生成报告(如有非法值)。如果合法性检查全部通过,则跳过步骤3,因为无需重复生成一模一样的规范数据集。

---

### 步骤2: 逐列声明列角色和名称

**Q: 为什么标签列不允许空值,但条件列允许?**
**A:** 标签列(如yield)是建模的目标值,必须有值才能训练模型。条件列(如temperature)是辅助特征,缺失值可以通过填充策略(如均值填充)或直接作为特殊标记处理,不影响建模流程。

**Q: 产物列为什么必须是单个SMILES?**
**A:** 这是目前采用的严格定义。产物列通常表示主产物,一个反应只有一个主产物。如果需要支持多产物,可以在文档中说明宽松定义,但需要修改后续描述符化逻辑。

**Q: 其他组分列为什么允许空值?**
**A:** 某些反应可能不需要某种组分(如无需催化剂的反应),空值表示"不存在该组分",这在化学上是合理的。

**Q: 列名称为什么禁止中文?**
**A:** 后续建模代码可能在不同环境(Linux/Windows/云端)运行,中文列名可能导致编码问题。强制英文列名可确保跨平台兼容性。

**Q: 为什么要维护`used_names`集合?**
**A:** 避免列名重复,导致后续数据处理时无法区分不同列。列名唯一性是数据规范化的基本要求。

---

### 步骤3: 生成规范数据集与非法输入排除报告

**Q: 为什么非法行仍参与后续检验?**
**A:** 一行数据可能在多个列都有非法值,完整检验所有列可以给用户提供完整的错误信息,便于修复数据源。如果检测到第一个非法值就跳过该行,用户需要多次运行才能发现所有错误。

**Q: 为什么全部合法时跳过步骤3.3?**
**A:** 如果没有非法行,规范数据集与原始数据集(仅列名和顺序不同)在内容上一模一样,重复生成没有意义。用户可以直接使用原始数据集配合列映射文件进行后续操作。

**Q: 规范数据集的列顺序为什么固定?**
**A:** 固定顺序便于后续描述符化脚本统一处理,不需要每次都读取列映射文件重新定位列位置。

---

### 步骤4: 指定描述符

**Q: 为什么所有顺序相关描述符默认都是reactant-others-product?**
**A:** 这是化学反应建模的常见约定,反应物→其他组分→产物的顺序符合化学反应的自然逻辑。用户可以根据实际需要调整。

**Q: DRFP为什么是固定模式?**
**A:** DRFP(Differential Reaction Fingerprint)是专门设计用于反应差异表征的描述符,其核心思想是reactant→product的转换,不适合任意调整顺序。

---

### 通用问题

**Q: 为什么不采纳配置文件导出功能?**
**A:** 不同数据集的格式大相径庭,列名、列数、数据类型都可能完全不同,导出的配置文件难以在其他数据集上复用。即使同一研究组的不同实验,列名也可能不一致。

**Q: 为什么不在规范数据集中保留原始列?**
**A:** 这会导致数据冗余,且混淆"原始列"和"规范列"的语义。用户如需查看原始格式,可直接打开原始数据集文件。

**Q: 为什么一键修复脚本只保留删除行这一手段?**
**A:** 自动修复非法SMILES(如尝试修正拼写错误)或非法数值(如自动填充)存在语义风险,可能改变原始数据的化学含义。删除非法行是最安全的修复方式。如需更复杂的修复策略,应由用户在原始数据集上手动处理后重新导入。

---

### 17.6 下一步行动建议

1. **实现步骤1-2的核心逻辑**(预计3.5小时):
   - 创建`dataset_input_wizard.py`主脚本
   - 实现步骤1的文件路径收集和验证
   - 实现步骤2的逐列声明逻辑(2.1-2.6)
   - 实现所有合法性检验函数

2. **实现步骤3的报告和数据集生成**(预计2.5小时):
   - 实现非法输入排除报告生成(Markdown格式)
   - 实现列映射说明表生成(CSV格式)
   - 实现规范数据集生成(两次扫描版本)

3. **实现步骤4-8的其他配置**(预计1小时):
   - 实现描述符选择和配置界面
   - 实现模型选择界面
   - 实现元信息收集
   - 实现报告格式选择
   - 实现修复后数据集生成

4. **测试和文档**(预计2小时):
   - 编写单元测试
   - 手动验证所有边界情况
   - 编写用户使用手册
   - 更新README

5. **Git提交**:
   - 完成开发后,执行`git add 数据集输入流程重构构建计划书.md`
   - 提交: `git commit -m "docs: 创建数据集输入流程重构构建计划书"`

**总预计工时:** 11小时

**优先级排序:**
1. 步骤1-2(核心输入逻辑,最高优先级)
2. 步骤3(输出规范数据集,次高优先级)
3. 测试(确保稳定性)
4. 步骤4-8(可在基础功能完成后逐步添加)

---

### 17.7 附录

### 附录A: 国内镜像源配置

#### pip镜像

```powershell
# 临时使用
pip install <包名> -i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn

# 永久配置
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

备用镜像:
- 阿里云: `https://mirrors.aliyun.com/pypi/simple`
- 中科大: `https://pypi.mirrors.ustc.edu.cn/simple`

#### conda镜像

```powershell
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
conda config --set show_channel_urls yes
```

### 附录B: 常见错误处理

#### 错误1: RDKit无法解析SMILES

**现象:** `Chem.MolFromSmiles()`返回`None`

**原因:**
1. SMILES字符串含有非法字符
2. SMILES语法错误(如括号不匹配)
3. 立体化学标记错误

**解决:**
1. 检查SMILES字符串是否含有特殊字符(如中文、空格)
2. 使用RDKit的`SanitizeMol()`尝试修复
3. 如无法修复,标记为非法行

#### 错误2: Pandas读取CSV失败

**现象:** `UnicodeDecodeError`

**原因:** CSV文件编码不是UTF-8

**解决:**
```python
# 尝试自动检测编码
import chardet

with open(csv_path, 'rb') as f:
    result = chardet.detect(f.read(100000))
    encoding = result['encoding']

df = pd.read_csv(csv_path, encoding=encoding)
```

#### 错误3: 列名重复

**现象:** Pandas自动添加`.1` `.2`后缀

**原因:** 原始CSV含有重复列名

**解决:**
1. 在步骤2.1展示列时,提醒用户注意重复列名
2. 要求用户在声明时使用唯一的新列名

---

## 十八、双入口重构计划（YONOD v2.0 架构升级）

### 18.1 改造目标

将 YONOD 项目的两个入口文件进行职责分离和协作优化：
- **dataset_input_wizard.py**：专注于数据准备（8步交互式向导），完成后自动调用 yonod.py
- **yonod.py**：专注于建模流程（删除交互式向导，纯CLI模式），直接读取向导生成的配置文件

### 18.2 支持的使用场景

#### 场景A：首次使用（完整向导流程）
```bash
python dataset_input_wizard.py
# → 8步向导收集配置
# → 自动调用 yonod.py 进行建模
```

#### 场景B：已有配置（直接建模）
```bash
python yonod.py --config project-folder/yonod_config.json
```

#### 场景C：传统CLI（向后兼容）
```bash
python yonod.py --csv data.csv --label-col yield --smiles-cols R1 R2 --descriptors morgan --models xgb
```

### 18.3 配置文件格式

**文件路径**: `{project_folder}/{project_name}_yonod_config.json`

```json
{
  "version": "1.0",
  "project_name": "amide_coupling_test",
  "dataset_path": "/absolute/path/to/normalized_dataset.csv",
  "column_mapping_path": "/absolute/path/to/column_mapping.csv",
  "descriptors": [
    {
      "descriptor": "morgan",
      "mode": "concat",
      "columns": ["reactant-1", "reactant-2", "others-1", "product-1"],
      "extra_reactants": []
    },
    {
      "descriptor": "drfp",
      "mode": "reaction",
      "columns": ["reactant-1", "reactant-2", "product-1"],
      "extra_reactants": ["others-1"]
    }
  ],
  "models": ["XGBoost", "Random Forest", "AutoGluon"],
  "metadata": {
    "repo_url": "https://github.com/example/dataset",
    "doi": "10.1000/example.doi",
    "notes": "Amide coupling reactions with yields"
  },
  "report_formats": ["HTML", "Markdown"],
  "column_roles": {
    "label": "yield",
    "reactants": ["reactant-1", "reactant-2"],
    "products": ["product-1"],
    "others": ["others-1", "others-2"],
    "conditions": ["temperature"]
  }
}
```

**字段说明**:
- `version`：配置格式版本号（用于未来兼容性）
- `dataset_path`：规范化数据集的绝对路径
- `column_mapping_path`：列映射文件的绝对路径
- `descriptors`：描述符配置数组
  - `mode`: `concat`（横向拼接）| `sum`（逐点加和）| `reaction`（反应模式，DRFP专用）
  - `columns`: 参与该描述符计算的列名
  - `extra_reactants`: DRFP 特有字段，额外加入反应物的 others 列
- `models`：模型名称列表
- `metadata`：数据集元信息
- `report_formats`：报告输出格式
- `column_roles`：列角色分类

### 18.4 实施步骤

#### 步骤1：配置文件生成（dataset_input_wizard.py）

在 `dataset_input_wizard.py` 中新增 `save_config_file()` 函数：
- 将向导步骤4-7的配置保存为 JSON
- 使用绝对路径确保跨目录兼容性
- 输出文件：`{project_name}_yonod_config.json`

#### 步骤2：自动调用建模（dataset_input_wizard.py）

新增 `step9_auto_launch_modeling()` 函数：
- 询问用户 "是否立即启动建模流程? [Y/n]"
- 使用 subprocess.run() 调用 `yonod.py --config <path>`
- 包含完善的错误处理

#### 步骤3：配置文件读取（yonod.py）

在 yonod.py 中新增：
- `--config` 参数
- `load_config_from_json()` 函数
- `config_to_args()` 函数
- `_MODEL_NAME_MAP` 模型名称映射

#### 步骤4：删除冗余代码（yonod.py）

删除约570行交互式向导代码：
- `wizard()` 函数及辅助函数
- 正则常量
- 新增 `_print_usage()` 友好提示

### 18.5 文件修改统计

| 文件 | 新增行数 | 删除行数 | 净变化 |
|------|---------|---------|-------|
| dataset_input_wizard.py | +157 | 0 | +157 |
| yonod.py | +200 | -576 | -376 |
| CLAUDE.md | +50 | -10 | +40 |
| **总计** | **+407** | **-586** | **-179** |

### 18.6 验证方法

1. **场景A测试**：运行完整向导，选择自动启动建模
2. **场景B测试**：使用配置文件运行 `python yonod.py --config <path>`
3. **场景C测试**：使用传统CLI参数运行，确认向后兼容

### 18.7 实施状态

| 步骤 | 状态 | 说明 |
|------|------|------|
| 步骤1：配置文件生成 | ✅ 完成 | save_config_file() 已实现 |
| 步骤2：自动调用建模 | ✅ 完成 | step9_auto_launch_modeling() 已实现 |
| 步骤3：配置文件读取 | ✅ 完成 | --config 参数已实现 |
| 步骤4：删除冗余代码 | ✅ 完成 | wizard() 已删除，-376行 |
| 步骤5：扩展 feature_builder | ⏳ 待实现 | mode/target_columns 参数 |
| 步骤6：load_csv_with_mapping | ⏳ 待实现 | 优先使用列映射文件 |
| 步骤7：集成测试 | ⏳ 待实现 | 三种场景端到端测试 |

### 18.8 后续优化方向

1. **断点续传**：向导支持中途保存进度
2. **配置校验工具**：独立的配置文件验证脚本
3. **Web 界面**：基于 Streamlit 的可视化向导
4. **批量处理**：支持多数据集批量建模
5. **结果对比**：生成多次运行的对比报告

---

## 十九、文件夹整理与轻量重构计划（VJETHBKM 复现配套）

> 本节基于 `project-docs/goal.md` 中“文件夹整理目标补充”追加，属于现有计划书的增量更新。执行原则是：先盘点、再分类、后清理；先保证 `main.py`、`yonod.py`、`dataset/`、`WEIGHTS/` 等入口路径不被破坏，再考虑移动、归档或删除。

### 19.1 项目概述

#### 19.1.1 整理目标

本轮文件夹整理不是一次性大搬迁，而是为 `VJETHBKM` 复现和 YONOD benchmark 工具化建立清晰的科研工程边界：

- **代码区**：保留 `yonod/` 作为核心源码包，保留 `main.py` 与 `yonod.py` 作为现有运行入口。
- **数据区**：保留 `dataset/` 作为数据集目录，不重命名、不移动已被 README、示例配置或脚本引用的数据文件。
- **文献区**：保留 `docs/参考文献/` 作为 Zotero 文献、主文、SI、补充材料和人工整理资料的本地目录；若需要入库共享，应另写轻量索引或复现说明到 `project-docs/`。
- **实验产物区**：继续使用 `results/` 和 `cache/` 存放可再生成的运行输出与特征缓存；重要基线快照另行确认后再选择是否固化。
- **项目文档区**：继续使用 `project-docs/` 记录目标、计划、构建日志、教学说明和复现决策。
- **本地缓存区**：`.vs`、`.vscode`、`.pytest_cache`、`__pycache__` 等优先由 `.gitignore` 管理；只有确认不影响恢复后才删除。
- **验证材料区**：`_verify/` 不能默认视为垃圾目录，应先确认脚本用途、Git 跟踪状态、是否仍能解释历史修复，再决定保留、归档或删除。

#### 19.1.2 当前已知现状

只读检查显示当前项目根目录已有以下主要分区：

```text
YONOD/
├── main.py                    # 现有建模入口之一，被 yonod.py 和帮助信息引用
├── yonod.py                   # 现有统一入口，被 README 和计划书大量引用
├── yonod/                     # 核心源码包
├── dataset/                   # 已入库样例/研究数据
├── docs/                      # 本地文献与参考材料，当前被 .gitignore 忽略
├── project-docs/              # 目标、计划、日志、教学说明
├── WEIGHTS/                   # 本地模型权重，当前按子目录忽略
├── results/                   # 运行输出，当前被忽略
├── cache/                     # 运行缓存，当前被忽略
├── _verify/                   # 本地验证脚本，当前被忽略
├── .vs/ .vscode/              # IDE 配置/缓存，当前被忽略
├── .pytest_cache/ __pycache__/ # Python/测试缓存，当前被忽略
└── .gitignore
```

`.gitignore` 已覆盖 `.vs`、`.vscode`、`.pytest_cache`、`__pycache__`、`_verify`、`docs/`、`WEIGHTS/MolMetaLM-base/`、`WEIGHTS/FISD/`、`results/`、`cache/` 等目录。后续执行阶段的重点不是“马上改 ignore 规则”，而是确认这些规则是否符合论文复现与协作共享需求。

### 19.2 可行性分析

#### 19.2.1 技术可行性

本轮整理高度可行，因为目标以轻量治理为主，不要求立刻改源码导入路径或数据路径。只要遵循以下约束，就能把风险控制在较低水平：

- 不移动 `main.py`、`yonod.py`、`yonod/`、`dataset/`、`WEIGHTS/`。
- 不直接删除 `_verify/`、`docs/参考文献/`、权重目录或任何被引用的实验产物。
- 对缓存和 IDE 文件先做 Git 跟踪检查、引用检查和可再生成性判断。
- 对确需保留但不适合入库的材料，用 `project-docs/` 记录索引、来源和用途，而不是把大文件强行纳入 Git。

#### 19.2.2 主要风险与应对

| 风险 | 影响 | 应对策略 |
|---|---|---|
| 移动入口脚本导致 README、示例命令、配置文件失效 | 高 | 本轮不移动 `main.py`、`yonod.py`；若未来迁移，必须同步更新引用并做端到端烟测 |
| 删除 `_verify/` 后丢失历史修复依据 | 中 | 先逐项记录脚本用途、最后运行时间、是否仍可复现问题，再决定归档或删除 |
| `docs/` 被整体忽略导致文献整理成果无法共享 | 中 | PDF/SI 可继续本地保存，关键文献信息、引用、复现依据写入 `project-docs/` 的轻量 Markdown |
| `results/` 全部忽略导致基线结果不可追溯 | 中 | 可再生成结果继续忽略；论文/答辩用关键快照需单独确认固化位置与脱敏规则 |
| 批量删除缓存误删用户配置 | 中 | 删除前执行路径解析、Git 跟踪检查和用户确认；禁止按名称盲删 |

#### 19.2.3 工作量估算

| 阶段 | 预计耗时 | 产出 |
|---|---:|---|
| 盘点与分类 | 0.5 天 | 当前目录职责表、引用关系检查结果 |
| 忽略规则审计 | 0.5 天 | `.gitignore` 调整建议或确认无需调整 |
| 缓存/临时文件清理 | 0.5 天 | 可删除、应保留、需确认清单 |
| 文献与实验结果分区说明 | 0.5 天 | VJETHBKM 复现材料索引与结果快照策略 |
| 验证与提交 | 0.5 天 | 烟测记录、Git 状态确认、整理日志 |

### 19.3 技术选型

#### 19.3.1 工具与运行环境

- `git >= 2.40`：检查跟踪状态、差异、回滚点。
- `ripgrep >= 14.0`：快速检索路径引用。
- `python >= 3.9`：运行现有 YONOD 脚本与必要的路径检查脚本。
- `PowerShell >= 5.1`：Windows 本地文件检查与受控清理命令。

#### 19.3.2 不新增运行依赖

本轮是目录治理和文档化计划，不需要安装新的 Python 包、Node 包或系统工具。若当前环境没有 `rg`，可退回 PowerShell `Select-String`，但正式执行阶段仍建议安装 `ripgrep` 以减少漏检。

```powershell
# 1. 首先尝试确认本机已有工具
git --version
rg --version
python --version

# 2. 若 rg 不存在，可临时使用 PowerShell 原生命令替代
Get-ChildItem -Recurse -File | Select-String -Pattern "dataset/|WEIGHTS/|yonod.py|main.py"
```

### 19.4 开发计划

### 步骤19.1：建立目录盘点清单

#### 目标说明

先把当前项目目录按“代码 / 数据 / 文献 / 实验结果 / 项目文档 / 本地缓存 / 待确认验证材料”分类，形成后续整理的事实依据。这样可以避免只凭目录名判断用途。

#### 具体操作

执行只读检查，输出目录、Git 跟踪状态和忽略规则匹配情况：

```powershell
git status --short
git ls-files
git check-ignore -v .vs .vscode .pytest_cache __pycache__ _verify docs WEIGHTS/MolMetaLM-base WEIGHTS/FISD cache results
rg -n "main\.py|yonod\.py|dataset/|dataset\\|WEIGHTS/|WEIGHTS\\|docs/参考文献|_verify|results/|cache/" README.md project-docs yonod.py main.py example.json .gitignore
```

建议形成如下清单：

| 分类 | 当前目录/文件 | 默认处理 |
|---|---|---|
| 代码 | `yonod/`, `main.py`, `yonod.py` | 保留原位 |
| 数据 | `dataset/` | 保留原位，禁止未同步引用时重命名 |
| 文献 | `docs/参考文献/` | 本地保留；重要索引写入项目文档 |
| 权重 | `WEIGHTS/` | 本地保留，不入库 |
| 运行产物 | `results/`, `cache/` | 可再生成，默认忽略 |
| IDE/缓存 | `.vs`, `.vscode`, `.pytest_cache`, `__pycache__` | 确认可恢复后可删除 |
| 验证材料 | `_verify/` | 需逐项确认，不默认删除 |

#### 验证方法

- 清单中每个顶层目录都有分类和处理建议。
- `git ls-files` 能说明哪些文件已被 Git 跟踪。
- `git check-ignore -v` 能说明哪些目录由哪条 `.gitignore` 规则管理。
- `rg` 结果能定位关键路径引用，尤其是 `main.py`、`yonod.py`、`dataset/`、`WEIGHTS/`。

#### 风险提示

`git check-ignore` 在部分 Windows 环境可能提示全局 ignore 文件权限问题；只要本仓库 `.gitignore` 的匹配结果正常输出，不影响本轮判断。若命令完全失败，应改用 `git status --ignored --short` 补充确认。

---

### 步骤19.2：固化轻量目标结构

#### 目标说明

在不大搬迁的前提下，为团队建立一致的目录理解，使后续 `VJETHBKM` 复现、benchmark 实验和答辩材料都知道“该放哪里”。

#### 具体操作

推荐采用以下目标结构语义：

```text
YONOD/
├── yonod/                 # 核心源码包：描述符、模型、特征、报告等模块
├── main.py                # 建模执行入口：保持现状
├── yonod.py               # 交互/统一入口：保持现状
├── dataset/               # 数据集：保留被示例与 README 引用的文件名
├── docs/参考文献/          # 文献主文、SI、补充材料、本地笔记
├── project-docs/          # 项目目标、计划、日志、教学文档、复现决策
├── WEIGHTS/               # 本地权重，不随 Git 分发
├── results/               # 可再生成实验输出
├── cache/                 # 可再生成特征缓存
└── _verify/               # 临时/回归验证脚本，待确认后归档或删除
```

如果需要新增“可共享的实验结果快照”，优先采用轻量文本或小体积 CSV，并在执行前确认位置。例如：

```text
project-docs/
└── benchmark-snapshots/   # 可选：仅存可共享、脱敏、体积小的关键结果摘要
```

注意：该目录是否创建应交给后续构建阶段处理，本计划阶段只提出规则，不创建目录。

#### 验证方法

- README、示例命令和配置文件中的现有路径仍然成立。
- 团队成员能根据目标结构判断新文件应该放入哪个分区。
- 文献 PDF、大模型权重、缓存和完整运行结果不会被误认为必须提交到 Git。

#### 风险提示

`docs/` 当前被 `.gitignore` 整体忽略。如果未来希望共享文献清单，不能简单取消忽略整个 `docs/`，否则可能把 PDF、补充材料或版权受限内容纳入仓库。更稳妥的做法是：保留本地文献目录，另在 `project-docs/` 写可共享索引。

---

### 步骤19.3：审计 `.gitignore` 与版本管理边界

#### 目标说明

确认 Git 只跟踪应共享的源码、轻量数据、项目文档和配置，不误纳入 IDE 缓存、运行缓存、大模型权重或可能涉及版权的文献附件。

#### 具体操作

执行以下只读命令：

```powershell
git status --short
git status --ignored --short
git ls-files dataset docs WEIGHTS results cache _verify .vs .vscode .pytest_cache __pycache__
git check-ignore -v docs WEIGHTS/MolMetaLM-base WEIGHTS/FISD results cache _verify .vs .vscode .pytest_cache __pycache__
```

判断规则：

- `dataset/`：已入库的样例数据可保留；新增私有或大体积数据应先确认许可证与体积。
- `WEIGHTS/`：默认不入库；README 说明下载方式即可。
- `docs/参考文献/`：PDF/SI 本地保留；可共享的文献信息写到 `project-docs/`。
- `results/` 与 `cache/`：默认忽略；只有精选基线摘要在确认后才固化。
- `.vs`、`.vscode`、`.pytest_cache`、`__pycache__`：默认忽略；删除前确认不含用户自定义配置。
- `_verify/`：默认忽略但不默认删除；若保留价值高，可考虑迁移为正式测试，但这是后续构建任务。

#### 验证方法

- `git status --ignored --short` 能清楚显示被忽略的缓存/产物。
- `git ls-files` 中不出现大模型权重、IDE 缓存和运行缓存。
- 若发现误跟踪文件，应先记录清单，再由用户确认是否通过后续 builder 任务调整。

#### 风险提示

不要使用 `git add .`、`git add -A` 或全目录拖拽提交来整理仓库。YONOD 当前同时包含数据、权重、文献和运行产物，粗粒度提交很容易把不该入库的内容带进去。

---

### 步骤19.4：制定缓存与临时文件清理流程

#### 目标说明

满足用户选择的“确认无用后删除”策略，同时保留可恢复、可审计的判断依据。

#### 具体操作

删除前必须逐项检查四件事：

1. **用途**：这个目录/文件是 IDE 缓存、Python 缓存、测试缓存、运行结果，还是历史验证材料？
2. **Git 状态**：是否被 Git 跟踪？是否存在未提交修改？
3. **引用关系**：README、脚本、配置或计划书是否引用它？
4. **可再生成性**：删除后是否可通过运行脚本重新生成，或是否会丢失人工整理信息？

可使用如下检查片段：

```powershell
$target = Resolve-Path -LiteralPath ".pytest_cache" -ErrorAction SilentlyContinue
if ($null -eq $target) {
    Write-Host "目标不存在，无需处理"
} else {
    git ls-files -- "$target"
    rg -n "\.pytest_cache" README.md project-docs yonod.py main.py .gitignore
    Write-Host "确认该路径为缓存且无需保留后，再进入删除步骤"
}
```

真正删除必须放到后续执行阶段，并采用明确路径：

```powershell
# 仅示例：执行前必须替换为已确认的具体路径
$confirmedTarget = Resolve-Path -LiteralPath ".pytest_cache"
if ($confirmedTarget.Path.StartsWith((Get-Location).Path)) {
    Remove-Item -LiteralPath $confirmedTarget.Path -Recurse -Force
} else {
    throw "目标路径不在当前项目内，停止删除"
}
```

#### 验证方法

- 删除清单中每一项都有“用途 / Git 状态 / 引用关系 / 可再生成性 / 处理结论”。
- 删除后 `git status --short` 不出现意外源码或文档变更。
- `python yonod.py --help` 或等价入口检查仍可运行。
- 示例数据路径仍能被脚本读取。

#### 风险提示

`.vscode/` 可能保存用户本地调试配置，`.vs/` 可能保存 Visual Studio 状态；它们通常可删除，但如果用户依赖本地调试任务，删除会影响使用体验。`_verify/` 更需要谨慎，因为它可能是历史 bug 的最短复现材料。

---

### 步骤19.5：保护入口路径与代码引用

#### 目标说明

确保整理动作不破坏现有运行入口、导入路径和 benchmark pipeline。这个步骤是整个目录整理的安全阀。

#### 具体操作

执行引用检查：

```powershell
rg -n "python yonod\.py|python main\.py|dataset/|WEIGHTS/|results/|cache/|docs/参考文献" README.md project-docs yonod.py main.py example.json
```

建立“不可直接移动”清单：

| 路径 | 原因 | 后续策略 |
|---|---|---|
| `main.py` | 被 `yonod.py` 和帮助信息引用 | 本轮不移动 |
| `yonod.py` | README 与计划书中的统一入口 | 本轮不移动 |
| `yonod/` | 核心源码包 | 本轮不移动 |
| `dataset/` | README、示例配置和命令引用 | 本轮不重命名 |
| `WEIGHTS/` | 描述符代码和 README 约定路径 | 本轮不移动 |
| `results/`、`cache/` | 运行流程默认输出/缓存目录 | 可清理内容，但不改变语义 |

如果未来确实需要移动入口或目录，必须作为独立构建任务处理，并同步更新：

- README 示例命令。
- `example.json` 或其他示例配置。
- `yonod.py`、`main.py` 中的路径定位逻辑。
- `project-docs/buildlog.md` 中的实际执行记录。
- 端到端烟测命令。

#### 验证方法

至少完成以下检查：

```powershell
python yonod.py --help
python main.py --help
python yonod.py --csv dataset/test-amide-coupling(additive_fixed).csv --label-col yield --smiles-cols sub_1_smiles sub_2_smiles --descriptors morgan --models rf --task-name folder-refactor-smoke
```

如果烟测耗时或依赖环境不满足，可先执行 `--help` 和 CSV 读取级别检查，并在构建日志中明确说明未运行完整建模的原因。

#### 风险提示

Windows PowerShell 对含括号的路径有解析差异，示例中的 `dataset/test-amide-coupling(additive_fixed).csv` 在实际执行时建议加引号。若命令失败，优先判断是路径转义问题还是代码问题。

---

### 步骤19.6：形成整理结果记录与回滚方案

#### 目标说明

把每次整理变成可审计的小步提交，避免“清爽了但没人知道删了什么”的情况。对于复现实验项目，可追溯性本身就是成果的一部分。

#### 具体操作

后续执行 agent 应在 `project-docs/buildlog.md` 记录：

- 执行日期和操作者。
- 删除/保留/归档清单。
- 每一项处理依据。
- 关键验证命令和结果摘要。
- 若未执行某些验证，说明原因。

建议提交粒度：

```powershell
git status --short
git add .gitignore README.md project-docs/buildlog.md
git commit -m "chore: 整理项目目录与忽略规则"
```

如果只是删除被忽略的本地缓存，通常不产生 Git 提交，但仍应在构建日志中记录本地处理结果。

回滚策略：

- 已提交的文档或 `.gitignore` 改动：用新的修正提交回滚，不使用 `git reset --hard`。
- 被忽略缓存删除：通常可由 IDE、pytest 或 YONOD 重新生成。
- `_verify/` 删除：只有在确认无历史价值后才允许；若仍有价值，优先归档或转为正式测试。
- 文献和权重：默认不删除，只整理索引和说明。

#### 验证方法

- 每次整理后 `git status --short` 只显示预期变更。
- 关键入口命令仍可执行。
- 文献、数据、权重和重要结果快照能够根据文档定位。
- 若产生提交，提交内容不包含大模型权重、IDE 缓存、PDF/SI 附件或临时产物。

#### 风险提示

本项目已有未提交的工作区变更时，执行 agent 不能回滚或覆盖它们。若整理需要修改同一文件，应先读清楚当前 diff，再只追加必要内容。

### 19.5 验收标准

本轮目录整理计划完成后，后续执行应满足：

1. 项目根目录能一眼区分代码、数据、文献、实验产物、项目文档和本地缓存。
2. `main.py`、`yonod.py`、`dataset/`、`WEIGHTS/` 的现有引用不被破坏。
3. `.vs`、`.vscode`、`.pytest_cache`、`__pycache__` 等缓存类目录要么继续被忽略，要么在确认可恢复后删除。
4. `_verify/` 的每个文件都有处理结论，不能仅凭目录名批量删除。
5. `docs/参考文献/` 中的文献与补充材料保留本地边界；可共享复现依据写入 `project-docs/`。
6. 任何实际移动、归档或删除都有检查记录和回滚思路。
7. 整理完成后至少通过入口帮助命令或轻量烟测。

### 19.6 Q&A 记录

### 通用问题

**Q：为什么不直接把 `.vs`、`.pytest_cache`、`__pycache__` 全删掉？**  
**A：** 它们大概率是可删除缓存，但项目整理不能只看名字。`.vs` 和 `.vscode` 可能包含本地调试配置，`_verify/` 可能保存历史 bug 的最短复现脚本；因此要先确认用途、Git 状态、引用关系和可再生成性。

**Q：`docs/参考文献/` 被 `.gitignore` 忽略，会不会影响 VJETHBKM 复现？**  
**A：** 不影响本地复现，但会影响团队共享。因此建议 PDF、SI、Zotero 导出等版权或体积敏感材料继续本地保存；把 DOI、官方数据代码链接、实验参数、复现差异和结论摘要写入 `project-docs/`。

**Q：为什么暂时不移动 `main.py`、`yonod.py`、`dataset/`、`WEIGHTS/`？**  
**A：** 它们已经被 README、示例配置、脚本和计划书多处引用。移动这些路径不是“整理文件夹”，而是入口兼容性重构，应交给后续 builder 任务单独实现和验证。

**Q：实验结果到底放 `results/` 还是 `project-docs/`？**  
**A：** 可再生成的完整运行输出放 `results/`，继续忽略；论文/答辩需要引用的轻量结果摘要、表格结论和复现说明写入 `project-docs/`。如果未来要保存精选 CSV 或图片快照，应先确认体积、隐私、许可证和是否适合入库。

### 19.7 下一步行动建议

1. 由 `project-builder-cn` 按步骤19.1 生成目录盘点清单，不做删除。
2. 基于清单让用户确认 `.vs`、`.vscode`、`.pytest_cache`、`__pycache__`、`_verify/` 的处理策略。
3. 用户确认后，再执行受控清理，并把实际处理结果写入 `project-docs/buildlog.md`。
4. 清理后运行入口帮助命令和最小数据烟测，确认现有 workflow 未被破坏。

---

**文档结束**
