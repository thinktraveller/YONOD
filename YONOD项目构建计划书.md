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
>   11. **包管理改用 conda**：环境约定 `yonod-yield`，T0.2 命令从 `py -3.9 -m venv` 改为 `conda create -n yonod-yield python=3.9`；所有 `Activate.ps1` 调用替换为 `conda activate yonod-yield`；离线 torch whl 仍通过 pip 装（conda 无同版本通道）。理由见 §九 Q11。
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
- 包管理：**conda**（Miniconda / Anaconda 均可），环境名约定 `yonod-yield`
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
2. 用 conda 创建 Python 3.9 环境 `yonod-yield`；
3. 在环境内先用 pip 离线安装 torch 三件套（顺序：torch → torchvision → torchaudio）；
4. 再用 pip 安装其余依赖（rdkit 也直接走 pip，与 transformers/sklearn 版本对齐更稳）。

**为什么 conda 内仍用 pip 装包？**
- 离线 torch whl 是 pip 格式，conda 没有同版本通道；
- transformers / xgboost / autogluon 等在 PyPI 上更新更及时；
- conda 主要作用是**隔离 Python 解释器**和后续可能引入的非 Python 二进制依赖（如未来要装 cudatoolkit、openbabel 时切换更方便）。

**产出物**：conda 环境 `yonod-yield`，依赖全部就绪。

**验收标准**：`python -c "import torch; print(torch.cuda.is_available())"` 输出 `True`。

**PowerShell 命令**：

```powershell
$root = "C:\Users\joyjo\Desktop\其他大学资料\大创\YONOD"
$whl  = "H:\AI模型及torch包\torch_wheels"

# 步骤 1：确认 conda 可用
conda --version

# 步骤 2：创建并激活环境（约定环境名 yonod-yield）
conda create -n yonod-yield python=3.9 -y
conda activate yonod-yield

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
conda activate yonod-yield
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
conda activate yonod-yield
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
conda activate yonod-yield
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
conda activate yonod-yield
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
conda activate yonod-yield
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
conda activate yonod-yield
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
conda activate yonod-yield
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
conda activate yonod-yield
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
conda activate yonod-yield
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
- `pip install autogluon.tabular==1.1.1`（需在 conda 环境 `yonod-yield` 激活后单独运行，体积大）
- 包装 `TabularPredictor(label='yield', problem_type='regression')`
- 设置 `presets='medium_quality'`、`time_limit=300`、`num_cpus=1`（Windows 兼容）
- 用 `holdout_frac=0.2` 代替 K 折（AutoGluon 内置 CV 策略）

**产出物**：`yonod_yield/models/autogluon_model.py`。

**验收标准**：500 条数据 5 分钟内出结果，无 multiprocessing 相关报错。

**PowerShell 命令**：

```powershell
$root = "C:\Users\joyjo\Desktop\其他大学资料\大创\YONOD"
conda activate yonod-yield
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
conda activate yonod-yield
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
conda activate yonod-yield
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
conda activate yonod-yield
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
conda activate yonod-yield
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
conda activate yonod-yield

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
conda activate yonod-yield
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
conda activate yonod-yield

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
conda create -n yonod-yield python=3.9 -y
conda activate yonod-yield

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
**答**：venv 是 Python 标准库自带、零额外安装、足够轻量；conda 的优势在化学/AI 项目场景下更明显：① RDKit 在 conda-forge 上的二进制最为稳定可靠（虽然 PyPI 的 `rdkit==2023.9.5` 也已可用）；② 后续若需要 cudatoolkit、openbabel、psi4 等非 Python 二进制依赖，conda 的安装体验明显优于 venv + pip；③ `conda env list` 在多项目场景下管理更直观。本项目 v0.3 正式约定环境名 `yonod-yield`。注意：离线 torch whl 仍通过 pip 安装（conda 没有同版本通道），二者不冲突。

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
