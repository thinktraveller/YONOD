# YONOD — 有机合成反应预测平台

> **Y**ou **O**nly **N**eed **O**utstanding **D**escriptors
>
> 以 SMILES、显式列角色和可审计的特征版本为输入，比较分子描述符与机器学习模型在有机合成反应预测任务中的表现。

## 当前接口

稳定、可复现的运行配置使用 schema-2 YAML。正式建模任务通过 `yonod.py` 验证并启动；它会把已验证的配置交给配置运行时：

```bash
python -u yonod.py
# 在第一个提示中输入已保存的 YAML 路径，例如 example.yaml
```

仓库根目录的 [example.yaml](example.yaml) 是一个可直接运行的 12 样本 Morgan × Random Forest、2-fold smoke。成功时应看到 `features=ready` 和 `morgan × rf: complete` 或 `reused`；运行 ID 是内容派生的，不应写死。Chemical VAE 的历史验收 YAML 已归档在 [configs_20260916.tar.gz](recovery_backups/configs_20260916.tar.gz)，不是 `example.yaml` 的隐式组成部分。

| 目的 | 当前入口 |
| --- | --- |
| 启动已保存的可复现配置 | `python -u yonod.py`，在首个提示中输入 YAML 路径 |
| 交互创建并执行配置 | `python yonod.py` |
| 只读查看或派生特征版本 | `python scripts/manage_features.py inspect ...` / `derive ...` |

`yonod.py` 是交互向导，**不解析** `--csv`、`--models` 等命令行参数。向导可接收 CSV，收集列角色并写出 `*_run.yaml`；也可在首个提示中输入现有 schema-2 YAML 后验证并启动。所有建模任务均须使用已保存的 YAML 并经 `yonod.py` 启动。

JSON 不再是 `main.py` 或向导的可执行运行配置，且不再维护 JSON 配置迁移或 JSON 依赖启动器的兼容性。`main.py --json ...` 会明确拒绝。历史 JSON 指标、折级审计、第三方资料和既有结果仍按原格式保留；它们不是训练入口，也不构成需要迁移的依赖。

## 快速开始

### 1. 环境

项目基线是 Python 3.9。Linux 开发或测试前，在同一个 shell 激活 `yonod`：

```bash
source /home/wangzh685/miniconda3/etc/profile.d/conda.sh
conda activate yonod
```

新环境可按以下顺序安装。PyTorch 的 CUDA wheel 应与本机 CUDA/驱动匹配；CPU 用户请把 wheel index 改为 PyTorch 官方 CPU index。

```bash
conda create -n yonod python=3.9 -y
conda activate yonod

# 例：CUDA 12.1
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 \
  --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install "autogluon.tabular[lightgbm,catboost]==1.1.1"
```

`requirements.txt` 列出了基础依赖；AutoGluon、LightGBM、FISD 和 MolMetaLM 可能需要额外资产或较多磁盘空间。若只运行 `example.yaml`，不需要 FISD 或 MolMetaLM 权重。

### 2. 运行 smoke

Linux 的建模任务应使用保存好的 YAML、`yonod.py` 和任务专属日志/PID；不要直接用 `main.py --config` 启动模型：

```bash
mkdir -p logs
nohup setsid bash -lc '
  source /home/wangzh685/miniconda3/etc/profile.d/conda.sh
  conda activate yonod
  printf "%s\n" "example.yaml" | python -u yonod.py
' </dev/null > logs/example_morgan_smoke.log 2>&1 &
echo $! > logs/example_morgan_smoke.pid

task_pid="$(cat logs/example_morgan_smoke.pid)"
ps -p "$task_pid" -o pid,ppid,stat,etime,cmd
tail -n 100 logs/example_morgan_smoke.log
```

该示例把输出写到 `derived/interface_migration/example_artifacts/` 与 `derived/interface_migration/example_results/`。它是测试产物位置，不应复用于正式研究；请复制 YAML 并为正式数据设置独立的 `artifacts.output_dir` 与 `outputs.root`。

### 3. 用向导创建配置

```bash
python yonod.py
```

向导会验证 CSV、声明标签/SMILES/数值列、选择特征和模型，并要求选择非空且唯一的稳定 `sample_id` 列。随后会在项目 `docs/` 目录写出 `*_run.yaml`，并可调用：

```bash
python -u yonod.py
# 在首个提示中输入 path/to/project_run.yaml
```

在 Windows 上请通过 Conda Prompt 激活同名环境。应用代码使用仓库相对路径和 `pathlib`；YAML 内的相对路径以 YAML 所在目录为基准解析。

## Schema-2 YAML

每个运行配置必须带有 `schema_version: "2.0"`、项目名、阶段、数据集和稳定样本 ID。解析器只接受 UTF-8 的 `.yaml`/`.yml` 单文档，拒绝重复键、别名、未知字段和不支持的模型/参数区段。

```yaml
schema_version: "2.0"
project_name: my_reaction_task
stage: all

dataset:
  path: ./reactions.csv
  sample_id_col: reaction_id
  column_roles:
    label: yield
    reactants: [reactant_smiles, reagent_smiles]
    products: []
    others: []
    conditions: [temperature_c]
    categoricals: [solvent]

descriptors:
  - id: morgan_r2
    descriptor: morgan
    mode: concat
    columns: [reactant_smiles, reagent_smiles]

artifacts:
  output_dir: ./artifacts

models: [rf]
model_params:
  rf:
    estimator:
      n_estimators: 300
      random_state: 42
      n_jobs: 1

evaluation:
  protocol: outer_kfold
  n_splits: 5
  n_repeats: 1
  shuffle: true
  seed: 42

outputs:
  root: ./results
  report_formats: [Markdown]
```

`dataset.column_roles.label` 是 `train` 与 `all` 的必填项；特征阶段可以不声明标签。`sample_id_col` 必须在数据集中存在、非空并且唯一。不要以 DataFrame 行号充当样本 ID。

### 阶段

| `stage` | 必要输入 | 行为 |
| --- | --- | --- |
| `features` | `dataset`、`descriptors`、`artifacts.output_dir` | 计算或复用特征包；不构造模型，也不需要标签。 |
| `train` | 含标签的 `dataset`、`artifacts.input_manifest`、`models` | 只读取指定特征包并训练；绝不回退到描述符计算。 |
| `all` | features 与训练所需声明 | 先令全部候选特征达到 ready/failed 终态，再训练所有就绪包；每个 ready feature × model 组合独立发布。 |

某个已经确定身份的 feature × model 在实际训练中失败时，YONOD 会在 `outputs.root/failed_runs/` 原子发布带失败原因、原始 YAML 和生效 YAML 的不可覆盖 manifest，并继续同级组合。失败组合不生成不完整 predictions/metrics；只要任何组合失败，`main.py --config` 返回码为 2，避免调度器把部分成功误报为整体成功。真实 fixture 证据见 [training_failure_isolation_evidence.md](derived/interface_migration/step30_acceptance/training_failure_isolation_evidence.md)。

### 报告与离线重建

对 schema-2 的 `train` / `all`，`outputs.report_formats` 省略时生成 HTML 和 Markdown；支持 `html`、`markdown`（或 `md`），空列表关闭自动报告。每个完成 run 从自身的 `run_manifest.yaml`、预测、fold 指标及 YAML 快照生成 `pictures/` 与 `report/`；散点图使用每个 repeat 的 OOF 预测，HTML 内嵌 PNG，Markdown 使用 `../pictures/` 相对链接。`features` 阶段不会生成报告。

无需原始数据、描述符权重或模型即可重建普通结果报告：`python scripts/rebuild_report.py --run-dir <outputs.root/runs/run-id> --output-root <新的报告目录>`。该命令会校验 manifest、配置快照哈希、OOF 身份和有限数值，不会训练或改写源结果；strict benchmark 仍使用 `scripts/rebuild_benchmark_report.py`。

特征服务把静态矩阵、样本 ID、有效性掩码、身份和哈希发布到独立包。训练服务把每个 `feature × model` 的结果发布到 `outputs.root/runs/<content-derived-run-id>/`，其中包括：

- `predictions.csv` 与 `fold_metrics.csv`；
- `run_manifest.yaml`；
- 原始 `source_config.yaml` 和规范化 `effective_config.yaml`；
- 对按折变换特征，保存折内变换状态和审计信息。

同一身份的完整包会被验证后复用；任何已有不完整结果都不会被静默当作完成。

### 特征

公开特征注册表当前包含：

- 静态分子/反应描述符：`morgan`、`mfp`、`maccs`、`fisd`、`molmetalm`、`maf`、`rdkit2d`、`drfp`。
- 已验证的静态分子描述符：`chemical_vae`（独立验收的 ZINC/v5 与 `zinc_properties` encoder/v5、原始 `z_mean_sample`；性质头不在描述符范围）。
- 按折变换特征：`ohe`。它保存原始类别输入；OneHotEncoder 只在每个外层训练折拟合，验证折只做转换。

`mfp` 是 Morgan **count** fingerprint，不是二值 `morgan` 的别名。它可显式设置 `radius`、`fp_size` 与 `profile`。`drfp` 需要满足反应输入约束；`fisd` 和 `molmetalm` 需要下文所述权重。

数值缩放、OHE、SVM 的 scaler/PCA/子采样，以及 XGBoost/LightGBM 的早停内部验证都限定在外层训练折内。外层验证折只用于最终预测。

### Chemical VAE（冻结、独立验收的 encoder）

Chemical VAE 是已接入、但刻意受限的可选描述符：它只接受各自经数值对照验证的 ZINC/v5 或 `zinc_properties` encoder/v5 conversion manifest，并从相邻的已哈希 PyTorch state 取编码器输出。`zinc_properties` 只开放 encoder 的原始 `z_mean_sample`，不加载其性质头；两种资产不能互相替代。它不加载解码器、性质头、训练 CSV、原 HDF5 或 TensorFlow/Keras；普通运行时只需要 PyTorch。

```yaml
descriptors:
  - id: chemical-vae-zinc-v5
    descriptor: chemical_vae
    lifecycle: static_descriptor
    mode: concat
    columns: [reactant_1_smiles, reactant_2_smiles]
    params:
      model_manifest: ../../WEIGHTS/chemical_vae/zinc-37e96cd3bc8f9680/v5/conversion_manifest.json
      backend: pytorch
      device: cpu
      batch_size: 4
      input_preprocessing: identity
      output: z_mean_sample
```

`model_manifest` 仍以 YAML 所在目录为基准解析。当前资产的每个分子列输出 196 维原始 `z_mean_sample`；上例按列拼接为 392 维。`backend` 只能是 `pytorch`，`input_preprocessing` 只能是 `identity`，`output` 只能是 `z_mean_sample`；未知参数和未验证资产会失败而不是静默回退。

输入语义是逐字节 identity：不 trim、不 canonicalize、不拆盐、不替换分隔符、不截断、不扩展字符表。适配器先检查固定 35 字符表、长度上限 120 和 RDKit 可解析性；缺失、超长、字符不支持或 RDKit 无效都会写入版本化 diagnostics sidecar。`concat` 保持既有语义：至少一个分子列成功时保留反应行，失败列为零块；所有选中列都失败才使该行的 mask 为 false。严格共同子集比较必须另行要求每个选中角色都成功，不能把零块留存集当作严格覆盖率。

步骤 31 的 Chemical VAE features/train/all、配对比较、计时/成本、GPU smoke 与 Windows 交接 YAML 均为历史工程验收输入，已保存在 [配置归档](recovery_backups/configs_20260916.tar.gz)。需要审计时，从仓库根解压该归档以恢复原始 `configs/` 路径；新任务应创建自己的独立 YAML，而不要修改或复用这些小样本验收输入。

示例和运行证据是工程验收，不是性能结论。Linux CPU 和两张可用 RTX 5090 中的 `cuda:0` 已真实运行；`zinc_properties` encoder 也已在冻结的 CPU float32 `atol=rtol=2e-5` 下经原 HDF5 TensorFlow/Keras 与独立 NumPy 对照后运行 GPU smoke。GPU 相对 CPU artifact 的差异单独记录，绝不回写成 C03 CPU parity。用户已授权将步骤31标记完成并延期 Windows CPU 验证；Windows CPU 仍没有实测，不能视为已支持。完整状态见 [C11 verification](derived/chemical_vae/step31_9_gpu_smoke/verification.json)、[step31 acceptance index](derived/chemical_vae/step31_acceptance/index.json) 与 [comparison report](derived/chemical_vae/step31_7_comparison/comparison_report.json)。

#### Chemical VAE 依赖与数值对照

`requirements.txt` 的 `h5py` 用于资产审计/转换，不是 PyTorch 运行时的必需导入。TensorFlow/Keras 不应被作为普通 Chemical VAE 特征生成、Morgan/MFP 或 train-only manifest 消费的前提。

### 模型与参数

支持的模型标识符是 `rf`、`xgb`、`svm`、`lightgbm` 和 `autogluon`。模型名别名会在 YAML 加载时规范化；为避免实验身份歧义，不要同时写别名和规范名。

模型参数必须放在下列明确 API 区段。字段未写出时，使用所安装受支持库的默认值；`null`、`false` 和 `0` 仍是显式值。

| 模型 | 可用区段 |
| --- | --- |
| `rf`、`xgb`、`lightgbm` | `estimator`、`fit`、`runtime` |
| `svm` | `estimator`、`fit`、`preprocessing`、`runtime` |
| `autogluon` | `predictor`、`fit`、`runtime` |

XGBoost 和 LightGBM 如需 early stopping，应只声明由外层训练折产生的内部验证策略，不能把外部 `eval_set`、回调或数组传进 `fit`：

```yaml
model_params:
  xgb:
    runtime:
      early_stopping: {rounds: 20, validation_fraction: 0.2, seed: 42}
```

完整可运行字段示例见 [example.yaml](example.yaml)；参数路由与契约见 [project-docs/step30-contracts.md](project-docs/step30-contracts.md)。

## 特征版本操作

查看已发布的特征包：

```bash
python scripts/manage_features.py inspect --manifest artifacts/features/<artifact-id>/manifest.yaml
```

从操作 YAML 派生新版本：

```bash
python scripts/manage_features.py derive --config derive.yaml
```

`derive_features` 只支持显式样本选择、特征选择及按唯一 sample ID 连接数值 CSV。它永远发布新包并记录父版本、操作、输入/输出映射和哈希；标签、重复/缺失 ID、列冲突和不安全的按折变换操作会失败。

旧 JSON 运行配置不属于当前可执行入口。仓库保留的迁移器只在能够显式恢复 schema-2 字段时生成新的 YAML 供人工复核；JSON 本身及依赖 JSON 的启动器不能作为新建或恢复建模任务的入口。

## 历史 CSV CLI

`main.py --csv ...` 仍保留为代码级调试/历史兼容接口，但不得用它启动建模任务。它的旧式 `docs/`、`pictures/`、`report/` 输出布局不属于 schema-2 产物或训练契约；新任务必须先保存独立 YAML，再由 `yonod.py` 启动。

## 专用 benchmark 与历史运行

strict benchmark 已迁移为 schema-2 的专用 `stage: benchmark`，其公开启动仍是 `yonod.py`。为每项研究创建独立且相同的 `artifacts.output_dir` / `outputs.root`，然后在 Linux 使用：

```bash
mkdir -p logs
nohup setsid bash -lc '
  source /home/wangzh685/miniconda3/etc/profile.d/conda.sh
  conda activate yonod
  printf "%s\n" "path/to/my_benchmark.yaml" | python -u yonod.py
' </dev/null > logs/my_benchmark.log 2>&1 &
echo $! > logs/my_benchmark.pid
```

strict benchmark 强制 `evaluation.protocol: manifest_outer_cv`、显式 `evaluation.grouping` 和 `benchmark.task_state: {backend: sqlite, resumable: true}`。它生成并绑定唯一 split manifest；当前不接受未经逐行等价核验的 `evaluation.split_manifest` 导入，也绝不回退成普通 `outer_kfold`。该路径需要 `pyarrow`（已列入 `requirements.txt`）或 `fastparquet`，以原子发布 split、逐折预测和指标证据。现有 Conda 环境可用不触及默认 channel 的方式安装固定版本：`conda install -y --override-channels -c conda-forge pyarrow=16.1.0 numpy=1.26.4`。多候选 Tukey HSD 另需 `statsmodels>=0.14`；一个模型/一个描述符的 smoke 不会调用该可选统计。

已完成的低成本 strict smoke 使用独立 12 行 fixture，经 `yonod.py` 产生 2 个 component-holdout 外部折、SQLite state、parquet 预测/指标和 HTML/Markdown 报告；配置与产物核验见 [benchmark_schema2_preflight_evidence.md](derived/interface_migration/step30_acceptance/benchmark_schema2_preflight_evidence.md)。

下列项目明确**不纳入迁移或完成标准**：所有 JSON 运行配置及依赖 JSON 的启动器（包括 paper-exact 和历史 YieldMaster shell 入口），以及 `different_order/amide-coupling/run_different_order_tasks.py`。它们保留为历史/归档材料，不是受支持的建模入口；不得为恢复这些路径而新增 JSON 兼容分支或补建 `different_order` runner。

迁移边界和验证证据见 [project-docs/step30-contracts.md](project-docs/step30-contracts.md) 与 [步骤 30 验收报告](derived/interface_migration/step30_acceptance/acceptance_report.md)。已有的 JSON 指标、预测、run/fold manifest、官方资料和 `reference-proejct/` 内容是历史/审计材料，应保留原格式。

## 数据与外部资产

### 酰胺缩合数据集

仓库内的 `dataset/amide-coupling.csv` 来自 [aichemeco/amide_coupling](https://github.com/aichemeco/amide_coupling/tree/main)（MIT 协议），含 47,015 条酰胺缩合反应，`yield` 归一化至 `[0, 1]`。`dataset/benchmark_smoke_fixture.csv` 是 schema-2 smoke 使用的小型 fixture；README 不再假设存在 `dataset/test-amide-coupling.csv`。

如将该数据集用于发表，请引用：Dai *et al.*, *Chem. Sci.*, 2025, DOI [10.1039/D5SC03364K](https://doi.org/10.1039/D5SC03364K)。

| 列名 | 类型 | 说明 |
| --- | --- | --- |
| `row_id` | int | 可作为经唯一性验证后的候选样本 ID |
| `sub_1_smiles`、`sub_2_smiles` | str | 底物 SMILES |
| `product_smiles` | str | 产物 SMILES；是否使用由 YAML 特征声明决定 |
| `activation`、`additive`、`base`、`solvent` | str | 反应组分 SMILES；空值和多组分语义应由特征声明处理 |
| `yield` | float | 回归目标 |

### MolMetaLM 权重

- Hugging Face: [wudejian789/MolMetaLM-base](https://huggingface.co/wudejian789/MolMetaLM-base)
- GitHub: [CSUBioGroup/MolMetaLM](https://github.com/CSUBioGroup/MolMetaLM)
- 论文: DOI [10.48550/arXiv.2411.15500](https://doi.org/10.48550/arXiv.2411.15500)

```bash
pip install -U "huggingface_hub>=0.34"
hf download wudejian789/MolMetaLM-base --local-dir WEIGHTS/MolMetaLM-base
```

下载约 500 MB；目录应包含 `config.json`、`model.safetensors`、tokenizer 文件等模型资产。

### FISD 权重

FISD 依赖 QM9 上预训练的三份 GCN 权重（约 19 MB）。从 [KeantChen/FISD model](https://github.com/KeantChen/FISD/tree/main/model) 获取后放到 `WEIGHTS/FISD/`：

```text
WEIGHTS/FISD/
├── qm_9_mse_model.pth
├── qm_9_cos_model.pth
└── qm_9_2in1_model.pth
```

相关论文 DOI：[10.1039/D5SC00451A](https://doi.org/10.1039/D5SC00451A)。缺少这些文件时，选择 `fisd` 的特征计算会明确失败，其他独立特征仍可完成。

### ATMOMACCS / MACCS

本项目的 `maccs` 特征调用 RDKit 标准 MACCS keys（167 位去掉占位 bit 0 后为 166 维），无需下载 ATMOMACCS 资产。若引用对应实验结果，请同时遵守上游 [Zenodo 18669279](https://zenodo.org/records/18669279) 和论文 [10.1063/5.0308548](https://doi.org/10.1063/5.0308548) 的许可与引用要求。

## 项目结构

目录用途、保留边界和清理记录见 [目录管理说明](project-docs/folder-organization.md)。按用途查找：[项目文档](project-docs/README.md) · [辅助脚本](scripts/README.md) · [配置归档](recovery_backups/configs_20260916.tar.gz) · [数据集](dataset/README.md)。

```text
YONOD/
├── main.py                     schema-2 底层配置运行时与 smoke 入口；保留历史 CSV 调试接口
├── yonod.py                    正式建模任务入口；生成/验证 schema-2 YAML
├── example.yaml                可运行的最小 schema-2 smoke
├── example.md                  阶段、操作和迁移的简明示例
├── recovery_backups/configs_20260916.tar.gz
│                               历史 schema-2 与 Chemical VAE 验收 YAML
├── WEIGHTS/chemical_vae/       内容校验的转换后编码器资产（本地，不从 YAML 训练）
├── yonod/
│   ├── config/                 YAML 加载和 schema 契约
│   ├── artifacts/              不可变特征包的读写与派生
│   ├── pipeline/               features、train 与 all 编排
│   ├── model_factory.py        五模型参数路由与生效参数审计
│   ├── descriptors/            特征注册与描述符实现
│   └── benchmark/              独立 benchmark / paper-exact 协议（迁移中）
├── scripts/                    特征操作、迁移及专用运行脚本
├── project-docs/               设计、契约和构建记录
├── dataset/                    数据集与 smoke fixture
└── derived/                    本地验收与生成产物
```

## 验证与贡献注意事项

从仓库根目录、在 `yonod` 环境中运行：

```bash
python -m unittest tests.test_chemical_vae_audit tests.test_chemical_vae_encoder_parity \
  tests.test_chemical_vae_descriptor tests.test_chemical_vae_artifact_lineage \
  tests.test_chemical_vae_training_isolation tests.test_mfp_descriptor -v
python -m unittest tests.test_step30_contracts tests.test_step30_yaml_loader \
  tests.test_step30_feature_service tests.test_step30_training_service -v
```

不要覆盖已提交数据集、历史运行输出、特征包或 `reference-proejct/`。项目在 Linux 开发，但应用代码必须保持 Windows 兼容：使用 `pathlib.Path`/`os.path`，不硬编码路径分隔符，并保留 UTF-8 中文路径支持。

## License

本项目代码和文档采用 [CC BY-NC 4.0](LICENSE)。第三方数据、模型权重和参考工程遵守各自许可证；使用或发布结果前请核对相应来源的条款和引用要求。
