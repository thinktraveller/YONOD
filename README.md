<a id="zh"></a>

# YONOD — 有机合成反应预测平台

[中文](#zh) · [English](#en)

> **Y**ou **O**nly **N**eed **O**utstanding **D**escriptors
>
> 以 SMILES、显式列角色和可审计的特征版本为输入，比较分子描述符与机器学习模型在有机合成反应预测任务中的表现。

## 当前接口

稳定、可复现的运行配置使用 schema-2 YAML。正式建模任务通过 `yonod.py` 验证并启动；它会把已验证的配置交给配置运行时：

```bash
python -u yonod.py
# 在第一个提示中输入已保存的 YAML 路径，例如 example.yaml
```

仓库根目录的 [example.yaml](example.yaml) 是一个可直接运行的 12 样本 Morgan × Random Forest、2-fold smoke。成功时应看到 `features=ready` 和 `morgan × rf: complete` 或 `reused`；运行 ID 是内容派生的，不应写死。Chemical VAE 的历史验收 YAML 已归档在 [configs_20260916.tar.gz](backup/configs_20260916.tar.gz)，不是 `example.yaml` 的隐式组成部分。

面向化学研究人员的[三种检验方式示例](example/index.html)集中在 [example/](example/README.md)：随机划分内测、固定名单两折内测和独立外测流程。三个 `run.yaml` 均已从各自目录实际运行，结果保存在本包 `result/`；无需复制配置。下载包附本轮已完成输出快照，数据均为人工教学记录。

| 目的 | 当前入口 |
| --- | --- |
| 启动已保存的可复现配置 | `python -u yonod.py`，在首个提示中输入 YAML 路径 |
| 交互创建并执行配置 | `python yonod.py` |
| 只读查看或派生特征版本 | `yonod.artifacts.operations` 的 `inspect_feature_artifact` / `derive_features` API |

`yonod.py` 是交互向导，**不解析** `--csv`、`--models` 等命令行参数。向导可接收 CSV，收集列角色并写出 `*_run.yaml`；也可在首个提示中输入现有 schema-2 YAML 后验证并启动。所有建模任务均须使用已保存的 YAML 并经 `yonod.py` 启动。

JSON 不再是 `main.py` 或向导的可执行运行配置，且不再维护 JSON 配置迁移或 JSON 依赖启动器的兼容性。`main.py --json ...` 会明确拒绝。历史 JSON 指标、折级审计、第三方资料和既有结果仍按原格式保留；它们不是训练入口，也不构成需要迁移的依赖。

## 三种教学样例：在自己的文件夹完整运行

[示例总入口](example/index.html) · [详细操作说明](example/README.md) · [下载全部示例](example/archives/all_examples.zip)

三个样例均已于2026-11-13在Linux的`yonod`环境中，通过`yonod.py`从各自目录读取`run.yaml`完整执行；进程返回码均为0，逐条预测、划分成员、评分和HTML/Markdown报告已验收。它们是教学样例，不必复制YAML到根`config/`；正式研究仍使用根`config/`中的独立任务配置。

| 方法 | 数据与配置 | 已完成的计算 | 本轮报告快照与验收记录 |
| --- | --- | --- | --- |
| 随机划分内测 | [12条记录](example/random_split/dataset/reactions.csv)、[run.yaml](example/random_split/run.yaml) | 同一批数据随机分成两份轮换；RF、XGBoost、LightGBM各两折 | [报告](example/random_split/completed_run/report/report.html) · [验收](example/random_split/verification.json) |
| 固定名单两折内测 | [12条记录](example/fixed_split/dataset/reactions.csv)、[成员清单](example/fixed_split/splits/fixed_split_manifest.csv)、[run.yaml](example/fixed_split/run.yaml) | 按事先指定的名单，两折各6条建模、6条检验，再交换；RF两折 | [报告](example/fixed_split/completed_run/report/benchmark_report.html) · [验收](example/fixed_split/verification.json) |
| 独立外测流程 | [12条训练记录](example/external_test/dataset/train.csv)、[6条外测记录](example/external_test/dataset/external.csv)、[run.yaml](example/external_test/run.yaml)、[来源核查](example/external_test/audit/external_audit.json) | 模型设置预先固定，RF使用全部训练表建立一次模型，再检验外测表；外测含3个胺组 | [报告](example/external_test/completed_run/report/external_test_report.html) · [验收](example/external_test/verification.json) |

从所选样例目录调用`../../yonod.py`，首次提示输入`run.yaml`。输入相对该YAML所在目录，新结果写入本包的`result/<任务名>/`，结构数值编码写入其中的`feature/`。Linux使用无头后台启动、恰好19个可用逻辑CPU和仓库根`logs/`中的专属日志/PID，任务依次运行；完整命令见[操作说明](example/README.md)。从样例目录调用主入口时，相对YAML路径会在切换子进程目录前固定，避免找错配置。

Windows Conda Prompt示例（从仓库根开始）：

```text
conda activate yonod
cd example\random_split
python -u ..\..\yonod.py
```

首次提示输入`run.yaml`；另两种样例改为进入相应目录。命令保留Windows用法，但本轮实际执行验收只覆盖Linux。

下载包的`completed_run/`保存本轮完整输出快照，活动`result/`不打入ZIP，因此解压后可从头运行；源码Git不跟踪这两个运行目录。源码克隆后可直接运行生成新结果，或解压下载包查看已完成报告。随机包的`report/`另保留2026-09-29的历史报告；`evidence/`中的YAML是历史追溯材料，不属于三个活动样例。

全部数据和产率均为人工教学记录，不是实验测量。固定名单不自动保证未见底物，当前接口要求至少两折、每条记录每轮恰好留出一次；另存外测CSV也不自动证明来源独立，本例只展示外测操作。R²（决定系数）不是预测正确率；0–1产率尺度上的MAE（平均绝对误差）乘100才是百分点。随机报告按每轮汇总留出预测评分，固定名单报告按逐折指标汇总，不能跨口径直接排名。

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

该示例使用仓库内的 `dataset/benchmark_smoke_fixture.csv`，并将特征、训练结果和报告统一写入 `result/smoke-test/`：

```text
result/smoke-test/
├── feature/
│   ├── feature_runs/<feature_run_id>.yaml
│   └── features/<artifact_id>/
├── pictures/
│   └── scatter_morgan_rf_repeat-1.png
├── report/
│   ├── report.html
│   └── report.md
└── runs/
    └── run_morgan-rf/
        ├── run_manifest.yaml
        ├── source_config.yaml
        ├── effective_config.yaml
        ├── predictions.csv
        └── fold_metrics.csv
```

路径按实际配置位置解析：根`config/`中的YAML对以`./`开头的引用按仓库根解析；显式`../`引用及`example/`等其他位置的配置按YAML所在目录解析；绝对路径保持绝对。`outputs.root`是任务主目录，普通运行加载器将有效配置的`artifacts.output_dir`统一为该目录下的`feature/`。启动日志与PID位于仓库`logs/`。该smoke输出不得复用于正式研究：研究任务应在`config/`新建YAML并使用专属`result/<任务名>/`；上述教学样例采用各自目录的`run.yaml`与`result/<任务名>/`。

### 3. 用向导创建配置

```bash
python yonod.py
```

向导会验证 CSV、声明标签/SMILES/数值列、选择特征和模型，并要求选择非空且唯一的稳定 `sample_id` 列。随后应将 `*_run.yaml` 保存到项目 `config/` 目录，并可调用：

```bash
python -u yonod.py
# 在首个提示中输入 path/to/project_run.yaml
```

在Windows上请通过Conda Prompt激活同名环境。应用代码使用可配置路径和`pathlib`；YAML引用遵循上面的配置位置规则，包括根`config/`中`./`按仓根解析的例外。

## Schema-2 YAML

唯一可复制、可校验的YAML规范是仓库根目录的[example.yaml](example.yaml)。每个新研究任务应在`config/`创建独立schema-2文件，仅替换任务专属的数据集、描述符、模型、评估参数和隔离结果路径；教学样例的`example/<样例>/run.yaml`是明确的就地运行例外。不要在Markdown中维护第二份完整YAML。所选RF/XGBoost/LightGBM使用`model_params.<model>.estimator.n_jobs: 19`，AutoGluon使用`model_params.autogluon.fit.num_cpus: 19`；仅声明实际选择的模型，任务串行。

步骤 2-14 的 HPO 在 Linux `yonod` 环境已通过普通 RF/XGBoost/LightGBM、strict 数值及 OHE 入口、软时限、断点恢复和独立最终模型的工程验收；Windows 实机验证依用户安排暂缓，仍为 `not_verified`。`hpo` 省略或 `enabled: false` 时沿用固定参数路径，不导入 Optuna；依赖版本固定为 4.5.0。预算用 `max_trials` 显式指定；字段、搜索空间和默认关闭示例仅见 [example.yaml](example.yaml)。每个外层训练折的 HPO study 与内层折独立持久化在该任务的 `result/<task_name>/hpo/`，不能把内层搜索分数当成外层 OOF 成绩；不以搜索分数上升作为通过标准。完整 Linux 验收入口为 `_verify/accept_hpo_all.py`，其脚本与本地证据按项目约定不进入主仓。

每个运行配置必须声明 schema 版本、项目名、阶段、数据集和稳定样本 ID。解析器只接受 UTF-8 的 `.yaml`/`.yml` 单文档，拒绝重复键、别名、未知字段和不支持的模型/参数区段。`dataset.column_roles.label` 是 `train` 与 `all` 的必填项；特征阶段可以不声明标签。`sample_id_col` 必须在数据集中存在、非空并且唯一，不能用 DataFrame 行号代替。

### 阶段

| `stage` | 必要输入 | 行为 |
| --- | --- | --- |
| `features` | `dataset`、`descriptors`、`artifacts.output_dir` | 初始化特征输出并计算或复用特征包；不构造模型，也不需要标签。建议同时显式声明 `outputs.root`。 |
| `train` | 含标签的 `dataset`、`artifacts.input_manifest`、`models` | 只读取指定特征包并训练；绝不回退到描述符计算。 |
| `all` | features 与训练所需声明 | 先令全部候选特征达到 ready/failed 终态，再训练所有就绪包；每个 ready feature × model 组合独立发布。 |
| `benchmark` | 含标签的`dataset`、结构输入、模型及专用`evaluation`/`benchmark`声明 | 使用`manifest_outer_cv`执行分组/固定名单内测，或使用`frozen_train_external_test`执行训练后单次外测；两种协议各有约束。 |

外部 feature manifest、标签和辅助列一律按稳定 `sample_id` 对齐。静态特征身份不包含标签，因此只修改标签会改变训练身份而不是特征身份。

某个已经确定身份的 feature × model 在实际训练中失败时，YONOD 会在 `outputs.root/failed_runs/` 原子发布带失败原因、原始 YAML 和生效 YAML 的不可覆盖 manifest，并继续同级组合。失败组合不生成不完整 predictions/metrics；只要任何组合失败，`main.py --config` 返回码为 2，避免调度器把部分成功误报为整体成功。真实 fixture 证据见 `derived/interface_migration/step30_acceptance/training_failure_isolation_evidence.md`。

下文以`derived/...`列出的六项历史本地验收证据未附在当前工作区，路径保留作追溯，不作为可点击下载链接；本轮没有重新执行这些历史验收。

### 报告与离线重建

对 schema-2 的 `train` / `all`，`outputs.report_formats` 省略时生成 HTML 和 Markdown；支持 `html`、`markdown`（或 `md`），空列表关闭自动报告。每个完成 run 从自身的 `run_manifest.yaml`、预测、fold 指标及 YAML 快照生成 `pictures/` 与 `report/`；散点图使用每个 repeat 的 OOF 预测，HTML 内嵌 PNG，Markdown 使用 `../pictures/` 相对链接。`features` 阶段不会生成报告。

无需原始数据、描述符权重或模型即可通过 `yonod.pipeline.reporting.rebuild_schema2_report` 重建普通结果报告。该函数会校验 manifest、配置快照哈希、OOF 身份和有限数值，不会训练或改写源结果。旧的专用脚本保存在本地 `backup/scripts/`，不再作为当前项目入口。

完成的 HPO 组合另有“超参数搜索与嵌套评估”区：逐 study/trial 轨迹、每折实际参数与来源、独立外层折指标，以及搜索、外层训练和预测的分项耗时。普通汇总表仍按每个 repeat 的 pooled OOF 计算，不能把它与逐折均值或内层搜索分数混称。显式最终模型单列开发集搜索和重训；没有独立测试时不填测试分数。重建器只读核验持久 study 导出的哈希，不打开数据库或重新拟合。HPO 失败/软截止的 `report/hpo_status.html` 与 `.md` 只展示状态和已核验搜索证据，不把不完整折当作 OOF。

### 训练与外层评估诊断（步骤 2-15，Linux 三树范围已验收）

九个隔离小样本任务覆盖普通与 strict 的 RF、XGBoost、LightGBM、fixed/HPO、诊断开关及普通早停；外层折预测、诊断数值、模型重载和双报告已独立核对。`logs/diag215_20260929_acceptance_resume5.log` 保存统一验收记录。历史 SVM/AutoGluon 结果可读并明确显示“未记录”诊断；本轮没有重新拟合 AutoGluon，Windows 实机仍未验证。这些工程验收不代表真实化学数据的模型效果。

`outputs.diagnostics.enabled` 控制 RF、XGBoost、LightGBM 的折级训练诊断（默认 `true`）；`outputs.diagnostics.save_train_predictions` 控制是否另外保存训练逐行预测（默认 `false`，验收任务应设为 `true`）。唯一完整 YAML 结构见 `example.yaml`。这些选项只控制诊断，不改变外层 OOF 的文件格式或评分口径。SVM、AutoGluon 暂无同口径训练诊断，原建模能力保持。

诊断复用最终外层折模型和已拟合特征状态，计算 outer-train 的训练分数、outer-heldout 分数，以及以该折 outer-train 标签算术均值作常数预测的 heldout 基线。R² 差距为训练减外层，RMSE/MAE 差距为外层减训练。早停时另记录实际 fit 与内部早停留出人口；训练分数是已见样本的拟合诊断，不代表泛化。无单位声明时单位标为未知，常量标签等不可定义的 R² 保留原因，不填 0。

普通运行的每折紧凑诊断位于 `runs/<组合>/diagnostics/repeat-NN-fold-NN.yaml`，可选逐行预测为同名 `-train.csv`，由 `run_manifest.yaml` 的 `diagnostics.folds` 通过相对路径和 SHA-256 引用。strict 运行的紧凑诊断位于任务根的 `diagnostics/<描述符>__<模型>__rNN__fNN.json`，可选逐行为同名 `-train.parquet`，由折元数据的 `diagnostic` 字段引用。报告只读这些证据；旧结果没有诊断时显示“未记录”，损坏或失败时显示原因，不从 OOF 或 HPO 内层分数反推训练成绩。离线重建报告不拟合模型，也不补算缺失诊断。

strict 报告的“全量外层留出样本与组追溯”区从保存的 split manifest 和预测分片重建。完整行位于 `report/diagnostic_sample_trace.csv` 与 `report/diagnostic_group_trace.csv`；报告展示预览并链接 CSV。逐样本表保留残差方向、组键和源行号；逐组表保留训练/留出数量、目标范围及误差。缺失、重复或错配的组身份显示 `unavailable`，单样本组标明不稳定性；不透明组键不能推断反应类型。

`yonod.diagnostic_repair` 是独立的**只预测后处理**，不执行 fit、HPO、特征 fit 或启动新的建模任务。它从已完成折的原始数据、模型和变换状态重算诊断，将证据写入新的 `result/<repair_task>/`；原 OOF、模型、study、manifest 和报告不改写，原报告也不会自动引用修复结果。仅支持有完整可核验模型包的 ordinary/strict × fixed/HPO 三树折；缺包、数据身份不符或保存的外层预测不一致时拒绝发布。所有正式建模仍须先建 `config/` 中的独立 schema-2 YAML，并通过 `yonod.py` 启动。

在仓库根、已激活 `yonod` 环境的 Bash 或 PowerShell 中，可对单个已保存折执行后处理（任务名和源路径按实际替换；日志保存在 `logs/`）：

```text
python -m yonod.diagnostic_repair --mode strict --source-root result/diag215_20260929_strict_fixed_v1 --dataset-path dataset/numeric_strict_smoke_fixture.csv --destination result/my_repair_task --feature morgan --model rf --repeat 1 --fold 1 > logs/my_repair_task.log 2>&1
```

成功时日志给出 `repair_manifest.json` 路径；同名目标目录已存在时会拒绝覆盖。修复产物中的 `fit_calls=0`、`search_calls=0` 是操作声明，验收通过拦截拟合/搜索调用、独立重载和源文件哈希对照来核实。

特征服务把静态矩阵、样本 ID、有效性掩码、身份和哈希发布到独立包。训练服务把每个 `feature × model` 的结果发布到 `outputs.root/runs/run_<feature-id>-<model>/`；内容派生的 `run_id` 保留在 manifest 中用于复用校验，其中包括：

- `predictions.csv` 与 `fold_metrics.csv`；
- `run_manifest.yaml`；
- 原始 `source_config.yaml` 和规范化 `effective_config.yaml`；
- 对按折变换特征，保存折内变换状态和审计信息。

同一身份的完整包会被验证后复用；任何已有不完整结果都不会被静默当作完成。失败记录位于组合目录的 `failed_attempts/` 下；同一组合目录已有不同身份结果时，运行会拒绝覆盖，应使用新的任务主目录。

### 特征

公开特征注册表当前包含：

- 静态分子/反应描述符：`morgan`、`mfp`、`maccs`、`fisd`、`molmetalm`、`maf`、`rdkit2d`、`drfp`。
- 已验证的静态分子描述符：`chemical_vae`（独立验收的 ZINC/v5 与 `zinc_properties` encoder/v5、原始 `z_mean_sample`；性质头不在描述符范围）。
- 按折变换特征：`ohe`。它保存原始类别输入；OneHotEncoder 只在每个外层训练折拟合，验证折只做转换。

`mfp` 是 Morgan **count** fingerprint，不是二值 `morgan` 的别名。它可显式设置 `radius`、`fp_size` 与 `profile`。`drfp` 需要满足反应输入约束；`fisd` 和 `molmetalm` 需要下文所述权重。

数值缩放、OHE、SVM 的 scaler/PCA/子采样，以及 XGBoost/LightGBM 的早停内部验证都限定在外层训练折内。外层验证折只用于最终预测。

### Chemical VAE（冻结、独立验收的 encoder）

Chemical VAE 是已接入、但刻意受限的可选描述符，并非 `example.yaml` 的默认描述符。它只接受各自经数值对照验证的 ZINC/v5 或 `zinc_properties` encoder/v5 conversion manifest，并从相邻的已哈希 PyTorch state 取编码器输出。声明时必须使用独立的 `chemical_vae` 描述符项，完整指定模型 manifest、`pytorch` 后端、设备、batch size、`identity` 输入预处理及 `z_mean_sample` 输出；`model_manifest` 以 YAML 所在目录为基准解析。`zinc_properties` 只开放 encoder 的原始 `z_mean_sample`，不加载其性质头；两种资产不能互相替代。它不加载解码器、性质头、训练 CSV、原 HDF5 或 TensorFlow/Keras；普通运行时只需要 PyTorch。

当前资产的每个分子列输出 196 维原始 `z_mean_sample`，两个列 `concat` 为 392 维。未知参数和未验证资产会失败而不是静默回退；历史完整 Chemical VAE YAML 位于 [配置归档](backup/configs_20260916.tar.gz)，不在 README 中复制。

输入语义是逐字节 identity：不 trim、不 canonicalize、不拆盐、不替换分隔符、不截断、不扩展字符表。适配器先检查固定 35 字符表、长度上限 120 和 RDKit 可解析性；缺失、超长、字符不支持或 RDKit 无效都会写入版本化 diagnostics sidecar。`concat` 保持既有语义：至少一个分子列成功时保留反应行，失败列为零块；所有选中列都失败才使该行的 mask 为 false。严格共同子集比较必须另行要求每个选中角色都成功，不能把零块留存集当作严格覆盖率。

步骤 31 的 Chemical VAE features/train/all、配对比较、计时/成本、GPU smoke 与 Windows 交接 YAML 均为历史工程验收输入，已保存在 [配置归档](backup/configs_20260916.tar.gz)。需要审计时，从仓库根解压该归档以恢复原始 `configs/` 路径；新任务应创建自己的独立 YAML，而不要修改或复用这些小样本验收输入。

示例和运行证据是工程验收，不是性能结论。Linux CPU 和两张可用 RTX 5090 中的 `cuda:0` 已真实运行；`zinc_properties` encoder 也已在冻结的 CPU float32 `atol=rtol=2e-5` 下经原 HDF5 TensorFlow/Keras 与独立 NumPy 对照后运行 GPU smoke。GPU 相对 CPU artifact 的差异单独记录，绝不回写成 C03 CPU parity。用户已授权将步骤31标记完成并延期 Windows CPU 验证；Windows CPU 仍没有实测，不能视为已支持。完整状态见 `derived/chemical_vae/step31_9_gpu_smoke/verification.json`、`derived/chemical_vae/step31_acceptance/index.json` 与 `derived/chemical_vae/step31_7_comparison/comparison_report.json`。

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

XGBoost 和 LightGBM 如需 early stopping，应在各自 `model_params.<model>.runtime.early_stopping` 中只声明由外层训练折产生的内部验证策略（轮数、验证比例和随机种子），不能把外部 `eval_set`、回调或数组传进 `fit`。

完整可运行字段示例见 [example.yaml](example.yaml)；参数路由与契约见 [Schema-2 契约](project-docs/docs/step30-contracts.md)。

## 特征版本操作

查看已发布的特征包，可调用 `yonod.artifacts.operations.inspect_feature_artifact(manifest_path)`；从操作 YAML 派生新版本，可调用 `yonod.artifacts.operations.derive_features(config_path)`。旧的命令行脚本已归档到本地 `backup/scripts/`。

`derive_features` 只支持显式样本选择、特征选择及按唯一 sample ID 连接数值 CSV。它永远发布新包并记录父版本、操作、输入/输出映射和哈希；标签、重复/缺失 ID、列冲突和不安全的按折变换操作会失败。

旧 JSON 运行配置不属于当前可执行入口。历史迁移器保存在本地 `backup/scripts/`，仅供审计；新建或恢复建模任务应参照 [example.yaml](example.yaml) 创建独立的 schema-2 YAML。

## 历史 CSV CLI

`main.py --csv ...` 仍保留为代码级调试/历史兼容接口，但不得用它启动建模任务。它的旧式 `docs/`、`pictures/`、`report/` 输出布局不属于 schema-2 产物或训练契约；新任务必须先保存独立 YAML，再由 `yonod.py` 启动。

## 专用 benchmark 与历史运行

专用执行器使用schema-2的`stage: benchmark`，公开启动仍为`yonod.py`。研究任务的`outputs.root`为独立`./result/<任务名>`，`artifacts.output_dir`必须是其中的`feature/`子目录，二者不能设置为同一目录。配置中给所选模型19CPU预算；Linux按[AGENTS.md](AGENTS.md)绑定恰好19个可用逻辑CPU后启动，完整后台命令见[示例操作说明](example/README.md)。启动形式如下：

```bash
source /home/wangzh685/miniconda3/etc/profile.d/conda.sh
conda activate yonod
benchmark_cpu_ids="$(python -c 'import os; ids=sorted(os.sched_getaffinity(0)); assert len(ids)>=19, "19 available logical CPUs are required"; print(",".join(map(str, ids[:19])))')"
if [ -n "$benchmark_cpu_ids" ]; then
mkdir -p logs
nohup setsid bash -lc '
  source /home/wangzh685/miniconda3/etc/profile.d/conda.sh
  conda activate yonod
  printf "%s\n" "config/my_benchmark.yaml" | taskset --cpu-list "$1" python -u yonod.py
' _ "$benchmark_cpu_ids" </dev/null > logs/my_benchmark.log 2>&1 &
echo $! > logs/my_benchmark.pid
fi
```

`manifest_outer_cv`要求显式`evaluation.grouping`和`benchmark.task_state: {backend: sqlite, resumable: true}`。提供`evaluation.split_manifest`时，程序核对CSV文件SHA、完整成员、折/轮次数、角色与组隔离，再导入执行；未提供时按分组规则生成清单。它不会回退到普通`outer_kfold`，但专用执行器也支持随机划分规则`repeated_kfold`，因此“strict”不自动表示未见底物。该路径需要`pyarrow`或`fastparquet`发布split、逐折预测及指标证据；`pyarrow`已列入`requirements.txt`。现有环境的固定版本安装方式为`conda install -y --override-channels -c conda-forge pyarrow=16.1.0 numpy=1.26.4`。多候选Tukey HSD另需`statsmodels>=0.14`；单模型/单描述符smoke不会调用它。

`frozen_train_external_test`使用单独的训练CSV、外测CSV、字段映射和来源核查文件；模型设置预先固定，每个编码与模型组合只用训练表建立一次模型，再评价有参考值的外测表。该协议不能混入CV折数、打乱或划分名单字段；当前不支持OHE、数值条件、HPO、AutoGluon或外测早停，也不保存可重用的最终模型包。来源独立性仍需研究者核查。报告为`report/external_test_report.html`和`.md`；固定名单/分组内测为`report/benchmark_report.html`和`.md`。

已完成的低成本strict smoke使用单独保存的12行人工fixture，经`yonod.py`产生两个按组分留出的内测外层折、SQLite状态、parquet预测/指标及HTML/Markdown报告；这些外层折不属于独立来源外测。配置与产物核验见`derived/interface_migration/step30_acceptance/benchmark_schema2_preflight_evidence.md`。

下列项目明确**不纳入迁移或完成标准**：所有 JSON 运行配置及依赖 JSON 的启动器（包括 paper-exact 和历史 YieldMaster shell 入口），以及 `different_order/amide-coupling/run_different_order_tasks.py`。它们保留为历史/归档材料，不是受支持的建模入口；不得为恢复这些路径而新增 JSON 兼容分支或补建 `different_order` runner。

迁移边界和验证证据见 [Schema-2 契约](project-docs/docs/step30-contracts.md) 与 `derived/interface_migration/step30_acceptance/acceptance_report.md`。已有的 JSON 指标、预测、run/fold manifest、官方资料和 `reference-proejct/` 内容是历史/审计材料，应保留原格式。

## 数据与外部资产

### 下载数据集与模型权重

在已激活 `yonod` 环境并安装 Hugging Face CLI（`hf`）后，从 YONOD 仓库根目录执行以下命令（Bash、PowerShell 或 Conda Prompt 均可）。

[YONOD-datasets](https://huggingface.co/datasets/thinktraveller/YONOD-datasets) 托管项目可分发的数据资产，其中 `USPTO/` 包含清洗和规范化后的 USPTO 数据。下载到本地 `dataset/USPTO/`：

```bash
hf download thinktraveller/YONOD-datasets --repo-type dataset --include "USPTO/**" --local-dir dataset
```

[YONOD-weights](https://huggingface.co/thinktraveller/YONOD-weights) 托管项目可分发的模型资产，目录布局与项目的 `WEIGHTS/` 一致。下载到本地 `WEIGHTS/`：

```bash
hf download thinktraveller/YONOD-weights --local-dir WEIGHTS
```

使用数据前请核对其来源与适用条款；使用模型权重前请核对上游模型的许可证与使用条款。

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

目录用途、保留边界和清理记录见 [目录管理说明](project-docs/docs/folder-organization.md)。按用途查找：[项目文档](project-docs/docs/README.md) · [配置归档](backup/configs_20260916.tar.gz) · [数据集](dataset/README.md)。

```text
YONOD/
├── main.py                     schema-2 底层配置运行时与 smoke 入口；保留历史 CSV 调试接口
├── yonod.py                    正式建模任务入口；生成/验证 schema-2 YAML
├── example.yaml                唯一可复制的 schema-2 YAML 规范与最小 smoke
├── example/                    三种已完整实跑的人工教学样例与下载包
│   ├── random_split/run.yaml   随机两折，RF/XGBoost/LightGBM
│   ├── fixed_split/run.yaml    预先指定两折名单，RF
│   └── external_test/run.yaml  训练12条后外测6条，RF
├── backup/                     不再活跃的结果、脚本和恢复材料（Git 忽略）
│   └── configs_20260916.tar.gz 历史 schema-2 与 Chemical VAE 验收 YAML
├── WEIGHTS/chemical_vae/       内容校验的转换后编码器资产（本地，不从 YAML 训练）
├── yonod/
│   ├── config/                 YAML 加载和 schema 契约
│   ├── artifacts/              不可变特征包的读写与派生
│   ├── pipeline/               features、train 与 all 编排
│   ├── model_factory.py        五模型参数路由与生效参数审计
│   ├── descriptors/            特征注册与描述符实现
│   └── benchmark/              固定/分组内测与冻结训练外测执行器
├── project-docs/               独立 Git 仓库：目标、计划、日志、学习笔记及 docs/
├── reference-projects/         本地参考项目（Git 忽略）
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

---

<a id="en"></a>

# YONOD — Organic Synthesis Reaction Prediction Platform

[中文](#zh) · [English](#en)

> **Y**ou **O**nly **N**eed **O**utstanding **D**escriptors
>
> Compare molecular descriptors and machine learning models for organic synthesis reaction prediction using SMILES, explicit column roles, and auditable feature versions.

## Current interface

Reproducible runs use schema-2 YAML. Start modeling tasks through `yonod.py`, which validates the saved configuration and passes it to the appropriate runtime:

```bash
python -u yonod.py
# At the first prompt, enter a saved YAML path, for example example.yaml.
```

The repository-root [example.yaml](example.yaml) is a runnable 12-record Morgan × Random Forest smoke configuration with two folds. A successful run reports `features=ready` and `morgan × rf: complete` or `reused`. Run IDs are derived from content; do not hard-code them. Historical Chemical VAE acceptance configurations are stored in [configs_20260916.tar.gz](backup/configs_20260916.tar.gz); they are not implicit parts of `example.yaml`.

The [three evaluation examples for chemistry researchers](example/index.html) are collected under [example/](example/README.md): random-split internal evaluation, predefined-membership two-fold internal evaluation, and an independent external-test workflow. All three `run.yaml` files have been executed from their own folders, with new outputs written to each package's `result/`. No configuration copy is needed. Downloadable packages include snapshots of the completed outputs. All example data are synthetic teaching records.

| Purpose | Current entry point |
| --- | --- |
| Run a saved, reproducible configuration | `python -u yonod.py`; enter the YAML path at the first prompt |
| Create and execute a configuration interactively | `python yonod.py` |
| Inspect or derive feature versions without modifying the source | `inspect_feature_artifact` / `derive_features` APIs in `yonod.artifacts.operations` |

`yonod.py` is an interactive wizard. It **does not parse** command-line options such as `--csv` or `--models`. It can accept a CSV, collect column roles, and write `*_run.yaml`, or validate and run an existing schema-2 YAML supplied at the first prompt. All modeling tasks must use a saved YAML configuration and start through `yonod.py`.

JSON is no longer an executable run configuration for `main.py` or the wizard. JSON configuration migration and JSON-dependent launchers are no longer maintained for compatibility. `main.py --json ...` explicitly rejects that input. Historical JSON metrics, fold audits, third-party material, and existing results retain their original formats; they are neither training entry points nor dependencies that need migration.

## Three teaching examples: complete runs from their own folders

[Example overview](example/index.html) · [Detailed instructions](example/README.md) · [Download all examples](example/archives/all_examples.zip)

All three examples completed on 2026-11-13 in the Linux `yonod` environment, starting through `yonod.py` and reading `run.yaml` from each example's own folder. Every process returned 0. Per-record predictions, split membership, scores, and HTML/Markdown reports passed acceptance checks. These teaching examples do not require copying their YAML into the root `config/` folder. Research tasks still require separate task configurations under the root `config/`.

| Method | Data and configuration | Completed computation | Report snapshot and acceptance record |
| --- | --- | --- | --- |
| Random-split internal evaluation | [12 records](example/random_split/dataset/reactions.csv), [run.yaml](example/random_split/run.yaml) | Randomly divide the same data into two parts and exchange their roles; two folds each for RF, XGBoost, and LightGBM | [Report](example/random_split/completed_run/report/report.html) · [Acceptance](example/random_split/verification.json) |
| Predefined-membership two-fold internal evaluation | [12 records](example/fixed_split/dataset/reactions.csv), [membership manifest](example/fixed_split/splits/fixed_split_manifest.csv), [run.yaml](example/fixed_split/run.yaml) | Follow predefined membership; six training and six held-out records per fold, then exchange roles; two RF folds | [Report](example/fixed_split/completed_run/report/benchmark_report.html) · [Acceptance](example/fixed_split/verification.json) |
| Independent external-test workflow | [12 training records](example/external_test/dataset/train.csv), [6 external-test records](example/external_test/dataset/external.csv), [run.yaml](example/external_test/run.yaml), [source audit](example/external_test/audit/external_audit.json) | Fix model settings in advance; fit RF once on the complete training table, then evaluate the external table, which contains three amine groups | [Report](example/external_test/completed_run/report/external_test_report.html) · [Acceptance](example/external_test/verification.json) |

From the selected example folder, launch `../../yonod.py` and enter `run.yaml` at the first prompt. Input paths are relative to that YAML's folder. New outputs go to the package-local `result/<task_name>/`, with numerical structure encodings under its `feature/` subfolder. On Linux, use a headless background process, exactly 19 available logical CPUs, and dedicated log/PID files in the repository-root `logs/`. Run tasks serially. Full commands are in the [instructions](example/README.md). When launching from an example folder, the entry point fixes the relative YAML path before changing the subprocess working directory, so the runtime reads the intended configuration.

Windows Conda Prompt example, starting at the repository root:

```text
conda activate yonod
cd example\random_split
python -u ..\..\yonod.py
```

Enter `run.yaml` at the first prompt. For either other example, enter its corresponding folder instead. These commands retain Windows usage, but this round of actual execution acceptance covered Linux only.

The download's `completed_run/` contains the complete output snapshot from this acceptance run. The active `result/` directory is excluded from the ZIP, leaving a fresh output destination after extraction. Neither runtime directory is tracked by source Git. A source checkout can generate new results by running the examples; extracting the download also provides the completed reports. The random package's `report/` separately retains the historical report from 2026-09-29. YAML files under `evidence/` document historical provenance and are not among the three active examples.

All data and yields are synthetic teaching records, not experimental measurements. Predefined membership does not automatically establish unseen-substrate evaluation. The current interface requires at least two folds and exactly one held-out appearance per record per repeat. A separately saved external CSV does not establish independent provenance either; this example demonstrates the external-test workflow. R², the coefficient of determination, is not prediction accuracy. For yields on a 0–1 scale, multiply MAE, the mean absolute error, by 100 to obtain percentage points. The random report scores pooled held-out predictions within each repeat, whereas the predefined-membership report summarizes fold-level metrics; do not directly rank results across these different aggregation rules.

## Quick start

### 1. Environment

The project baseline is Python 3.9. Before Linux development or testing, activate `yonod` in the same shell invocation:

```bash
source /home/wangzh685/miniconda3/etc/profile.d/conda.sh
conda activate yonod
```

For a new environment, install in the following order. Match the PyTorch CUDA wheel to the local CUDA/driver setup. CPU users should replace the wheel index with the official PyTorch CPU index.

```bash
conda create -n yonod python=3.9 -y
conda activate yonod

# Example: CUDA 12.1
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 \
  --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install "autogluon.tabular[lightgbm,catboost]==1.1.1"
```

`requirements.txt` lists the base dependencies. AutoGluon, LightGBM, FISD, and MolMetaLM may need additional assets or substantial disk space. Running only `example.yaml` does not require FISD or MolMetaLM weights.

### 2. Run the smoke configuration

Linux modeling tasks use saved YAML, `yonod.py`, and task-specific log/PID files. Do not launch a modeling task directly with `main.py --config`:

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

This configuration uses `dataset/benchmark_smoke_fixture.csv` from the repository and writes features, training outputs, and reports under `result/smoke-test/`:

```text
result/smoke-test/
├── feature/
│   ├── feature_runs/<feature_run_id>.yaml
│   └── features/<artifact_id>/
├── pictures/
│   └── scatter_morgan_rf_repeat-1.png
├── report/
│   ├── report.html
│   └── report.md
└── runs/
    └── run_morgan-rf/
        ├── run_manifest.yaml
        ├── source_config.yaml
        ├── effective_config.yaml
        ├── predictions.csv
        └── fold_metrics.csv
```

Paths resolve according to the configuration's location. For YAML under the root `config/`, references beginning with `./` resolve from the repository root. Explicit `../` references and configurations elsewhere, including `example/`, resolve from the YAML's own folder. Absolute paths remain absolute. `outputs.root` is the task root; the ordinary-run loader normalizes the effective `artifacts.output_dir` to its `feature/` subfolder. Startup logs and PID files belong in the repository's `logs/`. Do not reuse this smoke output for research. Research tasks need new YAML under `config/` and their own `result/<task_name>/`; the teaching examples above use `run.yaml` and `result/<task_name>/` inside their respective folders.

### 3. Create a configuration with the wizard

```bash
python yonod.py
```

The wizard validates the CSV, declares target/SMILES/numerical column roles, selects features and models, and requires a stable `sample_id` column with nonempty, unique values. Save the resulting `*_run.yaml` under the project's `config/` folder, then launch it through:

```bash
python -u yonod.py
# At the first prompt, enter path/to/project_run.yaml.
```

On Windows, activate the same environment through Conda Prompt. Application code uses configurable paths and `pathlib`. YAML references follow the location rules above, including the root `config/` exception for `./` references.

## Schema-2 YAML

The repository-root [example.yaml](example.yaml) is the sole authoritative, copyable, and validatable YAML specification. Create a separate schema-2 file under `config/` for each new research task, replacing only task-specific data, descriptors, models, evaluation settings, and isolated output paths. The teaching examples at `example/<example>/run.yaml` are an explicit exception that supports running in place. Do not maintain a second complete YAML specification in Markdown. Selected RF/XGBoost/LightGBM models use `model_params.<model>.estimator.n_jobs: 19`; AutoGluon uses `model_params.autogluon.fit.num_cpus: 19`. Declare only selected models, and run tasks serially.

Step 2-14 HPO passed engineering acceptance in the Linux `yonod` environment for ordinary RF/XGBoost/LightGBM, strict numerical and OHE entry points, soft time limits, resume, and independent final models. Windows hardware validation was deferred at the user's direction and remains `not_verified`. Omitting `hpo` or setting `enabled: false` preserves the fixed-parameter path and does not import Optuna; its dependency version is pinned to 4.5.0. Declare the budget explicitly with `max_trials`. Fields, search spaces, and the disabled-by-default example are defined only in [example.yaml](example.yaml). Each outer-training-fold HPO study and its inner folds are persisted separately under the task's `result/<task_name>/hpo/`. Inner search scores are not outer out-of-fold evaluation scores, and a higher search score is not an acceptance criterion. The full Linux acceptance entry point is `_verify/accept_hpo_all.py`; its script and local evidence are excluded from the main repository under project conventions.

Every run configuration declares a schema version, project name, stage, dataset, and stable sample ID. The parser accepts a single UTF-8 `.yaml`/`.yml` document and rejects duplicate keys, aliases, unknown fields, and unsupported model/parameter sections. `dataset.column_roles.label` is required for `train` and `all`; feature-only runs may omit it. `sample_id_col` must exist in the dataset and contain unique, nonempty values. DataFrame row numbers cannot replace it.

### Stages

| `stage` | Required inputs | Behavior |
| --- | --- | --- |
| `features` | `dataset`, `descriptors`, `artifacts.output_dir` | Initialize feature outputs and compute or reuse feature packages. No models are constructed and labels are not required. Explicitly declaring `outputs.root` is also recommended. |
| `train` | Labeled `dataset`, `artifacts.input_manifest`, `models` | Read only the specified feature packages and train. Never fall back to descriptor computation. |
| `all` | Declarations required by both features and training | First bring every candidate feature to a ready/failed terminal state, then train all ready packages. Publish each ready feature × model combination independently. |
| `benchmark` | Labeled `dataset`, structure inputs, models, and protocol-specific `evaluation`/`benchmark` declarations | Use `manifest_outer_cv` for grouped/predefined-membership internal evaluation, or `frozen_train_external_test` for one external test after training. Each protocol has its own constraints. |

External feature manifests, labels, and auxiliary columns are aligned by stable `sample_id`. Static feature identities exclude labels, so changing only labels changes the training identity rather than the feature identity.

If an identified feature × model combination fails during actual training, YONOD atomically publishes a non-overwritable manifest under `outputs.root/failed_runs/` containing the reason, original YAML, and effective YAML, then continues with sibling combinations. Failed combinations do not publish incomplete predictions/metrics. Any failed combination makes `main.py --config` return 2, preventing schedulers from treating partial success as complete success. Actual fixture evidence is in `derived/interface_migration/step30_acceptance/training_failure_isolation_evidence.md`.

The six historical local acceptance records listed below as `derived/...` paths are not included in the current checkout. Their paths are retained for traceability rather than as download links; those historical acceptance runs were not repeated for this documentation update.

### Reports and offline rebuilding

For schema-2 `train` / `all`, omitting `outputs.report_formats` generates both HTML and Markdown. Supported values are `html` and `markdown` (or `md`); an empty list disables automatic reports. Each completed run generates `pictures/` and `report/` from its own `run_manifest.yaml`, predictions, fold metrics, and YAML snapshots. Scatter plots use each repeat's out-of-fold predictions. HTML embeds PNG images; Markdown uses `../pictures/` relative links. The `features` stage does not generate reports.

Rebuild ordinary result reports with `yonod.pipeline.reporting.rebuild_schema2_report` without the original dataset, descriptor weights, or models. It validates manifests, configuration snapshot hashes, out-of-fold identities, and finite values; it neither trains nor rewrites source results. Old dedicated scripts are archived locally under `backup/scripts/` and are no longer current entry points.

Completed HPO combinations also include a “Hyperparameter search and nested evaluation” section: study/trial histories, actual parameters and their sources for every fold, independent outer-fold metrics, and separate search, outer-training, and prediction times. Ordinary summary tables still use pooled out-of-fold predictions within each repeat; do not equate these with mean fold scores or inner search scores. Explicit final models show development-set search and refitting separately, with no test score when there is no independent test. The report rebuilder verifies hashes of persisted study exports without opening databases or refitting. HPO failure/soft-time-limit reports at `report/hpo_status.html` and `.md` display status and verified search evidence only; incomplete folds are not presented as complete out-of-fold evaluation.

### Training and outer-evaluation diagnostics — Step 2-15, accepted Linux scope for three tree models

Nine isolated small-data tasks covered ordinary and strict RF, XGBoost, and LightGBM, fixed settings/HPO, diagnostics on/off, and ordinary early stopping. Outer-fold predictions, diagnostic values, model reloads, and both report formats were checked independently. `logs/diag215_20260929_acceptance_resume5.log` retains the unified acceptance record. Historical SVM/AutoGluon results remain readable and explicitly display diagnostics as “not recorded.” AutoGluon was not refitted in that acceptance round, and Windows hardware remained unverified. These engineering checks do not establish model performance on real chemistry data.

`outputs.diagnostics.enabled` controls fold-level diagnostics for RF, XGBoost, and LightGBM (default `true`). `outputs.diagnostics.save_train_predictions` controls whether per-record training predictions are saved (default `false`; acceptance tasks should set it to `true`). The sole complete YAML structure is in `example.yaml`. These options control diagnostics only; they do not change the outer out-of-fold file format or scoring rules. SVM and AutoGluon do not yet have diagnostics on the same basis; their existing modeling capabilities remain available.

Diagnostics reuse the final outer-fold model and fitted feature state. They compute outer-training scores, outer-held-out scores, and a held-out baseline that always predicts the arithmetic mean of that fold's outer-training labels. The R² gap is training minus outer-held-out; RMSE/MAE gaps are outer-held-out minus training. With early stopping, diagnostics also record the actual fit records and the internal early-stopping holdout. Training scores describe fit to previously seen records, not generalization. Units are marked unknown if undeclared. Undefined R², such as for constant labels, retains its reason rather than being replaced with 0.

Ordinary per-fold compact diagnostics are at `runs/<combination>/diagnostics/repeat-NN-fold-NN.yaml`, with optional per-record predictions in the corresponding `-train.csv`. The `diagnostics.folds` entries of `run_manifest.yaml` reference them by relative path and SHA-256. Strict compact diagnostics are at `diagnostics/<descriptor>__<model>__rNN__fNN.json` under the task root, with optional per-record `-train.parquet`; fold metadata references them through `diagnostic`. Reports read this evidence only. Missing historical diagnostics display “not recorded,” while corruption or failures retain their reasons. Training performance is not inferred from out-of-fold or HPO inner-fold scores. Offline report rebuilding neither fits models nor supplies missing diagnostics.

The strict report's “Complete outer-held-out sample and group trace” section is rebuilt from saved split manifests and prediction shards. Full rows are in `report/diagnostic_sample_trace.csv` and `report/diagnostic_group_trace.csv`; the report shows previews and links to the CSVs. Per-record tables retain residual direction, group keys, and source row indices. Per-group tables retain training/held-out counts, target ranges, and errors. Missing, duplicate, or mismatched group identities display `unavailable`; single-record groups are marked unstable. Opaque group keys cannot establish reaction types.

`yonod.diagnostic_repair` is a separate **prediction-only postprocessing operation**. It does not fit models, run HPO, fit features, or launch new modeling tasks. It recomputes diagnostics from the original data, model, and transformation state of a completed fold and writes evidence to a new `result/<repair_task>/`. Original out-of-fold predictions, models, studies, manifests, and reports remain unchanged, and the old report does not automatically reference the repair. It supports only ordinary/strict × fixed/HPO folds for the three tree models when complete, verifiable model bundles exist. Missing bundles, mismatched data identities, or disagreement with saved outer-fold predictions prevent publication. All formal modeling still requires a separate schema-2 YAML under `config/` and launch through `yonod.py`.

From the repository root in Bash or PowerShell, with `yonod` activated, postprocess one saved fold as follows. Replace the task name and source paths as appropriate; logs belong in `logs/`:

```text
python -m yonod.diagnostic_repair --mode strict --source-root result/diag215_20260929_strict_fixed_v1 --dataset-path dataset/numeric_strict_smoke_fixture.csv --destination result/my_repair_task --feature morgan --model rf --repeat 1 --fold 1 > logs/my_repair_task.log 2>&1
```

On success, the log provides the `repair_manifest.json` path. An existing destination directory is not overwritten. The repair outputs' `fit_calls=0` and `search_calls=0` are operational declarations; acceptance verifies them by intercepting fit/search calls, independently reloading outputs, and comparing source file hashes.

The feature service publishes static matrices, sample IDs, validity masks, identities, and hashes in separate packages. The training service publishes each feature × model result under `outputs.root/runs/run_<feature-id>-<model>/`. The content-derived `run_id` remains in the manifest for reuse validation. Outputs include:

- `predictions.csv` and `fold_metrics.csv`;
- `run_manifest.yaml`;
- the original `source_config.yaml` and normalized `effective_config.yaml`;
- fold-local transformation states and audits for features transformed within each fold.

A complete package with the same identity is validated before reuse. Incomplete existing results are never silently treated as complete. Failure records are stored under the combination directory's `failed_attempts/`. If the same combination directory already contains a different result identity, the run refuses to overwrite it; use a new task root.

### Features

The public feature registry currently includes:

- Static molecular/reaction descriptors: `morgan`, `mfp`, `maccs`, `fisd`, `molmetalm`, `maf`, `rdkit2d`, and `drfp`.
- Validated static molecular descriptor: `chemical_vae`, using the independently accepted ZINC/v5 and `zinc_properties` encoder/v5 assets and raw `z_mean_sample`; property heads are outside its scope.
- Fold-transformed features: `ohe`. It stores raw categorical inputs; OneHotEncoder is fitted only on each outer-training fold, and the validation fold is transformed only.

`mfp` is a Morgan **count** fingerprint, not an alias for binary `morgan`. It accepts explicit `radius`, `fp_size`, and `profile`. `drfp` requires compatible reaction inputs; `fisd` and `molmetalm` require the weights described below.

Numerical scaling, OHE, SVM scaling/PCA/subsampling, and internal early-stopping validation for XGBoost/LightGBM are restricted to the outer-training fold. The outer-validation fold is used only for final prediction.

### Chemical VAE — frozen encoder with independent acceptance

Chemical VAE is an integrated but deliberately restricted optional descriptor, not the default descriptor in `example.yaml`. It accepts only separately validated ZINC/v5 or `zinc_properties` encoder/v5 conversion manifests and obtains encoder outputs from adjacent, hashed PyTorch state. Declare it as a separate `chemical_vae` descriptor, specifying the model manifest, `pytorch` backend, device, batch size, `identity` input preprocessing, and `z_mean_sample` output. `model_manifest` resolves relative to the YAML's folder. `zinc_properties` exposes only the encoder's raw `z_mean_sample`, without loading its property head; the two assets are not interchangeable. It does not load decoders, property heads, training CSVs, original HDF5, or TensorFlow/Keras; ordinary runtime requires only PyTorch.

The current assets produce 196 raw `z_mean_sample` dimensions per molecular column, or 392 for two columns in `concat` mode. Unknown parameters or unverified assets fail explicitly rather than silently falling back. Historical complete Chemical VAE YAML is in the [configuration archive](backup/configs_20260916.tar.gz), not duplicated in this README.

Inputs use byte-for-byte identity semantics: no trimming, canonicalization, salt splitting, delimiter replacement, truncation, or vocabulary extension. The adapter first checks the fixed 35-character vocabulary, maximum length 120, and RDKit parseability. Missing, overlong, unsupported-character, or RDKit-invalid inputs are recorded in a versioned diagnostics sidecar. `concat` preserves existing semantics: retain a reaction row if at least one molecular column succeeds, replacing failed columns with zero blocks. Only failure of all selected columns makes the row's mask false. Strict common-subset comparisons must separately require success for every selected role; a retained set containing zero blocks is not strict coverage.

The Step 31 Chemical VAE features/train/all, paired comparisons, time/cost, GPU smoke, and Windows handoff YAML files are historical engineering acceptance inputs in the [configuration archive](backup/configs_20260916.tar.gz). For audits, extract it from the repository root to restore the original `configs/` paths. New tasks need their own separate YAML, rather than editing or reusing those small acceptance inputs.

Examples and execution evidence establish engineering acceptance, not predictive performance. Actual runs covered Linux CPU and `cuda:0`, one of the two available RTX 5090 GPUs. The `zinc_properties` encoder also ran a GPU smoke after comparison against original HDF5 TensorFlow/Keras and independent NumPy outputs under the frozen CPU float32 tolerance `atol=rtol=2e-5`. GPU-to-CPU-artifact differences are recorded separately and are never relabeled as C03 CPU parity. The user authorized marking Step 31 complete while deferring Windows CPU validation. Windows CPU has not been tested and must not be described as verified support. Full status is in `derived/chemical_vae/step31_9_gpu_smoke/verification.json`, the `derived/chemical_vae/step31_acceptance/index.json`, and the `derived/chemical_vae/step31_7_comparison/comparison_report.json`.

#### Chemical VAE dependencies and numerical comparisons

`h5py` in `requirements.txt` supports asset audits/conversion; it is not a required import for PyTorch runtime. TensorFlow/Keras must not become prerequisites for ordinary Chemical VAE feature generation, Morgan/MFP, or train-only manifest consumption.

### Models and parameters

Supported model identifiers are `rf`, `xgb`, `svm`, `lightgbm`, and `autogluon`. YAML loading normalizes model aliases. To avoid ambiguous experiment identities, do not specify an alias and its canonical name together.

Model parameters belong in the explicit API sections below. Omitted fields use the installed supported library's defaults; `null`, `false`, and `0` remain explicit values.

| Model | Available sections |
| --- | --- |
| `rf`, `xgb`, `lightgbm` | `estimator`, `fit`, `runtime` |
| `svm` | `estimator`, `fit`, `preprocessing`, `runtime` |
| `autogluon` | `predictor`, `fit`, `runtime` |

For XGBoost or LightGBM early stopping, declare only the internal validation strategy derived from the outer-training fold—rounds, validation fraction, and seed—under the model's `model_params.<model>.runtime.early_stopping`. Do not pass an external `eval_set`, callbacks, or arrays through `fit`.

Complete runnable fields are defined in [example.yaml](example.yaml); parameter routing and contracts are documented in the [Schema-2 contract](project-docs/docs/step30-contracts.md).

## Feature version operations

Inspect a published feature package with `yonod.artifacts.operations.inspect_feature_artifact(manifest_path)`. Derive a new version from an operations YAML with `yonod.artifacts.operations.derive_features(config_path)`. Old command-line scripts are archived locally under `backup/scripts/`.

`derive_features` supports only explicit sample selection, feature selection, and numerical CSV joins using unique sample IDs. It always publishes a new package and records the parent version, operations, input/output mappings, and hashes. Labels, duplicate/missing IDs, column conflicts, and unsafe fold-transformation operations are rejected.

Old JSON run configurations are not current executable entry points. Historical migration tools remain under local `backup/scripts/` for audits only. Create a separate schema-2 YAML following [example.yaml](example.yaml) for a new or resumed modeling task.

## Historical CSV CLI

`main.py --csv ...` remains a code-level debugging/historical compatibility interface, but must not launch modeling tasks. Its old `docs/`, `pictures/`, and `report/` layout is outside the schema-2 artifact and training contracts. New tasks must first save a separate YAML and then launch through `yonod.py`.

## Dedicated benchmarks and historical runs

Dedicated executors use schema-2 `stage: benchmark` and still start publicly through `yonod.py`. For research tasks, `outputs.root` is a separate `./result/<task_name>`, while `artifacts.output_dir` must be its `feature/` subfolder; they cannot be the same directory. Declare a 19-CPU budget for selected models. On Linux, bind exactly 19 available logical CPUs as required by [AGENTS.md](AGENTS.md); complete background-launch instructions are in the [example guide](example/README.md). The launch pattern is:

```bash
source /home/wangzh685/miniconda3/etc/profile.d/conda.sh
conda activate yonod
benchmark_cpu_ids="$(python -c 'import os; ids=sorted(os.sched_getaffinity(0)); assert len(ids)>=19, "19 available logical CPUs are required"; print(",".join(map(str, ids[:19])))')"
if [ -n "$benchmark_cpu_ids" ]; then
mkdir -p logs
nohup setsid bash -lc '
  source /home/wangzh685/miniconda3/etc/profile.d/conda.sh
  conda activate yonod
  printf "%s\n" "config/my_benchmark.yaml" | taskset --cpu-list "$1" python -u yonod.py
' _ "$benchmark_cpu_ids" </dev/null > logs/my_benchmark.log 2>&1 &
echo $! > logs/my_benchmark.pid
fi
```

`manifest_outer_cv` requires explicit `evaluation.grouping` and `benchmark.task_state: {backend: sqlite, resumable: true}`. When `evaluation.split_manifest` is provided, the program checks the CSV file SHA, complete membership, fold/repeat counts, roles, and group separation before importing and executing it. Otherwise, it generates a manifest from the grouping rules. It never falls back to ordinary `outer_kfold`. However, the dedicated executor also supports the random-split rule `repeated_kfold`, so “strict” does not automatically mean unseen substrates. This path needs `pyarrow` or `fastparquet` to publish splits, per-fold predictions, and metric evidence; `pyarrow` is listed in `requirements.txt`. A pinned installation command for an existing environment is `conda install -y --override-channels -c conda-forge pyarrow=16.1.0 numpy=1.26.4`. Multiple-candidate Tukey HSD additionally requires `statsmodels>=0.14`; a single-model/single-descriptor smoke does not invoke it.

`frozen_train_external_test` uses separate training and external-test CSVs, column mappings, and a source-audit file. Fix model settings in advance. For each encoding × model combination, fit once on the training table and then evaluate an external table with reference values. This protocol cannot contain CV fold-count, shuffling, or split-manifest fields. It currently does not support OHE, numerical conditions, HPO, AutoGluon, or external-test early stopping, and does not save a reusable final model bundle. Researchers must still establish independent provenance. Its reports are `report/external_test_report.html` and `.md`; predefined-membership/grouped internal evaluation reports are `report/benchmark_report.html` and `.md`.

A completed low-cost strict smoke used a separately saved 12-record synthetic fixture. Through `yonod.py`, it produced two component-held-out outer folds in internal cross-validation, SQLite state, parquet predictions/metrics, and HTML/Markdown reports. Those outer folds are not an independent external-source test. Configuration and output checks are in `derived/interface_migration/step30_acceptance/benchmark_schema2_preflight_evidence.md`.

The following are explicitly **outside migration and completion criteria**: all JSON run configurations and JSON-dependent launchers, including paper-exact and historical YieldMaster shell entry points, and `different_order/amide-coupling/run_different_order_tasks.py`. They remain historical/archive material, not supported modeling entry points. Do not add JSON compatibility branches or recreate the `different_order` runner to restore them.

Migration boundaries and verification evidence are in the [Schema-2 contract](project-docs/docs/step30-contracts.md) and `derived/interface_migration/step30_acceptance/acceptance_report.md`. Existing JSON metrics, predictions, run/fold manifests, official material, and `reference-proejct/` contents are historical/audit material and retain their original formats.

## Data and external assets

### Download datasets and model weights

With `yonod` activated and the Hugging Face CLI (`hf`) installed, run these commands from the YONOD repository root in Bash, PowerShell, or Conda Prompt.

[YONOD-datasets](https://huggingface.co/datasets/thinktraveller/YONOD-datasets) hosts redistributable project data assets, including cleaned and standardized USPTO data under `USPTO/`. Download into local `dataset/USPTO/`:

```bash
hf download thinktraveller/YONOD-datasets --repo-type dataset --include "USPTO/**" --local-dir dataset
```

[YONOD-weights](https://huggingface.co/thinktraveller/YONOD-weights) hosts redistributable model assets with a layout matching the project's `WEIGHTS/`. Download into local `WEIGHTS/`:

```bash
hf download thinktraveller/YONOD-weights --local-dir WEIGHTS
```

Check provenance and applicable terms before using data. Check upstream licenses and terms before using model weights.

### Amide-coupling dataset

The repository's `dataset/amide-coupling.csv` comes from [aichemeco/amide_coupling](https://github.com/aichemeco/amide_coupling/tree/main), under the MIT license. It contains 47,015 amide-coupling reactions, with `yield` normalized to `[0, 1]`. `dataset/benchmark_smoke_fixture.csv` is the small schema-2 smoke fixture. This README does not assume that `dataset/test-amide-coupling.csv` exists.

For publications using this dataset, cite Dai *et al.*, *Chem. Sci.*, 2025, DOI [10.1039/D5SC03364K](https://doi.org/10.1039/D5SC03364K).

| Column | Type | Meaning |
| --- | --- | --- |
| `row_id` | int | Candidate sample ID, subject to uniqueness validation |
| `sub_1_smiles`, `sub_2_smiles` | str | Substrate SMILES |
| `product_smiles` | str | Product SMILES; use is determined by YAML feature declarations |
| `activation`, `additive`, `base`, `solvent` | str | Reaction-component SMILES; feature declarations must handle missing and multicomponent semantics |
| `yield` | float | Regression target |

### MolMetaLM weights

- Hugging Face: [wudejian789/MolMetaLM-base](https://huggingface.co/wudejian789/MolMetaLM-base)
- GitHub: [CSUBioGroup/MolMetaLM](https://github.com/CSUBioGroup/MolMetaLM)
- Paper: DOI [10.48550/arXiv.2411.15500](https://doi.org/10.48550/arXiv.2411.15500)

```bash
pip install -U "huggingface_hub>=0.34"
hf download wudejian789/MolMetaLM-base --local-dir WEIGHTS/MolMetaLM-base
```

The download is approximately 500 MB. The directory should contain `config.json`, `model.safetensors`, tokenizer files, and related model assets.

### FISD weights

FISD requires three GCN weight files pretrained on QM9, approximately 19 MB in total. Obtain them from [KeantChen/FISD model](https://github.com/KeantChen/FISD/tree/main/model) and place them under `WEIGHTS/FISD/`:

```text
WEIGHTS/FISD/
├── qm_9_mse_model.pth
├── qm_9_cos_model.pth
└── qm_9_2in1_model.pth
```

Related paper: DOI [10.1039/D5SC00451A](https://doi.org/10.1039/D5SC00451A). Selecting `fisd` without these files causes an explicit feature-computation failure; other independent features can still complete.

### ATMOMACCS / MACCS

The project's `maccs` feature uses standard RDKit MACCS keys: 166 dimensions after removing placeholder bit 0 from the 167-bit representation. No ATMOMACCS asset download is required. If citing the corresponding experimental results, also follow the license and citation requirements of upstream [Zenodo 18669279](https://zenodo.org/records/18669279) and paper [10.1063/5.0308548](https://doi.org/10.1063/5.0308548).

## Project layout

Directory purposes, retention boundaries, and cleanup history are documented in the [directory guide](project-docs/docs/folder-organization.md). Browse [project documents](project-docs/docs/README.md), the [configuration archive](backup/configs_20260916.tar.gz), or [datasets](dataset/README.md).

```text
YONOD/
├── main.py                     Schema-2 runtime and smoke entry; historical CSV debugging interface
├── yonod.py                    Modeling-task entry point; generates/validates schema-2 YAML
├── example.yaml                Sole copyable schema-2 YAML specification and minimal smoke
├── example/                    Three executed synthetic teaching examples and downloads
│   ├── random_split/run.yaml   Random two-fold evaluation, RF/XGBoost/LightGBM
│   ├── fixed_split/run.yaml    Predefined two-fold membership, RF
│   └── external_test/run.yaml  Train on 12 records, externally test 6, RF
├── backup/                     Inactive results, scripts, and recovery material (Git-ignored)
│   └── configs_20260916.tar.gz Historical schema-2 and Chemical VAE acceptance YAML
├── WEIGHTS/chemical_vae/        Content-verified converted encoders (local; not trained from YAML)
├── yonod/
│   ├── config/                 YAML loading and schema contracts
│   ├── artifacts/              Immutable feature-package reading, writing, and derivation
│   ├── pipeline/               features, train, and all orchestration
│   ├── model_factory.py        Five-model parameter routing and effective-parameter audits
│   ├── descriptors/            Feature registry and descriptor implementations
│   └── benchmark/              Predefined/grouped internal and frozen-training external executors
├── project-docs/               Separate Git repository: goals, plan, build log, learning notes, docs/
├── reference-projects/         Local reference projects (Git-ignored)
├── dataset/                    Datasets and smoke fixtures
└── derived/                    Local acceptance and generated artifacts
```

## Verification and contribution notes

Run from the repository root with the `yonod` environment active:

```bash
python -m unittest tests.test_chemical_vae_audit tests.test_chemical_vae_encoder_parity \
  tests.test_chemical_vae_descriptor tests.test_chemical_vae_artifact_lineage \
  tests.test_chemical_vae_training_isolation tests.test_mfp_descriptor -v
python -m unittest tests.test_step30_contracts tests.test_step30_yaml_loader \
  tests.test_step30_feature_service tests.test_step30_training_service -v
```

Do not overwrite committed datasets, historical run outputs, feature packages, or `reference-proejct/`. Development takes place on Linux, but application code must remain Windows-compatible: use `pathlib.Path`/`os.path`, avoid hard-coded path separators, and preserve UTF-8 Chinese-path support.

## License

Project code and documentation use [CC BY-NC 4.0](LICENSE). Third-party data, model weights, and reference projects retain their own licenses. Check the relevant source's terms and citation requirements before using assets or publishing results.
