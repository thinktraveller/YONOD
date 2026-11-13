# YONOD — 化学反应数值预测与建模方法比较

[English](README.md) · **简体中文**

**You Only Need Outstanding Descriptors.** 从已有反应记录出发，将分子结构和记录条件转换为模型输入，建立产率等数值目标的预测模型，并输出逐条预测、误差和可追查的报告。

YONOD 面向化学研究人员，支持比较不同结构数值编码（descriptors）与回归方法。它需要数据中已有的参考值进行建模和评价；不会自动生成化学产物、推导反应机理或替你寻找最佳实验条件。

## 1. 快速上手

建议先完成[随机划分教学样例](example/random_split/README.md)，再查看固定名单与外测。三个样例均使用已随仓库提供的人工数据和 Morgan 编码，**不需要下载预训练模型权重**。

### 1.1 获取项目并准备环境

先安装 Git 和 Conda。项目以 Python 3.9 为基线；首次使用：

```bash
git clone https://github.com/thinktraveller/YONOD.git
cd YONOD
conda create -n yonod python=3.9 -y
conda activate yonod
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

这里安装的是 CPU 版 PyTorch，足以运行三个教学样例。需要 CUDA 时，按 [requirements.txt](requirements.txt) 中的版本说明选择与本机匹配的 PyTorch 安装源。LightGBM 和 SciPy 已列入 `requirements.txt`，PyTorch 仍需事先单独安装。只有使用 AutoGluon 时才需要额外安装：

```bash
pip install "autogluon.tabular[lightgbm,catboost]==1.1.1"
```

已有 `yonod` 环境可跳过创建和安装。Linux 每个新 shell 都应先初始化并激活 Conda；下面假设安装在 `~/miniconda3`，其他位置请相应修改。Windows 使用 Conda Prompt 激活环境。

### 1.2 在样例自己的文件夹启动

Linux Bash，在仓库根执行以下完整命令。它选择恰好19个可用逻辑 CPU，并将日志和 PID 保存到仓库根 `logs/`：

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate yonod
cd example/random_split

yonod_repo_root="$(cd ../.. && pwd)"
yonod_cpu_ids="$(python -c 'import os; ids=sorted(os.sched_getaffinity(0)); assert len(ids)>=19, "19 available logical CPUs are required"; print(",".join(map(str, ids[:19])))')"
if [ -n "$yonod_cpu_ids" ]; then
  mkdir -p "$yonod_repo_root/logs"
  nohup setsid bash -lc '
    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate yonod
    cd "$1" || exit 1
    printf "%s\n" "run.yaml" | taskset --cpu-list "$2" python -u ../../yonod.py
  ' _ "$PWD" "$yonod_cpu_ids" </dev/null > "$yonod_repo_root/logs/example_random_split_20261113.log" 2>&1 &
  echo $! > "$yonod_repo_root/logs/example_random_split_20261113.pid"
else
  echo "Fewer than 19 logical CPUs are available; task was not started."
fi
```

Windows Conda Prompt，从仓库根执行：

```text
conda activate yonod
cd example\random_split
python -u ..\..\yonod.py
```

首次提示输入 `run.yaml`。Linux 命令已通过标准输入提供同一答案。正式入口是 `yonod.py`；不要把 `--csv`、`--models` 或 `--config` 当作这个交互入口的参数。

**一次只运行一个任务。** 等当前任务完成后再运行另一个样例；固定名单与外测分别进入 `example/fixed_split/`、`example/external_test/`，同时更换日志和 PID 文件名。三个样例已在 Linux 完整运行并验收；Windows 命令说明用法，不代表所有可选后端都经过 Windows 实机验证。

### 1.3 查看进度与报告

仍在随机样例目录时：

```bash
task_pid="$(cat ../../logs/example_random_split_20261113.pid)"
ps -p "$task_pid" -o pid,ppid,stat,etime,cmd
taskset --pid --cpu-list "$task_pid"
tail -n 60 ../../logs/example_random_split_20261113.log
```

进程结束后，以日志中的完成状态及运行记录为准：普通流程应看到 `features=ready` 和三个模型组合 `complete` 或 `reused`；固定名单应有两折成功记录；外测应有完整预测表及两种报告。运行编号由内容决定，不应匹配固定编号。

新结果保存在**当前样例目录**的 `result/<任务名>/`。打开其中 `report/` 下的 HTML 报告；各协议的具体文件名见下一节。

## 2. 跟着 example 逐步理解数据、设置和输出

[离线导览](example/index.html) · [全部操作说明](example/README.md) · [下载三个完整样例](example/archives/all_examples.zip)

数据和产率全部是**人工教学值，不是实验测量**；分子搭配只展示输入格式，不声称代表真实反应。已保存的[随机](example/random_split/verification.json)、[固定名单](example/fixed_split/verification.json)、[外测](example/external_test/verification.json)验收记录证明流程完整运行，不能据此评价真实化学预测能力。

### 2.1 随机划分内测：从一张表开始

1. 打开 [reactions.csv](example/random_split/dataset/reactions.csv)。每行一条记录：`reaction_id` 是唯一编号，`yield` 是参考产率，两列 `*_smiles` 是分子结构文本（SMILES）。例如 `yield=0.10` 表示10%，不是0.10%。
2. 打开同目录 [run.yaml](example/random_split/run.yaml)。它指定两列结构、Morgan 编码、随机森林（Random Forest，RF）及 XGBoost、LightGBM 三种方法，以及随机两折轮换：先6条建模、6条检验，再交换角色。
3. 按快速上手启动本目录的 `run.yaml`。首次完整运行时三个模型各完成两折，共6次拟合；身份匹配的已有结果可以复用。
4. 查看 `result/example_random_split_20261113/report/report.html` 或 `report.md`；逐条留出预测位于 `runs/run_morgan-<模型>/predictions.csv`。

每条记录都由没有学习该记录的相应模型预测一次，这些预测汇总称为留出反应的内测预测（out-of-fold，OOF）。随机划分不保证检验中的底物从未出现在建模数据里。

### 2.2 固定名单内测：提前决定谁用于检验

1. 查看 [数据表](example/fixed_split/dataset/reactions.csv)与[角色解释表](example/fixed_split/splits/membership_guide.csv)。`fixed_block` 将12条记录分为两个教学组，每组6条；分组列不作为模型输入。
2. 查看程序实际读取的[固定成员清单](example/fixed_split/splits/fixed_split_manifest.csv)和 [run.yaml](example/fixed_split/run.yaml)。`stage: benchmark` 与 `manifest_outer_cv` 才会执行名单；仅固定随机种子不等于指定成员。
3. 在 `example/fixed_split/` 启动 `run.yaml`。第1折留出 `block_A`，第2折留出 `block_B`；RF 共建立两次模型。
4. 查看 `result/example_fixed_split_20261113/report/benchmark_report.html` 或 `.md`；实际采用的名单和预测保存在 `docs/manifests/`、`docs/predictions/`。

当前接口要求至少两折，每条记录每轮恰好留出一次，不能把本例改成仅一次“8条训练＋4条检验”的名单。修改数据后必须同步核对记录编号、源行号、成员、组隔离和 CSV 文件 SHA-256；详见[固定名单说明](example/fixed_split/README.md)。教学组名不具有底物或反应类型的化学含义。

### 2.3 外测流程：训练表与检验表分开

1. 对照 [train.csv](example/external_test/dataset/train.csv) 的12条训练记录与 [external.csv](example/external_test/dataset/external.csv) 的6条外测记录。两表列名可以不同；外测另有来源编号和3个胺组。
2. 在 [run.yaml](example/external_test/run.yaml) 查看 `external_test.column_map`：训练列 `amine_smiles` 对应外测列 `external_amine`，其他结构列同理。参考产率分别由训练 `label` 和外测 `label_col` 指定，不放入结构映射。
3. 核对[来源审计文件](example/external_test/audit/external_audit.json)、CSV 和审计文件的校验码。真实研究应重新核实来源和结构重叠，不能照抄示例中的零值。
4. 在 `example/external_test/` 启动 `run.yaml`。模型设置预先固定，RF 使用全部12条训练记录建立一次模型，再预测6条外测记录；外测参考值用于最后评分。
5. 查看 `result/example_external_test_20261113/report/external_test_report.html` 或 `.md`；预测为 `docs/predictions/morgan__rf.parquet`，指标为 `docs/metrics/external_test_metrics.csv`。

本例同时使用胺、酸和产物结构，真实使用时需确认预测时目标产物结构已知。当前外测适配器要求审计声明的胺、酸、产物精确规范化结构交集均为0，外测至少有两个非空组。本例实际检查了交集，但人工数据没有独立实验来源；另存 CSV 或结构零重叠也不自动证明科学独立性。仅有新结构而没有参考值的预测，不是这个外测评分流程。更多要求见[外测说明](example/external_test/README.md)。

### 2.4 怎样读结果

| 指标或文件 | 应怎样理解 |
| --- | --- |
| MAE，平均绝对误差 | 与目标同单位；0–1产率尺度的 `MAE=0.10` 是平均相差10个百分点 |
| RMSE，均方根误差 | 与目标同单位，对较大的偏差更敏感 |
| R²，决定系数 | 参照检验数据均值比较误差，可以为负；不是“预测正确率” |
| 训练分数 | 已学记录的拟合情况，不能替代留出或外测分数 |
| `verification.json` | 本例的运行与文件核验记录，不是模型性能保证 |

随机报告按每轮汇总 OOF 评分；固定名单报告汇总逐折指标。两者的 R² 计算口径不同，不跨路线直接排名。三个样例的低分与负 R² 均保留。

下载 ZIP 中的 `completed_run/` 是已完成输出的参考快照，默认 `result/` 不打入包，解压后可从头运行。源码 Git 不跟踪运行目录；仅克隆源码时，报告要在运行后生成，或从下载包中查看。随机包的 `report/` 是另存的历史报告，`evidence/` 中的 YAML 只作历史追溯。

## 3. YAML：把一次计算写清楚

[example.yaml](example.yaml) 是唯一完整格式规范；本页解释字段，不复制另一套 YAML 模板。三个教学 `run.yaml` 展示具体任务如何填写。

| 字段 | 含义与使用要点 |
| --- | --- |
| `schema_version` | 当前为 `"2.0"` |
| `project_name` | 独立任务名；对应自己的结果目录 |
| `stage` | 运行哪一个阶段，见下表 |
| `dataset.path`、`sample_id_col` | 数据表与稳定编号列；编号必须存在、非空且唯一 |
| `dataset.column_roles` | 指定预测目标 `label`、反应物 `reactants`、产物 `products`、其他结构 `others`、数值条件 `conditions`、类别列 `categoricals` |
| `dataset.numeric_conditions` | 缺失填补统计量和缩放参数从相应训练折拟合；单位换算由YAML显式指定，不自动猜测 |
| `descriptors` | 编码候选；`columns` 选输入列，`mode: concat` 按列合并。多个候选不会自动拼成一个编码 |
| `artifacts.output_dir` | 任务根下的 `feature/`；`train` 阶段另需 `artifacts.input_manifest` 指向可核验编码包 |
| `models`、`model_params` | 选择方法和模型设置；只声明被选择的模型参数 |
| `evaluation` | 随机轮换、固定/分组名单或外测的规则；不同协议的字段不能随意混用 |
| `hpo` | 自动比较和筛选模型设置（hyperparameter optimization），不是优化反应条件；三个样例均关闭 |
| `outputs.root`、`report_formats` | 独立结果根目录及 `[html, markdown]` 双报告 |
| `outputs.diagnostics` | 可选的训练与留出效果对照；关闭诊断不等于关闭正常评分 |
| `benchmark` | 专用协议的任务身份和 SQLite 恢复状态；普通随机流程不需要照抄此节 |
| `metadata` | 人工说明和备注 |

| `stage` | 行为 |
| --- | --- |
| `features` | 只生成或复用输入编码，不建立模型；可以没有参考标签 |
| `train` | 从 `artifacts.input_manifest` 读取编码后建模，不自动补算缺失编码 |
| `all` | 先准备编码，再对可用的编码与模型组合完成普通内测 |
| `benchmark` | 执行 `manifest_outer_cv` 或 `frozen_train_external_test` 的专用流程 |

普通 `all/train` 使用 KFold。真正固定成员或分组留出必须选 `stage: benchmark`；更改普通流程中的协议文字不会让它变成固定名单执行器。专用执行器也能采用随机划分（`repeated_kfold`），因此“strict”一词本身不表示未见底物。

### 路径、预算与自有研究任务

- 教学样例：`run.yaml` 留在自己的文件夹，输入与 `./result/<任务名>/` 均相对它解析。
- 正式研究：在仓库根 `config/` 新建独立 YAML。该目录内以 `./` 开头的引用按仓库根解析；显式 `../` 和其他位置的配置按 YAML 所在目录解析；绝对路径保持绝对。
- 所有任务的 `artifacts.output_dir` 应为 `outputs.root/feature`，不能与任务根相同。RF、XGBoost、LightGBM 的 `estimator.n_jobs` 为19；AutoGluon 的 `fit.num_cpus` 为19。任务、编码、组合和折不另开并行后台作业。

准备自己的随机内测配置，可从仓库根复制格式规范：

```bash
mkdir -p config
cp example.yaml config/my_reaction.yaml
```

在新文件中逐项修改：任务名、数据路径和列用途、编码与模型、模型预算、划分规则，以及 `./result/my_reaction` 和其中的 `feature/`。根格式样例使用12条人工记录、两折和10棵RF树，CPU预算已设为19；正式研究应为所选模型保留19CPU预算，并自行确定科学问题与计算预算。固定名单或外测另需参照对应教学样例补齐专用字段，不能只替换 `protocol` 字符串。修改后仍通过 `yonod.py` 首提示读取该 YAML。

解析器拒绝重复键、别名和未知字段。JSON 不是运行配置；`main.py` 是底层运行时，不是正式任务启动入口。`yonod.py` 也可从 CSV 交互收集列用途并生成设置；保存位置和任务隔离仍按 [AGENTS.md](AGENTS.md) 执行。

### 可用编码与模型

无需神经预训练权重的编码包括 `morgan`、`mfp`、`maccs`、`maf`、`rdkit2d`、`drfp` 和类别编码 `ohe`。`mfp` 是计数型 Morgan 编码，不是二值 `morgan` 的别名；`drfp` 有反应输入约束。`fisd`、`molmetalm`、`chemical_vae` 需要下一节所述资产。OHE、数值缺失值处理及缩放只从相应建模折学习，留出记录只做转换。

模型标识为 `rf`、`xgb`、`lightgbm`、`svm`、`autogluon`。RF/XGBoost/LightGBM 使用 `estimator`、`fit`、`runtime` 参数节；SVM 另有 `preprocessing`，AutoGluon 使用 `predictor`、`fit`、`runtime`。参数未声明时采用受支持安装版本的默认值，不能用另一版本的默认值猜测本次设置。

HPO 仅支持 RF/XGBoost/LightGBM 的已接入搜索流程，搜索只发生在相应训练数据内。当前冻结外测不支持 HPO、OHE、数值条件、AutoGluon或外测早停，不保存可复用最终模型，也不发布普通流程那种完整特征包；它按协议在内存中编码、建模和评分。外测配置不能加入 `n_splits`、`n_repeats`、`shuffle`、`grouping` 或 `split_manifest` 等内测划分字段。

## 4. 项目结构

```text
YONOD/
├── README.md / README.zh-CN.md  英文默认入口与中文说明
├── yonod.py                    正式交互启动入口
├── main.py                     普通配置运行时
├── example.yaml                完整schema-2格式规范与最小测试
├── requirements.txt            基础依赖；PyTorch等单独安装
├── LICENSE                     本项目许可证
├── example/                    三个自包含教学样例、导览与ZIP
│   ├── random_split/
│   ├── fixed_split/
│   └── external_test/
├── yonod/                      配置、编码、模型、流程、报告与专用协议代码
├── dataset/                    数据及来源说明
├── WEIGHTS/                    可选外部模型资产
├── config/                     本地科研任务YAML（Git忽略）
├── result/                     各科研任务的隔离结果（Git忽略）
├── logs/                       所有任务日志和PID（Git忽略）
├── project-docs/               独立文档Git仓库，不属于主仓
├── _verify/                    本地验证脚本（Git忽略）
├── backup/                     工程支持与历史材料（Git忽略）
└── reference-projects/         外部参考项目（Git忽略）
```

教学样例自己的 `result/` 与下载包中的 `completed_run/` 位于各包内。编码、模型、预测及报告的具体产物依协议而异；不要假设三个流程都保存相同的模型包。额外工程说明维护在独立的 [YONOD-docs](https://github.com/thinktraveller/YONOD-docs) 仓库，访问需要权限；主仓克隆不会自动获得它。

## 5. 外部数据与模型资产

### 数据与项目权重分发

先看[数据说明](dataset/README.md)和[权重说明](WEIGHTS/README.md)。三个教学样例的数据已随源码提供。项目分发入口是 [YONOD-datasets](https://huggingface.co/datasets/thinktraveller/YONOD-datasets) 和 [YONOD-weights](https://huggingface.co/thinktraveller/YONOD-weights)，它们不替代原始来源与许可记录。

以下命令在已激活 `yonod` 的仓库根执行；安装兼容 Transformers 4.40 的 Hugging Face CLI：

```bash
pip install "huggingface_hub>=0.34,<1.0"
hf download thinktraveller/YONOD-datasets --repo-type dataset --include "USPTO/**" --local-dir dataset
hf download thinktraveller/YONOD-weights --local-dir WEIGHTS
```

第一条下载命令只获取 USPTO 子目录，不是全部数据。按服务要求完成必要的登录或访问授权。二进制权重被 Git 忽略；已有配置、tokenizer或manifest文件不代表所有模型权重已具备。

### 可选的预训练编码

| 编码 | 所需资产与放置位置 |
| --- | --- |
| FISD | [上游权重](https://github.com/KeantChen/FISD/tree/main/model)放入 `WEIGHTS/FISD/`：`qm_9_mse_model.pth`、`qm_9_cos_model.pth`、`qm_9_2in1_model.pth` |
| MolMetaLM | [MolMetaLM-base](https://huggingface.co/wudejian789/MolMetaLM-base)的模型、配置与tokenizer放入 `WEIGHTS/MolMetaLM-base/`；[上游项目](https://github.com/CSUBioGroup/MolMetaLM) |
| Chemical VAE | 使用 `WEIGHTS/chemical_vae/<资产>/v5/` 的已验证转换manifest及相邻、哈希匹配的 `encoder_state_dict.pt`；ZINC与`zinc_properties`资产不能互换 |

单独恢复 MolMetaLM：

```bash
hf download wudejian789/MolMetaLM-base --local-dir WEIGHTS/MolMetaLM-base
```

Chemical VAE只使用冻结的PyTorch编码器原始 `z_mean_sample`，不在YONOD中训练VAE，不加载解码器或性质头。原始HDF5或仅有manifest不能直接替代转换后权重；普通运行不要求TensorFlow/Keras。具体约束见[加载器](yonod/descriptors/chemical_vae.py)。

只下载并配置实际要使用的资产。外部数据、权重和参考项目各自保留上游许可、引用和适用范围，不能由本项目许可证统一覆盖。

## 6. License

本项目代码和文档采用 [CC BY-NC 4.0](LICENSE)，要求署名并限于非商业使用。第三方数据、预训练权重及参考代码遵循其各自许可证。
