# YONOD

[English](README.md) · [简体中文](README.zh-CN.md)

YONOD（You Only Need Outstanding Descriptors）用已有化学反应记录建立数值预测模型，例如预测反应产率。你准备一张数据表，选好输入列和建模方法，程序会把分子结构转换成数值编码，完成建模与留出检验，并保存逐条预测和可在浏览器打开的报告。

如果你还不熟悉 Python，可以先跟着下面的步骤运行自带样例。你只需输入命令和配置文件名，无需编写程序或下载预训练模型权重。

程序的交互提示目前使用中文。

[安装](#2-安装) · [第一次运行](#3-完成第一次运行) · [读报告](#4-怎样读结果) · [自己的数据](#5-使用自己的-csv) · [更多样例](#6-尝试其他检验方式)

## 1. 它能帮你做什么

整理好一批反应物结构和实验产率后，YONOD 可以将结构转换为描述符（descriptor，即供模型使用的数值编码），用部分记录建立回归模型，再预测本次建模留出的记录。你也可以在同一批数据和划分上比较不同编码、随机森林等建模方法。

| 你提供什么 | 程序做什么 | 你得到什么 |
| --- | --- | --- |
| CSV 表格，每行一条记录 | 按配置读取编号、结构、参考值及可选条件 | 可核对的输入与运行记录 |
| 分子结构的 SMILES 文本 | 生成或复用所选数值编码 | 特征文件，或按训练折学习的编码 |
| 产率等数值参考值、检验规则 | 建立模型，对留出的记录预测并评分 | 逐条预测、误差指标、HTML 和 Markdown 报告 |

SMILES 是用字符表示分子结构的方法，例如 `CCO` 表示乙醇。结构列填写可解析的 SMILES，分子图片、化学名称或 CAS 号不能直接替代它。CSV 是可以用 Excel 打开的文本表格；保存时请选择“CSV UTF-8（逗号分隔）”。第一次练习可以直接用仓库已有文件。

YONOD 的主要流程需要已有数值参考值来建模和检验。它不会自动生成产物结构、推导反应机理或寻找最佳实验条件。若把产物结构作为输入，实际预测时就需要事先知道目标产物结构。没有参考值的新反应预测也不能直接套用本页的外测评分流程。

| 你的问题 | 从哪里开始 |
| --- | --- |
| 想先看看程序能不能在本机运行 | 跟随下面的随机划分样例，用随源码提供的数据。 |
| 想比较自己的结构编码或模型 | 先整理已有参考值的CSV，再生成独立配置，核对输入与划分。 |
| 已经决定哪些记录用于检验 | 使用固定名单样例，按记录编号准备成员清单。 |
| 有另一批带参考值、准备单独检验的数据 | 阅读外测样例，核实来源、结构重叠和列映射，提前固定模型设置。 |

## 2. 安装

### 准备终端和项目文件

先安装 [Miniconda](https://www.anaconda.com/docs/getting-started/miniconda/install) 和 [Git](https://git-scm.com/downloads)。Conda 用来为本项目建立独立的 Python 环境，Git 用来下载源码。这里使用名为 `yonod` 的环境，Python 版本为 3.9。

Linux 打开 Bash 终端。Windows 打开 **Anaconda PowerShell Prompt**；如果使用普通 PowerShell，需要先从 Anaconda Prompt 执行 `conda init powershell`，关闭后重新打开 PowerShell。下面命令在终端中逐行输入，每行按回车；不要在 Python 的 `>>>` 提示符后输入，也不要复制网页中显示的运行输出。若已经进入 `>>>`，输入 `exit()` 回到终端。

“项目根目录”指包含 `yonod.py`、`requirements.txt` 和 `example.yaml` 的 YONOD 文件夹。下载后用 `cd YONOD` 或 `Set-Location YONOD` 进入它。路径里有空格时，用引号包住路径。

### 首次安装依赖

以下步骤只需安装时完成一次。Linux 的 Conda 初始化命令假定 Miniconda 安装在 `$HOME/miniconda3`，若位置不同，请替换该前缀。已下载源码时，可跳过 `git clone`，直接进入已有的项目根目录。

Linux Bash：

```bash
source "$HOME/miniconda3/etc/profile.d/conda.sh"
git clone https://github.com/thinktraveller/YONOD.git
cd YONOD
conda create -n yonod python=3.9 -y
conda activate yonod
python -m pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

Windows Conda PowerShell：

```powershell
git clone https://github.com/thinktraveller/YONOD.git
Set-Location YONOD
conda create -n yonod python=3.9 -y
conda activate yonod
python -m pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

这里先安装 CPU 版 PyTorch，再安装 [requirements.txt](requirements.txt) 中的依赖，包括 LightGBM 和 SciPy。三个教学样例都可使用 CPU；首次使用无需配置 CUDA。需要 GPU 时，请结合本机驱动查看 [PyTorch 对应版本安装说明](https://pytorch.org/get-started/previous-versions/) 和依赖文件中的 CUDA 12.1 命令。

如果已经有可用的 `yonod` 环境，不必重复创建。Windows 的 Conda 初始化通常只需完成一次；Linux 的新 Bash 窗口按上面的初始化命令读入 Conda。每次打开新终端都需执行 `conda activate yonod`，然后进入项目目录。环境激活后，终端通常会显示 `(yonod)`。

### 检查环境

激活 `yonod` 后，以下命令在两种系统相同：

```text
python --version
python -c "import torch, rdkit, sklearn, xgboost, lightgbm; print('YONOD imports OK')"
```

应看到 Python 3.9 和 `YONOD imports OK`，说明主要依赖可以导入。保留当前终端，继续第一次运行。

## 3. 完成第一次运行

### 看一眼数据和设置

先打开[随机划分样例的数据表](example/random_split/dataset/reactions.csv)。它有12条人工记录：`reaction_id` 是唯一编号，两个 `*_smiles` 列是结构，`yield` 是参考产率。例如 `0.10` 表示10%。这些结构搭配和产率仅用于教学，不是实验测量，也不保证代表合理的化学反应。

同目录的 [run.yaml](example/random_split/run.yaml) 保存本次设置。YAML 是用缩进组织设置的文本文件，这里已经准备好，第一次运行无需修改。程序会使用 Morgan 编码，依次运行随机森林（RF）、XGBoost 和 LightGBM；将12条记录分成两份，每次6条建模、6条检验，再交换。首次完整运行共进行6次拟合。

### 启动并输入配置路径

从项目根目录开始，按系统选择下面一组命令。程序在当前终端显示进度；等待它结束后再启动下一个任务。

Linux Bash：

```bash
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate yonod
export PYTHONUTF8=1
cd example/random_split
python -u ../../yonod.py
```

Windows Conda PowerShell：

```powershell
conda activate yonod
$env:PYTHONUTF8 = "1"
Set-Location .\example\random_split
python -u ..\..\yonod.py
```

当程序询问 `请输入配置文件路径(schema-2 YAML):` 时，输入下面的文件名并按回车：

```text
run.yaml
```

程序随后校验配置、准备编码并运行模型。完成时应看到 `features=ready`，以及 `morgan × rf`、`morgan × xgb`、`morgan × lightgbm` 各自的 `complete` 或 `reused`。`reused` 表示程序核对身份后复用了已有结果。运行路径中的长编号由内容决定，不必与别人屏幕上的编号相同。

### 打开报告

仍在 `example/random_split/` 目录时，新报告位于 `result/example_random_split_20261113/report/report.html`。在文件管理器找到它，双击用浏览器打开即可。也可使用下列命令。

Linux Bash（有桌面环境时）：

```bash
xdg-open result/example_random_split_20261113/report/report.html
```

Windows Conda PowerShell：

```powershell
Invoke-Item .\result\example_random_split_20261113\report\report.html
```

同目录的 `report.md` 是文字版报告；逐条预测位于任务目录的 `runs/run_morgan-<模型>/predictions.csv`。这里的 `<模型>` 替换为 `rf`、`xgb` 或 `lightgbm`，不是文件名的一部分。

后续步骤从项目根目录开始。运行结束后，Linux 用 `cd ../..`，Windows PowerShell 用 `Set-Location ..\..` 返回。需要关闭终端后继续计算时，再参考[后台启动与状态查看说明](example/README.md)；所有操作日志、标准输入文件和 PID 文件统一放在仓库根 `logs/`。

教学包保存的完整运行验收来自 Linux。Windows 命令说明操作方法，尚不能据此认定各后端通过了 Windows 实机验收；请在本机完成安装与运行检查。

## 4. 怎样读结果

先看留出检验的指标和逐条预测。每条记录由没有学习该记录的相应模型预测，这些结果汇总称为 OOF（out-of-fold）预测。训练分数反映模型对已学记录的拟合情况，不能替代留出分数。

| 指标 | 含义 |
| --- | --- |
| MAE，平均绝对误差 | 预测值与参考值平均相差多少，与参考值同单位。产率按0到1记录时，`MAE=0.10` 表示平均相差10个百分点。 |
| RMSE，均方根误差 | 也与参考值同单位，对较大的偏差更敏感。 |
| R²，决定系数 | 将预测误差与检验数据均值基线比较，可以为负；它不是“预测正确率”。 |

小样例的低分和负 R² 都保留，目的是教你查看结果，不能用来判断某种方法对真实化学问题的预测能力。随机划分也不保证检验中的底物从未在建模记录中出现；要研究新底物，需要另外设计分组和检查结构重叠。

## 5. 使用自己的 CSV

### 先准备数据表

每行放一条反应记录，保留表头，并为记录准备稳定、非空且唯一的编号，例如 `record-001`。不要把编号作为模型输入。避免单独使用 `001`、`NA` 等可能在读取时改变含义的编号；需要改编号时保留原编号对应表。Excel中使用“另存为”选择CSV UTF-8，直接把 `.xlsx` 后缀改为 `.csv` 不会转换文件格式。

参考产率必须是数值，并统一尺度，例如始终用 `0.80` 表示80%，或始终用 `80` 表示80%。程序不自动换算产率尺度；不要在单元格里写 `80%`。列角色也要按化学含义核对，产率、转化率、LC面积比和 ee 不能混为同一目标。

当前配置生成向导必须选择一列数值参考值和一列产物 SMILES；反应物、其他组分和数值条件可按提示选择。若你的数据没有产物列，请使用本节末尾的手工配置方法。普通建模运行器支持 `products: []`，可以只用反应物；不要为通过向导虚构产物结构。

### 生成配置

从项目根目录运行生成器。它只准备数据并保存配置，完成后不会自动开始建模。

Linux Bash：

```bash
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate yonod
export PYTHONUTF8=1
python -u generate_yaml_config.py
```

Windows Conda PowerShell：

```powershell
conda activate yonod
$env:PYTHONUTF8 = "1"
python -u .\generate_yaml_config.py
```

按屏幕提示依次填写：

1. 输入 CSV 路径，给任务取一个新名字，例如 `my_reaction`。名称只用英文字母、数字、下划线或短横杠；已有同名配置或非空结果目录时需换名。
2. 选择稳定编号列，再依次选择参考值、反应物、产物、其他组分和数值条件列。按当前屏幕列号作答，各步骤的编号可能不同；没有其他组分或数值条件时可按提示回车跳过。
3. 核对无效行提示。生成器会另存规范化 CSV，并从中排除无效标签或结构等记录；运行前检查排除原因和剩余行数。
4. 第一次练习在描述符菜单输入 `1` 选择 Morgan；询问是否编辑编码配置时可回车使用默认设置。默认包含选中的反应物、其他组分和产物，需确认这些信息在实际预测时可获得。
5. 在模型菜单输入 `2` 选择 Random Forest。描述符或模型菜单直接回车会选择多种方法，其中有的需要额外权重或依赖，首次不要在这两处留空。
6. 按需填写网址、DOI 和备注；报告格式回车选择 HTML 与 Markdown。最后可选的“修复后数据集”是额外副本，回车不生成该副本也不会取消已经完成的规范化处理。

生成器将 YAML 保存为 `config/my_reaction.yaml`。规范化表在 `result/my_reaction/feature/docs/my_reaction_normalized_dataset.csv`；有无效行时，排除报告在 `result/my_reaction/report/my_reaction_invalid_report.md`。原始 CSV 保留。

### 检查后启动

用文本编辑器打开生成的 YAML，核对数据路径、列名、输入信息和结果目录。当前生成器默认是随机5折、1轮、种子42，HPO关闭，不会询问训练参数预算；有效编码记录至少要有5条才能执行5折，这也不足以保证有意义的科学评价。模型参数未声明时采用已安装库的默认值。

数值条件默认缺失即报错，不自动填零或换算单位。存在缺失条件时，先核对数据，或参考 [example.yaml](example.yaml) 的 `dataset.numeric_conditions` 显式设置缺失标记、填补和缩放规则。均值、中位数和缩放参数从相应训练折学习。

确认后，在同一个已激活 `yonod` 的终端、项目根目录启动：Linux用 `python -u yonod.py`，Windows用 `python -u .\yonod.py`。首次提示输入：

```text
config/my_reaction.yaml
```

新结果都保存在 `result/my_reaction/`，HTML 报告为 `report/report.html`。一次只运行一个任务；任务、编码、模型组合和交叉验证折依次执行。

### 手工准备 YAML

[example.yaml](example.yaml) 是完整 schema-2 配置的唯一格式规范。无产物列或需要自行设置时，从它另存新文件再编辑，不要覆盖原样例。以下文件操作从项目根目录执行。

Linux Bash：

```bash
mkdir -p config
cp example.yaml config/my_reaction.yaml
```

Windows Conda PowerShell：

```powershell
New-Item -ItemType Directory -Path config -Force | Out-Null
Copy-Item example.yaml config\my_reaction.yaml
```

选择尚未使用的任务名，并同步修改 `project_name`、数据路径和角色、编码与模型、检验设置、`outputs.root: ./result/my_reaction`、`artifacts.output_dir: ./result/my_reaction/feature`，保留 `outputs.report_formats: [html, markdown]`。只声明已选模型的参数。保存后仍通过 `yonod.py` 首次提示输入配置路径。

原始格式样例使用12条人工记录、两折和10棵RF树，只用于轻量工程检查。研究任务需要按自己的问题确定数据量、模型设置和检验设计。

## 6. 尝试其他检验方式

[样例总说明](example/README.md) · [浏览器导览](example/index.html) · [下载全部样例](example/archives/all_examples.zip)

| 样例 | 它检验什么 | 主要文件 |
| --- | --- | --- |
| [随机划分](example/random_split/README.md) | 同一批记录随机分组，两折轮换检验 | [数据](example/random_split/dataset/reactions.csv)、[配置](example/random_split/run.yaml) |
| [固定名单](example/fixed_split/README.md) | 采用预先保存的具体成员名单；12条分两组，每组6条，RF共拟合两次 | [名单解释](example/fixed_split/splits/membership_guide.csv)、[实际名单](example/fixed_split/splits/fixed_split_manifest.csv)、[配置](example/fixed_split/run.yaml) |
| [外测流程](example/external_test/README.md) | 提前固定模型设置，用12条训练记录拟合一次，预测另表6条记录 | [训练表](example/external_test/dataset/train.csv)、[外测表](example/external_test/dataset/external.csv)、[配置](example/external_test/run.yaml) |

运行另一个包时，先回项目根目录，进入 `example/fixed_split/` 或 `example/external_test/`，再按首次运行的方法启动 `../../yonod.py`（Windows 为 `..\..\yonod.py`），输入 `run.yaml`。配置留在各自包内，无需复制。报告分别是各任务目录的 `report/benchmark_report.html` 和 `report/external_test_report.html`，也都有 Markdown 版。

固定名单使用 `stage: benchmark` 和 `manifest_outer_cv`，不能用固定随机种子代替成员清单。更改数据后需要同步核对编号、源行号、组隔离及 CSV 的 SHA-256 校验码。当前要求至少两折，每条记录每轮恰好留出一次，不能直接改成仅一次“8条训练、4条检验”。教学组名没有化学类别含义。

外测使用 `frozen_train_external_test`。训练列与外测列可通过 `external_test.column_map` 对应，参考值由各自标签字段指定。当前适配器要求审计声明的胺、酸、产物精确规范化结构交集均为0，外测至少有两个非空组；真实任务需重新核实来源、映射、校验码和结构交集。另存 CSV 或结构零重叠不会自动证明科学独立性。这个样例也使用产物结构，实际预测时需已知目标产物。

三个包的数据都是人工教学值，外测包没有独立实验来源。随机报告按每轮汇总 OOF 评分，固定名单报告汇总逐折指标，R² 口径不同，不能跨路线直接排名。[随机](example/random_split/verification.json)、[固定](example/fixed_split/verification.json)、[外测](example/external_test/verification.json)的验收记录说明流程和文件核对结果，不是模型性能保证。

下载 ZIP 的 `completed_run/` 包含完成输出快照；重新运行写入各包 `result/`。源码 Git 不跟踪这些运行目录，克隆后需自己运行或从下载包查看快照。随机包另有历史 `report/` 与 `evidence/`，其中历史 YAML 用于追溯。

## 7. 进一步理解配置

配置用文本编辑器保存为 UTF-8 纯文本，缩进使用空格，保留样例的层级结构；确认实际扩展名是 YAML，而不是文本编辑器自动附加的TXT。可执行配置采用 schema-2 YAML；JSON 用于审计和运行记录，不能代替运行 YAML。解析器拒绝重复键、别名和未知字段。将已有 YAML 提供给 `generate_yaml_config.py` 只会校验；提供给 `yonod.py` 才会执行。

| 设置 | 用途 |
| --- | --- |
| `stage: features` | 只准备或复用编码，不拟合模型，可不提供标签。 |
| `stage: train` | 用 `artifacts.input_manifest` 指定的可核验编码包建模，不补算缺失编码。 |
| `stage: all` | 准备编码后，执行普通内测。 |
| `stage: benchmark` | 分派固定成员或冻结外测等专用协议。 |
| `descriptors` | 每项是一个编码候选；多个候选不会自动合并。`columns` 选择输入，`mode: concat` 按列拼接。 |
| `evaluation` | 划分、轮数和随机种子，或专用协议要求的成员与外测设置。 |
| `hpo` | 模型超参数搜索，只支持已接入的 RF/XGBoost/LightGBM；不是优化反应条件。三个样例均关闭。 |
| `outputs.report_formats` | 默认保留 `[html, markdown]`，`outputs.diagnostics` 可另开训练与留出效果对照。 |

根 `config/` 内 YAML 中以 `./` 开头的路径按项目根目录解析；显式 `../` 和其他位置 YAML 的相对路径按 YAML 所在文件夹解析，绝对路径保持绝对。每个研究任务使用独立的 `result/<task_name>/`，编码目录必须位于其中，例如 `feature/`；不要将最终结果放入 `derived/`。

无需预训练权重的编码有 `morgan`、`mfp`、`maccs`、`maf`、`rdkit2d`、`drfp`、`ohe`。`mfp` 是计数型 Morgan，`morgan` 是二值编码；`drfp` 使用反应角色输入。OHE 是类别独热编码，从相应训练折学习类别表，留出数据只做转换。

模型标识为 `rf`、`xgb`、`lightgbm`、`svm`、`autogluon`。RF/XGBoost/LightGBM 的参数节为 `estimator`、`fit`、`runtime`；SVM 另有 `preprocessing`，AutoGluon 使用 `predictor`、`fit`、`runtime`。需要限制线程时，根据本机设置 `model_params.<模型>.estimator.n_jobs`；AutoGluon 用 `model_params.autogluon.fit.num_cpus`。不指定时采用库默认值，无需固定CPU数量或绑定CPU。

普通 `all/train` 使用 KFold；指定固定成员或分组留出须采用相应 `benchmark` 协议。专用执行器也支持 `repeated_kfold`，因此“strict”一词本身不表示未见底物。当前冻结外测不支持 HPO、OHE、数值条件、AutoGluon或外测早停，不保存可复用最终模型或普通完整特征包，也不能混入 `n_splits`、`n_repeats`、`shuffle`、`grouping`、`split_manifest` 等内测字段。

## 8. 项目文件放在哪里

| 位置 | 内容 |
| --- | --- |
| `generate_yaml_config.py` / `yonod.py` | 配置生成器 / 建模入口 |
| [example.yaml](example.yaml) / [example/](example/) | 完整格式规范 / 三个教学包 |
| `config/` / `result/<task_name>/` | 本地研究配置 / 对应结果，均不由Git跟踪 |
| `logs/` | 任务操作日志、标准输入文件和PID |
| [dataset/](dataset/README.md) / [WEIGHTS/](WEIGHTS/README.md) | 数据来源说明 / 可选外部权重 |
| `yonod/` | 配置、编码、模型、检验和报告源码 |
| `project-docs/` | 独立文档仓库，主仓克隆不包含它 |
| `_verify/` / `backup/` / `reference-projects/` | 本地验证脚本 / 备份及工程支持 / 外部参考材料，均不提交 |

教学包的 `result/` 位于各自包内。更多维护文档在独立的 [YONOD-docs](https://github.com/thinktraveller/YONOD-docs)，访问需要权限。参与修改前请阅读 [AGENTS.md](AGENTS.md)，测试脚本放 `_verify/`，项目文档放独立文档仓库。

## 9. 按需安装可选依赖和资产

只有选择 AutoGluon 时，才需在已激活 `yonod` 的终端额外安装。下列命令适用于 Linux Bash 和 Windows Conda PowerShell：

```text
python -m pip install "autogluon.tabular[lightgbm,catboost]==1.1.1"
```

三个教学包已含数据且使用 Morgan。其他数据与权重可从 [YONOD-datasets](https://huggingface.co/datasets/thinktraveller/YONOD-datasets) 和 [YONOD-weights](https://huggingface.co/thinktraveller/YONOD-weights) 获取，下载前查看[数据说明](dataset/README.md)与[权重说明](WEIGHTS/README.md)。

以下命令在已激活环境的项目根目录执行，两种系统相同。先安装兼容 Transformers 4.40 的 Hugging Face CLI，再按需下载；数据命令只下载 USPTO 子目录。

```text
python -m pip install "huggingface_hub>=0.34,<1.0"
hf download thinktraveller/YONOD-datasets --repo-type dataset --include "USPTO/**" --local-dir dataset
hf download thinktraveller/YONOD-weights --local-dir WEIGHTS
```

按服务要求完成登录或访问授权。Git 忽略二进制权重；看见配置、tokenizer或manifest文件不代表权重已齐全。部分原始数据虽然扩展名是 CSV，实际使用制表符分隔，需先按子目录说明转换，不能直接送入逗号CSV向导。

| 编码 | 所需资产 |
| --- | --- |
| `fisd` | 从[上游](https://github.com/KeantChen/FISD/tree/main/model)获取 `qm_9_mse_model.pth`、`qm_9_cos_model.pth`、`qm_9_2in1_model.pth`，放入 `WEIGHTS/FISD/`。 |
| `molmetalm` | [MolMetaLM-base](https://huggingface.co/wudejian789/MolMetaLM-base) 的模型、配置与tokenizer，放入 `WEIGHTS/MolMetaLM-base/`；参见[上游项目](https://github.com/CSUBioGroup/MolMetaLM)。 |
| `chemical_vae` | `WEIGHTS/chemical_vae/<资产>/v5/` 下已验证转换manifest及相邻、哈希匹配的 `encoder_state_dict.pt`。ZINC 与 `zinc_properties` 资产不能互换。 |

Chemical VAE只使用冻结 PyTorch 编码器的原始 `z_mean_sample`，不训练 VAE，也不加载解码器或性质头。原始 HDF5 或只有manifest不能替代转换后权重；普通运行无需 TensorFlow/Keras，详见[加载器](yonod/descriptors/chemical_vae.py)。

## 10. 遇到问题时

| 现象 | 先检查什么 |
| --- | --- |
| `conda` 无法识别 | Linux先用本机 `conda.sh` 初始化；Windows使用 Anaconda PowerShell Prompt或初始化后重开终端。 |
| 输入命令后出现 `SyntaxError` | 若提示符是 `>>>`，先输入 `exit()` 退出Python，再在终端运行命令。 |
| 找不到 `yonod.py` 或 `requirements.txt` | 当前目录是否正确；根目录应能看见这两个文件，教学包启动则使用上面给出的两级上层路径。 |
| 找不到YAML，或文件名看着正确 | 在文件管理器显示扩展名，确认文件不是 `run.yaml.txt`；核对配置路径相对当前目录的位置。 |
| 路径中有空格 | 终端命令里用引号包住完整路径；进入程序的路径提示后填写实际路径，不附带启动命令。 |
| `ModuleNotFoundError` | 是否激活 `yonod`，是否先装 PyTorch再完成依赖安装；安装错误需要先解决，不能跳过后续检查。 |
| 编号或列名报错 | 核对CSV表头、唯一编号和YAML引用；生成器会重命名部分列，需检查规范化表。 |
| 数值条件或SMILES报错 | 查看具体记录编号与排除报告，核实结构和缺失规则。 |
| 没有生成报告 | 查看终端首次报错及最终完成状态；`result/` 中仅出现目录不代表任务完成。 |
| 同名任务或已存在外测结果被拒绝 | 另取任务名和配置名，同步设置独立结果路径；冻结外测会保护已有指标，不能按普通任务复用方式理解。 |

仍需帮助时，可在[项目 Issues](https://github.com/thinktraveller/YONOD/issues)中提供操作系统、Python版本、所用命令、配置及完整报错。提交前移除私人数据和凭据。开发者的轻量工程检查是激活环境后从根目录运行 `python yonod.py --config example.yaml`；它实际执行小型建模，不等同于仅校验配置。

## 11. License

本项目代码和文档采用 [CC BY-NC 4.0](LICENSE)，要求署名并限于非商业使用。第三方数据、预训练权重与参考代码遵循各自许可证和引用要求。

最近一次对照本仓库核对：2026-11-14。
