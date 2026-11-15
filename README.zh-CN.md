# YONOD

[English](README.md) · [简体中文](README.zh-CN.md)

[项目结构](#项目结构) · [致谢](#致谢) · [License](#license)

YONOD（You Only Need Outstanding Descriptors）用已有化学反应记录建立数值预测模型，例如预测反应产率。你准备一张数据表，选好输入列和建模方法，程序会把分子结构转换成数值编码，完成建模与留出检验，并保存逐条预测和可在浏览器打开的报告。

如果你还不熟悉 Python，可以先跟着下面的步骤运行自带样例。你只需输入命令和配置文件名，无需编写程序或下载预训练模型权重。

程序的交互提示目前使用中文。

[入门内容](#入门内容)带你安装并完成冒烟测试；[进阶内容](#进阶内容)说明三种检验方式与配置；[开发者内容](#开发者内容)提供完整参数参考及扩展方法。

## 入门内容

- [它能帮你做什么](#它能帮你做什么)
- [安装](#安装)
- [完成冒烟测试](#完成冒烟测试)
- [怎样读结果](#怎样读结果)
- [遇到问题时](#遇到问题时)

### 它能帮你做什么

- 整理好一批反应物结构和实验产率后，YONOD 可以将结构转换为描述符（descriptor，即供模型使用的数值编码），用部分记录建立回归模型，再预测本次建模留出的记录。
- 你也可以在同一批数据和划分上比较不同编码、随机森林等建模方法。

| 你提供什么 | 程序做什么 | 你得到什么 |
| --- | --- | --- |
| CSV 表格，每行一条记录 | 按配置读取编号、结构、参考值及可选条件 | 可核对的输入与运行记录 |
| 分子结构的 SMILES 文本 | 生成或复用所选数值编码 | 特征文件，或按训练折学习的编码 |
| 产率等数值参考值、检验规则 | 建立模型，对留出的记录预测并评分 | 逐条预测、误差指标、HTML 和 Markdown 报告 |

- SMILES 是用字符表示分子结构的方法，例如 `CCO` 表示乙醇。
- 结构列填写可解析的 SMILES，分子图片、化学名称或 CAS 号不能直接替代它。
- CSV 是可以用 Excel 打开的文本表格；保存时请选择“CSV UTF-8（逗号分隔）”。
- 首次使用可以直接用仓库已有文件，先完成下文的冒烟测试，检查本机能否运行主要流程。

- YONOD 的主要流程需要已有数值参考值来建模和检验。
- 它不会自动生成产物结构、推导反应机理或寻找最佳实验条件。
- 若把产物结构作为输入，实际预测时就需要事先知道目标产物结构。
- 没有参考值的新反应预测也不能直接套用本页的外测评分流程。

| 你的问题 | 从哪里开始 |
| --- | --- |
| 想先看看程序能不能在本机运行 | 跟随下面的随机划分样例，用随源码提供的数据。 |
| 想比较自己的结构编码或模型 | 先整理已有参考值的CSV，再生成独立配置，核对输入与划分。 |
| 已经决定哪些记录用于检验 | 使用固定名单样例，按记录编号准备成员清单。 |
| 有另一批带参考值、准备单独检验的数据 | 阅读外测样例，核实来源、结构重叠和列映射，提前固定模型设置。 |

### 安装

#### 准备终端和项目文件

- 先安装 [Miniconda](https://www.anaconda.com/docs/getting-started/miniconda/install) 和 [Git](https://git-scm.com/downloads)。
- Conda 用来为本项目建立独立的 Python 环境，Git 用来下载源码。
- 这里使用名为 `yonod` 的环境，Python 版本为 3.9。

- Linux 打开 Bash 终端。
- Windows 打开 **Anaconda PowerShell Prompt**；如果使用普通 PowerShell，需要先从 Anaconda Prompt 执行 `conda init powershell`，关闭后重新打开 PowerShell。
- 下面命令在终端中逐行输入，每行按回车；不要在 Python 的 `>>>` 提示符后输入，也不要复制网页中显示的运行输出。
- 若已经进入 `>>>`，输入 `exit()` 回到终端。

- “项目根目录”指包含 `yonod.py`、`requirements.txt` 和 `example.yaml` 的 YONOD 文件夹。
- 下载后用 `cd YONOD` 或 `Set-Location YONOD` 进入它。
- 路径里有空格时，用引号包住路径。

#### 首次安装依赖

- 以下步骤只需安装时完成一次。
- Linux 的 Conda 初始化命令假定 Miniconda 安装在 `$HOME/miniconda3`，若位置不同，请替换该前缀。
- 已下载源码时，可跳过 `git clone`，直接进入已有的项目根目录。

Linux Bash：

```bash
source "$HOME/miniconda3/etc/profile.d/conda.sh"
git clone https://github.com/thinktraveller/YONOD.git
cd YONOD
conda create -n yonod python=3.9 -y
conda activate yonod
python -m pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

Windows Conda PowerShell：

```powershell
git clone https://github.com/thinktraveller/YONOD.git
Set-Location YONOD
conda create -n yonod python=3.9 -y
conda activate yonod
python -m pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

- 这里先安装 CPU 版 PyTorch，再安装 [requirements.txt](requirements.txt) 中的依赖，包括 LightGBM 和 SciPy。
- 核心固定版本为 scikit-learn 1.5.2、LightGBM 4.6.0 和 tqdm 4.67.1。Linux x86_64 二进制轮子要求 glibc ≥2.28（包括 Ubuntu 22.04），Windows 面向 amd64；其他架构、musl 和更旧 glibc 需单独验收。
- 迁移依赖时，在独立 Conda 安装、虚拟机或 CI runner 中新建 `yonod`，保留已有冻结研究环境和结果。升级可能改变预处理、模型和预测；不要复用旧缓存，也不要假定模型支持跨版本加载。
- 三个首次使用样例都可使用 CPU；首次使用无需配置 CUDA。
- CPU 安装使用配套且支持 Python 3.9 的 Torch 2.8.0 / torchvision 0.23.0 / torchaudio 2.8.0；原 Torch 2.1.2 冻结环境单独保留。GPU 需另行匹配驱动并验收，参见 [PyTorch 对应版本安装说明](https://pytorch.org/get-started/previous-versions/)。
- Torch 2.8 仍有已公开漏洞，支持 Python 3.9 不代表整个环境安全。经审计的 Transformers `4.57.6+yonod.sec2` 对 pickle/RNG 加载要求 Torch ≥2.10，因此其 Python 3.9 路线使用获准的本地 safetensors。FISD 与 Chemical VAE 仅接收独立固定摘要批准的检查点字节；这限制了输入边界，没有修复原生 Torch 漏洞。

- 如果已经有可用的 `yonod` 环境，不必重复创建。
- Windows 的 Conda 初始化通常只需完成一次；Linux 的新 Bash 窗口按上面的初始化命令读入 Conda。
- 每次打开新终端都需执行 `conda activate yonod`，然后进入项目目录。
- 环境激活后，终端通常会显示 `(yonod)`。

#### 检查环境

激活 `yonod` 后，以下命令在两种系统相同：

```text
python --version
python -c "import torch, rdkit, sklearn, xgboost, lightgbm; print('YONOD imports OK')"
```

应看到 Python 3.9 和 `YONOD imports OK`，说明主要依赖可以导入。保留当前终端，继续冒烟测试。

### 完成冒烟测试

#### 什么是冒烟测试

- 冒烟测试（smoke test）是先用少量数据完整运行一遍主要流程，尽早发现安装、配置或程序衔接中的明显问题。
- 本项目的首次使用样例主要用于冒烟测试：检查所需依赖能否正常使用、配置和数据能否读取、结构编码能否生成、模型能否训练并保存预测，以及 HTML 和 Markdown 报告能否生成。运行结束后，还需按下文打开报告，确认可以查看。
- 三个样例都使用小规模人工数据。首次使用先运行下面的随机划分样例；另外两种流程的冒烟测试方法见进阶内容。
- 测试通过说明当前环境下、本次样例涉及的流程能够运行。它不能证明模型对真实化学数据的预测能力，也没有覆盖所有描述符、模型或操作系统。

#### 看一眼数据和设置

- 先打开[随机划分样例的数据表](example/random_split_test/dataset/reactions.csv)。
- 它有12条人工记录：`reaction_id` 是唯一编号，两个 `*_smiles` 列是结构，`yield` 是参考产率。
- 例如 `0.10` 表示10%。
- 这些结构搭配和产率是供冒烟测试使用的人工数据，不是实验测量，也不保证代表合理的化学反应。

- 本次设置已保存在配置目录中的 [run.yaml](config/example/random_split_test/run.yaml)。
- YAML 是用缩进组织设置的文本文件，这里已经准备好，运行冒烟测试无需修改。
- 程序会使用 Morgan 编码，依次运行随机森林（RF）、XGBoost 和 LightGBM；将12条记录分成两份，每次6条建模、6条检验，再交换。
- 首次完整运行共进行6次拟合。

#### 启动并输入配置路径

从项目根目录开始，按系统选择下面一组命令。程序在当前终端显示进度；等待它结束后再启动下一个任务。

Linux Bash：

```bash
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate yonod
export PYTHONUTF8=1
python -u yonod.py
```

Windows Conda PowerShell：

```powershell
conda activate yonod
$env:PYTHONUTF8 = "1"
python -u .\yonod.py
```

当程序询问 `请输入配置文件路径(schema-2 YAML):` 时，输入下面的文件名并按回车：

```text
config/example/random_split_test/run.yaml
```

- 该配置已保存在对应目录；随机划分测试初始只有输入 CSV 和配置，不附带预生成结果，无需复制 YAML 或运行 YAML 生成器。
- 程序随后校验配置、准备编码并运行模型。
- 完成时应看到 `features=ready`，以及 `morgan × rf`、`morgan × xgb`、`morgan × lightgbm` 各自的 `complete` 或 `reused`。
- `reused` 表示程序核对身份后复用了已有结果。
- 运行路径中的长编号由内容决定，不必与别人屏幕上的编号相同。

#### 打开报告

- 从项目根目录查找，新报告位于 `result/example/random_split_test/report/report.html`。
- 在文件管理器找到它，双击用浏览器打开即可。
- 也可使用下列命令。

Linux Bash（有桌面环境时）：

```bash
xdg-open result/example/random_split_test/report/report.html
```

Windows Conda PowerShell：

```powershell
Invoke-Item .\result\example\random_split_test\report\report.html
```

- 同目录的 `report.md` 是文字版报告；逐条预测位于任务目录的 `runs/run_morgan-<模型>/predictions.csv`。
- 这里的 `<模型>` 替换为 `rf`、`xgb` 或 `lightgbm`，不是文件名的一部分。

- 后续步骤仍从项目根目录开始。
- 需要关闭终端后继续计算时，参考[后台运行与状态查看](#后台运行与状态查看)；所有操作日志、标准输入文件和 PID 文件统一放在仓库根 `logs/`。

Windows 命令说明操作方法，不能据此认定各后端通过了 Windows 实机验收；请在本机完成安装与运行检查。

### 怎样读结果

- 先看留出检验的指标和逐条预测。
- 每条记录由没有学习该记录的相应模型预测，这些结果汇总称为 OOF（out-of-fold）预测。
- 训练分数反映模型对已学记录的拟合情况，不能替代留出分数。

| 指标 | 含义 |
| --- | --- |
| MAE，平均绝对误差 | 预测值与参考值平均相差多少，与参考值同单位。产率按0到1记录时，`MAE=0.10` 表示平均相差10个百分点。 |
| RMSE，均方根误差 | 也与参考值同单位，对较大的偏差更敏感。 |
| R²，决定系数 | 将预测误差与检验数据均值基线比较，可以为负；它不是“预测正确率”。 |

- 自己运行生成的报告应保留低分和负 R²，用于核对报告与逐条预测；冒烟测试不以分数高低判定是否通过，也不能用来判断某种方法对真实化学问题的预测能力。
- 随机划分也不保证检验中的底物从未在建模记录中出现；要研究新底物，需要另外设计分组和检查结构重叠。

### 遇到问题时

| 现象 | 先检查什么 |
| --- | --- |
| `conda` 无法识别 | Linux先用本机 `conda.sh` 初始化；Windows使用 Anaconda PowerShell Prompt或初始化后重开终端。 |
| 输入命令后出现 `SyntaxError` | 若提示符是 `>>>`，先输入 `exit()` 退出Python，再在终端运行命令。 |
| 找不到 `yonod.py` 或 `requirements.txt` | 当前目录是否正确；根目录应能看见这两个文件，首次使用样例启动则使用上面给出的两级上层路径。 |
| 找不到YAML，或文件名看着正确 | 在文件管理器显示扩展名，确认文件不是 `run.yaml.txt`；核对配置路径相对当前目录的位置。 |
| 路径中有空格 | 终端命令里用引号包住完整路径；进入程序的路径提示后填写实际路径，不附带启动命令。 |
| `ModuleNotFoundError` | 是否激活 `yonod`，是否先装 PyTorch再完成依赖安装；安装错误需要先解决，不能跳过后续检查。 |
| 编号或列名报错 | 核对CSV表头、唯一编号和YAML引用；生成器会重命名部分列，需检查规范化表。 |
| 数值条件或SMILES报错 | 查看具体记录编号与排除报告，核实结构和缺失规则。 |
| 没有生成报告 | 查看终端首次报错及最终完成状态；`result/` 中仅出现目录不代表任务完成。 |
| 同名任务或已存在外测结果被拒绝 | 另取任务名和配置名，同步设置独立结果路径；冻结外测会保护已有指标，不能按普通任务复用方式理解。 |

- 仍需帮助时，可在[项目 Issues](https://github.com/thinktraveller/YONOD/issues)中提供操作系统、Python版本、所用命令、配置及完整报错。
- 提交前移除私人数据和凭据。

## 进阶内容

- [使用自己的 CSV](#使用自己的-csv)
- [配置路径与执行阶段](#配置路径与执行阶段)
- [选择并运行检验方式](#选择并运行检验方式)
- [逐行理解 example.yaml](#逐行理解-exampleyaml)
- [调整数值条件](#调整数值条件)
- [使用 HPO 搜索模型设置](#使用-hpo-搜索模型设置)
- [其他流程字段与已有特征输入](#其他流程字段与已有特征输入)
- [根据问题选择要调节的设置](#根据问题选择要调节的设置)
- [后台运行与状态查看](#后台运行与状态查看)
- [测试输入与结果报告](#测试输入与结果报告)
- [按需安装可选依赖和资产](#按需安装可选依赖和资产)

### 使用自己的 CSV

可直接打开[示例 CSV 数据集](example/random_split_test/dataset/reactions.csv)，参考列名与记录格式；其中数值是冒烟测试使用的人工数据。

#### 先准备数据表

- 每行放一条反应记录，保留表头，并为记录准备稳定、非空且唯一的编号，例如 `record-001`。
- 不要把编号作为模型输入。
- 避免单独使用 `001`、`NA` 等可能在读取时改变含义的编号；需要改编号时保留原编号对应表。
- Excel中使用“另存为”选择CSV UTF-8，直接把 `.xlsx` 后缀改为 `.csv` 不会转换文件格式。

- 参考产率必须是数值，并统一尺度，例如始终用 `0.80` 表示80%，或始终用 `80` 表示80%。
- 程序不自动换算产率尺度；不要在单元格里写 `80%`。
- 列角色也要按化学含义核对，产率、转化率、LC面积比和 ee 不能混为同一目标。

- 当前配置生成向导必须选择一列数值参考值和一列产物 SMILES；反应物、其他组分和数值条件可按提示选择。
- 若你的数据没有产物列，请使用本节末尾的手工配置方法。
- 普通建模运行器支持 `products: []`，可以只用反应物；不要为通过向导虚构产物结构。

#### 生成配置

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
4. 首次使用在描述符菜单输入 `1` 选择 Morgan；询问是否编辑编码配置时可回车使用默认设置。默认包含选中的反应物、其他组分和产物，需确认这些信息在实际预测时可获得。
5. 在模型菜单输入 `2` 选择 Random Forest。描述符或模型菜单直接回车会选择多种方法，其中有的需要额外权重或依赖，首次不要在这两处留空。
6. 按需填写网址、DOI 和备注；报告格式回车选择 HTML 与 Markdown。最后可选的“修复后数据集”是额外副本，回车不生成该副本也不会取消已经完成的规范化处理。

- 生成器将 YAML 保存为 `config/my_reaction.yaml`。
- 规范化表在 `result/my_reaction/feature/docs/my_reaction_normalized_dataset.csv`；有无效行时，排除报告在 `result/my_reaction/report/my_reaction_invalid_report.md`。
- 原始 CSV 保留。

#### 检查后启动

- 用文本编辑器打开生成的 YAML，核对数据路径、列名、输入信息和结果目录。
- 当前生成器默认是随机5折、1轮、种子42，HPO关闭，不会询问训练参数预算；有效编码记录至少要有5条才能执行5折，但达到这个数量下限不代表数据足以支持有意义的科学评价。
- 模型参数未声明时采用已安装库的默认值。

- 数值条件默认缺失即报错，不自动填零或换算单位。
- 存在缺失条件时，先核对数据，或参考 [example.yaml](example.yaml) 的 `dataset.numeric_conditions` 显式设置缺失标记、填补和缩放规则。
- 均值、中位数和缩放参数从相应训练折学习。

- 确认后，在同一个已激活 `yonod` 的终端、项目根目录启动：Linux用 `python -u yonod.py`，Windows用 `python -u .\yonod.py`。
- 首次提示输入：

```text
config/my_reaction.yaml
```

- 新结果都保存在 `result/my_reaction/`，HTML 报告为 `report/report.html`。
- 一次只运行一个任务；任务、编码、模型组合和交叉验证折依次执行。

#### 手工准备 YAML

- [example.yaml](example.yaml) 是完整 schema-2 配置的唯一格式规范。
- 无产物列或需要自行设置时，从它另存新文件再编辑，不要覆盖原样例。
- 以下文件操作从项目根目录执行。

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

- 选择尚未使用的任务名，并同步修改 `project_name`、数据路径和角色、编码与模型、检验设置、`outputs.root: ./result/my_reaction`、`artifacts.output_dir: ./result/my_reaction/feature`，保留 `outputs.report_formats: [html, markdown]`。
- 只声明已选模型的参数。
- 保存后仍通过 `yonod.py` 首次提示输入配置路径。

原始格式样例使用12条人工记录、两折和10棵RF树，只用于轻量工程冒烟测试。研究任务需要按自己的问题确定数据量、模型设置和检验设计。

### 配置路径与执行阶段

- 推荐将运行配置统一保存在根 `config/`：研究配置使用 `config/<task_name>.yaml`，三个测试已准备好 `config/example/<测试名>/run.yaml`。
- 直接位于 `config/` 下的 YAML，以 `./` 开头的路径按项目根解析；显式 `../` 及其他位置配置内的相对路径按 YAML 所在目录解析，绝对路径保持绝对。
- 嵌套的测试 YAML 使用 `../../../example/<测试名>/` 读取输入，用 `../../../result/example/<测试名>` 将结果保存到项目根目录。
- 配置保存为 UTF-8 纯文本，缩进使用空格；Windows下检查文件没有被另加 `.txt` 后缀。

| `stage` | 执行内容 |
| --- | --- |
| `features` | 只准备或复用编码，可以没有参考标签。 |
| `train` | 从 `artifacts.input_manifest` 指定的可核验特征包训练，不补算缺失编码。 |
| `all` | 准备编码后进行普通内测。 |
| `benchmark` | 按所选协议及必需字段分派专用执行器。 |

- 推荐将所有运行结果统一放在项目根目录 `result/`，每项任务使用独立的 `result/<task_name>/`。
- 每个研究任务使用独立的 `outputs.root: ./result/<task_name>`，并将编码放入子目录，例如 `artifacts.output_dir: ./result/<task_name>/feature`，保留 `outputs.report_formats: [html, markdown]`。
- 操作日志和PID在根 `logs/`，最终建模结果在任务目录内，不放入 `derived/`。

### 选择并运行检验方式

- 三个首次使用样例都使用人工记录，主要用于冒烟测试，检查各自流程能否完整运行。
- 一次只运行一个任务。
- 每组命令都从项目根目录开始。
- 每个测试都有已准备好的 `config/example/<测试名>/run.yaml`；运行器首提示输入各例给出的完整路径。
- 以下报告路径均相对项目根目录。

| 你的问题 | 流程与配置 | 怎样分开建模和检验 |
| --- | --- | --- |
| 同一张表中，模型对留出记录预测得怎样？ | 普通 `stage: all`、`evaluation.protocol: outer_kfold`；[随机样例](config/example/random_split_test/run.yaml) | 各折中的训练行和留出行分开；同一种底物仍可能在两侧出现。 |
| 怎样执行指定的成员名单或整组留出？ | `stage: benchmark`、`manifest_outer_cv`；[固定名单样例](config/example/fixed_split_test/run.yaml) | 采用清单中真实的 `train`/`valid` 成员及分组键。 |
| 提前固定的建模方案，对另外一张带参考值的表预测得怎样？ | `stage: benchmark`、`frozen_train_external_test`；[外测样例](config/example/external_test/run.yaml) | 每个组合不使用外测标签拟合和预测，随后才读取标签评分。 |

#### 随机划分内测

1. 查看[12行数据表](example/random_split_test/dataset/reactions.csv)和 [run.yaml](config/example/random_split_test/run.yaml)。

   - 本例使用两列反应物结构、Morgan 编码、RF/XGBoost/LightGBM，进行两折、一轮检验。
2. 按系统运行下面的命令，在首次提示输入 `config/example/random_split_test/run.yaml`。

Linux Bash：

```bash
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate yonod
export PYTHONUTF8=1
python -u yonod.py
```

Windows Conda PowerShell：

```powershell
conda activate yonod
$env:PYTHONUTF8 = "1"
python -u .\yonod.py
```

3. 核对 `features=ready`、三个组合的 `complete` 或 `reused` 状态及两种报告。

   - 新 HTML 报告是 `result/example/random_split_test/report/report.html`；逐条预测位于该任务的 `runs/run_morgan-<model>/predictions.csv`。

4. 使用自己的 CSV 时，在根 `config/` 生成新配置，或从 [example.yaml](example.yaml) 另存独立副本。

   - 修改编号、目标、输入列、方法与独立输出。
   - `evaluation.n_splits` 至少为2，`n_repeats` 至少为1，`shuffle` 为布尔值，`seed` 控制打乱。
   - 每轮打乱使用 `seed + repeat - 1`；`shuffle: false` 时，多轮使用相同的行划分。
   - 有效编码行数必须不少于折数。

- 普通运行器始终建立 KFold 划分。
- 只在普通 `all/train` 配置中写 `manifest_outer_cv` 不会导入自己的清单；需要真正固定成员时，使用下一条专用 benchmark 路线。

本表没有产物、数值条件或类别输入列，第二列结构始终为 `CO`；恒定输入不能检验对任意新组分的预测能力。数据表中出现的列不会自动成为模型输入。

当前配置的三个模型各使用两棵树，关闭 HPO 与额外训练/留出效果诊断，不指定线程数，采用各模型库默认。这些小设置用于降低冒烟测试成本。

运行完成后，将输入 CSV 中的 `reaction_id`、`yield` 与各模型保存的 `runs/run_morgan-<model>/predictions.csv` 对齐，检查两折记录以及生成的两种报告。随机测试的全部结果都由本次运行写入 `result/example/random_split_test/`。

#### 固定名单与分组检验

1. 查看[角色解释表](example/fixed_split_test/splits/membership_guide.csv)、[实际24行清单](example/fixed_split_test/splits/fixed_split_manifest.csv)和[配置](config/example/fixed_split_test/run.yaml)。

   - 每折列出全部12条记录；第1折留出 `block_A`，第2折留出 `block_B`。
2. 保持 `stage: benchmark`、`evaluation.protocol: manifest_outer_cv`、分组声明、`evaluation.split_manifest` 与 `benchmark.task_state` 相互一致。
3. 从项目根目录启动，首提示输入 `config/example/fixed_split_test/run.yaml`。

Linux Bash：

```bash
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate yonod
export PYTHONUTF8=1
python -u yonod.py
```

Windows Conda PowerShell：

```powershell
conda activate yonod
$env:PYTHONUTF8 = "1"
python -u .\yonod.py
```

4. 核对 RF 两折成功记录。

   - 报告是 `result/example/fixed_split_test/report/benchmark_report.html`；实际采用的清单和预测在该任务的 `docs/manifests/`、`docs/predictions/`。

5. 用自己的数据时，先准备稳定编号和分组键，再分配折成员。

   - 另存新的 CSV 或 Parquet 清单，更新其数据校验码和成员，在新的根 `config/` YAML 中引用。
   - 分组列只用于划分时，不要把它放入描述符输入。
   - 新任务输出使用 `./result/<task_name>` 与 `./result/<task_name>/feature`。

- 当前接口至少要求两折。
- 每折的训练、验证部分均非空，编号全集与数据表一致，同一组的记录不能分散在该折的训练和验证部分，每条记录每轮恰好一次 `valid`。
- 仅一次“8条训练、4条测试”的名单不能代替这份交叉验证清单。
- 固定种子或关闭打乱也不能表达任意指定的成员。

这份 CSV 在记录编号、参考产率和两列结构之外增加 `fixed_block`。`reaction_id` 用于对齐名单；`yield` 是 0–1 尺度的人工参考值；`fixed_block` 只控制成员划分，不作为模型输入。

| 折 | `train` 建模成员 | `valid` 留出成员 |
| --- | --- | --- |
| 1 | `fixed-07` 至 `fixed-12`（`block_B`） | `fixed-01` 至 `fixed-06`（`block_A`） |
| 2 | `fixed-01` 至 `fixed-06`（`block_A`） | `fixed-07` 至 `fixed-12`（`block_B`） |

`block_A` 和 `block_B` 只是人工分组，没有底物、实验室或实验批次的化学含义。12 行解释表用 `held_out` 表示留出记录；程序读取的 24 行正式清单只使用 `train` 与 `valid`。固定名单决定哪些记录用于检验，固定模型参数决定怎样建模，两者作用不同。

若没有 `evaluation.split_manifest`，专用执行器会根据 `evaluation.grouping` 生成清单，目前支持五种策略：

| `grouping.strategy` | 设置与行为 |
| --- | --- |
| `repeated_kfold` | 不按化学结构分组；按原 CSV 行序进行打乱 KFold，每轮使用 `seed + repeat - 1`。这一专用分支固定打乱，`evaluation.shuffle` 不控制它。 |
| `precomputed_column` | 必需 `group_column`：读取现有非空分组键，内部加 `precomputed:` 前缀。 |
| `component_holdout` | 必需非空 `component_cols`：规范化这些结构列，以按顺序组成的联合身份分组。多列时按“组合”分组，不保证各单独组分都不交叉。 |
| `substrate_scaffold` | 必需 `substrate_col`：采用包含手性的 Murcko 骨架；无环结构退回规范化结构身份。 |
| `reaction_fingerprint_cluster` | `reaction_cols` 默认使用分子角色列；`similarity_threshold` 默认 `0.7`，范围 `(0, 1]`；分组 Morgan 的 `radius` 默认 `2`、至少为0，`n_bits` 默认 `2048`、至少为8。实现将分子组分合并后按 Tanimoto/Butina 聚类。 |
| 上述四种实际分组策略 | 可选 `max_group_fraction` 默认 `0.8`，范围 `(0, 1]`：生成划分时最大组占比超过该值则拒绝。组数至少等于折数，程序按组大小贪心平衡各折。 |

- 无法解析的分组结构使用明确的 fallback 键，不静默删行，也不自动改为普通 KFold。
- 分组名本身不能证明两侧不存在相同底物或来源。
- 导入已有清单时，以其中的组键和成员为准；修改分组设置不会重新生成导入清单的成员。

清单需要以下列：

| 列 | 含义与准备要求 |
| --- | --- |
| `run_id` | 全文件一个非空来源运行标识；导入后保存在 `source_run_id`，任务内副本使用新运行编号。 |
| `split_id` | 全文件同一个划分方案标识。 |
| `sample_id` | 数据表稳定编号；每折包含相同的完整记录全集。 |
| `group_id` | 已核对的组键；每折训练与验证的组不相交。 |
| `group_strategy` | 分组策略名称。 |
| `repeat`、`fold` | 从1开始准备轮数和折数，组合数量与 YAML 一致。 |
| `role` | 使用 `train` 或 `valid`，不是 `test`，也不是解释表中的 `held_out`。 |
| `seed` | 本轮保存的划分种子。 |
| `source_row_index` | 原 CSV 中从0开始的数据行位置，不含表头；准备清单时保持准确。 |
| `dataset_sha256` | 原 CSV 文件字节的 SHA-256，各行一致。 |
| `grouping_params_json` | 保存分组设置的 JSON 对象，放进 CSV 时正确转义引号。 |
| 可选 `population_id`、`split_hash` | 自动生成清单和较新协议使用的额外来源字段。 |

假设文件为 `dataset/my_reactions.csv`，先计算哈希再填清单。以下操作只读取文件。

Linux Bash：

```bash
sha256sum dataset/my_reactions.csv
```

Windows Conda PowerShell：

```powershell
Get-FileHash -Algorithm SHA256 -LiteralPath .\dataset\my_reactions.csv
```

导入器检查数据哈希、来源运行/划分标识、编号全集、各折覆盖情况和组隔离。仍需自己核对源行号、策略备注和组定义是否与 CSV 一致；程序不会重新推断它们的化学含义。

#### 冻结设置后进行外测

1. 查看[12行训练表](example/external_test/dataset/train.csv)、[6行外测表](example/external_test/dataset/external.csv)、[审计文件](example/external_test/audit/external_audit.json)和 [run.yaml](config/example/external_test/run.yaml)。

   - 先固定描述符输入、模型参数和评分方案，再查看外测结果。

2. 保留 `stage: benchmark`、`frozen_train_external_test`、`evaluation.seed` 和完整外测字段。

   - 不能混入 `n_splits`、`n_repeats`、`shuffle`、`grouping`、`split_manifest`、`population_manifest`。
3. 从项目根目录启动，首提示输入 `config/example/external_test/run.yaml`。

Linux Bash：

```bash
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate yonod
export PYTHONUTF8=1
python -u yonod.py
```

Windows Conda PowerShell：

```powershell
conda activate yonod
$env:PYTHONUTF8 = "1"
python -u .\yonod.py
```

4. RF 使用全部12条训练记录拟合一次，再预测6条外测记录。

   - 查看 `result/example/external_test/report/external_test_report.html`，以及该任务根下的 `docs/predictions/morgan__rf.parquet`、`docs/metrics/external_test_metrics.csv`。
   - 外测没有交叉验证；保存的分数也不是独立置信区间。

5. 真实任务需另建训练、外测与审计文件，再建立独立根 `config/` YAML 和任务输出目录。

   - `external_test.column_map` 从训练结构列映射到外测结构列；目标由训练 `column_roles.label` 与外测 `label_col` 分别指定。

| `evaluation.external_test` 下的字段 | 含义 |
| --- | --- |
| `path` | 外测 CSV 路径。 |
| `sample_id_col` | 非空唯一的外测编号；与训练编号不相交。 |
| `label_col` | 最终评分使用的外测数值参考值。 |
| `source_id_col` | 非空唯一的来源记录编号。审计声明 `physical_dataset_id` 时，每项须以该编号加 `:` 开头。 |
| `group_col` | 外测分组列，至少两个非空组；当前报告记录组数，预测中保留组键以供后续分组分析。 |
| `column_map` | 训练特征列名到外测列名的一对一映射，覆盖所选结构输入，不映射目标列。 |
| `expected_sha256` | 外测 CSV 的准确 SHA-256。 |
| `audit_manifest` | JSON 来源/重叠审计路径。 |
| `audit_manifest_sha256` | 审计 JSON 的准确 SHA-256。 |

#### 外测审计为何必需

[外测审计文件](example/external_test/audit/external_audit.json)是冻结外测流程的必需输入，记录来源与数据身份、规范化结构重叠，以及本例属于人工教学数据、不具有独立实验来源的事实。

- YAML 的 `external_test.expected_sha256` 固定外测 CSV 的版本；`audit_manifest` 指定已审阅的 JSON，`audit_manifest_sha256` 固定其准确字节，用于识别数据与审计版本。
- 审计记录 `candidate_source_sha256`、`physical_dataset_id`、行数，以及胺、酸、产物的 `exact_canonical_overlap`。本例的候选来源就是外测 CSV；真实研究也可能引用更上游的来源文件。
- 运行器在执行前检查文件存在、YAML 中固定的两份文件哈希、必需来源与零重叠声明、声明的行数，以及由 `physical_dataset_id` 确定的来源编号前缀；它不自行重算规范化结构交集，也不证明实验来源独立。
- 删除该文件会使外测校验失败。为通过校验而手改零重叠声明不等于完成审计；哈希只能固定文件版本，不能证明声明真实。

修改训练或外测记录时，应重新核查实际结构与来源，更新审计 JSON 及其哈希，并更新 YAML 中的外测 CSV 哈希。可按上文 Bash/PowerShell 命令计算正确文件的哈希，不能将本例零值或人工来源声明照搬到真实研究。结果中保留审计限制，避免将未经核查的数据版本或没有依据的独立性声明当作已审阅的外测记录。

- 当前适配器只接受静态描述符，不支持数值条件、OHE、AutoGluon、HPO 或外测早停，不保存普通完整特征/最终模型包。
- 已有外测运行输出会在拟合前被拒绝，不恢复或覆盖；需新建任务名和输出目录。
- 样例含产物输入，实际预测时须已知目标产物结构。
- 另存文件、精确结构零重叠，都不能证明独立实验来源或分布独立。

两份 CSV 故意使用不同列名：

| 训练表列 | 外测表列 | 作用 |
| --- | --- | --- |
| `amine_smiles` | `external_amine` | 胺结构，由 `column_map` 对应。 |
| `acid_smiles` | `external_acid` | 酸结构，由 `column_map` 对应。 |
| `product_smiles` | `external_product` | 预测时已知的产物结构，由 `column_map` 对应。 |
| `yield` | `measured_yield` | 单独声明的参考目标，不放入结构映射。 |
| `reaction_id` | `external_id` | 唯一记录编号，不作为分子输入。 |
| 无 | `source_record_id` | 来源编号，统一以 `teaching-external-v1:` 开头。 |
| 无 | `amine_group` | 外测的三个胺组，保留给后续分组分析。 |

`measured_yield` 只是演示常见列名，其中仍是人工参考值；`0.22` 表示 22%，本例没有自动百分数换算设置。这条外测评分流程需要参考目标；只有新结构、没有目标值的预测属于另一种操作路径。

本例审计记录给出下列规范化结构计数：

| 角色 | 训练中不同结构数 | 外测中不同结构数 | 完全相同的结构交集 |
| --- | --- | --- | --- |
| 胺 | 4 | 3 | 0 |
| 酸 | 3 | 3 | 0 |
| 产物 | 12 | 6 | 0 |

本例没有更上游的实测原始表，所以 `candidate_source_sha256` 指向外测 CSV，`physical_dataset_id` 为 `teaching-external-v1`。真实数据需根据实际文件重做来源、重叠与哈希核查；三个小组和零完全结构重叠不能证明独立实验来源。

### 逐行理解 example.yaml

- [example.yaml](example.yaml) 是唯一完整 YAML 规范。
- 下表对应当前104行文件中的每条有效非空行，用完整字段路径说明其含义。
- 缩进表示从属层级；`-` 表示列表项，`[]` 是空列表，`{}` 是空映射，`null` 是空值，`true`/`false` 是布尔值。
- 缩进用空格，不用制表符。
- 修改格式规范后，行号也会变化。

| 行号 | 完整字段路径 | 含义与可调整内容 |
| --- | --- | --- |
| 8 | `schema_version` | `"2.0"`，保持运行器支持的版本。 |
| 9 | `project_name` | 任务名；新任务使用新名字及对应独立输出路径。 |
| 10 | `stage` | 当前 `all`；按所需流程选择 `features`、`train`、`all` 或 `benchmark`。 |
| 12 | `dataset` | 数据表与列契约的容器。 |
| 13 | `dataset.path` | 冒烟测试的人工数据 CSV 路径；改为自己的输入表。 |
| 14 | `dataset.sample_id_col` | 稳定、非空、唯一的记录编号列。 |
| 15 | `dataset.column_roles` | 声明每列用途的容器。 |
| 16 | `dataset.column_roles.label` | 数值参考目标；建模与评分时必需。 |
| 17 | `dataset.column_roles.reactants` | 反应物结构列列表。 |
| 18 | `dataset.column_roles.products` | 产物结构列列表；本例为空。 |
| 19 | `dataset.column_roles.others` | 其他分子结构列；本例为空。 |
| 20 | `dataset.column_roles.conditions` | 按顺序输入的数值条件列；本例为空。 |
| 21 | `dataset.column_roles.categoricals` | 显式选择相应编码时使用的类别列。 |
| 27 | `dataset.numeric_conditions` | 数值预处理声明的容器。 |
| 28 | `dataset.numeric_conditions.defaults` | 各数值条件共用的规则。 |
| 29 | `dataset.numeric_conditions.defaults.missing` | 当前 `strategy: error`；也支持 `mean`、`median`、`constant`。 |
| 30 | `dataset.numeric_conditions.defaults.scaling` | 当前 `none`；可选 `standard`、`minmax`，只从训练行学习。 |
| 31 | `dataset.numeric_conditions.columns` | 按源列逐项覆盖；本例无数值条件，因此为空。 |
| 35 | `descriptors` | 独立编码候选的列表；多个候选不会自动合成一个编码。 |
| 36 | `descriptors[0].id` | 当前候选唯一标识 `morgan`，用于输出身份。 |
| 37 | `descriptors[0].descriptor` | 算法名称 `morgan`；使用已实现的注册名。 |
| 38 | `descriptors[0].lifecycle` | `static_descriptor`；按算法支持的生命周期填写，不能任意切换。 |
| 39 | `descriptors[0].mode` | `concat` 按顺序拼接所选组分编码；合法模式取决于算法。 |
| 40 | `descriptors[0].columns` | 该编码使用的源列及顺序。 |
| 41 | `descriptors[0].params` | 编码专属参数；Morgan 的当前 YAML 接口没有可调参数，保持空映射。 |
| 43 | `artifacts` | 特征产物位置。 |
| 44 | `artifacts.output_dir` | 当前任务的 `feature/` 子目录，随任务名同步修改；`train` 另需 `input_manifest`。 |
| 46 | `models` | 选择模型标识；本例只有 `rf`。 |
| 47 | `model_params` | 仅为所选模型声明参数节。 |
| 52 | `model_params.rf` | 随机森林设置。 |
| 53 | `model_params.rf.estimator` | 送入模型工厂的构造参数。 |
| 54 | `model_params.rf.estimator.n_estimators` | `10` 棵树，仅为轻量工程预算。 |
| 55 | `model_params.rf.estimator.max_depth` | `null`，不显式限制深度。 |
| 56 | `model_params.rf.estimator.min_samples_leaf` | `1`，每叶最少样本数。 |
| 57 | `model_params.rf.estimator.max_features` | `1.0`，分裂时考虑全部输入特征列。 |
| 58 | `model_params.rf.estimator.random_state` | 模型种子 `42`，与数据划分种子是不同参数。 |
| 59 | `model_params.rf.estimator.verbose` | 模型库输出级别 `0`。 |
| 66 | `hpo` | 可选模型参数搜索容器。 |
| 67 | `hpo.enabled` | `false`，不搜索；关闭时 HPO 节只接受 `enabled`。 |
| 86 | `evaluation` | 训练/留出成员规则的容器。 |
| 91 | `evaluation.protocol` | `outer_kfold`，普通随机交叉验证路线。 |
| 92 | `evaluation.n_splits` | `2` 折；整数至少为2，且不超过有效行数。 |
| 93 | `evaluation.n_repeats` | 完整交叉验证 `1` 轮；整数至少为1。 |
| 94 | `evaluation.shuffle` | `true`，普通 KFold 前打乱记录。 |
| 95 | `evaluation.seed` | 划分种子 `42`，不能代替具体成员名单。 |
| 97 | `outputs` | 结果位置与报告设置。 |
| 98 | `outputs.root` | 任务根目录：本冒烟测试使用 `result/example/yonod_schema2_smoke/`；自己的任务使用 `result/<task_name>/`。 |
| 99 | `outputs.report_formats` | 同时生成 `html` 和 `markdown`，日常使用保留两种。 |
| 100 | `outputs.diagnostics` | 可选训练与留出效果诊断设置。 |
| 101 | `outputs.diagnostics.enabled` | `true`，启用诊断，不改变划分成员。 |
| 102 | `outputs.diagnostics.save_train_predictions` | `false`，是否保存逐条训练预测；诊断关闭时不能设为真。 |
| 103 | `metadata` | 人工说明，不作为模型输入。 |
| 104 | `metadata.notes` | 说明人工 fixture 和独立任务约定；可替换为自己的来源与研究目的。 |

- 空行只便于阅读。
- 整行 `#` 注释及行末 `#` 后的文字不执行。
- 第1至7行说明启动位置；22至26行说明数值条件；33至34行说明候选和生命周期；48至51行说明模型节与资源；61至65行说明 HPO 范围；68至84行是未启用的 HPO 示例；87至90行说明协议切换。
- 注释中的 HPO 字段见后文。
- 不能在原有 `hpo` 旁再取消注释另一份同名节，YAML 重复键会被拒绝。

### 调整数值条件

- 在 `dataset.column_roles.conditions` 按输入顺序声明源列，不能与编号、标签、分子或类别角色重叠。
- 预处理设置写入 `dataset.numeric_conditions`，当前支持的全部键如下。

| 位置 | 支持的键、值与行为 |
| --- | --- |
| `defaults` | 只接受 `missing` 和 `scaling`；默认 `missing.strategy: error`、`scaling: none`。 |
| `columns.<source_column>` | 只接受 `name`、`missing_values`、`missing`、`scaling`、`unit`。源列必须已在 `conditions` 声明。 |
| `name` | 非空且在数值输出中唯一的名称；默认等于源列名。 |
| `missing_values` | 不重复的字符串和/或 `null` 列表，默认 `[]`。用 `null` 声明空值，用字符串如 `""` 声明对应原始标记；未声明的空值或非数值会报错。 |
| `missing.strategy` | `error`、`mean`、`median`、`constant`，也可直接写策略字符串。均值/中位数要求训练部分至少一个观测值。 |
| `missing.constant` | 仅 `constant` 策略必需的有限数值；在单位换算后的数值尺度上填补。 |
| `scaling` | `none`；`standard` 使用训练均值和总体标准差（`ddof=0`）；`minmax` 使用训练最小值和极差。训练尺度为0时替换为1，保留该列。 |
| `unit` | 只接受 `source`、`target`、`scale`、`offset`。声明时源/目标字符串必需；有限 `scale` 默认1且不能为0，有限 `offset` 默认0。转换值=原值×scale+offset；同单位必须为恒等换算。省略则为无量纲恒等转换。 |

单位转换是显式算术，不自动识别单位。填补和缩放只拟合当前训练部分，HPO 内层也分别拟合；留出与未来记录只使用已保存状态。冻结外测目前不接受数值条件输入。

### 使用 HPO 搜索模型设置

- HPO 在训练数据内搜索模型参数，不选择实验条件。
- 当前接入 RF、XGBoost 和 LightGBM，通过内外层划分隔离外层留出标签。
- 每个编码×模型×外层折分别搜索，以各内层折分数的等权算术平均为目标。

| HPO 字段 | 取值与限制 |
| --- | --- |
| `enabled` | 布尔值，省略或 `false` 表示关闭；关闭时不能声明其他 HPO 键。 |
| `models` | 顶层 `models` 的非空不重复子集，只含 `rf`、`xgb`、`lightgbm`。 |
| `sampler` | 仅支持 `tpe`。 |
| `seed` | 必需的整数搜索种子。 |
| `objective.metric`、`objective.direction` | `rmse`/`mae` 配 `minimize`；`r2`/`kendall_tau` 配 `maximize`，两者都要显式声明。 |
| `inner_cv.n_splits` | 整数至少为2，不超过当前外层训练记录数；分组搜索还需要足够的组数。 |
| `inner_cv.seed`、`inner_cv.shuffle` | 显式整数与布尔值。分组严格 HPO 使用 GroupKFold，必须 `shuffle: false`；`repeated_kfold` 为不分组的选择。 |
| `budget.max_trials` | 每次搜索（study）的参数尝试（trial）次数，必须为正整数。 |
| `budget.study_timeout_s` | 可选有限正数，study 时间软上限，正在运行的 trial 可以完成。 |
| `budget.task_timeout_s` | 可选有限正数，跨恢复会话保留的累计任务时间预算；是调度软上限，不是杀进程计时器。 |
| `search_spaces` | 与 `hpo.models` 精确对应，每个模型一项。 |
| `search_spaces.<model>.preset` | 可选 `compact-v1`，不接受其他预设名。 |
| `search_spaces.<model>.parameters` | 参数名到分布的映射；有预设时可省略，无预设时需非空。同名项整体覆盖预设分布；搜索值覆盖同名固定构造参数并记录此行为。 |
| `final_model.enabled` | 可选布尔值，默认关闭；关闭时不能带开发数据编号或最终预算。 |
| `final_model.development_population_id` | 开启最终模型时必需的非空开发数据身份。 |
| `final_model.budget` | 为最终模型单独安排的搜索预算，包含 `max_trials` 及可选 study/task 时间键。单独搜索后在完整开发数据上重训，不产生独立测试分数。 |

- 分布的 `type` 只支持 `int`、`float`、`categorical`。
- 整数/浮点分布需要有限 `low`、`high`，且 low≤high；可选正数 `step` 必须整除区间。
- 整数边界和步长必须为整数，默认步长1。
- 可选 `log` 为布尔值且要求 low>0；整数对数分布步长只能为1，浮点对数分布不能同时声明 step。
- 类别分布只接受非空 `choices`，由不重复的安全标量组成（`null`、字符串、布尔值及有限数值），不接受任意表达式或可调用对象。

| 模型 | 当前全部可搜索构造参数 |
| --- | --- |
| RF | `n_estimators`、`max_depth`、`min_samples_leaf`、`min_samples_split`、`max_features`、`max_samples`、`min_impurity_decrease`、`ccp_alpha` |
| XGBoost | `n_estimators`、`learning_rate`、`max_depth`、`min_child_weight`、`subsample`、`colsample_bytree`、`gamma`、`reg_alpha`、`reg_lambda`、`max_delta_step` |
| LightGBM | `n_estimators`、`learning_rate`、`num_leaves`、`min_child_samples`、`min_child_weight`、`subsample`、`colsample_bytree`、`reg_alpha`、`reg_lambda`、`min_split_gain`、`max_depth` |

- `compact-v1` 中，RF 搜索100至800棵树（步长100）、深度 `null/4/8/12/20/32`、叶最小样本1至20、特征比例0.3至1.0。
- XGBoost 搜索100至800棵树（步长100）、对数学习率0.01至0.2、深度2至10、对数子节点权重1至20、行比例0.6至1.0、特征比例0.5至1.0。
- LightGBM 搜索100至800棵树（步长100）、对数学习率0.01至0.2、叶数8至64、子节点最小样本5至50、特征比例0.5至1.0。
- 这些只是软件预设，不代表适合所有数据集的科学预算。

- 普通 HPO 要求 `outer_kfold`，严格 HPO 要求 `stage: benchmark` 和 `manifest_outer_cv`；冻结外测和固定论文复现协议不支持。
- 当前 HPO 拒绝 runtime 早停，严格 HPO 也拒绝非空模型 `fit` 节。
- 仅特征阶段只校验声明，不建立 study。
- LightGBM 搜索时不能固定被搜索参数的别名；搜索 `subsample` 需固定正数 `subsample_freq`，还会核对搜索 `num_leaves` 与固定正 `max_depth` 是否冲突。
- XGBoost 预设要求 `booster: gbtree`。

### 其他流程字段与已有特征输入

| 字段 | 当前用途 |
| --- | --- |
| `artifacts.input_manifest` | `stage: train` 必需，指向已有、可核验的特征包清单，不是 CSV 路径。该阶段不补算缺失编码，不能替换为另一份记录或输入契约的特征包。 |
| `benchmark.task_state.backend` | `stage: benchmark` 必须为 `sqlite`。 |
| `benchmark.task_state.resumable` | `stage: benchmark` 必须为 `true`。固定名单流程用任务状态核对并恢复进度；冻结外测适配器仍拒绝已有运行输出，不恢复该任务。 |
| `benchmark.population_id`、`benchmark.dataset_id` | 可选非空来源标识，描述记录集合及数据版本；字符串本身不证明来源独立。 |
| `benchmark.reproduction_protocol` | 为既有专用复现行为保留的可选映射，三个首次使用样例流程不需要；新普通任务不要照抄归档复现设置。 |
| `outputs.diagnostics.enabled` | 布尔值，在普通/固定名单流程对支持的模型默认启用。当前训练/留出诊断支持 `rf`、`xgb`、`lightgbm`，SVM 和 AutoGluon 记为不支持；冻结外测不生成这类配对诊断。 |
| `outputs.diagnostics.save_train_predictions` | 布尔值，默认关闭；保存支持诊断的逐条训练预测。诊断关闭时不能为真。关闭诊断不会关闭正常留出评分。 |

- `evaluation` 下的完整顶层白名单为：`protocol`、`n_splits`、`n_repeats`、`shuffle`、`seed`、`grouping`、`split_manifest`、`population_manifest`、`metrics`、`external_test`。
- 三条当前启动路线实际使用的字段以前述对应说明为准。
- 通用 schema 接受 `population_manifest` 或 `metrics`，不代表它们能在这三条路线切换记录集合或自定义评分，此处应省略。
- 冻结外测明确禁止 `population_manifest` 和内测字段。
- 字段通过通用格式校验，不等于适配器实现了该设置。

### 根据问题选择要调节的设置

先保存一份能核对的固定配置，再按研究问题调整设置。比较方法时保持检验成员和评分口径一致；下方开发者参数表列出完整字段及协议限制。

| 想改变什么 | YAML 位置 | 需要理解的影响 |
| --- | --- | --- |
| 输入信息 | `dataset.column_roles`、`descriptors[].columns` | 决定模型能使用哪些结构、条件或类别；预测时必须能获得这些信息。 |
| 普通随机检验 | `evaluation.n_splits`、`n_repeats`、`shuffle`、`seed` | 改变成员或重复次数；更多折与轮次会增加拟合次数。固定名单和外测采用各自协议。 |
| 计数型 Morgan 表示 | `mfp` 的 `descriptors[].params.radius`、`fp_size`、`profile` | 改变计数指纹；二值 `morgan` 未开放这些 YAML 参数。 |
| 树模型预算与复杂度 | `model_params.<model>.estimator.n_estimators`、`max_depth` | 控制已支持树模型的规模；各库接受的类型和边界不同。 |
| 随机森林采样 | `model_params.rf.estimator.max_features`、`min_samples_leaf`、`max_samples` | 控制特征选择、叶节点规模和自助采样；具体依赖关系见 RF 参数表。 |
| 提升树设置 | XGBoost/LightGBM 的 `estimator.learning_rate`、行列采样和正则项 | 改变提升过程；字段名称按所选模型填写。 |
| 支持向量回归 | `model_params.svm.estimator.C`、`epsilon`、`kernel`、`gamma` | 控制惩罚强度、不敏感误差带与核函数；缩放和 PCA 另在预处理节设置。 |
| AutoGluon 预算 | `model_params.autogluon.fit.time_limit`、`presets`、`num_cpus` | 控制训练时间、方案和CPU资源，需要可选依赖。 |
| 自动参数搜索 | `hpo.enabled`、`hpo.budget.max_trials`、`hpo.search_spaces` | 为支持的模型在每个外层训练折内搜索；启动前明确预算。 |
| 报告 | `outputs.report_formats`、`outputs.diagnostics` | 选择报告与可选训练诊断；诊断不能替代留出评价。 |

只声明已选模型的参数。更改研究配置时另取任务名并使用独立结果目录。线程设置由本机资源决定，各任务和折仍依次执行。

### 后台运行与状态查看

第一次冒烟测试建议前台运行，便于查看错误。需要退出终端后继续计算时，从项目根目录选择下面一组后台启动命令；将 `test_name` 或 `$testName` 设为 `random_split_test`、`fixed_split_test` 或 `external_test`，等当前任务完成后再启动下一个。

#### Linux Bash

```bash
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate yonod
export PYTHONUTF8=1
test_name=random_split_test
mkdir -p logs
printf '%s\n' "config/example/$test_name/run.yaml" > "logs/$test_name.stdin.txt"
nohup python -u yonod.py < "logs/$test_name.stdin.txt" > "logs/$test_name.log" 2>&1 &
echo $! > "logs/$test_name.pid"
```

需要时从同一项目根目录查询状态：

```bash
test_name=random_split_test
test_pid="$(cat "logs/$test_name.pid")"
ps -p "$test_pid" -o pid,ppid,stat,etime,cmd
tail -n 100 "logs/$test_name.log"
```

#### Windows Conda PowerShell

```powershell
conda activate yonod
$env:PYTHONUTF8 = "1"
$testName = "random_split_test"
$testLogs = Join-Path $PWD.Path "logs"
New-Item -ItemType Directory -Path $testLogs -Force | Out-Null
$testInput = Join-Path $testLogs "$testName.stdin.txt"
[IO.File]::WriteAllText($testInput, "config/example/$testName/run.yaml`n", [Text.UTF8Encoding]::new($false))
$testProcess = Start-Process -FilePath (Get-Command python).Source -ArgumentList @("-u", "yonod.py") -WorkingDirectory $PWD.Path -RedirectStandardInput $testInput -RedirectStandardOutput (Join-Path $testLogs "$testName.log") -RedirectStandardError (Join-Path $testLogs "$testName.stderr.log") -PassThru
$testProcess.Id | Set-Content -LiteralPath (Join-Path $testLogs "$testName.pid") -Encoding ascii
```

标准输出与错误输出分开保存，查询时检查两份日志：

```powershell
$testName = "random_split_test"
$testLogs = Join-Path $PWD.Path "logs"
$testPid = [int](Get-Content -LiteralPath (Join-Path $testLogs "$testName.pid"))
Get-Process -Id $testPid -ErrorAction SilentlyContinue
Get-Content -LiteralPath (Join-Path $testLogs "$testName.log") -Encoding UTF8 -Tail 100
Get-Content -LiteralPath (Join-Path $testLogs "$testName.stderr.log") -Encoding UTF8 -Tail 100
```

日志、PID 与运行器输入文件统一放在根 `logs/`，模型结果仍位于 `result/example/<测试名>/`。进程结束或出现结果目录不能单独证明成功；还需检查完成状态、逐条预测和两种报告。Linux 命令不能代替 Windows 实机验收。

### 测试输入与结果报告

三个测试使用对应的数据、配置和结果目录。以下报告路径指成功运行后生成的文件：

| 测试名 | 已保存配置 | 输入 | `result/example/` 下的新网页报告 |
| --- | --- | --- | --- |
| `random_split_test` | [run.yaml](config/example/random_split_test/run.yaml) | [12 行 CSV](example/random_split_test/dataset/reactions.csv) | `random_split_test/report/report.html` |
| `fixed_split_test` | [run.yaml](config/example/fixed_split_test/run.yaml) | [CSV](example/fixed_split_test/dataset/reactions.csv)、[角色解释表](example/fixed_split_test/splits/membership_guide.csv)、[正式名单](example/fixed_split_test/splits/fixed_split_manifest.csv) | `fixed_split_test/report/benchmark_report.html` |
| `external_test` | [run.yaml](config/example/external_test/run.yaml) | [训练 CSV](example/external_test/dataset/train.csv)、[外测 CSV](example/external_test/dataset/external.csv)、[审计](example/external_test/audit/external_audit.json) | `external_test/report/external_test_report.html` |

三个测试均提供输入与已准备好的 YAML，不附带预生成的报告、分数或预测；运行后才会将文件写入各自的 `result/example/<测试名>/` 目录。

每份生成报告的 `report/` 目录同时保存 Markdown `.md` 版。固定名单逐条预测与实际采用的清单分别位于任务的 `docs/predictions/`、`docs/manifests/`；外测逐条预测为 `docs/predictions/morgan__rf.parquet`，完整精度指标为 `docs/metrics/external_test_metrics.csv`。

输入目录中已移除可选的历史 `provenance.json`、`verification.json`、`checksums.json`。这些旁证文件的移除不涉及结果目录中运行后生成的 manifest、预测和其他记录；应保留生成文件用于核查完成的任务。外测必需的 `audit/external_audit.json` 仍作为输入保留，作用见上文。

#### 三个测试的 Linux 实际验收

2026-11-14 的一轮验收在 Linux、Python 3.9.23 中，经 `yonod.py` 串行运行三份已保存配置，全部正常退出；保存结果的独立验收通过 557 项检查，覆盖记录编号、划分成员、54 行预测（随机划分 36 行、固定名单 12 行、外测 6 行）、指标重算和两种报告。另用 Chrome 打开 HTML，检查表格与本地图像。

| 测试 / 协议 | 执行时的输出根目录 | HTML / Markdown 报告，相对该根目录 | 保存预测，相对该根目录 |
| --- | --- | --- | --- |
| `random_split_test` / `outer_kfold` | `result/example/random_split_test/` | `report/report.html`、`report/report.md` | 三模型各自的 `runs/run_morgan-<model>/predictions.csv` |
| `fixed_split_test` / `manifest_outer_cv` | `result/example/fixed_split_test/` | `report/benchmark_report.html`、`report/benchmark_report.md` | 两折对应的 `docs/predictions/` |
| `external_test` / `frozen_train_external_test` | `result/example/external_test/` | `report/external_test_report.html`、`report/external_test_report.md` | `docs/predictions/morgan__rf.parquet` |

验收后，完整输出已在本地归档到 `backup/three_examples_acceptance_20261114/completed_runs/<测试名>/`，三个源码 `result/example/<测试名>/` 目录均恢复为仅含 `.gitkeep`。该本地归档不包含在源码克隆中；自己重新运行时，仍在上表的原输出位置生成结果。该轮仅提供 Windows 命令说明，未在 Windows 执行。各提交的 Linux/Windows 运行证据见 [YONOD CI](https://github.com/thinktraveller/YONOD/actions/workflows/ci.yml)，以对应平台已完成且成功的保存结果验收任务为准。人工数据流程验收不证明真实化学预测效果。

#### 运行后怎样阅读报告

先看任务信息，再看结果矩阵、详细指标和散点图。每个留出点对照一条预测与人工参考值，越接近对角线越吻合。

- 普通随机报告在每轮汇总 OOF 预测评分，固定名单报告汇总逐折指标，R² 聚合口径不同，不能当作相同评价直接跨路线排名。
- 一套完整轮换不能证明稳定或没有不确定性，跨重复标准差为零也不能提供这种证据。
- MAE 与 RMSE 的单位和目标相同。样例采用 0–1 产率尺度，误差乘以 100 才是百分点；MAE 也会受大误差影响。
- R² 可以为负，也不是预测正确率。人工记录及其生成分数用于检查操作流程，不能代表真实化学预测效果。

### 按需安装可选依赖和资产

- 当前核心依赖支持 RF、XGBoost、LightGBM 和 SVM。AutoGluon 1.1.1 仅保留在独立的历史冻结环境：它要求 scikit-learn ≥1.3,<1.4.1，其 LightGBM extra 要求 LightGBM ≥3.3,<4.4，与当前核心版本冲突。不能在当前核心环境追加安装，也不能绕过依赖解析或降级核心来强行安装。
- AutoGluon 1.4.0 的公开元数据允许新的 sklearn/LightGBM 固定版本，但其 common 包要求 PyArrow <21，Torch 相关 extras 要求 Torch <2.8。它是需要单独迁移、完整解析和运行验收的候选，尚非经过验证的替代版。下文 AutoGluon 参数目录描述历史 1.1.1 适配器，不代表它与当前核心或 1.4.0 兼容。

- 三个首次使用样例已含数据且使用 Morgan。
- 其他数据与权重可从 [YONOD-datasets](https://huggingface.co/datasets/thinktraveller/YONOD-datasets) 和 [YONOD-weights](https://huggingface.co/thinktraveller/YONOD-weights) 获取，下载前查看[数据说明](dataset/README.md)与[权重说明](WEIGHTS/README.md)。

- 以下命令在已激活环境的项目根目录执行，两种系统相同。
- 先安装兼容 Transformers 4.40 的 Hugging Face CLI，再按需下载；数据命令只下载 USPTO 子目录。

```text
python -m pip install "huggingface_hub>=0.34,<1.0"
hf download thinktraveller/YONOD-datasets --repo-type dataset --include "USPTO/**" --local-dir dataset
```

- 按服务要求完成登录或访问授权。
- Git 忽略二进制权重；看见配置、tokenizer或manifest文件不代表权重已齐全。
- 部分原始数据虽然扩展名是 CSV，实际使用制表符分隔，需先按子目录说明转换，不能直接送入逗号CSV向导。

#### 下载预训练描述符权重

**选择 `fisd`、`chemical_vae` 或 `molmetalm` 前，必须先从 [Hugging Face](https://huggingface.co/thinktraveller/YONOD-weights) 下载对应的预训练权重。** 克隆源码只有配置和资产说明，不包含二进制权重；使用 Morgan 的冒烟测试不需要这些权重。

在项目根目录、已激活 `yonod` 的终端中，按需运行对应命令。以下命令在 Linux Bash 与 Windows Conda PowerShell 相同，使用上文安装的 CLI：

```text
hf download thinktraveller/YONOD-weights --include "FISD/**" --local-dir WEIGHTS
hf download thinktraveller/YONOD-weights --include "chemical_vae/**" --local-dir WEIGHTS
hf download thinktraveller/YONOD-weights --include "MolMetaLM-base/**" --local-dir WEIGHTS
```

三条命令分别恢复 `WEIGHTS/FISD/`、`WEIGHTS/chemical_vae/`、`WEIGHTS/MolMetaLM-base/`。若需一次下载全部权重，可用 `hf download thinktraveller/YONOD-weights --local-dir WEIGHTS`；更多选项见 [Hugging Face CLI 文档](https://huggingface.co/docs/huggingface_hub/v0.36.0/guides/cli)。

| 编码 | 所需资产 |
| --- | --- |
| `fisd` | 从 [YONOD-weights](https://huggingface.co/thinktraveller/YONOD-weights) 的 `FISD/` 下载 `qm_9_mse_model.pth`、`qm_9_cos_model.pth`、`qm_9_2in1_model.pth`，放入 `WEIGHTS/FISD/`。 |
| `molmetalm` | [MolMetaLM-base](https://huggingface.co/wudejian789/MolMetaLM-base) 的模型、配置与tokenizer，放入 `WEIGHTS/MolMetaLM-base/`；参见[上游项目](https://github.com/CSUBioGroup/MolMetaLM)。 |
| `chemical_vae` | `WEIGHTS/chemical_vae/<资产>/v5/` 下已验证转换manifest及相邻、哈希匹配的 `encoder_state_dict.pt`。ZINC 与 `zinc_properties` 资产不能互换。 |

- Chemical VAE只使用冻结 PyTorch 编码器的原始 `z_mean_sample`，不训练 VAE，也不加载解码器或性质头。
- FISD 只接受三份独立批准的原始检查点；Chemical VAE 只接受批准的 v5 manifest/state 配对，陌生资产不能靠随包提供的自报哈希授权加载。
- MolMetaLM 按作者方法逐字符加空格，使用固定词表的 BERT fast tokenizer，不扩充词表。含 `%`、`:`、`*` 等不支持字符或产生未知词的行会失败；默认512 token截断仍明确保留，并写入任务日志。带掩码的均值池化保持既有契约。
- 原始 HDF5 或只有manifest不能替代转换后权重；普通运行无需 TensorFlow/Keras，详见[加载器](yonod/descriptors/chemical_vae.py)。

#### 通过镜像下载权重

访问 Hugging Face 较慢时，可按 [HF-Mirror 的使用说明](https://hf-mirror.com/)切换下载端点。先完成上文 CLI 安装并激活 `yonod`，在项目根目录选择对应系统的命令；只需下载所选描述符的一项。环境变量仅作用于当前终端，最后一行恢复官方端点；`HF_HUB_DISABLE_XET=1` 使下载使用普通 HTTP。

Linux Bash：

```bash
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1
hf download thinktraveller/YONOD-weights --include "FISD/**" --local-dir WEIGHTS
hf download thinktraveller/YONOD-weights --include "chemical_vae/**" --local-dir WEIGHTS
hf download thinktraveller/YONOD-weights --include "MolMetaLM-base/**" --local-dir WEIGHTS
unset HF_ENDPOINT HF_HUB_DISABLE_XET
```

Windows Conda PowerShell：

```powershell
$env:HF_ENDPOINT = "https://hf-mirror.com"
$env:HF_HUB_DISABLE_XET = "1"
hf download thinktraveller/YONOD-weights --include "FISD/**" --local-dir WEIGHTS
hf download thinktraveller/YONOD-weights --include "chemical_vae/**" --local-dir WEIGHTS
hf download thinktraveller/YONOD-weights --include "MolMetaLM-base/**" --local-dir WEIGHTS
Remove-Item Env:HF_ENDPOINT, Env:HF_HUB_DISABLE_XET -ErrorAction SilentlyContinue
```

镜像下载仍保存到同一个 `WEIGHTS/` 目录，文件要求与上表一致。若访问受限仓库，先到 Hugging Face 官网完成授权。

## 开发者内容

- [开发与验证一个改动](#开发与验证一个改动)
- [描述符参数与协议支持](#描述符参数与协议支持)
- [新增描述符](#新增描述符)
- [模型参数与实际执行范围](#模型参数与实际执行范围)
- [增加新的模型](#增加新的模型)

### 开发与验证一个改动

- 先阅读 [AGENTS.md](AGENTS.md)，再定位准备支持的运行流程。
- schema-2 从 `yonod.py` 进入配置加载器，随后调用普通 `pipeline/` 或专用 `benchmark/` 适配器。
- 只修改 `yonod/models/` 中的旧适配器，不能完成 schema-2 模型工厂的扩展。

1. 明确新组件的输入类型、输出形状、支持参数、缺失输入处理和协议范围。配置校验与描述符计算、模型执行分开实现。
2. 添加实现和下文列出的注册、分派位置。可选依赖在被选中时再导入；按需记录库版本和资产身份。
3. 保留记录编号、有效掩码、列顺序、划分身份、哈希及原子发布规则。需要学习的转换和模型筛选只使用相应训练数据；输入或有效设置改变时，缓存身份也必须改变。
4. 验证脚本放 `_verify/`，检查无效/空输入、未知参数拒绝、输出对齐、固定种子、训练折隔离、恢复行为和报告路径。先做配置及模拟执行检查，再单独进行会实际训练的冒烟测试。
5. 使用 `pathlib.Path`、UTF-8文件和分系统命令，检查 Linux 与 Windows 路径处理，并记录实际运行的系统。

现有轻量工程冒烟测试需先激活 `yonod`，再从项目根目录执行。下面命令在 Linux Bash 与 Windows Conda PowerShell 相同：

```text
python yonod.py --config example.yaml
```

- 该命令会计算 Morgan 编码，并在人工 fixture 上拟合 RF。
- 测试新描述符或模型前，还应在 `config/` 保存独立 schema-2 YAML、使用唯一任务名，并隔离 `result/<task_name>/` 输出；正式任务仍通过保存YAML后的交互入口启动。
- 冒烟测试通过说明本次示例流程能执行，不等于科学有效性或未测试平台的验收。

- 工程支持与备份放 `backup/`，操作日志/PID放根 `logs/`，维护文档放独立 `project-docs/`。
- 逐项暂存本任务修改的文件，两个发生改动的仓库分别提交；不提交 `_verify/`、`config/` 下的非示例研究配置、`backup/` 或外部参考材料。`config/example/` 下三份预建配置属于跟踪例外。
- 直接 CSV 建模参数会被拒绝，与 `--config` 混用也会拒绝。新 CSV 先通过 `generate_yaml_config.py` 保存 YAML，再在 `yonod.py` 首提示提供路径；`--config` 继续作为工程冒烟接口。
- 贡献、自动检查与发布要求见 [CONTRIBUTING.md](CONTRIBUTING.md)，私密漏洞报告方式见 [SECURITY.md](SECURITY.md)。

### 描述符参数与协议支持

- 普通 schema-2 流程的特征注册表在 [registry.py](yonod/descriptors/registry.py)。
- 每个描述符条目使用下列字段。
- Python 构造器中的参数不一定已经开放给 YAML。

| 字段 | 可以怎样调整 |
| --- | --- |
| `id` | 当前特征候选的独立名称；不同参数候选使用不同 ID，同一任务内不得重复。 |
| `descriptor` | 下表中的10种注册名称之一。 |
| `lifecycle` | 由提供者固定：9种结构编码为 `static_descriptor`，OHE 为 `fold_transform`。普通配置可省略，benchmark 配置须显式声明。 |
| `columns` | 有顺序的输入列。普通分子编码为空或省略时使用所有已声明的 `reactants`、`products`、`others`；也可显式选择子集。OHE 必须提供非空、无重复的列名。目标列不得作为输入。benchmark 必须显式指定属于分子角色的列。 |
| `mode` | 使用下表中该描述符支持的组合方式；多个描述符条目仍是独立候选。 |
| `params` | 普通注册表只开放下表逐项列出的参数；没有开放参数的描述符省略该节或写 `{}`。 |
| `extra_reactants` | 普通 DRFP 流程将这些已声明的结构列追加到反应物一侧，默认为空。DRFP 本身按 `dataset.column_roles.reactants/products` 确定反应成员，不依靠 `columns` 指定反应角色。两种 benchmark 适配器均拒绝非空 `extra_reactants`，其他普通描述符不使用它。 |

以下维数只指结构或类别编码块，不包括之后追加的数值条件。`concat` 的固定单列维数乘以所选列数；OHE 的维数由每个训练折单独学到的类别数决定。

| 描述符 | 普通流程的生命周期 / 模式 | YAML `params` 与输出 |
| --- | --- | --- |
| `morgan` | `static_descriptor` / `concat` | 没有开放参数。二值 Morgan，固定半径2，每列1,024维。 |
| `mfp` | `static_descriptor` / `concat` | `radius`、`fp_size`、`profile`；计数型 Morgan，每列 `fp_size` 维。 |
| `maccs` | `static_descriptor` / `concat` | 没有开放参数。每列166个 MACCS key，去掉 RDKit 的第0位。 |
| `maf` | `static_descriptor` / `sum` | 没有开放参数。将所选分子组分的二值指纹加和为一个128维块，固定半径2。 |
| `rdkit2d` | `static_descriptor` / `concat` | 没有开放参数。当前固定清单每列210维；多组分单元格取第一个组分，计算出的非有限值替换为0。 |
| `drfp` | `static_descriptor` / `reaction` | 没有开放参数。一个2,048维反应块，反应成员来自角色列和可选 `extra_reactants`。 |
| `fisd` | `static_descriptor` / `concat` | 没有开放参数。每列50维；从 `WEIGHTS/FISD/` 读取独立固定摘要批准的原始权重，实现另保留旧路径回退。 |
| `molmetalm` | `static_descriptor` / `concat` | 没有开放参数。固定 float32 LlamaForCausalLM / BERT fast tokenizer 与作者逐字符空格输入，每列768维；从 `WEIGHTS/MolMetaLM-base/` 离线加载获准的本地 safetensors，不支持的分词行失败。 |
| `chemical_vae` | `static_descriptor` / `concat` | 开放下列6个键；已验证冻结编码器每列输出196维。 |
| `ohe` | `fold_transform` / `concat` | `missing_policy`、`dtype`、`handle_unknown`；维数是当前训练折中各列类别数之和。 |

#### 各描述符开放的全部专用参数

| 描述符 / 参数 | 默认值与可用值 | 含义 |
| --- | --- | --- |
| `mfp.radius` | `3`；整数 ≥ 0 | Morgan 邻域半径。 |
| `mfp.fp_size` | `1024`；整数 ≥ 8 | 每个输入列的计数指纹桶数。 |
| `mfp.profile` | `standard`；`standard` 或 `vjethbkm` | `standard` 使用普通输入规范化与有效行掩码；`vjethbkm` 直接编码原始 CSV 组分值，保留全部行，空白或无效组分贡献零块。这会改变输入与缺失处理语义，应另建任务。 |
| `chemical_vae.model_manifest` | 必填非空路径 | 已验证的 v5 转换 manifest，旁边必须有哈希匹配的 `encoder_state_dict.pt`。普通特征服务按 YAML 路径规则解析。 |
| `chemical_vae.backend` | `pytorch`；只接受 `pytorch` | 指定当前支持的冻结编码器后端，不能切换到其他后端。 |
| `chemical_vae.device` | `cpu`；合法 PyTorch 设备字符串 | 执行设备，例如 `cpu` 或 `cuda`；请求 CUDA 时必须实际可用。 |
| `chemical_vae.batch_size` | `64`；正整数 | 编码器推理批大小。 |
| `chemical_vae.input_preprocessing` | `identity`；只接受 `identity` | 保留输入原文，不拆盐、不规范化结构、不截断。 |
| `chemical_vae.output` | `z_mean_sample`；只接受 `z_mean_sample` | 使用未经采样或归一化的确定性编码器均值。当前已验证资产要求输入最大长度120、输出196维，这些是资产约束，不能通过 YAML 任意调整。 |
| `ohe.missing_policy` | `as_category`；`as_category`、`zero_block` 或 `error` | 将缺失标记编码为类别、将对应块强制置零，或拒绝缺失。训练中未出现的缺失类别在检验时仍按未知类别处理。 |
| `ohe.dtype` | `float32`；`float32` 或 `float64` | 输出矩阵的数据类型。 |
| `ohe.handle_unknown` | `ignore`；只接受 `ignore` | 检验记录中训练折未出现的类别对应全零块。 |

- 下列设置存在于 Python 实现，但**尚未开放为普通 YAML 的 `params`**：Morgan/MAF 的 `radius`、`n_bits`；RDKit2D 的 `descriptor_names`；FISD 的 `model_dir`、`device`、`batch_size`；MolMetaLM 的 `weight_path`、`device`、`batch_size`、`max_length`；OHE 的 `missing_token`。
- DRFP 的 Python 参数 `n_bits` 本身也只允许2,048。
- 将这些键写进普通 YAML 不会启用它们；新增支持还需修改校验、构造器路由与特征身份处理。

#### 三种流程的描述符支持有所不同

| 描述符或设置 | 普通随机内测 | 固定名单 benchmark | 冻结外测 |
| --- | --- | --- | --- |
| `morgan`、`maccs`、`fisd`、`molmetalm`、`rdkit2d` | `concat`，遵守普通注册表规则 | 显式列名与生命周期，通常使用按列 `concat` | 显式映射的列、`static_descriptor`，不得丢弃无效行 |
| `maf` | `sum` | 显式 `mode: sum` 保留加和语义 | 显式 `mode: sum`，所有行都须有效 |
| `mfp` | 上述3个参数 | 静态参数传至特征构建器，必须显式指定列 | 静态参数传至特征构建器，不得丢行 |
| `chemical_vae` | 上述6个参数，特征服务校验 manifest 身份 | 使用特征构建与资产路径，须保持有效 manifest 位置和已验证的相邻权重 | 同一冻结编码器约束，不得丢行 |
| `drfp` | 支持按角色构建 `reaction` 与额外反应物 | 当前 schema-2 适配器未接通相应反应角色与模式路由，不能直接使用 | 当前 schema-2 适配器未接通相应反应角色与模式路由，不能直接使用 |
| `ohe` | 按训练折应用上述参数 | 当前 `ReactionComponentOHE` 固定 `missing_policy=zero_block`、`dtype=float64`、`handle_unknown=ignore`，不转发普通 OHE 参数 | 拒绝；适配器只接受 `static_descriptor` |

- benchmark 采用独立校验和执行路径，模式检查只接受 `concat/sum`，也没有像普通特征服务那样传递 DRFP 反应角色。
- 描述符出现在菜单里，不代表它适用于每个协议；请从对应可运行样例开始设置。

### 新增描述符

当前扩展入口是源码，没有自动发现插件的机制。先接入普通流程，再按需要补专用协议支持。

1. 在 `yonod/descriptors/` 下实现遵守 [BaseDescriptor](yonod/descriptors/base.py) 的类，声明 `name`、`output_dim`，实现 `featurize(smiles_list) -> (features, mask)`。

   - 类返回 `(输入行数, output_dim)` 矩阵及原输入行数的一维 Boolean 掩码，顺序与输入一致；失败输入保留对应行，将数值置零并标为 `False`，不能仅凭零向量判为成功。

2. 在 [registry.py](yonod/descriptors/registry.py) 的 `_PROVIDERS` 中添加名称、生命周期、输入类型、模式和参数元数据，并在 `_normalise_params` 中实现默认值、取值校验与未知键拒绝。

   - 新的参数化提供者须同时开放 YAML 路由，不能只改构造器。

3. 在 [feature_builder.py](yonod/universal/feature_builder.py) 的 `_DESCRIPTOR_IMPORT_MAP` 中增加延迟导入，在 `_get_descriptor` 中把已校验参数传到构造器。

   - 当前通用分支无参构造；MFP 和 Chemical VAE 使用了显式参数路由。
   - 若需要公开类 API，再补 [descriptors/__init__.py](yonod/descriptors/__init__.py) 的延迟导出。

4. 在 [pipeline/features.py](yonod/pipeline/features.py) 核对组合与成员对齐。

   - 传给特征服务的计算函数返回 `sum(valid_mask)` 行矩阵和原始行数的 Boolean 掩码，过滤由构建器完成。
   - 经现有 artifact API 保存有序编号、特征名、有效性与哈希。
   - `FeatureSpec.schema_hash` 及特征身份包含规范化设置、解析后的列和角色；依赖外部权重时，还需绑定资产内容哈希，不能让同路径替换后的新权重复用旧编码。

5. 更新 [generate_yaml_config.py](generate_yaml_config.py) 的显式菜单和默认配置构建逻辑，核对 [yonod.py](yonod.py) 的名称选择。

   - 依赖较多或加载较慢的库继续延迟导入，按实际需要补安装说明。

6. 若需固定名单或外测，检查 [benchmark/config.py](yonod/benchmark/config.py)、[benchmark/runner.py](yonod/benchmark/runner.py) 与 [frozen_external.py](yonod/benchmark/frozen_external.py)，逐项接通角色、模式、参数和资产路径。

   - 保留冻结外测的总体、结构重叠与不得丢行等约束，不能假定它复用普通路由。

7. 验证脚本放 `_verify/`。

   - 检查错误声明、未知参数、无效输入、维数、行号/编号对齐、确定性身份，以及改变参数或资产内容后的缓存身份。
   - 后续确需建模验收时，使用独立配置和结果目录经 `yonod.py` 启动，遵循 Linux/Windows 路径与 UTF-8 规则。

- 如果表示方法从当前任务数据学习类别表、缩放或其他状态，应采用 `fold_transform`。
- 目前 OHE 在 `features` 阶段只保存原始输入，再由 [pipeline/training.py](yonod/pipeline/training.py) 在每个外层训练折单独拟合，检验数据只做转换。
- 新的折内转换器还需接入原始输入发布、训练/预测、[普通 HPO](yonod/hpo/ordinary.py)、严格 benchmark 预处理及适用的内层 CV 搜索与状态保存；这些路径当前含有 OHE 专用逻辑。
- 仅在注册表写 `lifecycle: fold_transform` 不会完成接入，也不能先在全量数据上拟合再划分。

- 实现改动如果改变了表示结果，应明确更新持久化身份版本，例如 `FEATURE_SERVICE_SCHEMA_VERSION`，或改用新的候选ID和配置。
- 特征身份不会自动对 Python 源文件求哈希；参数名相同不表示旧缓存仍适用于修改后的算法。

### 模型参数与实际执行范围

- schema-2 YAML 的模型参数以 [model_factory.py](yonod/model_factory.py)和[配置区段路由](yonod/config/contracts.py)为准。
- `yonod/models/` 中保留了旧适配器；其中300棵RF树、自动SVM缩放/PCA等默认值，不会自动成为普通schema-2任务的默认设置。

- 以下参数描述项目的参数白名单。核心版本面向 scikit-learn 1.5.2、XGBoost 2.0.3 和 LightGBM 4.6.0；AutoGluon 1.1.1 描述仅适用于独立历史环境。版本元数据不能代替运行兼容性验收。
- YAML 未声明的估计器参数采用已安装库的默认值。
- 项目在拟合前核对参数名，后端库还会检查参数值及组合。
- 安全YAML可表达数值、字符串、列表和映射，不能构造Python函数、回调对象或估计器类。

| 模型 | 参数区段 | 项目行为 |
| --- | --- | --- |
| `rf` | `estimator`、`fit`、`runtime` | RF构造器和原生拟合参数；`runtime` 必须为空。 |
| `xgb` | `estimator`、`fit`、`runtime` | XGBoost构造器和已接入拟合参数，可选设备策略和训练内早停。 |
| `lightgbm` | `estimator`、`fit`、`runtime` | 安装版本的构造器名加项目明确允许的Booster参数，可选训练内早停。 |
| `svm` | `estimator`、`fit`、`preprocessing`、`runtime` | SVR及可选的训练折内缩放、PCA、抽样；`runtime` 必须为空。 |
| `autogluon` | `predictor`、`fit`、`runtime` | 使用项目明确允许的参数名，还需满足后端版本限制。 |

修改时使用完整层级，例如 `model_params.rf.estimator.n_estimators`。只声明 `models` 中已选择模型的参数。

#### 随机森林（`rf`）

当前支持的scikit-learn安装版本允许在 `model_params.rf.estimator` 下使用以下18个参数。

| 参数 | 含义和取值 |
| --- | --- |
| `n_estimators` | 正整数，树的数量；库默认100。 |
| `criterion` | 分裂准则：`squared_error`、`absolute_error`、`friedman_mse`、`poisson`；默认 `squared_error`。Poisson有目标值限制。 |
| `max_depth` | 正整数深度上限，或 `null` 表示不显式限制；默认 `null`。 |
| `min_samples_split` | 分裂所需最少样本：至少2的整数或 `(0, 1]` 比例；默认2。 |
| `min_samples_leaf` | 叶节点最少样本：正整数或 `(0, 1)` 比例；默认1。 |
| `min_weight_fraction_leaf` | 叶节点最少加权样本比例，0到0.5；默认0。 |
| `max_features` | 每次分裂考虑的特征数：正整数、比例、`sqrt`、`log2` 或 `null`；默认1.0。 |
| `max_leaf_nodes` | 叶节点数上限，至少2的整数或 `null`；默认 `null`。 |
| `min_impurity_decrease` | 非负的最小分裂改善量；默认0。 |
| `bootstrap` | 每棵树是否有放回抽取训练记录；布尔值，默认 `true`。 |
| `max_samples` | Bootstrap记录数、比例或 `null`，仅在 `bootstrap: true` 时有意义；默认 `null`。 |
| `oob_score` | 是否计算bootstrap袋外分数，默认 `false`；不替代外层留出检验。 |
| `n_jobs` | CPU工作线程，`null`、正数与 `-1` 按库含义处理；首次使用样例配置省略此项。 |
| `random_state` | 整数种子或 `null`；库默认 `null`，与 `evaluation.seed` 分别控制。 |
| `verbose` | 非负日志等级；默认0。 |
| `warm_start` | 对同一估计器再次拟合时是否复用旧树；默认 `false`。YONOD每折新建估计器。 |
| `ccp_alpha` | 非负剪枝惩罚；默认0。 |
| `monotonic_cst` | 每个特征的 `-1`、`0`、`1` 单调约束或 `null`，默认 `null`；长度应对应最终特征矩阵。 |

- `rf.fit` 只接受 `sample_weight`，写为 `sample_weight: {column: weight_column}`，程序仅从当前训练行读取该列。
- 不要在YAML中放入权重数组。
- `rf.runtime` 没有可用的非空设置。
- [scikit-learn RF文档](https://scikit-learn.org/1.5/modules/generated/sklearn.ensemble.RandomForestRegressor.html)说明估计器自身的取值限制。

#### XGBoost（`xgb`）

- `model_params.xgb.estimator` 接受下列39个参数名。
- 省略参数采用XGBoost 2.0.3默认值；许多Python构造器字段默认为 `None`，由Booster决定最终设置。

| 参数 | 含义与限制 |
| --- | --- |
| `n_estimators`、`num_parallel_tree` | 提升轮数，以及每轮建立的树数。 |
| `max_depth`、`max_leaves`、`max_bin`、`grow_policy` | 树深、叶节点上限、直方图分箱数、增长策略（`depthwise` 或 `lossguide`）。 |
| `learning_rate`、`gamma`、`min_child_weight`、`max_delta_step` | 学习率、分裂损失改善阈值、子节点Hessian阈值、叶更新上限。 |
| `subsample`、`sampling_method` | 训练行抽样比例与策略（`uniform` 或GPU支持的 `gradient_based`）。 |
| `colsample_bytree`、`colsample_bylevel`、`colsample_bynode` | 每棵树、每层、每个节点的特征抽样比例。 |
| `reg_alpha`、`reg_lambda` | 非负L1、L2正则化。 |
| `objective`、`base_score` | 损失名称与初始预测偏置；回归器默认 `reg:squarederror`。本页流程应保持标量回归目标。 |
| `booster`、`tree_method` | Booster（`gbtree`、`gblinear`、`dart`）与树算法（`auto`、`exact`、`approx`、`hist`）；组合限制由后端决定。 |
| `device`、`n_jobs` | CPU/GPU设备与CPU工作线程，设备可用 `cpu`、`cuda` 或支持的CUDA编号。 |
| `random_state`、`verbosity`、`validate_parameters` | 种子、日志等级、原生参数检查。 |
| `missing` | 被视为缺失的输入值，构造器默认NaN；不绕过项目的特征有效性检查。 |
| `monotone_constraints`、`interaction_constraints` | 特征趋势约束与允许的交互组，对应最终特征矩阵。 |
| `importance_type` | 特征重要性的计算方式，不改变划分设计。 |
| `enable_categorical`、`feature_types`、`max_cat_to_onehot`、`max_cat_threshold` | 原生类别处理与分裂阈值；在这里填写源列名不能替代项目的OHE流程。 |
| `multi_strategy`、`scale_pos_weight` | 多输出策略与正类权重；参数名被原生接口识别，不代表本页标量回归流程新增其他任务类型。 |
| `eval_metric`、`early_stopping_rounds`、`callbacks` | 评价指标及早停/回调控制。Python回调对象不能由可移植的安全YAML构造；早停使用下述已接入路径。 |

- `xgb.fit` 允许 `sample_weight`、`eval_metric`、`early_stopping_rounds`、`verbose`。
- 权重按上面的列映射指定。
- XGBoost 2.x推荐将指标和早停预算放在构造器，不能在两处重复声明同一项。
- `fit` 拒绝 `eval_set`、`sample_weight_eval_set`、`base_margin`、`base_margin_eval_set`、`feature_weights`、`xgb_model`、`callbacks`，避免外部数组或对象进入折内拟合。

- `xgb.runtime.device_policy` 可设 `auto`、`cpu`、`cuda`。
- 若使用 `estimator.device`，需省略整个 `runtime` 区段；当前校验器拒绝 `estimator.device` 与任何显式 `runtime` 区段同时出现。
- `runtime.early_stopping` 有三个键：正整数 `rounds`、严格位于0和1之间的 `validation_fraction`、整数 `seed`（默认42）。
- 程序只从当前外层训练折再划内层验证记录。
- 需要早停时使用这个路径，不要仅填没有验证数据的裸早停参数。
- 原生含义见 [XGBoost 2.0参数文档](https://xgboost.readthedocs.io/en/release_2.0.0/parameter.html)。

#### LightGBM（`lightgbm`）

- `model_params.lightgbm.estimator` 接受以下71个参数名，其中部分是同一设置的别名。
- 一个设置选择一种写法，重复写别名可能触发后端的优先级规则。
- 项目名单不包含上游全部参数。

| 参数 | 含义与限制 |
| --- | --- |
| `boosting_type`、`objective`、`n_estimators`、`learning_rate` | 提升方式、回归损失、轮数预算、学习率；构造器默认分别为 `gbdt`、推断回归、100、0.1。 |
| `num_leaves`、`max_depth` | 叶节点与深度上限；默认31和-1（无正数深度上限）。 |
| `min_child_samples`、`min_child_weight`、`min_sum_hessian_in_leaf` | 叶节点最少记录数与Hessian下限，后两者是同一阈值的别名。 |
| `min_split_gain`、`min_gain_to_split` | 最小分裂改善量，互为别名。 |
| `subsample`、`bagging_fraction`、`subsample_freq`、`bagging_freq`、`bagging_seed` | 行抽样比例、执行频率和种子；前两者、中间两者各互为别名。抽样需要启用相应频率。 |
| `colsample_bytree`、`feature_fraction`、`feature_fraction_bynode`、`feature_fraction_seed` | 每棵树/每节点的特征抽样与种子；前两者互为别名。 |
| `reg_alpha`、`lambda_l1`、`reg_lambda`、`lambda_l2` | L1/L2惩罚及其别名。 |
| `subsample_for_bin`、`bin_construct_sample_cnt`、`max_bin`、`min_data_in_bin` | 构建分箱的抽样预算及其别名、分箱上限、每箱最少数据。 |
| `max_delta_step`、`path_smooth`、`linear_tree`、`extra_trees` | 叶更新上限、叶平滑、线性叶模型、随机阈值；需满足后端及特征限制。 |
| `cat_l2`、`cat_smooth`、`max_cat_to_onehot`、`max_cat_threshold`、`min_data_per_group` | 原生类别特征正则、平滑和分裂控制，作用于编码后的模型特征。 |
| `monotone_constraints`、`monotone_constraints_method`、`monotone_penalty` | 各特征单调性、约束算法与深度惩罚。 |
| `feature_contri`、`interaction_constraints` | 每个特征的增益乘数和允许的特征交互。 |
| `drop_rate`、`max_drop`、`skip_drop`、`xgboost_dart_mode`、`uniform_drop`、`dart_seed` | DART丢弃控制，仅在相应提升方式下有意义。 |
| `data_sample_strategy`、`top_rate`、`other_rate` | `bagging`/`goss`数据采样与GOSS保留比例。 |
| `deterministic`、`force_col_wise`、`force_row_wise` | CPU确定性、直方图按列/按行布局；不要同时强制两种布局。 |
| `device_type`、`n_jobs`、`num_threads` | 设备和CPU资源，后两者为相关的线程控制；GPU需要兼容构建。 |
| `random_state`、`seed`、`verbosity`、`importance_type` | 种子、日志与重要性方法（`split`/`gain`）。 |
| `boost_from_average`、`first_metric_only`、`metric`、`early_stopping_round`、`early_stopping_rounds` | 初始分数、指标选择和早停别名；普通评价的早停使用已接入内层验证路径。 |
| `class_weight`、`is_unbalance`、`scale_pos_weight`、`eval_at` | 名单中保留的分类/排序参数，不会将YONOD标量回归流程转换为分类或排序。 |
| `callbacks` | 项目名单中存在该名字，但安全YAML不能构造Python回调对象；这里不是已接入的早停路径。 |

- `lightgbm.fit` 允许 `sample_weight`、`eval_names`、`eval_metric`、`feature_name`、`categorical_feature`。
- 权重使用列映射；特征名和类别索引须对应模型矩阵，不能按不相关的源表列次序填写。
- `fit` 拒绝 `init_score`、`eval_set`、`eval_sample_weight`、`eval_init_score`、`init_model`、`callbacks`。
- `lightgbm.runtime` 只允许 `early_stopping`，内部同样使用 `rounds`、`validation_fraction`、`seed`，程序自己提供训练内验证集和LightGBM回调。
- 原生细节见 [LightGBM 4.6回归器API](https://lightgbm.readthedocs.io/en/v4.6.0/pythonapi/lightgbm.LGBMRegressor.html)和[参数文档](https://lightgbm.readthedocs.io/en/v4.6.0/Parameters.html)。

#### 支持向量回归（`svm`）

`model_params.svm.estimator` 接受以下11个参数，构造的是 `sklearn.svm.SVR`，不执行分类。

| 参数 | 含义与库默认值 |
| --- | --- |
| `kernel` | `linear`、`poly`、`rbf`、`sigmoid`、`precomputed`，默认 `rbf`；预计算核另需矩阵契约，普通描述符流程不直接提供该矩阵。 |
| `degree` | 非负多项式次数，默认3，仅影响 `poly`。 |
| `gamma` | 核系数，`scale`、`auto` 或非负数，默认 `scale`。 |
| `coef0` | 多项式/sigmoid核独立项，默认0。 |
| `C` | 正的误差惩罚参数，默认1；C越大，通常对误差的惩罚越强、正则化越弱。 |
| `epsilon` | 非负误差管宽度，默认0.1，与目标同单位。 |
| `tol` | 正的求解器停止容差，默认0.001。 |
| `shrinking` | 是否使用收缩启发式，默认 `true`。 |
| `cache_size` | 正的核缓存大小，单位MB，默认200。 |
| `verbose` | 求解日志，默认 `false`。 |
| `max_iter` | 迭代上限，-1表示不限，默认-1。 |

- `svm.fit` 只允许按源列映射的 `sample_weight`，工厂会在内部路由到最终Pipeline步骤。
- `svm.runtime` 必须为空。
- 普通schema-2工厂没有旧SVM适配器中隐含的缩放、PCA或训练行抽样，只有显式声明后才执行。

| `svm.preprocessing` 设置 | 全部字段与行为 |
| --- | --- |
| `scaler` | 必填布尔值 `enabled`；`params` 允许StandardScaler的 `copy`（复制输入，默认 `true`）、`with_mean`（列中心化，默认 `true`）、`with_std`（缩放为单位方差，默认 `true`），只在外层训练行拟合。 |
| `pca` | 必填布尔值 `enabled`；可选非负整数 `feature_threshold`，仅在输入维数严格大于该阈值时执行PCA，`null` 表示无阈值。 |
| `pca.params` | 下面逐项列出9个PCA构造参数；整数维数会按训练行数和特征数截断。 |
| `subsample` | `n_samples` 为正整数或 `null`；`random_state` 为非负整数，抽样时默认42。只减少SVR拟合行数，缩放/PCA仍在全部外层训练行拟合；实际抽样种子加入折序号偏移。 |

| `pca.params` 内的参数 | 含义与库默认值 |
| --- | --- |
| `n_components` | 输出维数、解释方差比例、`mle` 或 `null`；默认 `null`，须满足求解器要求。 |
| `copy` | 分解前是否复制训练输入，默认 `true`。 |
| `whiten` | 是否将主成分缩放为单位方差，默认 `false`。 |
| `svd_solver` | 分解方法：`auto`、`full`、`arpack`、`randomized`，默认 `auto`。 |
| `tol` | ARPACK收敛容差，非负，默认0。 |
| `iterated_power` | 随机SVD的幂迭代次数，整数或 `auto`，默认 `auto`。 |
| `n_oversamples` | 随机SVD额外随机向量数，正整数，默认10。 |
| `power_iteration_normalizer` | 幂迭代的归一化方法：`auto`、`QR`、`LU`、`none`，默认 `auto`。 |
| `random_state` | randomized/ARPACK求解器的种子，整数或 `null`，默认 `null`。 |

- 省略预处理构造参数时采用scikit-learn默认值：缩放默认中心化并标准化，PCA默认不白化。
- 原生细节见 [SVR](https://scikit-learn.org/1.5/modules/generated/sklearn.svm.SVR.html)、[StandardScaler](https://scikit-learn.org/1.5/modules/generated/sklearn.preprocessing.StandardScaler.html)、[PCA](https://scikit-learn.org/1.5/modules/generated/sklearn.decomposition.PCA.html)文档。

#### AutoGluon（`autogluon`）

- 以下参数名在普通schema-2流程的允许范围内；其中需要代码适配或会被固定版本拒绝的项已标明。
- 工厂将 `predictor.label` 绑定到 `dataset.column_roles.label`，需要明确任务类型时使用 `problem_type: regression`。
- 训练DataFrame只包含编码后的 `f0`、`f1` 等特征与目标，不自动保留任意原始CSV列。

| `autogluon.predictor` 参数 | 含义与当前限制 |
| --- | --- |
| `label`、`problem_type`、`eval_metric` | 目标名（须与数据集一致）、任务类型、模型选择指标。分类/分位数设置不构成本页标量回归流程已验收的新任务。 |
| `path`、`verbosity`、`log_to_file` | 模型文件路径、日志等级、文件日志；显式 `path` 优先于runtime路径。 |
| `sample_weight`、`weight_evaluation`、`groups` | 后端权重列名、加权评分、装袋分组列。原始权重/分组列不自动传入，因此不能视为已实现的任意CSV列适配。 |
| `quantile_levels` | 分位数回归所需列表；普通标量评分没有已验证的分位数输出契约。 |
| `learner_kwargs`、`learner_type` | 后端学习器定制；Python学习器类不能由安全YAML构造。 |
| `positive_class` | 虽在项目名称名单中，但被当前AutoGluon 1.1.1构造器拒绝，不应使用。 |

| `autogluon.fit` 参数 | 含义与当前限制 |
| --- | --- |
| `time_limit`、`presets`、`hyperparameters` | 拟合时间预算、预设名/列表、模型家族设置；不改变外层成员。 |
| `feature_metadata`、`num_cpus`、`num_gpus` | 特征类型与后端资源，按本机选择。 |
| `fit_weighted_ensemble`、`fit_full_last_level_weighted_ensemble` | 是否及如何拟合后端加权集成。 |
| `holdout_frac`、`num_bag_folds`、`num_bag_sets`、`num_stack_levels` | 内部留出比例、装袋折数/轮数、堆叠层数，只使用提供给后端的外层训练总体。 |
| `auto_stack`、`dynamic_stacking`、`use_bag_holdout` | 自动堆叠、堆叠过拟合检查、装袋留出选择。 |
| `included_model_types`、`excluded_model_types` | 允许或排除的后端模型家族。 |
| `refit_full`、`set_best_to_refit_full`、`keep_only_best`、`save_space` | 后端重新拟合与文件保留策略，需保留后续工作要用的模型文件。 |
| `ag_args`、`ag_args_fit`、`ag_args_ensemble` | 后端模型身份、单模型拟合、集成的嵌套设置。内部键由AutoGluon 1.1.1定义，不是YONOD自身的schema。 |
| `verbosity`、`calibrate` | 拟合日志和适用情况下的校准（`true`、`false`、`auto`）。 |
| `fit_extra`、`fit_strategy`、`delay_bag_sets`、`raise_on_no_models_fitted` | 虽在项目名称名单中，但不是当前AutoGluon 1.1.1 `fit` 参数校验器允许的项，不应作为YAML拟合参数。`fit_extra()` 是后端的另一个方法。 |

- `train_data` 由流程绑定，不能从YAML提供；`tuning_data`、`unlabeled_data` 也不允许。
- 上游存在但不在项目名单内的 `infer_limit`、`log_file_path` 等不会自动开放。
- `autogluon.runtime.save_path` 接受字符串或 `null`，默认模型位置是本折的 `autogluon/` 目录；`runtime.cleanup` 为布尔值，普通流程默认 `false`。
- 建议省略 `predictor.path` 和 `runtime.save_path`，保留系统的任务/逐折隔离目录：普通执行器不会为显式路径自动追加折后缀，共用路径可能混入或覆盖旧模型，也可能导致拟合失败。
- 自定义路径仍须位于本任务指定输出目录内，并隔离每折、每个组合。
- 这里没有一个能固定全部AutoGluon子模型的顶层 `random_state`。

- 固定版本的上游[构造器文档](https://github.com/autogluon/autogluon/blob/v1.1.1/tabular/src/autogluon/tabular/predictor/predictor.py#L222)和[拟合文档](https://github.com/autogluon/autogluon/blob/v1.1.1/tabular/src/autogluon/tabular/predictor/predictor.py#L392)也定义了 `hyperparameters`、`ag_args`、`ag_args_fit`、`ag_args_ensemble` 内部的嵌套范围。
- YONOD开放这些字典，但不逐项定义或校验每个后端模型家族的子参数，填写前需核对1.1.1源码/API。

#### 参数支持范围随检验协议变化

| 能力 | 普通 `all/train` | 固定成员manifest benchmark | 冻结外测 |
| --- | --- | --- | --- |
| 构造器参数 | 上述共同工厂目录 | 同一目录，但有协议种子等额外限制 | 支持的非AutoGluon模型使用共同目录 |
| RF/XGB/LightGBM/SVM非空 `fit` | 使用上述已接入原生路径 | 拒绝，此处尚未实现将这些数据选项逐折转换为拟合输入 | 拒绝 |
| XGB/LightGBM `runtime.early_stopping` | 外层训练折内再划验证集 | 拒绝 | 拒绝 |
| AutoGluon | 上述普通流程名单 | 仅 `fit.time_limit`、`fit.presets`、`fit.num_cpus`，以及已支持runtime路径/清理；省略 `predictor.label`，内部固定为 `yield` | 不支持 |
| SVM抽样 | 只减少最终SVR训练行 | 不执行 `preprocessing.subsample.n_samples` 裁样；显式缩放/PCA仍随工厂Pipeline执行 | 不执行 `preprocessing.subsample.n_samples` 裁样；显式缩放/PCA仍随工厂Pipeline执行 |

- 常规 `manifest_outer_cv` benchmark 不会用 `evaluation.seed` 替换 `estimator.random_state`。
- 显式复现设置 `benchmark.reproduction_protocol.name: vjethbkm_rf_5x5` 和单独的 `paper_exact_5x5` 评价路径会将RF模型种子绑定到逐折manifest，并拒绝冲突。
- 普通工厂接受某个参数，不代表所有benchmark适配器都会执行它。

- manifest benchmark的AutoGluon使用专用适配器，省略设置时采用300秒拟合预算、`medium_quality`、单CPU和启用清理。
- 这些是该适配器的兼容默认值；普通schema-2 AutoGluon通过共同predictor工厂采用已安装库默认。
- 使用专用路径时可明确声明资源。

### 增加新的模型

1. 实现或包装具有 `fit(X_train, y_train, **fit_options)`、`predict(X_valid)`、`get_params()` 的估计器。

   - 每条检验记录必须返回一个有限数值，拟合对象应可按流程模型格式保存和恢复；依赖数据的预处理只在训练折学习。

2. 在 [config/contracts.py](yonod/config/contracts.py) 的 `MODEL_PARAMETER_ROUTES` 中加入模型ID和参数区段。

   - 需要别名时，再更新 [config/loader.py](yonod/config/loader.py) 的 `_MODEL_ALIASES`。

3. 更新 [model_factory.py](yonod/model_factory.py) 中的 `_library_catalog`、`resolve_model_config`、`_validate_runtime`、`construct_estimator`（或明确的predictor专用路径）、`software_versions`。

   - 可选依赖延迟导入，在耗时计算前拒绝未知键和外部数据/回调对象。

4. 核对 [pipeline/training.py](yonod/pipeline/training.py) 的 `_default_fold_runner`、模型序列化和[保存模型预测](yonod/pipeline/prediction.py)。

   - 非sklearn模型需显式实现拟合选项路由和加载，不应假定通用runner已足够。
   - 需要训练诊断时，再验证训练/留出预测契约并加入支持。

5. 在 [generate_yaml_config.py](generate_yaml_config.py) 的模型菜单及 `_canonical_wizard_models` 中增加模型选项。`yonod.py` 中的 `_MODEL_NAMES` 仅属于保留的旧参数定义，不负责注册 schema-2 模型。

   - 旧 `_make_model` 和 `yonod/models/` 保留为实现历史；公开 CLI 已拒绝直接 CSV 建模。单独修改这些适配器不能注册 schema-2 模型。

6. HPO需另行更新 [hpo/contracts.py](yonod/hpo/contracts.py)、[hpo/spaces.py](yonod/hpo/spaces.py)、[hpo/search.py](yonod/hpo/search.py)及必要的最终模型流程。

   - 定义有边界、有版本的搜索空间和训练内评分；支持构造器不等于自动支持HPO。

7. 明确选择要兼容的专用协议，核对 [benchmark/config.py](yonod/benchmark/config.py)、[benchmark/executor.py](yonod/benchmark/executor.py)、[frozen_external.py](yonod/benchmark/frozen_external.py)。

   - 保留各协议的数据边界、模型限制和恢复证据；未实现适配时保持明确拒绝。

8. 验证脚本放 `_verify/`，检查未知参数拒绝、维度/非有限值、训练内预处理、外层标签隔离、模型保存/预测恢复、Linux/Windows路径。

   - 工程冒烟测试与真实研究任务分开；更新依赖、双语文档和独立文档仓日志，再按两仓实际更改分别提交。

最近一次对照本仓库核对：2026-11-14。

## 项目结构

三份可执行配置已保存在 `config/example/` 下；对应的 `random_split_test/`、`fixed_split_test/`、`external_test/` 结果目录初始仅含 `.gitkeep`，没有生成结果。其他本地任务目录会在准备或运行任务时建立。

```text
YONOD/
├── README.md                英文使用指南
├── README.zh-CN.md          中文使用指南
├── AGENTS.md                开发与目录约定
├── LICENSE                  项目许可证
├── yonod.py                 读取并执行已保存的 YAML
├── generate_yaml_config.py  准备数据、生成和校验 YAML，不建模
├── example.yaml             完整 schema-2 格式与工程冒烟测试
├── requirements.txt         依赖清单；PyTorch 先单独安装
├── example/                 冒烟测试输入与审计记录
│   ├── random_split_test/   随机划分输入 CSV
│   ├── fixed_split_test/    固定名单数据与划分清单
│   └── external_test/       训练/外测表及来源审计
├── yonod/                   Python 源码；下列为主要组件
│   ├── model_factory.py     schema-2 模型校验与构造
│   ├── config/              YAML 读取、校验与路径解析
│   ├── descriptors/         分子结构与类别编码
│   ├── features/            数据准备与数值条件处理
│   ├── models/              旧模型适配器
│   ├── pipeline/            普通编码、训练、预测与报告流程
│   ├── benchmark/           固定名单、外测等专用流程
│   ├── splits/              分组与成员清单
│   ├── artifacts/           特征包保存与完整性检查
│   ├── hpo/                 训练数据内的模型参数搜索
│   └── universal/           CSV 读取与特征拼装等公共组件
├── dataset/                 数据及来源说明
├── WEIGHTS/                 可选预训练资产
├── config/                  推荐统一存放运行配置
│   ├── example/             已准备好的冒烟测试配置
│   │   ├── random_split_test/
│   │   │   └── run.yaml
│   │   ├── fixed_split_test/
│   │   │   └── run.yaml
│   │   └── external_test/
│   │       └── run.yaml
│   └── <task_name>.yaml     本地研究任务配置
├── result/                  推荐统一存放运行结果
│   ├── example/             测试输出；初始仅有占位文件
│   │   ├── random_split_test/
│   │   ├── fixed_split_test/
│   │   └── external_test/
│   └── <task_name>/         同一个研究任务的全部输出
├── logs/                    操作日志与 PID 文件
├── _verify/                 本地验证脚本
├── backup/                  备份与工程支持文件
├── reference-projects/      外部参考材料
└── project-docs/            独立文档仓库
    ├── goal.md              项目目标
    ├── project-plan.md      开发计划
    ├── buildlog.md          变更与验证记录
    ├── teach.md             学习笔记
    └── docs/                其他维护文档
```

- 日常使用先看根目录的两个入口脚本、[example.yaml](example.yaml) 和 [example/](example/)。
- [数据说明](dataset/README.md)与[权重说明](WEIGHTS/README.md)分别解释输入来源和可选资产。

- `config/` 只跟踪 `example/` 下三份预建配置；`result/` 只跟踪三个示例目录的空 `.gitkeep`，生成的示例结果与工程冒烟结果都忽略。其他本地配置和结果，以及根目录 `logs/`、`_verify/`、`backup/`、`reference-projects/`、`project-docs/`，也不由主仓 Git 跟踪。
- `project-docs/` 在独立的 [YONOD-docs](https://github.com/thinktraveller/YONOD-docs) 仓库中维护，访问需要权限，克隆主仓不会自动获取它。
- 运行首次使用样例不需要该文档仓库。

一次普通建模任务的主要结果结构如下：

```text
result/<task_name>/
├── feature/                         编码及特征记录
├── runs/
│   └── run_<feature_id>-<model>/     一个编码与模型组合
│       ├── predictions.csv          逐条留出预测
│       ├── fold_metrics.csv         各折指标
│       └── run_manifest.yaml        配置与运行记录
├── pictures/                        报告图像
└── report/                          HTML 与 Markdown 报告
```

- 固定名单等 benchmark 将成员名单、预测、逐折记录、指标和恢复状态分别放在任务目录内的 `docs/manifests/`、`docs/predictions/`、`docs/folds/`、`docs/metrics/`、`docs/state/`。
- 冻结外测也使用 `docs/predictions/`、`docs/metrics/` 和 `report/`，但不保存普通流程那样完整的特征与最终模型包。
- 具体文件取决于所选协议。

- 示例的新结果统一位于项目根目录 `result/example/<task_name>/`，每个任务使用独立子目录。
- 推荐自己的配置放在根 `config/<task_name>.yaml`，结果放在 `result/<task_name>/`；研究配置及所有生成结果均由 Git 忽略。
- 参与修改前请阅读 [AGENTS.md](AGENTS.md)，验证脚本放 `_verify/`，维护文档放独立文档仓库。

## 致谢

感谢以下项目的作者与贡献者提供 YONOD 使用的分子表征方法和预训练资源：

- [DRFP](https://github.com/reymond-group/drfp)：感谢 Daniel Probst、Philippe Schwaller、Jean-Louis Reymond 及贡献者提供差分反应指纹方法与实现。
- [RDKit](https://www.rdkit.org/)：感谢开发者与贡献者提供分子解析、指纹与描述符计算等化学信息学工具；YONOD 的 MACCS 指纹也采用其实现。
- [MolMetaLM](https://github.com/CSUBioGroup/MolMetaLM)：感谢作者提供分子语言模型，以及 [MolMetaLM-base 预训练权重与 tokenizer](https://huggingface.co/wudejian789/MolMetaLM-base)。

## License

本项目代码和文档采用 [CC BY-NC 4.0](LICENSE)，要求署名并限于非商业使用。第三方数据、预训练权重与参考代码遵循各自许可证和引用要求。
