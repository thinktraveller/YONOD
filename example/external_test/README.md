# 训练数据与外测数据分开：固定模型设置后检验另一批记录

本包展示**训练CSV、外测CSV、列名对应关系和来源核查文件怎样配合**。这里有12条训练记录、6条外测记录，所有产率都为人工编造；结构只是形式上的酰胺示例，没有反应条件或实验测量依据。**本包演示外测操作，不构成独立实验来源的科学验证。**

本包已有完整运行与逐条验收记录。当前YAML不固定线程数；实际测试配置版本与平台见 `verification.json`。

## 文件如何分工

| 文件 | 用途 |
| --- | --- |
| [dataset/train.csv](dataset/train.csv) | 12条建立模型用的记录：反应编号、人工参考产率、胺、酸和产物结构。 |
| [dataset/external.csv](dataset/external.csv) | 6条留待外测的记录；额外保留来源编号和胺分组，并故意采用不同列名展示对应方法。 |
| [audit/external_audit.json](audit/external_audit.json) | 外测来源与结构重叠核查清单（audit manifest），明确人工教学身份及结构交集。 |
| [run.yaml](run.yaml) | 直接在本目录运行的任务配置；路径相对本文件所在目录。 |
| [provenance.json](provenance.json) | 训练/外测文件校验码、实际结构交集与本轮运行记录。 |

## 两份表怎样对应

训练表的`reaction_id`和外测表的`external_id`各自唯一且互不重叠。两份表可以列名不同，YAML明确告诉程序哪一列对应哪一列。

| 训练CSV列 | 外测CSV列 | 用途 |
| --- | --- | --- |
| `amine_smiles` | `external_amine` | 胺结构输入；由`column_map`对应。 |
| `acid_smiles` | `external_acid` | 酸结构输入；由`column_map`对应。 |
| `product_smiles` | `external_product` | 产物结构输入；由`column_map`对应。 |
| `yield` | `measured_yield` | 参考产率，用于建立模型或最终评分；分别由训练`label`和外测`label_col`声明，不放入`column_map`。 |
| `reaction_id` | `external_id` | 各自的唯一记录编号，不作为结构输入。 |
| 无 | `source_record_id` | 外测来源记录编号，如`teaching-external-v1:record-01`。 |
| 无 | `amine_group` | 同一胺的记录属于同组，共3组；保留给后续按组重采样等分析。 |

`measured_yield`只是用来演示真实项目常见列名，**本例仍是人工参考值，不是实测产率**。真实外测需要已知参考产率才能评分；仅有新结构、没有参考值的预测任务，是另一种工作流程。

## 程序按什么顺序工作

1. 先固定数据、结构编码和模型设置，核查外测文件与来源清单的SHA-256（文件校验码）、编号和结构列对应关系。
2. 每个编码与模型组合只使用全部12条训练记录建立一次模型。本例选择结构数值编码（Morgan指纹）和随机森林回归（RF），模型设置已经写死，关闭自动筛选设置（HPO）。
3. 用完成的模型预测外测6条记录；外测参考产率在得到预测后才读取并用于评分。
4. 输出外测逐条预测、整体误差和外测报告。这里没有两折轮换、折外预测或内测排名。

“固定”是先固定模型设置，再使用全部训练数据建立模型，不是读取一个现成模型文件。当前外测路线也不保存可供以后直接复用的最终模型包。

外测数据不能拿来筛选模型设置、决定何时停止训练，或反复修改方案后继续称为从未使用的最终外测。正式研究应先在训练数据内部做好这些选择。

## 本例如何核查结构重叠和来源

我们已将胺、酸和产物分别规范化为可比较的结构文字（RDKit规范化SMILES），实际计算下列交集，没有照抄零值：

| 结构角色 | 训练中的不同结构数 | 外测中的不同结构数 | 完全相同的结构交集 |
| --- | --- | --- | --- |
| 胺 | 4 | 3 | 0 |
| 酸 | 3 | 3 | 0 |
| 产物 | 12 | 6 | 0 |

当前这条外测执行路径要求审计文件中的胺、酸、产物完全结构交集均为零，还要求外测至少2个非空组。本例有3个胺组；组数少，不能把它们当成充分的独立实验重复。

程序检查审计声明及校验码，**不会代替研究者重新计算结构交集或核实文献来源**。另存一个CSV、改记录编号、文件校验码不同、或结构无完全重复，都不能单独证明数据独立。真实研究还须追溯论文/数据库/实验批次、重复记录、来源关系和测量口径，并如实报告允许覆盖的泛化范围。

`candidate_source_sha256`在本例指外测CSV，因为没有更上游的真实原始来源表。`physical_dataset_id`为明确标示人工来源的`teaching-external-v1`，外测`source_record_id`都以它加冒号开头。换成自己的数据时，需要完整重做来源审计，再更新CSV与审计JSON的两个SHA-256字段；不要仅修改零重叠声明。

## 修改自己的数据时应检查什么

1. 训练和外测的预测目标应相同、单位一致；本例`0.22`表示22%，没有程序自动完成百分数换算的设置。
2. 同一输入角色两表一一对应；先核对结构和实际化学角色，再填写`column_map`，不能只凭列顺序猜测。
3. 训练编号与外测编号不能相交；外测编号和来源编号分别唯一，外测分组至少两个且都非空。
4. 外测CSV、审计JSON及YAML中的校验码同步更新；保留真实来源链、结构重叠和排除依据。
5. 使用新的任务名，结果隔离到新`result/<任务名>/`。该协议不接受CV折数字段或分组划分清单，也不支持OHE类别编码、数值条件输入、自动设置筛选、AutoGluon或外测早停；本例仅展示支持的静态分子编码与固定RF。

这里同时使用胺、酸和产物结构，所以真正使用时须确认预测时确实已知目标产物。它不是仅输入反应物就自动推断产物的功能。

## 在本文件夹运行

从YONOD根目录进入 `example/external_test/`，检查同目录的 [run.yaml](run.yaml)，无需复制或移动设置。输入和新结果保留在本包中；当前YAML未指定线程数，采用模型库默认值，可按本机资源另设。每次只运行一个任务。

### Linux：Bash

先初始化本机Conda并激活环境，从YONOD根目录执行以下后台启动块：

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate yonod
export PYTHONUTF8=1
cd example/external_test
mkdir -p ../../logs
printf '%s\n' 'run.yaml' > ../../logs/example_external_test_20261113.stdin.txt
nohup python -u ../../yonod.py \
  < ../../logs/example_external_test_20261113.stdin.txt \
  > ../../logs/example_external_test_20261113.log 2>&1 &
echo $! > ../../logs/example_external_test_20261113.pid
```

若 `conda` 尚未加入shell路径，请先用本机安装目录的 `etc/profile.d/conda.sh` 初始化。日志和PID位于仓库根 `logs/`；[总操作说明](../README.md#linuxbash-后台运行)包含按需查询状态的命令。

### Windows：Conda PowerShell

在 Anaconda PowerShell Prompt 或已初始化Conda的PowerShell中，从YONOD根目录执行：

```powershell
conda activate yonod
$env:PYTHONUTF8 = '1'
Set-Location .\example\external_test
python -u ..\..\yonod.py
```

首次提示输入 `run.yaml`。需要后台运行时，使用[总操作说明中的Windows日志与PID步骤](../README.md#windowsconda-powershell)，同步替换示例目录和任务名。Windows不执行Linux的Bash命令，目录含空格或中文时保留引号。

新结果写入本目录的 `result/example_external_test_20261113/`，网页报告为其中的 `report/external_test_report.html`。下载包的 `completed_run/` 保留已完成运行的快照；具体测试配置版本和平台见 [verification.json](verification.json)。历史运行中实际记录的资源参数只用于追溯，不是当前YAML的要求。

## 真正完成运行后的报告在哪里

下载包的[本轮网页报告](completed_run/report/external_test_report.html)与[文字报告](completed_run/report/external_test_report.md)保存在 `completed_run/` 完整快照中。重新运行使用同目录YAML，在本包 `result/<任务名>/` 新建结果；快照不会占用默认结果位置。源码Git不跟踪运行目录。

以下路径均相对当前示例文件夹；运行成功后应生成并核对这些结果：

| 路径（相对当前示例目录） | 内容 |
| --- | --- |
| `result/example_external_test_20261113/report/external_test_report.html` | 外测网页报告，包含实际训练/外测规模与外测指标。 |
| `result/example_external_test_20261113/report/external_test_report.md` | Markdown版外测报告。 |
| `result/example_external_test_20261113/docs/metrics/external_test_metrics.csv` | 完整精度外测指标。 |
| `result/example_external_test_20261113/docs/predictions/morgan__rf.parquet` | 6条外测记录的参考值、预测值、来源编号与组信息。 |

R²（决定系数）不是预测正确率；MAE（平均绝对误差）在本例0–1产率表示下乘100才是百分点。人工小样例只用于检查操作流程，不能据其分数评价真实化学预测效果。

本轮逐条验收记录：[verification.json](verification.json)。
