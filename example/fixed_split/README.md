# 固定名单两折内测：先决定谁用于建模、谁用于检验

本包展示**怎样让所有模型严格使用事先指定的同一份名单**。12条记录和产率都是人工教学数据，结构搭配不代表经过实验验证的反应；**已有完整运行快照，实际测试配置版本与平台见verification.json。**

与随机划分不同，这里直接写明每条记录在每一轮的用途。固定名单属于数据划分方法，固定模型参数属于模型设置，两者是两件事。

## 先看哪些文件

| 文件 | 用途 |
| --- | --- |
| [dataset/reactions.csv](dataset/reactions.csv) | 12条人工记录，附加`fixed_block`分组列。 |
| [splits/membership_guide.csv](splits/membership_guide.csv) | 最易阅读的12行角色对照表：`held_out`是暂不用于建立模型的留出记录。 |
| [splits/fixed_split_manifest.csv](splits/fixed_split_manifest.csv) | 程序真正读取的24行划分清单（split manifest），每折列出全部12条记录。 |
| [run.yaml](run.yaml) | 直接在本目录运行的任务配置；所有输入与输出路径相对本文件所在目录。 |
| [provenance.json](provenance.json) | 数据来源、校验码与本轮运行记录。 |

## 数据表中的列是什么意思

| 列 | 内容及作用 |
| --- | --- |
| `reaction_id` | 唯一记录编号，如`fixed-01`；用于把名单和数据对上，不参与预测。 |
| `yield` | 人工参考产率；`0.10`表示10%，检验时与预测值比较。 |
| `reactant_1_smiles`、`reactant_2_smiles` | 结构文字表示（SMILES），转换为模型能处理的结构数值编码。 |
| `fixed_block` | 本例预先定义的两组：`block_A`为前6条，`block_B`为后6条；只约束划分，不输入模型。 |

`block_A`和`block_B`只是教学分组，**没有“不同底物”“不同实验室”或“不同反应类型”的化学含义**。真实研究可按底物、实验批次等定义组，但需要自行核查相应结构或来源确实没有跨组重复。

## 两折怎样轮换

| 轮换批次（fold） | 根据这些记录建立模型（train） | 用这些记录检验（valid） |
| --- | --- | --- |
| 第1折 | `fixed-07`至`fixed-12`，即`block_B` | `fixed-01`至`fixed-06`，即`block_A` |
| 第2折 | `fixed-01`至`fixed-06`，即`block_A` | `fixed-07`至`fixed-12`，即`block_B` |

这样每条记录恰好被留出一次，两次建立的是两个模型。汇总的是各记录**未参与对应模型建立时**得到的预测（折外预测，OOF）。同组记录始终一起分配。

当前程序的这条路线至少要求两折、每条记录每轮恰好留出一次。它不能用一份仅有“8条训练+4条测试”的单次名单替代本例。把`shuffle`改成`false`或只固定随机种子，也不能表达任意预先指定的成员名单。

## 修改自己的数据时，哪些文件需要一起改

1. 在CSV保留唯一编号、参考值和模型输入列。分组列只写在`evaluation.grouping.group_column`，不要误放进模型输入列。
2. 先确定各折成员，再修改正式划分清单。`role`只用`train`或`valid`；`source_row_index`从0开始，记录原CSV行序。
3. 每折必须包含全部记录且无重复，同一组不能同时出现在训练与留出部分；每轮每条记录恰好一次`valid`。本例采用1轮2折，因此12条记录对应24行清单。
4. 清单的`dataset_sha256`必须改为新CSV文件的SHA-256。它防止误用另一份数据，不能证明实验来源独立。`group_id`须与实际分组意义一致；不能为了通过检查随意编号。
5. 在YAML同步修改列名、折数、名单路径和独立任务名；将所有输出放在新的`result/<任务名>/`，避免复用旧结果目录。

程序会核查导入名单的成员、文件身份及组是否跨训练/留出；它不会仅凭组名判断化学结构是否真正未见。

## 在本文件夹运行

从YONOD根目录进入 `example/fixed_split/`，检查同目录的 [run.yaml](run.yaml)，无需复制或移动设置。输入和新结果保留在本包中；当前YAML未指定线程数，采用模型库默认值，可按本机资源另设。每次只运行一个任务。

### Linux：Bash

先初始化本机Conda并激活环境，从YONOD根目录执行以下后台启动块：

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate yonod
export PYTHONUTF8=1
cd example/fixed_split
mkdir -p ../../logs
printf '%s\n' 'run.yaml' > ../../logs/example_fixed_split_20261113.stdin.txt
nohup python -u ../../yonod.py \
  < ../../logs/example_fixed_split_20261113.stdin.txt \
  > ../../logs/example_fixed_split_20261113.log 2>&1 &
echo $! > ../../logs/example_fixed_split_20261113.pid
```

若 `conda` 尚未加入shell路径，请先用本机安装目录的 `etc/profile.d/conda.sh` 初始化。日志和PID位于仓库根 `logs/`；[总操作说明](../README.md#linuxbash-后台运行)包含按需查询状态的命令。

### Windows：Conda PowerShell

在 Anaconda PowerShell Prompt 或已初始化Conda的PowerShell中，从YONOD根目录执行：

```powershell
conda activate yonod
$env:PYTHONUTF8 = '1'
Set-Location .\example\fixed_split
python -u ..\..\yonod.py
```

首次提示输入 `run.yaml`。需要后台运行时，使用[总操作说明中的Windows日志与PID步骤](../README.md#windowsconda-powershell)，同步替换示例目录和任务名。Windows不执行Linux的Bash命令，目录含空格或中文时保留引号。

新结果写入本目录的 `result/example_fixed_split_20261113/`，网页报告为其中的 `report/benchmark_report.html`。下载包的 `completed_run/` 保留已完成运行的快照；具体测试配置版本和平台见 [verification.json](verification.json)。历史运行中实际记录的资源参数只用于追溯，不是当前YAML的要求。

## 运行完成后会看到什么

下载包的[本轮网页报告](completed_run/report/benchmark_report.html)与[文字报告](completed_run/report/benchmark_report.md)保存在 `completed_run/` 完整快照中。重新运行使用同目录YAML，在本包 `result/<任务名>/` 新建结果；快照不会占用默认结果位置。源码Git不跟踪运行目录。

以下路径均相对当前示例文件夹；运行成功后应生成并核对这些结果：

| 路径（相对当前示例目录） | 内容 |
| --- | --- |
| `result/example_fixed_split_20261113/report/benchmark_report.html` | 网页报告：模型结果、数据划分核对、每折完成情况和图表。 |
| `result/example_fixed_split_20261113/report/benchmark_report.md` | 同一次计算的Markdown报告。 |
| `result/example_fixed_split_20261113/docs/manifests/split_manifest.parquet` | 重新核查后实际采用的划分清单。 |
| `result/example_fixed_split_20261113/docs/predictions/` | 每折逐条留出预测，便于核对实际成员。 |

先检查两折都完成，再看预测误差。R²（决定系数）不是正确率；MAE（平均绝对误差）在本例0–1产率单位下乘100后，才是产率百分点。固定名单本身不保证模型好，也不自动证明未见底物预测能力。

本轮逐条验收记录：[verification.json](verification.json)。


### 生成其他任务的配置

本包run.yaml已经准备好，继续按上面的分系统步骤执行。为自己的CSV创建新配置时，回到主仓根使用generate_yaml_config.py，仅生成/校验标准YAML，随后由yonod.py启动。操作见[总说明](../README.md)；普通运行器现名为run_modeling.py，历史运行记录中的旧名不改写。
