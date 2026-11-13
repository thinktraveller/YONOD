# 固定名单两折内测：先决定谁用于建模、谁用于检验

本包展示**怎样让所有模型严格使用事先指定的同一份名单**。12条记录和产率都是人工教学数据，结构搭配不代表经过实验验证的反应；本次只准备文件并检查配置，**尚未运行，也没有本包的预测分数或最终报告**。

与随机划分不同，这里直接写明每条记录在每一轮的用途。固定名单属于数据划分方法，固定模型参数属于模型设置，两者是两件事。

## 先看哪些文件

| 文件 | 用途 |
| --- | --- |
| [dataset/reactions.csv](dataset/reactions.csv) | 12条人工记录，附加`fixed_block`分组列。 |
| [splits/membership_guide.csv](splits/membership_guide.csv) | 最易阅读的12行角色对照表：`held_out`是暂不用于建立模型的留出记录。 |
| [splits/fixed_split_manifest.csv](splits/fixed_split_manifest.csv) | 程序真正读取的24行划分清单（split manifest），每折列出全部12条记录。 |
| [run.yaml](run.yaml) | 包内阅读版配置，路径相对本包目录；与正式副本的数据和模型设置相同。 |
| [run_from_root.yaml](run_from_root.yaml) | 操作模板：复制到仓库根`config/example_fixed_split_20261113.yaml`后运行，`./`路径相对仓库根。 |
| [provenance.json](provenance.json) | 数据来源、校验码、预期输出与“未运行”状态。 |

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

## 如何运行本例

本次没有执行以下建模命令。先把整个`example/`放在YONOD仓库根目录，然后在**仓库根**操作。

Linux Bash：

```bash
mkdir -p config
cp example/fixed_split/run_from_root.yaml config/example_fixed_split_20261113.yaml
```

Windows PowerShell：

```powershell
New-Item -ItemType Directory -Force -Path config | Out-Null
Copy-Item example/fixed_split/run_from_root.yaml config/example_fixed_split_20261113.yaml
```

随后按[总操作说明](../README.md)启动`yonod.py`，首次提示时提供`config/example_fixed_split_20261113.yaml`。Linux需使用`yonod`环境、19个可用逻辑CPU、无头启动及任务专属`logs/`日志；任务按顺序运行。不要直接把包内`run.yaml`或`run_from_root.yaml`当成位于根`config/`的正式配置启动。

## 运行完成后会看到什么

以下是**预期路径和内容**，本次尚未生成：

| 路径（相对YONOD根目录） | 内容 |
| --- | --- |
| `result/example_fixed_split_20261113/report/benchmark_report.html` | 网页报告：模型结果、数据划分核对、每折完成情况和图表。 |
| `result/example_fixed_split_20261113/report/benchmark_report.md` | 同一次计算的Markdown报告。 |
| `result/example_fixed_split_20261113/docs/manifests/split_manifest.parquet` | 重新核查后实际采用的划分清单。 |
| `result/example_fixed_split_20261113/docs/predictions/` | 每折逐条留出预测，便于核对实际成员。 |

先检查两折都完成，再看预测误差。R²（决定系数）不是正确率；MAE（平均绝对误差）在本例0–1产率单位下乘100后，才是产率百分点。固定名单本身不保证模型好，也不自动证明未见底物预测能力。
