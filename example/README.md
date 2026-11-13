# YONOD 三种数据检验方式示例

先用浏览器打开 [index.html](index.html)。这里把数据表、任务设置文件（YAML）、划分清单和报告入口放在同一个 `example/` 文件夹中，方便逐项对照。

**全部数据都是人工教学记录，不是实验测量。** 分子结构文本用于说明格式，不代表所列组分一定能发生合理反应。随机划分包保留历史实际计算报告；固定划分和外测流程是新准备的输入示例，本次尚未运行，没有新的预测评分。

| 示例包 | 它回答的问题 | 主要文件与入口 |
|---|---|---|
| [随机划分内测](random_split/index.html) | 同一批已有反应中，随机留出一部分，能预测得怎样？ | [数据表](random_split/dataset/reactions.csv)、[设置文件](random_split/run.yaml)、[历史实际报告](random_split/report/report.html)、[操作说明](random_split/README.md) |
| [固定名单内测](fixed_split/README.md) | 按事先指定的成员名单留出反应，能预测得怎样？ | [数据表](fixed_split/dataset/reactions.csv)、[固定名单](fixed_split/splits/fixed_split_manifest.csv)、[名单解释表](fixed_split/splits/membership_guide.csv)、[设置文件](fixed_split/run.yaml)、[操作说明](fixed_split/README.md) |
| [独立外测流程](external_test/README.md) | 用训练表建立模型后，对另外提供的一批反应，能预测得怎样？ | [训练表](external_test/dataset/train.csv)、[外测表](external_test/dataset/external.csv)、[来源核查记录](external_test/audit/external_audit.json)、[设置文件](external_test/run.yaml)、[操作说明](external_test/README.md) |

可分别下载 [随机划分包](archives/random_split.zip)、[固定划分包](archives/fixed_split.zip)、[外测流程包](archives/external_test.zip)，或下载[全部示例](archives/all_examples.zip)。运行这些示例仍需要完整 YONOD 项目及其 `yonod` 环境。

## 三种方法的相同点和区别

三种方法都用已有参考产率与模型预测值比较。区别在于检验数据如何安排，以及结果能支持什么结论。

- **随机划分内测**：程序按随机种子打乱同一批记录，再安排哪些记录用于建立模型、哪些留出检验。本例采用两份轮换，每条记录得到一次留出预测。
- **固定名单内测**：使用者在计算前保存每批建模和留出成员的编号清单（split manifest），程序按清单执行。本例明确安排两折，各6条建模、6条检验，并交换角色。固定随机种子和“关闭打乱”都不等于表达了任意指定的成员名单。
- **独立外测流程**：训练表用于建立模型，外测表用于检验，模型设置在使用外测结果前固定。本例为12条训练、6条外测，两表列名通过设置中的映射对应。还需要来源核查记录、字段对应和重叠检查。另存一份 CSV 不自动证明独立性；本包人工数据只展示外测操作流程。

固定名单是否属于“未见底物检验”，取决于真实的底物分组与重叠情况。独立外测是否检验新底物，也需要按实际结构检查。不要仅从目录名或任务名称推断。

## 怎样操作

1. 打开对应示例包的操作说明，核对数据列名、记录编号与产率尺度。固定划分需要同时核对成员名单；外测需要同时核对训练／外测两表及来源核查文件。
2. 阅读包中的 `run.yaml`，再使用说明中对应的根 `config/` 正式任务配置。包内还提供 `run_from_root.yaml`，与该正式配置内容一致；可以将它复制到根 `config/` 后使用。请勿直接移动阅读用 `run.yaml`，因为两处的相对路径解析方式不同。准备自己的任务时另存新的配置文件，更新任务名、输入路径、`artifacts.output_dir` 和 `outputs.root`，让全部结果位于新的 `result/<任务名>/` 下。以仓库根 [example.yaml](../example.yaml) 为唯一完整格式规范，不从 Markdown 复制另一套模板。
3. 从仓库根初始化 Conda 并激活 `yonod`，通过 `yonod.py` 主入口提供配置路径。Linux 使用任务专属后台日志，选择恰好19个可用逻辑 CPU，日志与 PID 文件放在 `logs/`；任务依次运行。各包说明给出具体配置名称与操作命令。Windows 可用 Conda Prompt 按对应说明启动。
4. 完成后核对网页报告（HTML）、文字报告（Markdown）、逐条预测和运行记录。外测报告的文件名为 `report/external_test_report.html` 和 `report/external_test_report.md`；固定划分的具体产物以其说明为准。不要把包内历史报告当成新配置已经完成的结果。

留出反应的内测预测汇总（out-of-fold prediction，OOF）来自未学习该条记录的相应模型。R² 不是“预测正确率”；产率采用0–1尺度时，平均绝对误差（MAE）乘100才是相差的产率百分点。随机包中保留全部负 R²，供学习如何阅读结果。

本次整理没有生成结构编码、训练模型、调参或执行预测。三个包提供的内容与执行状态，以各自说明为准。

## 可复制到根 config/ 的正式设置

| 示例 | 可复制文件 | 复制到仓库根后的名称 |
|---|---|---|
| 随机划分 | [random_split/run_from_root.yaml](random_split/run_from_root.yaml) | `config/example_random_split_20261113.yaml` |
| 固定名单 | [fixed_split/run_from_root.yaml](fixed_split/run_from_root.yaml) | `config/example_fixed_split_20261113.yaml` |
| 外测流程 | [external_test/run_from_root.yaml](external_test/run_from_root.yaml) | `config/example_external_test_20261113.yaml` |

例如，Linux Bash 在仓库根复制随机示例设置：

```bash
mkdir -p config
cp example/random_split/run_from_root.yaml config/example_random_split_20261113.yaml
```

Windows PowerShell 在仓库根执行：

```powershell
New-Item -ItemType Directory -Force config | Out-Null
Copy-Item example/random_split/run_from_root.yaml config/example_random_split_20261113.yaml
```

随后按对应包说明激活环境并经 `yonod.py` 启动。上面的命令只复制设置，不启动计算。将示例放回完整 YONOD 项目的根 `example/` 目录后再操作，保留各包内部目录结构。

## Linux Bash：正式启动示例

以下以随机示例为例，在仓库根执行。先确认对应设置已在根 `config/` 中；只运行一个任务，结束后再启动另一个。代码会从当前允许使用的逻辑 CPU 中选择19个，不足19个时停止。

```bash
source /home/wangzh685/miniconda3/etc/profile.d/conda.sh
conda activate yonod

example_cpu_ids="$(python -c 'import os; ids=sorted(os.sched_getaffinity(0)); assert len(ids)>=19, "需要至少19个可用逻辑CPU"; print(",".join(map(str, ids[:19])))')"
if [ -z "$example_cpu_ids" ]; then
  echo "CPU选择失败；未启动任务。"
else
mkdir -p logs
nohup setsid bash -lc '
  source /home/wangzh685/miniconda3/etc/profile.d/conda.sh
  conda activate yonod
  printf "%s\n" "config/example_random_split_20261113.yaml" | taskset --cpu-list "$1" python -u yonod.py
' _ "$example_cpu_ids" </dev/null > logs/example_random_split_20261113.log 2>&1 &
echo $! > logs/example_random_split_20261113.pid
fi
```

固定名单或外测运行时，把配置路径、日志名与 PID 文件名一起改为各自的任务名。

确认启动后，可以按需查询状态：

```bash
task_pid="$(cat logs/example_random_split_20261113.pid)"
ps -p "$task_pid" -o pid,ppid,stat,etime,cmd
taskset --pid --cpu-list "$task_pid"
tail -n 100 logs/example_random_split_20261113.log
```

## Windows：Conda Prompt 启动

在仓库根打开 Conda Prompt，运行：

```text
conda activate yonod
python -u yonod.py
```

出现首个提示时输入 `config/example_random_split_20261113.yaml`。另两种示例输入各自配置名称；一次运行一个任务。示例的模型设置保留19线程预算，Windows 不使用 Linux 的 `taskset`、`nohup` 或 PID 查询命令。
