# YONOD 三种数据检验方式示例

先用浏览器打开 [index.html](index.html)。三个包都只需使用自己目录中的 `run.yaml`，不用复制设置到其他目录。数据、划分清单和新计算结果都留在各自包内。

**全部数据都是人工教学记录，不是实验测量。** 分子结构文本用于说明格式，不代表所列组分一定能发生合理反应。三个样例均有完整运行快照和逐条验收记录；实际测试的配置版本与平台见各包的 `verification.json`。随机包另保留2026-09-29的历史实际报告，供追查来源。

| 示例包 | 它回答的问题 | 主要文件与入口 |
|---|---|---|
| [随机划分内测](random_split/index.html) | 同一批已有反应中，随机留出一部分，能预测得怎样？ | [数据表](random_split/dataset/reactions.csv)、[设置文件](random_split/run.yaml)、[历史报告](random_split/report/report.html)、[操作说明](random_split/README.md) |
| [固定名单内测](fixed_split/README.md) | 按事先指定的成员名单留出反应，能预测得怎样？ | [数据表](fixed_split/dataset/reactions.csv)、[固定名单](fixed_split/splits/fixed_split_manifest.csv)、[名单解释表](fixed_split/splits/membership_guide.csv)、[设置文件](fixed_split/run.yaml) |
| [独立外测流程](external_test/README.md) | 用训练表建立模型后，对另外提供的一批反应，能预测得怎样？ | [训练表](external_test/dataset/train.csv)、[外测表](external_test/dataset/external.csv)、[来源核查记录](external_test/audit/external_audit.json)、[设置文件](external_test/run.yaml) |

可分别下载[随机划分包](archives/random_split.zip)、[固定划分包](archives/fixed_split.zip)、[外测流程包](archives/external_test.zip)，或下载[全部示例](archives/all_examples.zip)。下载包的 `completed_run/` 保存本轮完整输出快照；重新运行会生成本包的 `result/`，两者互不占用。源码克隆不包含这些运行目录。实际运行仍需要完整YONOD项目及其 `yonod` 环境。

## 三种方法的相同点和区别

三种方法都用参考产率与模型预测值比较。区别在于检验数据如何安排，以及结果能支持什么结论。

- **随机划分内测**：程序按随机种子打乱同一批记录，再安排哪些用于建立模型、哪些留出检验。本例两份轮换，每条记录得到一次留出预测。
- **固定名单内测**：使用者提前保存每批建模和留出成员的编号清单（split manifest）。本例两折各6条建模、6条检验，再交换角色。固定随机种子或关闭打乱，不能表达任意指定的成员名单。
- **独立外测流程**：训练表用于建立模型，外测表用于检验，模型设置提前固定。本例12条训练、6条外测，两表不同列名通过设置中的映射对应。另存一个CSV不自动证明独立性；本包人工数据只演示外测流程。

固定名单是否属于“未见底物检验”，取决于底物分组和重叠核查。独立外测是否检验新底物，也需要按实际结构检查。

## 在自己的示例文件夹运行

| 示例目录（相对YONOD根） | 首次提示时输入 | 新结果目录（相对该示例目录） |
|---|---|---|
| `example/random_split/` | `run.yaml` | `result/example_random_split_20261113/` |
| `example/fixed_split/` | `run.yaml` | `result/example_fixed_split_20261113/` |
| `example/external_test/` | `run.yaml` | `result/example_external_test_20261113/` |

1. 把整个 `example/` 放在完整YONOD项目根目录中，保持各包内部结构。
2. 进入选中的示例目录。先看CSV列名、编号和产率单位，再看同目录的 `run.yaml`；固定划分另核对名单，外测另核对映射和来源记录。
3. 激活 `yonod` 环境，通过 `../../yonod.py` 主入口启动，首次提示输入 `run.yaml`。输入与输出路径均按该YAML所在目录解析。
4. 等当前任务完成后再启动下一个。核对网页报告（HTML）、文字报告（Markdown）、逐条预测和运行清单。

准备自己的任务时另存独立YAML，同步修改任务名、输入路径、`artifacts.output_dir` 与 `outputs.root`。完整设置格式仍以仓库根 [example.yaml](../example.yaml) 为准，本说明不复制另一套YAML模板。

## Linux：Bash 后台运行

先在本机安装并初始化 Conda。以下从YONOD根目录启动随机示例；若 `conda` 尚不在 shell 的路径中，先用本机 Conda 安装目录内的 `etc/profile.d/conda.sh` 初始化。这里不限制可用CPU数量，也不绑定特定CPU编号。

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate yonod
export PYTHONUTF8=1
cd example/random_split

example_repo_root="$(cd ../.. && pwd)"
example_task="example_random_split_20261113"
mkdir -p "$example_repo_root/logs"
printf '%s\n' 'run.yaml' > "$example_repo_root/logs/$example_task.stdin.txt"
nohup python -u ../../yonod.py \
  < "$example_repo_root/logs/$example_task.stdin.txt" \
  > "$example_repo_root/logs/$example_task.log" 2>&1 &
echo $! > "$example_repo_root/logs/$example_task.pid"
```

固定名单或外测运行时，进入相应示例目录，并将 `example_task` 改为该包YAML中的任务名。每次只启动一个任务，完成后再运行下一个。输入与输出仍按同目录 `run.yaml` 解析；日志、标准输入文件与PID统一存于仓库根 `logs/`。

仍在随机示例目录时，按需查询状态：

```bash
task_pid="$(cat ../../logs/example_random_split_20261113.pid)"
ps -p "$task_pid" -o pid,ppid,stat,etime,cmd
tail -n 100 ../../logs/example_random_split_20261113.log
```

## Windows：Conda PowerShell

打开 Anaconda PowerShell Prompt 或已完成 Conda 初始化的 PowerShell，从YONOD根目录执行。直接交互运行最容易检查输入；在首次提示输入 `run.yaml`：

```powershell
conda activate yonod
$env:PYTHONUTF8 = '1'
Set-Location .\example\random_split
python -u ..\..\yonod.py
```

需要保存后台日志时，用下面的完整块代替上面的交互启动块，仍从YONOD根目录执行。它将所有操作文件放入根 `logs/`，工作目录与输入文件保持在示例包中：

```powershell
conda activate yonod
$env:PYTHONUTF8 = '1'
$exampleRepoRoot = (Get-Location).Path
$exampleWorkDir = Join-Path $exampleRepoRoot 'example\random_split'
$exampleTask = 'example_random_split_20261113'
$exampleLogs = Join-Path $exampleRepoRoot 'logs'
New-Item -ItemType Directory -Force -Path $exampleLogs | Out-Null
$exampleInput = Join-Path $exampleLogs "$exampleTask.stdin.txt"
Set-Content -LiteralPath $exampleInput -Value 'run.yaml' -Encoding ascii
$examplePython = (Get-Command python -CommandType Application | Select-Object -First 1).Source
$exampleProcess = Start-Process -FilePath $examplePython -ArgumentList '-u', '..\..\yonod.py' -WorkingDirectory $exampleWorkDir -RedirectStandardInput $exampleInput -RedirectStandardOutput (Join-Path $exampleLogs "$exampleTask.log") -RedirectStandardError (Join-Path $exampleLogs "$exampleTask.stderr.log") -PassThru
Set-Content -LiteralPath (Join-Path $exampleLogs "$exampleTask.pid") -Value $exampleProcess.Id -Encoding ascii
```

固定名单或外测只需修改 `$exampleWorkDir` 和 `$exampleTask`。不要同时运行交互块和后台块。保持上面的PowerShell窗口或重设 `$exampleLogs` 后，可查询进程和最近输出：

```powershell
$exampleTaskPid = [int](Get-Content -LiteralPath (Join-Path $exampleLogs "$exampleTask.pid"))
Get-Process -Id $exampleTaskPid -ErrorAction SilentlyContinue
Get-Content -LiteralPath (Join-Path $exampleLogs "$exampleTask.log") -Encoding UTF8 -Tail 100
Get-Content -LiteralPath (Join-Path $exampleLogs "$exampleTask.stderr.log") -Encoding UTF8 -Tail 100
```

Windows不使用Bash的 `source`、`nohup`、`ps` 或输入重定向写法。路径含空格或中文时保留引号，并保持整个包的相对层级。上述Windows步骤说明运行方法；已完成的实机验收平台以 `verification.json` 为准。

## 计算资源由自己的机器决定

当前三个 `run.yaml` 不声明固定线程数，采用所用模型库的默认值，不要求某种工作站或最少核数。需要控制资源时，在所选RF/XGBoost/LightGBM的 `model_params.<模型>.estimator.n_jobs` 中填写适合本机的设置；不同库对参数默认值和特殊值的含义可能不同，不要跨库照搬。AutoGluon使用 `model_params.autogluon.fit.num_cpus`。无论线程设置为何，各任务、描述符组合和折都依次执行。

## 报告先看哪里

下载包中的本轮快照：[随机报告](random_split/completed_run/report/report.html) · [固定名单报告](fixed_split/completed_run/report/benchmark_report.html) · [外测报告](external_test/completed_run/report/external_test_report.html)。快照位于各包 `completed_run/`，重新运行的报告位置如下。

| 示例 | 新网页报告（相对该示例目录） |
|---|---|
| 随机划分 | `result/example_random_split_20261113/report/report.html` |
| 固定名单 | `result/example_fixed_split_20261113/report/benchmark_report.html` |
| 外测 | `result/example_external_test_20261113/report/external_test_report.html` |

留出反应的内测预测汇总（out-of-fold prediction，OOF）来自未学习该条记录的相应模型；外测则每个组合只用训练表建立一次模型。R²不是“预测正确率”；0–1产率尺度上的平均绝对误差（MAE）乘100才是产率百分点。人工小样例的评分用于学习操作与读报告，不用于推荐真实化学预测模型。

逐条验收记录：[随机划分](random_split/verification.json) · [固定名单](fixed_split/verification.json) · [外测](external_test/verification.json)。固定名单报告中的平均逐折指标与随机包汇总留出预测的指标口径不同，不直接跨路线比较分数。
