# 随机划分内测示例：先看数据，再看设置，最后看报告

这是供化学研究人员阅读的完整示例入口。建议先用浏览器打开 [index.html](index.html)，再按下面顺序看文件。

**数据是12条人工构造的流程测试记录，不是实验测量数据。** 分子结构文本用于演示输入格式，不表示这两列分子一定能发生合理反应。本页保留同一批数据、同一模型与随机划分设置的历史实际计算输出，供对照来源。本轮已用本目录YAML完整运行并验收，新增结果与历史报告分开保存。

本轮下载快照：[网页报告](completed_run/report/report.html) · [文字报告](completed_run/report/report.md)。快照保存在 `completed_run/`，重新运行仍写入独立的 `result/example_random_split_20261113/`。

## 1. 示例文件夹里有什么

| 文件 | 先看什么 |
|---|---|
| [dataset/reactions.csv](dataset/reactions.csv) | 反应数据表（CSV）：每行一条记录，含编号、参考产率和两列分子结构 |
| [run.yaml](run.yaml) | 可直接在本文件夹运行的建模任务设置文件（YAML）：说明选哪些列、怎样编码、采用哪些建模方法和怎样测试 |
| [tables/split_assignments.csv](tables/split_assignments.csv) | 每条记录在哪一批测试中被留出；它们不是三套不同模型各自随意拆出的数据 |
| [tables/metrics.csv](tables/metrics.csv) | 从已保存逐条预测核对得到的评分，保留全部三种方法及负分 |
| [report/report.html](report/report.html) | 原始完整网页报告，保留原任务名和原报告用语 |
| [report/report.md](report/report.md) | 原始完整文字报告；图片使用包内 `pictures/` 相对路径 |
| `pictures/` | 逐反应预测对照图与计算耗时图 |
| `evidence/run_morgan-rf/` 等 | 历史原始设置、生效设置、逐条预测、每批评分及运行记录清单的审计节选 |
| [provenance.json](provenance.json) | 数据和报告来自哪里、哪些文件是原字节复制、哪些是本次整理的展示内容 |

直接进入本目录，通过 `../../yonod.py` 选择 `run.yaml`。输入和结果路径都相对本YAML所在目录，无须复制到根 `config/`。新报告位于 `result/example_random_split_20261113/report/`；旧报告仍位于 `report/`，来自 `diag215_20260929_ordinary_fixed_off_v1`，保留原任务名称与日期。

`evidence/` 不是整套可重载模型的交付。保存预测和评分，并不等于保存了全部模型、转换文件及原运行环境；请勿据此直接调用模型预测接口。

## 2. 数据表怎样写

本例使用四列。列名可以按自己的数据命名，但必须在设置中准确对应。

| 反应记录编号（`reaction_id`） | 人工参考产率（`yield`） | 第一列结构（`reactant_1_smiles`） | 第二列结构（`reactant_2_smiles`） |
|---|---:|---|---|
| smoke-01 | 0.10 | C | CO |
| smoke-02 | 0.16 | CC | CO |
| smoke-03 | 0.22 | CCC | CO |

- **反应记录编号（sample_id）**：必须非空、不重复；它负责把结构、参考产率和预测值对应起来。
- **预测目标列（label）**：这里是 `yield`。本表采用0–1尺度，所以0.10表示10%，0.76表示76%。这里的数值是人工示例值。
- **分子结构文本（SMILES）**：这里选入两列结构作为已知输入。第二列全部是 `CO`，不构成对任意新组分预测能力的检验。
- 本例没有产物结构、温度等数值条件或组分类别列。其他列不会仅因出现在数据表中就自动参与建模。

## 3. 设置文件里哪些字段最值得看

本页不复制第二份完整YAML模板；唯一权威结构仍是仓库根的 [example.yaml](../../example.yaml)。直接打开示例设置文件，可以看到本任务采用的具体值。

| 设置字段 | 本例的值或选择 | 通俗含义 |
|---|---|---|
| `schema_version` | `2.0` | 使用当前设置格式 |
| `project_name` | `example_random_split_20261113` | 新示例任务的独立名称；历史报告保留原任务名 |
| `stage` | `all` | 连续完成输入编码、建立模型与内测报告 |
| `dataset.path` | 指向本包的 `dataset/reactions.csv` | 读取本目录中的反应表，输入和输出都按YAML所在目录解析 |
| `dataset.sample_id_col` | `reaction_id` | 哪列是反应记录编号 |
| `dataset.column_roles.label` | `yield` | 哪列是希望预测的参考产率 |
| `dataset.column_roles.reactants` | 两列结构文本 | 这些列可以作为已知分子输入 |
| `descriptors` | `morgan`，两列按固定顺序合并 | 用结构片段数值编码（Morgan fingerprint）表达两列分子 |
| `models` | `rf`、`xgb`、`lightgbm` | 分别比较三种建模方法 |
| `model_params.*.estimator.n_estimators` | 均为 `2` | 这是低成本流程演示，设置很小，不用于追求化学预测表现 |
| `model_params.*.estimator.n_jobs` | 均为 `19` | 线程预算；不表示建了19个模型，也不是反应数量 |
| `hpo.enabled` | `false` | 不自动比较和筛选模型设置；更不表示优化反应条件 |
| `evaluation.protocol` | `outer_kfold` | 执行随机分批轮换内测 |
| `evaluation.n_splits` | `2` | 分成两份，轮流留一份测试 |
| `evaluation.n_repeats` | `1` | 完整轮换一套，不重复重新划分 |
| `evaluation.shuffle` | `true` | 先打乱记录顺序后再划分 |
| `evaluation.seed` | `42` | 固定随机安排，方便核对划分；这个数字没有化学含义 |
| `artifacts.output_dir` | 新任务的 `feature/` 子目录 | 将结构编码保存到任务自己的文件夹 |
| `outputs.root` | `./result/example_random_split_20261113` | 在当前示例目录内保存新结果，与历史报告分开 |
| `outputs.report_formats` | `html`、`markdown` | 完成时生成网页和文字两种报告 |
| `outputs.diagnostics.enabled` | `false` | 关闭已学反应与留出反应的额外效果对照；不关闭正常内测预测和评分 |

随机森林（Random Forest，RF）将多棵决策树的预测汇总；逐轮修正误差的树模型（XGBoost、LightGBM）使用不同软件实现。这里采用同一批反应、同一编码与同一数据划分比较它们。

## 4. 两折随机内测实际怎样安排

分批轮换内测（K-fold cross-validation）会让每条记录轮到一次留出检验。本例的固定安排如下：

| 测试批次 | 用于建立模型的记录 | 暂不让本轮模型学习、用于检验的记录 |
|---|---|---|
| 第1批 | 02、04、05、07、08、12 | 01、03、06、09、10、11 |
| 第2批 | 01、03、06、09、10、11 | 02、04、05、07、08、12 |

表中编号省略共同前缀 `smoke-`。两批各用6条建立模型、6条测试；角色交换后，12条记录各得到一次未参与相应模型学习时的预测。这些预测合起来叫**留出反应的内测预测汇总（out-of-fold prediction，OOF）**。

三种方法各执行两批，共有6次模型拟合。这里不是做了6次独立化学实验。随机划分也不保证测试中的底物或结构片段从未出现在建模数据中；本例不是“未见底物内测”，更不是独立外测。

## 5. 最终报告先看哪里

先看“任务信息”，确认12条记录；再看“结果矩阵”和“详细指标”；最后看每种方法的散点图。每个点对应一条留出预测，横纵坐标比较参考值与预测值，越接近对角线越吻合。

下表保留从已保存预测复核的全部结果。R²和误差取全部12条留出预测汇总，不能拿每批评分的简单平均替代。

| 建模方法 | 对测试结果差异的预测评分（R²） | 更强调大误差的综合误差（RMSE，0–1尺度） | 平均绝对误差（MAE，0–1尺度） | 平均相差的产率百分点 |
|---|---:|---:|---:|---:|
| 随机森林（RF） | -0.638112 | 0.265094 | 0.222500 | 22.2500 |
| XGBoost | -0.102794 | 0.217508 | 0.184805 | 18.4805 |
| LightGBM | -0.173512 | 0.224374 | 0.189520 | 18.9520 |

平均绝对误差表示每条预测与参考值相差的绝对大小，再求平均。这里RF的MAE为0.2225，对应平均相差22.25个百分点。RMSE先将误差平方、求平均后再开平方，因此更强调大误差；同样使用0–1数值尺度。

R²比较模型与统一使用该测试结果均值预测的表现。负值表示在这个参照下更差，不是“负的正确率”，也不能据此把模型方法判定为对所有化学问题无效。本例数据人工构造、记录极少、模型设置很小，分数只用于展示报告如何呈现。

报告中的标准差为0，是因为只安排了一套完整轮换，不能据此声称稳定、没有不确定性，或已经做过多次独立验证。即便报告把某组合排第一，也只是在这个示例范围内的排序。

## 6. 原报告中几个容易误读的地方

- 开头“训练与外层评估诊断”的 `disabled` 表示额外效果对照被关闭。该诊断汇总的完成数为0，不表示三种模型的正常两折内测失败；正常评分表和各运行清单显示完整内测结果。
- “严格外层CV加权排名”是历史报告的标题。本例实际协议为随机 `outer_kfold`，不能从“严格”两个字推断采用了底物分组留出。
- 任务信息中的“SMILES列”为 `—` 是旧报告展示项未填；实际模型明确使用两列分子结构，见YAML和原始设置记录。
- 原任务名、生成日期和原项目路径有意保留，用来追查历史来源；这份旧报告没有改写；本轮新任务的报告另存于包内的 `result/`。
- 原报告保留历史通用指标文案，其中固定R²/误差阈值及用途建议不能直接作为本例或其他化学体系的可靠性标准。R²按“1减去误差平方和／测试值离均差平方和”理解；0–1尺度的误差乘100才是百分点，MAE也并非完全不受极端误差影响。本包的指标解读以本页和导览页为准。
- 原报告网页可能用在线公式排版资源；主要图片已内嵌。文字版的图片位于同包的 `pictures/`，请一起保留。

想看现有平台的整体功能，可读[解释版结构说明](../../project-docs/docs/yonod-structure-behavior-20261109.md)或[术语对照版](../../project-docs/docs/yonod-structure-behavior-20261109-technical.md)。


## 在本文件夹运行

1. 从YONOD根目录进入 `example/random_split/`，检查数据和同目录的 [run.yaml](run.yaml)，无需复制设置。
2. Linux按[总操作说明](../README.md)使用后台命令，激活 `yonod` 环境并绑定19个可用逻辑CPU，通过 `python -u ../../yonod.py` 启动，首次提示输入 `run.yaml`。日志与PID位于根 `logs/example_random_split_20261113_selfcontained.log` 和 `.pid`。
3. 新结果写入本目录的 `result/example_random_split_20261113/`；网页与文字报告为其中的 `report/report.html` 和 `report/report.md`。本轮已完整运行并验收，最终完成状态以运行记录为准。

Windows Conda Prompt从YONOD根目录执行：

```text
cd example\random_split
conda activate yonod
python -u ..\..\yonod.py
```

首次提示输入 `run.yaml`。每次只运行一个示例。

本目录的历史数据、报告、图片和预测证据保留原字节；`evidence/`中的历史YAML仅用于追溯原计算，不是本轮的任务入口。下载压缩包附本轮完整输出；源码克隆后自行运行生成包内 `result/`。

本轮逐条验收记录：[verification.json](verification.json)。
