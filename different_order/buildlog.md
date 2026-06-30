# 构建日志

## [2026-06-30 22:39] 步骤 1 完成：生成 different_order 任务目录与配置

### 执行的任务
- 阅读 `different_order/project-goal.md`，确认项目目标为验证 Morgan 描述符向量拼接顺序对建模结果的影响。
- 检查模板配置 `different_order/012345/012345_yonod_config.json` 与现有目录状态。
- 按文档任务名列表创建任务目录并生成 `<task>/<task>_yonod_config.json`。
- 文档写明“一共需要创建21个文件夹”，但实际任务名列表包含 22 个任务，本次以任务名列表为准。
- 已存在且已有建模结果的任务：`012345`、`01`。本次仅更新/检查配置，不重新运行这两个建模任务。

### 实际任务清单
- 共 22 个：`01`、`10`、`012`、`021`、`210`、`201`、`013`、`103`、`0123`、`0213`、`2103`、`2013`、`01345`、`10345`、`012345`、`021345`、`210345`、`201345`、`0132`、`1032`、`013452`、`103452`

### 关键判断
- 项目目标文档定义列序号映射为：
  - 0 = `reactant-amide`
  - 1 = `reactant-acid`
  - 2 = `product`
  - 3 = `activation`
  - 4 = `base`
  - 5 = `solvent`
- 模板 `012345_yonod_config.json` 中 `descriptors.columns` 原顺序为 `reactant-amide, reactant-acid, activation, base, solvent, product`，这实际对应逻辑顺序 `013452`，与文档中 `012345 = 0-1-2-3-4-5` 的编号说明不一致。
- 现有 `main.py` 的配置模式会通过 `column_roles` 生成实际 `smiles_cols`，而不是直接使用 `descriptors[*].columns` 参与特征拼接。既有 `01` 与 `012345` 的 `metrics_summary.csv` 都显示 `n_smiles_cols=7`、`feature_dim=7168`，且四个模型结果完全一致，说明历史运行没有按任务名限制/调整 Morgan 拼接列。
- 规范化 CSV 实际列为 `reactant-1, reactant-2, activation-1, activation-2, base, solvent, product, yield`。其中逻辑列 `activation` 被拆为实际建模列块 `activation-1 + activation-2`。因此包含编号 `3` 的任务会使用两个实际列。
- 为保证后续建模命令实际按任务顺序拼接 Morgan 向量，本次配置同时写入：
  - `descriptors.columns`：任务编号对应的逻辑语义列名；
  - `column_roles.others`：`main.py` 实际读取的规范化 CSV 列顺序。

### 关键变更
- 新增任务目录：`10`、`012`、`021`、`210`、`201`、`013`、`103`、`0123`、`0213`、`2103`、`2013`、`01345`、`10345`、`021345`、`210345`、`201345`、`0132`、`1032`、`013452`、`103452`
- 更新/生成配置文件：
  - `different_order/<task>/<task>_yonod_config.json`
- 新增日志文件：
  - `different_order/buildlog.md`

### 遇到的问题及解决方案
- 问题：文档任务数量与任务名列表不一致。
  - 解决：以文档中的实际任务名列表为准，记录实际数量 22。
- 问题：模板 `012345` 的 columns 顺序与编号说明不一致。
  - 解决：以编号说明和用户指定映射为准，重新生成所有任务的 `descriptors.columns`。
- 问题：`main.py` 不直接使用 `descriptors.columns` 控制拼接顺序。
  - 解决：同步写入 `column_roles.others`，使实际 `smiles_cols` 顺序与任务名一致。

### 快速验证方法
```bash
find different_order -maxdepth 2 -name '*_yonod_config.json' | sort
```

预期输出包含 22 个配置文件。

### 下一步计划
- 按顺序跳过已完成任务 `01`、`012345`，对其余 20 个任务执行建模命令：
  `python main.py --config "different_order/<task>/<task>_yonod_config.json" --csv "different_order/amide-coupling_normalized_dataset.csv"`

---
