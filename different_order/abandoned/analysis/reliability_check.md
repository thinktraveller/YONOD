# different_order 结果可靠性确认

## 1. 结果完整性

- 已发现任务目录：32 个（31 个新任务 + `1234567` 基准任务）。
- 已汇总指标行数：768 行。
- 每个任务均包含 24 条结果：6 个 descriptor × 4 个 model。
- 未发现缺失的 descriptor/model 组合。

输出文件：

- `all_metrics.csv`
- `completeness_summary.csv`

结论：从运行完成度和结果文件完整性看，所有任务都已成功产出指标。

## 2. 配置生效审计

审计发现当前结果不能可靠回答“不同任务配置的分子列组合/拼接顺序是否影响 Morgan 结果”。

关键证据：

- `config_effect_audit.csv` 共 768 行，其中 768 行的 `reported_feature_dim` 都不符合“按 descriptor columns 建模”时的预期维度。
- 所有任务的 `reported_n_smiles_cols` 都是 8，不随任务名中的列数变化。
- 标准化数据集实际列为：
  `reactant-1, reactant-2, activation-1, activation-2, additive, base, solvent, product, yield`
- 当前配置中的 descriptor `columns` 使用的是：
  `reactant-amide, reactant-acid, product, activation, additive, base, solvent`
  这与标准化 CSV 中的部分列名不一致。
- `morgan` 在 32 个任务上的 4 个模型指标完全一致：
  - `morgan + xgb`: `r2_mean = 0.687192764045`
  - `morgan + rf`: `r2_mean = 0.765973382388`
  - `morgan + svm`: `r2_mean = 0.642834191938`
  - `morgan + autogluon`: `r2_mean = 0.811031679749`

输出文件：

- `config_effect_audit.csv`
- `metric_variation_by_descriptor_model.csv`

结论：结果文件本身完整，但当前实验结果不适合直接用于证明 Morgan 描述符拼接顺序是否有影响。更合理的解释是：建模代码没有按每个 descriptor 的 `columns` 配置选择列，而是统一使用 `column_roles` 得到的 8 个 SMILES 列。

## 建议

下一步应先修改建模流程，使每个 descriptor 使用配置中的 `columns` 字段，或重新生成配置使 `column_roles` 与任务列组合一致。然后至少重跑 Morgan 相关任务，再进行正式结果分析。
