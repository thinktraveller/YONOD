# Ablation2 正式运行台账（结果前冻结）

本台账在第一项正式模型启动前建立。正式任务均采用严格 `manifest_outer_cv`，由 `yonod.py` 读取任务专属 schema-2 YAML 启动；一个任务只写入自己的 `result/<task_name>/`，并使用 19 个 CPU（Linux CPU IDs `0-18`）。任务严格串行，完成和验收一个任务后才启动下一个。

| 项目 | 已冻结值 |
| --- | --- |
| population | `result/ablation2_amide_data_prep_v1/data/standardized_population.csv`；47,015 rows；SHA `5fae4b314ffdb4d3f5eddcd6a59b627cee8ea5a939a9c375dbeb3a0f81c3989c` |
| 组合 | `full`, `minus_a`, `minus_b`, `minus_p`, `minus_a_p`, `minus_b_p`, `product_conditions`, `conditions_only`；C 顺序为 activation, additive, base, solvent |
| protocols | `random_repeat_group`, `unseen_amine`, `unseen_acid`, `unseen_substrate_pair`；各自预生成 5 folds × 3 repeats 外部 parquet manifest |
| primary | Morgan ECFP4 defaults（radius 2, 1024 bits）× RF：300 trees，`max_features: 1.0`，`min_samples_leaf: 1`，seed 20260918，`n_jobs: 19` |
| robustness | MFP radius 3, 1024, standard × LightGBM：500 trees，learning rate 0.05，31 leaves，`min_child_samples: 1`，seed 20260918，`n_jobs: 19` |
| first task | `config/ablation2_formal_morgan_rf_random_repeat_group_full_v1.yaml`；这是新的 full baseline，绝不引用历史结果 |

同一协议内八个组合必须导入同一外部 manifest；下一个任务的 YAML 在启动前独立保存并经真实 strict loader 校验。pair 留出只保证底物对未见，不能表述为两种单底物均未见。此文件不记录正式性能结果。
