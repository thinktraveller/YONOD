# YONOD 严谨模型比较报告

生成时间：`2026-09-23 02:21:45`  
run_id：`liao-rh-ch-amidation-6d81a08f620d`

> 本报告只读取 `docs/manifests/`、`docs/predictions/`、`docs/folds/` 和 `docs/metrics/`，不重新训练模型。

## 实验可追溯性

| run_id | config_hash | dataset_sha256 | code_git_commit | dataset_path | grouping | cv | derived_from |
|---|---|---|---|---|---|---|---|
| liao-rh-ch-amidation-6d81a08f620d | 6d81a08f620d59353eae56f50b4dac01e2e77b171bda7585d3059be787d1461b | 7b9e22069ddb458b5d8b9d6f0a1691ded11d581b38dc075ce6a972a1757e09a0 | 47523b54a2dd5c75103ef3de4e91514c7f4d71e2 | /home/wangzh685/桌面/ord-data/YONOD/result/pnecessity_utility_prepare_v1/data/liao_rh_ch_amidation.csv | {"group_column": "non_p_input_group_id", "max_group_fraction": 0.8, "protocol_label": "pnecessity_within_source_non_p_input", "strategy": "precomputed_column"} | {"n_repeats": 3, "n_splits": 5, "seed": 20260918} | docs/manifests/, docs/predictions/, docs/folds/, docs/metrics/ |

## 论文协议对齐状态

此处记录特征、切分与随机种子的运行证据；流程对齐不等同于论文数值复现，仍需核对数据和软件版本。

| protocol | status |
|---|---|
| 未声明论文协议 | 通用 benchmark；不可据此宣称论文流程对齐 |

## 切分审计

| split_id | repeat | fold | n_train | n_valid | n_train_groups | n_valid_groups | max_group_size | group_leakage |
|---|---|---|---|---|---|---|---|---|
| split-b74a2f8d8950 | 1 | 1 | 800 | 200 | 800 | 200 | 1 | False |
| split-b74a2f8d8950 | 1 | 2 | 800 | 200 | 800 | 200 | 1 | False |
| split-b74a2f8d8950 | 1 | 3 | 800 | 200 | 800 | 200 | 1 | False |
| split-b74a2f8d8950 | 1 | 4 | 800 | 200 | 800 | 200 | 1 | False |
| split-b74a2f8d8950 | 1 | 5 | 800 | 200 | 800 | 200 | 1 | False |
| split-b74a2f8d8950 | 2 | 1 | 800 | 200 | 800 | 200 | 1 | False |
| split-b74a2f8d8950 | 2 | 2 | 800 | 200 | 800 | 200 | 1 | False |
| split-b74a2f8d8950 | 2 | 3 | 800 | 200 | 800 | 200 | 1 | False |
| split-b74a2f8d8950 | 2 | 4 | 800 | 200 | 800 | 200 | 1 | False |
| split-b74a2f8d8950 | 2 | 5 | 800 | 200 | 800 | 200 | 1 | False |
| split-b74a2f8d8950 | 3 | 1 | 800 | 200 | 800 | 200 | 1 | False |
| split-b74a2f8d8950 | 3 | 2 | 800 | 200 | 800 | 200 | 1 | False |
| split-b74a2f8d8950 | 3 | 3 | 800 | 200 | 800 | 200 | 1 | False |
| split-b74a2f8d8950 | 3 | 4 | 800 | 200 | 800 | 200 | 1 | False |
| split-b74a2f8d8950 | 3 | 5 | 800 | 200 | 800 | 200 | 1 | False |

## 描述符预计算状态

建模任务只读取 `descriptors/*.npz`；失败描述符的模型任务被隔离跳过。

| descriptor | status | stage | artifact_path | n_total | n_valid | feature_dim | skipped_model_count | reason |
|---|---|---|---|---|---|---|---|---|
| mfp_lightgbm_full | computed | precompute | /home/wangzh685/桌面/ord-data/YONOD/result/pnecessity_utility_liao_rh_ch_amidation_mfp_lightgbm_full_v1/feature/mfp_lightgbm_full.npz | 1000 | 1000 | 3072 | 0 | 首次生成描述符文件 |

## 模型协议与严格排名守卫

strict benchmark 的比较/排名只纳入 `evaluation_protocol=manifest_outer_cv` 且 expected/valid fold 完整的组合；旧 `autogluon_internal_holdout` 只展示并明确排除。

| split_id | descriptor | model | evaluation_protocol | expected_folds | valid_metric_folds | is_complete | strict_rank_eligible | strict_rank_exclusion_reason | autogluon_time_limit | autogluon_presets | autogluon_num_cpus | autogluon_seed_policy | autogluon_version | model_artifact_cleanup |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| split-b74a2f8d8950 | mfp_lightgbm_full | lightgbm | manifest_outer_cv | 15 | 15 | True | True | 可进入 strict benchmark 比较/排名 | — | — | — | — | — | — |

## 性能矩阵与完成度

| split_id | evaluation_protocol | descriptor | model | r2 | r2_std | rmse | rmse_std | mae | mae_std | complete | expected_folds | valid_metric_folds |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| split-b74a2f8d8950 | manifest_outer_cv | mfp_lightgbm_full | lightgbm | 0.8332 | 0.0306 | 0.0817 | 0.0063 | 0.0580 | 0.0034 | True | 15 | 15 |

| run_id | config_hash | split_id | evaluation_protocol | descriptor | model | expected_folds | available_folds | valid_metric_folds | is_complete | missing_or_excluded_reason |
|---|---|---|---|---|---|---|---|---|---|---|
| liao-rh-ch-amidation-6d81a08f620d | 6d81a08f620d59353eae56f50b4dac01e2e77b171bda7585d3059be787d1461b | split-b74a2f8d8950 | manifest_outer_cv | mfp_lightgbm_full | lightgbm | 15 | 15 | 15 | True |  |

## 建模耗时与成本对比

时间口径：`total_model_time_s = total_train_time_s + total_predict_time_s`，均为该组合全部有效外部 CV fold 的累计值。描述符特征化与 CLI 端到端墙钟时间不计入柱状图；不完整、缺失或非法时间的组合保留状态，但不进入耗时排序。描述符和建模方法图是组合成本的两种汇总视图，不应与组合图相加，也不把共享特征化时间重复归因给模型。

### 描述符 × 建模方法组合

| run_id | config_hash | split_id | evaluation_protocol | descriptor | model | expected_folds | completed_folds | is_complete | time_status | time_status_detail | is_time_comparable | total_train_time_s | mean_train_time_s | median_train_time_s | max_train_time_s | total_predict_time_s | total_model_time_s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| liao-rh-ch-amidation-6d81a08f620d | 6d81a08f620d59353eae56f50b4dac01e2e77b171bda7585d3059be787d1461b | split-b74a2f8d8950 | manifest_outer_cv | mfp_lightgbm_full | lightgbm | 15 | 15 | True | complete_and_comparable |  | True | 23.0560 | 1.5371 | 1.5356 | 1.5633 | 0.0168 | 23.0727 |

![组合级建模耗时柱状图](../pictures/combination_modeling_time.png)

### 按描述符汇总

每根柱为该描述符下所有完整且时间可比较模型组合的累计时间。

| run_id | config_hash | split_id | evaluation_protocol | aggregation_dimension | item | expected_combinations | comparable_combinations | noncomparable_combinations | is_time_comparable | time_status | time_status_detail | total_train_time_s | total_predict_time_s | total_model_time_s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| liao-rh-ch-amidation-6d81a08f620d | 6d81a08f620d59353eae56f50b4dac01e2e77b171bda7585d3059be787d1461b | split-b74a2f8d8950 | manifest_outer_cv | descriptor | mfp_lightgbm_full | 1 | 1 | 0 | True | complete_and_comparable |  | 23.0560 | 0.0168 | 23.0727 |

![描述符累计建模耗时柱状图](../pictures/descriptor_modeling_time.png)

### 按建模方法汇总

每根柱为该建模方法在所有描述符下完整且时间可比较组合的累计时间。

| run_id | config_hash | split_id | evaluation_protocol | aggregation_dimension | item | expected_combinations | comparable_combinations | noncomparable_combinations | is_time_comparable | time_status | time_status_detail | total_train_time_s | total_predict_time_s | total_model_time_s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| liao-rh-ch-amidation-6d81a08f620d | 6d81a08f620d59353eae56f50b4dac01e2e77b171bda7585d3059be787d1461b | split-b74a2f8d8950 | manifest_outer_cv | model | lightgbm | 1 | 1 | 0 | True | complete_and_comparable |  | 23.0560 | 0.0168 | 23.0727 |

![建模方法累计建模耗时柱状图](../pictures/model_modeling_time.png)

## 预测与稳定性

![fold_metric_stability.png](../pictures/fold_metric_stability.png)
![prediction_residuals.png](../pictures/prediction_residuals.png)

## 双维统计比较

### 模型维度统计比较（固定描述符）

- 没有可比较的完整组合；这不是性能排名结论。

无可展示记录；请查看相应排除/完整性表。

## 描述符维度统计比较（固定模型）

- 没有可比较的完整组合；这不是性能排名结论。

无可展示记录；请查看相应排除/完整性表。

## Tukey HSD 多重比较

### 模型维度

无可展示记录；请查看相应排除/完整性表。

### 描述符维度

无可展示记录；请查看相应排除/完整性表。

## 成本、任务状态与失败

| run_dir | disk_bytes | all_valid_fold_cumulative_train_time_s | all_valid_fold_cumulative_predict_time_s | all_valid_fold_cumulative_model_time_s | time_comparable_combinations | time_noncomparable_combinations | metric_exclusions | comparison_exclusions |
|---|---|---|---|---|---|---|---|---|
| /home/wangzh685/桌面/ord-data/YONOD/result/pnecessity_utility_liao_rh_ch_amidation_mfp_lightgbm_full_v1 | 1206668 | 23.0560 | 0.0168 | 23.0727 | 1 | 0 | 0 | 0 |

| state_store | count |
|---|---|
| succeeded | 15 |

无可展示记录；请查看相应排除/完整性表。

## 统计限制

比较单位是 CV fold。折之间并非完全独立，p 值不是唯一证据；必须结合差值、bootstrap CI、稳定性图和缺失任务解读。`no_significant_difference` 不代表性能完全相同。Tukey HSD 与配对检验并列呈现，不可任选有利结果。
