# ChemRxiv 冻结外测统计综合 v1

## 输入与方法

- 已对 12/12 个冻结 development→external 任务再次运行逐项完整性核验；统计只读取其保存的 prediction parquet，未训练、覆盖或新增模型。
- 每项任务均为 47,015 行 development population 单次拟合后，对同一 957 行 ChemRxiv external population 的一次预测；它不是交叉验证，逐任务分数也不是模型重复。
- 推断单位是预注册的 10 个 `external_group_id` 胺结构。每个胺组先计算组内平均绝对误差差，再按组等权重汇总。95% 区间为固定随机种子、10000 次 percentile cluster bootstrap（重采样 10 个胺组）。
- 两个预注册的 A 边际对比使用双侧组级 Wilcoxon signed-rank，并在每一条模型线的 frozen-external track 内对这两个 p 值作 Holm 校正。派生的 Δsub,A 只报告预注册 bootstrap 区间，不额外扩增 p 值家族。

## 描述性外测指标

| line_label | combo | input_blocks | mae | rmse | r2 | kendall_tau |
| --- | --- | --- | --- | --- | --- | --- |
| Morgan ECFP4 × RF | full | A+B+P+C | 0.1966 | 0.2512 | -0.6984 | 0.0547 |
| Morgan ECFP4 × RF | minus_a | B+P+C | 0.1585 | 0.2030 | -0.1098 | 0.0313 |
| Morgan ECFP4 × RF | minus_p | A+B+C | 0.1974 | 0.2579 | -0.7903 | 0.0358 |
| Morgan ECFP4 × RF | minus_a_p | B+C | 0.2360 | 0.2976 | -1.3841 | -0.0188 |
| Morgan ECFP4 × RF | product_conditions | P+C | 0.1636 | 0.2118 | -0.2080 | 0.0310 |
| Morgan ECFP4 × RF | conditions_only | C only | 0.2204 | 0.2798 | -1.1072 | -0.0031 |
| MFP × LightGBM | full | A+B+P+C | 0.1969 | 0.2567 | -0.7741 | 0.0397 |
| MFP × LightGBM | minus_a | B+P+C | 0.1828 | 0.2363 | -0.5038 | 0.0828 |
| MFP × LightGBM | minus_p | A+B+C | 0.2169 | 0.2832 | -1.1590 | -0.0467 |
| MFP × LightGBM | minus_a_p | B+C | 0.2364 | 0.3039 | -1.4862 | -0.0281 |
| MFP × LightGBM | product_conditions | P+C | 0.1680 | 0.2315 | -0.4430 | 0.0414 |
| MFP × LightGBM | conditions_only | C only | 0.2257 | 0.2925 | -1.3040 | 0.0346 |

## A 组分的配对统计

正值表示移除 A 会提高 MAE；Δsub,A 为“无 P 时移除 A 的代价”减去“有 P 时移除 A 的代价”。其为条件预测信息比较，不能解释为化学因果或互信息。

| line_label | contrast | effect_group_mean | ci_low | ci_high | p_raw | p_holm | rank_biserial | conclusion |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Morgan ECFP4 × RF | a_with_product | -0.0382 | -0.0695 | -0.0065 | 0.0488 | 0.0977 | -0.7091 | marginal A information comparison; interpret with its interval and Holm-adjusted test |
| Morgan ECFP4 × RF | a_without_product | 0.0386 | -0.0049 | 0.0824 | 0.1602 | 0.1602 | 0.5273 | marginal A information comparison; interpret with its interval and Holm-adjusted test |
| Morgan ECFP4 × RF | a_product_substitution | 0.0768 | 0.0334 | 0.1051 | — | — | — | supports the pre-specified conditional predictive substitution pattern in this external collection |
| MFP × LightGBM | a_with_product | -0.0142 | -0.0364 | 0.0072 | 0.2754 | 0.5508 | -0.4182 | marginal A information comparison; interpret with its interval and Holm-adjusted test |
| MFP × LightGBM | a_without_product | 0.0194 | -0.0260 | 0.0689 | 0.5566 | 0.5566 | 0.2364 | marginal A information comparison; interpret with its interval and Holm-adjusted test |
| MFP × LightGBM | a_product_substitution | 0.0336 | -0.0127 | 0.0752 | — | — | — | uncertain at the pre-specified practical MAE boundary |

## 受限解释

- 只有当 Δsub,A 的 95% 区间下界超过预注册的 0.01 MAE 阈值，才可在该外部集合中称为支持“P 减少 A 的条件预测贡献”的模式；所有其他情形均保留为不确定或方向相反。
- Morgan ECFP4 × RF 和 MFP × LightGBM 是预注册的两条模型线，用于稳健性检查，而不是相互独立的化学重复。
- 外部来源只有一种酸；结果不能声称未见酸泛化、跨反应家族普适性、因果替代或模型等效。非显著结果也不能证明没有效应。
- 这份综合不启动同源 grouped unseen-A 验证；该后续研究需单独决策，且不能与本冻结外测混为一谈。

## 可复算产物

- `tables/task_provenance.json`：12 项输入配置、预测哈希及完整性核验摘要。
- `tables/overall_external_metrics.csv`：从保存预测重算的 957 行描述性指标。
- `tables/group_mae.csv` 与 `tables/group_contrasts.csv`：十个胺组层面的配对源数据。
- `tables/statistical_summary.csv`：bootstrap、Wilcoxon 与 Holm 输出。
- `figures/`：外测 MAE 与 Δsub,A 图（PNG/PDF）。
