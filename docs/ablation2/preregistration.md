# Ablation2 预注册 v0.2（正式结果前）

日期：2026-09-18。v0.1 在任何正式模型任务启动前冻结；本 v0.2 仅记录预先规定的 2-6 准入条件已经满足，仍在第一项正式模型启动前冻结。微型验收只检查契约、资源和完整性，不用于选择性能结论。

## 问题与边界

主问题是：已知产物 P 时，P 的结构表征是否会降低 A 或 B 的条件预测贡献；这种模式在随机、未见 A、未见 B 和未见 A+B 对留出时是否仍成立。结果只解释为给定信息可用性设定下的模型预测贡献，绝不解释为化学因果效应、互信息或溶剂普遍重要性。

所有组合采用同一标准化 population（SHA `5fae4b314ffdb4d3…`），相同来源 SHA、相同 sample ID 和同一协议内的外部 manifest。C 的拼接顺序固定为 `activation_smiles, additive_smiles, base_smiles, solvent_smiles`。

| 标识 | 分子输入列（固定顺序） |
| --- | --- |
| `full` | A+B+P+C |
| `minus_a` | B+P+C |
| `minus_b` | A+P+C |
| `minus_p` | A+B+C |
| `minus_a_p` | B+C |
| `minus_b_p` | A+C |
| `product_conditions` | P+C |
| `conditions_only` | C |

## 评估与模型

四套预先生成的 5 folds × 3 repeats 评价：`random_repeat_group`、`unseen_amine`、`unseen_acid`、`unseen_substrate_pair`。随机协议只描述数据集内性能；pair 留出不等同于两个底物都未见。每折的确切 sample IDs 与 group-intersection 审计是配对依据，不能以共享 seed 代替。

首要模型/描述符为 binary Morgan ECFP4（radius 2、1024 bits、concat；当前公开 provider 的固定默认）× RF（300 trees、`max_features=1.0`、`min_samples_leaf=1`、random state 20260918、`n_jobs=19`）。预先指定的稳健性组合为 Morgan count MFP（radius 3、1024 bins、standard profile、concat）× LightGBM（500 trees、learning rate 0.05、31 leaves、`min_child_samples=1`、random state 20260918、`n_jobs=19`）。第二组合只会在 2-6 确认全八块、四协议和资源契约均可执行后进入正式矩阵；其是否纳入不依赖任何正式性能分数。

## v0.2 微型准入记录（不含正式性能）

在任何正式任务启动前，八个合成、contract-only 微型任务均由 `yonod.py` 串行完成：Morgan × RF 和 MFP × LightGBM 各覆盖随机重复组、未见胺、未见酸与未见底物对四种协议。每项均产生 8 个预注册输入块 × 2 个外折的 16 份预测和 16 份折元数据，并生成 HTML/Markdown 报告；外部 manifest 的来源路径/SHA256、任务本地 run ID、逐折组隔离与模型 `n_jobs: 19` 均已核验。此记录仅满足预先定义的第二组合资源/契约门槛；没有查看或用任何正式分数决定 MFP × LightGBM 是否进入正式矩阵。因此，两个预先指定组合均保留，正式参数不再因微型结果调整。

主指标为逐样本配对 MAE；RMSE、R²、Kendall τ 为辅助。外层测试折不会用于数值预处理拟合、早停或调参。AutoGluon 不属于本轮预注册矩阵。

## 假设、效应和统计

令同一测试单元的 MAE 为 `L`。A 的预注册替代效应是 `[L(B+C)-L(A+B+C)]-[L(B+P+C)-L(A+B+P+C)]`；B 对称。正值仅与“P 使被删底物的条件预测贡献降低”一致，不能单独证明替代机制。

正式汇总从逐样本预测重算指标和贡献差。置信区间以 repeat 为最外层相关单元、在 repeat 内按造成留出的 group 重采样；不把折或不同模型当成独立化学重复。主比较共四个（A/B × 有/无 P），Holm 校正；报告 95% 区间、原始和校正 p 值以及预先定义的实际 MAE 阈值 0.01（yield fraction）。若区间跨零或未跨该阈值，结论是未决而非“等效”。

## 停止、失败与修订

- 任一组合如无法在共享 population/split 上完成，保留 failed manifest；不得静默换样本或从历史指标补数。
- 指标在常数标签折不可定义时，原样列入 exclusions；不降级为普通 KFold。
- 仅在正式启动前、且有资源/兼容性证据时才能修订预算；修订创建新的 pre-registration 版本和新任务名。
- 第二数据集只在 2-9 的可复查试点结论后，按独立来源、产率标签和角色可得性单独预注册；不把同源派生 CSV 当外部验证。
