# descriptor 结果一致性审计

基于 `metric_variation_by_descriptor_model.csv`，对 32 个任务在每个 `descriptor + model` 组合上的指标一致性进行审计。

## 总体结论

不是所有描述符的结果都完全相同。

但所有 descriptor/model 组合的 `unique_n_smiles_cols` 都是 8，`unique_feature_dims` 也都固定不变。这说明当前任务间差异并不是来自配置中的 `descriptors[*].columns` 被正确应用后产生的特征维度变化。对于出现轻微差异的组合，更合理的解释是模型训练或特征计算中的非确定性，而不是列组合或拼接顺序造成的差异。

## 分 descriptor 结论

| descriptor | 是否所有模型都完全相同 | 说明 |
| --- | --- | --- |
| morgan | 是 | 4 个模型在 32 个任务上指标完全一致。 |
| maccs | 是 | 4 个模型在 32 个任务上指标完全一致或仅有浮点零级误差。 |
| maf | 否 | xgb/rf/svm 完全一致，autogluon 有极小波动，`r2_range=0.000129746604978`。 |
| molmetalm | 否 | xgb/rf/svm 完全一致，autogluon 有小幅波动，`r2_range=0.000612930819411`。 |
| rdkit2d | 否 | xgb/rf/svm 完全一致，autogluon 有极小波动，`r2_range=0.0000621288670751`。 |
| fisd | 否 | 4 个模型都存在变化，其中 autogluon 波动最大，`r2_range=0.0189122007798`。 |

## 关键证据

### 完全相同的组合

- `morgan + autogluon/rf/svm/xgb`
- `maccs + autogluon/rf/svm/xgb`
- `maf + rf/svm/xgb`
- `molmetalm + rf/svm/xgb`
- `rdkit2d + rf/svm/xgb`

### 存在变化的组合

- `fisd + autogluon`: `r2_range=0.0189122007798`
- `fisd + xgb`: `r2_range=0.00432869320141`
- `fisd + rf`: `r2_range=0.00133126811479`
- `fisd + svm`: `r2_range=5.3100811942e-11`
- `maf + autogluon`: `r2_range=0.000129746604978`
- `molmetalm + autogluon`: `r2_range=0.000612930819411`
- `rdkit2d + autogluon`: `r2_range=0.0000621288670751`

## 可靠性判断

当前结果可以用于确认：

- 32 个任务都成功产出指标。
- 大多数 descriptor/model 组合在不同任务间完全相同。
- 少数组合存在小幅波动。

当前结果不适合用于确认：

- 某个 descriptor 是否受列组合影响。
- Morgan 或其他 descriptor 是否受拼接顺序影响。

原因是 `config_effect_audit.csv` 已显示所有任务实际报告的 `n_smiles_cols=8`，并且所有行的 `reported_feature_dim` 都不匹配“按配置 columns 建模”的预期维度。
