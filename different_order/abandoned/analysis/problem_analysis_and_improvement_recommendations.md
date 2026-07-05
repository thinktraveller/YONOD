# different_order 问题分析报告与改进建议

## 1. 总体结论

本轮 `different_order` 实验结果无效，不能用于判断 Morgan 描述符或其他描述符的分子向量拼接顺序是否影响建模结果。

根本原因不是任务没有跑完，也不是 JSON 文件没有差异，而是 `main.py` 在执行 JSON 配置时，没有把 `descriptors[*].columns` 真正传入特征构建流程。因此，不同任务虽然配置了不同的列组合，但实际建模时都使用了同一组 8 个 SMILES 列，导致所有任务结果中的 `n_smiles_cols` 始终为 8。

问题主要出在 `main.py` 的 JSON 配置解释和建模调度逻辑，不是 `yonod` 底层描述符实现本身。

## 2. 审计证据

### 2.1 JSON 文件并非完全一致

对 `different_order/tasks` 下 32 个 JSON 文件的审查结果显示：

- 32 个 JSON 的原始文件哈希均不同。
- `project_name` 有 32 种。
- `descriptors[*].columns` 有 32 种，确实随任务名变化。
- 去掉 `project_name` 和 `descriptors[*].columns` 后，32 个 JSON 完全一致。
- `column_roles` 在 32 个 JSON 中完全一致，只有 1 种。

审计结果已保存至：

- `different_order/analysis/tasks_json_audit.csv`

这说明：任务配置文件表面上确实不同，但真正被当前建模入口用于确定 SMILES 列的 `column_roles` 没有随任务变化。

### 2.2 所有任务的 `n_smiles_cols` 不变

根据以下审计文件：

- `different_order/analysis/config_effect_audit.csv`
- `different_order/analysis/metric_variation_by_descriptor_model.csv`

可确认：

- 所有任务实际报告的 `n_smiles_cols` 都是 8。
- 所有 descriptor/model 组合的 `unique_n_smiles_cols` 都是 8。
- 任务名对应的配置列数从 2 到 7 不等，但结果文件没有反映这种变化。

例如：

| 任务 | 配置列数预期 | 实际 `n_smiles_cols` |
| --- | ---: | ---: |
| `12` | 2 | 8 |
| `123` | 3 | 8 |
| `1234567` | 7 | 8 |

如果任务配置真正参与了建模，`n_smiles_cols` 应该随任务列数变化，而不是始终固定为 8。

### 2.3 `feature_dim` 未按配置列数变化

以 Morgan 为例，单列 Morgan 指纹维度为 1024。如果建模脚本正确使用 `descriptors[*].columns`，不同任务的 Morgan 特征维度应随列数变化：

| 任务 | 预期 Morgan 维度 | 实际 Morgan `feature_dim` |
| --- | ---: | ---: |
| `12` | `2 × 1024 = 2048` | 8192 |
| `123` | `3 × 1024 = 3072` | 8192 |
| `1234567` | `7 × 1024 = 7168` | 8192 |

实际所有任务的 Morgan `feature_dim` 均为 8192，即 `8 × 1024`。这表明建模实际使用的是固定的 8 个 SMILES 列，而不是任务配置中的列组合。

### 2.4 Morgan 结果完全相同

Morgan 是本实验重点关注的描述符。审计发现 Morgan 在 32 个任务中的 4 个模型指标完全一致：

| descriptor | model | 任务数 | 唯一指标签名数 | 是否完全相同 |
| --- | --- | ---: | ---: | --- |
| morgan | xgb | 32 | 1 | 是 |
| morgan | rf | 32 | 1 | 是 |
| morgan | svm | 32 | 1 | 是 |
| morgan | autogluon | 32 | 1 | 是 |

这与“不同列组合或不同拼接顺序参与建模”的实验预期不一致。

### 2.5 其他描述符的结果情况

其他描述符并非全部完全相同，但这些差异不能解释为列组合或拼接顺序导致，因为所有 descriptor/model 组合的 `n_smiles_cols` 都固定为 8，`feature_dim` 也不随任务配置变化。

已观察到：

- `maccs`：4 个模型在 32 个任务上完全相同。
- `maf`：`xgb/rf/svm` 完全相同，`autogluon` 有极小波动。
- `molmetalm`：`xgb/rf/svm` 完全相同，`autogluon` 有小幅波动。
- `rdkit2d`：`xgb/rf/svm` 完全相同，`autogluon` 有极小波动。
- `fisd`：4 个模型都存在变化，其中 `fisd + autogluon` 波动最大。

这些波动更可能来自模型训练或特征计算中的非确定性，而不是任务配置的列组合生效。

## 3. JSON 执行路径分析

当前 JSON 执行链路如下。

### 3.1 `--config` 被映射为 `--json`

用户运行：

```bash
python main.py --config xxx_yonod_config.json --csv ...
```

在 `main.py` 中，`--config` 会被兼容映射为 `--json`：

```python
if args.config is not None and args.json is None:
    args.json = args.config
```

### 3.2 `load_config_from_json()` 读取 JSON

`load_config_from_json()` 会读取 JSON，并验证以下字段：

- `project_name`
- `column_roles`
- `descriptors`
- `models`

该步骤只确认字段存在和数据集路径存在，并不会把 `descriptors[*].columns` 转换为实际建模列。

### 3.3 `config_to_args()` 构建全局 `smiles_cols`

`config_to_args()` 从 `column_roles` 构建全局 SMILES 列：

```python
reactant_cols = column_roles.get('reactants', [])
product_cols = column_roles.get('products', [])
other_cols = column_roles.get('others', [])
smiles_cols = reactant_cols + product_cols + other_cols
```

由于 32 个 JSON 的 `column_roles` 完全一致，所以这里得到的 `smiles_cols` 也完全一致。

### 3.4 `descriptors[*].columns` 被丢弃

同一个函数虽然读取了 `descriptors`，但只保留 descriptor 名称：

```python
descriptor_configs = config.get('descriptors', [])
descriptors = [cfg['descriptor'] for cfg in descriptor_configs]
```

这里丢弃了：

- `descriptors[*].columns`
- `descriptors[*].mode`

这一步是当前问题的核心。

### 3.5 建模循环使用同一组 `smiles_cols`

后续建模循环中，每个 descriptor 都使用同一个 `smiles_cols`：

```python
for desc_name in args.descriptors:
    X_smiles, X_numeric, mask = build_universal_features(
        smiles_cols=smiles_cols,
        numeric_cols=numeric_cols,
        df=df,
        desc_name=desc_name,
        smiles_roles=smiles_roles,
    )
```

因此，不同任务的 `descriptors[*].columns` 不会影响实际特征构建。

### 3.6 报告展示造成误导

在报告生成阶段，`main.py` 会把原始配置中的 `descriptors` 放入 `task_info`：

```python
task_info["descriptors"] = args._config.get("descriptors", [])
```

`yonod/universal/report.py` 会展示 descriptor 的 `columns`。这会让报告看起来像是不同 descriptor 使用了不同配置列，但实际上这些列没有参与建模。

## 4. 问题定位

### 4.1 主要问题在 `main.py`

直接责任点在 `main.py` 的 JSON 配置解释层和建模调度层：

- `config_to_args()` 没有保留完整 descriptor config。
- 建模循环只遍历 descriptor 名称。
- `build_universal_features()` 只收到全局 `smiles_cols`。
- 指标中的 `n_smiles_cols` 也是全局 `smiles_cols` 的长度。

因此，任务配置中的 `descriptors[*].columns` 没有进入实际建模路径。

### 4.2 `yonod` 底层不是直接责任点

`yonod/universal/feature_builder.py` 当前按传入的 `smiles_cols` 构建特征。它的接口说明也明确指出：

```text
smiles_cols: SMILES 列名列表，顺序决定拼接顺序。
```

也就是说，`feature_builder` 本身没有“忽略 JSON”。它根本没有收到 JSON 中每个 descriptor 的 `columns` 配置。

更准确地说：

- `yonod` 底层特征构建逻辑按传入列执行。
- `main.py` 没有把 JSON 中的 descriptor columns 转换成传入列。
- 因此问题主要发生在 `main.py` 和 `yonod` 之间的调度接口层。

### 4.3 列名体系还存在不一致

即使修复 `main.py`，也不能简单地把 `descriptors[*].columns` 直接传给 `feature_builder`，因为 JSON 配置列名与标准化 CSV 实际列名不一致。

JSON 中使用：

```text
reactant-amide
reactant-acid
product
activation
additive
base
solvent
```

标准化 CSV / `column_roles` 实际使用：

```text
reactant-1
reactant-2
activation-1
activation-2
additive
base
solvent
product
```

特别是 `activation` 在 JSON 中是单个逻辑列，但标准化 CSV 中对应 `activation-1` 和 `activation-2` 两个实际列。修复时必须明确这个映射规则。

## 5. 结果有效性判断

### 5.1 本轮结果可以说明什么

本轮结果可以说明：

- 32 个任务都成功完成了建模流程。
- 每个任务都生成了完整的指标文件。
- 共汇总 768 行指标。
- 每个任务均包含 24 条结果，即 6 个 descriptor × 4 个 model。
- 未发现缺失的 descriptor/model 组合。
- 在当前错误建模逻辑下，多数 descriptor/model 组合结果完全一致，少数组合存在非确定性波动。

### 5.2 本轮结果不能说明什么

本轮结果不能说明：

- Morgan 描述符的拼接顺序是否影响结果。
- 不同分子列组合是否影响结果。
- 其他 descriptor 是否受列顺序或列组合影响。
- `descriptors[*].columns` 配置是否合理。
- `mode=concat` 或 `mode=sum` 是否对建模产生影响。

因此，本轮实验结果应判定为无效，不应进入正式结果分析或论文、汇报、项目结论。

## 6. 改进建议

### 6.1 修复 `main.py` 的配置解释逻辑

`config_to_args()` 不应只保留 descriptor 名称，而应保留完整 descriptor 配置。

建议新增或保留：

```python
descriptor_configs = config.get("descriptors", [])
```

后续建模循环应遍历 descriptor config，而不是只遍历 descriptor name。

### 6.2 每个 descriptor 使用自己的 `columns`

当前逻辑是：

```python
for desc_name in args.descriptors:
    build_universal_features(smiles_cols=smiles_cols, ...)
```

建议改为类似：

```python
for desc_cfg in args.descriptor_configs:
    desc_name = desc_cfg["descriptor"]
    configured_columns = desc_cfg.get("columns", [])
    actual_columns = resolve_descriptor_columns(configured_columns)
    build_universal_features(smiles_cols=actual_columns, ...)
```

这样 `n_smiles_cols` 和 `feature_dim` 才会随任务变化。

### 6.3 增加配置列名到实际 CSV 列名的映射

需要建立明确的列名映射规则。例如：

| JSON 配置列 | 实际 CSV 列 |
| --- | --- |
| `reactant-amide` | `reactant-1` |
| `reactant-acid` | `reactant-2` |
| `product` | `product` |
| `activation` | `activation-1`, `activation-2` |
| `additive` | `additive` |
| `base` | `base` |
| `solvent` | `solvent` |

其中 `activation` 对应两个实际列，这是必须明确的设计点。建议在代码中把这种一对多映射显式实现，并在日志中打印映射后的实际列。

### 6.4 明确或实现 `mode`

当前 JSON 中存在：

```json
"mode": "concat"
"mode": "sum"
```

但现有建模路径没有真正使用 `mode` 控制拼接或求和逻辑。

建议二选一：

1. 如果 `mode` 是有效配置，则在特征构建中实现 `concat` 和 `sum` 的行为差异。
2. 如果当前阶段暂不支持 `mode`，则应在配置和报告中标注为未生效，避免误导。

### 6.5 增加运行前配置校验

在正式建模前，建议增加 dry-run 或 audit 模式，只解析配置，不训练模型。

每个 descriptor 至少输出并校验：

- 当前任务名
- descriptor 名称
- JSON 配置列
- 映射后的实际 CSV 列
- 实际 `n_smiles_cols`
- 预期 `feature_dim`

如果配置列数和实际列数不一致，或配置列无法映射到 CSV 列，应直接报错，不应继续运行。

### 6.6 修复后先做最小验证

修复后不要立即重跑全部任务。建议先选择 3 个任务验证：

- `12`
- `123`
- `1234567`

并优先只跑 Morgan 描述符，检查：

- `12` 的 Morgan `feature_dim` 是否符合配置列数预期。
- `123` 的 Morgan `feature_dim` 是否符合配置列数预期。
- `1234567` 的 Morgan `feature_dim` 是否符合映射后的实际列数。
- 三者的 `n_smiles_cols` 是否不同。
- 日志中打印的实际使用列是否符合任务配置。

通过后再批量重跑。

## 7. 推荐实施顺序

建议按以下顺序修复和验证：

1. 修改 `main.py`，保留完整 descriptor config。
2. 增加 descriptor columns 到实际 CSV 列名的解析函数。
3. 修改建模循环，使每个 descriptor 使用自己的实际列。
4. 增加运行日志，输出每个 descriptor 实际使用列。
5. 增加 dry-run/audit 模式，先验证配置解析结果。
6. 使用 `12`、`123`、`1234567` 三个任务只重跑 Morgan。
7. 检查 `n_smiles_cols` 和 `feature_dim` 是否随任务变化。
8. 验证通过后，再重跑全部 Morgan 任务。
9. Morgan 确认无误后，再决定是否重跑其他 descriptor。

## 8. 附录：相关审计文件

本报告依据以下文件生成：

- `different_order/analysis/tasks_json_audit.csv`
- `different_order/analysis/config_effect_audit.csv`
- `different_order/analysis/metric_variation_by_descriptor_model.csv`
- `different_order/analysis/reliability_check.md`
- `different_order/analysis/descriptor_identity_audit.md`
- `different_order/analysis/invalid_results_summary.md`

最终判断：本次问题的直接责任点在 `main.py` 的配置执行逻辑；`yonod` 底层特征构建代码目前只是按传入列执行，未收到正确的任务列配置。本轮实验结果应整体判定为无效，需要修复建模配置执行逻辑后重新运行。
