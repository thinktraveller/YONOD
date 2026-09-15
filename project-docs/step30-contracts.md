# 步骤 30.1：配置、产物与迁移契约

状态：30.1–30.7 的契约、YAML 层、产物读取、features/derive、模型工厂和 manifest-only train/all 已实施并有聚焦测试；`main.py --config`、`yonod.py` 向导、操作 CLI 和 strict benchmark 的 schema-2 `stage: benchmark` 公开路径已完成。锁定 Conda 环境的五模型 staged/all、OHE/PCA/数值缩放/inner early-stopping，以及经 `yonod.py` 启动的 strict benchmark 真实 fixture smoke 均已通过。JSON 运行配置、依赖 JSON 的启动器和 `different_order` 已明确排除；旧 benchmark 根、外部 split 导入和未经审计的 tree early stopping 均严格拒绝，而不作猜测性转换。

本文件是 `yonod.config.contracts` 和 `yonod.artifacts.contracts` 的人工可读补充。它不将尚未迁移的 JSON 运行入口误写成已支持 YAML，也不改变结果、审计或官方只读资料的 JSON 格式。

## 四类独立文件

| 文件 | 根字段/版本 | 用途 | 禁止混入的内容 |
| --- | --- | --- | --- |
| 运行配置 | `schema_version: "2.0"`、`stage` | `features`、`train`、`all` 的可执行意图 | 中间编辑步骤 |
| strict benchmark 运行配置 | `schema_version: "2.0"`、`stage: benchmark`、`benchmark.task_state` | manifest 外部折、grouping 与可恢复 SQLite state 的可执行意图 | 普通 `outer_kfold`、未经验证的外部 split、隐式 task state |
| 操作配置 | `schema_version: "2.0"`、`operation: derive_features` | 基于已有 manifest 生成新的特征版本 | 模型、模型参数和训练设置 |
| 产物 manifest | `schema_version: "2.0"`、`artifact_id`、`status` | 一个版本化特征包的内容、映射、哈希和血缘 | 生成器对象、绝对路径、训练结果 |

配置内的路径由配置文件位置解析；manifest 内所有文件路径必须是相对 manifest 的包内路径，且不允许 `..`。产物内容身份使用文件内容 SHA-256、矩阵/样本/有效性结构和来源身份；不使用 manifest 路径、创建时间或 YAML 的注释、键排序。

## 步骤 30.2：共享安全 YAML loader 与显式覆盖

`yonod.config.loader` 已提供独立的 `load_run_config()`、`load_operation_config()`、`load_yaml_mapping()`、`merge_config_layers()` 与 `build_explicit_cli_overrides()`。它们不接入当前 `main.py`、向导或批处理脚本；入口迁移仍严格留给步骤 30.8。

- 只接收 `.yaml` / `.yml` UTF-8 文件，`.json` 立即拒绝并提示一次性迁移器；空文档、多个文档、非 mapping 根节点、非字符串键、重复键、锚点别名和不安全 YAML tag 均拒绝。
- 对 `models` 与 `model_params` 仅规范化已登记的模型别名（`xgboost → xgb`、`random forest/random_forest → rf`、`light gbm/lgbm → lightgbm`、`auto_gluon → autogluon`）。两个原始字段归一为同一键时，错误会指出完整原始路径，如 `models[1]` 与 `models[0]`。
- 解析后先执行 30.1 的静态运行/操作契约，再合并任何库默认；因此默认值不能掩盖 YAML 缺少的阶段必填字段。合并后再次验证结果。
- 合并顺序固定为“库默认 → YAML 显式值 → 显式 CLI 覆盖”。未传入 `explicit_cli`（`MISSING`）时没有第三层；传入覆盖 mapping 时，`null`、`false`、`0` 都是明确值，不会被当成缺失。调用方必须按真实命令行 token 建立 `explicit_fields`，不能把 argparse 默认 Namespace 整体传入覆盖层。

这一步只实现安全、纯配置层：参数名对底层库签名的完整校验和库默认采集归 30.6；`main.py` 与向导的 YAML 接入已在 30.8 完成，其余历史入口迁移另列为未完成项。

## 运行阶段最小语义

| 阶段 | 必需 | 不要求 | 说明 |
| --- | --- | --- | --- |
| `features` | 数据路径、`sample_id_col`、输入列角色、静态/折内特征声明、`artifacts.output_dir` | 标签、模型、模型参数 | 只能建立/复用特征；不能构造预测模型。 |
| `train` | 数据路径、`sample_id_col`、标签角色、`artifacts.input_manifest`、模型 | 原始 SMILES 列、描述符算法参数 | 只读取已完成产物，稍后由读取层核验 ID 对齐。 |
| `all` | 上述两个阶段所需的字段 | — | 仅编排两个独立服务，不能凭进程内对象传递特征。 |
| `benchmark` | 含标签数据、描述符、模型、相同 `artifacts.output_dir`/`outputs.root`、`manifest_outer_cv`、grouping、SQLite state | 普通 outer CV、外部 split 猜测导入 | 只经 `yonod.py` 分派 strict runner；先发布 split manifest，再执行可恢复外部折。 |

普通 `train`/`all` 中，每个已经完成配置、特征、标签和 split identity 的 feature × model 都是独立结果边界。真实 fold 训练异常会原子发布到 `outputs.root/failed_runs/<run-id>-<attempt>/`：failed manifest 记录模型请求、身份、异常类型/原因和 source/effective YAML，且不写出半成品 predictions 或 fold metrics。其他组合继续；任何 failed 组合使公开 `main.py --config` 返回 2，不能把部分完成误报为成功。完整 fixture 证据见 `derived/interface_migration/step30_acceptance/training_failure_isolation_evidence.md`。

`model_params.<model>` 目前仅能放入已登记的接口区段。字段名与具体依赖版本的构造器/`fit` 参数校验属于步骤 30.6，不能把任意 YAML 键宣称为可传递的 `**kwargs`。

| 模型 | 可路由区段 | 目标接口（后续统一模型工厂） |
| --- | --- | --- |
| `rf` | `estimator`、`fit`、`runtime` | `sklearn.ensemble.RandomForestRegressor` 构造/拟合 |
| `xgb` | `estimator`、`fit`、`runtime` | `xgboost.XGBRegressor` 构造/拟合；设备策略在 runtime 解析 |
| `svm` | `estimator`、`fit`、`preprocessing`、`runtime` | `SVR` 与训练折内 scaler/PCA/抽样 |
| `lightgbm` | `estimator`、`fit`、`runtime` | `lightgbm.LGBMRegressor` 构造/拟合 |
| `autogluon` | `predictor`、`fit`、`runtime` | `TabularPredictor` 构造和 `.fit()`；标签名必须同时控制训练表 |

支持版本基线以 [requirements.txt](/home/wangzh685/桌面/ord-data/YONOD/requirements.txt) 锁定的 `scikit-learn==1.4.0`、`xgboost==2.0.3`、`PyYAML==6.0.1` 为准；AutoGluon 按文件注释的 `1.1.1` 可选安装，LightGBM 需在实际环境记录版本。步骤 30.6 必须在这些版本范围内建立完整参数目录和生效值审计。

## Manifest 的就绪产物结构

`ready` 和 `partial` manifest 必须包含：

- `files.matrix`、`files.sample_ids`、`files.validity_mask`：每项都包含相对 `path` 与小写 SHA-256；
- `matrix`：引用的文件、`dtype`、二维 `shape` 和列结构；
- `samples`：样本 ID 文件、ID 字段、总行数和矩阵行映射语义；
- `validity`：掩码文件、`n_total`、`n_valid` 和语义；矩阵行数必须等于有效行数；
- `sources.dataset_identity` 与 `sources.feature_config_identity`；
- 派生产物还必须有父 `artifact_id`、操作声明及非空的输入/输出映射。

`failed` manifest 必须给出 `failure.reason`，且不可被读取层当作就绪特征。

## 步骤 30.3：独立读取、原子发布与旧 NPZ 导入

`yonod.artifacts.reader` 是纯读取层：仅依赖标准库、NumPy 与共享安全 YAML reader，不导入 RDKit、Torch、描述符注册表或特征生成器。`load_feature_artifact()` 默认只接受 `status: ready`，返回只读矩阵、完整 `sample_ids`、完整 `valid_mask` 和由掩码计算出的有效行索引；`partial` 只有调用方显式 `allow_partial=True` 才可读取。

- 对 manifest 的每一个 `files.*` 条目，先验证相对包内路径、存在性、可选字节数和 SHA-256；符号链接解析到包外同样拒绝。
- schema-2 发布器使用三个无 pickle 的 `.npy` 实体文件（matrix、sample IDs、validity mask）。静态描述符 matrix 必须为数值数组；`fold_transform` 可保存未拟合的字符串原始类别矩阵。读取时核验实际 dtype/二维 shape、命名列结构、唯一非空字符串 ID、bool 掩码长度/有效数和 `valid_rows_in_sample_order` 行映射；任一不一致拒绝。
- `publish_feature_artifact()` 只发布不存在的新目标目录：先在同级临时目录写所有文件、生成 YAML manifest、通过公开读取 API 复验全部实体后，再原子重命名目录。失败清理临时目录，现有版本绝不覆盖。
- `import_legacy_npz()` 仅接收已知 schema-v1 的 `X_smiles/sample_ids/valid_mask/metadata_json` NPZ。调用者必须显式提供 artifact/feature ID、数据身份、特征配置身份和列结构；不会推测标签、辅助表或样本顺序，也不写回旧 NPZ。导入后的新包记录源文件名、SHA-256 与旧 schema 版本。

30.3 只提供“已落盘特征的读取/发布”基础，不会自动绑定标签或辅助表；训练时按样本 ID 的标签/辅助表连接和 split 身份校验留给步骤 30.7。

## 步骤 30.4：独立 features 服务与特征就绪状态

内部 API `yonod.pipeline.features.run_features(<features.yaml>)` 只接收 schema-2 `stage: features` 配置。它通过 30.2 loader 读取无标签、无模型配置，输出两类持久化引用：`artifacts.output_dir/features/<artifact-id>/manifest.yaml` 中的独立特征包，以及 `artifacts.output_dir/feature_runs/<run-id>.yaml` 的本次总状态。返回对象仅含这些路径、ID、状态和理由，不保留 DataFrame、闭包或训练对象。

- 每个静态候选惰性调用既有 `yonod.universal.feature_builder.build_universal_features` 生产路径；测试可注入轻量计算器，但服务本体不导入预测模型工厂、不会创建 `docs/`、`report/` 等训练输出目录。
- 特征 identity 由 CSV 内容 SHA-256、规范化 feature spec、解析后的输入列和角色决定。相同 identity 只会读取并验证已有 package 后标为 `reused`；目标存在但损坏或身份冲突时失败而不覆盖。输入/算法变化会指向新的 artifact ID。
- 所有候选都依次完成、复用或记录失败后才原子写入 run status：全成功/复用为 `ready`，成功与失败共存为 `partial`，全失败为 `failed`。一个候选失败不会删除或遮蔽其他就绪包。
- OHE 不进行全数据拟合；服务将声明的原始类别列和完整 sample ID 保存成 `lifecycle: fold_transform` 字符串 artifact，并记录 `fit_scope: training_fold_only`、参数、列和缺失哨兵。折内实际编码和状态保存留待训练阶段。

`main.py --config <schema-2.yaml>` 现已调度该 API；配置内 `stage: features` 是只计算公开入口。`stage: all` 仅由 `run_all()` 在完成全部特征终态后调用训练服务。

## 步骤 30.5：只读 inspect 与不可变 derive

内部 API `yonod.artifacts.operations.inspect_feature_artifact(<manifest.yaml>)` 只读取并完整验证 schema-2 包，返回 ID、矩阵结构、样本/有效数、来源身份、父 ID 和内容身份等不可变描述，不创建缓存或改写任何文件。`derive_features(<operations.yaml>)` 使用同一安全 YAML loader 和操作配置契约；公开入口为 `scripts/manage_features.py inspect|derive`。

操作 YAML 的最小形式如下；`protected_columns` 可选，用于将项目特有的标签/目标名加入默认受保护集合（`label`、`target`、`yield`、`y`）。

```yaml
schema_version: "2.0"
operation: derive_features
input_manifest: source/manifest.yaml
output_dir: derived
protected_columns: [experimental_yield]
operations:
  - kind: select_samples
    sample_ids: [sample-3, sample-1]
  - kind: select_features
    columns: [mfp_17, mfp_42]
  - kind: join_features
    table_path: side_features.csv
    sample_id_col: sample_id
    columns: [temperature_feature]
    prefix: side_
```

- `select_samples` 只接受唯一、存在的 sample ID，并严格按 YAML 声明的 ID 顺序输出；掩码和有效矩阵行随该顺序重新映射。`select_features` 只接受 `matrix.columns.kind: named` 的唯一已知列，且按声明列顺序输出。
- `join_features` 当前只接受 UTF-8 CSV 和显式列清单，要求外部 `sample_id_col` 唯一、非空，并与父 population 的每个 ID 一对一匹配。外部表可有额外 ID，但绝不扩充 population；其数量会写入输入/输出映射。外部行即使重排也按父 sample ID 顺序数值对齐。未知列、缺失/重复 ID、非数值/缺失值、输出列冲突、空前缀及受保护标签列均立即失败。
- 为避免把未拟合类别错误地写成普通数值特征，当前派生仅接受数值 `static_descriptor` 或数值 `derived` 包；`fold_transform` 原始类别仍只能由后续训练阶段按训练折拟合。
- 每个操作按顺序经原子发布器生成一个新的 `lifecycle: derived` 目录、artifact ID 和 manifest；现有同名目录一律拒绝覆盖。manifest 保存直接父 ID、操作声明及 identity、父内容 hash，以及哈希化的 sample/列/外部表输入输出映射。输出根若位于父包内同样拒绝，父包绝不修改。链中较早的成功版本在随后操作失败时仍保持可独立读取。

## 运行配置入口清单与迁移界限

| 当前位置 | 当前运行配置格式/用途 | 步骤 30.8 的迁移落点 | 验收方式 |
| --- | --- | --- | --- |
| `main.py` | schema-2 `--config`；`--json` 仅作拒绝提示 | 已调用共享 loader；作为底层配置运行时和 smoke 入口 | YAML 分阶段与五模型 smoke；JSON 被入口拒绝 |
| `yonod.py` | 已接收现有 schema-2 YAML，向导生成/验证 `*_run.yaml` | 已调用共享 loader；显式选择唯一 sample ID；正式建模任务的启动入口 | 向导输出由同一 loader 校验 |
| `scripts/run_benchmark.py`、`yonod/benchmark/config.py` | schema-2 `stage: benchmark`；public launch 由 `yonod.py` 分派 | 已实现严格 adapter，保存 grouping、manifest CV、feature lifecycle 与 SQLite state | schema adapter/拒绝测试和 12 行真实 fixture smoke |
| `yonod/benchmark/paper_exact*.py`、`scripts/run_yieldmaster_paper_exact*` | paper-exact JSON 运行配置 | **明确不迁移**；仅保留为历史/归档材料 | 不计入 schema-2 支持或验收缺口 |
| 依赖 JSON 的 `scripts/run_yieldmaster*` 启动器 | 批处理和历史静态矩阵资源配置 | **明确不迁移**；不得新增 JSON 兼容入口或猜测性 YAML 改写 | 不计入 schema-2 支持或验收缺口 |
| `different_order/amide-coupling/run_different_order_tasks.py` | 缺失目标 runner 的 compatibility wrapper | **明确不迁移**；不补建 runner | 不计入验收缺口 |

JSON 指标、结果元数据、官方资料、NPZ 内嵌 JSON 和已完成任务的审计文件不属于运行配置，保留其格式。不会继续迁移 JSON 运行配置或为其维护兼容性。旧 benchmark 根同样不是已迁移 YAML：无法证明逐行语义等价的字段（尤其外部 split）会被拒绝；用户须写出独立 schema-2 YAML，或先实现受审计的等价 importer。

## QA-R013 到工作包的映射

| 已确认决策 | 实现位置 |
| --- | --- |
| 计算/建模必须可独立启动 | 30.1 阶段字段与 manifest 契约；30.3/30.4/30.7 实现读取、features、train/all |
| 查看、筛选、拼接须另存版本 | 30.1 `derive_features` 专用操作配置和派生血缘字段；30.5 实现操作 |
| YAML 唯一运行配置，省略即库默认 | 30.1 路由契约；30.2 单一 YAML loader；30.6 模型工厂；30.8 删除 JSON 运行分支 |

## 本步骤可执行验证

`python3 -m unittest tests.test_step30_contracts tests.test_step30_yaml_loader tests.test_step30_artifact_reader tests.test_step30_feature_service tests.test_step30_artifact_operations -v` 覆盖普通配置、产物和派生契约。strict benchmark 另以 `tests.test_step30_benchmark_schema tests.test_repeated_kfold_manifest tests.test_ohe_fold_preprocessor tests.test_benchmark_autogluon_executor tests.test_benchmark_report_matplotlib_compat -v` 覆盖 schema-2 adapter、legacy/outer-CV/external-split/early-stop fail-closed、manifest、fold-local OHE、模型工厂与 Matplotlib 新旧标签 API；完整 fixture 证据见 `derived/interface_migration/step30_acceptance/benchmark_schema2_preflight_evidence.md`。
