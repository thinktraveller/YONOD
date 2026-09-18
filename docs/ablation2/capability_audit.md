# Ablation2 平台能力与执行契约审计

审计日期：2026-09-18。范围是本轮酰胺偶联组分消融；历史 `result/ablation_experiments/` 未作为输入或证据。

| 需求 | 审计结论与源码位置 | 本轮处置 / 验收 |
| --- | --- | --- |
| A、B、P、C 的显式输入角色 | 原 `yonod/benchmark/config.py` 只允许描述符列位于 `reactants`，会迫使 P/C 伪装为反应物。`yonod/pipeline/features.py` 已能处理 `reactants/products/others`。 | strict adapter 现把三类分子角色的并集作为可选列，且 `contracts.py` 拒绝同一分子列跨角色重复声明。A/B 为 `reactants`，P 为 `products`，C 为 `others`；数值 `conditions` 仍不被误作 SMILES。 |
| 合法的任务输出目录 | 原 strict adapter 要求 `artifacts.output_dir == outputs.root`，与 schema-2 的 `outputs.root/feature` 约定冲突；其内容派生 run-id 又使同一任务分散到两个目录。 | strict adapter 现要求并实际使用 `outputs.root/feature`；整个 strict task 写在唯一的 `outputs.root`。不匹配路径在昂贵计算前失败。 |
| `config/` 路径 | `resolve_config_path` 原先总相对 YAML 文件，使 `config/foo.yaml` 的 `./dataset` 解析成 `config/dataset`。 | 位于仓库 `config/` 树的 YAML 统一按仓库根解析；其它外部/临时 YAML 保持历史 source-relative 行为。公开 `yonod.py` 路由 smoke 已验证。 |
| 共同外层划分 | 普通 `pipeline/training.py::_split_rows` 无论 `manifest_outer_cv` 字段均调用 `KFold`，不能作为本轮严格路径。strict adapter 原先也拒绝 `evaluation.split_manifest`。 | 普通训练仍明确受限、不得用于本研究。strict runner 现导入外部 manifest，检查 dataset SHA、唯一 source run/split ID、每折完整 sample population 和 group leakage；只将 task-local `run_id` 改写后落盘，并保留 source run ID。 |
| 随机重复与组分留出 | `yonod/splits/grouping.py` 有 component/scaffold/cluster，但没有可复用的准备阶段结构键。 | 新增 `precomputed_column`，只消费准备阶段的审计键，不重新把其解析为 SMILES。四套 manifest 均由同一 population 创建并有每折 ID/交集审计。 |
| 折级预测、指标及报告 | strict `executor.py` 逐折保存 sample-level parquet 预测和 JSON 元数据；`metrics.py` 重建 MAE/RMSE/R²，报告模块生成 HTML/Markdown。 | 所有正式任务只走 strict runner；结果身份含配置、数据、外部 split 内容 SHA、特征和模型参数。 |
| AutoGluon | 外层 fold adapter 存在，但其内部选择/随机性边界需要单独审计，且本轮首轮不需要它。 | 不纳入本轮预注册矩阵；不把历史 AutoGluon 数字作为严格 CV。 |
| 公开 strict 启动 | `yonod.py` 导入 `_verify.run_benchmark`；该模块是唯一的 strict manifest-outer-CV 编排器。 | `_verify/test_ablation2_strict_launcher.py` 在不拟合模型的 mock 下验证 `yonod.py → _verify.run_benchmark.run_benchmark` 的实际分派。 |

## 明确限制

- Windows 路径兼容由 `pathlib` 保持，但本轮未在 Windows 实测，状态为 `not_verified`。
- 非严格 `stage: all/train` 的 `manifest_outer_cv` 声明不能替代 strict manifest 外层折，故不用于 2-7 至 2-9。
- 条件空白被保留为来源层面的 `blank_source_value`；它不是已确认的“未添加”化学对照。
- strict configuration 目前固定特征子目录名为 `feature`。这是显式拒绝而非忽略用户的另一合法-looking路径，避免产物错位。

## 验收计划

1. 用 `_verify/test_ablation2_strict_launcher.py` 验证 YAML path、角色和 `yonod.py` 分派。
2. 用 `_verify/create_ablation2_splits.py` 产生四套 5×3 split；未知 ID、SHA 不同和 group leakage 作为外部导入负例。
3. 用 2-6 的合成微型数据经 `yonod.py` 产生实际预测、fold metadata 与双报告；检查 8 个输入块、四协议和 19-CPU 配置。
