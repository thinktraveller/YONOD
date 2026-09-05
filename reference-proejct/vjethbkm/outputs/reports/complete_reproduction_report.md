# VJETHBKM 完整复现工作报告

> 复现对象：Ismail, Landrum, Riniker, “Yield Smarter, Not Harder: Good Practices for Machine Learning of Reaction Outcomes”  
> Zotero 条目：`VJETHBKM`  
> 论文 DOI：`10.1021/jacs.6c02213`  
> 补充材料 DOI：`10.1021/jacs.6c02213.s001`  
> 官方数据与代码 DOI：`10.3929/ethz-c-000800856`  
> 报告日期：2026-08-25  
> 实验与结果产物基线提交：`ae503402`（本报告文本在该基线之后生成）

## 摘要

本项目在 `reference-proejct/vjethbkm/` 中建立了与 YONOD 主代码隔离的复现工作区，并对 VJETHBKM 论文的低成本描述符、随机森林、5×5 重复交叉验证、外部验证、组分留出结构和产率不平衡分析进行了分层复现。

核心对照覆盖 4 个数据集、3 类描述符和 4 个指标，共 `4 × 3 × 4 = 48` 个 Table S3 风格目标。所有 48 行均获得本地结果，但完成等级不同：`32` 项达到数值复现、`12` 项为接近官方值的 OHE 兼容复跑、`4` 项为 SM/OHE 未解差异。

- MFP 的 16 个指标与官方结果完全一致，最大绝对差为 `0.0`。
- PhysChem 的 16 个指标与官方结果一致，最大绝对差为 `1.776 × 10⁻¹⁵`，属于浮点尾差。这 32 项均依赖官方预计算特征，因此复现的是“官方特征输入后的 RF/CV/指标链”，不包含从原始分子结构重新生成 MFP/PhysChem 的过程。
- OHE 在 BH、BH2 和 SLAP 上与官方结果接近；最大绝对差分别为 `0.006902`、`0.007504` 和 `0.109521`。
- SM/OHE 的 4 个指标存在无法由官方包内信息解释的系统差异，最大绝对差为 `1.900650`。法证检查已经排除行顺序、标签、KFold 切分、数值精度以及是否加入 `catalyst_smiles` 等原因，因此该项被标记为“官方产物差异未解”，没有通过硬编码伪造一致。

除核心交叉验证外，本项目还完成了 BH2 外部预测文件审计、4 个数据集的 0D/1D/2D 留出结构审计，以及基于官方 OOF 预测的产率分桶和高产率识别分析。DFT、SOAP、其他传统模型、重加权实验、论文精确的组件留出训练和 BH2 本地外部重新预测尚未完成，因此本报告将这些内容明确列为未复现，而不把官方已有输出当成本地训练结果。

## 1. 复现目标与判定标准

本次工作的目标不是仅运行官方脚本，而是建立一条可审计的复现链：文献结论能够追溯到原始来源，数据和参数能够追溯到官方包，本地运行能够产生结构化结果，差异能够被量化，不能复现的部分保留证据边界。

采用以下状态词：

| 状态 | 含义 |
|---|---|
| 数值复现 | 给定同一官方预计算特征，切分、模型和指标定义对齐，结果与官方目标完全一致或仅有浮点尾差；不自动代表特征生成链也已复现。 |
| 兼容复跑 | 按已知官方逻辑重新实现并获得本地结果，但未逐字调用原脚本，允许存在小幅依赖或实现差异。 |
| 审计验证 | 对官方包中的数据、预测或结构进行重新计算和交叉检查，但没有重新训练相应模型。 |
| 未解差异 | 已完成针对性排查，仍缺少足够信息解释官方结果与本地结果的差异。 |
| 未复现 | 尚未执行本地训练或缺少论文精确定义、依赖、模型文件或计算资源。 |

## 2. 证据来源与复现边界

文献层证据来自三个层次：

1. 前序会话通过 Z-AIHub 阅读 Zotero 条目 `VJETHBKM` 的正文和补充材料。正文文档 ID 为 `doc_9c01f86b3093`，11 页、43 个分块；补充材料文档 ID 为 `doc_a270fcc24f77`，13 页、37 个分块。
2. ACS、Crossref 和 ETH Research Collection 的公开记录用于核对题名、作者、论文 DOI、补充材料 DOI、开放许可和数据代码 DOI。
3. 用户提供的官方 `yieldsmarter` 数据包用于核对数据、代码、配置、预计算特征、训练输出和许可证。

完整证据台账见 [`docs/literature-ledger.md`](../../docs/literature-ledger.md)。本地没有把论文 PDF 或大体积官方数据复制进 Git；原始包由 `.gitignore` 排除，只提交脚本、小型表格、manifest 和报告。

## 3. 官方数据包审计

官方包位于 `reference-proejct/vjethbkm/yieldsmarter/`，根目录说明文件实际名为 `README.txt`，并非 `README.md`。审计结果如下：

| 项目 | 结果 |
|---|---:|
| 文件数 | 904 |
| 总大小 | 351,207,591 bytes，约 351 MB |
| CSV | 147 |
| JSON | 287 |
| NPZ | 149 |
| Python 源文件 | 23 |
| XYZ 几何文件 | 287 |
| 许可证 | MIT License，Copyright © 2025 ETH Zurich |

关键正式数据集：

| 报告名称 | 官方文件 | 样本数 | 组分列数 | 目标范围 | 目标均值 |
|---|---|---:|---:|---:|---:|
| BH | `BH1/BH1.csv` | 3,955 | 4 | 0–99.99999 | 33.0853 |
| BH2 | `BH2/BH2.csv` | 3,359 | 5 | 0–101 | 26.2410 |
| SM | `SM/SM.csv` | 5,760 | 5 | 0–100 | 40.1095 |
| SLAP | `SL1/SL1.csv` | 1,150 | 3 | 0–630.473598 | 31.5147 |
| BH2 external | `187_with_corrected_catalyst_smiles.csv` | 187 | — | 0–100 | 43.4171 |

需要特别注意：SM 的 OHE 复跑使用 `SM.csv` 的 5,760 行；官方 MFP/PhysChem `.npz` 仅包含 4,620 行。因此，不同描述符下 SM 的高产率样本计数和性能不能被解释为同一测试总体上的严格横向对照。

包审计详情见 [`yieldsmarter_package_audit.json`](../../data/manifest/yieldsmarter_package_audit.json) 和 [`yieldsmarter_dataset_schema.csv`](../../data/manifest/yieldsmarter_dataset_schema.csv)。

## 4. 运行环境

本次验证使用隔离环境 `reference-proejct/vjethbkm/.venv`：

| 软件 | 版本 |
|---|---:|
| Python | 3.13.9 |
| pandas | 3.0.5 |
| NumPy | 2.5.2 |
| SciPy | 1.18.1 |
| scikit-learn | 1.9.0 |
| RDKit | 2026.3.5 |
| LightGBM | 4.7.0 |
| DScribe | 2.1.2 |
| ASE | 3.29.0 |
| pytest | 9.1.1 |

官方 `yieldsmarter.yml` 以 Python 3.10、RDKit 2025.03.6 等版本为基础。本地环境版本不同，但 MFP 和 PhysChem 仍逐指标复现，说明这些预计算特征路径对当前版本差异不敏感。OHE 的细小偏差则可能包含 scikit-learn 版本和编码细节影响；SM/OHE 的大偏差无法单靠版本差异得到证明。

环境检查时 Matplotlib 因用户目录写权限未能写入字体缓存，但检查脚本退出码为 0，且不影响本报告所用的表格计算与测试。

## 5. 完整复现工作流

| 阶段 | 输入 | 处理 | 主要输出 | 复现性质 |
|---|---|---|---|---|
| 1. 文献审计 | 正文、SI、公开元数据 | 提取数据集、描述符、模型、指标和验证协议 | 文献台账、图表目标表 | 证据整理 |
| 2. 包审计 | 官方 `yieldsmarter/` | 统计文件、校验许可证、生成 checksum 和 schema | package audit、schema、checksums | 审计验证 |
| 3. 参数对齐 | `Train_OHE.py`、`Train_descriptors.py`、JSON 配置 | 核对 OHE fit 范围、KFold、RF 参数和指标 | training params audit | 审计验证 |
| 4. Smoke 验证 | YONOD 10 行样本 | OHE/Morgan/PhysChem + RF 小规模运行 | smoke tables/reports | 工程验证 |
| 5. OHE 正式复跑 | 4 个官方 CSV | 每折训练集拟合 OHE，RF 5×5 CV | 4 组 OHE summary | 兼容复跑 |
| 6. MFP/PhysChem 正式复跑 | 官方预计算 `.npz` | RF 5×5 CV | 8 组 summary | 数值复现 |
| 7. Table S3 合并 | 官方 summary + 本地 summary | 按数据集/描述符/指标计算绝对差和相对差 | 48 行主差异表 | 数值核验 |
| 8. SM/OHE 法证 | SM/2SM、官方 OOF、配置和脚本 | 核对行序、标签、fold、dtype、催化剂列 | 法证 manifest/report | 未解差异 |
| 9. BH2 外部验证 | 官方 per-product 和 187 行预测 | 重新计算 MAE并核对文件完整性 | external tables/report | 审计验证 |
| 10. 组件留出审计 | 4 个正式 CSV | 统计 0D、1D、2D 留出组和测试集大小 | 25 行 schema summary | 结构审计 |
| 11. 不平衡分析 | 官方 RF OOF 预测 | 分桶误差、高产率 precision/recall | 69 行 bucket error、12 行 high-yield summary | 审计验证 |
| 12. 自动汇总与测试 | 所有小型产物 | 生成状态报告并运行单元测试 | 状态 JSON、报告、24 项测试 | 质量控制 |

### 5.1 统一训练协议

核心训练协议与官方脚本对齐：

- `5` 次外层重复，每次 `5` 折，共 `25` 个测试折。
- `KFold(n_splits=5, shuffle=True, random_state=1000 + repeat_index)`。
- `RandomForestRegressor(n_estimators=500, max_features=0.3, n_jobs=-1)`。
- RF 的 `random_state` 同样随 repeat 使用 `1000 + repeat_index`。
- OHE 在每一个 fold 内仅用训练集拟合，再转换训练集和测试集，避免类别编码泄漏。本地兼容实现使用 `OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=float32)`，缺失值标记为 `<missing>`，输入列来自各数据集 component manifest；官方配置记录的缺失标记和细节并非逐项完全相同，因此 OHE 不被标为位级数值复现。
- MFP、PhysChem 使用官方预计算 `.npz` 中的 `X` 和 `y`；浮点特征的 NaN 在本地 runner 中置零。
- 每个指标先在单个测试 fold 上计算，再对 5 次重复 × 5 折的 25 个 fold 做算术平均；不是先拼接全部 OOF 再计算一次指标。
- 报告指标为 MAE、RMSE、R² 和 Kendall τ；官方脚本还计算 MedAE，但它不在本次 48 行主对照表中。

MAE 和 RMSE 保持各数据集原始产率尺度，没有跨数据集归一化。因 SLAP 存在大于 100 的值，不能把所有数据集的误差都机械解释为标准百分率误差。

## 6. 核心交叉验证结果

下表均为本地 5×5 CV 结果。MFP 和 PhysChem 来自官方 `.npz` 复跑，因此只证明特征输入之后的训练与评价链；OHE 来自本地兼容实现。“最大绝对差”仅用于逐行描述偏差，不作为跨 MAE、RMSE、R²、Kendall τ 的统一统计量，也没有预设一个把 OHE 判为“通过”的容差阈值。

| 数据集 | 描述符 | MAE ↓ | RMSE ↓ | R² ↑ | Kendall τ ↑ | 与官方结果 |
|---|---|---:|---:|---:|---:|---|
| BH | MFP | 4.5315 | 6.8520 | 0.9365 | 0.8484 | 精确一致 |
| BH | OHE | 6.4349 | 9.3105 | 0.8830 | 0.7951 | 最大绝对差 0.006902 |
| BH | PhysChem | 6.0446 | 8.7507 | 0.8965 | 0.8109 | 浮点精度内一致 |
| BH2 | MFP | 12.0560 | 17.8925 | 0.7250 | 0.6841 | 精确一致 |
| BH2 | OHE | 15.2443 | 21.8620 | 0.5896 | 0.6143 | 最大绝对差 0.007504 |
| BH2 | PhysChem | 13.8611 | 19.4802 | 0.6743 | 0.6611 | 精确一致 |
| SM | MFP | 7.3989 | 11.0086 | 0.8528 | 0.7615 | 精确一致 |
| SM | OHE | 7.8416 | 11.1586 | 0.8418 | 0.7515 | 未解；最大绝对差 1.900650 |
| SM | PhysChem | 7.3738 | 10.8609 | 0.8567 | 0.7654 | 浮点精度内一致 |
| SLAP | MFP | 18.0434 | 35.1230 | 0.7682 | 0.5095 | 精确一致 |
| SLAP | OHE | 21.9328 | 45.7516 | 0.6075 | 0.4811 | 最大绝对差 0.109521 |
| SLAP | PhysChem | 18.4253 | 34.6871 | 0.7736 | 0.4901 | 精确一致 |

完整的 48 行逐指标差异见 [`table_s3_local_vs_official_differences.csv`](../tables/table_s3_local_vs_official_differences.csv)。

### 6.1 结果解读

- BH 和 BH2 上，MFP 在四项指标上都优于 OHE 和 PhysChem。
- SM 上，PhysChem 在四项指标上略优于 MFP；两者差距很小。
- SLAP 上，MFP 的 MAE 和 Kendall τ 更好，而 PhysChem 的 RMSE 和 R² 略好，说明“最佳描述符”依赖评价指标。
- 在 BH、BH2 和 SLAP 的各自官方口径下，OHE 整体弱于 MFP/PhysChem；它仍是必须保留的低成本基线。SM 因 OHE 与 MFP/PhysChem 使用的样本总体不同，不据此做严格横向排名。
- 当前本地证据足以支持“较简单的 MFP/PhysChem 可以形成很强的反应产率基线”，但尚不能用本地复跑比较 DFT/SOAP 的成本—性能关系，因为这两类描述符尚未重新训练。

## 7. SM/OHE 未解差异法证

官方 SM/OHE 目标与本地兼容复跑结果如下：

| 指标 | 官方值 | 本地值 | 绝对差 |
|---|---:|---:|---:|
| MAE | 9.3629 | 7.8416 | 1.5213 |
| RMSE | 13.0593 | 11.1586 | 1.9007 |
| R² | 0.7834 | 0.8418 | 0.0584 |
| Kendall τ | 0.7102 | 0.7515 | 0.0414 |

已完成的排查：

- `SM.csv` 为 `5760 × 6`，SHA-256 为 `111da4e67305f0e00ae7551a0252b78aee4438d434dd5f20e80917e2d0ab596a`。
- `2SM.csv` 为 `5760 × 7`，只多出 `catalyst_smiles`；共享列和 Yield 完全一致。
- 官方 OOF 的 `y_true`、`y_pred`、`fold_id` 长度均为 28,800，即 `5760 × 5`。
- 官方 `y_true` 顺序和 `fold_id` 与本地按种子 1000–1004 生成的 KFold 完全一致。
- 强制 float64 OHE 不能匹配官方值。
- 使用 `2SM.csv + catalyst_smiles` 也不能匹配官方值。

因此，行顺序、标签、fold、浮点 dtype 和催化剂列都不是充分解释。最受当前证据支持的结论是：官方包内的 Suzuki OHE RF summary/OOF 可能由不同代码修订、不同依赖版本或未记录配置生成，现有 README、配置和脚本不足以恢复该生成路径。完整法证见 [`sm_ohe_forensics_report.md`](sm_ohe_forensics_report.md)。

## 8. BH2 外部验证

官方包包含 13 个 per-product 输入文件和相应预测文件，但 `BH2_holdout_predict.py` 所需的 `model_MFP_RF.pkl` 不在包内。因此，本步骤重新计算并核对官方预测文件中的误差，不声称完成本地模型重新预测。

### 8.1 187 条 corrected catalyst smiles

| 预测列 | MAE |
|---|---:|
| `predict` | 20.2406 |
| `ours` | 18.6105 |

这里的 `predict` 和 `ours` 都是**官方 187 行 CSV 中已有的字段名**，`ours` 不代表本项目训练的模型。官方字段 `ours` 相比官方字段 `predict` 的 MAE 降低约 `1.6301`，相对降低约 `8.05%`。重新计算值与文件中的 `AE_predict`、`AE_ours` 均值相互吻合。

### 8.2 逐 product 审计

13 个 product 的官方 MFP MAE 范围为 `5.4558–48.0846`：最低为 product `l`，最高为 product `e`。这种大幅波动表明外部泛化性能强烈依赖被留出的化学子空间，仅汇报总体随机 CV 指标会掩盖这一风险。

详细结果见 [`bh2_external_per_product_mae.csv`](../tables/bh2_external_per_product_mae.csv) 和 [`bh2_external_validation_report.md`](bh2_external_validation_report.md)。

## 9. 0D/1D/2D 组件留出结构审计

本项目对 4 个正式数据集生成了 25 行留出结构摘要：

| 数据集 | 0D 随机折测试大小 | 示例 2D 留出组合 | 2D 组数 | 每组测试样本范围 |
|---|---:|---|---:|---:|
| BH | 791 | Aryl halide + Additive | 330 | 11–12 |
| BH2 | 671–672 | Amine + Bromide | 123 | 12–49 |
| SM | 1,152 | Reactant 1 + Reactant 2 | 15 | 384 |
| SLAP | 230 | Aldehyde 1 + bifunctional reagent | 71 | 6–72 |

1D 留出按单个组件唯一值分组，保证同一被留出组件值不会同时出现在训练集和测试集中；2D 审计按配置中的前两个组件列形成 pair。该步骤只验证 schema、组数、测试集大小和泄漏边界，没有执行论文精确的组件留出模型训练，也没有证明当前前两列映射等同于论文所有 2D 定义。

完整结构见 [`official_component_split_schema_summary.csv`](../tables/official_component_split_schema_summary.csv)。

## 10. 产率不平衡与高产率识别

该分析使用官方 RF OOF 预测，不是重新训练后的本地 OOF。产率分桶为 `<20`、`20–40`、`40–60`、`60–80`、`80–100` 和 `>100`；高产率阈值固定为 `Yield ≥ 80`。

数据分布具有明显不平衡：

- BH2 中 `<20` 的样本为 `2106/3359`，占 `62.70%`。
- SLAP 中 `<20` 的样本占 `71.13%`，同时有 `87/1150` 个样本大于 100，说明其原始尺度存在长尾。
- 由于 OOF 汇总连接了 5 次重复，高产率表中的 `true_high`、TP、FP、FN 是重复后的累计计数，而不是唯一反应数。

高产率识别的代表性结果：

| 数据集 | 描述符 | Precision | Recall |
|---|---|---:|---:|
| BH | OHE | 0.9019 | 0.5450 |
| BH | MFP | 0.9088 | 0.6193 |
| BH2 | OHE | 0.6995 | 0.0620 |
| BH2 | MFP | 0.8990 | 0.2295 |
| BH2 | PhysChem | 0.8868 | 0.1401 |
| SM | MFP | 0.8822 | 0.6997 |
| SM | PhysChem | 0.8966 | 0.6785 |
| SLAP | MFP | 0.7521 | 0.8552 |
| SLAP | PhysChem | 0.7703 | 0.8686 |

最重要的观察是 BH2：三个描述符的 high-yield precision 尚可，但 recall 很低，特别是 OHE 仅为 `0.0620`。这说明良好的总体回归指标不等于能够找全高产率反应。对用于实验筛选的 YONOD 项目，至少应把 high-yield recall、precision 和分桶误差作为与 MAE/R² 并列的指标。

完整结果见 [`official_oof_high_yield_summary.csv`](../tables/official_oof_high_yield_summary.csv) 和 [`official_oof_bucket_error_summary.csv`](../tables/official_oof_bucket_error_summary.csv)。SM/OHE 的相关 OOF 分析继承其官方产物未解边界，不应与其他已解释结果等量齐观。

## 11. 测试与质量控制

2026-08-25 使用本地隔离环境执行：

```powershell
.\.venv\Scripts\python.exe -m pytest tests -q
```

结果：`24 passed in 4.95s`。

测试覆盖数据包存在性与结构、数据源、smoke pipeline、指标提取、训练参数审计、官方 NPZ runner、差异合并、SM/OHE 法证、BH2 external、组件切分、不平衡分析和状态报告生成。

Git 在生成本报告前状态为 clean。官方原始包、`.venv`、大体积 `.npz`、模型文件、详细 run 目录和图像产物均按复现区 `.gitignore` 策略不纳入普通提交。

## 12. 在当前工作区重复执行

以下命令均从 `reference-proejct/vjethbkm/` 执行。它们用于在**已经配置好的当前工作区**重复生成结果，不是完整的全自动数据下载和环境安装流程。

前置条件：

- 当前仓库包含报告基线提交 `ae503402` 或其后续兼容版本。
- 官方数据包已经从 DOI `10.3929/ethz-c-000800856` 获取并解压到 `yieldsmarter/`，其中应直接包含 `Data/`、`Src/`、`Results/`、`README.txt`、`LICENSE.txt` 和 `yieldsmarter.yml`。
- `.venv` 已经安装第 4 节列出的核心依赖。本复现区当前没有独立 lockfile；官方完整环境定义位于 `yieldsmarter/yieldsmarter.yml`，因此新机器安装时应优先参考该文件，并记录最终解析版本。
- 可运行 `scripts/audit_yieldsmarter_package.py` 重新生成关键文件 checksum，并与 `data/manifest/yieldsmarter_key_file_checksums.csv` 核对。

当前仓库没有提供官方包的自动下载脚本，也没有提供可从空环境一键构建的锁定文件。这是本次复现的工程局限，不能仅凭下列命令声称在任意新机器上实现完全无条件重建。

### 12.1 环境、来源与参数审计

```powershell
.\.venv\Scripts\python.exe scripts\check_environment.py
.\.venv\Scripts\python.exe scripts\audit_yieldsmarter_package.py
.\.venv\Scripts\python.exe scripts\audit_official_training_params.py
.\.venv\Scripts\python.exe scripts\extract_official_metrics.py
```

### 12.2 OHE 5×5 正式复跑

```powershell
.\.venv\Scripts\python.exe scripts\run_benchmark.py --stage core_rf_5x5 --dataset BH --descriptors ohe
.\.venv\Scripts\python.exe scripts\run_benchmark.py --stage core_rf_5x5 --dataset BH2 --descriptors ohe
.\.venv\Scripts\python.exe scripts\run_benchmark.py --stage core_rf_5x5 --dataset SM --descriptors ohe
.\.venv\Scripts\python.exe scripts\run_benchmark.py --stage core_rf_5x5 --dataset SLAP --descriptors ohe
```

### 12.3 MFP/PhysChem 5×5 正式复跑

```powershell
.\.venv\Scripts\python.exe scripts\run_official_npz_benchmark.py --dataset BH --descriptors MFP,PhysChem
.\.venv\Scripts\python.exe scripts\run_official_npz_benchmark.py --dataset BH2 --descriptors MFP,PhysChem
.\.venv\Scripts\python.exe scripts\run_official_npz_benchmark.py --dataset SM --descriptors MFP,PhysChem
.\.venv\Scripts\python.exe scripts\run_official_npz_benchmark.py --dataset SLAP --descriptors MFP,PhysChem
```

MFP 维度较高，BH、BH2 和 SM 的运行时间明显长于 PhysChem。建议逐数据集执行，并保留足够内存和数分钟以上的运行窗口。

### 12.4 差异、法证和扩展分析

```powershell
.\.venv\Scripts\python.exe scripts\merge_official_differences.py
.\.venv\Scripts\python.exe scripts\forensic_sm_ohe.py
.\.venv\Scripts\python.exe scripts\analyze_bh2_external.py
.\.venv\Scripts\python.exe scripts\audit_official_component_splits.py
.\.venv\Scripts\python.exe scripts\analyze_official_imbalance.py
.\.venv\Scripts\python.exe scripts\build_reproduction_status_report.py
.\.venv\Scripts\python.exe -m pytest tests -q
```

## 13. 复现完成度

| 论文工作项 | 当前状态 | 证据边界 |
|---|---|---|
| 官方数据包与 schema | 已完成 | 本地官方包审计、checksum 和 schema 已生成。 |
| RF 5×5 协议 | 已完成 | 参数和切分逻辑已从官方脚本核对。 |
| OHE + RF | 部分完成 | BH/BH2/SLAP 兼容复跑接近官方；SM/OHE 未解。 |
| MFP + RF | 已完成训练评价链 | 使用官方预计算特征；4 数据集、16 指标精确一致，未复现特征生成。 |
| PhysChem + RF | 已完成训练评价链 | 使用官方预计算特征；4 数据集、16 指标在浮点精度内一致，未复现特征生成。 |
| DFT + RF | 未复现 | 官方目标和 NPZ 存在，但尚未本地复跑。 |
| SOAP + RF | 未复现 | 官方目标和 NPZ 存在，但尚未本地复跑。 |
| LightGBM/KNN/Ridge | 未复现 | 仅从论文/SI 和官方包识别。 |
| 0D/1D/2D 泛化训练 | 部分完成 | 完成 schema/组数/泄漏边界审计，未完成论文精确训练。 |
| BH2 外部重新预测 | 部分完成 | 官方输出已审计；缺少 `model_MFP_RF.pkl`，未本地 repredict。 |
| 产率分桶与 high-yield | 已完成审计 | 基于官方 OOF 预测重新统计。 |
| 重加权实验 | 未复现 | 官方 `train_weighting.py` 已识别，尚未运行 weighted vs unweighted 对照。 |

## 14. 对 YONOD 项目设计的指导意义

1. **先建立低成本强基线。** MFP 和 PhysChem 在本次复现中稳定、可追溯且性能强，应成为 YONOD 默认基线，再判断复杂表征是否带来足够增益。
2. **统一协议比增加模型更优先。** 固定 split、seed、特征输入、指标定义和版本记录，是比较描述符和模型的前提。
3. **避免只看随机 CV。** BH2 外部结果和组件组大小表明，随机 KFold 不能代表对新底物、新组合或新化学空间的泛化。
4. **把高产率检出作为独立目标。** 对实验筛选而言，BH2 的低 recall 比总体 MAE 更有决策意义；报告应固定输出 high-yield precision、recall 和分桶误差。
5. **保留原始尺度和异常值信息。** SLAP 的大于 100 长尾说明不能在未核实实验定义前强行截断或统一归一化。
6. **对异常结果采用证据化法证。** SM/OHE 案例表明，复现失败本身也能形成有效结果；应记录排除项、hash、split 和版本，而不是为了“对齐”修改目标值。
7. **区分本地训练与官方输出审计。** 这一区分应保留在 YONOD 的自动报告中，以防把读取官方预测误认为模型复现。

## 15. 后续优先级

建议按以下顺序继续：

1. 使用现有官方 NPZ runner 复跑 DFT 和 SOAP，扩展到 4 数据集 × 5 描述符 × 4 指标的完整 RF 矩阵。
2. 读取并严格复刻官方 `train_weighting.py`，生成 weighted vs unweighted 的总体指标、分桶误差和 high-yield 指标对照。
3. 从 SI 或官方绘图脚本确定论文精确的 0D/1D/2D 组件映射，再执行正式泛化训练。
4. 对 BH2 从训练集重训 holdout 模型，或补齐官方 `model_MFP_RF.pkl`，完成真正的本地 external reprediction。
5. 最后扩展 LightGBM、KNN 和 Ridge，避免在核心 RF 证据尚未闭环前扩大模型矩阵。

## 16. 关键产物索引

- 复现状态总览：[`reproduction_status_report.md`](reproduction_status_report.md)
- 机器可读状态：[`reproduction_status_summary.json`](../../data/manifest/reproduction_status_summary.json)
- Table S3 主差异表：[`table_s3_local_vs_official_differences.csv`](../tables/table_s3_local_vs_official_differences.csv)
- 官方 RF 目标表：[`table_s3_official_rf_targets.csv`](../tables/table_s3_official_rf_targets.csv)
- SM/OHE 法证：[`sm_ohe_forensics_report.md`](sm_ohe_forensics_report.md)
- BH2 external：[`bh2_external_validation_report.md`](bh2_external_validation_report.md)
- 组件留出结构：[`official_component_split_audit_report.md`](official_component_split_audit_report.md)
- 不平衡分析：[`official_imbalance_report.md`](official_imbalance_report.md)
- 构建过程日志：[`project-docs/buildlog.md`](../../../../project-docs/buildlog.md)

## 结论

本次复现已经完成 VJETHBKM 核心低成本描述符 + RF 工作流的可审计实现：MFP 和 PhysChem 达到逐指标数值复现，OHE 在三个数据集上获得接近官方的兼容结果，并对 SM/OHE 的异常进行了可重复的排除式法证。外部验证、组件泛化和产率不平衡也已形成结构化审计产物。

因此，当前最稳健的项目结论不是“所有论文结果均已完整复现”，而是：**核心 MFP/PhysChem 的 RF 训练评价链已经高可信复现；OHE 为兼容复跑且 SM/OHE 未解；泛化与不平衡结论获得了官方产物支持的审计验证；高成本描述符、重加权和论文精确组件留出训练仍是明确的下一阶段。**
