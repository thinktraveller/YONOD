# YONOD 建模报告 — 1234567

> 生成时间：2026-07-06 03:55:51  
> 评估组合数：24

---

## 任务信息

| 项目 | 内容 |
|---|---|
| 任务名称 | 1234567 |
| 项目文件夹 | `different_order/1234567` |
| CSV 路径 | `different_order/amide-coupling(additive_fixed)_normalized_dataset.csv` |
| 有效样本量 | 47015 |
| SMILES 列 | `['reactant-amide', 'reactant-acid', 'product', 'activation-1', 'activation-2', 'additive', 'base', 'solvent']` |
| 数值辅助列 | `（无）` |
| 标签列 | `yield` |

---

## 结果矩阵（R²）

| 描述符 \ 模型 | xgb | rf | svm | autogluon |
|---|---|---|---|---|
| morgan | 0.6872 | 0.7660 | 0.6428 | 0.8110 |
| maccs | 0.7013 | 0.7586 | 0.6400 | 0.8015 |
| rdkit2d | 0.7454 | 0.7735 | 0.6400 | 0.8080 |
| fisd | 0.7294 | 0.7657 | -0.0218 | 0.7736 |
| molmetalm | 0.7048 | 0.7084 | -0.0219 | 0.7386 |
| maf | 0.6710 | 0.7107 | 0.6400 | 0.7802 |

---

## 详细指标

| 描述符 | 模型 | R² 均值 | R² 标准差 | RMSE | MAE | 用时 |
|---|---|---|---|---|---|---|
| morgan | xgb | 0.6872 | 0.0067 | 0.1686 | 0.1306 | 20.1s |
| morgan | rf | 0.7660 | 0.0036 | 0.1458 | 0.0926 | 18m 49s |
| morgan | svm | 0.6428 | 0.0071 | 0.1801 | 0.1402 | 1m 12s |
| morgan | autogluon | 0.8110 | 0.0000 | 0.1313 | 0.0900 | 6m 18s |
| maccs | xgb | 0.7013 | 0.0048 | 0.1647 | 0.1256 | 4.8s |
| maccs | rf | 0.7586 | 0.0047 | 0.1481 | 0.0957 | 2m 14s |
| maccs | svm | 0.6400 | 0.0081 | 0.1808 | 0.1393 | 46.1s |
| maccs | autogluon | 0.8015 | 0.0000 | 0.1345 | 0.0924 | 5m 2s |
| rdkit2d | xgb | 0.7454 | 0.0067 | 0.1521 | 0.1132 | 7.3s |
| rdkit2d | rf | 0.7735 | 0.0043 | 0.1435 | 0.0909 | 6m 45s |
| rdkit2d | svm | 0.6400 | 0.0087 | 0.1809 | 0.1391 | 47.1s |
| rdkit2d | autogluon | 0.8080 | 0.0000 | 0.1323 | 0.0893 | 5m 2s |
| fisd | xgb | 0.7294 | 0.0071 | 0.1568 | 0.1177 | 4.5s |
| fisd | rf | 0.7657 | 0.0073 | 0.1459 | 0.0993 | 9m 42s |
| fisd | svm | -0.0218 | 0.0056 | 0.3047 | 0.2664 | 1m 55s |
| fisd | autogluon | 0.7736 | 0.0000 | 0.1437 | 0.1038 | 5m 6s |
| molmetalm | xgb | 0.7048 | 0.0085 | 0.1638 | 0.1225 | 29.6s |
| molmetalm | rf | 0.7084 | 0.0075 | 0.1628 | 0.1073 | 84m 40s |
| molmetalm | svm | -0.0219 | 0.0056 | 0.3047 | 0.2664 | 1m 23s |
| molmetalm | autogluon | 0.7386 | 0.0000 | 0.1544 | 0.1128 | 5m 3s |
| maf | xgb | 0.6710 | 0.0070 | 0.1729 | 0.1353 | 2.3s |
| maf | rf | 0.7107 | 0.0061 | 0.1621 | 0.1152 | 41.2s |
| maf | svm | 0.6400 | 0.0106 | 0.1808 | 0.1408 | 16.3s |
| maf | autogluon | 0.7802 | 0.0000 | 0.1416 | 0.0998 | 3m 48s |

---

## 推荐组合（加权排名）

| 名次 | 描述符 | 模型 | R² | RMSE | MAE | 综合分 | 用时 | 推荐理由 |
|---|---|---|---|---|---|---|---|---|
| 🥇 1 | **morgan** | **autogluon** | 0.811 | 0.1313 | 0.0900 | 0.709 | 6m 18s | 综合评分 0.709 排名靠前 |
| 🥈 2 | **rdkit2d** | **autogluon** | 0.808 | 0.1323 | 0.0893 | 0.707 | 5m 2s | 综合评分 0.707 排名靠前 |
| 🥉 3 | **maccs** | **autogluon** | 0.802 | 0.1345 | 0.0924 | 0.699 | 5m 2s | 综合评分 0.699 排名靠前 |

<details>
<summary>展开完整排名</summary>

| 名次 | 描述符 | 模型 | R² | RMSE | MAE | 综合分 | 用时 | 推荐理由 |
|---|---|---|---|---|---|---|---|---|
| 1 | morgan | autogluon | 0.811 | 0.1313 | 0.0900 | 0.709 | 6m 18s ||
| 2 | rdkit2d | autogluon | 0.808 | 0.1323 | 0.0893 | 0.707 | 5m 2s ||
| 3 | maccs | autogluon | 0.802 | 0.1345 | 0.0924 | 0.699 | 5m 2s ||
| 4 | rdkit2d | rf | 0.773 | 0.1435 | 0.0909 | 0.677 | 6m 45s ||
| 5 | maf | autogluon | 0.780 | 0.1416 | 0.0998 | 0.676 | 3m 48s ||
| 6 | morgan | rf | 0.766 | 0.1458 | 0.0926 | 0.670 | 18m 49s ||
| 7 | fisd | autogluon | 0.774 | 0.1437 | 0.1038 | 0.667 | 5m 6s ||
| 8 | fisd | rf | 0.766 | 0.1459 | 0.0993 | 0.665 | 9m 42s ||
| 9 | maccs | rf | 0.759 | 0.1481 | 0.0957 | 0.662 | 2m 14s ||
| 10 | rdkit2d | xgb | 0.745 | 0.1521 | 0.1132 | 0.638 | 7.3s ||
| 11 | molmetalm | autogluon | 0.739 | 0.1544 | 0.1128 | 0.633 | 5m 3s ||
| 12 | fisd | xgb | 0.729 | 0.1568 | 0.1177 | 0.622 | 4.5s ||
| 13 | molmetalm | rf | 0.708 | 0.1628 | 0.1073 | 0.613 | 84m 40s ||
| 14 | maf | rf | 0.711 | 0.1621 | 0.1152 | 0.609 | 41.2s ||
| 15 | molmetalm | xgb | 0.705 | 0.1638 | 0.1225 | 0.599 | 29.6s ||
| 16 | maccs | xgb | 0.701 | 0.1647 | 0.1256 | 0.594 | 4.8s ||
| 17 | morgan | xgb | 0.687 | 0.1686 | 0.1306 | 0.580 | 20.1s ||
| 18 | maf | xgb | 0.671 | 0.1729 | 0.1353 | 0.564 | 2.3s ||
| 19 | morgan | svm | 0.643 | 0.1801 | 0.1402 | 0.539 | 1m 12s ||
| 20 | rdkit2d | svm | 0.640 | 0.1809 | 0.1391 | 0.538 | 47.1s ||
| 21 | maccs | svm | 0.640 | 0.1808 | 0.1393 | 0.537 | 46.1s ||
| 22 | maf | svm | 0.640 | 0.1808 | 0.1408 | 0.536 | 16.3s ||
| 23 | fisd | svm | -0.022 | 0.3047 | 0.2664 | -0.011 | 1m 55s ||
| 24 | molmetalm | svm | -0.022 | 0.3047 | 0.2664 | -0.011 | 1m 23s ||

</details>

---

## 列映射与分类

| 原始列名 | 角色 | 规范名称 |
|---|---|---|
| yield | 标签列 | `yield` |
| sub_1_smiles | 反应物 | `reactant-amide` |
| sub_2_smiles | 反应物 | `reactant-acid` |
| product_smiles | 产物 | `product` |
| activation | 其他分子 | `activation` |
| additive | 其他分子 | `additive` |
| base | 其他分子 | `base` |
| solvent | 其他分子 | `solvent` |

---

## 描述符配置

| 描述符 | 模式 | 应用列 |
|---|---|---|
| **morgan** | 拼接 | `reactant-amide, reactant-acid, product, activation, additive, base, solvent` |
| **maccs** | 拼接 | `reactant-amide, reactant-acid, product, activation, additive, base, solvent` |
| **rdkit2d** | 拼接 | `reactant-amide, reactant-acid, product, activation, additive, base, solvent` |
| **fisd** | 拼接 | `reactant-amide, reactant-acid, product, activation, additive, base, solvent` |
| **molmetalm** | 拼接 | `reactant-amide, reactant-acid, product, activation, additive, base, solvent` |
| **maf** | 求和 | `reactant-amide, reactant-acid, product, activation, additive, base, solvent` |

---

## 数据路径

| 项目 | 路径 |
|---|---|
| 原始数据集 | `dataset/amide-coupling(additive_fixed).csv` |
| 规范化数据集 | `different_order/amide-coupling(additive_fixed)_normalized_dataset.csv` |
| 配置文件 | `different_order/1234567/1234567_yonod_config.json` |

---

## 指标说明

**R²（决定系数）**：衡量模型预测方差占真实方差的比例，取值上限为 1。

- R² > 0.85：预测可靠性较强，可用于辅助筛选实验条件
- R² 0.7～0.85：中等预测能力，趋势判断可参考，具体数值需谨慎
- R² < 0.7 或负值：拟合效果弱；样本量极少时负值属正常现象

**RMSE（均方根误差）**：对大误差样本更敏感。标签归一化到 [0,1] 时，单位等同于产率百分点。

- RMSE < 0.05：平均误差约 5 个百分点，接近实验重复性误差范围
- RMSE 0.05～0.10：中等误差，可区分高产率和低产率区间
- RMSE > 0.10：误差偏大，不建议用于定量预测

**MAE（平均绝对误差）**：对每个样本的预测偏差取绝对值后平均。

- MAE < 0.04：典型偏差极小，预测稳定性好
- MAE 0.04～0.08：中等偏差，结合 R² 综合评估
- MAE > 0.08：典型偏差较大

**综合评分**：S = R² × 0.5 + (1 − RMSE / RMSE_max) × 0.3 + (1 − MAE / MAE_max) × 0.2

---

*由 YONOD report.py 自动生成 · 2026-07-06 03:55:51*