# YONOD 建模报告 — 123457

> 生成时间：2026-07-06 22:22:46  
> 评估组合数：24

---

## 任务信息

| 项目 | 内容 |
|---|---|
| 任务名称 | 123457 |
| 项目文件夹 | `different_order/123457` |
| CSV 路径 | `different_order/amide-coupling(additive_fixed)_normalized_dataset.csv` |
| 有效样本量 | 47015 |
| SMILES 列 | `['reactant-amide', 'reactant-acid', 'product', 'activation-1', 'activation-2', 'additive', 'base', 'solvent']` |
| 数值辅助列 | `（无）` |
| 标签列 | `yield` |

---

## 结果矩阵（R²）

| 描述符 \ 模型 | xgb | rf | svm | autogluon |
|---|---|---|---|---|
| morgan | 0.6696 | 0.7296 | 0.6262 | 0.7700 |
| maccs | 0.6816 | 0.7225 | 0.6179 | 0.7626 |
| rdkit2d | 0.7213 | 0.7371 | 0.6185 | 0.7723 |
| fisd | 0.7076 | 0.7403 | -0.0218 | 0.7575 |
| molmetalm | 0.6847 | 0.6803 | -0.0220 | 0.7155 |
| maf | 0.6588 | 0.7013 | 0.6273 | 0.7525 |

---

## 详细指标

| 描述符 | 模型 | R² 均值 | R² 标准差 | RMSE | MAE | 用时 |
|---|---|---|---|---|---|---|
| morgan | xgb | 0.6696 | 0.0069 | 0.1733 | 0.1344 | 15.1s |
| morgan | rf | 0.7296 | 0.0036 | 0.1567 | 0.1003 | 14m 14s |
| morgan | svm | 0.6262 | 0.0087 | 0.1843 | 0.1429 | 1m 5s |
| morgan | autogluon | 0.7700 | 0.0000 | 0.1448 | 0.0991 | 6m 0s |
| maccs | xgb | 0.6816 | 0.0057 | 0.1701 | 0.1300 | 4.4s |
| maccs | rf | 0.7225 | 0.0061 | 0.1588 | 0.1036 | 1m 59s |
| maccs | svm | 0.6179 | 0.0086 | 0.1863 | 0.1430 | 47.7s |
| maccs | autogluon | 0.7626 | 0.0000 | 0.1471 | 0.1011 | 5m 1s |
| rdkit2d | xgb | 0.7213 | 0.0063 | 0.1591 | 0.1188 | 6.7s |
| rdkit2d | rf | 0.7371 | 0.0046 | 0.1545 | 0.0984 | 5m 53s |
| rdkit2d | svm | 0.6185 | 0.0098 | 0.1862 | 0.1425 | 47.1s |
| rdkit2d | autogluon | 0.7723 | 0.0000 | 0.1441 | 0.0991 | 5m 1s |
| fisd | xgb | 0.7076 | 0.0059 | 0.1630 | 0.1226 | 4.4s |
| fisd | rf | 0.7403 | 0.0066 | 0.1536 | 0.1050 | 8m 43s |
| fisd | svm | -0.0218 | 0.0056 | 0.3047 | 0.2664 | 1m 23s |
| fisd | autogluon | 0.7575 | 0.0000 | 0.1487 | 0.1037 | 5m 40s |
| molmetalm | xgb | 0.6847 | 0.0075 | 0.1693 | 0.1267 | 27.3s |
| molmetalm | rf | 0.6803 | 0.0070 | 0.1704 | 0.1129 | 69m 28s |
| molmetalm | svm | -0.0220 | 0.0056 | 0.3047 | 0.2664 | 1m 18s |
| molmetalm | autogluon | 0.7155 | 0.0000 | 0.1610 | 0.1165 | 5m 3s |
| maf | xgb | 0.6588 | 0.0053 | 0.1761 | 0.1368 | 2.3s |
| maf | rf | 0.7013 | 0.0062 | 0.1647 | 0.1113 | 37.2s |
| maf | svm | 0.6273 | 0.0092 | 0.1840 | 0.1427 | 15.4s |
| maf | autogluon | 0.7525 | 0.0000 | 0.1502 | 0.1056 | 3m 2s |

---

## 推荐组合（加权排名）

| 名次 | 描述符 | 模型 | R² | RMSE | MAE | 综合分 | 用时 | 推荐理由 |
|---|---|---|---|---|---|---|---|---|
| 🥇 1 | **rdkit2d** | **autogluon** | 0.772 | 0.1441 | 0.0991 | 0.670 | 5m 1s | 综合评分 0.670 排名靠前 |
| 🥈 2 | **morgan** | **autogluon** | 0.770 | 0.1448 | 0.0991 | 0.668 | 6m 0s | 综合评分 0.668 排名靠前 |
| 🥉 3 | **maccs** | **autogluon** | 0.763 | 0.1471 | 0.1011 | 0.661 | 5m 1s | 综合评分 0.661 排名靠前 |

<details>
<summary>展开完整排名</summary>

| 名次 | 描述符 | 模型 | R² | RMSE | MAE | 综合分 | 用时 | 推荐理由 |
|---|---|---|---|---|---|---|---|---|
| 1 | rdkit2d | autogluon | 0.772 | 0.1441 | 0.0991 | 0.670 | 5m 1s ||
| 2 | morgan | autogluon | 0.770 | 0.1448 | 0.0991 | 0.668 | 6m 0s ||
| 3 | maccs | autogluon | 0.763 | 0.1471 | 0.1011 | 0.661 | 5m 1s ||
| 4 | fisd | autogluon | 0.758 | 0.1487 | 0.1037 | 0.655 | 5m 40s ||
| 5 | maf | autogluon | 0.752 | 0.1502 | 0.1056 | 0.649 | 3m 2s ||
| 6 | rdkit2d | rf | 0.737 | 0.1545 | 0.0984 | 0.643 | 5m 53s ||
| 7 | fisd | rf | 0.740 | 0.1536 | 0.1050 | 0.640 | 8m 43s ||
| 8 | morgan | rf | 0.730 | 0.1567 | 0.1003 | 0.635 | 14m 14s ||
| 9 | maccs | rf | 0.722 | 0.1588 | 0.1036 | 0.627 | 1m 59s ||
| 10 | rdkit2d | xgb | 0.721 | 0.1591 | 0.1188 | 0.615 | 6.7s ||
| 11 | molmetalm | autogluon | 0.716 | 0.1610 | 0.1165 | 0.612 | 5m 3s ||
| 12 | maf | rf | 0.701 | 0.1647 | 0.1113 | 0.605 | 37.2s ||
| 13 | fisd | xgb | 0.708 | 0.1630 | 0.1226 | 0.601 | 4.4s ||
| 14 | molmetalm | rf | 0.680 | 0.1704 | 0.1129 | 0.588 | 69m 28s ||
| 15 | molmetalm | xgb | 0.685 | 0.1693 | 0.1267 | 0.581 | 27.3s ||
| 16 | maccs | xgb | 0.682 | 0.1701 | 0.1300 | 0.576 | 4.4s ||
| 17 | morgan | xgb | 0.670 | 0.1733 | 0.1344 | 0.563 | 15.1s ||
| 18 | maf | xgb | 0.659 | 0.1761 | 0.1368 | 0.553 | 2.3s ||
| 19 | maf | svm | 0.627 | 0.1840 | 0.1427 | 0.525 | 15.4s ||
| 20 | morgan | svm | 0.626 | 0.1843 | 0.1429 | 0.524 | 1m 5s ||
| 21 | rdkit2d | svm | 0.619 | 0.1862 | 0.1425 | 0.519 | 47.1s ||
| 22 | maccs | svm | 0.618 | 0.1863 | 0.1430 | 0.518 | 47.7s ||
| 23 | fisd | svm | -0.022 | 0.3047 | 0.2664 | -0.011 | 1m 23s ||
| 24 | molmetalm | svm | -0.022 | 0.3047 | 0.2664 | -0.011 | 1m 18s ||

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
| **morgan** | 拼接 | `reactant-amide, reactant-acid, product, activation, additive, solvent` |
| **maccs** | 拼接 | `reactant-amide, reactant-acid, product, activation, additive, solvent` |
| **rdkit2d** | 拼接 | `reactant-amide, reactant-acid, product, activation, additive, solvent` |
| **fisd** | 拼接 | `reactant-amide, reactant-acid, product, activation, additive, solvent` |
| **molmetalm** | 拼接 | `reactant-amide, reactant-acid, product, activation, additive, solvent` |
| **maf** | 求和 | `reactant-amide, reactant-acid, product, activation, additive, solvent` |

---

## 数据路径

| 项目 | 路径 |
|---|---|
| 原始数据集 | `dataset/amide-coupling(additive_fixed).csv` |
| 规范化数据集 | `different_order/amide-coupling(additive_fixed)_normalized_dataset.csv` |
| 配置文件 | `different_order/123457/123457_yonod_config.json` |

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

*由 YONOD report.py 自动生成 · 2026-07-06 22:22:46*