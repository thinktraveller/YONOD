# YONOD 建模报告 — 124567

> 生成时间：2026-07-06 15:04:07  
> 评估组合数：24

---

## 任务信息

| 项目 | 内容 |
|---|---|
| 任务名称 | 124567 |
| 项目文件夹 | `different_order/124567` |
| CSV 路径 | `different_order/amide-coupling(additive_fixed)_normalized_dataset.csv` |
| 有效样本量 | 47015 |
| SMILES 列 | `['reactant-amide', 'reactant-acid', 'product', 'activation-1', 'activation-2', 'additive', 'base', 'solvent']` |
| 数值辅助列 | `（无）` |
| 标签列 | `yield` |

---

## 结果矩阵（R²）

| 描述符 \ 模型 | xgb | rf | svm | autogluon |
|---|---|---|---|---|
| morgan | 0.6861 | 0.7658 | 0.6427 | 0.8120 |
| maccs | 0.6940 | 0.7586 | 0.6256 | 0.8010 |
| rdkit2d | 0.7417 | 0.7779 | 0.6278 | 0.8139 |
| fisd | 0.7097 | 0.7596 | -0.0218 | 0.7674 |
| molmetalm | 0.6673 | 0.6691 | -0.0219 | 0.7159 |
| maf | 0.6381 | 0.7048 | 0.6515 | 0.7727 |

---

## 详细指标

| 描述符 | 模型 | R² 均值 | R² 标准差 | RMSE | MAE | 用时 |
|---|---|---|---|---|---|---|
| morgan | xgb | 0.6861 | 0.0076 | 0.1689 | 0.1307 | 14.7s |
| morgan | rf | 0.7658 | 0.0028 | 0.1459 | 0.0929 | 10m 15s |
| morgan | svm | 0.6427 | 0.0066 | 0.1802 | 0.1401 | 1m 6s |
| morgan | autogluon | 0.8120 | 0.0000 | 0.1309 | 0.0897 | 5m 1s |
| maccs | xgb | 0.6940 | 0.0067 | 0.1667 | 0.1280 | 4.3s |
| maccs | rf | 0.7586 | 0.0049 | 0.1481 | 0.0961 | 1m 48s |
| maccs | svm | 0.6256 | 0.0067 | 0.1844 | 0.1418 | 41.7s |
| maccs | autogluon | 0.8010 | 0.0000 | 0.1347 | 0.0930 | 5m 2s |
| rdkit2d | xgb | 0.7417 | 0.0053 | 0.1532 | 0.1147 | 6.0s |
| rdkit2d | rf | 0.7779 | 0.0039 | 0.1421 | 0.0903 | 4m 40s |
| rdkit2d | svm | 0.6278 | 0.0086 | 0.1839 | 0.1413 | 43.6s |
| rdkit2d | autogluon | 0.8139 | 0.0000 | 0.1303 | 0.0880 | 5m 3s |
| fisd | xgb | 0.7097 | 0.0062 | 0.1624 | 0.1236 | 4.2s |
| fisd | rf | 0.7596 | 0.0062 | 0.1478 | 0.1036 | 7m 34s |
| fisd | svm | -0.0218 | 0.0056 | 0.3047 | 0.2664 | 1m 35s |
| fisd | autogluon | 0.7674 | 0.0000 | 0.1456 | 0.1069 | 5m 9s |
| molmetalm | xgb | 0.6673 | 0.0052 | 0.1739 | 0.1314 | 22.2s |
| molmetalm | rf | 0.6691 | 0.0040 | 0.1734 | 0.1183 | 53m 35s |
| molmetalm | svm | -0.0219 | 0.0057 | 0.3047 | 0.2664 | 1m 21s |
| molmetalm | autogluon | 0.7159 | 0.0000 | 0.1609 | 0.1174 | 5m 3s |
| maf | xgb | 0.6381 | 0.0053 | 0.1813 | 0.1430 | 2.3s |
| maf | rf | 0.7048 | 0.0033 | 0.1638 | 0.1167 | 34.3s |
| maf | svm | 0.6515 | 0.0107 | 0.1779 | 0.1376 | 16.4s |
| maf | autogluon | 0.7727 | 0.0000 | 0.1440 | 0.1028 | 3m 23s |

---

## 推荐组合（加权排名）

| 名次 | 描述符 | 模型 | R² | RMSE | MAE | 综合分 | 用时 | 推荐理由 |
|---|---|---|---|---|---|---|---|---|
| 🥇 1 | **rdkit2d** | **autogluon** | 0.814 | 0.1303 | 0.0880 | 0.713 | 5m 3s | 综合评分 0.713 排名靠前 |
| 🥈 2 | **morgan** | **autogluon** | 0.812 | 0.1309 | 0.0897 | 0.710 | 5m 1s | 综合评分 0.710 排名靠前 |
| 🥉 3 | **maccs** | **autogluon** | 0.801 | 0.1347 | 0.0930 | 0.698 | 5m 2s | 综合评分 0.698 排名靠前 |

<details>
<summary>展开完整排名</summary>

| 名次 | 描述符 | 模型 | R² | RMSE | MAE | 综合分 | 用时 | 推荐理由 |
|---|---|---|---|---|---|---|---|---|
| 1 | rdkit2d | autogluon | 0.814 | 0.1303 | 0.0880 | 0.713 | 5m 3s ||
| 2 | morgan | autogluon | 0.812 | 0.1309 | 0.0897 | 0.710 | 5m 1s ||
| 3 | maccs | autogluon | 0.801 | 0.1347 | 0.0930 | 0.698 | 5m 2s ||
| 4 | rdkit2d | rf | 0.778 | 0.1421 | 0.0903 | 0.681 | 4m 40s ||
| 5 | morgan | rf | 0.766 | 0.1459 | 0.0929 | 0.670 | 10m 15s ||
| 6 | maf | autogluon | 0.773 | 0.1440 | 0.1028 | 0.667 | 3m 23s ||
| 7 | maccs | rf | 0.759 | 0.1481 | 0.0961 | 0.661 | 1m 48s ||
| 8 | fisd | autogluon | 0.767 | 0.1456 | 0.1069 | 0.660 | 5m 9s ||
| 9 | fisd | rf | 0.760 | 0.1478 | 0.1036 | 0.657 | 7m 34s ||
| 10 | rdkit2d | xgb | 0.742 | 0.1532 | 0.1147 | 0.634 | 6.0s ||
| 11 | molmetalm | autogluon | 0.716 | 0.1609 | 0.1174 | 0.611 | 5m 3s ||
| 12 | maf | rf | 0.705 | 0.1638 | 0.1167 | 0.604 | 34.3s ||
| 13 | fisd | xgb | 0.710 | 0.1624 | 0.1236 | 0.602 | 4.2s ||
| 14 | maccs | xgb | 0.694 | 0.1667 | 0.1280 | 0.587 | 4.3s ||
| 15 | morgan | xgb | 0.686 | 0.1689 | 0.1307 | 0.579 | 14.7s ||
| 16 | molmetalm | rf | 0.669 | 0.1734 | 0.1183 | 0.575 | 53m 35s ||
| 17 | molmetalm | xgb | 0.667 | 0.1739 | 0.1314 | 0.564 | 22.2s ||
| 18 | maf | svm | 0.652 | 0.1779 | 0.1376 | 0.547 | 16.4s ||
| 19 | morgan | svm | 0.643 | 0.1802 | 0.1401 | 0.539 | 1m 6s ||
| 20 | maf | xgb | 0.638 | 0.1813 | 0.1430 | 0.533 | 2.3s ||
| 21 | rdkit2d | svm | 0.628 | 0.1839 | 0.1413 | 0.527 | 43.6s ||
| 22 | maccs | svm | 0.626 | 0.1844 | 0.1418 | 0.525 | 41.7s ||
| 23 | fisd | svm | -0.022 | 0.3047 | 0.2664 | -0.011 | 1m 35s ||
| 24 | molmetalm | svm | -0.022 | 0.3047 | 0.2664 | -0.011 | 1m 21s ||

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
| **morgan** | 拼接 | `reactant-amide, reactant-acid, activation, additive, base, solvent` |
| **maccs** | 拼接 | `reactant-amide, reactant-acid, activation, additive, base, solvent` |
| **rdkit2d** | 拼接 | `reactant-amide, reactant-acid, activation, additive, base, solvent` |
| **fisd** | 拼接 | `reactant-amide, reactant-acid, activation, additive, base, solvent` |
| **molmetalm** | 拼接 | `reactant-amide, reactant-acid, activation, additive, base, solvent` |
| **maf** | 求和 | `reactant-amide, reactant-acid, activation, additive, base, solvent` |

---

## 数据路径

| 项目 | 路径 |
|---|---|
| 原始数据集 | `dataset/amide-coupling(additive_fixed).csv` |
| 规范化数据集 | `different_order/amide-coupling(additive_fixed)_normalized_dataset.csv` |
| 配置文件 | `different_order/124567/124567_yonod_config.json` |

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

*由 YONOD report.py 自动生成 · 2026-07-06 15:04:07*