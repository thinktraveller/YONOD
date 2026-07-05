# YONOD 建模报告 — 12567

> 生成时间：2026-07-04 23:32:27  
> 评估组合数：24

---

## 任务信息

| 项目 | 内容 |
|---|---|
| 任务名称 | 12567 |
| 项目文件夹 | `different_order/12567` |
| CSV 路径 | `different_order/amide-coupling(additive_fixed)_normalized_dataset.csv` |
| 有效样本量 | 47015 |
| SMILES 列 | `['reactant-1', 'reactant-2', 'product', 'activation-1', 'activation-2', 'additive', 'base', 'solvent']` |
| 数值辅助列 | `（无）` |
| 标签列 | `yield` |

---

## 结果矩阵（R²）

| 描述符 \ 模型 | xgb | rf | svm | autogluon |
|---|---|---|---|---|
| morgan | 0.6872 | 0.7660 | 0.6428 | 0.8110 |
| maccs | 0.7013 | 0.7586 | 0.6400 | 0.8015 |
| rdkit2d | 0.7454 | 0.7735 | 0.6400 | 0.8080 |
| fisd | 0.7291 | 0.7650 | -0.0218 | 0.7746 |
| molmetalm | 0.7048 | 0.7084 | -0.0219 | 0.7392 |
| maf | 0.7165 | 0.7691 | 0.6505 | 0.8106 |

---

## 详细指标

| 描述符 | 模型 | R² 均值 | R² 标准差 | RMSE | MAE | 用时 |
|---|---|---|---|---|---|---|
| morgan | xgb | 0.6872 | 0.0067 | 0.1686 | 0.1306 | 16.7s |
| morgan | rf | 0.7660 | 0.0036 | 0.1458 | 0.0926 | 18m 52s |
| morgan | svm | 0.6428 | 0.0071 | 0.1801 | 0.1402 | 1m 10s |
| morgan | autogluon | 0.8110 | 0.0000 | 0.1313 | 0.0900 | 6m 18s |
| maccs | xgb | 0.7013 | 0.0048 | 0.1647 | 0.1256 | 4.8s |
| maccs | rf | 0.7586 | 0.0047 | 0.1481 | 0.0957 | 2m 15s |
| maccs | svm | 0.6400 | 0.0081 | 0.1808 | 0.1393 | 44.5s |
| maccs | autogluon | 0.8015 | 0.0000 | 0.1345 | 0.0924 | 5m 2s |
| rdkit2d | xgb | 0.7454 | 0.0067 | 0.1521 | 0.1132 | 7.3s |
| rdkit2d | rf | 0.7735 | 0.0043 | 0.1435 | 0.0909 | 6m 45s |
| rdkit2d | svm | 0.6400 | 0.0087 | 0.1809 | 0.1391 | 49.4s |
| rdkit2d | autogluon | 0.8080 | 0.0000 | 0.1323 | 0.0893 | 5m 3s |
| fisd | xgb | 0.7291 | 0.0075 | 0.1569 | 0.1179 | 4.5s |
| fisd | rf | 0.7650 | 0.0070 | 0.1461 | 0.0995 | 9m 46s |
| fisd | svm | -0.0218 | 0.0056 | 0.3047 | 0.2664 | 1m 54s |
| fisd | autogluon | 0.7746 | 0.0000 | 0.1433 | 0.1038 | 5m 6s |
| molmetalm | xgb | 0.7048 | 0.0085 | 0.1638 | 0.1225 | 29.4s |
| molmetalm | rf | 0.7084 | 0.0075 | 0.1628 | 0.1073 | 84m 32s |
| molmetalm | svm | -0.0219 | 0.0056 | 0.3047 | 0.2664 | 1m 25s |
| molmetalm | autogluon | 0.7392 | 0.0000 | 0.1542 | 0.1126 | 5m 3s |
| maf | xgb | 0.7165 | 0.0067 | 0.1605 | 0.1226 | 4.0s |
| maf | rf | 0.7691 | 0.0036 | 0.1448 | 0.0930 | 3m 3s |
| maf | svm | 0.6505 | 0.0067 | 0.1782 | 0.1386 | 43.7s |
| maf | autogluon | 0.8106 | 0.0000 | 0.1314 | 0.0894 | 5m 1s |

---

## 推荐组合（加权排名）

| 名次 | 描述符 | 模型 | R² | RMSE | MAE | 综合分 | 用时 | 推荐理由 |
|---|---|---|---|---|---|---|---|---|
| 🥇 1 | **maf** | **autogluon** | 0.811 | 0.1314 | 0.0894 | 0.709 | 5m 1s | 综合评分 0.709 排名靠前 |
| 🥈 2 | **morgan** | **autogluon** | 0.811 | 0.1313 | 0.0900 | 0.709 | 6m 18s | 综合评分 0.709 排名靠前 |
| 🥉 3 | **rdkit2d** | **autogluon** | 0.808 | 0.1323 | 0.0893 | 0.707 | 5m 3s | 综合评分 0.707 排名靠前 |

<details>
<summary>展开完整排名</summary>

| 名次 | 描述符 | 模型 | R² | RMSE | MAE | 综合分 | 用时 | 推荐理由 |
|---|---|---|---|---|---|---|---|---|
| 1 | maf | autogluon | 0.811 | 0.1314 | 0.0894 | 0.709 | 5m 1s ||
| 2 | morgan | autogluon | 0.811 | 0.1313 | 0.0900 | 0.709 | 6m 18s ||
| 3 | rdkit2d | autogluon | 0.808 | 0.1323 | 0.0893 | 0.707 | 5m 3s ||
| 4 | maccs | autogluon | 0.802 | 0.1345 | 0.0924 | 0.699 | 5m 2s ||
| 5 | rdkit2d | rf | 0.773 | 0.1435 | 0.0909 | 0.677 | 6m 45s ||
| 6 | maf | rf | 0.769 | 0.1448 | 0.0930 | 0.672 | 3m 3s ||
| 7 | morgan | rf | 0.766 | 0.1458 | 0.0926 | 0.670 | 18m 52s ||
| 8 | fisd | autogluon | 0.775 | 0.1433 | 0.1038 | 0.668 | 5m 6s ||
| 9 | fisd | rf | 0.765 | 0.1461 | 0.0995 | 0.664 | 9m 46s ||
| 10 | maccs | rf | 0.759 | 0.1481 | 0.0957 | 0.662 | 2m 15s ||
| 11 | rdkit2d | xgb | 0.745 | 0.1521 | 0.1132 | 0.638 | 7.3s ||
| 12 | molmetalm | autogluon | 0.739 | 0.1542 | 0.1126 | 0.633 | 5m 3s ||
| 13 | fisd | xgb | 0.729 | 0.1569 | 0.1179 | 0.622 | 4.5s ||
| 14 | molmetalm | rf | 0.708 | 0.1628 | 0.1073 | 0.613 | 84m 32s ||
| 15 | maf | xgb | 0.716 | 0.1605 | 0.1226 | 0.608 | 4.0s ||
| 16 | molmetalm | xgb | 0.705 | 0.1638 | 0.1225 | 0.599 | 29.4s ||
| 17 | maccs | xgb | 0.701 | 0.1647 | 0.1256 | 0.594 | 4.8s ||
| 18 | morgan | xgb | 0.687 | 0.1686 | 0.1306 | 0.580 | 16.7s ||
| 19 | maf | svm | 0.651 | 0.1782 | 0.1386 | 0.546 | 43.7s ||
| 20 | morgan | svm | 0.643 | 0.1801 | 0.1402 | 0.539 | 1m 10s ||
| 21 | rdkit2d | svm | 0.640 | 0.1809 | 0.1391 | 0.538 | 49.4s ||
| 22 | maccs | svm | 0.640 | 0.1808 | 0.1393 | 0.537 | 44.5s ||
| 23 | fisd | svm | -0.022 | 0.3047 | 0.2664 | -0.011 | 1m 54s ||
| 24 | molmetalm | svm | -0.022 | 0.3047 | 0.2664 | -0.011 | 1m 25s ||

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
| **morgan** | 拼接 | `reactant-amide, reactant-acid, additive, base, solvent` |
| **maccs** | 拼接 | `reactant-amide, reactant-acid, additive, base, solvent` |
| **rdkit2d** | 拼接 | `reactant-amide, reactant-acid, additive, base, solvent` |
| **fisd** | 拼接 | `reactant-amide, reactant-acid, additive, base, solvent` |
| **molmetalm** | 拼接 | `reactant-amide, reactant-acid, additive, base, solvent` |
| **maf** | 求和 | `reactant-amide, reactant-acid, additive, base, solvent` |

---

## 数据路径

| 项目 | 路径 |
|---|---|
| 原始数据集 | `different_order/amide-coupling(additive_fixed).csv` |
| 规范化数据集 | `different_order/amide-coupling(additive_fixed)_normalized_dataset.csv` |
| 配置文件 | `different_order/12567/12567_yonod_config.json` |

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

*由 YONOD report.py 自动生成 · 2026-07-04 23:32:27*