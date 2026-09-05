# YONOD 建模报告 — 123456

> 生成时间：2026-07-06 19:51:38  
> 评估组合数：24

---

## 任务信息

| 项目 | 内容 |
|---|---|
| 任务名称 | 123456 |
| 项目文件夹 | `different_order/123456` |
| CSV 路径 | `different_order/amide-coupling(additive_fixed)_normalized_dataset.csv` |
| 有效样本量 | 47015 |
| SMILES 列 | `['reactant-amide', 'reactant-acid', 'product', 'activation-1', 'activation-2', 'additive', 'base', 'solvent']` |
| 数值辅助列 | `（无）` |
| 标签列 | `yield` |

---

## 结果矩阵（R²）

| 描述符 \ 模型 | xgb | rf | svm | autogluon |
|---|---|---|---|---|
| morgan | 0.6872 | 0.7661 | 0.6428 | 0.8110 |
| maccs | 0.7013 | 0.7585 | 0.6400 | 0.8015 |
| rdkit2d | 0.7454 | 0.7735 | 0.6400 | 0.8080 |
| fisd | 0.7323 | 0.7651 | 0.6293 | 0.7880 |
| molmetalm | 0.7048 | 0.7085 | 0.5738 | 0.7395 |
| maf | 0.6710 | 0.7107 | 0.6449 | 0.7800 |

---

## 详细指标

| 描述符 | 模型 | R² 均值 | R² 标准差 | RMSE | MAE | 用时 |
|---|---|---|---|---|---|---|
| morgan | xgb | 0.6872 | 0.0067 | 0.1686 | 0.1306 | 15.1s |
| morgan | rf | 0.7661 | 0.0036 | 0.1458 | 0.0926 | 14m 18s |
| morgan | svm | 0.6428 | 0.0071 | 0.1801 | 0.1402 | 1m 6s |
| morgan | autogluon | 0.8110 | 0.0000 | 0.1313 | 0.0900 | 6m 22s |
| maccs | xgb | 0.7013 | 0.0048 | 0.1647 | 0.1256 | 4.4s |
| maccs | rf | 0.7585 | 0.0048 | 0.1481 | 0.0957 | 2m 12s |
| maccs | svm | 0.6400 | 0.0081 | 0.1808 | 0.1393 | 45.0s |
| maccs | autogluon | 0.8015 | 0.0000 | 0.1345 | 0.0924 | 5m 2s |
| rdkit2d | xgb | 0.7454 | 0.0067 | 0.1521 | 0.1132 | 6.8s |
| rdkit2d | rf | 0.7735 | 0.0042 | 0.1435 | 0.0909 | 6m 40s |
| rdkit2d | svm | 0.6400 | 0.0087 | 0.1809 | 0.1391 | 42.2s |
| rdkit2d | autogluon | 0.8080 | 0.0000 | 0.1323 | 0.0893 | 5m 3s |
| fisd | xgb | 0.7323 | 0.0066 | 0.1559 | 0.1170 | 4.3s |
| fisd | rf | 0.7651 | 0.0065 | 0.1461 | 0.0992 | 9m 2s |
| fisd | svm | 0.6293 | 0.0053 | 0.1835 | 0.1416 | 45.9s |
| fisd | autogluon | 0.7880 | 0.0000 | 0.1390 | 0.0972 | 5m 5s |
| molmetalm | xgb | 0.7048 | 0.0085 | 0.1638 | 0.1225 | 27.0s |
| molmetalm | rf | 0.7085 | 0.0074 | 0.1627 | 0.1072 | 77m 38s |
| molmetalm | svm | 0.5738 | 0.0064 | 0.1968 | 0.1518 | 1m 7s |
| molmetalm | autogluon | 0.7395 | 0.0000 | 0.1541 | 0.1125 | 5m 3s |
| maf | xgb | 0.6710 | 0.0070 | 0.1729 | 0.1353 | 2.3s |
| maf | rf | 0.7107 | 0.0061 | 0.1621 | 0.1152 | 41.9s |
| maf | svm | 0.6449 | 0.0103 | 0.1796 | 0.1397 | 15.4s |
| maf | autogluon | 0.7800 | 0.0000 | 0.1416 | 0.0996 | 4m 29s |

---

## 推荐组合（加权排名）

| 名次 | 描述符 | 模型 | R² | RMSE | MAE | 综合分 | 用时 | 推荐理由 |
|---|---|---|---|---|---|---|---|---|
| 🥇 1 | **morgan** | **autogluon** | 0.811 | 0.1313 | 0.0900 | 0.587 | 6m 22s | 综合评分 0.587 排名靠前 |
| 🥈 2 | **rdkit2d** | **autogluon** | 0.808 | 0.1323 | 0.0893 | 0.585 | 5m 3s | 综合评分 0.585 排名靠前 |
| 🥉 3 | **maccs** | **autogluon** | 0.802 | 0.1345 | 0.0924 | 0.574 | 5m 2s | 综合评分 0.574 排名靠前 |

<details>
<summary>展开完整排名</summary>

| 名次 | 描述符 | 模型 | R² | RMSE | MAE | 综合分 | 用时 | 推荐理由 |
|---|---|---|---|---|---|---|---|---|
| 1 | morgan | autogluon | 0.811 | 0.1313 | 0.0900 | 0.587 | 6m 22s ||
| 2 | rdkit2d | autogluon | 0.808 | 0.1323 | 0.0893 | 0.585 | 5m 3s ||
| 3 | maccs | autogluon | 0.802 | 0.1345 | 0.0924 | 0.574 | 5m 2s ||
| 4 | fisd | autogluon | 0.788 | 0.1390 | 0.0972 | 0.554 | 5m 5s ||
| 5 | rdkit2d | rf | 0.773 | 0.1435 | 0.0909 | 0.548 | 6m 40s ||
| 6 | maf | autogluon | 0.780 | 0.1416 | 0.0996 | 0.543 | 4m 29s ||
| 7 | morgan | rf | 0.766 | 0.1458 | 0.0926 | 0.539 | 14m 18s ||
| 8 | fisd | rf | 0.765 | 0.1461 | 0.0992 | 0.529 | 9m 2s ||
| 9 | maccs | rf | 0.758 | 0.1481 | 0.0957 | 0.527 | 2m 12s ||
| 10 | rdkit2d | xgb | 0.745 | 0.1521 | 0.1132 | 0.492 | 6.8s ||
| 11 | molmetalm | autogluon | 0.740 | 0.1541 | 0.1125 | 0.487 | 5m 3s ||
| 12 | fisd | xgb | 0.732 | 0.1559 | 0.1170 | 0.474 | 4.3s ||
| 13 | molmetalm | rf | 0.708 | 0.1627 | 0.1072 | 0.465 | 77m 38s ||
| 14 | maf | rf | 0.711 | 0.1621 | 0.1152 | 0.456 | 41.9s ||
| 15 | molmetalm | xgb | 0.705 | 0.1638 | 0.1225 | 0.441 | 27.0s ||
| 16 | maccs | xgb | 0.701 | 0.1647 | 0.1256 | 0.434 | 4.4s ||
| 17 | morgan | xgb | 0.687 | 0.1686 | 0.1306 | 0.414 | 15.1s ||
| 18 | maf | xgb | 0.671 | 0.1729 | 0.1353 | 0.394 | 2.3s ||
| 19 | maf | svm | 0.645 | 0.1796 | 0.1397 | 0.365 | 15.4s ||
| 20 | morgan | svm | 0.643 | 0.1801 | 0.1402 | 0.362 | 1m 6s ||
| 21 | rdkit2d | svm | 0.640 | 0.1809 | 0.1391 | 0.361 | 42.2s ||
| 22 | maccs | svm | 0.640 | 0.1808 | 0.1393 | 0.361 | 45.0s ||
| 23 | fisd | svm | 0.629 | 0.1835 | 0.1416 | 0.348 | 45.9s ||
| 24 | molmetalm | svm | 0.574 | 0.1968 | 0.1518 | 0.287 | 1m 7s ||

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
| **morgan** | 拼接 | `reactant-amide, reactant-acid, product, activation, additive, base` |
| **maccs** | 拼接 | `reactant-amide, reactant-acid, product, activation, additive, base` |
| **rdkit2d** | 拼接 | `reactant-amide, reactant-acid, product, activation, additive, base` |
| **fisd** | 拼接 | `reactant-amide, reactant-acid, product, activation, additive, base` |
| **molmetalm** | 拼接 | `reactant-amide, reactant-acid, product, activation, additive, base` |
| **maf** | 求和 | `reactant-amide, reactant-acid, product, activation, additive, base` |

---

## 数据路径

| 项目 | 路径 |
|---|---|
| 原始数据集 | `dataset/amide-coupling(additive_fixed).csv` |
| 规范化数据集 | `different_order/amide-coupling(additive_fixed)_normalized_dataset.csv` |
| 配置文件 | `different_order/123456/123456_yonod_config.json` |

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

*由 YONOD report.py 自动生成 · 2026-07-06 19:51:38*