# YONOD 建模报告 — 123467

> 生成时间：2026-07-07 00:43:56  
> 评估组合数：24

---

## 任务信息

| 项目 | 内容 |
|---|---|
| 任务名称 | 123467 |
| 项目文件夹 | `different_order/123467` |
| CSV 路径 | `different_order/amide-coupling(additive_fixed)_normalized_dataset.csv` |
| 有效样本量 | 47015 |
| SMILES 列 | `['reactant-amide', 'reactant-acid', 'product', 'activation-1', 'activation-2', 'additive', 'base', 'solvent']` |
| 数值辅助列 | `（无）` |
| 标签列 | `yield` |

---

## 结果矩阵（R²）

| 描述符 \ 模型 | xgb | rf | svm | autogluon |
|---|---|---|---|---|
| morgan | 0.5564 | 0.5298 | 0.4893 | 0.6153 |
| maccs | 0.5679 | 0.5316 | 0.4948 | 0.6119 |
| rdkit2d | 0.5987 | 0.5355 | 0.4950 | 0.6183 |
| fisd | 0.5835 | 0.5843 | -0.0218 | 0.6069 |
| molmetalm | 0.5631 | 0.4789 | -0.0220 | 0.5749 |
| maf | 0.5505 | 0.5130 | 0.5041 | 0.5931 |

---

## 详细指标

| 描述符 | 模型 | R² 均值 | R² 标准差 | RMSE | MAE | 用时 |
|---|---|---|---|---|---|---|
| morgan | xgb | 0.5564 | 0.0062 | 0.2008 | 0.1580 | 14.9s |
| morgan | rf | 0.5298 | 0.0126 | 0.2067 | 0.1443 | 13m 31s |
| morgan | svm | 0.4893 | 0.0090 | 0.2154 | 0.1669 | 1m 9s |
| morgan | autogluon | 0.6153 | 0.0000 | 0.1873 | 0.1395 | 5m 0s |
| maccs | xgb | 0.5679 | 0.0078 | 0.1981 | 0.1539 | 4.4s |
| maccs | rf | 0.5316 | 0.0134 | 0.2063 | 0.1448 | 1m 49s |
| maccs | svm | 0.4948 | 0.0092 | 0.2142 | 0.1653 | 45.8s |
| maccs | autogluon | 0.6119 | 0.0000 | 0.1881 | 0.1394 | 2m 30s |
| rdkit2d | xgb | 0.5987 | 0.0093 | 0.1909 | 0.1444 | 6.6s |
| rdkit2d | rf | 0.5355 | 0.0128 | 0.2054 | 0.1429 | 5m 24s |
| rdkit2d | svm | 0.4950 | 0.0072 | 0.2142 | 0.1649 | 49.8s |
| rdkit2d | autogluon | 0.6183 | 0.0000 | 0.1865 | 0.1383 | 5m 21s |
| fisd | xgb | 0.5835 | 0.0088 | 0.1945 | 0.1493 | 4.4s |
| fisd | rf | 0.5843 | 0.0097 | 0.1944 | 0.1411 | 9m 3s |
| fisd | svm | -0.0218 | 0.0056 | 0.3047 | 0.2664 | 1m 20s |
| fisd | autogluon | 0.6069 | 0.0000 | 0.1893 | 0.1407 | 4m 59s |
| molmetalm | xgb | 0.5631 | 0.0078 | 0.1993 | 0.1524 | 27.0s |
| molmetalm | rf | 0.4789 | 0.0118 | 0.2176 | 0.1541 | 65m 35s |
| molmetalm | svm | -0.0220 | 0.0056 | 0.3047 | 0.2664 | 1m 19s |
| molmetalm | autogluon | 0.5749 | 0.0000 | 0.1969 | 0.1492 | 5m 3s |
| maf | xgb | 0.5505 | 0.0068 | 0.2021 | 0.1587 | 2.3s |
| maf | rf | 0.5130 | 0.0126 | 0.2104 | 0.1501 | 34.7s |
| maf | svm | 0.5041 | 0.0085 | 0.2123 | 0.1655 | 17.3s |
| maf | autogluon | 0.5931 | 0.0000 | 0.1926 | 0.1449 | 1m 27s |

---

## 推荐组合（加权排名）

| 名次 | 描述符 | 模型 | R² | RMSE | MAE | 综合分 | 用时 | 推荐理由 |
|---|---|---|---|---|---|---|---|---|
| 🥇 1 | **rdkit2d** | **autogluon** | 0.618 | 0.1865 | 0.1383 | 0.522 | 5m 21s | 综合评分 0.522 排名靠前 |
| 🥈 2 | **morgan** | **autogluon** | 0.615 | 0.1873 | 0.1395 | 0.519 | 5m 0s | 综合评分 0.519 排名靠前 |
| 🥉 3 | **maccs** | **autogluon** | 0.612 | 0.1881 | 0.1394 | 0.516 | 2m 30s | 综合评分 0.516 排名靠前 |

<details>
<summary>展开完整排名</summary>

| 名次 | 描述符 | 模型 | R² | RMSE | MAE | 综合分 | 用时 | 推荐理由 |
|---|---|---|---|---|---|---|---|---|
| 1 | rdkit2d | autogluon | 0.618 | 0.1865 | 0.1383 | 0.522 | 5m 21s ||
| 2 | morgan | autogluon | 0.615 | 0.1873 | 0.1395 | 0.519 | 5m 0s ||
| 3 | maccs | autogluon | 0.612 | 0.1881 | 0.1394 | 0.516 | 2m 30s ||
| 4 | fisd | autogluon | 0.607 | 0.1893 | 0.1407 | 0.511 | 4m 59s ||
| 5 | rdkit2d | xgb | 0.599 | 0.1909 | 0.1444 | 0.503 | 6.6s ||
| 6 | maf | autogluon | 0.593 | 0.1926 | 0.1449 | 0.498 | 1m 27s ||
| 7 | fisd | rf | 0.584 | 0.1944 | 0.1411 | 0.495 | 9m 3s ||
| 8 | fisd | xgb | 0.584 | 0.1945 | 0.1493 | 0.488 | 4.4s ||
| 9 | molmetalm | autogluon | 0.575 | 0.1969 | 0.1492 | 0.482 | 5m 3s ||
| 10 | maccs | xgb | 0.568 | 0.1981 | 0.1539 | 0.473 | 4.4s ||
| 11 | molmetalm | xgb | 0.563 | 0.1993 | 0.1524 | 0.471 | 27.0s ||
| 12 | morgan | xgb | 0.556 | 0.2008 | 0.1580 | 0.462 | 14.9s ||
| 13 | rdkit2d | rf | 0.536 | 0.2054 | 0.1429 | 0.458 | 5m 24s ||
| 14 | maf | xgb | 0.551 | 0.2021 | 0.1587 | 0.457 | 2.3s ||
| 15 | maccs | rf | 0.532 | 0.2063 | 0.1448 | 0.454 | 1m 49s ||
| 16 | morgan | rf | 0.530 | 0.2067 | 0.1443 | 0.453 | 13m 31s ||
| 17 | maf | rf | 0.513 | 0.2104 | 0.1501 | 0.437 | 34.7s ||
| 18 | maf | svm | 0.504 | 0.2123 | 0.1655 | 0.419 | 17.3s ||
| 19 | rdkit2d | svm | 0.495 | 0.2142 | 0.1649 | 0.413 | 49.8s ||
| 20 | maccs | svm | 0.495 | 0.2142 | 0.1653 | 0.412 | 45.8s ||
| 21 | molmetalm | rf | 0.479 | 0.2176 | 0.1541 | 0.410 | 65m 35s ||
| 22 | morgan | svm | 0.489 | 0.2154 | 0.1669 | 0.407 | 1m 9s ||
| 23 | fisd | svm | -0.022 | 0.3047 | 0.2664 | -0.011 | 1m 20s ||
| 24 | molmetalm | svm | -0.022 | 0.3047 | 0.2664 | -0.011 | 1m 19s ||

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
| **morgan** | 拼接 | `reactant-amide, reactant-acid, product, activation, base, solvent` |
| **maccs** | 拼接 | `reactant-amide, reactant-acid, product, activation, base, solvent` |
| **rdkit2d** | 拼接 | `reactant-amide, reactant-acid, product, activation, base, solvent` |
| **fisd** | 拼接 | `reactant-amide, reactant-acid, product, activation, base, solvent` |
| **molmetalm** | 拼接 | `reactant-amide, reactant-acid, product, activation, base, solvent` |
| **maf** | 求和 | `reactant-amide, reactant-acid, product, activation, base, solvent` |

---

## 数据路径

| 项目 | 路径 |
|---|---|
| 原始数据集 | `dataset/amide-coupling(additive_fixed).csv` |
| 规范化数据集 | `different_order/amide-coupling(additive_fixed)_normalized_dataset.csv` |
| 配置文件 | `different_order/123467/123467_yonod_config.json` |

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

*由 YONOD report.py 自动生成 · 2026-07-07 00:43:56*