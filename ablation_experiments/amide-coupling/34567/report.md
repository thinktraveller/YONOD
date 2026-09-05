# YONOD 建模报告 — 134567

> 生成时间：2026-07-09 16:55:20  
> 评估组合数：24

---

## 任务信息

| 项目 | 内容 |
|---|---|
| 任务名称 | 134567 |
| 项目文件夹 | `/home/wangzh685/桌面/YONOD/ablation_experiments/amide-coupling/134567` |
| CSV 路径 | `/home/wangzh685/桌面/YONOD/ablation_experiments/amide-coupling/amide-coupling(additive_fixed)_normalized_dataset.csv` |
| 有效样本量 | 47015 |
| SMILES 列 | `['reactant-amide', 'reactant-acid', 'product', 'activation-1', 'activation-2', 'additive', 'base', 'solvent']` |
| 数值辅助列 | `（无）` |
| 标签列 | `yield` |

---

## 结果矩阵（R²）

| 描述符 \ 模型 | xgb | rf | svm | autogluon |
|---|---|---|---|---|
| morgan | 0.6841 | 0.7662 | 0.6499 | 0.8103 |
| maccs | 0.6929 | 0.7554 | 0.6272 | 0.7989 |
| rdkit2d | 0.7398 | 0.7724 | 0.6261 | 0.8080 |
| fisd | 0.7131 | 0.7622 | -0.0218 | 0.7834 |
| molmetalm | 0.7001 | 0.7066 | -0.0220 | 0.7348 |
| maf | 0.6694 | 0.7274 | 0.6472 | 0.7831 |

---

## 详细指标

| 描述符 | 模型 | R² 均值 | R² 标准差 | RMSE | MAE | 用时 |
|---|---|---|---|---|---|---|
| morgan | xgb | 0.6841 | 0.0076 | 0.1694 | 0.1312 | 19.2s |
| morgan | rf | 0.7662 | 0.0036 | 0.1458 | 0.0925 | 11m 11s |
| morgan | svm | 0.6499 | 0.0067 | 0.1784 | 0.1383 | 1m 5s |
| morgan | autogluon | 0.8103 | 0.0000 | 0.1315 | 0.0898 | 5m 41s |
| maccs | xgb | 0.6929 | 0.0076 | 0.1670 | 0.1277 | 4.5s |
| maccs | rf | 0.7554 | 0.0044 | 0.1491 | 0.0967 | 1m 58s |
| maccs | svm | 0.6272 | 0.0067 | 0.1840 | 0.1411 | 41.3s |
| maccs | autogluon | 0.7989 | 0.0000 | 0.1354 | 0.0936 | 5m 2s |
| rdkit2d | xgb | 0.7398 | 0.0078 | 0.1538 | 0.1147 | 7.1s |
| rdkit2d | rf | 0.7724 | 0.0043 | 0.1438 | 0.0913 | 6m 8s |
| rdkit2d | svm | 0.6261 | 0.0062 | 0.1843 | 0.1411 | 43.8s |
| rdkit2d | autogluon | 0.8080 | 0.0000 | 0.1323 | 0.0899 | 5m 2s |
| fisd | xgb | 0.7131 | 0.0065 | 0.1615 | 0.1217 | 4.5s |
| fisd | rf | 0.7622 | 0.0079 | 0.1470 | 0.1009 | 9m 30s |
| fisd | svm | -0.0218 | 0.0056 | 0.3047 | 0.2664 | 1m 22s |
| fisd | autogluon | 0.7834 | 0.0000 | 0.1405 | 0.0989 | 6m 52s |
| molmetalm | xgb | 0.7001 | 0.0092 | 0.1651 | 0.1237 | 55.2s |
| molmetalm | rf | 0.7066 | 0.0079 | 0.1633 | 0.1080 | 71m 31s |
| molmetalm | svm | -0.0220 | 0.0058 | 0.3048 | 0.2664 | 1m 13s |
| molmetalm | autogluon | 0.7348 | 0.0000 | 0.1555 | 0.1130 | 5m 3s |
| maf | xgb | 0.6694 | 0.0075 | 0.1733 | 0.1347 | 2.4s |
| maf | rf | 0.7274 | 0.0053 | 0.1574 | 0.1099 | 50.4s |
| maf | svm | 0.6472 | 0.0084 | 0.1790 | 0.1382 | 16.1s |
| maf | autogluon | 0.7831 | 0.0000 | 0.1406 | 0.0987 | 3m 44s |

---

## 推荐组合（加权排名）

| 名次 | 描述符 | 模型 | R² | RMSE | MAE | 综合分 | 用时 | 推荐理由 |
|---|---|---|---|---|---|---|---|---|
| 🥇 1 | **morgan** | **autogluon** | 0.810 | 0.1315 | 0.0898 | 0.708 | 5m 41s | 综合评分 0.708 排名靠前 |
| 🥈 2 | **rdkit2d** | **autogluon** | 0.808 | 0.1323 | 0.0899 | 0.706 | 5m 2s | 综合评分 0.706 排名靠前 |
| 🥉 3 | **maccs** | **autogluon** | 0.799 | 0.1354 | 0.0936 | 0.696 | 5m 2s | 综合评分 0.696 排名靠前 |

<details>
<summary>展开完整排名</summary>

| 名次 | 描述符 | 模型 | R² | RMSE | MAE | 综合分 | 用时 | 推荐理由 |
|---|---|---|---|---|---|---|---|---|
| 1 | morgan | autogluon | 0.810 | 0.1315 | 0.0898 | 0.708 | 5m 41s ||
| 2 | rdkit2d | autogluon | 0.808 | 0.1323 | 0.0899 | 0.706 | 5m 2s ||
| 3 | maccs | autogluon | 0.799 | 0.1354 | 0.0936 | 0.696 | 5m 2s ||
| 4 | fisd | autogluon | 0.783 | 0.1405 | 0.0989 | 0.679 | 6m 52s ||
| 5 | maf | autogluon | 0.783 | 0.1406 | 0.0987 | 0.679 | 3m 44s ||
| 6 | rdkit2d | rf | 0.772 | 0.1438 | 0.0913 | 0.676 | 6m 8s ||
| 7 | morgan | rf | 0.766 | 0.1458 | 0.0925 | 0.670 | 11m 11s ||
| 8 | fisd | rf | 0.762 | 0.1470 | 0.1009 | 0.661 | 9m 30s ||
| 9 | maccs | rf | 0.755 | 0.1491 | 0.0967 | 0.658 | 1m 58s ||
| 10 | rdkit2d | xgb | 0.740 | 0.1538 | 0.1147 | 0.632 | 7.1s ||
| 11 | molmetalm | autogluon | 0.735 | 0.1555 | 0.1130 | 0.629 | 5m 3s ||
| 12 | maf | rf | 0.727 | 0.1574 | 0.1099 | 0.626 | 50.4s ||
| 13 | molmetalm | rf | 0.707 | 0.1633 | 0.1080 | 0.612 | 71m 31s ||
| 14 | fisd | xgb | 0.713 | 0.1615 | 0.1217 | 0.606 | 4.5s ||
| 15 | molmetalm | xgb | 0.700 | 0.1651 | 0.1237 | 0.595 | 55.2s ||
| 16 | maccs | xgb | 0.693 | 0.1670 | 0.1277 | 0.586 | 4.5s ||
| 17 | morgan | xgb | 0.684 | 0.1694 | 0.1312 | 0.577 | 19.2s ||
| 18 | maf | xgb | 0.669 | 0.1733 | 0.1347 | 0.563 | 2.4s ||
| 19 | morgan | svm | 0.650 | 0.1784 | 0.1383 | 0.546 | 1m 5s ||
| 20 | maf | svm | 0.647 | 0.1790 | 0.1382 | 0.544 | 16.1s ||
| 21 | maccs | svm | 0.627 | 0.1840 | 0.1411 | 0.527 | 41.3s ||
| 22 | rdkit2d | svm | 0.626 | 0.1843 | 0.1411 | 0.526 | 43.8s ||
| 23 | fisd | svm | -0.022 | 0.3047 | 0.2664 | -0.011 | 1m 22s ||
| 24 | molmetalm | svm | -0.022 | 0.3048 | 0.2664 | -0.011 | 1m 13s ||

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
| **morgan** | 拼接 | `reactant-amide, product, activation, additive, base, solvent` |
| **maccs** | 拼接 | `reactant-amide, product, activation, additive, base, solvent` |
| **rdkit2d** | 拼接 | `reactant-amide, product, activation, additive, base, solvent` |
| **fisd** | 拼接 | `reactant-amide, product, activation, additive, base, solvent` |
| **molmetalm** | 拼接 | `reactant-amide, product, activation, additive, base, solvent` |
| **maf** | 求和 | `reactant-amide, product, activation, additive, base, solvent` |

---

## 数据路径

| 项目 | 路径 |
|---|---|
| 原始数据集 | `dataset/amide-coupling(additive_fixed).csv` |
| 规范化数据集 | `/home/wangzh685/桌面/YONOD/ablation_experiments/amide-coupling/amide-coupling(additive_fixed)_normalized_dataset.csv` |
| 配置文件 | `/home/wangzh685/桌面/YONOD/ablation_experiments/amide-coupling/134567/134567_yonod_config.json` |

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

*由 YONOD report.py 自动生成 · 2026-07-09 16:55:20*