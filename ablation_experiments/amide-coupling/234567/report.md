# YONOD 建模报告 — 234567

> 生成时间：2026-07-09 19:21:43  
> 评估组合数：22

---

## 任务信息

| 项目 | 内容 |
|---|---|
| 任务名称 | 234567 |
| 项目文件夹 | `/home/wangzh685/桌面/YONOD/ablation_experiments/amide-coupling/234567` |
| CSV 路径 | `/home/wangzh685/桌面/YONOD/ablation_experiments/amide-coupling/amide-coupling(additive_fixed)_normalized_dataset.csv` |
| 有效样本量 | 47015 |
| SMILES 列 | `['reactant-amide', 'reactant-acid', 'product', 'activation-1', 'activation-2', 'additive', 'base', 'solvent']` |
| 数值辅助列 | `（无）` |
| 标签列 | `yield` |

---

## 结果矩阵（R²）

| 描述符 \ 模型 | rf | svm | autogluon | xgb |
|---|---|---|---|---|
| morgan | 0.7635 | 0.6290 | 0.8091 | — |
| maccs | 0.7413 | 0.6043 | 0.7918 | 0.7001 |
| rdkit2d | 0.7645 | 0.6007 | 0.8006 | — |
| fisd | 0.7434 | -0.0218 | 0.7661 | 0.6910 |
| molmetalm | 0.6686 | -0.0220 | 0.7120 | 0.6763 |
| maf | 0.7016 | 0.6127 | 0.7748 | 0.6552 |

---

## 详细指标

| 描述符 | 模型 | R² 均值 | R² 标准差 | RMSE | MAE | 用时 |
|---|---|---|---|---|---|---|
| morgan | rf | 0.7635 | 0.0038 | 0.1466 | 0.0935 | 12m 15s |
| morgan | svm | 0.6290 | 0.0078 | 0.1836 | 0.1435 | 1m 2s |
| morgan | autogluon | 0.8091 | 0.0000 | 0.1319 | 0.0907 | 5m 42s |
| maccs | xgb | 0.7001 | 0.0048 | 0.1651 | 0.1265 | 4.8s |
| maccs | rf | 0.7413 | 0.0052 | 0.1533 | 0.1002 | 1m 56s |
| maccs | svm | 0.6043 | 0.0103 | 0.1896 | 0.1466 | 43.3s |
| maccs | autogluon | 0.7918 | 0.0000 | 0.1378 | 0.0954 | 5m 1s |
| rdkit2d | rf | 0.7645 | 0.0050 | 0.1463 | 0.0936 | 6m 23s |
| rdkit2d | svm | 0.6007 | 0.0100 | 0.1905 | 0.1473 | 41.4s |
| rdkit2d | autogluon | 0.8006 | 0.0000 | 0.1348 | 0.0921 | 5m 3s |
| fisd | xgb | 0.6910 | 0.0057 | 0.1676 | 0.1294 | 4.4s |
| fisd | rf | 0.7434 | 0.0062 | 0.1527 | 0.1059 | 9m 33s |
| fisd | svm | -0.0218 | 0.0056 | 0.3047 | 0.2664 | 1m 29s |
| fisd | autogluon | 0.7661 | 0.0000 | 0.1460 | 0.1043 | 5m 26s |
| molmetalm | xgb | 0.6763 | 0.0081 | 0.1715 | 0.1295 | 26.7s |
| molmetalm | rf | 0.6686 | 0.0052 | 0.1735 | 0.1159 | 66m 18s |
| molmetalm | svm | -0.0220 | 0.0057 | 0.3048 | 0.2664 | 1m 17s |
| molmetalm | autogluon | 0.7120 | 0.0000 | 0.1620 | 0.1187 | 5m 3s |
| maf | xgb | 0.6552 | 0.0064 | 0.1770 | 0.1388 | 2.4s |
| maf | rf | 0.7016 | 0.0048 | 0.1647 | 0.1165 | 46.5s |
| maf | svm | 0.6127 | 0.0118 | 0.1876 | 0.1472 | 15.8s |
| maf | autogluon | 0.7748 | 0.0000 | 0.1433 | 0.1021 | 3m 36s |

---

## 推荐组合（加权排名）

| 名次 | 描述符 | 模型 | R² | RMSE | MAE | 综合分 | 用时 | 推荐理由 |
|---|---|---|---|---|---|---|---|---|
| 🥇 1 | **morgan** | **autogluon** | 0.809 | 0.1319 | 0.0907 | 0.707 | 5m 42s | 综合评分 0.707 排名靠前 |
| 🥈 2 | **rdkit2d** | **autogluon** | 0.801 | 0.1348 | 0.0921 | 0.698 | 5m 3s | 综合评分 0.698 排名靠前 |
| 🥉 3 | **maccs** | **autogluon** | 0.792 | 0.1378 | 0.0954 | 0.689 | 5m 1s | 综合评分 0.689 排名靠前 |

<details>
<summary>展开完整排名</summary>

| 名次 | 描述符 | 模型 | R² | RMSE | MAE | 综合分 | 用时 | 推荐理由 |
|---|---|---|---|---|---|---|---|---|
| 1 | morgan | autogluon | 0.809 | 0.1319 | 0.0907 | 0.707 | 5m 42s ||
| 2 | rdkit2d | autogluon | 0.801 | 0.1348 | 0.0921 | 0.698 | 5m 3s ||
| 3 | maccs | autogluon | 0.792 | 0.1378 | 0.0954 | 0.689 | 5m 1s ||
| 4 | maf | autogluon | 0.775 | 0.1433 | 0.1021 | 0.670 | 3m 36s ||
| 5 | rdkit2d | rf | 0.764 | 0.1463 | 0.0936 | 0.668 | 6m 23s ||
| 6 | morgan | rf | 0.763 | 0.1466 | 0.0935 | 0.667 | 12m 15s ||
| 7 | fisd | autogluon | 0.766 | 0.1460 | 0.1043 | 0.661 | 5m 26s ||
| 8 | maccs | rf | 0.741 | 0.1533 | 0.1002 | 0.645 | 1m 56s ||
| 9 | fisd | rf | 0.743 | 0.1527 | 0.1059 | 0.642 | 9m 33s ||
| 10 | molmetalm | autogluon | 0.712 | 0.1620 | 0.1187 | 0.607 | 5m 3s ||
| 11 | maf | rf | 0.702 | 0.1647 | 0.1165 | 0.601 | 46.5s ||
| 12 | maccs | xgb | 0.700 | 0.1651 | 0.1265 | 0.593 | 4.8s ||
| 13 | fisd | xgb | 0.691 | 0.1676 | 0.1294 | 0.583 | 4.4s ||
| 14 | molmetalm | rf | 0.669 | 0.1735 | 0.1159 | 0.576 | 66m 18s ||
| 15 | molmetalm | xgb | 0.676 | 0.1715 | 0.1295 | 0.572 | 26.7s ||
| 16 | maf | xgb | 0.655 | 0.1770 | 0.1388 | 0.549 | 2.4s ||
| 17 | morgan | svm | 0.629 | 0.1836 | 0.1435 | 0.526 | 1m 2s ||
| 18 | maf | svm | 0.613 | 0.1876 | 0.1472 | 0.511 | 15.8s ||
| 19 | maccs | svm | 0.604 | 0.1896 | 0.1466 | 0.505 | 43.3s ||
| 20 | rdkit2d | svm | 0.601 | 0.1905 | 0.1473 | 0.502 | 41.4s ||
| 21 | fisd | svm | -0.022 | 0.3047 | 0.2664 | -0.011 | 1m 29s ||
| 22 | molmetalm | svm | -0.022 | 0.3048 | 0.2664 | -0.011 | 1m 17s ||

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
| **morgan** | 拼接 | `reactant-acid, product, activation, additive, base, solvent` |
| **maccs** | 拼接 | `reactant-acid, product, activation, additive, base, solvent` |
| **rdkit2d** | 拼接 | `reactant-acid, product, activation, additive, base, solvent` |
| **fisd** | 拼接 | `reactant-acid, product, activation, additive, base, solvent` |
| **molmetalm** | 拼接 | `reactant-acid, product, activation, additive, base, solvent` |
| **maf** | 求和 | `reactant-acid, product, activation, additive, base, solvent` |

---

## 数据路径

| 项目 | 路径 |
|---|---|
| 原始数据集 | `dataset/amide-coupling(additive_fixed).csv` |
| 规范化数据集 | `/home/wangzh685/桌面/YONOD/ablation_experiments/amide-coupling/amide-coupling(additive_fixed)_normalized_dataset.csv` |
| 配置文件 | `/home/wangzh685/桌面/YONOD/ablation_experiments/amide-coupling/234567/234567_yonod_config.json` |

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

*由 YONOD report.py 自动生成 · 2026-07-09 19:21:43*