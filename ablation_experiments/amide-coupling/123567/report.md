# YONOD 建模报告 — 123567

> 生成时间：2026-07-07 02:41:28  
> 评估组合数：24

---

## 任务信息

| 项目 | 内容 |
|---|---|
| 任务名称 | 123567 |
| 项目文件夹 | `different_order/123567` |
| CSV 路径 | `different_order/amide-coupling(additive_fixed)_normalized_dataset.csv` |
| 有效样本量 | 47015 |
| SMILES 列 | `['reactant-amide', 'reactant-acid', 'product', 'activation-1', 'activation-2', 'additive', 'base', 'solvent']` |
| 数值辅助列 | `（无）` |
| 标签列 | `yield` |

---

## 结果矩阵（R²）

| 描述符 \ 模型 | xgb | rf | svm | autogluon |
|---|---|---|---|---|
| morgan | 0.4624 | 0.4459 | 0.3880 | 0.5223 |
| maccs | 0.4736 | 0.4435 | 0.3943 | 0.5206 |
| rdkit2d | 0.4952 | 0.4438 | 0.3901 | 0.5246 |
| fisd | 0.4770 | 0.4809 | -0.0218 | 0.5120 |
| molmetalm | 0.4608 | 0.3725 | -0.0220 | 0.4827 |
| maf | 0.4538 | 0.4287 | 0.4038 | 0.5055 |

---

## 详细指标

| 描述符 | 模型 | R² 均值 | R² 标准差 | RMSE | MAE | 用时 |
|---|---|---|---|---|---|---|
| morgan | xgb | 0.4624 | 0.0126 | 0.2210 | 0.1753 | 13.1s |
| morgan | rf | 0.4459 | 0.0141 | 0.2244 | 0.1630 | 12m 15s |
| morgan | svm | 0.3880 | 0.0175 | 0.2358 | 0.1812 | 1m 8s |
| morgan | autogluon | 0.5223 | 0.0000 | 0.2087 | 0.1579 | 4m 43s |
| maccs | xgb | 0.4736 | 0.0111 | 0.2187 | 0.1711 | 4.0s |
| maccs | rf | 0.4435 | 0.0148 | 0.2249 | 0.1642 | 1m 22s |
| maccs | svm | 0.3943 | 0.0166 | 0.2346 | 0.1796 | 44.7s |
| maccs | autogluon | 0.5206 | 0.0000 | 0.2091 | 0.1587 | 2m 6s |
| rdkit2d | xgb | 0.4952 | 0.0126 | 0.2142 | 0.1649 | 6.0s |
| rdkit2d | rf | 0.4438 | 0.0151 | 0.2248 | 0.1633 | 4m 11s |
| rdkit2d | svm | 0.3901 | 0.0166 | 0.2354 | 0.1802 | 51.3s |
| rdkit2d | autogluon | 0.5246 | 0.0000 | 0.2082 | 0.1573 | 4m 59s |
| fisd | xgb | 0.4770 | 0.0133 | 0.2180 | 0.1701 | 4.1s |
| fisd | rf | 0.4809 | 0.0134 | 0.2172 | 0.1635 | 7m 26s |
| fisd | svm | -0.0218 | 0.0056 | 0.3047 | 0.2664 | 56.3s |
| fisd | autogluon | 0.5120 | 0.0000 | 0.2109 | 0.1621 | 5m 1s |
| molmetalm | xgb | 0.4608 | 0.0139 | 0.2213 | 0.1726 | 22.8s |
| molmetalm | rf | 0.3725 | 0.0159 | 0.2388 | 0.1736 | 49m 57s |
| molmetalm | svm | -0.0220 | 0.0056 | 0.3047 | 0.2664 | 1m 20s |
| molmetalm | autogluon | 0.4827 | 0.0000 | 0.2172 | 0.1664 | 5m 2s |
| maf | xgb | 0.4538 | 0.0117 | 0.2228 | 0.1774 | 2.3s |
| maf | rf | 0.4287 | 0.0132 | 0.2278 | 0.1694 | 28.3s |
| maf | svm | 0.4038 | 0.0181 | 0.2327 | 0.1807 | 16.6s |
| maf | autogluon | 0.5055 | 0.0000 | 0.2123 | 0.1629 | 1m 21s |

---

## 推荐组合（加权排名）

| 名次 | 描述符 | 模型 | R² | RMSE | MAE | 综合分 | 用时 | 推荐理由 |
|---|---|---|---|---|---|---|---|---|
| 🥇 1 | **rdkit2d** | **autogluon** | 0.525 | 0.2082 | 0.1573 | 0.439 | 4m 59s | 综合评分 0.439 排名靠前 |
| 🥈 2 | **morgan** | **autogluon** | 0.522 | 0.2087 | 0.1579 | 0.437 | 4m 43s | 综合评分 0.437 排名靠前 |
| 🥉 3 | **maccs** | **autogluon** | 0.521 | 0.2091 | 0.1587 | 0.435 | 2m 6s | 综合评分 0.435 排名靠前 |

<details>
<summary>展开完整排名</summary>

| 名次 | 描述符 | 模型 | R² | RMSE | MAE | 综合分 | 用时 | 推荐理由 |
|---|---|---|---|---|---|---|---|---|
| 1 | rdkit2d | autogluon | 0.525 | 0.2082 | 0.1573 | 0.439 | 4m 59s ||
| 2 | morgan | autogluon | 0.522 | 0.2087 | 0.1579 | 0.437 | 4m 43s ||
| 3 | maccs | autogluon | 0.521 | 0.2091 | 0.1587 | 0.435 | 2m 6s ||
| 4 | fisd | autogluon | 0.512 | 0.2109 | 0.1621 | 0.427 | 5m 1s ||
| 5 | maf | autogluon | 0.505 | 0.2123 | 0.1629 | 0.421 | 1m 21s ||
| 6 | rdkit2d | xgb | 0.495 | 0.2142 | 0.1649 | 0.413 | 6.0s ||
| 7 | fisd | rf | 0.481 | 0.2172 | 0.1635 | 0.404 | 7m 26s ||
| 8 | molmetalm | autogluon | 0.483 | 0.2172 | 0.1664 | 0.403 | 5m 2s ||
| 9 | fisd | xgb | 0.477 | 0.2180 | 0.1701 | 0.396 | 4.1s ||
| 10 | maccs | xgb | 0.474 | 0.2187 | 0.1711 | 0.393 | 4.0s ||
| 11 | molmetalm | xgb | 0.461 | 0.2213 | 0.1726 | 0.383 | 22.8s ||
| 12 | morgan | xgb | 0.462 | 0.2210 | 0.1753 | 0.382 | 13.1s ||
| 13 | morgan | rf | 0.446 | 0.2244 | 0.1630 | 0.380 | 12m 15s ||
| 14 | rdkit2d | rf | 0.444 | 0.2248 | 0.1633 | 0.378 | 4m 11s ||
| 15 | maccs | rf | 0.443 | 0.2249 | 0.1642 | 0.377 | 1m 22s ||
| 16 | maf | xgb | 0.454 | 0.2228 | 0.1774 | 0.374 | 2.3s ||
| 17 | maf | rf | 0.429 | 0.2278 | 0.1694 | 0.363 | 28.3s ||
| 18 | maf | svm | 0.404 | 0.2327 | 0.1807 | 0.337 | 16.6s ||
| 19 | maccs | svm | 0.394 | 0.2346 | 0.1796 | 0.331 | 44.7s ||
| 20 | rdkit2d | svm | 0.390 | 0.2354 | 0.1802 | 0.328 | 51.3s ||
| 21 | morgan | svm | 0.388 | 0.2358 | 0.1812 | 0.326 | 1m 8s ||
| 22 | molmetalm | rf | 0.372 | 0.2388 | 0.1736 | 0.321 | 49m 57s ||
| 23 | fisd | svm | -0.022 | 0.3047 | 0.2664 | -0.011 | 56.3s ||
| 24 | molmetalm | svm | -0.022 | 0.3047 | 0.2664 | -0.011 | 1m 20s ||

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
| **morgan** | 拼接 | `reactant-amide, reactant-acid, product, additive, base, solvent` |
| **maccs** | 拼接 | `reactant-amide, reactant-acid, product, additive, base, solvent` |
| **rdkit2d** | 拼接 | `reactant-amide, reactant-acid, product, additive, base, solvent` |
| **fisd** | 拼接 | `reactant-amide, reactant-acid, product, additive, base, solvent` |
| **molmetalm** | 拼接 | `reactant-amide, reactant-acid, product, additive, base, solvent` |
| **maf** | 求和 | `reactant-amide, reactant-acid, product, additive, base, solvent` |

---

## 数据路径

| 项目 | 路径 |
|---|---|
| 原始数据集 | `dataset/amide-coupling(additive_fixed).csv` |
| 规范化数据集 | `different_order/amide-coupling(additive_fixed)_normalized_dataset.csv` |
| 配置文件 | `different_order/123567/123567_yonod_config.json` |

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

*由 YONOD report.py 自动生成 · 2026-07-07 02:41:28*