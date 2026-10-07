# 其他来源反应数据

本目录收录未归入 ORD 标准化集合的数据。当前共有 17,904 条本地记录（不含表头），按来源分为 SuFEx、四个原归在 `yieldmaster` 名下的 HTE 数据集，以及 GraphRXN 的自建 Buchwald–Hartwig HTE 表。四个 HTE 数据集现已按反应类型直接放在本目录下。各文件的标签量纲并不完全相同；特别是不能把 LC–MS 响应或产物比直接解释为分离收率。2026-10-07 的[来源与原文基线审计](../../project-docs/docs/dataset_literature_analysis_20261007.md)另有逐文件哈希、引用清单和证据边界。

## 文件与来源

| 本地位置 | 记录数 | 反应/标签 | 数据来源与说明 |
| --- | ---: | --- | --- |
| `SuFEx/SuFExPredictor.csv` | 2,122 | SuFEx；`yield` | Tan *et al.*，[*Data-Driven, Mechanistically Guided Prediction of Yield and Chemoselectivity in SuFEx Reactions*](https://doi.org/10.1021/jacs.6c04549)，*J. Am. Chem. Soc.* **2026**, 148, 24138–24150 的[补充信息（SI）](https://pubs.acs.org/doi/10.1021/jacs.6c04549)。本地表记录碱、反应物、溶剂、温度、时间、添加剂和文献 DOI；论文说明其训练数据来自文献整理，涵盖成功、低收率和失败的 SuFEx 反应。 |
| `Buchwald-Hartwig 胺化（Doyle-Ahneman HTE）/doyle_ahneman_buchwald_hartwig_amination_hte_yield.csv` | 3,955 | Buchwald–Hartwig 胺化；`Yield`（%） | Ismail、Landrum、Riniker 的[原文](https://doi.org/10.1021/jacs.6c02213)将其记作 BH1：Doyle 团队的 HTE 数据，在 DMSO 中以 *p*-toluidine 为固定胺，系统改变 15 个芳基/杂芳基卤化物、22 个添加剂、3 个碱和 4 个配体。 |
| `Buchwald-Hartwig 胺化（Denmark HTE）/denmark_buchwald_hartwig_amination_hte_yield.csv` | 3,359 | Buchwald–Hartwig 胺化；`Yield`（%） | 同一原文将其记作 BH2：Denmark 团队的数据，121 个胺–溴化物组合各在 24 个条件下测得；条件空间涉及 20 个 Pd 配体、3 个溶剂和 3 个碱。 |
| `Suzuki-Miyaura 偶联（SM）/SM.csv` | 5,760 | Suzuki–Miyaura 偶联；`Yield`（%） | 同一原文记作 SM：7 个亲电体、4 个亲核体、11 个 Pd 配体、7 个碱和 4 个溶剂构成的 HTE 数据。本地文件保留完整的 5,760 条记录；原文的描述符比较另行移除了配体或其他试剂信息缺失的行，实际使用 4,620 条，因此两者不可混称。 |
| `SLAP 三组分反应（SL1）/SL1.csv` | 1,150 | SLAP 三组分反应；`Yield` | 同一原文记作 SL1：50 个醛、5 个双官能 SLAP 试剂和 60 个醛反应伙伴的组合，反应条件不变。原文明确指出该标签是 **LC–MS 产物比**，量纲与前三个数据集的收率百分数不同。 |
| `Buchwald-Hartwig C-N 偶联（Doyle-Ahneman Ratio 子集）/doyle_ahneman_buchwald_hartwig_c_n_coupling_ratio_subset.csv` | 1,558 | Buchwald–Hartwig C–N 偶联；`Ratio` | **来源更正**：Li *et al.*，[*A deep learning framework for accurate reaction prediction and its application on high-throughput experimentation data*](https://doi.org/10.1186/s13321-023-00732-w)，*Journal of Cheminformatics* **2023**, 15, 72（GraphRXN）。本轮与[原始补充表](https://media.springernature.com/original/springer-static/esm/art%3A10.1186%2Fs13321-023-00732-w/MediaObjects/13321_2023_732_MOESM1_ESM.xlsx)核对全部 1,558×6 单元格和行序，除 `Output` 改名 `Ratio` 外完全一致。目录和文件沿用历史名称，实际不是 Doyle–Ahneman 子集。标签为经内标当量缩放的 UV 响应比，不是校准分离收率。 |

## 原 `yieldmaster` 四个 HTE 数据集的文献依据

`yieldmaster` 是此前用于汇集这四个文件的本地目录名；目录已扁平化，但其来源文献不变：Ismail, I.; Landrum, G. A.; Riniker, S. [*Yield Smarter, Not Harder: Good Practices for Machine Learning of Reaction Outcomes*](https://doi.org/10.1021/jacs.6c02213), *Journal of the American Chemical Society* **2026**, 148, 31593–31603。原文以这四个公开 HTE 数据集比较不同复杂度的分子描述符，并使用重复 5 折交叉验证；BH1、BH2 和 SM 的目标为收率百分数，SL1 的目标为 LC–MS 产物比。论文的“Data and Code Availability”还指向其 [ETH Research Collection 归档](https://doi.org/10.3929/ethz-c-000800856)，适合需要作者发布代码、处理细节或复现实验的人查阅。

## Buchwald–Hartwig 数据之间的关系

`Buchwald-Hartwig 胺化（Doyle-Ahneman HTE）/doyle_ahneman_buchwald_hartwig_amination_hte_yield.csv` 与 `Buchwald-Hartwig 胺化（Denmark HTE）/denmark_buchwald_hartwig_amination_hte_yield.csv` 是**不同的原始 HTE 数据集**：来源团队、可变反应组分、条件空间和记录数均不同。历史名称带“Doyle-Ahneman Ratio 子集”的 1,558 行表则来源于 GraphRXN 2023 的自建 HTE，是第三个原始实验来源；不能再凭旧目录名认定其与 3,955 行 BH1 同源。前者记录可变溴化物、胺和产物及 UV 响应比，后者记录芳基卤化物、添加剂、碱、配体及 `Yield`，固定胺为 p-toluidine。

SuFEx 表的 `doi` 列有 131 个原始化学文献标识，对应 2,122 行；Tan 2026 是汇编与建模论文。逐引用计数及元数据见独立文档仓库的[SuFEx 引用清单](../../project-docs/docs/dataset_literature_sufex_references_20261007.csv)，各原始化学论文的实验对照尚未逐篇核读。

除非另有明确的许可证或发布说明，本目录的本地副本不应被视为已经获得独立的再分发许可。应按原论文、补充信息、数据归档及其许可证条件使用和引用。
