# 其他来源反应数据

本目录收录未归入 ORD 标准化集合的数据。当前共有 17,904 条本地记录（不含表头），按来源分为 SuFEx、四个原归在 `yieldmaster` 名下的 HTE 数据集，以及一个 Buchwald–Hartwig 外部基准子集。四个 HTE 数据集现已按反应类型直接放在本目录下。各文件的标签量纲并不完全相同；特别是不能把 LC–MS 响应或产物比直接解释为分离收率。

## 文件与来源

| 本地位置 | 记录数 | 反应/标签 | 数据来源与说明 |
| --- | ---: | --- | --- |
| `SuFEx/SuFExPredictor.csv` | 2,122 | SuFEx；`yield` | Tan *et al.*，[*Data-Driven, Mechanistically Guided Prediction of Yield and Chemoselectivity in SuFEx Reactions*](https://doi.org/10.1021/jacs.6c04549)，*J. Am. Chem. Soc.* **2026**, 148, 24138–24150 的[补充信息（SI）](https://pubs.acs.org/doi/10.1021/jacs.6c04549)。本地表记录碱、反应物、溶剂、温度、时间、添加剂和文献 DOI；论文说明其训练数据来自文献整理，涵盖成功、低收率和失败的 SuFEx 反应。 |
| `Buchwald-Hartwig 胺化（Doyle-Ahneman HTE）/doyle_ahneman_buchwald_hartwig_amination_hte_yield.csv` | 3,955 | Buchwald–Hartwig 胺化；`Yield`（%） | Ismail、Landrum、Riniker 的[原文](https://doi.org/10.1021/jacs.6c02213)将其记作 BH1：Doyle 团队的 HTE 数据，在 DMSO 中以 *p*-toluidine 为固定胺，系统改变 15 个芳基/杂芳基卤化物、22 个添加剂、3 个碱和 4 个配体。 |
| `Buchwald-Hartwig 胺化（Denmark HTE）/denmark_buchwald_hartwig_amination_hte_yield.csv` | 3,359 | Buchwald–Hartwig 胺化；`Yield`（%） | 同一原文将其记作 BH2：Denmark 团队的数据，121 个胺–溴化物组合各在 24 个条件下测得；条件空间涉及 20 个 Pd 配体、3 个溶剂和 3 个碱。 |
| `Suzuki-Miyaura 偶联（SM）/SM.csv` | 5,760 | Suzuki–Miyaura 偶联；`Yield`（%） | 同一原文记作 SM：7 个亲电体、4 个亲核体、11 个 Pd 配体、7 个碱和 4 个溶剂构成的 HTE 数据。本地文件保留完整的 5,760 条记录；原文的描述符比较另行移除了配体或其他试剂信息缺失的行，实际使用 4,620 条，因此两者不可混称。 |
| `SLAP 三组分反应（SL1）/SL1.csv` | 1,150 | SLAP 三组分反应；`Yield` | 同一原文记作 SL1：50 个醛、5 个双官能 SLAP 试剂和 60 个醛反应伙伴的组合，反应条件不变。原文明确指出该标签是 **LC–MS 产物比**，量纲与前三个数据集的收率百分数不同。 |
| `Buchwald-Hartwig C-N 偶联（Doyle-Ahneman Ratio 子集）/doyle_ahneman_buchwald_hartwig_c_n_coupling_ratio_subset.csv` | 1,558 | Buchwald–Hartwig C–N 偶联；`Ratio` | Ahneman *et al.*，[*Predicting Reaction Performance in C–N Cross-Coupling Using Machine Learning*](https://doi.org/10.1126/science.aar5169)，*Science* **2018**, 360, 186–190；公开整理数据见 [doyle-lab-ucla/ochem-data 的 `CN` 数据](https://github.com/doyle-lab-ucla/ochem-data/tree/main/CN)。这是外部基准数据，不是廖矿标课题组论文；本地 CSV 是适配后的子集/表格版本，不能与原始 HTE 全量记录逐条等同。 |

## 原 `yieldmaster` 四个 HTE 数据集的文献依据

`yieldmaster` 是此前用于汇集这四个文件的本地目录名；目录已扁平化，但其来源文献不变：Ismail, I.; Landrum, G. A.; Riniker, S. [*Yield Smarter, Not Harder: Good Practices for Machine Learning of Reaction Outcomes*](https://doi.org/10.1021/jacs.6c02213), *Journal of the American Chemical Society* **2026**, 148, 31593–31603。原文以这四个公开 HTE 数据集比较不同复杂度的分子描述符，并使用重复 5 折交叉验证；BH1、BH2 和 SM 的目标为收率百分数，SL1 的目标为 LC–MS 产物比。论文的“Data and Code Availability”还指向其 [ETH Research Collection 归档](https://doi.org/10.3929/ethz-c-000800856)，适合需要作者发布代码、处理细节或复现实验的人查阅。

## Buchwald–Hartwig 数据之间的关系

`Buchwald-Hartwig 胺化（Doyle-Ahneman HTE）/doyle_ahneman_buchwald_hartwig_amination_hte_yield.csv` 与 `Buchwald-Hartwig 胺化（Denmark HTE）/denmark_buchwald_hartwig_amination_hte_yield.csv` 是**不同的原始 HTE 数据集**：来源团队、可变反应组分、条件空间和记录数均不同。`Buchwald-Hartwig C-N 偶联（Doyle-Ahneman Ratio 子集）/doyle_ahneman_buchwald_hartwig_c_n_coupling_ratio_subset.csv` 则是第三份**不同的本地表**，但它与 Doyle-Ahneman HTE 表都追溯到 Ahneman/Doyle 的 C–N 偶联数据脉络；前者是 1,558 条、以 `Ratio`、溴化物和胺为核心字段的适配子集/表格版本，后者是 3,955 条、包含添加剂、碱、配体和 `Yield` 的 HTE 表。因此，三份本地文件应作为不同输入表处理，但不能把这两份 Doyle-Ahneman 派生表误称为彼此独立的原始实验数据来源。

除非另有明确的许可证或发布说明，本目录的本地副本不应被视为已经获得独立的再分发许可。应按原论文、补充信息、数据归档及其许可证条件使用和引用。
