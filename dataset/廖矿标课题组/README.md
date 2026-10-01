# 廖矿标课题组反应数据集：文献与公开获取位置

本目录收录 8 类反应数据。下表将本地文件与其**原始/对应文献**、以及作者公开的数据或代码位置对应起来，方便引用、复现与追溯。

## 使用说明

- `✅ 代码与数据仓库`：作者或研究团队公开的可复现仓库；优先从该处获取原始版本。
- `📄 补充信息（SI）`：原始实验/HTE 数据在出版社公开的补充信息中；这表示数据可访问，**不等同于代码采用开源许可证**。
- `⚠️ 未找到独立仓库`：检索到论文和数据描述，但未找到由作者维护的、可确认对应的公开代码/数据仓库。为避免误归因，未以第三方复刻仓库替代原始地址。
- CSV 行数均为本地文件的有效数据行（不含表头）；预处理后的 RDKit 特征、UMAP/t-SNE 与 SHAP 文件不重复列为独立数据集。

| 本地数据集 | 本地文件与规模 | 对应文献来源 | 公开数据/代码位置 | 说明 |
|---|---|---|---|---|
| 光诱导钯催化 C–H 官能团化三步串联 | `Pd_catalyzed_CH_functionalization.csv`（1,032） | Qiu, J. *et al.* [Selective functionalization of hindered meta-C–H bond of *o*-alkylaryl ketones promoted by automation and deep learning](https://doi.org/10.1016/j.chempr.2022.08.015), *Chem* **2022**, 8, 3275–3287. | 📄 公开数据位于 [论文 SI（由原始论文页面获取）](https://www.sciencedirect.com/science/article/pii/S2451929422004284)；未找到独立作者仓库。 | 文中说明以 1,032 个实测产率建立 CMPRY；名称中的“光诱导”对应三步一锅法的首步 C–H 羧化。 |
| Rh 催化芳醛腙邻位 C–H 酰胺化 | `all_HTE_set.csv`（1,000）；`model_set (600).csv` / `external_set (400).csv` 及 RDKit 特征文件 | Zhang, Y. *et al.* [High-throughput screening and machine learning prediction of Rh-catalyzed *ortho*-C(sp²)–H amidation of arylaldehyde hydrazones](https://doi.org/10.1039/D6QO00026F), *Organic Chemistry Frontiers* **2026**, 13, 2730–2741. | ✅ [C-H_Amidation_Prediction](https://github.com/zy2402/C-H_Amidation_Prediction) | 论文的 Data availability 明确给出该仓库；仓库含 600/400 训练–外部验证划分、RDKit 特征和 KNIME 工作流。 |
| Ru（钌）催化亚砜叶立德对膦酸 O–H 键插入 | `Ru_catalyzed_insertion.csv`（756） | Lin, A. *et al.* [High-throughput experimentation and machine learning-promoted synthesis of α-phosphoryloxy ketones via Ru-catalyzed P(O)O–H insertion reactions of sulfoxonium ylides](https://doi.org/10.1007/s11426-024-2313-5), *Science China Chemistry* **2025**, 68, 679–686. | 📄 公开数据位于 [出版社 SI（HTE 与模型开发部分）](https://files.sciengine.com/gridfs/1878695122409881600/edit_scc-2024-0862-File003.pdf) | 数据文件名维持 `Ru_catalyzed_insertion.csv`。 |
| 镍催化对映选择性交叉偶联 | `Raw_Dataset.csv`（6,590） | Gao, Y. *et al.* [Artificial Intelligence-Driven Development of Nickel-Catalyzed Enantioselective Cross-Coupling Reactions](https://doi.org/10.1021/acscatal.4c04277), *ACS Catalysis* **2024**, 14, 18457–18468. | ✅ [Enantioselective-Cross-Coupling-Prediction](https://github.com/TheLiaoGroup/Enantioselective-Cross-Coupling-Prediction) | 论文 Accession Codes 指向该仓库；其中包含 `Raw_Dataset.xlsx`、MAF/RDKit/DFT 描述符、坐标、Python 脚本与 KNIME 工作流。 |
| 铜催化选择性氟代烯烃环丙烷化 | `acceptor_HTE.csv`（763）；`donor_acceptor_HTE.csv`（959） | Yan, C. *et al.* [Data-driven copper catalysis enables stereoselective fluorocyclopropanation](https://doi.org/10.1016/j.chempr.2025.102922), *Chem* **2026**, 102922. | ✅ [Enantioselecitve-Fluorocyclopropanation](https://github.com/TheLiaoGroup/Enantioselecitve-Fluorocyclopropanation) | 论文 Data and code availability 指向该库。仓库名中 `Enantioselecitve` 为作者实际拼写；含两类 HTE 数据、聚类结果和预测模型，采用 MIT License。 |
| 酰胺缩合产率 | `amide-coupling.csv` 与 `amide-coupling(additive_fixed).csv`（各 47,015） | Zhang, C. *et al.* [Intermediate knowledge enhanced the performance of the amide coupling yield prediction model](https://doi.org/10.1039/D5SC03364K), *Chemical Science* **2025**, 16, 11809–11822. | ✅ [aichemeco/amide_coupling](https://github.com/aichemeco/amide_coupling/tree/main) | 论文 Data availability 明确给出该库；`data/` 中提供 47,000+ HTE 反应、单条件数据、USPTO 与虚拟化合物数据，并采用 MIT License。两份本地 CSV 的差异在于 additive 字段处理方式。 |
| Ir 催化羧酸与亚砜叶立德选择性 O–H 插入 | `Ir_catalyzed_OH_insertion.csv`（653）；`model_set (412).xlsx`、`external_set (235).xlsx`、`bioactive molecules (6).xlsx` | Xu, Y. *et al.* [HTE and machine learning-assisted development of iridium(I)-catalyzed selective O–H bond insertion reactions toward carboxymethyl ketones](https://doi.org/10.1039/D2QO01954J), *Organic Chemistry Frontiers* **2023**, 10, 1153–1159. | 📄 公开数据位于 [RSC SI](https://www.rsc.org/suppdata/d2/qo/d2qo01954j/d2qo01954j2.pdf)；✅ [KNIME 公开工作流](https://hub.knime.com/theliaogroup/spaces/O-H_bond_insertion)；[在线预测服务](https://www.pangu-drug.com/ylide) | SI 明确给出 412 个建模反应、235 个外部验证反应及工作流地址；本地 653 行 CSV 对应三份数据合并，另有 6 个生物活性分子案例。在线服务用于预测，不应当视为源码发布。 |
| DHP 衍生物催化羧酸脱羧硒化 | `DHP_catalyzed_decarboxylative_selenation.csv`（868） | Yu, Z. *et al.* [HTE- and AI-assisted development of DHP-catalyzed decarboxylative selenation](https://doi.org/10.1039/D2CC06217H), *Chemical Communications* **2023**, 59, 2935–2938. | 📄 公开数据位于 [RSC SI](https://www.rsc.org/suppdata/d2/cc/d2cc06217h/d2cc06217h3.pdf)；未找到独立作者仓库。 | 原始实验与 HTE 资料位于 SI；本次未检索到能与该论文精确对应的作者公开代码库。 |

## Ir O–H 插入合并去重数据集

使用独立文件 [Ir_catalyzed_OH_insertion_merged_deduplicated.csv](<Ir 催化羧酸与亚砜叶立德选择性 O–H 插入/Ir_catalyzed_OH_insertion_merged_deduplicated.csv>)（653 行）。2026-10-02 将两个同源目录中的 6 份 XLSX 与既有 653 行 CSV 合并，按 `acid + ylide + product + yield (%)` 去除重复观测：共读入 1,959 行，去除 1,306 行重复副本。两个目录的对应 XLSX 字节相同；既有 CSV 与一套 XLSX 拼接后逐行一致。所有源文件保留。

输出保留原五列和 Well 顺序，编码为 UTF-8 BOM。`W1–W412` 属于原建模集，`W413–W647` 属于原外部验证集，`W648–W653` 属于生物活性分子案例。规范化分子结构后有 650 个唯一 `acid + ylide + product` 组合；`W74/W580`、`W234/W595`、`W426/W610` 三对产率不同，均保留，不做任意取舍或平均。它们跨原建模/外部验证分区，后续评估需考虑结构重叠。详细输入指纹与验收记录位于独立文档仓库 `project-docs/docs/ir-oh-insertion-merge-20261002.md`。

## 检索与核验记录

1. 通过 Z-AIHub 的已向量化“课题组文章”记录，按数据集反应名称匹配原论文，并读取其 Zotero 元数据（标题、作者、年份、DOI、期刊）。
2. 对论文正文与 SI 进行“data availability / code availability / GitHub”证据检索，确认酰胺、镍、Rh 与 Cu 四项的仓库地址；Ir 的 KNIME 工作流地址由其 SI 明示。
3. 逐项核对本地文件名、列名与行数。对没有作者仓库的项目保留 DOI/SI 原始出处并明确标记，避免把二次整理库错误写成作者开源地址。

建议引用时同时引用表中原始论文；使用代码/数据时还应遵守对应仓库许可证与出版社 SI 的使用条件。
