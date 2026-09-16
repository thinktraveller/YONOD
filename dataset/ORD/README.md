# ORD 标准化反应数据

本目录保存 19 个可直接建模的 CSV 表，共 **128,712** 条数据记录（不含表头；于 2026-09-16 按本地文件统计）。文件名中的 `structure_v1_normalized_dataset` 表示本项目采用的扁平化、结构化字段版本；它们不是完整的原始 ORD 反应 JSON。

## 来源与追溯

所有文件来自 [thinktraveller/ord-datasets](https://github.com/thinktraveller/ord-datasets) 的 **ORD model-ready** 发布物。该发布物将 Open Reaction Database（ORD）中的反应整理为可建模的表格，并为各目标提供字段模式、行级映射、筛选审计和固定的上游 ORD 源链接。建议引用或复核时使用其固定的预发布版本 [v0.1.0-model-ready-preview-1](https://github.com/thinktraveller/ord-datasets/releases/tag/v0.1.0-model-ready-preview-1)（提交 `7d52dd2`），而不是仅依赖会继续变化的 `main` 分支。

本目录仅保留了 CSV 表，没有同步源发布物中的 `row-map.csv`、`audit.jsonl`、`source-links.json`、`metadata.json` 和校验文件。因此，本地行数可用于确认这里的快照，但不能替代源发布物提供的逐行 ORD 谱系、标签选择理由和哈希核验。数据许可和再分发条件也应以源仓库及其每个数据包的说明为准。

## 文件清单

下表行数均不含 CSV 表头；“目标”由本地末列名称记录，不能自动等同于分离收率。

| 文件 | 记录数 | 目标 |
| --- | ---: | --- |
| `ahneman_yield_structure_v1_normalized_dataset.csv` | 4,312 | `yield_percent` |
| `asymmetric_alkylation_ee_s_minus_r_v1_normalized_dataset.csv` | 1,430 | `ee_s_minus_r_percent` |
| `c_n_yield_structure_v1_normalized_dataset.csv` | 50,668 | `yield_percent` |
| `catechol_product_2_research_full_v1_normalized_dataset.csv` | 1,227 | `product_2_yield_percent` |
| `catechol_product_3_research_full_v1_normalized_dataset.csv` | 1,227 | `product_3_yield_percent` |
| `catechol_total_product_research_full_v1_normalized_dataset.csv` | 1,227 | `total_product_yield_percent` |
| `cernak_member_2be11_conversion_structure_v1_normalized_dataset.csv` | 1,152 | `conversion_percent` |
| `cernak_member_3b8a2e_conversion_structure_v1_normalized_dataset.csv` | 1,320 | `conversion_percent` |
| `chan_lam_yield_structure_v1_normalized_dataset.csv` | 9,602 | `yield_percent` |
| `chemrxiv_amide_yield_structure_v1_normalized_dataset.csv` | 957 | `yield_percent` |
| `chemrxiv_ch_arylation_yield_structure_v1_normalized_dataset.csv` | 1,529 | `yield_percent` |
| `nano_yield_structure_v1_normalized_dataset.csv` | 1,728 | `yield_percent` |
| `nicolit_yield_structure_v1_normalized_dataset.csv` | 1,669 | `yield_percent` |
| `pfizer_lcms_v1_normalized_dataset.csv` | 39,347 | `lc_area_percent` |
| `photodehalogenation_conversion_structure_v1_normalized_dataset.csv` | 1,152 | `conversion_percent` |
| `roche_borylation_yield_structure_v1_normalized_dataset.csv` | 890 | `yield_percent` |
| `science_hte_diverse_nucleophile_relative_lc_area_ratio_structure_v1_normalized_dataset.csv` | 1,531 | `relative_product_lc_area_ratio` |
| `shields_yield_structure_v1_normalized_dataset.csv` | 1,984 | `yield_percent` |
| `suzuki_yield_structure_v1_normalized_dataset.csv` | 5,760 | `yield_percent` |

使用前请读取各 CSV 的表头：反应组分以角色（如 `reactant-*`、`reagent-*`、`catalyst-*`、`solvent-*`）展开，空单元格不代表化学上必然不存在该角色，而可能是源记录未结构化或未报告。若需要完整实验过程、测量方法、原始数据集或论文出处，应回到源发布物的对应数据包及其 `source-links.json`。
