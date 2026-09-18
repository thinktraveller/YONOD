# Ablation2 原始数据与角色审计

数据准备任务为 `ablation2_amide_data_prep_v1`，不拟合模型。原始来源是 `dataset/廖矿标课题组/酰胺缩合产率/amide-coupling.csv`，SHA-256 为 `79bd0d92e0dc9932ab68e2eb239b973d7e4db0243a9ae69db93d21acfe4e4a8a`。

标准化总体位于 `result/ablation2_amide_data_prep_v1/data/standardized_population.csv`，SHA-256 为 `5fae4b314ffdb4d3f5eddcd6a59b627cee8ea5a939a9c375dbeb3a0f81c3989c`，共有 47,015 条记录；没有因本轮质量规则排除的行。

| 研究角色 | 原始列 | 标准化列 | 说明 |
| --- | --- | --- | --- |
| A（胺） | `sub_1_smiles` | `amine_smiles` | 47,015 行均含氮；不以列名本身推断角色。 |
| B（羧酸） | `sub_2_smiles` | `acid_smiles` | 47,015 行均匹配羧酸 SMARTS。 |
| P（产物） | `product_smiles` | `product_smiles` | 632 个底物对各对应一个规范化产物键；46920 行匹配简单酰胺 SMARTS，余者保留并不据此删行。 |
| C（条件） | `activation`,`additive`,`base`,`solvent` | 对应 `*_smiles` | 都作为 `others` 分子角色；不放入 `reactants`，也不当作数值 `conditions`。 |

标准表保存未改写的字符串、逗号转点的独立规范化字符串、状态和规范化结构键。标签 `yield` 为 `[0.0000317, 1.0]` 的分数而非百分数。A/B/P/C 未检出非法 SMILES；activation 有 1 个空白，additive 有 23,220 个空白，base 有 9,686 个空白，solvent 无空白且恒为来源数据内 DMF。空白仅标作 `blank_source_value`，不声称“未添加”。

完整输入的 `repeat_group_id` 基于 A+B+P+四个 C 键，共检出 797 个重复组、1,600 条记录，其中 794 组的产率不一致。因此随机评估按该键分组，防止相同完整输入跨训练/测试；标签冲突被保留为复测/重复来源未明的质量事实，并未直接称为数据泄漏。

划分任务 `ablation2_amide_splits_v1` 生成随机重复组、未见胺、未见酸和未见底物对四套 5×3 manifest。每套只有同一 population 的 sample IDs。底物对留出只保证 pair 不见，不声称胺或酸各自也不见。
