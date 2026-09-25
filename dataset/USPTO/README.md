# USPTO 专利反应–产率数据集

本目录保存一个本地派生、固定 train/valid/test 切分的 USPTO 反应–产率快照，以及可直接供 YONOD/Pandas 读取的标准 CSV。它不是 Daniel Lowe 在 Figshare 发布的完整原始归档。固定切分边界属于这个本地快照，不能被任意重切分后仍称为同一数据集。

## 当前布局

```text
USPTO/
├── README.md
├── train.csv                         # 直接使用的标准训练集
├── valid.csv                         # 直接使用的标准验证集
├── test.csv                          # 直接使用的标准测试集
└── corpus/
    ├── US_patents_..._{train,valid,test}.csv  # 原始本地 TSV 快照
    ├── provenance.csv                          # 标准行到原始行的映射
    ├── conversion_manifest.yaml                # 转换输入/输出哈希与契约
    └── quality_report.json                     # 结构、化学与重复审计汇总
```

根目录的 `train.csv`、`valid.csv` 和 `test.csv` 是唯一直接建模输入。`corpus/` 保存原始输入和最小溯源链；其中长文件名虽使用 `.csv` 扩展名，实际是 UTF-8/ASCII 兼容的制表符分隔（TSV）文件。不要把它们当作标准逗号分隔 CSV 直接读取。

## 标准 CSV

三份标准表均为 UTF-8、逗号分隔、单行表头且没有写入 Pandas 索引。共同字段为：

`sample_id, split, reactants_smiles, reagents_smiles, products_smiles, yield_percent, reaction_smiles`

| 切分 | 文件 | 行数 | 大小（字节） | SHA-256 |
| --- | --- | ---: | ---: | --- |
| train | `train.csv` | 473,963 | 138,755,334 | `b6ea60f9d09cf47fafe1518f9b57f3638c081ec17b03c2c41d29ea1249a1f47f` |
| valid | `valid.csv` | 26,101 | 7,626,225 | `9c8ed5e04057d8df66d18f4ae28e6daf68076909f722d8f4601d461521438823` |
| test | `test.csv` | 26,404 | 7,664,804 | `b17c71724d6acda2f6e4c6867ce23d0585b3d0b61480d4e8b1f956faab1d945a` |
| **合计** |  | **526,468** | **154,046,363** |  |

- `sample_id` 为 `uspto_<split>_<myID>`，在全部三个切分中唯一；原始 `myID` 和无名索引会跨切分重复，不能作全局主键。
- 三个角色字段逐字保留 `CanonicalizedReaction` 的 `reactants>reagents>products` 区段。点号连接的多组分没有截断、重排或去盐；`reagents_smiles` 为空表示原始记录没有试剂，不能以虚构分子填充。
- `yield_percent` 保留原始 `Yield` 的 0–100 百分比单位、数值和精度；未除以 100、取整或裁剪。其更上游的提取方法未知。
- `reaction_smiles` 可由三个角色字段以 `>` 可逆重组，是审计字段；不要把完整 reaction SMILES 直接传给只接受分子 SMILES 的描述符。

```python
from pathlib import Path

import pandas as pd

data_dir = Path("dataset") / "USPTO"
train = pd.read_csv(data_dir / "train.csv", dtype={"sample_id": "string"})
assert train["sample_id"].is_unique
assert train["yield_percent"].between(0, 100).all()
```

首次进行分子特征计算时，建议仅将 `reactants_smiles` 声明为 reactants/descriptor 输入。`products_smiles` 和 `reagents_smiles` 是否可用取决于实际预测时点，必须明确记录；默认纳入产物会带来信息泄漏风险。多组分 reactants 可作为一个点号分隔的 SMILES 区段供 Morgan 使用，不能随意强拆为固定数量的分子列。

## 原始快照与溯源链

原始本地输入位于 `corpus/`，行数不含表头，SHA-256 针对完整未压缩字节内容计算。

| 切分 | 文件 | 行数 | 大小（字节） | SHA-256 |
| --- | --- | ---: | ---: | --- |
| train | `corpus/US_patents_1976-Sep2016_1product_reactions_yield_ok_cropped_data_train.csv` | 473,963 | 373,902,727 | `e93dbe419385402444ad8ac992c095c139a25c68d1a004312a8acc49048a77f7` |
| valid | `corpus/US_patents_1976-Sep2016_1product_reactions_yield_ok_cropped_data_valid.csv` | 26,101 | 20,530,981 | `996adf61ee4c8671788a35f3bc29b11ae62a70c06f6cbec2c14d1aaf38e53379` |
| test | `corpus/US_patents_1976-Sep2016_1product_reactions_yield_ok_cropped_data_test.csv` | 26,404 | 20,777,603 | `4f7293a54386198578c720a54e65a984ccb22bffc4e73b66277757d390d807ed` |
| **合计** |  | **526,468** | **415,211,311** |  |

原始 TSV 的逻辑列为无名索引、`myID`、`Source`、`Target`、`CanonicalizedReaction`、`OriginalReaction` 和 `Yield`。`Source`/`Target` 是 tokenized 模型文本，且 Source 可含 `A_…` 试剂别名，不是直接可用的分子 SMILES。`OriginalReaction` 含原子映射；映射不是无误真值。

`corpus/provenance.csv` 有 526,468 行、405,471,924 字节，并按 train、valid、test 标准表顺序一一对应。字段为：

`sample_id, split, source_file, source_row, source_index, source_myid, source_tokens, target_tokens, original_reaction, yield_raw`

`source_row` 是包含表头的 1-based 逻辑记录号，首条数据为 2。标准表加 provenance 可恢复每条记录的逻辑源字段；要验证**字节级**来源，必须保留三份原始 TSV，并将其 SHA-256 与 `corpus/conversion_manifest.yaml` 的 `input_files` 比对。

转换清单还记录了转换器内容哈希、Python/RDKit 版本、字段契约、输出表哈希和发布状态。质量报告记录此次转换的结果：526,468 条记录全部接受、36,311 条空试剂、所有反应物/非空试剂/产物均可由 RDKit 2022.09.5 解析，且原文及规范化 reaction 均未发现完全重复。

按当前目录整理要求，`rejected_rows.csv`、`chemistry_issues.csv`、`duplicates.csv` 和 smoke `acceptance/` 已移除：前面三者在此次完整转换中均只有表头，质量报告已保留其零计数；转换清单的 `output_files` 是转换发布当时的不可变文件清单，因此仍列出这些历史零记录审计文件，不应被误作当前目录树。

## 固定切分与来源限制

当前普通 YONOD `train/all` 工作流仍使用 KFold，不能仅凭标准表的 `split` 列复现固定 train/valid/test 协议。不要合并三表后运行普通 KFold，再称结果为原始 test 评估；固定三分训练、调参与测试隔离仍需单独实现。

本地快照最合理的候选上游是 Daniel Lowe，*Chemical reactions from US patents (1976-Sep2016)*，Figshare，2017，DOI：[10.6084/m9.figshare.5104873.v1](https://doi.org/10.6084/m9.figshare.5104873.v1)。Schwaller 等人的 [MolecularTransformer 仓库](https://github.com/pschwllr/MolecularTransformer) 描述了 Lowe 数据的 canonicalized 固定切分子集，相关论文见 [Schwaller *et al.* (2019)](https://doi.org/10.1021/acscentsci.9b00576)。本地没有上游归档哈希、完整转换记录、筛选规则或切分算法，故这些资料不能证明本地副本与公开包逐字节对应。候选上游采用 [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/)；本地派生副本没有独立许可证，使用或再发布前应自行确认适用条款。

## 复核

```bash
source /home/wangzh685/miniconda3/etc/profile.d/conda.sh
conda activate yonod

# 核对原始 TSV 的字节身份
sha256sum dataset/USPTO/corpus/US_patents_*.csv

# 结构审计；结果写入 result/uspto_standard_csv/planning_audit/
python _verify/audit_uspto_csv.py

# 新建一次独立标准表核验，不覆盖既有结果
python _verify/verify_uspto_standard_csv.py \
  --snapshot dataset/USPTO/corpus \
  --source-dir dataset/USPTO/corpus \
  --acceptance-dir result/uspto_standard_csv/revalidation_YYYYMMDD
```
