# 数据集总览

本目录是 YONOD 的数据入口，包含通用反应数据库的建模表、固定切分的专利反应–产率数据、专题 HTE 数据，以及一个仅用于测试配置的小型 fixture。各子目录的 README 是来源、许可证、字段和逐行可追溯性说明的权威位置；本文件只提供选型与导航，不应替代子目录的详细记录。

## 快速选择

| 需求 | 建议位置 | 注意事项 |
| --- | --- | --- |
| 在多个公开反应类型上直接建模 | [ORD/](ORD/) | 19 个扁平化 CSV；包含产率、转化率、LC 响应比和 ee 等不同标签，先核对目标列。 |
| 使用大规模、固定 train/valid/test 切分的专利反应–产率数据 | [USPTO/](USPTO/) | 文件扩展名为 CSV，但实际以制表符分隔；保留既有切分。 |
| 使用单一反应类型的 HTE 或文献整理数据 | [others/](others/) | 包含 SuFEx、两套独立 Buchwald–Hartwig HTE、Suzuki–Miyaura、SLAP，以及同源的 BH 适配子集。 |
| 使用廖矿标课题组相关的专题反应数据、公开仓库或 SI 数据 | [廖矿标课题组/](廖矿标课题组/) | 目录名不能单独证明数据独立性；以其 README 列出的八类反应和原始论文/仓库为准。 |
| 验证配置、描述符或端到端流程 | `benchmark_smoke_fixture.csv` | 仅 12 条记录；供 [`example.yaml`](../example.yaml) 的 2-fold smoke 使用，不应用于科学结果。 |

## 顶层目录索引

| 位置 | 当前内容与规模 | 主要标签/格式 | 来源与使用边界 | 详细说明 |
| --- | --- | --- | --- | --- |
| [ORD/](ORD/) | 19 个模型就绪 CSV，合计 128,712 条记录 | `yield_percent`、`conversion_percent`、`lc_area_percent`、相对 LC 面积比、`ee_s_minus_r_percent` 等 | 来自 `thinktraveller/ord-datasets` 的 model-ready 预发布快照；本地仅保存 CSV，缺少源发布物的行映射、审计、上游链接与校验文件。 | [ORD/README.md](ORD/README.md) |
| [USPTO/](USPTO/) | train/valid/test 三个固定切分，共 526,468 条记录 | `Yield`；UTF-8 兼容的制表符分隔文本 | 本地派生快照；Lowe USPTO 归档是最合理的候选上游，`Schwaller_` 前缀只说明处理语境，不能证明与某个公开包逐字节对应。 | [USPTO/README.md](USPTO/README.md) |
| [others/](others/) | 6 个专题表，共 17,904 条记录 | 收率、`Ratio` 与 LC–MS 产物比并存 | SuFEx 来自论文 SI；四个原 `yieldmaster` HTE 数据集来自其对应论文/归档；另有一个 Doyle–Ahneman C–N 偶联的适配子集。 | [others/README.md](others/README.md) |
| [廖矿标课题组/](廖矿标课题组/) | README 已文档化 8 类反应；包括 CSV、XLSX 和预处理结果 | 以各反应项目的实验产率/选择性等标签为准 | 每一类数据对应论文 SI、作者数据/代码仓库或公开工作流；有些条目是外部基准，不能仅因位于本目录而归为课题组原始数据。 | [廖矿标课题组/README.md](廖矿标课题组/README.md) |
| `benchmark_smoke_fixture.csv` | 12 条合成的最小 fixture | `yield` | 项目内 smoke 测试输入，不是外部基准或实验数据集。 | [`example.yaml`](../example.yaml) |

## 跨目录使用注意事项

- **不要跨目录直接相加记录数。** ORD 的不同目标表可复用同一反应；`others/` 中 Doyle–Ahneman HTE 表与其 C–N `Ratio` 子集共享上游数据脉络；廖矿标目录还包含建模/外部验证划分及预处理结果。
- **收率并非统一标签。** `yield_percent` / `Yield`、转化率、LC 面积比、LC–MS 产物比和 ee 的实验含义与量纲不同。跨表比较或合并前必须先查看目标列及其子目录说明。
- **保留既有切分。** USPTO 的 train/valid/test 边界是本地快照的一部分；部分专题数据也已提供训练/外部验证划分，不应在不了解论文方案时覆盖。
- **廖矿标目录的文件夹数不等于独立数据集数。** 当前有 12 个一级子目录，而其 README 只将内容归为 8 类反应；存在不同命名的目录和派生结果。聚合、去重或建模前请按该 README 的本地文件清单核对实际来源与角色。
- **引用和许可。** 需要论文出处、作者代码、SI、许可证、哈希或行级追溯时，进入对应子目录 README 并回到其链接的原始发布物；不要将本地副本自动视作具有独立再分发许可。

## 基本读取约定

- 除 USPTO 外，当前表主要为逗号分隔 CSV；字段包含 SMILES、反应角色、条件和标签。各表的列名与缺失值约定不同，读取前先检查表头。
- USPTO 的 `.csv` 文件必须显式使用 `sep="\t"` 读取；其首列为冗余索引，可按 [USPTO 说明](USPTO/README.md) 的示例处理。
- 请使用 `pathlib.Path` 或 `os.path` 构造路径，以兼容 Linux 与 Windows，并保留 UTF-8 编码以支持中文目录名。
