# 更新日志

本项目遵循 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/) 与 [语义化版本 2.0.0](https://semver.org/lang/zh-CN/)。

格式：每个版本下分 `Added` / `Changed` / `Fixed` / `Removed` / `Deprecated` / `Security` 六类（按需出现）。
发布标签由 `git tag vX.Y.Z` 创建后 push。

---

## [Unreleased]

> 当前开发分支的修改，下次发布时会归并到具体版本号。
> **发布标签建议**：`v1.2.0`（Track B 新增 + 数据集整理 + 通用化规划，属 MINOR 增量）。

### Changed

**README.md：补充各数据集与模型的来源标注**
- 酰胺缩合数据集（`dataset/amide-coupling.csv`）新增引用提示，指向 DOI:10.1039/D5SC03364K
- FISD 权重小节：新增 GitHub 链接（`KeantChen/FISD/tree/main/model`）、论文 DOI（10.1039/D5SC00451A），获取方式 A 改为"GitHub 直接下载"，原方式 A/B 顺延为 B/C
- MolMetaLM 权重小节：新增 GitHub 链接（`CSUBioGroup/MolMetaLM`）、论文 DOI（10.48550/arXiv.2411.15500）
- 算法实现对照表：FISD 行补充上游仓库链接与 DOI；MolMetaLM 行补充 GitHub 链接与 DOI
- 引用小节：新增酰胺缩合数据集、FISD、MolMetaLM 三条 BibTeX 模板
- License 表：FISD 行补充 DOI 链接

**FISD 权重移出 git 追踪，改为需用户自行获取**
- **动机**：`WEIGHTS/FISD/` 中的 3 个 `.pth` 文件（`qm_9_mse_model.pth`、`qm_9_cos_model.pth`、`qm_9_2in1_model.pth`）来自上游 FISD 项目，该项目未附任何明确开源许可证；在许可证不明确的情况下，将上游产物作为二进制文件随仓库分发存在法律风险
- **`.gitignore`**：新增 `WEIGHTS/FISD/` 条目，更新注释说明原因；原"bundled with the repo"注释改为"no explicit license; not redistributed"
- **`git rm --cached`**：将 3 个 `.pth` 文件移出 git 索引，本地文件不受影响
- **`README.md`**：
  - 「第二步：克隆仓库」去除"克隆后获得 FISD 权重"说明，改为提示 FISD 和 MolMetaLM 均需单独获取
  - 「仓库结构」`WEIGHTS/FISD/` 条目从 `★ 已随仓库提供` 改为 `✗ 需单独获取（无明确许可证）`
  - 「外部资产说明」总表新增 FISD 权重行（❌ 需单独获取）
  - 「FISD 权重」小节重写：说明不再随仓库分发的原因，提供两种获取方式（方式 A：从本地 `化学描述符相关项目/FISD/` 复制；方式 B：重训练上游 FISD 笔记本）
  - 「License」表中 FISD 条目注明"无明确许可证，不随本仓库分发"

**README.md：「示例结果」简化为结构示例，移除具体数值**
- 将控制台输出、metrics_summary.csv 矩阵、report.html 排名表中的所有具体数值替换为 `X.XXXX` 占位符
- 去除「全量数据集基线（供参考）」子节（含历史实测 R² 数值）
- 保留完整的输出格式说明，使读者了解输出结构，具体数值以实际运行结果为准

**README.md：简化文档，去除 Track A/B 区分**
- 通篇移除"Track A"/"Track B"标签，以酰胺缩合反应为唯一示例贯穿文档
- 「第六步：准备数据集」数据集目录标注去除 "Track A" 前缀，改为直接描述文件内容
- 「第七步：快速验证」完整替换为交互向导步骤示例：展示 8 步向导的每一步提示文本、示例用户输入（以 `>` 标注）及预期程序输出，以 `dataset/test-amide-coupling.csv`（10 行样本）为示范，引导用户用 `B C E F G H` 字母序号选取 SMILES 列，描述符选 `morgan`、模型选 `xgb`，秒级完成冒烟测试
- 「第八步：完整运行」去除 Track B（镍催化偶联）命令，仅保留酰胺缩合全量 4×4 grid 的 CLI 示例，并补充向导全选提示
- 「主要结果」删除 Track B 镍催化对映选择性结果表，节标题改为「酰胺缩合产率预测」
- 「仓库结构与文件说明」：去除 `features/` 说明中的 Track A/B 标注，删除 `dataset/镍催化偶联数据集/` 目录条目，`ee_metrics.py` 说明去除"Track B 专用"字样
- 「外部资产说明」：资产总表删除 ECC 数据集行，小节标题"Track A 数据集"改为"酰胺缩合数据集"，完整删除"Track B 数据集"小节（schema 表 + 论文引用说明）
- 「命令行参数速查」：删除 Track B 镍催化示例命令，调试模式示例补全为酰胺缩合的具体参数

**tests/ 移出 git 追踪**
- `tests/test_csv_loader.py`、`tests/test_feature_builder.py`、`tests/test_report.py` 通过 `git rm --cached` 移出 git 索引；`tests/` 在 `.gitignore` 中已存在，本次使远端历史与声明保持一致
- 动机：单元测试为本地开发辅助脚本，不属于可分发源码；与 `.gitignore` 注释保持一致（"仅供本地开发使用"）

**数据集目录重命名 & 样本数据集纳入仓库**
- 数据集目录从 `数据集/` 重命名为 `dataset/`（英文路径，避免跨平台中文路径问题）
- `dataset/amide-coupling.csv`（47015 条）和 `dataset/test-amide-coupling.csv`（10 条样本）正式纳入 git 追踪，来源：[aichemeco/amide_coupling](https://github.com/aichemeco/amide_coupling/tree/main)，遵循 MIT 协议
- `.gitignore` 更新：`dataset/*` 整体忽略，通过 `!dataset/amide-coupling.csv` / `!dataset/test-amide-coupling.csv` 豁免两个样本 CSV；保留旧 `数据集/` 规则以兼容历史
- `README.md` 同步更新：
  - 第六步「准备数据集」：说明 Track A 样本已随仓库提供，更新放置路径示意
  - 烟测命令：改为使用 `dataset/test-amide-coupling.csv`（10 行，秒级），并修正列名为 `activation additive base solvent`
  - 完整运行命令：路径更新为 `dataset/amide-coupling.csv` / `dataset/镍催化偶联数据集/Raw_Dataset.csv`
  - 外部资产表：两个样本 CSV 标记为 ✅ 仓库已含（MIT）
  - 仓库结构树：`数据集/` → `dataset/`，标注各文件来源

**仓库清理：移除大型数据集和第三方源码的 git 追踪**
- `git rm --cached -r 数据集/ 化学描述符相关项目/`：共 18685 个文件从 git 索引中移除，远端不再保存这两个目录；本地文件不受影响
- 两者在 `.gitignore` 中已存在（`化学描述符相关项目/` 与 `数据集/` 条目），本次操作使远端历史与 gitignore 声明保持一致
- 动机：`化学描述符相关项目/` 含第三方上游源码（ATMOMACCS/FISD 等），各有独立 LICENSE；`数据集/` 体积较大且非源码，不适合随仓库分发

**README.md 全面重写**
- 新增「目录」章节：快速跳转到各节
- 新增「新手快速上手」章节（8 步图文指引）：
  - 前置知识说明（命令行 / conda / SMILES）
  - 克隆仓库与环境准备
  - GPU / CPU 双路径安装 PyTorch 说明
  - 数据集放置目录示意
  - 烟测命令与预期输出
  - 完整运行命令示例
- 新增「仓库结构与文件说明」章节：带注释的树状目录，标明每个文件/目录的用途及是否随仓库提供
- 更新入口脚本引用：全部改为 `yonod.py`（合并版），删除旧 `run_yield_prediction.py` / `run_ecc_prediction.py` 引用
- 更新外部资产表：`数据集/酰胺缩合数据集.csv` 与 `数据集/镍催化偶联数据集/Raw_Dataset.csv` 标记为 ❌ 需单独获取
- 更新算法实现对照表：路径从 `yonod_yield/` 修正为 `yonod/`；新增 `csv_loader.py` / `feature_builder.py` 两行
- 更新命令行参数速查：统一为 `yonod.py` 的参数格式，新增 `--label-col` / `--smiles-cols` / `--numeric-cols` / `--task-name` 等参数说明

### Fixed

**项目清理：移除废弃文件、补入遗漏源文件**
- **移除** `yonod.bat`：已由 `yonod.py`（Python 交互向导）完全替代，bat 方案因 cmd.exe 解析缺陷已弃用
- **移除** `verify_morgan_rf.py`：开发期一次性性能诊断工具，功能已无需保留
- **移除** `数据集/测试.csv`、`数据集/镍催化偶联数据集/test.csv`：临时测试用小 CSV，不应入库
- **补入** `yonod_yield/universal/__init__.py`、`yonod_yield/universal/csv_loader.py`、`test_csv_loader.py`：universal 子包的核心文件及测试脚本此前未被 git 追踪，本次一并入库
- **更新** `.gitignore`：新增 `result/`（本地运行输出目录）排除规则
- **更新** `run_yonod.py`：`_Tee` 日志类改为直接持有 `sys.__stdout__` 引用，去掉冗余的 `real_stream` 构造参数

**report.py：推荐组合表格加入模型用时列**
- `rank_combinations()`：若输入 `metrics_df` 含 `train_time_s` 列，将其保留至返回 DataFrame（列位于 `score` 与 `reason` 之间）
- `_section_ranking()`：前三名推荐表和完整排名表新增「用时」列（`has_time` 条件渲染，不含 `train_time_s` 时自动隐藏，向后兼容）
- 新增 `_fmt_time()` 辅助函数：秒数 < 60 显示 `3.2s`，≥ 60 显示 `1m 23s`

**feature_builder.py：SMILES 行有效性判断改为「至少一列有效」**
- **根本原因**：原逻辑对所有 SMILES 列使用**与掩码**（`row_mask &= mask`），导致任意一列为空（如可选试剂列 `additive`/`base` 留空）即将整行标记为无效并丢弃；10 行测试数据中仅 4 行全列填充，其余 6 行被误丢
- **修复**：改为**或掩码**（`row_mask |= mask`），初始化 `row_mask = np.zeros(n, dtype=bool)`；只要一行中至少有一列含有效 SMILES 即保留该行；空列/无效列已由描述符 `featurize` 返回零向量，拼接结果正确（零向量 = 该试剂无贡献），无需额外处理
- **影响**：含可选试剂列的多列数据集（酰胺缩合、ECC 等）有效行数大幅提升；仅当一行中所有 SMILES 列均为空或解析失败时才被过滤

**yonod.py + feature_builder.py：逗号分隔阴阳离子 SMILES 规范化**
- **背景**：部分试剂以逗号区分阴阳离子（如 `CCN=C=NCCCN(C)C,Cl`），而 RDKit 的合法片段分隔符为 `.`；原逻辑直接调用 `Chem.MolFromSmiles()` 导致验证误报无效、描述符计算时静默丢行
- **yonod.py**：新增 `_normalize_smiles(smi)` 辅助函数（`smi.replace(",", ".")`），在 `_validate_smiles` 中对每个值规范化后再调用 `Chem.MolFromSmiles()`，逗号分隔的离子对不再触发误报
- **feature_builder.py**：在 `build_universal_features` 的 SMILES 列提取循环中，对 `smiles_list` 逐值执行 `s.replace(",", ".")`，确保描述符收到的 SMILES 均为合法格式，逗号分隔的行不再被 `valid_mask` 静默过滤

**yonod.py v3：标签列数值验证 + SMILES 列 RDKit 全量验证**
- **标签列验证**（`_validate_label`）：用户确认标签列后立即对全量数据执行 `pd.to_numeric(errors='coerce')`；首个原始非空但转换失败的格子报错 `第 N 列（列名：'col'）第 M 行的值 'xxx' 不是数值` 并退出脚本（行号 = CSV 物理行号，表头为第 1 行）
- **SMILES 列验证**（`_validate_smiles`）：用户确认 SMILES 列后对每列逐行调用 `Chem.MolFromSmiles()`；首个解析失败的格子报错 `第 N 列（列名：'col'）第 M 行的值 'xxx' 不是有效的 SMILES` 并退出脚本；RDKit 未安装时仅打印警告，不中断流程
- **加载时机调整**：CSV 在路径确认后立即完整加载（不再仅读表头），两次验证复用同一 DataFrame，避免重复 IO
- **辅助函数** `_col_display(col, columns)`：统一生成 `第 N 列（列名：'col'）` 格式，供两个验证函数共用

**yonod.py v2 + run_yonod.py：交互向导重构，列必须显式指定**
- **yonod.py 重构**：
  - 加载 CSV 后显示带字母序号（A/B/C…）和数字序号（1/2/3…）的列索引表，用户可通过**列名 / 单字母 / 序号**三种方式指定任意列，每种方式均有正则验证
  - **标签列**：单列必填，`_RE_LETTER` / `_RE_NUMBER` 解析后回显确认
  - **SMILES 列**：多列必填（至少一列），空格分隔，自动去重并检查与标签列的重叠
  - **数值辅助列**：多列可选，同上
  - **任务名称**：必填，`_RE_TASK = ^[A-Za-z0-9_\-]+$` 强制英文，不通过则循环重新输入
  - **输出目录**：放在任务名称之后询问；默认值为 `<脚本目录>/result/<task_name>`，自动提示给用户，直接回车接受
  - **描述符 / 模型**：可选，输入非法值时打印警告并过滤，不中断流程
  - 执行前打印等效命令预览，按 Enter 确认后调用 `run_yonod.main()`
- **run_yonod.py 变更**：
  - `--label-col` 改为 `required=True`（原为 `default=None` 允许自动推断）
  - `--smiles-cols` 改为 `required=True`（原为 `default=None` 允许自动探测）
  - 移除 `--smiles-threshold` 参数（自动探测路径已关闭，阈值参数无意义）
  - `load_csv_with_roles()` 调用移除 `smiles_threshold` 关键字参数
- **迁移说明**：直接使用 `run_yonod.py` 的命令行调用需补充 `--label-col` 和 `--smiles-cols`，否则 argparse 报错退出

**yonod.py：以 Python 交互脚本替代 yonod.bat**
- **动机**：`yonod.bat` 经三轮修复后仍因 cmd.exe 变量展开时序、括号解析、编码等问题频繁报错；根本原因是 cmd.exe 脚本语言本身的解析缺陷无法通过修补彻底消除
- **替代方案**：新建 `yonod.py`，使用标准 Python `input()` 完成全部 8 步交互，构建参数列表后直接调用 `run_yonod.main()`（不经过 shell），无需 conda activate 命令、无编码问题、无特殊字符限制
- **等效功能**：CSV 路径（自动去引号）→ 标签列 → SMILES 列 → 数值辅助列 → 输出目录 → 任务名称 → 描述符选择 → 模型选择；执行前打印等效命令预览，按 Enter 确认后运行
- **用法**：在激活 conda 环境后执行 `python yonod.py`，或在 IDE / Jupyter 终端中运行

**yonod.bat 变量拼接失效修复（三次修复）**
- **根本原因**：`if` 块内使用 `set CMD=%CMD% ...` 拼接命令时，cmd.exe 在**解析阶段**（parse time）即展开 `%CMD%`，导致多行 `if (...) set CMD=...` 块内的变量值为空，产生 `'PUT_DIR' 不是内部或外部命令`、`'CMDSMILES_COLS' 不是内部或外部命令` 等运行时错误
- **修复**：在文件头加 `setlocal enabledelayedexpansion`，将 `if` 块内所有 `%CMD%` 改为 `!CMD!`（延迟展开，在执行阶段取值）；同时将多行 `if (...) { set }` 改为单行 `if ... set`（去除括号），彻底消除括号解析歧义
- **验证**：8 条输入分支（SMILES_COLS / NUMERIC_COLS / OUTPUT_DIR / TASK_NAME / DESCS / MODELS 均为空或非空）下，命令拼接结果与预期一致

**yonod.bat 乱码与交互失效修复（二次修复）**
- **二次修复**：`set /p` 提示字符串中的 `( )` 被 cmd.exe 解析为**代码块分隔符**（括号内文字被当作命令执行），`< >` 被解析为**重定向符**，导致 `'3' 不是内部或外部命令` 等运行时错误；修复方法：从全部 8 条 `set /p` 提示字符串中彻底去除 `() <>`，改用逗号和中文标点替代，验证无 BAD 字符

**yonod.bat 乱码与交互失效修复（首次修复）**
- **根本原因**：bat 文件原以 UTF-8 写入，但 `cmd.exe` 默认按系统 ANSI 代码页（GBK/CP936）解析 bat 文件源文本，导致中文字符乱码；同时 `chcp 65001 + set /p` 在 Windows 上有已知 bug（提示字符串乱码、无法接收输入）
- **修复**：使用 PowerShell `[System.IO.File]::WriteAllText()` 以 GBK（CP936）编码写入 bat 文件，去掉 `chcp 65001` 改为依赖系统默认代码页
- **新增步骤 [5/8]**：增加输出目录输入步骤，留空走默认路径；`OUTPUT_DIR` 非空时以带引号方式传入 `--output-dir "%OUTPUT_DIR%"`，支持含空格和中文的路径
- **验证**：`run_yonod.py --output-dir "results/测试输出目录"` 正常输出 metrics_summary.csv + report.html

### Added

**通用化子包 yonod_yield/universal/（第5步，14.6.3）**
- `yonod_yield/universal/report.py`：通用报告生成器（§14.5 完整实现）
  - `rank_combinations(metrics_df)`：加权排名函数，公式 R²×0.5 + (1−RMSE/max)×0.3 + (1−MAE/max)×0.2；前三名自动生成中文推荐理由；列名兼容 run_yonod.py（`descriptor`/`r2_mean`）和 rank_combinations 直接调用（`desc_name`/`r2`）两种格式
  - `generate_report(metrics_df, task_info, out_dir)`：输出自包含 HTML，含 5 个段落：任务信息头 / 描述符×模型结果矩阵（R²颜色渐变，浅蓝→深绿）/ §14.5.2 固定指标解释 / 加权排名推荐（前三名+折叠完整排名）/ 散点图画廊（扫描 scatter_*.png 内嵌 base64）
- `test_report.py`：5 个验证用例（排名顺序 / 推荐理由非空 / HTML 含指标解释文字 / HTML 嵌入任务名 / 全 NaN 安全处理），全部通过
- `run_yonod.py`（更新）：grid 评估完成后自动调用 `generate_report()`，输出 `results/<task>/report.html`；报告失败不阻断主流程

**已验证的关键行为（供后续步骤参考）**：
- mock 4 行 metrics_df 排名：morgan×rf（R²=0.87）正确排第1，score=0.759
- HTML 报告文件含所有§14.5.2 固定文字和权重参数（0.5 / 0.3 / 0.2）
- 全链路冒烟测试（200 行酰胺缩合，morgan×rf）：`metrics_summary.csv` + `report.html` 均正常生成

**通用化子包 yonod_yield/universal/（第4步，14.6.3）**
- `yonod.bat`：Windows 双击交互式向导（按 §14.4.3 模板实现）
  - 7 步引导：CSV 路径 → 标签列 → SMILES 列（可选） → 数值辅助列（可选） → 任务名称（可选） → 描述符选择（可选） → 模型选择（可选）
  - 拖拽支持：自动去除路径首尾引号
  - 命令预览：执行前打印完整 `python run_yonod.py ...` 命令供用户核对，按任意键确认
  - conda 环境激活：`call conda activate yonod-yield`，失败时报错暂停
  - 已知局限（见 §14.4.3）：列名含 `&`/`|`/`>`/`<` 特殊字符时不适用，建议改用命令行直接传参

**通用化子包 yonod_yield/universal/（第3步，14.6.3）**
- `run_yonod.py`：通用化 CLI 入口（267 行），完整实现 csv_loader → feature_builder → 模型评估 → metrics_summary.csv 的 pipeline
  - 全 §14.3.1 CLI 参数：`--csv`、`--label-col`、`--smiles-cols`、`--numeric-cols`、`--task-name`、`--descriptors`、`--models`、`--smiles-threshold`、`--output-dir`、`--cv`、`--nrows`、`--append`、`--heartbeat`、`--svm-subsample` 等
  - 无数值列时直接调用 `model.cross_validate(X_smiles, y)`，复用现有模型适配器接口
  - 有数值列时执行 `_cv_with_numeric()`：KFold 循环内每折独立 `StandardScaler`（防数据泄露，符合 §14.2 规范）；AutoGluon 使用全局 scaler 并打印警告（已知局限）
  - 多描述符 × 多模型 grid 自动执行，含心跳线程、时间戳镜像日志
  - 输出目录：`results/<task-name>建模报告/`，`--append` 支持增量写入
- 验证（冒烟）：`python run_yonod.py --csv 数据集/酰胺缩合数据集.csv --label-col yield --descriptors morgan --models rf --nrows 200`

**已验证的关键行为（供后续步骤参考）**：
- SMILES 自动探测：200 行子集中 `sub_1_smiles`/`sub_2_smiles`/`product_smiles`/`base_id`/`solvent_id` 正确纳入（5 列），`activation_id`/`additive_id` 正确排除
- 特征矩阵：5 SMILES 列 × 1024 Morgan 维 = shape (161, 5120)
- 指标输出与 `metrics_summary.csv` 写入正常；日志镜像文件自动创建

**通用化子包 yonod_yield/universal/（第2步，14.6.3）**
- `yonod_yield/universal/feature_builder.py`：通用特征构建器（119 行）
  - `_get_descriptor()`：按需懒加载描述符，避免 torch_geometric / HuggingFace 等重依赖在无关场景预先导入
  - `build_universal_features(smiles_cols, numeric_cols, df, desc_name)`：主入口，返回 `(X_smiles, X_numeric, valid_mask)`
    - 多 SMILES 列描述符向量按列拼接，顺序与 `smiles_cols` 一致
    - 数值辅助列以原始 float32 返回，**不做归一化**（归一化在 KFold 循环内逐折完成，见 §14.2）
    - 失败行（RDKit 解析失败）通过 `valid_mask` 统一过滤，并打印 stderr 警告
- `test_feature_builder.py`：5 个验证用例（单 SMILES 列维度 / 双列维度 = 2×d / 数值列形状 / 无效 SMILES 过滤 / 未知描述符名称报错），全部通过

**已验证的关键行为（供后续步骤参考）**：
- `morgan` 描述符：单 SMILES 列 → shape (n, 1024)；双列 → shape (n, 2048)
- 数值列不参与归一化，`X_numeric` 原样输出 float32，shape (n, k)；无数值列时返回 `None`
- 描述符懒加载：`morgan`/`maccs` 不依赖 PyTorch，可独立验证；`fisd`/`molmetalm` 仅在被选用时导入

**通用化子包 yonod_yield/universal/（第1步，14.6.3）**
- `yonod_yield/universal/__init__.py`：子包初始化
- `yonod_yield/universal/csv_loader.py`：通用 CSV 加载器（205 行）
  - `auto_detect_smiles_cols()`：对每列随机抽样 50 行用 RDKit 检测 SMILES 有效率，超过阈值（默认 0.5）则纳入 SMILES 列集合
  - `load_csv_with_roles()`：主入口，支持显式指定或自动探测 smiles_cols / numeric_cols / label_col；标签列 NaN 行自动过滤；Windows GBK 终端 UTF-8 输出兼容
  - `LoadedDataset` dataclass：统一返回 df + 列角色元数据
- `test_csv_loader.py`：5 个验证用例（酰胺缩合自动探测 / 显式指定 / ECC 显式 + 温度辅助列 / ECC 自动探测 / 错误处理），全部通过

**已验证的关键行为（供后续步骤参考）**：
- 酰胺缩合数据集中 `base_id` / `solvent_id` 存储真实 SMILES，自动探测正确纳入；`activation_id` / `additive_id` 含逗号多片段 SMILES 和 `(无)` 占位符，有效率低于阈值，正确排除
- ECC 数据集中只有 `Ligand_SMILES` / `Product_SMILES` 通过探测；`Temp (K)` 为 float 列，不被探测为 SMILES，需用户显式声明 `--numeric-cols`

**Track B：Ni 催化不对称偶联 ΔΔG 预测**
- `run_ecc_prediction.py`：ECC 专用入口，4×4 grid（2 描述符 × 4 模型），支持 `--append`、`--svm-subsample`、`--heartbeat` 等 9 个 CLI 旋钮；输出到 `results/ecc_results/`
- `yonod_yield/features/ecc_dataset.py`：ECC 数据加载器（读取 `Raw_Dataset.csv`，提取 Ligand_SMILES / Product_SMILES / Temperature / ΔΔG，过滤无效 SMILES）
- `yonod_yield/metrics/ee_metrics.py`：`ddG_to_ee()` 转换函数及 `ee_mae()` 指标（ΔΔG → 对映体过量 ee%，R=0.001987 kcal/(mol·K)）
- `yonod_yield/metrics/__init__.py`：新指标子包
- `test_ecc_load.py`：ECC 数据加载与特征构建烟测

**数据集整理**
- `数据集/镍催化偶联数据集/`：从 Enantioselective-Cross-Coupling-Prediction 提取整理（6590 条 `Raw_Dataset.csv` + 5 个预计算描述符 CSV + 文献 PDF/MD + 数据集说明 `README.md`）
- `数据集/GraphRXN数据集/`：从 `GraphRXN-master` 批量提取并转换，共 142 个 CSV，含四个子集：
  - `BuchwaldHartwig/`：Dreher & Doyle HTE 数据 3955 条（原始 xlsx 转 CSV，16 个分折文件 + 合并原始文件）
  - `SuzukiMiyaura/`：aap9112 Science 2019 数据集 5760 条（16 列，含 SMILES）
  - `Denmark/`：相转移催化 10-CV 预分割（40 个 CSV）
  - `InHouse/`：实验室内部 HTE 1558 条，按底物分 4 组，各 5-CV 分割
  - `Stat/`：GraphRXN 与基线方法的 R² 性能对比表（4 个 CSV）
- `数据集/酰胺缩合数据集.csv`：原数据集文件重命名（内容不变，47015 条）

**文档**
- `CHANGELOG.md`（本文件）
- `YONOD项目构建计划书.md` 新增 §11–§14：
  - §11 ECC 数据集调研记录（数据规模、列说明、与 Track A 的架构共性）
  - §12 ECC-Track 扩展实施计划（文件清单、特征构建方式、ee% 换算公式）
  - §13 通用化重构方案分析（对 5 条设计方案的逐条评价与改进建议）
  - §14 通用化重构确认方案与实施计划（数值列归一化量化分析、CLI 参数规范、BAT 向导设计、文件新增清单）

### Removed

**废弃入口脚本清理（yonod.py 合并后遗留）**
- **删除** `run_yield_prediction.py`：Track A 专用入口，已由 `yonod.py` 通用 CLI 完全替代
- **删除** `run_ecc_prediction.py`：Track B (ECC) 专用入口，已由 `yonod.py --smiles-cols ... --numeric-cols ... --label-col ddG` 替代
- **删除** `generate_report.py`：独立报告生成器，报告生成已集成进 `yonod.py` pipeline（通过 `yonod_yield/universal/report.py`）
- **删除** `test_run_yield.py`：通过 subprocess 调用 `run_yield_prediction.py` 的冒烟测试，被删文件不复存在
- **删除** `test_ecc_load.py`：ECC 专用加载器的冒烟测试，与 Track B 旧入口一并废弃
- **迁移** `test_csv_loader.py` / `test_feature_builder.py` / `test_report.py` → `tests/`：仍有效的通用子包单元测试，集中到 `tests/` 目录并加入 `.gitignore`（仅供本地开发使用）

### Changed

**管线通用化（v1.1.0 范围，随本版本一同发布）**
- 数据集读取契约改为通用 2 列 schema：第 1 列 SMILES、第 2 列浮点标签；其余列忽略。原 9 列酰胺反应级特征改为可选 legacy 路径。
- 反应级 6 分子拼接（`ReactionFeaturizer`）从默认 pipeline 移除；新默认 `MoleculeFeaturizer`：单分子 SMILES → 描述符向量。
- 输出目录从 `results/` 改为 `results/<数据集 stem>建模报告/`，多数据集横向对比时互不覆盖。
- HTML 报告标题与术语改为通用版本：标题用数据集名，"产率" → "标签 / target"，"反应" 字样在通用语境中删除。

**文件级修改**
- `README.md`：增加双轨道（Track A / Track B）说明，ECC 实验结果表，GraphRXN 参考项说明，更新仓库目录树
- `yonod_yield/evaluate.py`：新增可选 `ee_mae` 字段，向后兼容现有 Track A 调用
- `yonod_yield/plot.py`：散点图标签适配通用 target 名称
- `generate_report.py`：报告模板术语通用化
- `run_yield_prediction.py`：CLI 旋钮与输出路径更新

### Fixed
- `run_ecc_prediction.py`：补充 `--append` 参数定义，修复 `unrecognized arguments: --append` 运行时错误

### Removed
- `数据集/酰胺缩合反应数据集.csv`：已重命名为 `数据集/酰胺缩合数据集.csv`（内容不变）

### Deprecated
- `ReactionFeaturizer` + `reagent_cache.py`：仍保留可调用，但不在 `evaluate.py` 的注册表中；后续版本可能完全移除或抽到 `legacy/` 子包。

---

## [1.0.0] — 2026-05-16

首个公开发布，**酰胺缩合反应产率预测专题**。

### Added
- 4 类分子描述符
  - `MorganDescriptor`（RDKit ECFP4，1024 bit）
  - `ATMOMACCSDescriptor`（RDKit MACCS keys，166 维）
  - `FISDDescriptor`（QM9 预训练双 GCN + TwoInOne MLP，50 维，权重已 inline 至 `WEIGHTS/FISD/`）
  - `MolMetaLMDescriptor`（HuggingFace Llama base，768 维，attention-masked mean-pool）
- 4 类机器学习适配器
  - `XGBYieldModel`（GPU `tree_method='hist'`，5 折 CV）
  - `RFYieldModel`（sklearn 300 棵树，5 折 CV）
  - `SVMYieldModel`（RBF SVR + 自动 PCA，子采样训练以应对 O(n²) 复杂度）
  - `AutoGluonYieldModel`（80/20 holdout + bagging+stacking）
- `ReactionFeaturizer`：6 个分子（2 底物 + 4 试剂）描述符向量拼接
- `reagent_cache.py`：试剂 SMILES 去重 + 描述符预计算缓存（`(无)` → 零向量、`,` → `.` 多片段预处理）
- 主入口 `run_yield_prediction.py`：4×4 grid 自动执行 + 心跳 + 文件日志 + `--append` 合并 + 7 个 CLI 旋钮
- 报告生成器 `generate_report.py`：自包含 HTML（base64 内嵌 PNG），含术语解释、推荐表、热力图、散点画廊
- 诊断工具 `verify_morgan_rf.py`：单独验证 RF 在高维特征上的耗时
- 烟测脚本 `test_run_yield.py`
- 文档：`README.md`、`MIGRATION.md`（云端/Linux/GitHub 迁移指南）、`YONOD项目构建计划书.md`（含设计决策与 13 条 Q&A）
- 协议：CC BY-NC 4.0
- `.gitignore` + `requirements.txt`

### Notes
- 数据集：47015 条酰胺缩合反应（公开来源），R²（5 折 CV）实测范围 0.58~0.87；冠军组合 Morgan × AutoGluon (R²=0.874)。
- 仓库初始大小约 40 MB（含数据 CSV + FISD 3 个 `.pth`）；MolMetaLM 权重 500 MB 需从 HuggingFace 单独下载。
- 已在 Windows 11 + Python 3.9 + CUDA 12.1 + torch 2.1.2 上跑通；云端 Linux 部署指南见 `MIGRATION.md`。

---

[Unreleased]: https://github.com/thinktraveller/YONOD/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/thinktraveller/YONOD/releases/tag/v1.0.0
