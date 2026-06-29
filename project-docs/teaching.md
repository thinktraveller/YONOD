# YONOD 学习笔记

> **创建时间**：2026-06-11 11:08
> **项目地址**：https://github.com/thinktraveller/YONOD

---

## 项目概览

- **一句话描述**：以 SMILES 为统一输入，系统比较 4 种分子描述符 × 4 种机器学习算法在有机合成反应预测任务上的性能
- **技术栈**：Python 3.9 / PyTorch 2.1.2 / RDKit / Transformers / scikit-learn / XGBoost / AutoGluon
- **项目类型**：化学信息学研究平台 / 机器学习基准测试工具
- **核心价值**：为化学反应预测任务提供标准化的描述符性能评估框架

---

## 架构与模块

### 描述符层 (`yonod/descriptors/`)

项目汇聚了 **5 种分子描述符**，代表了从经典指纹到深度学习的不同技术路线：

#### 1. Morgan ECFP4 (`morgan.py`)
- **维度**：1024 位
- **类型**：环形指纹（Circular Fingerprint）
- **实现**：RDKit `GetMorganFingerprintAsBitVect(radius=2, nBits=1024)`
- **原理**：以每个原子为中心，向外扩展 2 跳（radius=2），将局部子结构哈希映射到 1024 位向量
- **优势**：捕捉分子局部结构特征，对相似子结构敏感
- **应用场景**：药物发现、分子相似性搜索、QSAR 建模

#### 2. ATMOMACCS (`atmomaccs.py`)
- **维度**：166 位
- **类型**：基于预定义化学子结构的二进制指纹
- **实现**：RDKit `GetMACCSKeysFingerprint()` 去除占位 bit 0
- **原理**：检测 166 种预定义的化学结构模式（如羟基、苯环、酯基等）
- **优势**：可解释性强，维度低，计算速度快
- **引用**：上游项目 [Zenodo 18669279](https://zenodo.org/records/18669279) / [DOI:10.1063/5.0308548](https://doi.org/10.1063/5.0308548)

#### 3. FISD (`fisd.py`)
- **维度**：50 维
- **类型**：图神经网络嵌入
- **实现**：双路 GCN（MSE + Cosine 监督）+ TwoInOne MLP 融合
- **架构细节**：
  - **原子特征**：45 维（原子类型 + 度数 + 杂化方式 + 手性 + 物理属性）
  - **GCN_MSE**：5 层 GCNConv (512 隐层) → 50 维
  - **GCN_COS**：5 层 GCNConv (512 隐层) → 50 维
  - **TwoInOne MLP**：concat(MSE, COS) → 4 层全连接 (512 隐层) → 50 维
- **权重**：在 QM9 数据集上预训练（3 个 .pth 文件，约 19 MB）
- **优势**：维度极低（仅 50 维），适合大规模筛选；捕捉分子图拓扑信息
- **引用**：[DOI:10.1039/D5SC00451A](https://doi.org/10.1039/D5SC00451A)

#### 4. MolMetaLM (`molmetalm.py`)
- **维度**：768 维
- **类型**：预训练语言模型嵌入（Physicochemical Knowledge-Guided）
- **实现**：基于 Llama 架构的 Transformer 编码器 + attention-masked mean-pool
- **原理**：将 SMILES 视为"分子语言"，结合 RDKit 提取的物理化学描述符构建 `<S, P, O>` 知识三元组
- **权重**：HuggingFace [wudejian789/MolMetaLM-base](https://huggingface.co/wudejian789/MolMetaLM-base)（约 500 MB）
- **预训练数据**：PubChem 1.1 亿分子数据集
- **专注提取的特征**（关键）：
  - **药物相关物理化学性质**：LogP（脂水分配系数）、TPSA（拓扑极性表面积）、分子量、氢键供受体数
  - **药物成药性指标**：Lipinski 规则（RO5）、血脑屏障渗透性（BBBP）
  - **生物活性性质**：毒性（Tox21）、活性、溶解度、代谢稳定性
  - **分子语义**：通过 Transformer 学习 SMILES 的化学语法和结构-性质关系
- **优势**：在药物发现任务上表现优异，适合预测 ADME 性质
- **局限**（在 YONOD 任务中暴露）：学到的性质面向药物化学，对催化反应预测几乎无用（催化反应需要电子效应、空间位阻、配位能力等）
- **引用**：[DOI:10.48550/arXiv.2411.15500](https://doi.org/10.48550/arXiv.2411.15500)

#### 5. MAF (Molecular Additive Fingerprint) (`maf.py`)
- **维度**：128 位
- **类型**：多分子加和指纹（Multi-Molecule Additive Fingerprint）
- **实现**：对反应中每个组分生成 ECFP (radius=2, 128 bits)，按位加和
- **原理**：
  - 输入格式：点分隔的多分子 SMILES，如 `"CCO.c1ccccc1.CC(=O)Cl.CCNCC.CN(C)C.ClCCl"`
  - 每个组分独立生成二进制 ECFP (每位 0 或 1)
  - 按位加和得到整数向量 (每位 0~6，取决于有多少个组分含该子结构)
- **优势**：
  - 捕捉"集体子结构特征"：某子结构在多少个组分中出现
  - 维度低 (128-d)，适合多组分反应 (6~10 个分子)
  - 相比 Morgan 的"横向拼接"(6×1024=6144-d)，维度降低 48 倍
  - 计算速度快：~1.5 ms/行（6 分子反应），比深度学习描述符快 30~50 倍
- **局限**：
  - 丢失了"哪个组分含该子结构"的信息 (只知道计数)
  - 不适用于需要区分组分角色的任务 (如"催化剂必须含 Pd"的分类)
- **适用场景**：反应产率预测、多组分反应筛选、高通量虚拟筛选
- **注意事项**：
  - MAF **不能使用** `reagent_cache.py` (需要完整反应的多个分子信息)
  - 输入必须经过 `build_universal_features()` 自动预处理为点分隔格式

---

### 特征工程层 (`yonod/features/`)

#### 反应级特征拼接器 (`reaction_featurizer.py`)
- **职责**：将多分子反应（如酰胺缩合的 6 个 SMILES 列）的描述符横向拼接
- **关键文件**：`reaction_featurizer.py:33-57`
- **对外接口**：`featurize_reaction(smiles_dict, descriptor) -> (features, mask)`
- **特殊处理**：
  - `(无)` 字符串 → 零向量
  - `,` 分隔的盐型 SMILES → 转换为 `.` 格式

#### 试剂缓存机制 (`reagent_cache.py`)
- **职责**：缓存已计算的试剂描述符，避免重复计算
- **实现**：首次运行时将试剂 SMILES → 描述符的映射保存到 `cache/reagent_feats_<desc>.pkl`
- **性能提升**：对于酰胺缩合数据集（47015 行，底物组合 >> 试剂组合），可将 FISD/MolMetaLM 计算时间从小时级降至分钟级

---

### 通用数据处理 (`yonod/universal/`)

#### CSV 自动探测 (`csv_loader.py`)
- **职责**：自动识别 CSV 中的 SMILES 列和标签列
- **算法**：用 RDKit 解析每列样本，若解析成功率 > 阈值则判定为 SMILES 列
- **关键文件**：`csv_loader.py`

#### 特征矩阵构建 (`feature_builder.py`)
- **职责**：SMILES 描述符 + 数值辅助列（如温度、压力）拼接
- **对外接口**：`build_features(smiles_cols, numeric_cols, descriptor) -> feature_matrix`

#### 报告生成器 (`report.py`)
- **职责**：生成 HTML 可视化报告（内嵌散点图画廊）
- **输出**：`result/<task-name>/report.html`

---

### 模型层 (`yonod/models/`)

#### XGBoost (`xgb_model.py`)
- **实现**：`xgboost.XGBRegressor(tree_method='hist', device='cuda')`
- **优势**：GPU 加速，适合高维稀疏特征

#### Random Forest (`rf_model.py`)
- **实现**：`sklearn.ensemble.RandomForestRegressor(n_estimators=300, n_jobs=-1)`
- **优势**：多核并行，鲁棒性强

#### SVM (`svm_model.py`)
- **实现**：`sklearn.svm.SVR(kernel='rbf')` + 自动 PCA 降维（输入 > 512 维触发）
- **特殊处理**：每折随机子采样 8000 条（RBF SVR O(n²) 复杂度限制）

#### AutoGluon (`autogluon_model.py`)
- **实现**：`autogluon.tabular.TabularPredictor(preset='medium_quality')`
- **优势**：自动集成 16 个子模型（LightGBM、XGBoost、RandomForest、ExtraTrees、CatBoost 等）
- **已知局限**：若使用数值辅助列，采用全局 StandardScaler（非折内归一化）

---

## 核心数据流

```
CSV 输入 (例：dataset/amide-coupling.csv)
  ↓
[csv_loader.py] 自动探测 SMILES 列 + 标签列
  ↓
[descriptors/] 描述符计算（4 种并行）
  ├─ Morgan ECFP4: 1024-d 二进制向量
  ├─ MACCS keys: 166-d 二进制向量
  ├─ FISD: 50-d 连续向量（双 GCN 嵌入）
  └─ MolMetaLM: 768-d 连续向量（Llama 嵌入）
  ↓
[reaction_featurizer.py] 反应级特征拼接
  └─ 例：6 分子 × 1024-d (Morgan) = 6144-d 特征向量
  ↓
[evaluate.py] 5 折交叉验证 × 4 种模型
  ├─ XGBoost (GPU hist)
  ├─ Random Forest (300 树)
  ├─ SVM (RBF + PCA)
  └─ AutoGluon (medium_quality)
  ↓
[metrics_summary.csv] 性能评估（R²、RMSE、MAE）
  ↓
[report.py] HTML 报告 + 散点图
```

---

## 设计说明

### 为什么选择这 4 种描述符？

1. **维度跨越 3 个数量级**：
   - 50 维 (FISD) → 166 维 (MACCS) → 768 维 (MolMetaLM) → 1024 维 (Morgan)
   - 可评估维度对模型性能的影响

2. **技术路线多样性**：
   - **经典指纹**：Morgan（哈希指纹）、MACCS（专家规则）
   - **深度学习**：FISD（图神经网络）、MolMetaLM（预训练语言模型）

3. **计算成本差异**：
   - **快速**：Morgan、MACCS（RDKit API，毫秒级）
   - **中等**：FISD（GCN 推理，秒级）
   - **慢速**：MolMetaLM（Transformer 推理，秒级，GPU 建议）

---

### 为什么需要试剂缓存机制？

在酰胺缩合数据集（47015 条）中：
- **底物组合**：sub_1 × sub_2 = 数千种
- **试剂组合**：activation × additive × base × solvent ≈ 数百种

由于试剂复用率高，缓存机制可以：
1. 避免对相同试剂重复计算描述符
2. 将 FISD/MolMetaLM 的总计算时间从 **~2 小时降至 ~10 分钟**

---

### FISD 的双路 GCN 为何要合并？

- **GCN_MSE**：用均方误差监督训练，关注数值预测精度
- **GCN_COS**：用余弦相似度监督训练，关注向量方向（拓扑相似性）
- **TwoInOne MLP**：融合两路特征，兼顾数值预测和结构相似性

---

### SVM 为什么要 PCA 降维和子采样？

1. **PCA 降维**：RBF 核 SVM 在高维空间（>512 维）容易过拟合，PCA 降至 256 维可缓解
2. **子采样**：RBF SVR 的时间复杂度为 O(n²)，对 47015 条数据训练一次需 > 1 小时，子采样至 8000 条可在 10 分钟内完成

---

## Q&A 记录

### yonod.py 脚本架构（2026-06-29 新增）

**Q：** 请详细分析 yonod.py 的脚本架构，包括整体结构、入口点、外部模块调用、核心函数和数据流。

**A：** 以下是 yonod.py（1097 行）的完整架构解析。

---

#### 1. 整体架构：双模式入口 + 统一 Pipeline

yonod.py 采用**单文件集成**设计，包含 5 个主要功能区块：

```
yonod.py (1097 行)
├─ [L1-72]    模块导入 + 常量定义
├─ [L74-123]  辅助类：_Tee（日志镜像）、_Heartbeat（心跳打印）
├─ [L125-225] 模型工厂 + KFold CV 逻辑（带数值列的特殊处理）
├─ [L227-524] CLI 模式：argparse 解析 + main() 主流程
├─ [L526-1089] 交互向导模式：wizard() 逐步收集参数
└─ [L1093-1097] 入口路由：根据 sys.argv 长度选择模式
```

**双模式入口机制**：
```python
if __name__ == "__main__":
    if len(sys.argv) > 1:
        sys.exit(main())      # CLI 模式：python yonod.py --csv ...
    else:
        wizard()               # 向导模式：python yonod.py
```

**设计亮点**：
- 向导模式在收集完参数后，构建 `argv` 列表并调用 `main(argv)`，**复用 CLI 逻辑**，避免代码重复
- 日志镜像机制（`_Tee` 类）：stdout 自动同步写入 `run_<timestamp>.log`，每行前缀时间戳
- 心跳监控（`_Heartbeat` 类）：后台线程每 30 秒打印 "still running"，防止长时间运行时无输出

---

#### 2. 入口点与参数解析

##### 2.1 CLI 模式入口（`main(argv)`，L310-523）

**核心流程**：
```python
def main(argv: Optional[List[str]] = None) -> int:
    # 1. 解析参数（argparse）
    args = parse_args(argv)

    # 2. 加载数据集（CSV → LoadedDataset）
    dataset = load_csv_with_roles(
        csv_path=args.csv,
        smiles_cols=args.smiles_cols,
        numeric_cols=args.numeric_cols or [],
        label_col=args.label_col,
        reactant_cols=args.reactant_cols,  # 三分类模式
        product_cols=args.product_cols,
        other_cols=args.other_cols,
        nrows=args.nrows,
    )

    # 3. 描述符 × 模型 笛卡尔积循环
    for desc_name in args.descriptors:
        # 3.1 计算描述符特征矩阵
        X_smiles, X_numeric, mask = build_universal_features(
            smiles_cols=dataset.smiles_cols,
            numeric_cols=dataset.numeric_cols,
            df=dataset.df,
            desc_name=desc_name,
            smiles_roles=dataset.smiles_roles,
        )
        y = dataset.df[dataset.label_col].to_numpy(dtype=np.float64)[mask]

        for model_name in args.models:
            # 3.2 创建模型实例
            model = _make_model(model_name, args)

            # 3.3 KFold CV（有/无数值列分支）
            if X_numeric is None:
                metrics = model.cross_validate(X_smiles, y, cv=args.cv)
            else:
                metrics = _cv_with_numeric(
                    model_name, model, X_smiles, X_numeric, y,
                    cv=args.cv, svm_subsample=args.svm_subsample,
                )

            # 3.4 记录指标 + 生成散点图
            rows.append({...metrics, ...metadata})
            plot_scatter(oof_pred, oof_y_true, desc_name, model_name, out_dir)

    # 4. 生成报告（HTML + Markdown）
    generate_report(metrics_df, task_info, out_dir)
    generate_markdown_report(metrics_df, task_info, out_dir)
```

**关键参数组（argparse，L229-273）**：

| 参数组 | 关键参数 | 说明 |
|---|---|---|
| **必填** | `--csv`, `--label-col` | CSV 路径和标签列名 |
| **SMILES 列** | `--smiles-cols` | 传统模式：所有 SMILES 列等价 |
| **三分类模式** | `--reactant-cols`, `--product-cols`, `--other-cols` | 区分反应物/产物/其他参与者 |
| **数值辅助列** | `--numeric-cols` | 温度、压力等数值特征 |
| **描述符/模型** | `--descriptors`, `--models` | 默认全选 7 种描述符 × 4 种模型 |
| **输出控制** | `--task-name`, `--output-dir`, `--output-format` | 任务名、输出目录、报告格式 |
| **数据集元信息** | `--dataset-citation`, `--dataset-url`, `--dataset-notes` | 显示在报告中 |

##### 2.2 交互向导模式（`wizard()`，L768-1088）

**核心流程**：11 步逐步收集参数（部分步骤合并显示）

```python
def wizard() -> None:
    # [1/11] CSV 路径
    csv_path = _ask_required("路径")
    df = pd.read_csv(csv_path, encoding='utf-8-sig' or 'gbk')
    _show_columns(df.columns)  # 展示列序号和列名

    # [2/11] 标签列
    label_col = _ask_single_col("标签列", df.columns, required=True)
    _validate_label(df, label_col, columns)  # 数值合法性验证

    # [3/11] SMILES 列
    smiles_cols = _ask_multi_cols("SMILES 列", df.columns, required=True)
    _validate_smiles(df, smiles_cols, columns)  # RDKit 逐行验证

    # [3a-3d/11] 列角色分类（可选）
    enable_roles = _ask_optional("是否启用列角色分类？[y/N]", default="N")
    if enable_roles in ("y", "yes"):
        reactant_cols = _ask_multi_cols("反应物列", smiles_cols, required=True)
        product_cols = _ask_multi_cols("产物列", remaining_cols, required=True)
        other_cols = [自动归类剩余列]

    # [4/11] 数值辅助列
    numeric_cols = _ask_multi_cols("数值辅助列", df.columns, required=False)

    # [5/11] 任务名称
    task_name = _ask_required("任务名称")  # 正则校验：仅允许 [A-Za-z0-9_-]

    # [6/11] 输出目录
    output_dir = _ask_optional("输出目录", default=f"result/{task_name}")

    # [7/11] 描述符选择
    descs = _ask_optional("描述符（空格分隔，留空=全选）")

    # [8/11] 模型选择
    models = _ask_optional("模型（空格分隔，留空=全选）")

    # [9/11] 数据集来源
    dataset_citation = _ask_optional("文献引用")
    dataset_url = _ask_optional("开源地址")
    dataset_notes = _ask_optional("备注")

    # [10/11] 报告输出格式
    fmt = _ask_optional("输出格式", default="both")  # html / md / both

    # [11/11] 确认配置
    print("即将执行（等效命令）:")
    print(" ".join(cmd_parts))
    _input("按 Enter 确认执行，Ctrl+C 取消... ")

    # 构建 argv 并调用 main()
    argv = ["--csv", str(csv_path), "--label-col", label_col, ...]
    sys.exit(main(argv))
```

**辅助函数（L528-671）**：

| 函数名 | 作用 |
|---|---|
| `_ask_required()` | 必填项输入，空值时循环提示 |
| `_ask_optional()` | 可选项输入，支持默认值 |
| `_col_index_to_excel()` / `_excel_to_col_index()` | 列索引 ↔ Excel 风格列名（A, B, ..., Z, AA, AB, ...） |
| `_show_columns()` | 展示列序号和列名，格式：`A(1)  sub_1_smiles` |
| `_ask_single_col()` / `_ask_multi_cols()` | 解析用户输入（支持列名/字母/序号） |
| `_validate_label()` | 验证标签列数值合法性（支持 JSON 数组格式如 `[0.85]`） |
| `_validate_smiles()` | RDKit 逐行验证 SMILES（支持多组分，逗号/分号/点分隔） |
| `_normalize_smiles()` | 将各类分隔符统一为 RDKit 标准点分隔（`,;*~` → `.`） |

**用户体验优化**：
- **Excel 列名支持**：用户可输入 `B` 而非 `1`，对应第 2 列
- **多字母列名支持**：支持 `AA`、`AB`、`AZ` 等（Excel 26 列后的列名）
- **智能分隔符识别**：支持复制粘贴含 `,` 或 `;` 的多列选择
- **实时验证反馈**：SMILES 格式错误时，打印精确的**行号+列名+组分索引**

---

#### 3. 外部模块调用关系

yonod.py 的依赖关系（按调用顺序）：

```
yonod.py
  ├─ yonod.universal.csv_loader
  │   └─ load_csv_with_roles() → LoadedDataset
  │       ├─ 自动探测 SMILES 列（RDKit 解析率 > 阈值）
  │       └─ 三分类角色映射（reactant/product/other）
  │
  ├─ yonod.universal.feature_builder
  │   └─ build_universal_features() → (X_smiles, X_numeric, mask)
  │       ├─ 懒加载描述符（_get_descriptor）
  │       ├─ 逐列调用 descriptor.featurize(smiles_list)
  │       ├─ 特殊处理：DRFP 使用反应物+产物，其他用全部列
  │       └─ 横向拼接多列 + 合并 mask
  │
  ├─ yonod.descriptors.base
  │   └─ split_multi_smiles() - 多组分 SMILES 拆分工具
  │
  ├─ yonod.models.*
  │   ├─ XGBYieldModel (xgb_model.py)
  │   ├─ RFYieldModel (rf_model.py)
  │   ├─ SVMYieldModel (svm_model.py)
  │   └─ AutoGluonYieldModel (autogluon_model.py)
  │       └─ 统一接口：cross_validate(X, y, cv) → metrics_dict
  │
  ├─ yonod.plot
  │   └─ plot_scatter() - 生成 OOF 预测散点图
  │
  └─ yonod.universal.report
      ├─ generate_report() → HTML 报告（内嵌 Base64 图片）
      └─ generate_markdown_report() → Markdown 报告
```

**模块导入特点**：
- **延迟导入**：模型类在 `_make_model()` 内按需 import，避免启动时加载全部依赖
- **懒加载描述符**：`feature_builder` 通过 `importlib` 动态加载描述符模块
- **依赖隔离**：不使用描述符时，PyTorch / torch_geometric 不被导入

---

#### 4. 核心函数详解

##### 4.1 `_make_model(model_name, args)` - 模型工厂（L127-145）

```python
def _make_model(model_name: str, args: argparse.Namespace) -> Any:
    if model_name == "xgb":
        from yonod.models.xgb_model import XGBYieldModel
        return XGBYieldModel()
    if model_name == "rf":
        from yonod.models.rf_model import RFYieldModel
        return RFYieldModel(
            n_jobs=args.rf_n_jobs,           # 默认 -1（全核并行）
            n_estimators=args.rf_n_estimators,  # 默认 300
            max_depth=args.rf_max_depth,     # 默认 None（不限制）
        )
    if model_name == "svm":
        from yonod.models.svm_model import SVMYieldModel
        sub = args.svm_subsample if args.svm_subsample > 0 else None
        return SVMYieldModel(subsample_n=sub)  # 默认 8000
    if model_name == "autogluon":
        from yonod.models.autogluon_model import AutoGluonYieldModel
        return AutoGluonYieldModel()
    raise ValueError(f"未知模型名称 {model_name!r}")
```

**设计细节**：
- 仅 RF 和 SVM 接受超参数配置（通过 CLI 参数）
- XGBoost / AutoGluon 使用固定配置（简化用户选择）

##### 4.2 `_cv_with_numeric()` - 带数值列的 KFold CV（L150-224）

**问题背景**：当有数值辅助列（如温度、压力）时，需在**每折内部**对数值列 fit StandardScaler，防止数据泄露。

**实现逻辑**：
```python
def _cv_with_numeric(
    model_name: str,
    model: Any,
    X_smiles: np.ndarray,   # 已计算的描述符矩阵
    X_numeric: np.ndarray,  # 原始数值矩阵（未归一化）
    y: np.ndarray,
    cv: int,
    svm_subsample: Optional[int],
) -> Dict[str, Any]:
    # AutoGluon 例外处理：使用全局 scaler + 打印警告
    if model_name == "autogluon":
        print("[warning] AutoGluon + numeric cols: 使用全局 StandardScaler（非折内归一化）")
        scaler = StandardScaler()
        X_num_scaled = scaler.fit_transform(X_numeric)
        X_combined = np.hstack([X_smiles, X_num_scaled])
        return model.cross_validate(X_combined, y, cv=cv)

    # 标准 KFold 流程
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)
    r2s, rmses, maes = [], [], []
    oof = np.full(len(y), np.nan, dtype=np.float64)

    for fold_idx, (tr, te) in enumerate(kf.split(X_smiles)):
        # 关键：每折独立 fit scaler
        scaler = StandardScaler()
        X_num_tr = scaler.fit_transform(X_numeric[tr])    # 仅 train 子集 fit
        X_num_te = scaler.transform(X_numeric[te])        # test 子集 transform

        X_tr = np.hstack([X_smiles[tr], X_num_tr])
        X_te = np.hstack([X_smiles[te], X_num_te])
        y_tr, y_te = y[tr], y[te]

        # SVM 子采样（仅 train 集）
        if model_name == "svm" and svm_subsample and svm_subsample < len(X_tr):
            rng = np.random.default_rng(seed=fold_idx)
            sub_idx = rng.choice(len(X_tr), size=svm_subsample, replace=False)
            X_tr_fit = X_tr[sub_idx]
            y_tr_fit = y_tr[sub_idx]
        else:
            X_tr_fit = X_tr
            y_tr_fit = y_tr

        # 模型训练与预测
        est = model._build(n_features=X_tr.shape[1], n_train=len(X_tr_fit))
        est.fit(X_tr_fit, y_tr_fit)
        pred = est.predict(X_te)

        oof[te] = pred
        r2s.append(r2_score(y_te, pred))
        rmses.append(float(np.sqrt(mean_squared_error(y_te, pred))))
        maes.append(float(mean_absolute_error(y_te, pred)))

    return {
        "r2_mean": float(np.mean(r2s)),
        "r2_std": float(np.std(r2s)),
        "rmse_mean": float(np.mean(rmses)),
        "mae_mean": float(np.mean(maes)),
        "train_time_s": float(time.time() - t0),
        "device": "cpu",
        "oof_pred": oof,       # Out-of-Fold 预测（用于散点图）
        "oof_y_true": y,
    }
```

**关键设计决策**：
- **折内归一化**：StandardScaler 在每折内独立 fit，确保测试集未见过训练集统计量
- **SVM 子采样在 fold 内**：每折独立随机采样，测试集完整预测（无泄漏）
- **AutoGluon 妥协**：无法嵌入 KFold，使用全局 scaler + 显式警告

##### 4.3 `_validate_smiles()` - SMILES 合法性验证（L726-765）

**调用位置**：向导模式第 3 步（`wizard()` L823）

**核心逻辑**：
```python
def _validate_smiles(df: pd.DataFrame, smiles_cols: list, columns: list) -> None:
    """用 RDKit 逐行检查 SMILES 列；发现第一个无效 SMILES 则退出。"""
    try:
        from rdkit import Chem
        from rdkit import RDLogger
        RDLogger.DisableLog("rdApp.*")  # 静默 RDKit 警告
    except ImportError:
        print("  [警告] RDKit 未安装，跳过 SMILES 格式验证。")
        return

    from yonod.descriptors.base import split_multi_smiles

    for col in smiles_cols:
        col_desc = _col_display(col, columns)  # 生成 "第 2 列（列名：'sub_1_smiles'）"
        print(f"  [验证] 检查 SMILES 列 '{col}'（{len(df)} 行）...")

        for raw_idx, val in enumerate(df[col]):
            if pd.isna(val):
                continue

            # 规范化分隔符：, ; * ~ → .
            normalized = _normalize_smiles(str(val))

            # 逐组分验证（支持多组分 SMILES）
            components = split_multi_smiles(normalized)
            for comp_idx, comp in enumerate(components):
                comp = comp.strip()
                if not comp:
                    continue
                if Chem.MolFromSmiles(comp) is None:
                    display_row = raw_idx + 2  # Excel 行号（标题行 + 1-based）
                    comp_desc = f"第 {comp_idx + 1} 个组分 '{comp}'" if len(components) > 1 else f"'{val}'"
                    print(f"\n  [错误] {col_desc} 第 {display_row} 行的 {comp_desc} 不是有效的 SMILES。")
                    print("         请检查数据（是否有乱码、截断或占位符）后重新运行。")
                    sys.exit(1)

        print(f"  [验证] SMILES 列 '{col}' 全量验证通过。")
```

**错误定位精度**：
- **行号**：`display_row = raw_idx + 2`（Excel 1-based + 标题行）
- **列名**：`第 2 列（列名：'sub_1_smiles'）`
- **组分索引**：`第 2 个组分 'Cl'`（多组分 SMILES 时）

**分隔符支持**：
- 逗号（`,`）：阴阳离子对，如 `CCN=C=NCCCN(C)C,Cl`
- 分号（`;`）：组分分隔符，如 `CCO;CC(=O)O`
- 星号（`*`）：反应步骤分隔符，如 `A.B*C.D*E`
- 波浪线（`~`）：组分替代表示，如 `CC(=O)O~CC(=O)O~[Pd]`

##### 4.4 `_normalize_smiles()` - SMILES 规范化（L673-723）

**核心功能**：将各类非标准分隔符统一转换为 RDKit 标准的点分隔形式（`.`）。

**处理流程**：
```python
def _normalize_smiles(smi: str) -> str:
    s = smi.strip()

    # JSON 数组格式预处理（如 ["CCO","CC(=O)O"] → CCO.CC(=O)O）
    if s.startswith('[') and s.endswith(']'):
        if re.match(r'^\[\s*\]$', s):  # 空数组 [] → ""
            return ''
        if '"' in s:  # 检测 JSON 引号
            s = s[1:-1]  # 删除最外层 []
            s = re.sub(r'"([^"]*)"', r'\1', s)  # 删除成对的双引号

    # 步骤 1：替换逗号、分号、波浪线 → .
    s = re.sub(r'\s*[,;~]\s*', '.', s)

    # 步骤 2：替换星号分隔符（但保留原子映射星号，如 [*:1]）
    # 使用否定前瞻断言 (?!:) 确保星号后面不是冒号
    s = re.sub(r'\s*\*\s*(?!:)', '.', s)

    # 步骤 3：压缩连续点号（如 .. → .）
    s = re.sub(r'\.{2,}', '.', s)

    # 步骤 4：移除首尾点号
    return s.strip('.')
```

**设计亮点**：
- **原子映射保护**：`[*:1]` 中的 `*` 不会被替换（化学反应映射标记）
- **JSON 数组支持**：自动识别并删除最外层方括号和双引号
- **空数组处理**：`[]`、`[ ]`、`[  ]` 等空数组统一返回空字符串

---

#### 5. 完整数据流（从 CSV 到报告）

```
┌─────────────────────────────────────────────────────────────┐
│ 用户输入：python yonod.py --csv dataset/amide-coupling.csv │
│           --smiles-cols sub_1 sub_2 ... --label-col yield   │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ Step 1: CSV 加载与列角色推断                                 │
│   csv_loader.load_csv_with_roles()                          │
│     ├─ pd.read_csv(encoding='utf-8-sig')                    │
│     ├─ 自动探测 SMILES 列（RDKit 解析率 > 50%）            │
│     ├─ 三分类角色映射（reactant / product / other）         │
│     └─ 返回 LoadedDataset {df, smiles_cols, numeric_cols,  │
│                             label_col, smiles_roles}        │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ Step 2: 描述符计算（外层循环：7 种描述符）                   │
│   for desc_name in ["morgan", "maccs", ...]:                │
│     build_universal_features(smiles_cols, numeric_cols,     │
│                              df, desc_name, smiles_roles)   │
│       ├─ 根据 desc_name 选择使用的列                        │
│       │   └─ DRFP: 仅反应物+产物                            │
│       │   └─ 其他: 所有 SMILES 列                           │
│       ├─ 逐列调用 descriptor.featurize(smiles_list)         │
│       │   └─ 返回 (features, mask)                          │
│       ├─ 横向拼接多列：np.concatenate([f1, f2, ...], axis=1)│
│       ├─ 合并 mask：row_mask = mask_1 & mask_2 & ...        │
│       └─ 过滤失败行：X_smiles = X_full[row_mask]            │
│                                                             │
│   输出：X_smiles (n_valid, n_cols * desc_dim)               │
│         X_numeric (n_valid, n_numeric_cols) 或 None         │
│         mask (n_total,)                                     │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ Step 3: 模型训练（内层循环：4 种模型）                       │
│   for model_name in ["xgb", "rf", "svm", "autogluon"]:     │
│     model = _make_model(model_name, args)                   │
│                                                             │
│     if X_numeric is None:  # 纯 SMILES 特征                 │
│       metrics = model.cross_validate(X_smiles, y, cv=5)     │
│     else:  # 有数值辅助列                                   │
│       metrics = _cv_with_numeric(                           │
│         model_name, model, X_smiles, X_numeric, y,          │
│         cv=5, svm_subsample=8000                            │
│       )                                                     │
│       └─ KFold 循环内每折独立 fit StandardScaler            │
│                                                             │
│     输出：metrics_dict {r2_mean, r2_std, rmse_mean,         │
│                         mae_mean, train_time_s, device,     │
│                         oof_pred, oof_y_true}               │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ Step 4: 结果记录与可视化                                     │
│   rows.append({                                             │
│     "task_name": "amide_coupling",                          │
│     "descriptor": desc_name,                                │
│     "model": model_name,                                    │
│     "n_samples": int(mask.sum()),                           │
│     "n_total": int(len(mask)),                              │
│     "coverage": float(mask.sum() / len(mask)),              │
│     "feature_dim": int(X_smiles.shape[1]),                  │
│     "cv": 5,                                                │
│     **metrics,  # r2_mean, r2_std, rmse_mean, mae_mean, ... │
│   })                                                        │
│                                                             │
│   plot_scatter(oof_pred, oof_y_true, desc_name, model_name,│
│                out_dir / "pictures")                        │
│     └─ 生成 PNG 散点图：<desc>_<model>.png                  │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ Step 5: 报告生成                                             │
│   df = pd.DataFrame(rows)                                   │
│   df.to_csv(out_dir / "metrics_summary.csv")               │
│                                                             │
│   generate_report(df, task_info, out_dir)                  │
│     └─ 生成 HTML 报告（内嵌 Base64 图片）                   │
│                                                             │
│   generate_markdown_report(df, task_info, out_dir)         │
│     └─ 生成 Markdown 报告（外链图片）                       │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 输出文件（默认 results/<task-name>/）                        │
│   ├─ metrics_summary.csv   - 所有组合的指标汇总             │
│   ├─ report.html           - HTML 可视化报告               │
│   ├─ report.md             - Markdown 报告                 │
│   ├─ run_<timestamp>.log   - 控制台镜像日志                │
│   └─ pictures/             - 散点图画廊                     │
│       ├─ morgan_xgb.png                                     │
│       ├─ morgan_rf.png                                      │
│       └─ ...                                                │
└─────────────────────────────────────────────────────────────┘
```

**关键性能优化点**：
1. **试剂缓存**（酰胺缩合专用）：复用率高的试剂 SMILES 仅计算一次
2. **懒加载描述符**：未选中的描述符模块不加载（避免 PyTorch 预先导入）
3. **批处理推理**（FISD/MolMetaLM）：batch_size=32，充分利用 GPU
4. **SVM 子采样**：训练集降采样至 8000 条，测试集完整预测
5. **心跳监控**：长时间运行任务（如 MolMetaLM）每 30 秒打印进度

---

### 通用问题

**Q：YONOD 项目中汇聚了多少种描述符？每个描述符的作用是什么？**

**A：** YONOD 项目汇聚了 **5 种分子描述符**，作用如下：

1. **Morgan ECFP4** (1024 维)
   - **作用**：捕捉分子的局部结构特征，通过圆形指纹编码以每个原子为中心的 2 跳邻域
   - **优势**：对相似子结构敏感，广泛用于药物发现和分子相似性搜索

2. **ATMOMACCS** (166 维)
   - **作用**：基于 166 种预定义的化学子结构模式（如官能团）生成二进制指纹
   - **优势**：可解释性强（每一位对应明确的化学结构），计算速度快

3. **FISD** (50 维)
   - **作用**：通过双路图神经网络将分子图压缩为 50 维连续向量，捕捉拓扑和语义信息
   - **优势**：维度极低，适合大规模筛选；在 QM9 数据集上预训练，泛化能力强

4. **MolMetaLM** (768 维)
   - **作用**：将 SMILES 视为"分子语言"，用 Transformer 编码器学习化学语法和语义
   - **优势**：在大规模分子数据上预训练，适合迁移学习和零样本预测

5. **MAF (Molecular Additive Fingerprint)** (128 维)
   - **作用**：对多组分反应中每个分子生成 ECFP 并按位加和，捕捉"集体子结构特征"
   - **优势**：维度低（相比横向拼接降低 48 倍），计算速度快，适合多组分反应预测

**总结**：这 4 种描述符代表了从经典指纹（Morgan、MACCS）到深度学习（FISD、MolMetaLM）的技术演进，维度从 50 到 1024 跨越 3 个数量级，为化学反应预测任务提供了多样化的特征表示选择。

---

**Q：阅读《描述符综合比较.html》报告后，为什么分子指纹（Morgan/MACCS）作为"不包含化学性质的描述符"，相比其他描述符有近乎碾压的优势？**

**A（启发性思考方向，非确定结论）：**

这是化学信息学中最耐人寻味的悖论之一。报告显示：

**核心矛盾**：
- **Morgan/MACCS**：纯拓扑结构哈希，不含电荷、轨道能量、pKa 等化学性质 → **8/9 体系最优**
- **FISD/MolMetaLM**：声称学到了"化学语义"和"分子性质" → 大部分体系显著落后

以下是 10 个启发性思考方向：

### 1. 任务本质：拓扑 vs 性质
- 这些反应预测任务是在学习**"相似底物 → 相似结果"的统计规律**，还是在理解**化学反应机理**？
- Morgan 的圆形指纹天然擅长捕捉**局部拓扑匹配**（"底物 A 和 B 的取代基在同一位置是否相似"）
- 类比：预测房价时，GPS 坐标（拓扑）比装修风格（语义）更重要

### 2. 预训练数据域不匹配
- FISD 在 QM9（小分子量子化学性质）上预训练，MolMetaLM 在大规模分子数据上预训练
- 但下游任务是**催化反应**，涉及金属配体、溶剂、特殊试剂
- 报告证据：BH 偶联为钯催化，但钯未输入，molmetalm 金属描述符"完全失效"
- 启示：预训练学到的"化学性质"可能不是任务需要的性质

### 3. 浅层模型无法激活深度特征
- XGBoost/RF/SVM 都是浅层模型，依赖特征工程
- Morgan 的二进制稀疏向量天然适合树模型（if bit[42]=1 then ...）
- FISD/MolMetaLM 的连续嵌入可能需要深度模型（MLP）才能充分利用
- 报告证据：FISD 在 Rh 体系唯一领先，用的是 **AutoGluon**（集成了神经网络）

### 4. 信息密度 vs 冗余
- Morgan 1024 位，MACCS 166 位 —— 稀疏但每一位有明确意义
- MolMetaLM 768 维 —— 密集但可能包含大量**任务无关的冗余**
- 在中小数据集（500-7000 样本）上，冗余信息可能干扰学习

### 5. 多分子拼接稀释全局语义
- 酰胺缩合：6 列 SMILES 拼接 → Morgan 每分子的局部子结构信息仍保留在对应 1024 位
- MolMetaLM 拼接后：6 个"全局分子表示"横向拼接成 4608 维
- 浅层模型难以学习"哪 768 维属于底物，哪 768 维属于溶剂"

### 6. Occam's Razor（奥卡姆剃刀）
- 简单假设更可能正确：反应预测可能只需"拓扑相似性"
- DHP 体系 R²=0.920，报告诊断："产物 SMILES 提供了隐式反应完成度信息"
- 暗示：模型可能在学**表观相关性**而非**因果机理**

### 7. 错误的先验知识有害
- MolMetaLM 预训练时学到了"钯配合物的结构-性质关系"
- 但在 Pd 催化任务中，钯催化剂本身**没有输入**
- 模型可能试图从底物 SMILES 中"推断催化剂类型"，导致错误归纳偏置
- 启示：预训练的"化学知识"是高度上下文依赖的

### 8. FISD 唯一领先的体系
- Rh 酰胺化：腙（R-CH=N-NH-R'）、恶唑啉酮（五元含氮杂环）
- FISD 的 45 维原子特征（杂化、手性、范德华半径）捕捉到了 Morgan 遗漏的精细结构差异
- 启示：深度学习描述符不是普遍更好，而是在**特定结构域**有优势

### 9. 数值化学性质 vs 隐式化学性质
- Ni 体系：显式加入**温度（K）**作为数值辅助列，R²=0.857
- 但 FISD/MolMetaLM 的隐式化学性质反而无用
- 可能原因：下游任务需要的性质（温度、极性、pKa）是**任务特定的**，而预训练模型编码的是"溶解度"等**通用性质**

### 10. 小数据集过拟合风险
- FISD 双 GCN + MLP 参数量 > 1M，MolMetaLM Transformer 参数量 > 100M
- 下游任务只有 653-6590 样本
- 可能过拟合了预训练数据的分布，泛化能力不如无参数的 Morgan

---

**综合启示**：这个现象可能是多种因素的叠加效应 —— 任务本质、数据域不匹配、模型容量、特征拼接方式、样本量限制。

**可验证的假说**：
1. 换成深度下游模型（MLP 替代 XGBoost），看 FISD/MolMetaLM 性能是否提升
2. 在 QM9 数据集上比较 FISD vs Morgan，验证是否真的学到了有用的化学性质
3. 在 Morgan 特征中加入 RDKit 的 200 维理化描述符（MolWt、LogP、TPSA），看性能是否提升
4. 控制变量实验：固定模型（都用 AutoGluon），只改变描述符，观察差异

---

**Q：MolMetaLM 专注于将分子的什么特征提取出来？**

**A：** MolMetaLM 专注于提取**药物相关的物理化学性质（Physicochemical Knowledge）**，而非纯拓扑结构或催化反应相关性质。

**核心机制**：
- 将分子 SMILES 与 RDKit 提取的物理化学描述符构建成 `<S, P, O>` 知识三元组
- 在 PubChem 1.1 亿分子数据集上通过自监督学习（恢复被掩码的 Token）预训练

**具体提取的特征**：

1. **药物成药性指标**（Lipinski 规则）：
   - LogP（脂水分配系数）：反映分子在脂质和水相中的分配，预测口服吸收
   - TPSA（拓扑极性表面积）：预测血脑屏障渗透能力
   - 分子量（MolWt）
   - 氢键供体/受体数（HBD/HBA）

2. **生物活性性质**：
   - 毒性（Tox21 数据集）
   - 血脑屏障渗透性（BBBP）
   - 溶解度（Solubility）
   - 代谢稳定性

3. **分子语义**：
   - 通过 Transformer 学习 SMILES 的化学语法
   - 学习结构-性质的深层语义关系

**为什么在催化反应预测中表现最差？**

催化反应预测需要的性质与药物化学完全不同：

| 药物化学（MolMetaLM 擅长） | 催化反应（YONOD 任务需要） |
|---|---|
| LogP、TPSA、毒性 | Hammett 常数、HOMO/LUMO 能级 |
| 血脑屏障渗透性 | Sterimol 参数（空间位阻）|
| 代谢稳定性 | Tolman 锥角（配体配位能力）|
| 溶解度 | pKa、离去基团能力 |

**类比**：就像让一个精通**建筑美学**（对称性、色彩、风格）的专家去评估**建筑抗震能力**（需要力学、材料学），虽然都是关于"建筑"的知识，但**知识域完全不匹配**。

**报告中的证据**：
- BH 偶联（钯催化）：钯未输入 → 金属相关知识变成噪声
- Pd C-H 官能团化：MolMetaLM vs Morgan 差距达 0.227（最大）
- 诊断："molmetalm 金属描述符完全失效"

**启示**：预训练模型的"化学知识"是高度**任务和数据域依赖**的，不能盲目认为"深度学习 = 更好"。

---

### 描述符模块

**Q：描述符模块是如何定义和运作的？请从架构设计、接口约定、计算逻辑、调用关系四个维度详细解析。**

**A：** 以下是 YONOD 描述符模块的完整技术解析。

---

#### 1. 架构设计：基于抽象基类的插件式架构

**核心设计原则**：所有描述符遵循统一接口规范，通过继承 `BaseDescriptor` 抽象基类实现多态。

```python
# yonod/descriptors/base.py
class BaseDescriptor(abc.ABC):
    name: str = "base"           # 描述符唯一标识符（用于注册和缓存）
    output_dim: int = 0          # 输出向量维度（类属性，子类必须设置）

    @abc.abstractmethod
    def featurize(self, smiles_list: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """核心接口：SMILES 列表 → (特征矩阵, 有效性掩码)"""
        raise NotImplementedError
```

**设计亮点**：

1. **类属性契约**：`name` 和 `output_dim` 在类加载时即可访问，无需实例化（工厂模式友好）
2. **统一返回格式**：`(features, mask)` 元组
   - `features`：`np.ndarray` shape `(n, output_dim)`，失败行填充零向量
   - `mask`：`bool` 数组 shape `(n,)`，`True` 表示该行成功计算
3. **失败容错机制**：无效 SMILES 不中断批处理，而是标记在 `mask` 中，由调用方统一过滤

**当前注册的 4 个具体实现**（在 `yonod/evaluate.py` 中注册）：

```python
DESCRIPTOR_REGISTRY = {
    "morgan":    MorganDescriptor,      # 1024 维 ECFP4 圆形指纹
    "atmomaccs": ATMOMACCSDescriptor,   # 166 维 MACCS keys 结构指纹
    "fisd":      FISDDescriptor,        # 50 维双 GCN 图嵌入
    "molmetalm": MolMetaLMDescriptor,   # 768 维 Transformer 语义嵌入
}
```

---

#### 2. 接口约定：统一的 `featurize` 协议

**方法签名**：

```python
def featurize(self, smiles_list: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Args:
        smiles_list: SMILES 字符串列表，长度 n
                     - 空字符串 / None 调用方已预处理为 ""
                     - 试剂"(无)"已在上游转换为 None，本层不感知

    Returns:
        features: shape (n, output_dim), dtype=np.float32
                 - 成功行：描述符向量
                 - 失败行：全零向量（保持矩阵对齐）
        mask:     shape (n,), dtype=bool
                 - True: RDKit 解析成功 + 描述符计算成功
                 - False: 解析失败 / 模型推理异常
    """
```

**调用方责任**（上游 `reaction_featurizer.py` / `feature_builder.py`）：

1. 传入的 `smiles_list` 长度与数据集行数对齐
2. 根据返回的 `mask` 过滤 `y` 标签向量：`y_valid = y[mask]`
3. **不做二次过滤**：下游模型接收到的 X 已是纯净的有效特征

**实现方模板**（所有子类通用模式）：

```python
def featurize(self, smiles_list: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    n = len(smiles_list)
    features, mask = self._empty_outputs(n, dtype=np.float32)  # 预分配零矩阵

    for i, smi in enumerate(smiles_list):
        if not smi:  # 空字符串跳过（mask[i] 保持 False）
            continue
        mol = Chem.MolFromSmiles(smi)
        if mol is None:  # RDKit 解析失败
            continue
        # 计算描述符，写入 features[i]
        # ...
        mask[i] = True  # 标记成功

    return features, mask
```

---

#### 3. 计算逻辑：四种描述符的具体实现

##### 3.1 Morgan ECFP4（`morgan.py`）

**算法原理**：以每个原子为中心，向外扩展 `radius=2` 跳，将局部子结构哈希到 1024 位二进制指纹。

**实现细节**：

```python
# yonod/descriptors/morgan.py:41-48
fp = AllChem.GetMorganFingerprintAsBitVect(
    mol, radius=2, nBits=1024
)
arr = np.zeros(1024, dtype=np.float32)
ConvertToNumpyArray(fp, arr)  # RDKit BitVect → NumPy 0/1 浮点数组
features[i] = arr
```

**性能特点**：
- **计算速度**：毫秒级（RDKit C++ 后端）
- **内存占用**：1024 × 4 字节 = 4 KB/分子（稀疏，但以密集矩阵存储）
- **无外部依赖**：仅需 RDKit

---

##### 3.2 ATMOMACCS（`atmomaccs.py`）

**算法原理**：检测 166 种预定义的化学结构模式（官能团、环系统、拓扑特征）。

**实现细节**：

```python
# yonod/descriptors/atmomaccs.py:51-54
fp = GetMACCSKeysFingerprint(mol)  # 返回 167 位（bit 0 占位）
full = np.zeros(167, dtype=np.float32)
ConvertToNumpyArray(fp, full)
features[i] = full[1:]  # 去除 bit 0 → 166 位
```

**设计决策**：
- **为什么去除 bit 0？** RDKit 的 MACCS bit 0 是保留位（永远为 0），上游论文 [DOI:10.1063/5.0308548](https://doi.org/10.1063/5.0308548) 也去除了它
- **为什么不使用 `化学描述符相关项目/ATMOMACCS/` 包？** 该包需要从磁盘读取 `smiles.txt`，不适配内存批处理接口

---

##### 3.3 FISD（`fisd.py`）

**算法原理**：双路图神经网络（GCN_MSE + GCN_COS）→ TwoInOne MLP 融合 → 50 维嵌入。

**网络架构**：

```
SMILES → RDKit Mol → 分子图 (45-d 原子特征, 邻接矩阵)
  ├─ GCN_MSE: 5层GCN (512隐层) → global_pool → MLP → 50-d
  ├─ GCN_COS: 5层GCN (512隐层) → global_pool → MLP → 50-d
  └─ concat(MSE, COS) → TwoInOne MLP (4层, 512隐层) → 50-d
```

**原子特征化**（`fisd.py:105-131`）：45 维 = 以下 one-hot/归一化特征拼接

| 特征组 | 维度 | 编码方式 |
|---|---|---|
| 元素符号 | 9 | one-hot (C, N, O, S, F, P, Cl, Br, Unknown) |
| 度数 | 6 | one-hot (0, 1, 2, 3, 4, MoreThanFour) |
| 形式电荷 | 8 | one-hot (-3~+3, Extreme) |
| 杂化方式 | 7 | one-hot (S, SP, SP2, SP3, SP3D, SP3D2, OTHER) |
| 成环性 | 1 | 布尔值 |
| 芳香性 | 1 | 布尔值 |
| 原子质量 | 1 | 归一化：(mass - 10.812) / 116.092 |
| 范德华半径 | 1 | 归一化：(Rvdw - 1.5) / 0.6 |
| 共价半径 | 1 | 归一化：(Rcov - 0.64) / 0.76 |
| 手性 | 4 | one-hot (CHI_UNSPECIFIED, CW, CCW, OTHER) |
| 氢原子数 | 6 | one-hot (0~4, MoreThanFour) |

**关键实现逻辑**：

```python
# fisd.py:134-150 分子 → PyG Data 对象
def _smiles_to_graph(smi: str) -> Optional[Data]:
    mol = Chem.MolFromSmiles(smi)
    if mol is None or mol.GetNumAtoms() == 0:
        return None
    # 提取 45-d 原子特征矩阵 X
    X = np.zeros((n_atoms, 45), dtype=np.float32)
    for atom in mol.GetAtoms():
        X[atom.GetIdx()] = _atom_features(atom)
    # 构建边索引（双向无向边）
    rows, cols = np.nonzero(GetAdjacencyMatrix(mol))
    edge_index = torch.from_numpy(np.stack([rows, cols]))
    return Data(x=torch.from_numpy(X), edge_index=edge_index)

# fisd.py:218-224 批量推理
def _encode_batch(self, graphs: List[Data]) -> np.ndarray:
    batch = Batch.from_data_list(graphs).to(self.device)
    mse_out = self._mse(batch.x, batch.edge_index, batch.batch)  # (B, 50)
    cos_out = self._cos(batch.x, batch.edge_index, batch.batch)  # (B, 50)
    combined = torch.cat([mse_out, cos_out], dim=1)             # (B, 100)
    fisd = self._two(combined)                                   # (B, 50)
    return fisd.detach().cpu().numpy()
```

**权重懒加载**（`fisd.py:186-213`）：
- 首次调用 `featurize()` 时触发 `_load()`
- 从 `WEIGHTS/FISD/` 加载 3 个 `.pth` 文件（总计 ~19 MB）
- 自动检测 CUDA；若不可用降级到 CPU

**批处理策略**：
- `batch_size=32`（可配置）
- 无效 SMILES 不进入 GPU，避免浪费显存

---

##### 3.4 MolMetaLM（`molmetalm.py`）

**算法原理**：将 SMILES 作为"分子语言"输入预训练 Transformer（Llama 架构），取 `last_hidden_state` 的 attention-masked mean-pool 作为 768 维嵌入。

**网络流程**：

```
SMILES → Tokenizer → [input_ids, attention_mask] → Llama Encoder
  → last_hidden_state (B, seq_len, 768)
  → attention-masked mean-pool → (B, 768)
```

**实现细节**：

```python
# molmetalm.py:100-128
def _encode_batch(self, batch_smiles: List[str]) -> np.ndarray:
    tokens = self._tokenizer(
        batch_smiles,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    ).to(self.device)

    # Llama 不接受 token_type_ids，仅传递支持的参数
    model_inputs = {k: v for k, v in tokens.items() if k in {"input_ids", "attention_mask"}}

    with torch.no_grad():
        out = self._model(**model_inputs)
    hidden = out.last_hidden_state  # (B, seq_len, 768)

    # 加权平均池化（忽略 padding 位置）
    mask = tokens["attention_mask"].unsqueeze(-1)  # (B, seq_len, 1)
    summed = (hidden * mask).sum(dim=1)            # (B, 768)
    pooled = summed / mask.sum(dim=1).clamp(min=1) # (B, 768)
    return pooled.cpu().numpy().astype(np.float32)
```

**错误恢复机制**：
- **CUDA OOM**：自动减半 `batch_size` 并重试（`molmetalm.py:159-166`）
- **非 OOM 失败**：打印首次错误堆栈，后续同类错误静默跳过

**权重加载兼容性**（`molmetalm.py:62-96`）：
- 优先尝试 `AutoModel.from_pretrained()`（无 LM head）
- 若失败则降级到 `AutoModelForCausalLM`，手动提取 encoder 部分
- 自动检测 `hidden_size` 并覆盖 `output_dim`（支持非标准检查点）

---

#### 4. 调用关系与数据流

**完整调用链路**（以酰胺缩合数据集为例）：

```
┌─────────────────────────────────────────────────────────────────┐
│ 1. 主入口：yonod.py                                              │
│    └─ 读取 CSV，调用 csv_loader.load_csv_with_roles()           │
│       └─ 自动探测 SMILES 列和标签列                             │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ 2. 特征构建：feature_builder.build_universal_features()         │
│    └─ 懒加载描述符实例：_get_descriptor(desc_name)              │
│    └─ 逐列调用 descriptor.featurize(smiles_list)                │
│       ├─ 第 1 列（如 sub_1_smiles）→ features_1, mask_1         │
│       ├─ 第 2 列（如 sub_2_smiles）→ features_2, mask_2         │
│       └─ ...                                                     │
│    └─ 横向拼接多列：np.concatenate([f1, f2, ...], axis=1)       │
│    └─ 合并 mask：row_mask = mask_1 & mask_2 & ...               │
│    └─ 过滤失败行：X_smiles = X_full[row_mask]                   │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ 3. 试剂缓存优化（酰胺缩合专用路径）                              │
│    └─ reagent_cache.build_reagent_feat_cache()                  │
│       ├─ 提取 4 列试剂的唯一 SMILES（~58 种）                   │
│       ├─ 批量调用 descriptor.featurize(unique_reagents)         │
│       ├─ 构建字典：{smiles → ndarray(d,)}                       │
│       └─ 保存到 cache/reagent_feats_<desc>.pkl                  │
│    └─ reaction_featurizer.ReactionFeaturizer.transform()        │
│       ├─ 底物：直接调用 featurize(sub_1), featurize(sub_2)     │
│       ├─ 试剂：从缓存字典 O(1) 查找                             │
│       └─ 拼接：[sub1, sub2, act, add, base, solv] → 6*d 维      │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ 4. 交叉验证与模型训练（yonod.py:_cv_with_numeric）              │
│    └─ KFold(n_splits=5).split(X_smiles)                         │
│       └─ 每折：                                                  │
│          ├─ X_train, X_test = X[train_idx], X[test_idx]         │
│          ├─ model.fit(X_train, y_train)                         │
│          └─ y_pred = model.predict(X_test)                      │
└─────────────────────────────────────────────────────────────────┘
```

**数据形状变换示例**（酰胺缩合，Morgan 1024-d）：

```
CSV (47015 行 × 9 列)
  ↓ featurize(sub_1_smiles)  → (47015, 1024), mask_1
  ↓ featurize(sub_2_smiles)  → (47015, 1024), mask_2
  ↓ 试剂缓存查找 × 4         → (47015, 1024) × 4
  ↓ 横向拼接                 → (47015, 6144)
  ↓ 过滤失败行（mask_1 & mask_2）→ (46892, 6144)  # 丢弃 123 行解析失败
  ↓ KFold 切分               → train (37513, 6144) / test (9379, 6144)
  ↓ model.fit()              → 训练 XGBoost / RF / SVM / AutoGluon
```

**描述符间调用隔离**：
- 4 种描述符**完全独立**，无交叉依赖
- 主循环并行调用（`for desc in descriptors: for model in models:`）
- FISD 和 MolMetaLM 的 PyTorch 模块仅在对应描述符被选中时加载

---

#### 5. 性能优化机制

##### 5.1 试剂缓存（`reagent_cache.py`）

**问题**：酰胺缩合数据集有 47015 行，但试剂组合仅 ~300 种，底物组合 ~1500 种。

**优化策略**：
1. 扫描 4 列试剂，收集唯一 SMILES（含 `(无)` → `None` 映射）
2. 批量计算 58 个唯一试剂的描述符（而非 47015 × 4 次）
3. 保存到 `cache/reagent_feats_{desc}.pkl`
4. 每行试剂查找改为字典 O(1) 访问

**加速效果**：
- **FISD**：2 小时 → 10 分钟（GCN 推理主导）
- **MolMetaLM**：3 小时 → 15 分钟（Transformer 推理主导）
- **Morgan / MACCS**：无明显差异（本身已是毫秒级）

##### 5.2 懒加载（Lazy Loading）

**FISD 和 MolMetaLM 的模型权重仅在首次 `featurize()` 时加载**：

```python
# fisd.py:186-213
def _load(self) -> None:
    if self._mse is not None:  # 已加载，跳过
        return
    self.device = "cuda" if torch.cuda.is_available() else "cpu"
    # 加载 3 个 .pth 检查点...
    self._mse = mse.to(self.device).eval()
    self._cos = cos.to(self.device).eval()
    self._two = two.to(self.device).eval()
```

**好处**：
1. `import yonod.descriptors.fisd` 不触发 PyTorch 加载，模块导入速度快
2. 用户若只选择 Morgan + MACCS，FISD 权重永不加载（节省 ~500 MB 内存）

##### 5.3 批处理与 GPU 利用

| 描述符 | batch_size | 设备选择 | OOM 恢复 |
|---|---|---|---|
| Morgan | 单分子循环 | CPU only | N/A |
| MACCS | 单分子循环 | CPU only | N/A |
| FISD | 32（可配置）| 自动检测 CUDA | ✗ |
| MolMetaLM | 32（可配置）| 自动检测 CUDA | ✓ 自动减半 batch_size |

---

#### 6. 扩展新描述符的标准流程

若要添加新描述符（如 SOAP、MBTR、ChemBERTa），需完成以下 4 步：

**Step 1：实现 `BaseDescriptor` 子类**

```python
# yonod/descriptors/my_descriptor.py
from .base import BaseDescriptor

class MyDescriptor(BaseDescriptor):
    name = "my_desc"
    output_dim = 128  # 根据实际维度设置

    def featurize(self, smiles_list):
        n = len(smiles_list)
        features, mask = self._empty_outputs(n, dtype=np.float32)
        for i, smi in enumerate(smiles_list):
            if not smi:
                continue
            # 计算描述符...
            features[i] = ...
            mask[i] = True
        return features, mask
```

**Step 2：注册到 `DESCRIPTOR_REGISTRY`**

```python
# yonod/evaluate.py
from .descriptors.my_descriptor import MyDescriptor

DESCRIPTOR_REGISTRY = {
    "morgan": MorganDescriptor,
    "atmomaccs": ATMOMACCSDescriptor,
    "fisd": FISDDescriptor,
    "molmetalm": MolMetaLMDescriptor,
    "my_desc": MyDescriptor,  # 新增
}
```

**Step 3：更新 CLI 参数（可选）**

```python
# yonod.py
_DESCRIPTOR_NAMES = ["morgan", "maccs", "fisd", "molmetalm", "my_desc"]
```

**Step 4：运行测试**

```bash
python yonod.py --csv dataset/test.csv --smiles-cols smiles --label-col y \
    --descriptors my_desc --models rf
```

---

#### 7. 已知设计权衡

| 设计决策 | 优点 | 代价 |
|---|---|---|
| **失败行填充零向量** | 保持矩阵对齐，mask 机制统一过滤 | 浪费内存（无效行占空间） |
| **float32 而非 uint8** | XGBoost / sklearn 通用接受 | Morgan 指纹内存翻倍（1024 字节 → 4096 字节） |
| **懒加载权重** | 按需加载，节省内存 | 首次调用延迟高（FISD ~5s，MolMetaLM ~20s） |
| **试剂缓存持久化** | 避免重复计算 | 首次运行需构建缓存，pkl 文件占磁盘 |
| **批处理 batch_size=32** | GPU 利用率高 | 大分子可能触发 OOM |

---

**总结**：YONOD 的描述符模块通过**抽象基类 + 注册表 + 懒加载**实现了高度模块化和可扩展性，同时通过**试剂缓存 + 批处理 + GPU 加速**优化了实际运行性能。4 种描述符覆盖了从经典指纹（Morgan/MACCS）到深度学习（FISD/MolMetaLM）的完整技术谱系，为下游模型性能对比提供了多样化的特征表示。

---

## 二次开发备忘

> 本节记录用户的二次开发需求和改造预分析，便于后续切换至 project-planner-cn 时保留上下文。

---

