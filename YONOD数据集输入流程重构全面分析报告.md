# YONOD 数据集输入与合法性检验流程重构全面分析报告

> **报告生成时间**：2026-06-28
> **分析对象**：YONOD 数据集输入流程重构需求文档
> **当前实现版本**：向导模式 v5 (支持列角色分类)

---

## 目录

1. [可行性分析报告](#1-可行性分析报告)
2. [步骤分析报告](#2-步骤分析报告)
3. [建议和疑问](#3-建议和疑问)
4. [脚本交互逻辑流程图](#4-脚本交互逻辑流程图)

---

## 1. 可行性分析报告

### 1.1 技术可行性评估

#### ✅ 高可行性部分

**1. 列角色声明机制（步骤 2）**
- **现状**：当前 `yonod.py` 已实现三分类模式（`--reactant-cols`、`--product-cols`、`--other-cols`），`csv_loader.py` 已支持角色映射
- **重构点**：将现有 CLI 参数改造为交互式向导，逐步引导用户声明
- **技术难度**：⭐⭐☆☆☆（中低难度）
- **实现方案**：
  - 复用 `_ask_single_col()` 和 `_ask_multi_cols()` 函数
  - 新增 `_ask_col_name()` 函数，强制用户输入英文列名（正则校验 `^[A-Za-z0-9_]+$`）
  - 维护一个 `remaining_cols` 列表，每次声明后动态移除已选列

**2. SMILES 合法性检验（步骤 2.2-2.5）**
- **现状**：`yonod.py:726-766` 已实现 `_validate_smiles()` 函数，支持多组分 SMILES 验证
- **重构点**：改为**累积错误日志模式**，而不是遇到第一个错误就退出
- **技术难度**：⭐⭐☆☆☆（中低难度）
- **实现方案**：
  ```python
  invalid_rows = []  # 存储 (行号, 列名, 非法值) 三元组
  for col in smiles_cols:
      for idx, val in enumerate(df[col]):
          if not is_valid_smiles(val):
              invalid_rows.append((idx+2, col, val))  # idx+2 是显示行号
  ```

**3. 列映射文件（步骤 1.2 和 3.2）**
- **现状**：无现成实现，但格式简单（CSV 三列表）
- **技术难度**：⭐☆☆☆☆（低难度）
- **实现方案**：
  ```python
  mapping_df = pd.DataFrame({
      'origin_name': [...],
      'role': [...],
      'name': [...]
  })
  mapping_df.to_csv('列映射说明.csv', index=False, encoding='utf-8-sig')
  ```

**4. 规范数据集生成（步骤 3.3）**
- **现状**：`descriptors/base.py:split_multi_smiles()` 已实现 SMILES 拆分逻辑
- **重构点**：需新增"动态列扩展"逻辑（根据最大组分数创建 `reactant-1`, `reactant-2`, ...）
- **技术难度**：⭐⭐⭐☆☆（中等难度）
- **实现方案**：
  ```python
  # 1. 统计每个角色的最大组分数
  max_reactants = max(len(split_multi_smiles(s)) for s in df['reactant_col'])

  # 2. 动态创建列名
  reactant_cols = [f'reactant-{i+1}' for i in range(max_reactants)]

  # 3. 逐行填充（允许部分列为空）
  for idx, row in df.iterrows():
      smiles_list = split_multi_smiles(row['reactant_col'])
      for i, smi in enumerate(smiles_list):
          normalized_df.at[idx, f'reactant-{i+1}'] = smi
  ```

#### ⚠️ 中等风险部分

**1. 产物列单 SMILES 校验（步骤 2.4）**
- **挑战**：需额外检测每个单元格是否包含多个 SMILES（点号、分号、逗号分隔）
- **风险点**：离子对化合物（如 `[Na+].[Cl-]`）会被点号拆分为 2 个组分，可能误判
- **缓解方案**：
  - 使用 RDKit 解析每个组分，判断是否为离子（`mol.GetFormalCharge() != 0`）
  - 若所有组分都是离子且总电荷为 0，则视为单一盐型产物
  - 否则拒绝输入并提示用户

**2. 非法输入排除报告（步骤 3.1）**
- **挑战**：Markdown 表格格式化（需处理含特殊字符的 SMILES）
- **风险点**：SMILES 中可能含有管道符 `|`，会破坏 Markdown 表格
- **缓解方案**：
  ```python
  def escape_md_table(s: str) -> str:
      return s.replace('|', '\\|').replace('\n', ' ')
  ```

**3. 描述符嵌入配置（步骤 4）**
- **挑战**：需设计一个交互式界面，让用户自定义列拼接顺序
- **风险点**：用户可能输入错误的列序号，导致嵌入逻辑混乱
- **缓解方案**：
  - 每次配置前展示当前选择，用户确认后才生效
  - 提供"恢复默认"选项（`reactant-others-product`）

#### ❌ 高风险/不可行部分

**1. 列映射文件自动加载逻辑（步骤 1.2）**
- **问题**：需求中说"如果输入列映射文件，则直接进行数据合法性检验"
- **矛盾点**：
  - 合法性检验依赖 RDKit 解析每个单元格的 SMILES
  - 列映射文件仅存储"列名 → 角色"关系，不存储具体数据
  - 因此**仍需读取原始数据集进行检验**，无法"跳过"交互步骤
- **建议**：将列映射文件定位为**历史配置复用**，而非跳过检验的捷径
  - 用户输入列映射文件 → 自动填充步骤 2 的列角色（但仍需确认）
  - 仍然执行全量 SMILES 和数值合法性检验

---

### 1.2 与现有代码的兼容性

#### 需要修改的现有模块

| 模块 | 修改范围 | 影响评估 |
|------|---------|---------|
| `yonod.py:wizard()` | 完全重写步骤 2-3，新增非法值累积逻辑 | ⚠️ 高影响（核心交互流程） |
| `csv_loader.py:load_csv_with_roles()` | 新增参数 `custom_col_names: Dict[str, str]` 支持用户自定义列名 | ✅ 低影响（向后兼容） |
| `universal/feature_builder.py` | 新增参数 `descriptor_config` 支持自定义嵌入顺序 | ⚠️ 中影响（需修改 DRFP 等描述符调用逻辑） |
| `descriptors/drfp_desc.py` | 新增 `featurize_with_roles()` 方法，支持将 `other` 列并入 `reactant` | ✅ 低影响（扩展接口） |

#### 不需要修改的模块

- **模型层** (`models/*.py`)：完全不受影响，输入仍是特征矩阵 `(n, d)`
- **评估与报告** (`evaluate.py`, `plot.py`, `universal/report.py`)：完全不受影响
- **描述符核心逻辑** (`descriptors/morgan.py` 等)：仅需修改调用参数，不改内部实现

---

### 1.3 潜在风险识别

#### 🔴 高风险

**R1. 规范数据集与原始数据集不一致导致可追溯性丧失**
- **场景**：用户在步骤 3 生成规范数据集后，发现某行数据有误，但已无法对应回原始数据集的行号
- **影响**：用户无法定位并修复原始数据源
- **缓解措施**：
  - 在规范数据集中新增 `_original_row_index` 隐藏列（建模时自动排除）
  - 在非法输入报告中标注"原始行号"和"规范数据集行号"的映射

**R2. 用户误操作：将同一列同时声明为多个角色**
- **场景**：用户在步骤 2.3 将某列声明为 `reactant`，又在步骤 2.5 声明为 `solvent`
- **影响**：规范数据集中该列数据重复，导致模型过拟合
- **缓解措施**：
  - 维护 `used_cols` 集合，每次声明前检查是否重复
  - 若检测到重复，拒绝输入并提示用户

#### 🟡 中风险

**R3. 描述符配置界面过于复杂，用户理解成本高**
- **场景**：步骤 4.4 要求用户输入列序号并理解"输入顺序即拼接顺序"
- **影响**：用户可能输入错误配置，导致建模失败
- **缓解措施**：
  - 提供"使用推荐配置"快捷选项（默认 `reactant-others-product`）
  - 每次配置后立即展示拼接后的示例（如 `reactant-1 → reactant-2 → solvent → product`）

**R4. 非法值报告文件过大**
- **场景**：数据集有 50000 行，其中 10000 行存在非法值
- **影响**：Markdown 文件过大（>100MB），用户浏览器无法打开
- **缓解措施**：
  - 限制报告最多展示前 1000 行非法数据
  - 超出部分在报告末尾注明："共检测到 10000 行非法数据，此处仅展示前 1000 行，完整列表见 `invalid_rows.csv`"

#### 🟢 低风险

**R5. 列名冲突**
- **场景**：用户将多个列都命名为 `reactant`
- **影响**：规范数据集列名重复，Pandas 会自动重命名为 `reactant.1`, `reactant.2`
- **缓解措施**：
  - 在步骤 2 中维护 `used_names` 集合，禁止重复命名
  - 或自动追加后缀（`reactant-1`, `reactant-2`）并告知用户

---

### 1.4 开发工作量估算

| 模块 | 预估工时（人天） | 依赖关系 |
|------|----------------|---------|
| 1. 重构 `wizard()` 交互流程 | 3-4 | 无 |
| 2. 非法值累积与报告生成 | 2-3 | 依赖模块 1 |
| 3. 规范数据集生成逻辑 | 2-3 | 依赖模块 1、2 |
| 4. 列映射文件读写 | 0.5 | 无 |
| 5. 描述符配置界面 | 2-3 | 依赖模块 3 |
| 6. 产物列单 SMILES 校验 | 1 | 无 |
| 7. 集成测试与边界情况处理 | 2-3 | 依赖所有模块 |
| **总计** | **12-17 人天** | — |

**关键路径**：模块 1 → 模块 2 → 模块 3 → 模块 5 → 模块 7

**并行开发机会**：模块 4、6 可与其他模块并行开发

---

### 1.5 技术选型建议

**1. 交互界面库**
- **推荐**：继续使用内置 `input()` 函数（与现有代码一致）
- **原因**：
  - 无需额外依赖
  - 用户已熟悉当前交互方式
  - 足以支持所需功能

**2. 列映射文件格式**
- **推荐**：CSV 格式（如需求文档所示）
- **原因**：
  - Pandas 原生支持
  - 用户可用 Excel 直接编辑
  - 人类可读性强

**3. 规范数据集格式**
- **推荐**：CSV 格式（UTF-8 with BOM）
- **原因**：
  - 与现有 `load_csv_with_roles()` 兼容
  - Excel 可正确打开（BOM 防止中文乱码）

**4. 非法值报告格式**
- **推荐**：Markdown 格式
- **原因**：
  - 表格可读性强
  - 支持加粗（标注非法值）
  - GitHub/VS Code 原生预览

---

## 1.6 列角色与合法性检验机制总览

下表汇总了重构方案中定义的所有列角色及其对应的合法性检验规则：

| 列角色 | 英文标识 | 用户输入数量 | 默认名称 | 合法性检验规则 | 空值处理 | 非法值处理 |
|--------|----------|--------------|----------|----------------|----------|------------|
| **标签列** | `label` | 仅 1 列 | `yield` | 必须为**数值类型**（可转换为 `float`） | ❌ **不允许空值** | 标记该行为非法，生成规范数据集时跳过 |
| **反应物列** | `reactant` | 可多列 | `reactant` | 必须为**有效 SMILES**（RDKit 可解析） | ❌ **不允许空值** | 标记该行为非法，生成规范数据集时跳过 |
| **产物列** | `product` | 仅 1 列 | `product` | 必须为**单一 SMILES**（每个单元格仅含 1 个独立分子，离子对除外） | ❌ **不允许空值** | 标记该行为非法，生成规范数据集时跳过 |
| **其他组分列** | `others` | 可多列（循环添加） | 原列名或用户指定 | 必须为**有效 SMILES**（RDKit 可解析） | ✅ **允许空值** | 标记该行为非法，生成规范数据集时跳过 |
| **条件数值列** | `condition` | 可多列（循环添加） | 原列名或用户指定 | 必须为**数值类型**（可转换为 `float`） | ✅ **允许空值** | 标记该行为非法，生成规范数据集时跳过 |

### 检验规则详解

#### 1. SMILES 合法性检验（适用于 `reactant`、`product`、`others`）

```python
from rdkit import Chem

def is_valid_smiles(smiles: str) -> bool:
    """判断 SMILES 是否合法（RDKit 可解析）"""
    if not smiles or pd.isna(smiles):
        return False
    try:
        mol = Chem.MolFromSmiles(str(smiles))
        return mol is not None
    except:
        return False
```

#### 2. 数值合法性检验（适用于 `label`、`condition`）

```python
def is_valid_numeric(value) -> bool:
    """判断值是否为有效数值"""
    if pd.isna(value):
        return False  # label 列：空值非法；condition 列：空值合法
    try:
        float(value)
        return True
    except (ValueError, TypeError):
        return False
```

#### 3. 产物列单 SMILES 约束（仅适用于 `product`）

```python
def is_single_smiles(smiles: str) -> bool:
    """
    判断是否为单一 SMILES（产物列专用）
    - 允许离子对（如 [Na+].[Cl-]，总电荷为 0）
    - 不允许多组分混合物（如 CCO.CC）
    """
    components = split_multi_smiles(smiles)  # 按 . ; , 拆分
    if len(components) == 1:
        return True

    # 检查是否为离子对
    mols = [Chem.MolFromSmiles(c) for c in components]
    if any(m is None for m in mols):
        return False

    charges = [mol.GetFormalCharge() for mol in mols]
    return all(c != 0 for c in charges) and sum(charges) == 0
```

### 非法值处理策略对比

| 处理方式 | 原流程 | 重构后流程 |
|----------|--------|------------|
| 检测到非法值时 | 立即退出程序，要求用户修复 | **累积记录**，继续检验后续列 |
| 非法值报告 | 无 | 生成 Markdown 报告，逐行展示 |
| 规范数据集生成 | 非法值导致无法生成 | **跳过非法行**，仅保留合法行 |
| 用户确认流程 | 无 | 生成报告后要求用户浏览确认 |

### 列角色与描述符嵌入关系

| 列角色 | 参与描述符嵌入 | 嵌入方式说明 |
|--------|----------------|--------------|
| `reactant` | ✅ 所有描述符 | 横向拼接 / 逐点加和 / DRFP 反应物 |
| `product` | ✅ 所有描述符 | 横向拼接 / 逐点加和 / DRFP 产物 |
| `others` | ✅ 所有描述符 | 横向拼接 / 逐点加和 / 可选加入 DRFP 反应物侧 |
| `condition` | ❌ 不参与 | 作为辅助特征直接拼接到特征向量末尾 |
| `label` | ❌ 不参与 | 作为建模目标 (y 值) |

---

## 2. 步骤分析报告

### 步骤 1：指定初始数据集、列映射文件、项目名称和文件夹

#### 实现要点

**1.1 初始数据集输入**
- 复用现有 `yonod.py:777-785` 的文件路径输入逻辑
- 支持拖拽文件（自动去除引号）
- 多编码尝试（UTF-8-sig → GBK）

**1.2 列映射文件输入（可选）**
```python
def _load_column_mapping(csv_path: Path) -> Optional[Dict[str, Tuple[str, str]]]:
    """加载列映射文件，返回 {origin_name: (role, name)} 字典。"""
    if not csv_path.exists():
        return None
    df = pd.read_csv(csv_path, encoding='utf-8-sig')
    required_cols = {'origin_name', 'role', 'name'}
    if not required_cols.issubset(df.columns):
        print(f"  [错误] 列映射文件缺少必需列：{required_cols - set(df.columns)}")
        return None
    return {
        row['origin_name']: (row['role'], row['name'])
        for _, row in df.iterrows()
    }
```

**1.3 项目名称与文件夹**
- 复用现有 `yonod.py:923-940` 的任务名称和输出目录逻辑
- **新增**：在项目文件夹下创建子目录结构
  ```
  project_folder/
  ├── 非法输入排除报告.md        # 步骤 3.1 生成
  ├── 列映射说明.csv              # 步骤 3.2 生成
  ├── 规范数据集.csv              # 步骤 3.3 生成
  └── 建模报告/                  # 原有输出目录
      ├── metrics_summary.csv
      └── report.html
  ```

#### 关键边界情况

- **BC1.1**：列映射文件列名与当前数据集不匹配 → 提示用户但继续执行（视为历史配置参考）
- **BC1.2**：项目文件夹已存在 → 询问用户是否覆盖或追加时间戳（避免数据丢失）

---

### 步骤 2：逐列声明列角色和名称

#### 2.1 读取初始数据集并筛选备选列

```python
def _select_candidate_cols(columns: List[str]) -> List[str]:
    """让用户从所有列中选择需要的列。"""
    print("\n所有可用列：")
    _show_columns(columns)

    raw = _ask_optional("请选择需要的列（空格分隔，留空=全选）")
    if not raw:
        return columns

    selected = []
    for token in raw.split():
        col = _resolve_one(token, columns)
        if col:
            selected.append(col)
        else:
            print(f"  [警告] 无法识别 '{token}'，已跳过")

    return selected if selected else columns
```

#### 2.2 声明标签列

**核心逻辑**：
```python
def _declare_label_col(candidate_cols: List[str]) -> Tuple[str, str, List[int]]:
    """
    返回：(原列名, 用户指定的新列名, 非法值行号列表)
    """
    _show_columns(candidate_cols)
    origin_col = _ask_single_col("请选择标签列", candidate_cols, required=True)

    # 询问用户自定义列名
    while True:
        new_name = _ask_optional("请输入标签列名称", default="yield")
        if re.match(r'^[A-Za-z0-9_]+$', new_name):
            break
        print("  [错误] 列名只能包含英文字母、数字、下划线")

    # 合法性检验（累积模式，不中断）
    invalid_rows = []
    series = df[origin_col]
    numeric = pd.to_numeric(series, errors='coerce')
    bad_mask = numeric.isna() & series.notna()
    invalid_rows = list(bad_mask[bad_mask].index + 2)  # +2 转为显示行号

    if invalid_rows:
        print(f"  [警告] 检测到 {len(invalid_rows)} 行非数值数据，将在生成规范数据集时排除")

    return origin_col, new_name, invalid_rows
```

#### 2.3 声明反应物 SMILES 列

**特殊处理**：允许多列，每列可独立命名或统一命名为 `reactant`

```python
def _declare_reactant_cols(candidate_cols: List[str], df: pd.DataFrame) -> Dict[str, Any]:
    """
    返回：{
        'columns': [(原列名, 新列名), ...],
        'invalid_rows': [行号, ...]
    }
    """
    _show_columns(candidate_cols)
    origin_cols = _ask_multi_cols("请选择反应物列（可多列）", candidate_cols, required=True)

    # 询问是否统一命名
    if len(origin_cols) == 1:
        new_name = _ask_optional("请输入列名称", default="reactant")
        col_mapping = [(origin_cols[0], new_name)]
    else:
        choice = _ask_optional("是否统一命名为 'reactant'？[Y/n]", default="Y")
        if choice.lower() in ('y', 'yes', ''):
            col_mapping = [(col, "reactant") for col in origin_cols]
        else:
            col_mapping = []
            for col in origin_cols:
                name = _ask_optional(f"  请为列 '{col}' 指定名称", default="reactant")
                col_mapping.append((col, name))

    # SMILES 合法性检验（累积模式）
    invalid_rows = set()
    for origin_col, _ in col_mapping:
        for idx, val in enumerate(df[origin_col]):
            if pd.isna(val):  # 反应物列不允许空值
                invalid_rows.add(idx + 2)
            elif not _is_valid_smiles(str(val)):
                invalid_rows.add(idx + 2)

    return {
        'columns': col_mapping,
        'invalid_rows': sorted(invalid_rows)
    }
```

#### 2.4 声明产物 SMILES 列

**特殊约束**：每个单元格必须恰好包含 1 个 SMILES（不计离子对）

```python
def _validate_single_smiles(smiles: str) -> Tuple[bool, str]:
    """
    检查是否为单个 SMILES（允许离子对如 [Na+].[Cl-]）。

    返回：(是否有效, 错误消息)
    """
    from rdkit import Chem
    components = split_multi_smiles(smiles)

    if len(components) == 1:
        return True, ""

    # 检查是否为离子对
    mols = [Chem.MolFromSmiles(c) for c in components]
    if any(m is None for m in mols):
        return False, f"包含无效组分"

    charges = [mol.GetFormalCharge() for mol in mols]
    if all(c != 0 for c in charges) and sum(charges) == 0:
        return True, ""  # 离子对，视为单一产物

    return False, f"包含 {len(components)} 个非离子组分，产物列只能有 1 个 SMILES"
```

#### 2.5 声明其他组分 SMILES 列

**交互流程**：循环询问，直到用户选择"不再添加"

```python
def _declare_other_cols(candidate_cols: List[str], df: pd.DataFrame) -> Dict[str, Any]:
    """
    返回：{
        'columns': [(原列名, 新列名, 角色标签), ...],
        'invalid_rows': [行号, ...]
    }
    """
    col_mapping = []
    invalid_rows = set()

    while True:
        if not candidate_cols:
            print("  [提示] 无剩余列可选")
            break

        _show_columns(candidate_cols)
        choice = _ask_optional("是否添加其他组分列（溶剂/催化剂/碱等）？[y/N]", default="N")
        if choice.lower() not in ('y', 'yes'):
            break

        origin_col = _ask_single_col("请选择列", candidate_cols, required=True)

        # 推荐名称（根据原列名猜测）
        suggested_name = _suggest_name(origin_col)  # 如 'solvent_1' → 'solvent'
        new_name = _ask_optional(f"请输入列名称（常用：solvent/catalyst/base/reagent）",
                                 default=suggested_name)

        col_mapping.append((origin_col, new_name))
        candidate_cols.remove(origin_col)

        # SMILES 合法性检验（允许空值）
        for idx, val in enumerate(df[origin_col]):
            if pd.notna(val) and not _is_valid_smiles(str(val)):
                invalid_rows.add(idx + 2)

    return {
        'columns': col_mapping,
        'invalid_rows': sorted(invalid_rows)
    }
```

#### 2.6 声明条件数值列

**与 2.5 类似**，但合法性检验改为数值检查

```python
# 合法性检验逻辑（允许空值）
for idx, val in enumerate(df[origin_col]):
    if pd.notna(val):
        try:
            float(val)
        except (ValueError, TypeError):
            invalid_rows.add(idx + 2)
```

---

### 步骤 3：生成规范数据集与非法输入排除报告

#### 3.1 生成非法输入排除报告

**Markdown 模板**：

```markdown
# 非法输入排除报告

**生成时间**：2026-06-28 14:30:15
**原始数据集**：dataset/example.csv
**总行数**：5000
**排除行数**：127

---

## 排除原因统计

| 步骤 | 原因 | 行数 |
|------|------|------|
| 2.2 标签列检验 | 非数值值 | 45 |
| 2.3 反应物列检验 | 非法 SMILES | 32 |
| 2.3 反应物列检验 | 空值 | 18 |
| 2.4 产物列检验 | 多组分 SMILES | 12 |
| 2.5 其他组分列检验 | 非法 SMILES | 20 |

---

## 详细数据

### 第 5 行（来源：步骤 2.3 反应物列检验）

| sub_1 | sub_2 | yield |
|-------|-------|-------|
| **CCX** | CCO | 0.85 |

**非法值**：`sub_1` 列的 `CCX`（无效 SMILES）

---

### 第 17 行（来源：步骤 2.2 标签列检验）

| sub_1 | sub_2 | yield |
|-------|-------|-------|
| CCO | CC | **high** |

**非法值**：`yield` 列的 `high`（非数值）

---

（后续省略，共展示前 1000 行）
```

**实现关键点**：

```python
def _generate_invalid_report(invalid_data: List[Dict], output_path: Path) -> None:
    """
    invalid_data 格式：[
        {
            'row_num': 5,
            'step': '2.3 反应物列检验',
            'reason': '非法 SMILES',
            'invalid_cols': ['sub_1'],
            'row_data': {'sub_1': 'CCX', 'sub_2': 'CCO', 'yield': 0.85}
        },
        ...
    ]
    """
    if not invalid_data:
        return  # 无非法数据，不生成报告

    with open(output_path, 'w', encoding='utf-8') as f:
        # 写入头部
        f.write(f"# 非法输入排除报告\n\n")
        f.write(f"**生成时间**：{datetime.now()}\n")
        f.write(f"**排除行数**：{len(invalid_data)}\n\n")

        # 统计表
        f.write("## 排除原因统计\n\n")
        # ... 分组统计逻辑

        # 详细数据（限制前 1000 行）
        f.write("## 详细数据\n\n")
        for i, item in enumerate(invalid_data[:1000]):
            f.write(f"### 第 {item['row_num']} 行（来源：{item['step']}）\n\n")

            # 生成 Markdown 表格
            row_data = item['row_data']
            f.write("| " + " | ".join(row_data.keys()) + " |\n")
            f.write("|" + "------|" * len(row_data) + "\n")

            # 加粗非法值
            values = []
            for col, val in row_data.items():
                if col in item['invalid_cols']:
                    values.append(f"**{escape_md_table(str(val))}**")
                else:
                    values.append(escape_md_table(str(val)))
            f.write("| " + " | ".join(values) + " |\n\n")

            f.write(f"**非法值**：{item['reason']}\n\n---\n\n")
```

#### 3.2 生成列映射说明表

```python
def _generate_column_mapping(
    label_info: Tuple[str, str],
    reactant_info: List[Tuple[str, str]],
    product_info: Tuple[str, str],
    other_info: List[Tuple[str, str]],
    condition_info: List[Tuple[str, str]],
    output_path: Path
) -> None:
    """生成列映射 CSV 文件。"""
    rows = []

    # 标签列
    rows.append({
        'origin_name': label_info[0],
        'role': 'label',
        'name': label_info[1]
    })

    # 反应物列
    for origin, name in reactant_info:
        rows.append({'origin_name': origin, 'role': 'reactant', 'name': name})

    # 产物列
    rows.append({
        'origin_name': product_info[0],
        'role': 'product',
        'name': product_info[1]
    })

    # 其他组分列
    for origin, name in other_info:
        rows.append({'origin_name': origin, 'role': 'others', 'name': name})

    # 条件列
    for origin, name in condition_info:
        rows.append({'origin_name': origin, 'role': 'condition', 'name': name})

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
```

#### 3.3 生成规范数据集

**动态列扩展逻辑**：

```python
def _generate_normalized_dataset(
    df_raw: pd.DataFrame,
    column_mapping: Dict[str, Tuple[str, str]],  # {原列名: (角色, 新列名)}
    invalid_rows: Set[int],
    output_path: Path
) -> None:
    """
    生成规范数据集，列顺序：
    reactant-1, reactant-2, ..., solvent, catalyst, ..., temperature, ..., product, yield
    """
    # 1. 过滤非法行
    valid_indices = [i for i in df_raw.index if (i + 2) not in invalid_rows]
    df = df_raw.loc[valid_indices].reset_index(drop=True)

    # 2. 按角色分组列
    reactant_cols = [(o, n) for o, (r, n) in column_mapping.items() if r == 'reactant']
    product_col = [(o, n) for o, (r, n) in column_mapping.items() if r == 'product'][0]
    other_cols = [(o, n) for o, (r, n) in column_mapping.items() if r == 'others']
    condition_cols = [(o, n) for o, (r, n) in column_mapping.items() if r == 'condition']
    label_col = [(o, n) for o, (r, n) in column_mapping.items() if r == 'label'][0]

    # 3. 统计每个角色的最大组分数
    def count_components(df, cols):
        max_counts = {}
        for origin, name in cols:
            max_count = df[origin].apply(
                lambda x: len(split_multi_smiles(str(x))) if pd.notna(x) else 0
            ).max()
            max_counts[name] = max_count
        return max_counts

    reactant_max = count_components(df, reactant_cols)
    other_max = count_components(df, other_cols)

    # 4. 构建新列名
    new_columns = []

    # 反应物列（如 reactant-1, reactant-2）
    for name, max_count in reactant_max.items():
        if max_count == 1:
            new_columns.append(name)
        else:
            new_columns.extend([f"{name}-{i+1}" for i in range(max_count)])

    # 其他组分列（如 solvent, catalyst-1, catalyst-2）
    for name, max_count in other_max.items():
        if max_count == 1:
            new_columns.append(name)
        else:
            new_columns.extend([f"{name}-{i+1}" for i in range(max_count)])

    # 条件列（不拆分，如 temperature, pressure）
    new_columns.extend([name for _, name in condition_cols])

    # 产物列（固定 1 列）
    new_columns.append(product_col[1])

    # 标签列（固定 1 列）
    new_columns.append(label_col[1])

    # 5. 逐行填充数据
    normalized_data = {col: [] for col in new_columns}
    normalized_data['_original_row_index'] = []  # 隐藏列，记录原始行号

    for idx, row in df.iterrows():
        normalized_data['_original_row_index'].append(idx + 2)  # 原始显示行号

        # 填充反应物
        for origin, name in reactant_cols:
            components = split_multi_smiles(str(row[origin])) if pd.notna(row[origin]) else []
            max_count = reactant_max[name]
            if max_count == 1:
                normalized_data[name].append(components[0] if components else '')
            else:
                for i in range(max_count):
                    col_name = f"{name}-{i+1}"
                    normalized_data[col_name].append(components[i] if i < len(components) else '')

        # 填充其他组分（逻辑同反应物）
        for origin, name in other_cols:
            components = split_multi_smiles(str(row[origin])) if pd.notna(row[origin]) else []
            max_count = other_max[name]
            if max_count == 1:
                normalized_data[name].append(components[0] if components else '')
            else:
                for i in range(max_count):
                    col_name = f"{name}-{i+1}"
                    normalized_data[col_name].append(components[i] if i < len(components) else '')

        # 填充条件列（直接复制数值）
        for origin, name in condition_cols:
            normalized_data[name].append(row[origin] if pd.notna(row[origin]) else '')

        # 填充产物（固定 1 个 SMILES）
        normalized_data[product_col[1]].append(row[product_col[0]])

        # 填充标签
        normalized_data[label_col[1]].append(row[label_col[0]])

    # 6. 保存为 CSV
    df_normalized = pd.DataFrame(normalized_data)
    df_normalized.to_csv(output_path, index=False, encoding='utf-8-sig')

    print(f"  [完成] 规范数据集已生成：{output_path}")
    print(f"  [统计] 原始行数：{len(df_raw)}，有效行数：{len(df)}，排除行数：{len(invalid_rows)}")
```

---

### 步骤 4：指定描述符

#### 4.1-4.3 描述符嵌入配置

**交互界面设计**：

```python
def _configure_descriptor_embedding(
    normalized_columns: List[str],  # 规范数据集的所有列
    column_roles: Dict[str, str]     # {列名: 角色}
) -> Dict[str, Any]:
    """
    返回：{
        'morgan': {'order': ['reactant-1', 'reactant-2', 'solvent', 'product']},
        'maf': {'columns': ['reactant-1', 'reactant-2', 'solvent', 'product']},
        'drfp': {'reactants': ['reactant-1', 'reactant-2', 'catalyst'], 'product': 'product'},
        ...
    }
    """
    print("\n可用描述符及其嵌入方式：")
    print("  1. morgan / maccs / rdkit2d / fisd / molmetalm")
    print("     → 横向拼接（可编辑列选择和拼接顺序）")
    print("  2. maf")
    print("     → 逐点加和（可编辑参与列，无顺序）")
    print("  3. drfp")
    print("     → 固定格式 reactant->product（可指定额外反应物）")
    print()

    # 展示当前规范数据集列
    smiles_cols = [c for c, r in column_roles.items() if r in ('reactant', 'product', 'others')]
    print("当前可用 SMILES 列：")
    for i, col in enumerate(smiles_cols):
        print(f"  {i+1}. {col} ({column_roles[col]})")
    print()

    config = {}

    # 询问是否使用推荐配置
    choice = _ask_optional("是否使用推荐配置（reactant → others → product）？[Y/n]", default="Y")
    if choice.lower() in ('y', 'yes', ''):
        reactant_cols = [c for c in smiles_cols if column_roles[c] == 'reactant']
        product_cols = [c for c in smiles_cols if column_roles[c] == 'product']
        other_cols = [c for c in smiles_cols if column_roles[c] == 'others']
        default_order = reactant_cols + other_cols + product_cols

        config['morgan'] = {'order': default_order}
        config['maccs'] = {'order': default_order}
        config['rdkit2d'] = {'order': default_order}
        config['fisd'] = {'order': default_order}
        config['molmetalm'] = {'order': default_order}
        config['maf'] = {'columns': default_order}
        config['drfp'] = {
            'reactants': reactant_cols,
            'product': product_cols[0]
        }

        print(f"  [配置] 推荐配置已应用：{' → '.join(default_order)}")
        return config

    # 手动配置（逐个描述符询问）
    # ... 实现略（交互复杂度高，建议大部分用户使用推荐配置）
```

#### 4.4 配置确认与预览

```python
def _preview_descriptor_config(config: Dict[str, Any]) -> None:
    """展示描述符配置预览，供用户确认。"""
    print("\n描述符配置预览：")
    print()

    for desc, cfg in config.items():
        if desc == 'drfp':
            print(f"  {desc.upper():12s}  {', '.join(cfg['reactants'])} → {cfg['product']}")
        elif desc == 'maf':
            print(f"  {desc.upper():12s}  逐点加和：{', '.join(cfg['columns'])}")
        else:
            print(f"  {desc.upper():12s}  {' → '.join(cfg['order'])}")
    print()
```

---

### 步骤 5-8：指定模型、数据集信息、报告格式、确认命令行

**这些步骤与现有实现完全一致，无需修改。**

---

## 3. 建议和疑问

### 3.1 设计改进建议

#### 建议 1：引入"配置文件导出"功能

**背景**：用户完成一次完整的交互流程后，可能需要在其他数据集上复用相同配置（如列角色声明、描述符配置）。

**建议**：在步骤 8 确认命令行后，询问用户是否导出配置文件（JSON 格式）：

```json
{
  "version": "1.0",
  "column_mapping": [
    {"origin_name": "r1", "role": "reactant", "name": "reactant"},
    {"origin_name": "p", "role": "product", "name": "product"}
  ],
  "descriptor_config": {
    "morgan": {"order": ["reactant", "product"]},
    "drfp": {"reactants": ["reactant"], "product": "product"}
  },
  "model_selection": ["xgb", "rf"],
  "output_format": "both"
}
```

**优势**：
- 团队协作时可共享配置
- 批量处理多个数据集时节省时间
- 可作为"模板"供初学者参考

---

#### 建议 2：在规范数据集中保留原始列作为隐藏列

**背景**：步骤 3.3 生成的规范数据集会将多组分 SMILES 拆分为多列，但用户可能需要查看原始格式（如调试时）。

**建议**：在规范数据集中新增 `_raw_<角色>` 隐藏列（以下划线开头，建模时自动忽略）：

| reactant-1 | reactant-2 | _raw_reactant | product | yield |
|------------|------------|---------------|---------|-------|
| CCO        | CC         | CCO.CC        | CCOC    | 0.85  |

**优势**：
- 用户可通过 Excel 打开规范数据集，直接查看原始数据
- 方便排查"拆分逻辑是否正确"

---

#### 建议 3：为"其他组分列"提供智能命名建议

**背景**：步骤 2.5 要求用户为每个其他组分列输入名称，但用户可能不清楚应该用什么名称。

**建议**：根据原列名关键词自动推荐：

```python
def _suggest_name(origin_col: str) -> str:
    """根据原列名推荐角色名称。"""
    col_lower = origin_col.lower()
    if 'solvent' in col_lower or 'solv' in col_lower:
        return 'solvent'
    if 'catalyst' in col_lower or 'cat' in col_lower or 'pd' in col_lower:
        return 'catalyst'
    if 'base' in col_lower:
        return 'base'
    if 'reagent' in col_lower or 'additive' in col_lower:
        return 'reagent'
    return origin_col.lower().replace(' ', '_')  # 默认使用原列名
```

**优势**：
- 减少用户输入负担
- 提供"最佳实践"命名约定

---

#### 建议 4：非法值报告支持"一键修复"脚本

**背景**：非法值报告展示了所有错误，但用户仍需手动编辑原始 CSV 修复。

**建议**：生成配套的 Python 修复脚本（`fix_invalid_data.py`）：

```python
# 自动生成的修复脚本
import pandas as pd

df = pd.read_csv('dataset/example.csv')

# 修复第 5 行：sub_1 列的 'CCX' → 删除此行（或替换为正确 SMILES）
df = df.drop(index=4)  # 行号 5 → 索引 4

# 修复第 17 行：yield 列的 'high' → 删除此行（或替换为数值）
df = df.drop(index=16)

df.to_csv('dataset/example_fixed.csv', index=False, encoding='utf-8-sig')
print("修复完成！请检查 example_fixed.csv")
```

**优势**：
- 用户可直接运行脚本批量删除非法行
- 或在脚本中手动编辑替换值，再运行

---

### 3.2 需要澄清的问题

#### 问题 1：列映射文件的加载时机

**需求文档中的表述**（步骤 1.2）：
> "可选输入列映射文件，如果输入，则直接进行数据合法性检验"

**疑问**：
- "直接进行数据合法性检验"是指**跳过步骤 2（逐列声明）**，还是**自动填充步骤 2 的默认值，用户确认后再检验**？
- 如果跳过步骤 2，如何处理列映射文件中缺失的列？（如原数据集有 10 列，但列映射文件只定义了 8 列）

**建议澄清**：
- **方案 A（推荐）**：列映射文件作为"历史配置参考"，自动填充步骤 2 的默认值，但用户仍可修改
- **方案 B**：列映射文件作为"强制配置"，跳过步骤 2，直接进入步骤 3；但需增加"列映射完整性校验"（确保所有必需列都已定义）

---

#### 问题 2：非法值的处理策略

**需求文档中的表述**（步骤 2.2）：
> "对于含有非法值的行，在创建规范数据集时跳过这些行（而不是出现非法值就退出程序），但这些行依然参与到后续的数值合法性检验中"

**疑问**：
- "依然参与到后续的数值合法性检验中"的意义是什么？
- 如果某行在步骤 2.2（标签列检验）已被标记为非法，为何还要在步骤 2.6（条件列检验）再次检验？

**可能的理解**：
- **理解 A**：为了在非法值报告中全面展示该行的所有错误（如"第 5 行既有非法标签，又有非法条件值"）
- **理解 B**：表述有误，应为"被标记为非法的行不参与后续检验"

**建议澄清**：
- 如果采用理解 A，需在实现中维护"已标记行"集合，但仍对其进行检验（仅记录错误，不重复计数）
- 如果采用理解 B，需修改需求文档表述

---

#### 问题 3：产物列的"单 SMILES"定义

**需求文档中的表述**（步骤 2.4）：
> "用户只能输入一列，且该列中所有单元格都只能包含一个独立的 SMILES（这是产物列独有的）"

**疑问**：
- 离子对化合物（如 `[Na+].[Cl-]`）应被视为 1 个 SMILES 还是 2 个 SMILES？
- 溶剂化物（如 `CCO.O`，乙醇水合物）应被视为 1 个 SMILES 还是 2 个 SMILES？

**建议澄清**：
- **严格定义**："单 SMILES"指不含任何分隔符（点号、分号、逗号）的 SMILES 字符串
  - 优势：定义清晰，易于实现
  - 劣势：用户需手动修改原始数据（如将 `[Na+].[Cl-]` 改写为单一盐型 SMILES，但 RDKit 可能不支持）
- **宽松定义**："单 SMILES"指化学意义上的单一产物，允许离子对和溶剂化物
  - 优势：更符合化学实际
  - 劣势：需实现复杂的"离子对检测"和"溶剂化物检测"逻辑

---

#### 问题 4：描述符配置的复杂度

**需求文档中的表述**（步骤 4.4）：
> "用户可以通过列序号进行编辑，对于可编辑顺序的描述符，用户输入的顺序就是描述符拼接顺序（需要强调给用户）"

**疑问**：
- 这是否要求为**每个描述符单独配置**拼接顺序？（如 morgan 用 `reactant-product`，fisd 用 `product-reactant`）
- 还是所有横向拼接描述符共用同一配置？

**建议澄清**：
- **方案 A（推荐）**：所有横向拼接描述符共用配置（简化交互）
- **方案 B**：允许为每个描述符单独配置（提供最大灵活性，但交互复杂度高）

---

#### 问题 5：规范数据集的列名冲突

**场景**：
- 用户在步骤 2.3 将列 `r1` 命名为 `reactant`
- 用户在步骤 2.5 将列 `catalyst` 命名为 `reactant`（误操作）

**疑问**：
- 是否允许不同角色的列使用相同名称？
- 如果允许，规范数据集中如何区分？（如 `reactant-1` 来自反应物列，`reactant-2` 来自催化剂列）

**建议澄清**：
- **方案 A（推荐）**：强制列名唯一，步骤 2 中维护 `used_names` 集合，禁止重复
- **方案 B**：允许重复，但在规范数据集生成时自动追加角色前缀（如 `reactant_reactant-1`、`others_reactant-1`）

---

### 3.3 潜在的用户体验问题

#### UX 问题 1：步骤 2 的交互流程过长

**问题**：
- 步骤 2.1 → 2.2 → 2.3 → 2.4 → 2.5 → 2.6，共 6 个子步骤
- 对于含有 10 列的数据集，用户可能需要回答 20+ 次问题
- 容易疲劳和误操作

**建议**：
- 提供"快速模式"：自动根据列名关键词推荐角色，用户一次性确认所有列
  ```
  自动检测到以下列角色（按 Enter 确认，输入 'edit' 手动修改）：
    列 'yield' → 标签（label）
    列 'r1', 'r2' → 反应物（reactant）
    列 'product' → 产物（product）
    列 'solvent' → 其他组分（solvent）
  ```
- 允许用户在步骤 3.1（查看非法值报告）后返回步骤 2 修改配置

---

#### UX 问题 2：非法值报告可能过大

**问题**：
- 如果数据集有 50000 行，其中 10000 行非法
- 生成的 Markdown 文件可能达到 100+ MB
- 用户浏览器无法打开，或加载极慢

**建议**：
- 限制报告最多展示前 1000 行
- 超出部分生成单独的 CSV 文件（`invalid_rows_full.csv`）
- 在报告末尾提供下载链接

---

#### UX 问题 3：规范数据集列名可能过长

**问题**：
- 某个反应最多有 8 个反应物，生成的列名为 `reactant-1` 到 `reactant-8`
- 用户在 Excel 中查看时，列名可能被截断

**建议**：
- 提供"列名缩写"选项（如 `r1`, `r2`, ..., `r8` 替代 `reactant-1`, `reactant-2`, ...）
- 在列映射文件中保留完整名称，规范数据集使用缩写

---

## 4. 脚本交互逻辑流程图

```mermaid
flowchart TD
    Start([开始: 启动交互向导]) --> Input1{输入初始数据集路径}
    Input1 --> Load[读取 CSV 文件<br/>尝试 UTF-8-sig / GBK 编码]
    Load --> ShowCols[展示所有列<br/>序号 + 列名]

    ShowCols --> Input2{可选: 输入列映射文件?}
    Input2 -->|是| LoadMapping[加载列映射文件<br/>JSON 格式]
    Input2 -->|否| Input3
    LoadMapping --> AutoFill[自动填充步骤2的默认值]
    AutoFill --> Input3

    Input3[输入项目名称<br/>正则校验: ^[A-Za-z0-9_-]+$]
    Input3 --> Input4[输入项目文件夹路径<br/>默认: ./result/<项目名>]
    Input4 --> CreateDir[创建项目文件夹结构<br/>├ 非法输入排除报告.md<br/>├ 列映射说明.csv<br/>└ 规范数据集.csv]

    CreateDir --> Step2_1[步骤2.1: 筛选备选列]
    Step2_1 --> ShowCandidate[展示所有列<br/>用户选择需要的列]
    ShowCandidate --> Step2_2[步骤2.2: 声明标签列]

    Step2_2 --> SelectLabel[选择1列作为标签列<br/>允许: 列名/字母/序号]
    SelectLabel --> NameLabel{输入标签列名称<br/>默认: yield}
    NameLabel -->|非法字符| NameLabel
    NameLabel -->|合法| ValidateLabel[合法性检验: 数值类型<br/>累积非法行号]
    ValidateLabel --> RemoveLabel[从备选列中移除]

    RemoveLabel --> Step2_3[步骤2.3: 声明反应物列]
    Step2_3 --> SelectReactant[选择多列作为反应物<br/>空格分隔]
    SelectReactant --> NameReactant{为每列输入名称?}
    NameReactant -->|统一命名| UnifyName[统一命名为 reactant]
    NameReactant -->|单独命名| IndivName[逐列输入名称]
    UnifyName --> ValidateReactant
    IndivName --> ValidateReactant[合法性检验: SMILES格式<br/>不允许空值<br/>累积非法行号]
    ValidateReactant --> RemoveReactant[从备选列中移除]

    RemoveReactant --> Step2_4[步骤2.4: 声明产物列]
    Step2_4 --> SelectProduct[选择1列作为产物列]
    SelectProduct --> NameProduct{输入产物列名称<br/>默认: product}
    NameProduct -->|非法字符| NameProduct
    NameProduct -->|合法| ValidateProduct[合法性检验: SMILES格式<br/>+ 单SMILES约束<br/>累积非法行号]
    ValidateProduct --> CheckSingle{是否为单SMILES?}
    CheckSingle -->|否| ErrorSingle[报错: 产物列只能包含<br/>1个SMILES<br/>离子对除外]
    CheckSingle -->|是| RemoveProduct[从备选列中移除]
    ErrorSingle --> SelectProduct

    RemoveProduct --> Step2_5[步骤2.5: 声明其他组分列]
    Step2_5 --> AskOther{是否添加其他组分?<br/>溶剂/催化剂/碱等}
    AskOther -->|否| Step2_6
    AskOther -->|是| SelectOther[选择1列]
    SelectOther --> NameOther{输入列名称<br/>推荐: solvent/catalyst/base}
    NameOther -->|非法字符| NameOther
    NameOther -->|合法| ValidateOther[合法性检验: SMILES格式<br/>允许空值<br/>累积非法行号]
    ValidateOther --> RemoveOther[从备选列中移除]
    RemoveOther --> AskOther

    Step2_6[步骤2.6: 声明条件数值列]
    Step2_6 --> AskCondition{是否添加条件列?<br/>温度/压力等}
    AskCondition -->|否| Step3
    AskCondition -->|是| SelectCondition[选择1列]
    SelectCondition --> NameCondition{输入列名称<br/>推荐: temperature/pressure}
    NameCondition -->|非法字符| NameCondition
    NameCondition -->|合法| ValidateCondition[合法性检验: 数值类型<br/>允许空值<br/>累积非法行号]
    ValidateCondition --> RemoveCondition[从备选列中移除]
    RemoveCondition --> AskCondition

    Step3[步骤3: 生成规范数据集<br/>与非法值报告]
    Step3 --> CheckInvalid{是否存在非法值?}
    CheckInvalid -->|是| GenReport[步骤3.1: 生成非法输入<br/>排除报告 Markdown]
    CheckInvalid -->|否| GenMapping
    GenReport --> ShowReport[展示报告路径<br/>要求用户浏览确认]
    ShowReport --> UserConfirm1{用户确认?}
    UserConfirm1 -->|否| AskFix[提示用户修复原始数据<br/>或继续]
    AskFix --> FixChoice{修复或继续?}
    FixChoice -->|修复| End1([结束: 用户去修复数据])
    FixChoice -->|继续| GenMapping
    UserConfirm1 -->|是| GenMapping

    GenMapping[步骤3.2: 生成列映射说明表<br/>CSV格式]
    GenMapping --> GenNormalized[步骤3.3: 生成规范数据集]
    GenNormalized --> CountComponents[统计每个角色的<br/>最大组分数]
    CountComponents --> CreateCols[动态创建列名<br/>reactant-1, reactant-2, ...]
    CreateCols --> FillData[逐行填充数据<br/>拆分多组分SMILES]
    FillData --> SaveNormalized[保存规范数据集 CSV<br/>排除非法行]
    SaveNormalized --> ShowNormalized[展示规范数据集路径<br/>要求用户浏览确认]
    ShowNormalized --> UserConfirm2{用户确认?}
    UserConfirm2 -->|否| AskRedo[提示用户重新配置<br/>或继续]
    AskRedo --> RedoChoice{重新配置或继续?}
    RedoChoice -->|重新配置| Step2_1
    RedoChoice -->|继续| Step4
    UserConfirm2 -->|是| Step4

    Step4[步骤4: 指定描述符]
    Step4 --> ShowDescriptors[展示所有描述符<br/>说明嵌入方式]
    ShowDescriptors --> AskConfig{使用推荐配置?<br/>reactant→others→product}
    AskConfig -->|是| ApplyDefault[应用推荐配置<br/>所有横向拼接描述符]
    AskConfig -->|否| ManualConfig[手动配置每个描述符<br/>输入列序号]
    ApplyDefault --> PreviewConfig
    ManualConfig --> ConfigMorgan[配置 morgan<br/>输入列序号和顺序]
    ConfigMorgan --> ConfigMAF[配置 maf<br/>输入参与列]
    ConfigMAF --> ConfigDRFP[配置 drfp<br/>指定额外反应物]
    ConfigDRFP --> PreviewConfig[展示配置预览<br/>如: reactant-1 → solvent → product]
    PreviewConfig --> UserConfirm3{用户确认?}
    UserConfirm3 -->|否| ManualConfig
    UserConfirm3 -->|是| Step5

    Step5[步骤5: 指定建模模型]
    Step5 --> SelectModels[选择模型<br/>xgb/rf/svm/autogluon<br/>空格分隔, 留空=全选]
    SelectModels --> Step6[步骤6: 补充数据集信息]
    Step6 --> InputCitation[输入文献引用 可选]
    InputCitation --> InputURL[输入开源地址 可选]
    InputURL --> InputNotes[输入备注 可选]
    InputNotes --> Step7[步骤7: 选择报告输出格式]
    Step7 --> SelectFormat[选择格式<br/>html/md/both<br/>默认: both]
    SelectFormat --> Step8[步骤8: 确认命令行]
    Step8 --> ShowCommand[展示等效 CLI 命令<br/>--csv ... --label-col ...]
    ShowCommand --> UserConfirm4{按 Enter 确认执行<br/>Ctrl+C 取消}
    UserConfirm4 -->|取消| End2([结束: 用户取消])
    UserConfirm4 -->|确认| BuildArgv[构建 argv 参数列表]
    BuildArgv --> CallMain[调用 main argv<br/>执行建模 pipeline]
    CallMain --> End3([结束: 生成建模报告])

    style Start fill:#e1f5e1
    style End1 fill:#ffe1e1
    style End2 fill:#ffe1e1
    style End3 fill:#e1f5e1
    style ErrorSingle fill:#ffcccc
    style GenReport fill:#fff4e1
    style GenNormalized fill:#e1f0ff
    style CallMain fill:#e1ffe1
```

### 流程图说明

#### 关键决策点

1. **列映射文件加载** (Input2)
   - 若提供，自动填充步骤 2 的默认值
   - 仍需用户确认，不跳过合法性检验

2. **产物列单 SMILES 校验** (CheckSingle)
   - 检测多组分 SMILES（点号/分号/逗号）
   - 离子对特殊处理（`[Na+].[Cl-]` 视为单一产物）

3. **非法值报告生成** (CheckInvalid)
   - 仅在存在非法值时生成
   - 用户可选择修复原始数据或继续

4. **描述符配置** (AskConfig)
   - 推荐配置：一键应用，适合 95% 用户
   - 手动配置：高级用户自定义拼接顺序

#### 循环与迭代

- **步骤 2.5** (声明其他组分): 循环询问直到用户选择"不再添加"
- **步骤 2.6** (声明条件列): 循环询问直到用户选择"不再添加"
- **步骤 3** (确认规范数据集): 用户可返回步骤 2.1 重新配置

#### 错误处理

- **非法字符检测**: 列名仅允许 `[A-Za-z0-9_]`，循环要求重新输入
- **单 SMILES 约束**: 产物列检测到多组分时拒绝并提示
- **列名冲突**: 维护 `used_names` 集合，禁止重复

#### 累积验证模式

- 所有合法性检验均采用**累积模式**（不立即退出）
- 在步骤 3.1 统一展示所有非法值
- 用户可一次性修复所有问题

---

## 总结与后续步骤

### 可行性结论

✅ **技术上完全可行**，主要原因：
1. 现有代码已实现 70% 的所需功能（列角色分类、SMILES 验证、CSV 加载）
2. 需新增的功能（规范数据集生成、非法值报告）技术难度中等
3. 无需引入新的外部依赖，Pandas + RDKit 足以支撑

⚠️ **需注意的风险**：
1. 产物列"单 SMILES"定义需明确（离子对、溶剂化物的处理）
2. 描述符配置界面交互复杂度高，建议大部分用户使用推荐配置
3. 非法值报告可能过大，需限制展示行数

### 推荐实施路径

#### 第一阶段（核心功能，1-2 周）
1. 重构 `wizard()` 函数，实现步骤 1-2 的逐列声明
2. 实现累积式合法性检验（非法值不中断流程）
3. 生成列映射文件和规范数据集

#### 第二阶段（报告与配置，1 周）
4. 实现非法输入排除报告（Markdown 格式）
5. 实现描述符配置界面（推荐配置 + 手动配置）

#### 第三阶段（优化与测试，1 周）
6. 集成测试（边界情况、大数据集、异常输入）
7. 用户体验优化（错误提示、配置预览、快速模式）

### 关键待澄清问题

建议与需求方确认以下 5 个问题后再开始开发：

1. **列映射文件加载逻辑**：是否跳过步骤 2？如何处理缺失列？
2. **非法值检验策略**：已标记行是否仍参与后续检验？
3. **产物列单 SMILES 定义**：离子对和溶剂化物如何处理？
4. **描述符配置粒度**：是否为每个描述符单独配置拼接顺序？
5. **列名冲突处理**：是否允许不同角色使用相同列名？

---

**报告完成时间**：2026-06-28
**下一步行动**：等待需求方反馈，澄清上述 5 个问题后进入开发阶段。
