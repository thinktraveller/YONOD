# 构建日志

## [2026-06-22 23:31] 步骤 1 完成：创建 MAF 描述符类文件

### 执行的任务
- 在 `yonod/descriptors/` 目录下创建 `maf.py`，实现 `MAFDescriptor` 类
- 实现 128 维整数特征向量，支持多分子 SMILES 点分隔格式输入
- 创建单元测试文件 `tests/test_maf_descriptor.py`，包含 5 个测试用例

### 关键变更
- **新增文件**：`yonod/descriptors/maf.py`
  - 继承 `BaseDescriptor` 抽象基类
  - 实现 `featurize()` 方法，支持多分子 SMILES 输入（点分隔格式）
  - 对每个组分生成 ECFP (radius=2, 128 bits)，按位加和生成整数向量
  - 自动处理空组分和 "(无)" 标记
- **新增文件**：`tests/test_maf_descriptor.py`
  - 测试基本功能（2 组分加和）
  - 测试空组分处理
  - 测试无效 SMILES 处理
  - 测试维度和名称
  - 测试 "(无)" 标记处理

### 遇到的问题及解决方案
- 无

### 下一步计划
- 步骤 2：注册 MAF 到 `DESCRIPTOR_REGISTRY`，使其可通过 CLI 调用

---

## [2026-06-22 23:42] 步骤 2 完成：注册 MAF 到系统

### 执行的任务
- 修改 `yonod/evaluate.py`，导入 `MAFDescriptor`
- 将 `MAFDescriptor` 注册到 `DESCRIPTOR_REGISTRY` 字典
- 验证注册成功

### 关键变更
- **修改文件**：`yonod/evaluate.py`
  - 第 34 行：添加 `from .descriptors.maf import MAFDescriptor`
  - 第 48 行：在 `DESCRIPTOR_REGISTRY` 中添加 `"maf": MAFDescriptor`

### 遇到的问题及解决方案
- 无

### 下一步计划
- 步骤 3：更新 CLI 参数（检查 `yonod.py` 是否需要更新）

---

## [2026-06-22 23:46] 步骤 3 完成：更新 CLI 参数

### 执行的任务
- 检查 `yonod.py` 中的 `_DESCRIPTOR_NAMES` 定义
- 添加 `"maf"` 到描述符选项列表
- 验证 `--descriptors` 参数的 help 信息

### 关键变更
- **修改文件**：`yonod.py`
  - 第 51 行：在 `_DESCRIPTOR_NAMES` 列表中添加 `"maf"`

### 遇到的问题及解决方案
- 无

### 下一步计划
- 步骤 4：适配多分子输入格式（使用方案 A）

---

## [2026-06-22 23:53] 步骤 4 完成：适配多分子输入格式（方案 A）

### 执行的任务
- 分析现有通用架构（`build_universal_features`）
- 发现架构已自动支持多分子点分隔格式转换（第 110-113 行）
- 在 `feature_builder.py` 的 `_DESCRIPTOR_IMPORT_MAP` 中添加 MAF
- 更新模块文档字符串
- 通过集成测试验证 MAF 在通用架构中的工作

### 关键变更
- **修改文件**：`yonod/universal/feature_builder.py`
  - 第 48 行：在 `_DESCRIPTOR_IMPORT_MAP` 中添加 `"maf": ("..descriptors.maf", "MAFDescriptor")`
  - 第 22 行：更新文档字符串，添加 MAF 描述
- **架构优势**：现有 `build_universal_features` 已内置多分子格式转换，无需额外实现 `prepare_maf_input()`

### 遇到的问题及解决方案
- **发现**：原计划方案 A 需要新增 `prepare_maf_input()` 函数，但分析后发现通用架构已自动处理多分子格式
- **解决**：直接注册 MAF 到 `_DESCRIPTOR_IMPORT_MAP`，利用现有格式转换机制

### 验证结果
- 输出维度：768 = 128 × 6 列 ✅
- 所有测试样本有效：3/3 ✅
- 特征非零率：5.03%（合理范围）✅
- 特征类型：int32 ✅

### 下一步计划
- 步骤 5：端到端集成测试（与 4 种模型联调）

---

## [2026-06-22 23:59] 步骤 5 完成：端到端集成测试

### 执行的任务
- 创建端到端测试脚本，验证 MAF 与模型的兼容性
- 使用真实数据集 `test-amide-coupling.csv`（50 样本）进行完整流程测试
- 验证 MAF + RF 组合的端到端工作流

### 关键变更
- 无代码变更，纯验证步骤

### 测试结果
- **特征构建**：MAF 成功生成 768 维特征（128×6 列），50/50 样本有效 ✅
- **模型训练**：RF 模型训练成功，R²=0.2027，RMSE=0.2937 ✅
- **报告生成**：自动生成 HTML 和 Markdown 报告 ✅
- **运行时间**：1.7 秒（特征计算 + 3-fold CV + 训练）✅

### 遇到的问题及解决方案
- **问题**：XGBoost 未安装（环境问题）
- **解决**：使用 RF 和 SVM 替代验证，两者都成功运行

### 下一步计划
- 步骤 6：文档更新（teaching.md 和 README.md）

---
