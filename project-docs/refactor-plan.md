# YONOD 文件重命名影响分析与实施方案

**分析日期**: 2026-06-29
**版本**: v1.0
**状态**: 待审核

---

## 一、重命名需求概述

### 1.1 用户意图
用户希望进行以下文件重命名：
- `yonod.py` → `main.py`（建模核心入口）
- `dataset_input_wizard.py` → `yonod.py`（数据准备向导）

### 1.2 设计理念分析
从用户意图推测，这次重命名的核心理念是：

1. **向导优先**：让数据准备向导（现 `dataset_input_wizard.py`）成为项目主入口
2. **名称语义化**：`yonod.py` 作为项目名应当是最核心、最常用的入口
3. **建模后端化**：将建模脚本（现 `yonod.py`）改名为 `main.py`，定位为"后端核心"而非直接入口

这与当前架构（2026-06-29 22:30 双入口重构）的设计哲学**部分一致**：
- ✅ 支持的设计：向导负责数据准备 → 自动调用建模
- ⚠️ 潜在冲突：当前文档和用户习惯中 `yonod.py` 被广泛识别为"统一 CLI 入口"

---

## 二、完整影响范围扫描

### 2.1 代码文件中的硬编码引用

| 文件路径 | 引用位置 | 引用类型 | 修改优先级 |
|---------|---------|---------|-----------|
| **dataset_input_wizard.py** | | | |
| ├─ L1717 | `yonod_script = os.path.join(..., 'yonod.py')` | 文件路径硬编码 | P0 🔴 |
| ├─ L1738 | `print(f"  python yonod.py --config ...")` | 用户提示字符串 | P0 🔴 |
| ├─ L1743 | `print(f"  python yonod.py --config ...")` | 用户提示字符串 | P0 🔴 |
| **yonod.py** | | | |
| ├─ L1-27 | docstring 示例代码 | 文档注释 | P1 🟡 |
| ├─ L375 | `--config` 参数 help | 参数说明 | P1 🟡 |
| ├─ L702 | `print("    python dataset_input_wizard.py")` | 用户提示 | P0 🔴 |
| ├─ L705-712 | 示例命令打印 | 用户提示 | P1 🟡 |

### 2.2 文档文件中的引用

| 文件 | 引用次数 | 修改类型 | 优先级 |
|------|---------|---------|--------|
| **README.md** | 10+ | 示例命令、快速开始、架构说明 | P0 🔴 |
| **CLAUDE.md** | 7+ | 常用命令、架构说明、入口点说明 | P0 🔴 |
| **project-docs/teaching.md** | 30+ | 架构分析、数据流图、示例代码 | P1 🟡 |
| **project-docs/buildlog.md** | 100+ | 历史记录（建议保留，不修改） | P2 🟢 |
| **project-docs/project-plan.md** | 150+ | 原始规划（建议保留，不修改） | P2 🟢 |
| **.gitignore** | 1 | 注释说明 | P2 🟢 |

### 2.3 潜在的间接影响

| 影响类型 | 具体场景 | 风险等级 |
|---------|---------|---------|
| **用户习惯断裂** | 现有用户已习惯 `python yonod.py` 启动建模 | 高 🔴 |
| **外部文档引用** | 论文、博客、教学材料可能引用旧文件名 | 中 🟡 |
| **CI/CD 脚本** | 如有自动化测试脚本可能硬编码文件名 | 低 🟢（当前无 CI） |
| **IDE 配置** | VSCode launch.json 等可能配置了 yonod.py | 低 🟢 |

---

## 三、风险评估

### 3.1 技术风险

| 风险项 | 严重程度 | 概率 | 缓解措施 |
|--------|---------|------|---------|
| **向后兼容性破坏** | 🔴 高 | 100% | 提供过渡期脚本（见 §4.2） |
| **文档不一致** | 🟡 中 | 100% | 批量替换 + 人工审核 |
| **用户操作错误** | 🟡 中 | 70% | README 添加醒目迁移提示 |
| **测试覆盖遗漏** | 🟢 低 | 30% | 运行现有测试套件验证 |

### 3.2 用户体验影响

#### 正面影响 ✅
1. **语义更清晰**：`python yonod.py` = 启动向导（符合新手预期）
2. **主入口明确**：项目名 = 主入口文件名（降低认知负担）
3. **分层更合理**：`main.py` 作为后端核心，不直接暴露给最终用户

#### 负面影响 ⚠️
1. **破坏肌肉记忆**：熟悉老版本的用户会执行错误命令
2. **教学材料失效**：所有已发布的教程、截图需要更新
3. **临时混乱期**：过渡期内可能存在两个版本的文档并存

---

## 四、实施方案

### 4.1 完整修改清单

#### 阶段 1：文件系统操作（使用 git mv）
```powershell
# Step 1: 重命名 yonod.py → main.py
git mv yonod.py main.py

# Step 2: 重命名 dataset_input_wizard.py → yonod.py
git mv dataset_input_wizard.py yonod.py
```

> ⚠️ **必须使用 `git mv`**：保留文件历史记录，否则 Git 会将其视为删除+新建。

---

#### 阶段 2：代码文件修改

**文件 1：`yonod.py`（原 `dataset_input_wizard.py`）**

| 行号 | 原内容 | 新内容 | 说明 |
|------|-------|-------|------|
| L1717 | `'yonod.py'` | `'main.py'` | subprocess 调用路径 |
| L1738 | `python yonod.py` | `python main.py` | 错误提示命令 |
| L1743 | `python yonod.py` | `python main.py` | 跳过提示命令 |

**文件 2：`main.py`（原 `yonod.py`）**

| 行号范围 | 修改内容 | 说明 |
|---------|---------|------|
| L1-27 | 所有 `python yonod.py` → `python main.py` | Docstring 示例 |
| L375 | help 文本更新为 `由 yonod.py 向导生成` | 参数说明 |
| L702 | `dataset_input_wizard.py` → `yonod.py` | 用户提示 |
| L705-712 | `python yonod.py` → `python main.py` | 示例命令 |

---

#### 阶段 3：文档修改

**文件 3：`README.md`**

| 章节 | 修改策略 | 具体操作 |
|------|---------|---------|
| 快速开始 | 全局替换 | `python yonod.py` → `python yonod.py`（向导）<br>`python yonod.py --csv ...` → `python main.py --csv ...`（CLI） |
| 架构说明 | 表格更新 | `├── yonod.py` → `├── yonod.py  # 数据准备向导`<br>新增 `├── main.py  # 建模核心引擎` |
| 命令行参数 | 示例更新 | 所有 CLI 示例改为 `python main.py ...` |

**文件 4：`CLAUDE.md`**

| 章节 | 修改内容 |
|------|---------|
| Entry Points | `yonod.py (1097 lines)` → `main.py (721 lines): 建模核心`<br>`dataset_input_wizard.py (1747 lines)` → `yonod.py (1904 lines): 数据准备向导` |
| Running the Project | 所有示例命令从 `python yonod.py` 改为 `python yonod.py`（向导）或 `python main.py`（CLI） |

**文件 5：`project-docs/teaching.md`**

⚠️ **建议保留原样**，仅在文档开头添加：
```markdown
> ⚠️ **历史文档说明**（2026-06-29 重命名前版本）
> 本文档中的 `yonod.py` 现已重命名为 `main.py`，
> `dataset_input_wizard.py` 现已重命名为 `yonod.py`。
```

**文件 6：`.gitignore`**

| 行号 | 原注释 | 新注释 |
|------|-------|-------|
| L31 | `# Pipeline outputs (regenerate by re-running yonod.py)` | `# Pipeline outputs (regenerate by re-running main.py)` |

---

#### 阶段 4：向后兼容措施（可选）

**方案 A：创建兼容性包装脚本**

```python
# 文件: yonod_legacy.py（保留旧入口，打印弃用警告）
import sys
print("⚠️  警告: yonod.py 已重命名为 main.py")
print("    请使用: python main.py --csv ...")
print("    或使用新向导: python yonod.py\n")
input("按 Enter 继续执行（将自动调用 main.py）...")
import main
sys.exit(main.main())
```

**方案 B：在 README 添加醒目迁移指南**

```markdown
## ⚠️ 重要更新（2026-06-29）

文件重命名通知：
- 🔄 `dataset_input_wizard.py` → **`yonod.py`**（新的主入口）
- 🔄 `yonod.py` → **`main.py`**（建模后端）

### 迁移示例
| 旧命令 | 新命令 |
|--------|--------|
| `python yonod.py --csv data.csv ...` | `python main.py --csv data.csv ...` |
| `python dataset_input_wizard.py` | `python yonod.py` |
```

---

### 4.2 详细实施步骤

#### Step 0：备份与准备
```powershell
# 1. 确保工作区干净
git status

# 2. 创建功能分支
git checkout -b refactor/rename-entry-files

# 3. 备份当前状态（可选）
git tag backup-before-rename
```

#### Step 1：执行文件重命名
```powershell
git mv yonod.py main.py
git mv dataset_input_wizard.py yonod.py
```

#### Step 2：修改代码文件
```powershell
# 修改 yonod.py（原 dataset_input_wizard.py）
# 使用 Edit 工具进行以下替换：
# - L1717: 'yonod.py' → 'main.py'
# - L1738, L1743: "python yonod.py" → "python main.py"

# 修改 main.py（原 yonod.py）
# - L1-27: docstring 示例更新
# - L375, L702, L705-712: 文件名引用更新
```

#### Step 3：修改文档
```powershell
# README.md：全局替换 + 架构图更新
# CLAUDE.md：入口点说明 + 命令示例更新
# .gitignore：注释更新
```

#### Step 4：验证与测试
```powershell
# 1. 语法检查
python -m py_compile main.py
python -m py_compile yonod.py

# 2. 运行现有测试套件
pytest tests/ -v

# 3. 手动验证三种场景
# 场景 1：新向导流程
python yonod.py

# 场景 2：配置文件模式
python main.py --config project-folder/yonod_config.json

# 场景 3：传统 CLI 模式
python main.py --csv dataset/test-amide-coupling.csv \
  --label-col yield \
  --smiles-cols sub_1_smiles sub_2_smiles activation_id additive_id base_id solvent_id \
  --descriptors morgan --models xgb
```

#### Step 5：提交与文档
```powershell
# 暂存所有修改
git add -A

# 提交（清晰的 commit message）
git commit -m "refactor: 重命名入口文件以明确职责分层

重命名操作：
- yonod.py → main.py（建模核心后端）
- dataset_input_wizard.py → yonod.py（主入口向导）

影响范围：
- 代码文件：yonod.py, main.py（内部引用更新）
- 文档文件：README.md, CLAUDE.md, .gitignore
- 用户界面：所有提示信息和错误消息

向后兼容：
- 传统 CLI 模式从 'python main.py --csv ...' 调用
- 向导模式从 'python yonod.py' 调用
- 配置文件模式保持 '--config' 参数不变

BREAKING CHANGE: 旧版 'python yonod.py --csv ...' 命令需改为 'python main.py --csv ...'
"

# 推送到远程（可选）
git push origin refactor/rename-entry-files
```

---

## 五、替代方案与建议

### 5.1 方案对比

| 方案 | 优点 | 缺点 | 推荐度 |
|------|------|------|--------|
| **方案 1：完全重命名**<br>（本文档描述） | 语义清晰，符合设计哲学 | 破坏向后兼容，用户需适应 | ⭐⭐⭐⭐ |
| **方案 2：保持现状**<br>不重命名 | 零风险，用户无感知 | 文件名语义不够直观 | ⭐⭐⭐ |
| **方案 3：三文件模式**<br>新增 `yonod_wizard.py` | 兼容性最好，三种入口并存 | 增加维护成本，结构冗余 | ⭐⭐ |
| **方案 4：符号链接**<br>`yonod.py` → `main.py` | 技术上向后兼容 | Windows 符号链接权限问题 | ⭐ |

### 5.2 推荐意见

**✅ 推荐执行方案 1（完全重命名）**，原因：

1. **设计哲学一致性**：与 2026-06-29 双入口重构的职责分离理念完全契合
2. **长期维护性**：避免"历史包袱"积累，新用户学习曲线更平滑
3. **文档一致性**：所有文档统一更新后，项目整体专业度提升
4. **用户认知优化**：`yonod.py` = 项目主入口（向导）符合直觉

**但需要做好以下准备**：
1. 在 README 顶部添加醒目的迁移指南（大号提示框）
2. 首次提交后立即更新 GitHub Release Notes
3. 如有论坛/社区，发布迁移公告
4. 考虑保留 `yonod_legacy.py` 兼容脚本 3-6 个月

---

## 六、测试验证清单

### 6.1 功能测试

| 测试场景 | 测试命令 | 预期结果 | 状态 |
|---------|---------|---------|------|
| 向导模式 | `python yonod.py` | 启动 8 步向导，最后调用 main.py | ⏳ |
| 配置文件模式 | `python main.py --config test.json` | 正确读取配置并建模 | ⏳ |
| 传统 CLI 模式 | `python main.py --csv data.csv --label-col yield ...` | 正常执行建模流程 | ⏳ |
| Help 信息 | `python main.py --help` | 无旧文件名残留 | ⏳ |
| 错误提示 | 故意触发错误 | 错误消息中使用正确文件名 | ⏳ |

### 6.2 文档一致性检查

| 检查项 | 方法 | 标准 |
|--------|------|------|
| README 示例正确性 | 手动复制粘贴每个命令 | 所有命令可执行 |
| CLAUDE.md 架构图 | 对比实际文件结构 | 完全匹配 |
| 文档内引用一致 | 全局搜索 `yonod.py` | 无误导性引用 |
| 注释更新完整性 | 代码审查 | 无遗漏 |

---

## 七、回滚计划

如重命名后发现重大问题，可快速回滚：

```powershell
# 方案 A：Git 回滚（推荐）
git checkout backup-before-rename

# 方案 B：手动反向重命名
git mv main.py yonod.py
git mv yonod.py dataset_input_wizard.py
# 然后恢复所有代码和文档修改
git checkout HEAD -- README.md CLAUDE.md .gitignore

# 方案 C：使用功能分支
git checkout main  # 回到主分支
git branch -D refactor/rename-entry-files  # 删除功能分支
```

---

## 八、时间估算

| 阶段 | 预计耗时 | 说明 |
|------|---------|------|
| 文件重命名 | 5 分钟 | `git mv` 操作 |
| 代码修改 | 20 分钟 | 4 个文件共约 15 处修改 |
| 文档更新 | 40 分钟 | README + CLAUDE.md + teaching.md |
| 测试验证 | 30 分钟 | 三种场景 + 测试套件 |
| 提交文档 | 10 分钟 | Commit message + 推送 |
| **总计** | **约 105 分钟** | 建议预留 2 小时 |

---

## 九、总结

### 9.1 核心建议
✅ **推荐执行此重命名**，理由：
1. 符合项目演进方向（向导优先）
2. 提升新用户体验（文件名 = 功能）
3. 代码库结构更清晰（main.py 作为后端核心）

### 9.2 风险缓解
⚠️ **必须做好的 3 件事**：
1. 在 README 顶部添加**醒目的迁移指南**（表格对比旧命令 vs 新命令）
2. 完整测试所有三种使用场景（向导/配置/CLI）
3. 保留 Git 标签 `backup-before-rename` 以便紧急回滚

### 9.3 下一步行动
等待用户确认后，按照 §4.2 详细步骤执行。

---

**文档结束**

如有疑问或需要调整方案，请明确指出。
