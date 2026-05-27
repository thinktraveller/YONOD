# YONOD 项目迁移指南

> 把 YONOD 从本地 Windows 11 迁到云端 Linux 服务器，并维护 GitHub 仓库。
> 写给：项目作者本人，假设了解 Python，对 Linux/Git 有基础认识。
> 适用版本：v1.2.0（2026-05-27）

本文档分三部分：①云端硬件选型与传输；②Linux 适配要点；③GitHub 维护。

---

## 一、迁移到云端服务器

### 1.1 为什么要迁

| 痛点 | 本地 (Win11, 4 核, 16GB RAM) | 云端 (推荐配置) |
|---|---|---|
| RF 在 Morgan/MolMetaLM 全量上 | 单折 7~8 分钟，5 折 38 分钟 | 16 核 + 32GB 单折预计 2~3 分钟 |
| 4×4 完整 grid 总耗时 | 2~4 小时 | 30~60 分钟 |
| 需要熬夜守着脚本 | 是 | 否（提交后断开 SSH 也行） |
| GPU 利用率 | MolMetaLM 阶段 80%、RF 阶段 0% | 同左（除非用 cuML） |

### 1.2 推荐配置

| 资源 | 最低 | 推荐 | 说明 |
|---|---|---|---|
| **CPU** | 8 核 | **16~32 核** | RF 训练是 CPU 瓶颈，核越多 sklearn `n_jobs=-1` 越快 |
| **RAM** | 16 GB | **32 GB** | Morgan 全量 X ≈ 1.15 GB；RF 中间状态再吃几 GB；AutoGluon stack 时还会临时分配 |
| **GPU** | 6 GB 显存 | RTX 3060 / T4 / A10 (8~24 GB) | 仅 MolMetaLM forward 需要；不强求高端卡 |
| **磁盘** | 10 GB | 20 GB | 数据 + 权重 + 缓存 + 结果总计 < 2 GB；预留 swap 余量 |
| **OS** | Ubuntu 22.04 LTS | 同 | 与本项目 Python 3.9 + CUDA 12.1 兼容 |

### 1.3 国内云服务商速查

| 平台 | 适配场景 | 备注 |
|---|---|---|
| **AutoDL** | 学生 / 短时任务首选 | 按小时计费、GPU 列表透明、网盘传文件方便、有 PyTorch 2.1+CUDA 12.1 现成镜像 |
| **阿里云 ECS GPU** | 长期/生产 | 配置灵活，但起步价比 AutoDL 高 |
| **腾讯云 GN7** | 同上 | 同上 |
| **Lambda Labs / Vast.ai** | 海外 | 国内访问慢，需翻墙；价格便宜 |

**对本项目**：AutoDL 一个 RTX 3090 + 16 核 + 32GB 实例（每小时 ~2 元）跑完一次完整 4×4 grid 约 1 小时，成本 2~3 元，性价比最高。

### 1.4 需要上传的资产清单

| 资产 | 本地路径 | 体积 | 必需性 | 备注 |
|---|---|---|---|---|
| 数据集（完整） | `dataset/amide-coupling.csv` | ~10 MB | ✅ 必需 | UTF-8 编码；MIT 协议 |
| 数据集（样本） | `dataset/test-amide-coupling.csv` | < 1 KB | 可选 | 10 行，用于烟测 |
| 核心代码包 | `yonod/` | < 1 MB | ✅ 必需 | 含 4 个 descriptor + 4 个 model + universal + features |
| 统一入口脚本 | `yonod.py` | < 100 KB | ✅ 必需 | 交互向导 + CLI 合并入口 |
| MolMetaLM 权重 | `WEIGHTS/MolMetaLM-base/` | ~500 MB | ⚠️ 用 MolMetaLM 才需 | 可跳过该描述符则不传 |
| FISD 权重 | `WEIGHTS/FISD/` | ~19 MB | ⚠️ 用 FISD 才需 | 3 个 `.pth` 文件 |
| 试剂缓存 | `cache/reagent_feats_*.pkl` | < 5 MB | 可选 | 不传则首次运行重建（多 5~10 秒） |
| 第三方源码 | `化学描述符相关项目/` | 任意 | ❌ 不需要 | 当前实现已内联到 `yonod/`，无需上游源码 |

**总传输量**：~520 MB（含全部权重）或 ~10 MB（只跑 morgan/maccs）。

> **注意**：`dataset/amide-coupling.csv` 已随 GitHub 仓库提供（MIT 协议），
> 云端直接 `git clone` 即可获得，无需单独上传。

### 1.5 上传方式

**方法 A：AutoDL 网盘上传（推荐）**

1. AutoDL 控制台 → 数据盘 → 上传文件（支持拖拽，最大 50GB）
2. 实例启动后，文件位于 `/root/autodl-tmp/`

**方法 B：scp 命令**

```bash
# 在本地 Windows 端打开 PowerShell（或 Git Bash）
scp -r "C:\Users\joyjo\Desktop\其他大学资料\大创\YONOD" root@云端IP:/root/
```

**方法 C：rsync（断点续传，大文件首选）**

```bash
# 仅在 Git Bash / WSL 下可用
rsync -avzP \
    --exclude '.git' \
    --exclude 'results' \
    --exclude 'cache' \
    --exclude '化学描述符相关项目' \
    "/c/Users/joyjo/Desktop/其他大学资料/大创/YONOD/" \
    root@云端IP:/root/YONOD/
```

**方法 D：先发 GitHub，云端 git clone**（适合代码部分；大权重仍要单独传）

```bash
git clone https://github.com/thinktraveller/YONOD.git
cd YONOD
# 然后单独 scp 上传 WEIGHTS/ 目录
```

### 1.6 云端环境搭建

> 假定云端是 Ubuntu 22.04，已预装 conda 或 miniconda。AutoDL 镜像通常自带。

```bash
# 1) 创建 Python 3.9 环境
conda create -n yonod python=3.9 -y
conda activate yonod

# 2) 装 torch（云端有网，走官方 CUDA 12.1 通道，无需离线 whl）
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 \
    --index-url https://download.pytorch.org/whl/cu121

# 3) 验证 GPU
python -c "import torch; print('CUDA:', torch.cuda.is_available(), '| device:', torch.cuda.get_device_name(0))"
# 期望: CUDA: True | device: NVIDIA GeForce RTX 3090 (或同级)

# 4) 化学/ML 主依赖
pip install -r requirements.txt

# 5) AutoGluon（含 lightgbm/catboost；体积较大，耐心等待）
pip install "autogluon.tabular[lightgbm,catboost]==1.1.1"

# 6) 修复 molmetalm.py 的 Windows 默认路径（见 §2.1）
sed -i 's|DEFAULT_WEIGHT_PATH = .*|DEFAULT_WEIGHT_PATH = "/root/YONOD/WEIGHTS/MolMetaLM-base"|' \
    yonod/descriptors/molmetalm.py

# 7) 烟测（10 行，秒级完成）
cd /root/YONOD
python yonod.py \
    --csv dataset/test-amide-coupling.csv \
    --label-col yield \
    --smiles-cols sub_1_smiles sub_2_smiles activation_id additive_id base_id solvent_id \
    --descriptors morgan --models xgb --task-name smoke-test
```

### 1.7 启动全量任务（推荐 tmux 防 SSH 断线）

```bash
# 进入 tmux 会话；断开 SSH 也不会停
tmux new -s yonod

# 在 tmux 里启动（向导模式，全量 4×4 grid）
cd /root/YONOD
conda activate yonod
python yonod.py \
    --csv dataset/amide-coupling.csv \
    --label-col yield \
    --smiles-cols sub_1_smiles sub_2_smiles activation_id additive_id base_id solvent_id \
    --task-name amide-full \
    --heartbeat 60

# 按 Ctrl+B 然后 D 脱离 tmux（保留运行）
# 重新进入: tmux attach -t yonod
```

完成后预计 30~60 分钟（视实例配置），`results/amide-full/` 下会有散点图、metrics_summary.csv、log、report.html。

跳过 MolMetaLM（无权重时）：

```bash
python yonod.py \
    --csv dataset/amide-coupling.csv \
    --label-col yield \
    --smiles-cols sub_1_smiles sub_2_smiles activation_id additive_id base_id solvent_id \
    --descriptors morgan maccs fisd \
    --task-name amide-no-llm \
    --heartbeat 60
```

### 1.8 结果回传

```bash
# 云端打包
cd /root/YONOD/results/amide-full
tar czf yonod_results.tar.gz \
    metrics_summary.csv \
    report.html \
    pictures/ \
    *.log

# 本地下载（在 PowerShell 里）
scp root@云端IP:/root/YONOD/results/amide-full/yonod_results.tar.gz "C:\Users\joyjo\Desktop\"
```

或在 AutoDL 上直接 web 下载。

### 1.9 进阶：GPU 加速 RF（cuML）

> 选做，能让 RF 训练时间从 30 分钟降到 < 5 分钟。

```bash
# 仅 Linux + NVIDIA GPU，且 driver 与 CUDA 12.x 匹配
conda install -c rapidsai -c conda-forge -c nvidia cuml=24.04 python=3.9 cuda-version=12.1
```

然后写一个 `yonod/models/cuml_rf_model.py` 替换 `rf_model.py` 的 sklearn 调用为 `cuml.ensemble.RandomForestRegressor`，并在 `yonod/evaluate.py` 的注册表中添加对应 key。**注意 cuml 在 Windows 上不支持，是迁 Linux 的核心收益之一**。本项目当前未集成此优化，作为后续扩展点。

---

## 二、Linux 适配要点

### 2.1 已知的 Windows-only 代码点

| 文件 | 问题 | 修复 |
|---|---|---|
| `yonod/descriptors/molmetalm.py:28` | `DEFAULT_WEIGHT_PATH = r"C:\Users\joyjo\..."` **硬编码 Windows 路径** | 见 §1.6 第 6 步，用 `sed` 替换为 Linux 绝对路径，或传 `weight_path=` 参数覆盖 |

> `reagent_cache.py` 已用 `Path(__file__).parents[2]` 自动定位，`fisd.py` 默认从 `WEIGHTS/FISD/` 加载，均无跨平台问题。

迁移后快速检查残留 Windows 路径：

```bash
grep -rn "C:" yonod/ yonod.py
# 期望：仅 molmetalm.py:28 一行（修复后应无输出）
```

### 2.2 文件名编码

`dataset/` 目录下的文件名均为英文，不存在跨平台中文路径问题。Ubuntu 默认 UTF-8 环境下直接可用：

```bash
locale | grep -i utf
# 期望看到 LANG=en_US.UTF-8 或 zh_CN.UTF-8
ls dataset/
# 期望：amide-coupling.csv  test-amide-coupling.csv
```

### 2.3 PowerShell ↔ Bash 命令对照

| 操作 | PowerShell (Win) | Bash (Linux) |
|---|---|---|
| 变量赋值 | `$root = "/path"` | `root="/path"` |
| 环境变量 | `$env:HF_TOKEN = "xxx"` | `export HF_TOKEN=xxx` |
| 续行 | `` ` `` (反引号) | `\` (反斜杠) |
| 激活 conda | `conda activate yonod` | 完全一致 |
| 删文件 | `Remove-Item file` | `rm file` |
| 列目录 | `Get-ChildItem` / `ls` | `ls` |
| 路径分隔符 | `\` 或 `/` | 仅 `/` |
| 路径中含空格 | `"..."` 双引号 | 同上 |
| 短路 `&&` | PS 5.1 不支持，要 `; if ($?) {...}` | 原生支持 |

### 2.4 sklearn n_jobs 在 Linux 上的差异

Linux 上 sklearn 用 `fork` 启动 worker（Windows 用 `spawn`），有两个好处：
- worker 启动快 10 倍
- 不需要 pickle 大 X 数组传给 worker（fork 直接继承父进程内存）

所以本地遇到的"Morgan × RF n_jobs=-1 启动延迟"在 Linux 上不会出现。

### 2.5 nohup / tmux / screen

| 工具 | 用途 | 推荐场景 |
|---|---|---|
| `nohup ... &` | 后台运行，标准输出到 nohup.out | 一次性长任务 |
| `tmux` | 会话管理器，可断线重连 | **本项目首选** |
| `screen` | 类似 tmux，更老 | 兼容性最好 |

tmux 速查：

```bash
tmux new -s yonod          # 新建会话
tmux attach -t yonod       # 重连
tmux ls                    # 列所有会话
# 会话内: Ctrl+B 然后 D = 脱离，Ctrl+B 然后 [ = 滚屏
```

---

## 三、GitHub 仓库维护

> 仓库已建立：https://github.com/thinktraveller/YONOD

### 3.1 已入库 vs 未入库资产

| 文件类型 | 入库状态 | 理由 |
|---|---|---|
| `yonod/` 源码 | ✅ 已入库 | 核心交付物 |
| `yonod.py` | ✅ 已入库 | 统一入口 |
| `YONOD项目构建计划书.md` | ✅ 已入库 | 设计文档 |
| `.gitignore` / `LICENSE` / `README.md` / `CHANGELOG.md` / `requirements.txt` | ✅ 已入库 | 标配 |
| `dataset/amide-coupling.csv`（47015 条） | ✅ 已入库 | MIT 协议，来自 aichemeco/amide_coupling |
| `dataset/test-amide-coupling.csv`（10 条） | ✅ 已入库 | 同上，用于调试 |
| `MIGRATION.md`（本文件） | ❌ gitignored | 含内部部署细节，不公开分发 |
| `WEIGHTS/MolMetaLM-base/` (~500 MB) | ❌ gitignored | 超 GitHub 单文件 100 MB 限制；从 HuggingFace 单独下载 |
| `WEIGHTS/FISD/` (~19 MB) | ❌ gitignored | 无明确许可证；从上游 KeantChen/FISD 获取 |
| `cache/`, `results/`, `*.log` | ❌ gitignored | 运行产物 |
| `化学描述符相关项目/` | ❌ gitignored | 第三方源码，有各自 LICENSE |
| `__pycache__/`, `.venv/` | ❌ gitignored | Python/conda 产物 |

### 3.2 日常推送流程

```bash
cd "C:\Users\joyjo\Desktop\其他大学资料\大创\YONOD"

# 1) 确认变更
git status
git diff

# 2) 暂存并提交（按语义化版本规范写 message）
git add yonod/ yonod.py README.md CHANGELOG.md requirements.txt .gitignore
git commit -m "feat: <简要描述>"

# 3) 推送到 GitHub
git push origin main
```

### 3.3 提交后检查清单

```bash
# 仓库大小（应 < 50 MB；超过说明误入了大文件）
git count-objects -vH

# 检查是否误入了不该入库的文件
git ls-files | grep -E "WEIGHTS|cache|results|\.pth$|\.safetensors$|MIGRATION\.md"
# 期望: 无输出。任何输出都是漏掉的 .gitignore 规则
```

### 3.4 大文件用 Git LFS（如要把权重也入库）

> 不推荐除非必要。建议把权重链接放在 README 里指向 HuggingFace。

```bash
# 装 git-lfs（一次）
# Win: 从 https://git-lfs.com 下载安装
# Ubuntu: sudo apt install git-lfs

git lfs install
git lfs track "*.safetensors" "*.pth"
git add .gitattributes
git commit -m "chore: track large model weights via LFS"
# 然后正常 git add WEIGHTS/  并 push
```

GitHub 免费 LFS 配额 1 GB/月，超出收费；权重 500 MB 入库会很快用完。**强烈建议不入库**。

### 3.5 安全提醒

| 风险 | 应对 |
|---|---|
| HuggingFace token 写进代码 | 用 `.env` 文件（`.gitignore` 已忽略） |
| 内部部署细节（服务器 IP、路径）泄露 | `MIGRATION.md` 已 gitignore，不入库 |
| 第三方项目源码版权 | `.gitignore` 排除 `化学描述符相关项目/`，README 中链接原仓库 |
| 中间结果含个人路径 | `results/*.log` 已 ignored；若手动入库 log，注意脱敏 |
| commit message 暴露内部信息 | 避免在 message 里贴 IP / 密码 / 实验数据 |

---

## 四、整合工作流（推荐顺序）

1. **本地验证**
   - `python yonod.py --csv dataset/test-amide-coupling.csv ...` 烟测通过
2. **代码推送到 GitHub**
   - 按 §3.2 流程 push
3. **云端部署**
   - `git clone https://github.com/thinktraveller/YONOD.git`
   - 单独上传 `WEIGHTS/`（AutoDL 网盘或 scp）
   - 按 §1.6 装环境，**务必执行第 6 步修复 `molmetalm.py` 路径**
   - 烟测通过后 tmux + 全量 run，~1 小时
4. **回传结果**
   - 打包 `results/` 下载
   - 把 `report.html` 发给老师/团队
5. **结果回写仓库**（选做）
   - 把 `results/<task>/metrics_summary.csv` 入库作为基线快照
   - PNG 散点图体积较大，酌情不入库

---

## 五、常见坑（项目历史踩过的）

| 坑 | 触发场景 | 修复 |
|---|---|---|
| PowerShell 不支持 `&&` | 直接抄了 bash 命令 | 用 `; if ($?) { ... }` |
| AutoGluon `fastai` 链上的 spacy 要求 Python 3.10+ | `pip install autogluon[fastai]` 在 py3.9 上 | 只装 `[lightgbm,catboost]`，放弃 fastai |
| RF 默认 `verbose=0` 像卡死 | 全量 + 高维 + 300 树 | 加 `--heartbeat 60`，本项目已支持 |
| RBF SVR 在 n>10k 时 O(n²) 爆炸 | 47015 行单折 75+ 分钟未完 | `--svm-subsample 8000`（默认已设） |
| 试剂列里逗号 `,` 不是 RDKit 多片段分隔符 | `MolFromSmiles("A,B")` 返回 None | `yonod.py` 已自动替换 `,` → `.` |
| MolMetaLM 是 Llama 架构，不接受 `token_type_ids` | tokenizer 默认返回了它 | `molmetalm.py` 已过滤参数白名单 |
| `molmetalm.py` 默认路径为 Windows 绝对路径 | 直接在 Linux 上运行 | §1.6 第 6 步 sed 替换，或传 `weight_path=` |
| Win11 sklearn n_jobs=-1 的 pickle 启动延迟 | Win 用 spawn 而非 fork | Linux 上不存在；本地可在代码里临时改 `n_jobs=1` |
| `R^2` 标注渲染成字面量 | matplotlib 默认不解析 caret | 用 mathtext `$R^2$`（本项目已修复） |

---

## 六、长期演进建议

| 方向 | 改动量 | 收益 |
|---|---|---|
| 引入 cuML RF | 中（写 `yonod/models/cuml_rf_model.py` + 注册到 evaluate.py） | RF 训练 30 min → 3 min |
| MolMetaLM fine-tune | 大（需 yield 标签 + 训练循环） | MolMetaLM R² 可能从 0.70 提到 0.85+ |
| 描述符特征磁盘缓存 | 小 | 第二次跑同数据 < 5 秒 |
| 加 Linear / Lasso / Ridge 基线 | 小（注册新 model class） | 学术汇报常要"线性模型对照" |
| 加分类任务支持 | 中（pass / fail 二分类） | 适配产率阈值切分场景 |
| 把 `yonod.py` 改成 CLI 包 | 中（pyproject.toml + console_scripts） | `pip install yonod` 后命令行直用 |

---

*文档完。如有遗漏请补充。*
