# YONOD 项目迁移指南

> 把 YONOD 从本地 Windows 11 迁到云端 Linux 服务器，并发布到 GitHub。
> 写给：项目作者本人，假设了解 Python，对 Linux/Git 有基础认识。
> 适用版本：v0.3.2（2026-05-16）

本文档分三部分：①云端硬件选型与传输；②Linux 适配要点；③GitHub 发布。

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
| **RAM** | 16 GB | **32 GB** | Morgan 全量 X = 1.15 GB；RF 中间状态再吃几 GB；AutoGluon stack 时还会临时分配 |
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
| 数据集 | `数据集/酰胺缩合反应数据集.csv` | ~10 MB | ✅ 必需 | UTF-8 编码 |
| 主代码包 | `yonod_yield/` | < 1 MB | ✅ 必需 | 含 4 个 descriptor + 4 个 model + features + evaluate + plot |
| 入口脚本 | `run_yield_prediction.py` 等 4 个根脚本 | < 100 KB | ✅ 必需 | run / generate_report / verify_morgan_rf / test_run_yield |
| MolMetaLM 权重 | `WEIGHTS/MolMetaLM-base/` | ~500 MB | ⚠️ 用 MolMetaLM 才需 | 可不传则跳过该描述符 |
| FISD 模型 | `化学描述符相关项目/FISD/model/qm_9_*.pth` (3 个) | ~100 MB | ⚠️ 用 FISD 才需 | 同上 |
| 试剂缓存 | `cache/reagent_feats_*.pkl` | < 5 MB | 可选 | 不传则首次运行重建（多 5~10 秒） |
| 试剂映射表 | `数据集/试剂编号-SMILES对应表.md` | < 10 KB | ❌ 不需要 | CSV 已含 SMILES，仅作人类参考 |
| ATMOMACCS 源码 | `化学描述符相关项目/ATMOMACCS/` | < 1 MB | ❌ 不需要 | 当前实现直接用 RDKit，未依赖该项目 |
| HSPOC / DFT 项目 | `化学描述符相关项目/{HSPOC,DFTDescriptorPipeline}/` | 任意 | ❌ 不需要 | v0.3 已暂缓 |

**总传输量**：~620 MB（含全部权重）或 ~10 MB（只跑 Morgan/ATMOMACCS）。

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
    --exclude '化学描述符相关项目/HSPOC' \
    --exclude '化学描述符相关项目/DFTDescriptorPipeline' \
    "/c/Users/joyjo/Desktop/其他大学资料/大创/YONOD/" \
    root@云端IP:/root/YONOD/
```

**方法 D：先发 GitHub，云端 git clone**（适合代码部分；大权重仍要单独传）

见 §三。

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
pip install rdkit==2023.9.5 transformers==4.40.0 networkx==3.2.1
pip install scikit-learn==1.4.0 xgboost==2.0.3
pip install pandas==2.0.3 numpy==1.26.4 matplotlib==3.8.4
pip install joblib==1.3.2 pyyaml==6.0.1 tqdm==4.66.2

# 5) FISD 用的 PyTorch Geometric
pip install torch_geometric==2.5.3

# 6) AutoGluon（含 lightgbm/catboost）
pip install "autogluon.tabular[lightgbm,catboost]==1.1.1"

# 7) 烟测一遍主脚本
cd /root/YONOD
python test_run_yield.py
```

### 1.7 启动全量任务（推荐 nohup + tmux 防 SSH 断线）

```bash
# 进入 tmux 会话；断开 SSH 也不会停
tmux new -s yonod

# 在 tmux 里启动
cd /root/YONOD
conda activate yonod
python run_yield_prediction.py \
    --svm-subsample 8000 \
    --rf-verbose 1 \
    --heartbeat 60 \
    --log-file results/run_cloud_full.log

# 按 Ctrl+B 然后 D 脱离 tmux（保留运行）
# 重新进入: tmux attach -t yonod
```

完成后预计 30~60 分钟（视实例配置），`results/` 下会有 16 张 PNG、metrics_summary.csv、log、report.html。

### 1.8 结果回传

```bash
# 云端打包
cd /root/YONOD/results
tar czf yonod_results.tar.gz \
    metrics_summary.csv \
    scatter_*.png \
    report.html \
    *.log

# 本地下载（在 PowerShell 里）
scp root@云端IP:/root/YONOD/results/yonod_results.tar.gz "C:\Users\joyjo\Desktop\"
```

或在 AutoDL 上直接 web 下载。

### 1.9 进阶：GPU 加速 RF（cuML）

> 选做，能让 RF 训练时间从 30 分钟降到 < 5 分钟。

```bash
# 仅 Linux + NVIDIA GPU，且 driver 与 CUDA 12.x 匹配
conda install -c rapidsai -c conda-forge -c nvidia cuml=24.04 python=3.9 cuda-version=12.1
```

然后写一个 `cuml_rf_model.py` 替换 `rf_model.py` 的 sklearn 调用为 `cuml.ensemble.RandomForestRegressor`。**注意 cuml 在 Windows 上不支持，是迁 Linux 的核心收益之一**。本项目当前未集成此优化，作为后续扩展点。

---

## 二、Linux 适配要点

### 2.1 已知的 Windows-only 代码点

| 文件 | 问题 | 修复 |
|---|---|---|
| `molmetalm.py` | `DEFAULT_WEIGHT_PATH = r"C:\Users\..."` | 用环境变量 `YONOD_WEIGHTS` 或运行时传 `weight_path=` |
| `verify_morgan_rf.py` | `DEFAULT_CSV = ROOT / "数据集" / "..."` | `ROOT` 用 `Path(__file__).resolve().parent`，相对路径会自动适配 |
| `fisd.py` | `DEFAULT_MODEL_DIR = ROOT / "化学描述符相关项目" / "FISD" / "model"` | 同上，相对路径 OK |
| 各 powershell 文档示例 | `$root = "..."` / 反引号续行 | 见 §2.3 命令对照 |

实际只有 `molmetalm.py` 的默认路径需要改。`reagent_cache.py` 已经用 `Path(__file__).parents[2]` 自动定位，跨平台无忧。建议在云端 deploy 前做一次：

```bash
# 把 molmetalm.py 默认路径改为环境变量
sed -i 's|DEFAULT_WEIGHT_PATH = r"C:\\\\.*|DEFAULT_WEIGHT_PATH = os.environ.get("YONOD_WEIGHTS", "/root/YONOD/WEIGHTS/MolMetaLM-base")|' \
    yonod_yield/descriptors/molmetalm.py

# 或更简单：保持默认路径硬编码 Linux 路径
```

更稳妥的做法：迁移后跑一次 `grep -rn "C:" yonod_yield/ run_yield_prediction.py verify_morgan_rf.py` 检查所有残留的 Windows 路径。

### 2.2 文件名编码

数据集和映射表文件名含中文，确保用 **UTF-8 文件系统**（Ubuntu 默认 OK）。AutoDL/阿里云的 Ubuntu 镜像都默认 zh_CN.UTF-8 或 en_US.UTF-8，都行。可以验证：

```bash
locale | grep -i utf
# 期望看到 LANG=en_US.UTF-8 或 zh_CN.UTF-8
ls 数据集/
# 期望看到中文文件名正常显示
```

scp/rsync 传输时使用 OpenSSH 默认配置即可保持 UTF-8。若用某些图形 SCP 客户端（如老版本 WinSCP）可能会 mangle 中文，建议用命令行 scp。

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

PowerShell 命令直接复制到 Bash 一般跑不动，但语义近似，按表照搬即可。

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

## 三、发布到 GitHub

### 3.1 仓库要不要传整个项目？

| 文件类型 | 入库？ | 理由 |
|---|---|---|
| `yonod_yield/` 源码 | ✅ | 核心交付物 |
| 根脚本（run/test/verify/generate_report） | ✅ | 同上 |
| `YONOD项目构建计划书.md` / `MIGRATION.md` | ✅ | 文档 |
| `.gitignore` / `LICENSE` / `README.md` | ✅ | 标配 |
| `数据集/酰胺缩合反应数据集.csv` | ⚠️ 取决于来源 | 若数据集来自他人发表/未授权，**不入库**；若自己生成有权发布，可入库 |
| `WEIGHTS/MolMetaLM-base/` (500 MB) | ❌ | 超 GitHub 单文件 100 MB 限制；用 LFS 或外部链接 |
| `化学描述符相关项目/FISD/model/*.pth` (~100 MB) | ❌ | 同上 |
| `化学描述符相关项目/` 其他子项目 | ⚠️ | 取决于其 LICENSE。ATMOMACCS / FISD / MolMetaLM 都是第三方仓库，建议 **不入库**，写在 README 里指向上游 |
| `cache/`, `results/`, `*.log` | ❌ | 运行产物 |
| `__pycache__/`, `.venv/` | ❌ | Python/conda 产物 |

### 3.2 .gitignore（建议内容）

```gitignore
# Python
__pycache__/
*.py[cod]
*.egg-info/
.venv/
.conda/

# 运行产物
cache/
results/
*.log
AutogluonModels/

# 模型权重和大数据
WEIGHTS/
化学描述符相关项目/FISD/model/*.pth
化学描述符相关项目/FISD/test_MLMS/*.npy
*.safetensors

# 数据（若不打算公开数据集）
数据集/*.csv

# 第三方项目（避免重复维护）
化学描述符相关项目/

# 凭据
.env
.env.*
*.token

# IDE
.vscode/
.idea/
.DS_Store
```

如果决定**公开数据集**，从上面 `数据集/*.csv` 删除即可。

### 3.3 README.md 建议结构

```markdown
# YONOD: Yield prediction for amide condensation reactions

5-fold CV comparison of 4 molecular descriptors × 4 ML algorithms on
47015 amide-bond-formation reactions.

## Quick start

```bash
conda create -n yonod python=3.9 -y && conda activate yonod
pip install -r requirements.txt
python test_run_yield.py            # smoke test
python run_yield_prediction.py      # full 4x4 grid (~1 hour on 16 cores + GPU)
```

## Results (R² on 5-fold CV)

| Descriptor | XGB | RF | SVM | AutoGluon |
|---|---|---|---|---|
| Morgan ECFP4 | 0.73 | 0.85 | 0.70 | **0.87** |
| ATMOMACCS | 0.74 | ... | ... | ... |
| ...

Full report: `results/report.html`

## Reproduce

Dependencies:
- Python 3.9, CUDA 12.1, torch 2.1.2
- See `MIGRATION.md` for cloud deployment guide.

External assets (not in repo):
- Dataset: place at `数据集/酰胺缩合反应数据集.csv`
- MolMetaLM weights: download from [HuggingFace wudejian789/MolMetaLM-base](https://huggingface.co/wudejian789/MolMetaLM-base) to `WEIGHTS/MolMetaLM-base/`
- FISD pretrained weights: see [上游仓库](...)

## Citation

If you use this code, please cite ...

## License

MIT (or whichever you choose).
```

### 3.4 requirements.txt

从当前 conda env 导出：

```bash
conda activate yonod
pip freeze | grep -Ev "^(autogluon|lightgbm|catboost)" > requirements.txt
# 然后手工把 autogluon.tabular[lightgbm,catboost]==1.1.1 加到末尾
```

或者干脆手写一份精简版：

```text
torch==2.1.2
torchvision==0.16.2
torchaudio==2.1.2
torch_geometric==2.5.3
rdkit==2023.9.5
transformers==4.40.0
networkx==3.2.1
scikit-learn==1.4.0
xgboost==2.0.3
autogluon.tabular[lightgbm,catboost]==1.1.1
pandas==2.0.3
numpy==1.26.4
matplotlib==3.8.4
joblib==1.3.2
pyyaml==6.0.1
tqdm==4.66.2
```

### 3.5 第一次 push 流程

```bash
cd "C:\Users\joyjo\Desktop\其他大学资料\大创\YONOD"   # 或 Linux 上 cd /root/YONOD

# 1) 初始化（如果还没 git init）
git init
git branch -m main

# 2) 配置 user
git config user.name "你的名字"
git config user.email "你的邮箱"

# 3) 添加 .gitignore（贴上面的内容）
# 然后:
git add .gitignore
git commit -m "chore: add .gitignore"

# 4) 加 README.md / LICENSE / MIGRATION.md / requirements.txt
git add README.md LICENSE MIGRATION.md requirements.txt
git commit -m "docs: project README + migration guide"

# 5) 加源码和文档
git add yonod_yield/ run_yield_prediction.py generate_report.py test_run_yield.py verify_morgan_rf.py
git add YONOD项目构建计划书.md
git commit -m "feat: initial YONOD v0.3.2 source"

# 6) 在 GitHub 创建空仓库（网页操作）：
#    https://github.com/new  → 仓库名比如 yonod  → 创建（不要勾选 README/gitignore，已经有了）

# 7) 关联远端 + 推送
git remote add origin https://github.com/你的用户名/yonod.git
git push -u origin main
```

### 3.6 大文件用 Git LFS（如要把权重也入库）

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

### 3.7 安全提醒

| 风险 | 应对 |
|---|---|
| HuggingFace token 写进代码 | 用 `.env` 文件（`.gitignore` 已忽略） |
| 数据集含课题组未发表数据 | `.gitignore` 排除 `数据集/*.csv`，README 写"获取方式见 ..." |
| 第三方项目源码版权 | `.gitignore` 排除 `化学描述符相关项目/`，README 中链接原仓库 |
| 中间结果含个人路径 | `results/*.log` 已 ignored；若手动入库 log，注意脱敏 |
| commit message 暴露内部信息 | 避免在 message 里贴 IP / 密码 / 实验数据 |

### 3.8 commit 提交后检查清单

```bash
# 仓库大小（应 < 50 MB；超过说明误入了大文件）
git count-objects -vH

# 列举所有被追踪的大文件（> 1 MB）
git ls-files | xargs -I{} du -b "{}" 2>/dev/null | sort -rn | head -20

# 检查是否漏掉了不该入库的目录
git ls-files | grep -E "WEIGHTS|cache|results|\.pth$|\.safetensors$"
# 期望: 无输出。任何输出都是漏掉的 .gitignore 规则
```

---

## 四、整合工作流（推荐顺序）

1. **本地清理与验证**
   - `python test_run_yield.py` 一次 → 确认本地能跑通
   - `python generate_report.py` 一次 → 确认报告生成正常
2. **GitHub 仓库**（先发代码不发权重，体积小、迭代快）
   - 按 §3 流程 push
   - 同时把 README 里的"下载权重"链接写好
3. **云端部署**
   - 实例上 `git clone <你的仓库>`
   - 单独 scp 上传 `数据集/` + `WEIGHTS/` + `化学描述符相关项目/FISD/model/`
   - 按 §1.6 装环境
   - `python test_run_yield.py` 烟测
   - tmux + 全量 run，~1 小时
4. **回传结果**
   - 打包 `results/` 下载
   - 把 `report.html` 发给老师/团队
5. **结果回写仓库**（选做）
   - 把 `results/metrics_summary.csv` 入库作为基线快照
   - PNG 散点图体积大可不入库

---

## 五、常见坑（项目历史踩过的）

| 坑 | 触发场景 | 修复 |
|---|---|---|
| PowerShell 不支持 `&&` | 直接抄了 bash 命令 | 用 `; if ($?) { ... }` |
| AutoGluon `fastai` 链上的 spacy 要求 Python 3.10+ | `pip install autogluon[fastai]` 在 py3.9 上 | 只装 `[lightgbm,catboost]`，放弃 fastai |
| RF 默认 `verbose=0` 像卡死 | 全量 + 6144 维 + 300 树 | 加 `verbose=1` 或心跳，本项目已实现 |
| RBF SVR 在 n>10k 时 O(n²) 爆炸 | 47015 行单折 75+ 分钟未完 | 子采样 `subsample_n=8000` |
| 试剂列里逗号 `,` 不是 RDKit 多片段分隔符 | `MolFromSmiles("A,B")` 返回 None | 预处理替换 `,` → `.` |
| MolMetaLM 是 Llama 架构，不接受 `token_type_ids` | tokenizer 默认返回了它 | 过滤参数白名单 `{input_ids, attention_mask, position_ids}` |
| Win11 sklearn n_jobs=-1 的 pickle 启动延迟 | Win 用 spawn 而非 fork | Linux 上不存在；本地可改 `n_jobs=1` 临时规避 |
| `R^2` 标注渲染成字面量 | matplotlib 默认不解析 caret | 用 mathtext `$R^2$` |

---

## 六、长期演进建议

| 方向 | 改动量 | 收益 |
|---|---|---|
| 引入 cuML RF | 中（写 cuml_rf_model.py + 注册到 evaluate.py） | RF 训练 30 min → 3 min |
| MolMetaLM fine-tune | 大（需 yield 标签 + 训练循环） | MolMetaLM R² 可能从 0.70 提到 0.85+ |
| 描述符特征磁盘缓存 | 小（已规划为 T5.1） | 第二次跑同数据 < 5 秒 |
| 加 Linear / Lasso / Ridge 基线 | 小（注册新 model class） | 学术汇报常要"线性模型对照" |
| 加分类任务支持 | 中（pass / fail 二分类） | 适配产率阈值切分场景 |
| 把 `run_yield_prediction.py` 改成 CLI 包 | 中（pyproject.toml + console_scripts） | `pip install yonod` 后命令行直用 |

---

*文档完。如有遗漏请补充。*
