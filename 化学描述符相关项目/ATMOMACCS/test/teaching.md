# [归档] ATMOMACCS 项目教学文档

> 归档说明：本文件为 test 目录实验副本，不再作为主文档维护。
> 最新版本请查看主工作目录：`teaching.md`。

## 1. 项目概览
ATMOMACCS 是一个面向大气有机分子性质预测的机器学习科研项目，核心目标是：
1. 构建和生成分子描述符（尤其是 ATMOMACCS 系列）。
2. 使用核岭回归（KRR）预测目标理化性质。
3. 用 SHAP 做可解释性分析，解释描述符对预测的贡献。

它支持多个数据集（如 Wang、GeckoQ、Ferraz-Caetano、Li），可以用于对比不同描述符的效果，并输出学习曲线、预测结果、均值和标准差统计文件。

## 2. 项目可以做什么（用户最关心）
这个项目可以用于：
1. 从 SMILES/分子数据生成多类描述符（ATMOMACCS、MACCS、TopFP）。
2. 训练 KRR 模型预测目标性质，例如 log_p_sat、log_kwg、log_kwiomg、dvap、tg。
3. 在固定测试集上评估模型性能（MAE、MSE、R2、AAVRE）。
4. 做不同训练样本规模下的学习曲线分析（观察数据量提升带来的误差变化）。
5. 多随机种子重复实验并聚合统计（均值/标准差），用于更稳健地比较方法。
6. 对训练好的模型做 SHAP 分析，解释哪些描述符维度最重要。
7. 结合 notebooks 和 visualization 脚本，生成论文/汇报可用图表。

## 3. 目录与架构

### 3.1 核心目录
- src/main.py: 统一 CLI 入口，串联“描述符生成 -> KRR 训练/预测”。
- src/model/: 描述符构建、KRR、SHAP 等核心算法代码。
- src/visualization/: 结果聚合和绘图分析脚本。
- src/slurm/: 批处理脚本，便于在集群上批量跑实验。
- data/: 四个数据集及其各自输出目录（descriptors、KRR_output、pickle、shap、plots）。
- notebooks/: 分析与可视化 notebook。

### 3.2 简化流程图
```mermaid
flowchart LR
    A[输入数据和目标文件] --> B[生成描述符 ATMOMACCS/MACCS/TopFP]
    B --> C[KRR 网格搜索训练]
    C --> D[预测与指标输出]
    C --> E[保存训练模型 pickle]
    E --> F[SHAP 解释]
    D --> G[结果聚合 mean/std]
    F --> H[可解释性图与分析]
```

## 4. 关键文件作用
1. src/main.py
- 命令行入口。
- 根据参数选择描述符并触发生成，然后调用 KRR 主流程。

2. src/model/krr.py
- 训练与评估核心。
- 使用 GridSearchCV 对 KernelRidge 的 alpha/gamma 做网格搜索。
- 输出 result CSV、预测值 CSV、学习曲线图，并持久化最佳模型。

3. src/model/generate_ATMOMACCS.py 与 ATMOMACCS*.py
- 定义并生成 ATMOMACCS 描述符（不同版本）。

4. src/model/shap_vals.py
- 加载 pickled 训练模型和测试集，计算并保存 SHAP 值。

5. src/visualization/process_results.py
- 汇总多随机种子结果，生成 mean/std 文件，支持稳健比较。

6. notebooks/*.ipynb
- 数据分析、SHAP 展示、性能提升分析等科研复现内容。

## 5. 运行前准备
1. Python 3.12 环境（README 指定）。
2. 安装依赖（matplotlib、pandas、rdkit、scikit-learn、scipy、shap 等）。
3. 确认目标数据集目录下有必要文件：
- smiles.txt
- 目标值文件（如 log_p_sat.txt）
- descriptors 目录（可由脚本自动生成）

## 6. 典型运行方式
示例（Wang 数据集，ATMOMACCS v4，目标 log_p_sat）：

```bash
python src/main.py -v 4 -d data/Wang -t log_p_sat.txt -ds Wang -s 2435
```

执行后通常会在数据集目录产生：
1. descriptors/: 描述符文件。
2. KRR_output/results/: 每次训练规模的性能结果 CSV。
3. KRR_output/output_predictions/: 测试集预测值。
4. pickle/: 训练好的模型。
5. shap/: SHAP 输入与输出文件。
6. plots/: 学习曲线或其他图表。

## 7. 部署/执行建议
该项目更偏科研脚本与实验流水线，不是 Web 服务部署型项目。推荐三种运行环境：
1. 本地开发机：用于单次调参与分析。
2. Jupyter 环境：用于结果可视化和论文图绘制。
3. Linux/集群环境：配合 src/slurm 脚本做批量实验。

验证成功的标准：
1. 目标数据集目录下生成 descriptors、KRR_output、pickle、shap、plots。
2. result_*.csv 和 mean/std 文件内容完整。
3. notebook 可读取输出文件并画图。

## 8. 常见问题排查
1. 报错找不到目标文件：检查 -d 与 -t 是否匹配数据集路径。
2. SHAP 运行很慢：先使用 samples 或 kmeans 参数，或后处理时再单独跑 shap_vals.py。
3. 结果目录为空：确认随机种子、数据格式和描述符文件是否正确生成。
4. RDKit 安装失败：优先使用 conda 环境安装，避免纯 pip 在部分平台的兼容问题。

## 9. 对话记录
### 用户问题
- 告诉我这个项目可以干什么

### 回答摘要
- 本项目可以生成大气化学相关分子描述符，训练 KRR 预测多个理化性质，输出多指标评估结果，支持多随机种子统计与学习曲线分析，并可通过 SHAP 完成可解释性研究。
- 项目定位是科研建模与分析流水线，适合复现实验、比较描述符与制作可视化图表。

### 用户问题
- 数据文件可以放在哪里以发挥作用

### 回答摘要
- 最推荐将数据按 data/<数据集名>/... 的结构放在项目根目录下，这样与描述符生成脚本中的硬编码路径完全一致。
- 数据集名需与代码一致：Wang、GeckoQ、Ferraz-Caetano、Li。
- 每个数据集目录至少包含 smiles.txt、目标文件（如 log_p_sat.txt）和 descriptors 子目录。
- 训练结果会自动写入该数据集目录下的 KRR_output、pickle、shap、plots 等子目录。
