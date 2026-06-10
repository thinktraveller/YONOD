# 构建日志

## [2026-06-10] 构建记录

### 执行的任务
- 读取项目构建计划书（项目构建计划书.md），理解 9 个反应体系的描述符性能分析框架
- 逐一读取 10 个体系（含 Cu 的两个子任务）的真实实验数据：
  - result/*/metrics_summary.csv（各体系 4 描述符 × 4 模型的 R²、RMSE、MAE 等指标）
- 基于真实数值生成完整 HTML 可视化报告：descriptor_analysis_report.html

### 关键变更
- 新增文件：`descriptor_analysis_report.html`（约 500 行，内联 CSS + 纯 JS，无外部依赖）
  - 现代深色导航栏 + 白色内容区响应式布局
  - 全局概览统计卡片（6 项核心指标）
  - 全局汇总表（10 个建模任务对比，含 R²进度条、性能徽章）
  - 10 张体系详情卡片（每张包含最优组合标签、4 描述符完整对比表、诊断摘要）
  - 横向规律总结（6 条规律卡片 + 预测目标难度对比表）
  - 10 条改进建议卡片（含优先级标注、具体操作步骤）
  - 最终推荐方案汇总表
- 新增文件：`buildlog.md`

### 遇到的问题及解决方案
- Cu 体系有两个子目录（acceptor_HTE 和 donor_acceptor_HTE），分别读取各自的 metrics_summary.csv
- DHP 体系的 maccs SVM 结果缺失（CSV 中无对应行），报告中标注为 "—"
- Cu-acceptor_HTE 的 morgan SVM 结果缺失，同样标注为 "—"

### 数据来源说明
所有 R²、RMSE、MAE 数值均直接来自 result/*/metrics_summary.csv，未经人工编造或估算。

---
