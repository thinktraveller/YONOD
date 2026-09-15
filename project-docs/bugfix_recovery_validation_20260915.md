# 恢复后测试故障修复（2026-09-15）

## 结果

全量回归共 143 项：138 项通过，5 项按配置跳过，0 失败、0 错误，耗时 36.567 秒。日志：`logs/bugfix_full_unittest_20260915.log`。原有两个全量建模任务保持暂停，本轮没有恢复执行它们。

本轮采用 bug-detective 的证据排查流程，区分测试接口过期、进程导入状态污染和实际运行缺陷，再通过针对测试与完整回归验证。

## 修复内容

- 两项旧 JSON 测试迁移到独立 schema-2 YAML，保留真实外层折划分、AutoGluon 参数传递、完整 OOF 与 OHE 训练折拟合验证；新增旧 JSON 被拒绝及迁移提示验证。未重新开放 JSON 运行配置入口。
- 特征服务的两项导入隔离测试在全新 Python 进程中运行，继续检查不导入模型/OHE 模块，避免全量 discover 中其他测试的 `sys.modules` 状态导致误报。
- benchmark 的 schema-2 估计器参数快照兼容原有顶层字段，同时保留嵌套工厂审计信息；`max_features` 和 `model_random_seed` 可被既有消费者正确读取。
- 修复论文 RF 协议的逐折随机种子未传入 schema-2 估计器的问题。仅在论文协议声明种子时绑定，拒绝冲突的显式种子，不修改调用者配置；普通任务保留其原始种子行为。新增回归检查，并在两种特征各 25 折的测试中核对实际估计器快照。
- 补齐 benchmark Tukey HSD 报告依赖：`requirements.txt` 声明 `statsmodels>=0.14,<0.15`；当前 yonod 环境安装 `statsmodels 0.14.6`、`patsy 1.0.3`，其他现有依赖保持原版本。安装日志：`logs/bugfix_dependencies_20260915.log`。
- `yonod.py` 的 YAML 路径传播普通建模子进程和 strict benchmark 的返回码；无效配置、配置加载异常和启动异常返回 1，脚本入口使用 `sys.exit(main())`。CSV 交互原有 `None` 返回值仍映射成功退出。7 项测试覆盖这些状态。

## 验证

| 验证 | 结果 |
| --- | --- |
| 两个迁移后的主入口测试文件 | 6 项通过 |
| schema-2 测试 | 44 项通过 |
| 论文协议端到端测试 | 两种特征各 25 折完成，报告重建通过 |
| YAML 向导退出码 | 7 项通过 |
| 全量 unittest discover | 143 项中 138 通过、5 跳过，无失败 |

5 个跳过项为默认关闭的 4 项 launcher 集成测试和 1 项真实 AutoGluon smoke。它们需要显式设置相应环境变量，本轮未启用；Tukey 检验测试现已实际执行通过。

复现全量回归（Linux Bash，仓库根目录）：

```bash
source /home/wangzh685/miniconda3/etc/profile.d/conda.sh
conda activate yonod
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m unittest discover -s tests -p 'test_*.py' -v
```

历史 benchmark 已落盘结果没有被改写，也不会因本次代码修复自动重新训练。需要重新验证历史论文协议结果时，应使用独立配置与新输出目录。
