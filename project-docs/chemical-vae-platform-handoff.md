# Chemical VAE C11 平台交接

截至本文件更新，Linux CPU、Linux GPU（`cuda:0`）的真实 ZINC/v5
`features -> train` 路径均已有证据；`zinc_properties` encoder 也有独立的
TensorFlow/NumPy/PyTorch 数值对照和 Linux GPU smoke。**Windows CPU 仍为
`not_verified`**：Linux 的路径兼容性测试、YAML 校验或本交接配置都不能替代
一台真实 Windows 主机的执行。用户已授权将步骤31标记完成并暂时延期这项验证；
延期不改变本文件中 `not_verified` 的事实状态。

## Windows CPU 执行者步骤

在 Windows 10/11 实机或 Windows CI worker 的仓库根目录完成以下事项：

1. 使用项目支持的 Python 3.9 `yonod` Conda 环境，并确认 PyTorch CPU、RDKit、
   scikit-learn 和已提交的 `WEIGHTS/chemical_vae/zinc-37e96cd3bc8f9680/v5/`
   都可读。常规 Chemical VAE 推理不需要 TensorFlow；TensorFlow 仅用于另一套
   Linux 已完成的 parity 协议。
2. 不改 `example.yaml`，直接使用独立 schema-2 输入
   [`step31_9_zinc_v5_windows_cpu_all.yaml`](../configs/chemical_vae/step31_9_zinc_v5_windows_cpu_all.yaml)。
   它使用相对路径、自己的 artifact/output root 和 `device: cpu`。
3. 在 **Anaconda Prompt** 或 PowerShell 中通过公共向导入口运行，并保留完整日志：

   ```powershell
   conda activate yonod
   New-Item -ItemType Directory -Force logs | Out-Null
   $config = "configs/chemical_vae/step31_9_zinc_v5_windows_cpu_all.yaml"
   $config | python -u yonod.py *>&1 | Tee-Object -FilePath "logs/step31_9_zinc_v5_windows_cpu_all.log"
   if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
   ```

4. 保存该日志、`feature_runs/*.yaml`、feature `manifest.yaml`、训练
   `run_manifest.yaml`、`predictions.csv` 和 `fold_metrics.csv`，并记录
   `python --version`、`python -c "import torch; print(torch.__version__)"`、
   Windows 版本与 CPU 架构。验收时应看到 `features=ready` 和
   `chemical-vae-zinc-v5-windows-cpu × rf: complete`，12 行、392 列且所有有效。
5. 将这些 Windows 原始证据链接到 C11 后，才可把 `windows_cpu` 从
   `not_verified` 更新为 `passed`。失败时应保留日志与资产 hash，不要把 Linux
   结果复制为 Windows 结果。

CI 可以执行相同 PowerShell 块；其 Windows worker、环境创建日志和以上输出都
应作为平台证据的一部分。Linux 的 `nohup`/`setsid` 规则不适用于 Windows，
因此不要在 Windows 命令中伪造 PID 或 Unix 信号语义。
