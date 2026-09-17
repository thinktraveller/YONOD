# Schema-2 YAML 使用示例

下方是 [example.yaml](example.yaml) 的完整内容：一个可解析、可执行的小型 Morgan × RF smoke 配置。独立的 YAML 文件仍保留，供启动命令直接读取；修改示例时请同步更新两处内容。

```yaml
# 可直接复制的 schema-2 YAML 示例。此文件使用仓库内的小型 fixture，
# 仅运行 Morgan × RF 的 2-fold smoke；项目规模实验应另存配置和输出目录。
schema_version: "2.0"
project_name: yonod_schema2_smoke
stage: all

dataset:
  path: ./dataset/benchmark_smoke_fixture.csv
  sample_id_col: reaction_id
  column_roles:
    label: yield
    reactants: [reactant_1_smiles, reactant_2_smiles]
    products: []
    others: []
    conditions: []
    categoricals: []

# 每项均是单独的特征候选；不同描述符不会自动拼接。
descriptors:
  - id: morgan
    descriptor: morgan
    mode: concat
    columns: [reactant_1_smiles, reactant_2_smiles]

artifacts:
  output_dir: ./result/smoke-test/feature

models: [rf]
model_params:
  # 示例显式采用项目既有值；删除 estimator 中的字段即使用 sklearn 1.4.0 的库默认。
  rf:
    estimator:
      n_estimators: 10
      max_depth: null
      min_samples_leaf: 1
      max_features: 1.0
      n_jobs: 1
      random_state: 42
      verbose: 0

evaluation:
  protocol: outer_kfold
  n_splits: 2
  n_repeats: 1
  shuffle: true
  seed: 42

outputs:
  root: ./result/smoke-test
  report_formats: [Markdown]
metadata:
  notes: "小型可运行 smoke。正式研究请使用独立 YAML、特征包和输出根目录。"
```

建模应经 `yonod.py` 启动，而不是直接调用 `main.py --config`。以下为 Linux Bash 命令，在仓库根目录执行：

```bash
mkdir -p logs
nohup setsid bash -lc '
  source /home/wangzh685/miniconda3/etc/profile.d/conda.sh
  conda activate yonod
  printf "%s\n" "example.yaml" | python -u yonod.py
' </dev/null > logs/example_morgan_smoke.log 2>&1 &
echo $! > logs/example_morgan_smoke.pid
```

它使用仓库内 `dataset/benchmark_smoke_fixture.csv`，将特征、训练结果和报告统一写入 `result/smoke-test/`：

```text
result/smoke-test/
├── feature/
│   ├── feature_runs/<feature_run_id>.yaml
│   └── features/<artifact_id>/
├── pictures/
│   └── scatter_morgan_rf_repeat-1.png
├── report/
│   └── report.md
└── runs/
    └── run_morgan-rf/
        ├── run_manifest.yaml
        ├── source_config.yaml
        ├── effective_config.yaml
        ├── predictions.csv
        └── fold_metrics.csv
```

所有相对路径均以 YAML 所在目录为基准。`outputs.root` 决定任务主目录；当前加载器将有效配置中的 `artifacts.output_dir` 统一为该主目录下的 `feature/`，示例显式填写相同路径。默认只生成 Markdown 报告；若要同时生成 HTML，将 `report_formats` 改为 `[html, markdown]`。`project_name` 是任务显示名称，不会覆盖 `outputs.root`。

组合目录使用 `run_<特征ID>-<模型>` 命名；内容生成的 `run_id` 仍保留在 manifest 中用于复用校验。失败记录保存在组合目录的 `failed_attempts/` 下。启动日志与 PID 位于仓库 `logs/`，不占用结果主目录的子文件夹。

正式实验请复制该文件，并改为专属数据集、输出根目录、特征声明和模型参数；不要重用 smoke 输出目录。

## Chemical VAE YAML

Chemical VAE 不是 `example.yaml` 的默认描述符。可选资产是分别经真实 HDF5 参考数值对齐的 ZINC/v5 和 `zinc_properties` encoder/v5 PyTorch encoder；两者只能通过各自 manifest 使用，`zinc_properties` 的性质头不属于本描述符。配置必须完整声明其定位器和冻结运行语义：

```yaml
  - id: chemical-vae-zinc-v5
    descriptor: chemical_vae
    lifecycle: static_descriptor
    mode: concat
    columns: [reactant_1_smiles, reactant_2_smiles]
    params:
      model_manifest: ../../WEIGHTS/chemical_vae/zinc-37e96cd3bc8f9680/v5/conversion_manifest.json
      backend: pytorch
      device: cpu
      batch_size: 4
      input_preprocessing: identity
      output: z_mean_sample
```

路径相对 YAML 文件本身解析。每个成功组分输出 196 维原始 `z_mean_sample`；两个列 `concat` 为 392 维。该路径不加载原 HDF5、TensorFlow/Keras、解码器或性质头；训练阶段只读已发布的 feature manifest。

步骤 31 的 features、train、all、配对比较、GPU smoke 与 Windows handoff YAML 都是历史验收输入，已归档为 [configs_20260916.tar.gz](recovery_backups/configs_20260916.tar.gz)。其中的 `artifacts.output_dir` 与 `outputs.root` 已彼此隔离；如需审计，应在仓库根解压归档以恢复原路径。新任务应新建 YAML 和输出目录，不能修改或复用这些验收配置及其历史产物。

Chemical VAE 输入保持 identity：不 trim、canonicalize、拆盐、截断或扩字表。长度超过 120、字符不支持、缺失或 RDKit 无效的值会进入 diagnostics sidecar；普通 `concat` 中单个角色失败仅产生零块，严格比较则必须筛掉任一角色失败的行。详细运行资产、依赖与平台限制见 [README](README.md) 与 [comparison report](derived/chemical_vae/step31_7_comparison/comparison_report.json)。

Linux CPU 和 `cuda:0` GPU 均有真实 smoke；ZINC/v5 与 `zinc_properties` encoder/v5 都有独立 CPU float32 HDF5/NumPy/PyTorch 数值证据。用户已授权将步骤31标记完成并延期 Windows CPU 验证；Windows CPU 本身仍是 `not_verified`，不能以这份 YAML、Linux 路径测试或任何复制产物替代真实 Windows 运行。GPU/资产证据及范围见 [C11 verification](derived/chemical_vae/step31_9_gpu_smoke/verification.json)。

## 三个阶段

- `features`：需要 `dataset`、`descriptors` 和 `artifacts.output_dir`；建议显式声明 `outputs.root`。不需要标签或模型，会初始化四个子目录，但只生成特征产物，不执行训练。
- `train`：需要 `dataset`（含标签）、`artifacts.input_manifest` 和 `models`；绝不会回退到描述符计算。
- `all`：需要 features 和 train 的声明；它先完成/隔离全部特征候选，再对就绪包调用同一训练服务。

外部 feature manifest、标签和辅助列一律按稳定 `sample_id` 对齐。静态特征身份排除标签，所以只修改标签会改变训练身份，而非特征身份。同一组合目录已有不同身份的结果时会拒绝覆盖；应使用新的任务主目录，并可通过 `stage: train` 引用原特征 manifest。

## 中间版本

```bash
python scripts/manage_features.py inspect --manifest result/smoke-test/feature/features/<id>/manifest.yaml
python scripts/manage_features.py derive --config select-or-join.yaml
```

`derive_features` 只支持显式 sample ID/列选择和 ID 一对一的数值 CSV 拼接。它总是发布新包，记录父版本、操作声明、输入/输出映射和哈希；标签列、重复或缺失 ID 以及列冲突会失败。

## 参数和旧配置

模型参数按 `model_params.<model>` 中的 API 区段配置。字段省略表示使用该依赖版本的库默认，`null`、`false` 和 `0` 则仍是显式值。SVM 的 scaler、PCA、子采样只有明确声明时才启用；OHE 仅在训练折拟合。

旧 JSON/benchmark YAML 不能作为 `main.py --config` 的输入。请先运行：

```bash
python scripts/migrate_config_to_yaml.py --input old.json --output run.yaml --sample-id-col sample_id
```

迁移器会生成同级 `.migration.json` 报告。它无法确定稳定 ID、旧 grouping 或协议语义时会停止，不会猜测或覆盖旧文件。
