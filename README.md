# YONOD — Numerical Reaction Prediction and Modeling Method Comparison

**English** · [简体中文](README.zh-CN.md)

**You Only Need Outstanding Descriptors.** Start with existing reaction records, convert molecular structures and recorded conditions into model inputs, build models for numerical targets such as yield, and obtain per-record predictions, errors, and traceable reports.

YONOD helps chemistry researchers compare numerical structure encodings (descriptors) and regression methods. It needs reference values in the data for modeling and evaluation. It does not automatically generate chemical products, infer reaction mechanisms, or find the best experimental conditions for you.

## 1. Quick start

Start with the [random-split teaching example](example/random_split/README.md), then explore fixed membership and external testing. All three examples use synthetic data included in the repository and Morgan encodings. **No pretrained model weights need to be downloaded.**

### 1.1 Get the project and prepare the environment

Install Git and Conda first. The project baseline is Python 3.9. Choose the commands for your shell. The Linux examples assume Conda is under `$HOME/miniconda3`; replace that prefix with your installation directory. For Windows, open **Anaconda PowerShell Prompt** or a PowerShell session initialized with `conda init powershell` and restarted. Conda Prompt also supports the plain `conda`, `pip`, and `python` commands, but the PowerShell background/status blocks below require PowerShell.

**Linux Bash — first-time setup**

```bash
source "$HOME/miniconda3/etc/profile.d/conda.sh"
git clone https://github.com/thinktraveller/YONOD.git
cd YONOD
conda create -n yonod python=3.9 -y
conda activate yonod
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

**Windows Conda PowerShell — first-time setup**

```powershell
git clone https://github.com/thinktraveller/YONOD.git
Set-Location YONOD
conda create -n yonod python=3.9 -y
conda activate yonod
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

These commands install CPU PyTorch, which is sufficient for the three teaching examples. For CUDA, follow the version notes in [requirements.txt](requirements.txt) and choose a PyTorch wheel source appropriate for your machine. LightGBM and SciPy are included in `requirements.txt`; PyTorch must be installed separately first. Install the following only when using AutoGluon; it works in either Linux Bash or Windows Conda PowerShell after environment activation:

```text
pip install "autogluon.tabular[lightgbm,catboost]==1.1.1"
```

If you already have a `yonod` environment, skip creation and installation, but activate it in each new terminal. The examples leave thread counts unspecified and do not require CPU affinity. Installed model-library defaults apply; configure resources explicitly only when needed for your machine.

### 1.2 Launch an example from its own folder

Both launch blocks below start from the YONOD root, supply `run.yaml` to the first `yonod.py` prompt, and save logs and the PID in the repository-root `logs/` folder.

**Linux Bash — background execution**

```bash
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate yonod
export PYTHONUTF8=1
cd example/random_split
mkdir -p ../../logs
printf '%s\n' 'run.yaml' > ../../logs/example_random_split_20261113.stdin.txt
nohup python -u ../../yonod.py < ../../logs/example_random_split_20261113.stdin.txt > ../../logs/example_random_split_20261113.log 2>&1 &
echo $! > ../../logs/example_random_split_20261113.pid
```

**Windows Conda PowerShell**

```powershell
conda activate yonod
$env:PYTHONUTF8 = "1"
Set-Location example\random_split
$yonodLogs = [IO.Path]::GetFullPath((Join-Path $PWD.Path "..\..\logs"))
New-Item -ItemType Directory -Path $yonodLogs -Force | Out-Null
$yonodInput = Join-Path $yonodLogs "example_random_split_20261113.stdin.txt"
"run.yaml" | Set-Content -LiteralPath $yonodInput -Encoding ascii
$yonodProcess = Start-Process -FilePath (Get-Command python).Source -ArgumentList @("-u", "..\..\yonod.py") -WorkingDirectory $PWD.Path -RedirectStandardInput $yonodInput -RedirectStandardOutput (Join-Path $yonodLogs "example_random_split_20261113.log") -RedirectStandardError (Join-Path $yonodLogs "example_random_split_20261113.stderr.log") -PassThru
$yonodProcess.Id | Set-Content -LiteralPath (Join-Path $yonodLogs "example_random_split_20261113.pid") -Encoding ascii
```

Alternatively, activate the environment and enter the example folder, run `python -u ../../yonod.py` on Linux or `python -u ..\..\yonod.py` on Windows, then enter `run.yaml` at the first prompt. The public entry point is `yonod.py`; do not pass `--csv`, `--models`, or `--config` as arguments to this interactive entry point.

**Run one task at a time.** Wait for the current task to finish before starting another example. For fixed membership or external testing, enter `example/fixed_split/` or `example/external_test/` and change the log and PID filenames accordingly. Saved full-execution acceptance comes from Linux. The Windows commands explain usage; they do not establish that every optional backend has been tested on a Windows machine.

### 1.3 Check progress and read the report

While still in the random-split example folder, choose the commands for your operating system. On Windows, standard output and errors are saved separately in `.log` and `.stderr.log`; inspect both.

**Linux Bash**

```bash
task_pid="$(cat ../../logs/example_random_split_20261113.pid)"
ps -p "$task_pid" -o pid,ppid,stat,etime,cmd
tail -n 60 ../../logs/example_random_split_20261113.log
```

**Windows Conda PowerShell**

```powershell
$taskPid = [int](Get-Content ..\..\logs\example_random_split_20261113.pid)
Get-Process -Id $taskPid -ErrorAction SilentlyContinue
Get-Content ..\..\logs\example_random_split_20261113.log -Encoding UTF8 -Tail 60
Get-Content ..\..\logs\example_random_split_20261113.stderr.log -Encoding UTF8 -Tail 60
```

After the process exits, inspect the completion messages and saved run records. The ordinary workflow should report `features=ready` and `complete` or `reused` for all three model combinations. Fixed membership should have two successful fold records. External testing should have a complete prediction table and both reports. Run IDs are content-derived; do not match a hard-coded ID.

New results are written to `result/<task_name>/` **inside the current example folder**. Open its HTML file in a file manager, with `xdg-open <report-path>` on Linux, or with `Invoke-Item <report-path>` in Windows PowerShell. The next section lists the filenames for each protocol.

## 2. Follow the examples: data, settings, and outputs

[Offline guide](example/index.html) · [All operating instructions](example/README.md) · [Download all three complete examples](example/archives/all_examples.zip)

All data and yields are **synthetic teaching values, not experimental measurements**. Molecular pairings illustrate the input format and are not claimed to represent real reactions. The saved acceptance records for [random splitting](example/random_split/verification.json), [fixed membership](example/fixed_split/verification.json), and [external testing](example/external_test/verification.json) establish complete workflow execution. They do not establish predictive performance on real chemistry.

### 2.1 Random-split internal evaluation: start with one table

1. Open [reactions.csv](example/random_split/dataset/reactions.csv). Each row is one record: `reaction_id` is its unique ID, `yield` is the reference yield, and the two `*_smiles` columns contain molecular structure strings (SMILES). For example, `yield=0.10` means 10%, not 0.10%.
2. Open [run.yaml](example/random_split/run.yaml) in the same folder. It selects the two structure columns, Morgan encodings, Random Forest (RF), XGBoost, LightGBM, and random two-fold rotation: use six records for modeling and six for evaluation, then exchange their roles.
3. Launch this folder's `run.yaml` as described in the quick start. The first complete run performs two folds per model, giving six fits in total; existing results with matching identities can be reused.
4. Read `result/example_random_split_20261113/report/report.html` or `report.md`. Per-record held-out predictions are stored in `runs/run_morgan-<model>/predictions.csv`.

Each record is predicted once by the corresponding model that did not learn from that record. Combining these predictions gives out-of-fold internal predictions (OOF). Random splitting does not ensure that substrates in the evaluation records were absent from the modeling records.

### 2.2 Fixed-membership internal evaluation: choose evaluation members in advance

1. Read the [data table](example/fixed_split/dataset/reactions.csv) and [role guide](example/fixed_split/splits/membership_guide.csv). `fixed_block` divides the 12 records into two teaching groups of six. The grouping column is not a model input.
2. Read the [fixed-membership manifest](example/fixed_split/splits/fixed_split_manifest.csv) consumed by the program and [run.yaml](example/fixed_split/run.yaml). Executing the membership plan requires `stage: benchmark` and `manifest_outer_cv`. Fixing a random seed alone does not specify members.
3. Launch `run.yaml` from `example/fixed_split/`. Fold 1 holds out `block_A`; fold 2 holds out `block_B`. RF is fitted twice.
4. Read `result/example_fixed_split_20261113/report/benchmark_report.html` or `.md`. The executed membership plan and predictions are saved under `docs/manifests/` and `docs/predictions/`.

The current interface requires at least two folds, with every record held out exactly once per repeat. This example cannot be replaced by a single “eight training records plus four evaluation records” plan. After changing the data, check record IDs, source row indices, members, group separation, and the CSV file's SHA-256 together. See the [fixed-membership instructions](example/fixed_split/README.md). The teaching group names have no substrate or reaction-class meaning.

### 2.3 External testing: separate training and evaluation tables

1. Compare the 12 training records in [train.csv](example/external_test/dataset/train.csv) with the six external records in [external.csv](example/external_test/dataset/external.csv). Column names may differ. The external table also has source IDs and three amine groups.
2. Inspect `external_test.column_map` in [run.yaml](example/external_test/run.yaml): training column `amine_smiles` maps to external column `external_amine`, with equivalent mappings for the other structure columns. Reference yields are specified separately through training `label` and external `label_col`, not through the structure map.
3. Check the [source audit file](example/external_test/audit/external_audit.json) and the checksums of the CSV and audit files. For real research, recheck provenance and structural overlap; do not copy the example's zero counts.
4. Launch `run.yaml` from `example/external_test/`. Model settings are fixed in advance. RF is fitted once on all 12 training records, then predicts the six external records. External reference values are used for final scoring.
5. Read `result/example_external_test_20261113/report/external_test_report.html` or `.md`. Predictions are in `docs/predictions/morgan__rf.parquet`, and metrics are in `docs/metrics/external_test_metrics.csv`.

This example uses amine, acid, and product structures; in real use, confirm that the target product structure is known at prediction time. The current external adapter requires the audit to declare zero exact canonical structure overlap for amines, acids, and products, and at least two nonempty external groups. Those intersections were actually checked for this example, but synthetic data have no independent experimental source. A separate CSV or zero structural overlap does not automatically establish scientific independence. Predicting values for new structures without reference measurements is a different workflow from this external evaluation. See the [external-test instructions](example/external_test/README.md) for further requirements.

### 2.4 Interpret the results

| Metric or file | How to read it |
| --- | --- |
| MAE, mean absolute error | Same units as the target; on a 0–1 yield scale, `MAE=0.10` means an average difference of 10 percentage points |
| RMSE, root mean squared error | Same units as the target and more sensitive to large errors |
| R², coefficient of determination | Compares error against the evaluation data's mean; it can be negative and is not “prediction accuracy” |
| Training score | Fit to records already learned; it cannot replace held-out or external scores |
| `verification.json` | Run and file acceptance record for the example, not a predictive-performance guarantee |

Random-split reports score pooled OOF predictions within each repeat. Fixed-membership reports summarize fold-level metrics. Their R² aggregation differs, so do not rank methods directly across these workflows. Low scores and negative R² values are retained in all three examples.

The ZIP packages include `completed_run/` as a reference snapshot of completed outputs. The default `result/` folder is excluded, leaving a fresh output destination after extraction. Runtime folders are not tracked in source Git. With a source-only clone, reports are generated after execution or can be viewed in a downloaded package. The random package's separate `report/` contains a historical report; YAML files under `evidence/` are for historical traceability only.

## 3. YAML: specify one computation clearly

[example.yaml](example.yaml) is the sole complete format specification. This page explains fields without duplicating a YAML template. The three teaching `run.yaml` files show concrete tasks.

### Two entry scripts

| Script | Role |
| --- | --- |
| `generate_yaml_config.py` | Generate and validate a standard schema-2 YAML interactively from a CSV; save `config/<task_name>.yaml` without starting modeling |
| `yonod.py` | Task launcher and modeling runtime; read the prepared YAML at the first prompt, then run ordinary `features`/`train`/`all` stages or the selected benchmark protocol |

### Generate a YAML from your own CSV

Start from the repository root. Linux Bash:

```bash
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate yonod
export PYTHONUTF8=1
python -u generate_yaml_config.py
```

Windows Conda PowerShell:

```powershell
conda activate yonod
$env:PYTHONUTF8 = "1"
python -u .\generate_yaml_config.py
```

Enter the CSV path at the first prompt, then complete the column-role, encoding, model, and task questions. The generator saves the validated configuration directly to the root `config/<task_name>.yaml`; no configuration copy is needed. It may save normalized/prepared CSVs and invalid-input reports under the isolated `result/<task_name>/feature/docs/` while collecting roles. Generation does not compute descriptors, fit models, predict, or launch a run. Supplying an existing YAML to the generator validates it and returns without running it.

Review the generated YAML and its input/output paths, then start `python -u yonod.py` on Linux or `python -u .\yonod.py` on Windows and enter that `config/<task_name>.yaml` path. The three teaching examples already supply their own `run.yaml`, so they do not need this generation step.

| Field | Meaning and usage |
| --- | --- |
| `schema_version` | Currently `"2.0"` |
| `project_name` | Separate task name, associated with its own output directory |
| `stage` | Which phase to execute; see the table below |
| `dataset.path`, `sample_id_col` | Data table and stable ID column; IDs must exist, be nonempty, and be unique |
| `dataset.column_roles` | Assign target `label`, `reactants`, `products`, other structures `others`, numerical `conditions`, and categorical columns `categoricals` |
| `dataset.numeric_conditions` | Fit imputation statistics and scaling parameters on the corresponding training fold; unit conversion is declared explicitly in YAML, never guessed |
| `descriptors` | Encoding candidates; `columns` selects inputs and `mode: concat` joins columns in order. Multiple candidates are not automatically combined into one encoding |
| `artifacts.output_dir` | The task root's `feature/` child; `train` also needs `artifacts.input_manifest` pointing to verifiable encoding artifacts |
| `models`, `model_params` | Methods and model settings; declare parameters only for selected models |
| `evaluation` | Rules for random rotation, fixed/grouped membership, or external testing; fields from different protocols cannot be mixed arbitrarily |
| `hpo` | Automatic comparison and selection of model settings (hyperparameter optimization), not optimization of reaction conditions; disabled in all three examples |
| `outputs.root`, `report_formats` | Isolated output root and both `[html, markdown]` reports |
| `outputs.diagnostics` | Optional comparison of training and held-out performance; disabling diagnostics does not disable ordinary scoring |
| `benchmark` | Task identity and SQLite recovery state for dedicated protocols; do not copy this section into the ordinary random workflow |
| `metadata` | Human-readable notes, not model inputs |

| `stage` | Behavior |
| --- | --- |
| `features` | Generate or reuse input encodings only; no model is fitted and reference labels may be omitted |
| `train` | Read encodings through `artifacts.input_manifest`, then fit; missing encodings are not computed automatically |
| `all` | Prepare encodings, then complete ordinary internal evaluation for available encoding/model combinations |
| `benchmark` | Execute the dedicated `manifest_outer_cv` or `frozen_train_external_test` workflow |

Ordinary `all/train` uses KFold. Actual fixed-membership or grouped evaluation requires `stage: benchmark`; changing a protocol string in the ordinary workflow does not create a membership-manifest executor. The dedicated executor can also use random splitting (`repeated_kfold`), so the word “strict” does not by itself mean unseen substrates.

### Paths, budgets, and your own research tasks

- Teaching examples: keep `run.yaml` in its own folder. Inputs and `./result/<task_name>/` resolve relative to that file.
- Research tasks: create an independent YAML under the repository-root `config/` folder. References beginning with `./` there resolve from the repository root. Explicit `../` references and configurations elsewhere resolve from their YAML's directory. Absolute paths remain absolute.
- Each task's `artifacts.output_dir` should be `outputs.root/feature`, not the task root itself. Examples omit thread parameters and use installed model-library defaults. To limit resources, explicitly configure RF/XGBoost/LightGBM `estimator.n_jobs` or AutoGluon `fit.num_cpus` for your machine. Do not start tasks, encodings, combinations, or folds as separate parallel background jobs.

To edit a configuration manually instead of using the generator, copy the format specification from the repository root. Linux Bash:

```bash
mkdir -p config
cp example.yaml config/my_reaction.yaml
```

Windows Conda PowerShell:

```powershell
New-Item -ItemType Directory -Path config -Force | Out-Null
Copy-Item example.yaml config\my_reaction.yaml
```

In the new file, update the task name, data path and column roles, encodings and models, model budget, split rules, and `./result/my_reaction` with its `feature/` child. The root format example uses 12 synthetic records, two folds, and 10 RF trees; thread counts follow the installed model library. For research, determine the scientific question, model settings, and computation budget yourself. Fixed membership and external testing also require their dedicated fields from the corresponding teaching example; changing only the `protocol` string is insufficient. Continue to supply the edited YAML at the first `yonod.py` prompt.

The parser rejects duplicate keys, aliases, and unknown fields. Executable run configurations must be schema-2 YAML, not JSON; JSON audit files, manifests, and other evidence remain valid for their documented purposes. Use `generate_yaml_config.py` to prepare settings and `yonod.py` to validate and run the saved YAML. With no arguments, the runtime prompts for an existing YAML; CSV preparation belongs to the generator. The engineering smoke interface is `python yonod.py --config example.yaml`. Configuration locations and task isolation still follow [AGENTS.md](AGENTS.md).

### Available encodings and models

Encodings without neural pretrained weights include `morgan`, `mfp`, `maccs`, `maf`, `rdkit2d`, `drfp`, and categorical encoding `ohe`. `mfp` is a count-based Morgan encoding, not an alias for binary `morgan`. `drfp` has reaction-input constraints. `fisd`, `molmetalm`, and `chemical_vae` need the assets described below. OHE, numerical missing-value handling, and scaling learn only from the corresponding modeling fold; held-out records are only transformed.

Model identifiers are `rf`, `xgb`, `lightgbm`, `svm`, and `autogluon`. RF/XGBoost/LightGBM use `estimator`, `fit`, and `runtime` sections. SVM additionally has `preprocessing`; AutoGluon uses `predictor`, `fit`, and `runtime`. Omitted parameters use defaults from the supported installed version. Do not infer a run's settings from another version's defaults.

HPO is supported only through the integrated RF/XGBoost/LightGBM search workflows and operates within the corresponding training data. The current frozen external workflow does not support HPO, OHE, numerical conditions, AutoGluon, or external-test early stopping. It saves neither a reusable final model nor the complete feature artifacts used by the ordinary workflow; it encodes, fits, and scores in memory according to its protocol. External-test configurations must not include internal split fields such as `n_splits`, `n_repeats`, `shuffle`, `grouping`, or `split_manifest`.

## 4. Project structure

```text
YONOD/
├── README.md / README.zh-CN.md  English default entry and Chinese guide
├── generate_yaml_config.py     Generate and validate schema-2 YAML without modeling
├── yonod.py                    Interactive YAML launcher and modeling runtime
├── example.yaml                Complete schema-2 format and minimal smoke test
├── requirements.txt            Base dependencies; PyTorch and others are installed separately
├── LICENSE                     Repository license
├── example/                    Three self-contained teaching examples, guide, and ZIPs
│   ├── random_split/
│   ├── fixed_split/
│   └── external_test/
├── yonod/                      Configuration, encoding, models, pipelines, reports, and dedicated protocols
├── dataset/                    Data and provenance notes
├── WEIGHTS/                    Optional external model assets
├── config/                     Local research-task YAMLs (Git-ignored)
├── result/                     Isolated research-task outputs (Git-ignored)
├── logs/                       All task logs and PIDs (Git-ignored)
├── project-docs/               Independent documentation Git repository, outside the main repository
├── _verify/                    Local validation scripts (Git-ignored)
├── backup/                     Engineering support and historical material (Git-ignored)
└── reference-projects/         External reference projects (Git-ignored)
```

Example-local `result/` folders and the downloadable packages' `completed_run/` snapshots live inside their respective packages. Specific encoding, model, prediction, and report artifacts depend on the protocol; do not assume that all three workflows save the same model bundles. Additional engineering notes are maintained in the separate [YONOD-docs](https://github.com/thinktraveller/YONOD-docs) repository, which requires access permission. Cloning the main repository does not automatically provide it.

## 5. External data and model assets

### Data and project-weight distribution

Read the [data notes](dataset/README.md) and [weight notes](WEIGHTS/README.md) first. Data for the three teaching examples are included in the source. Project distribution entry points are [YONOD-datasets](https://huggingface.co/datasets/thinktraveller/YONOD-datasets) and [YONOD-weights](https://huggingface.co/thinktraveller/YONOD-weights); they do not replace original provenance and license records.

The following commands work in Linux Bash and Windows Conda PowerShell from the repository root with `yonod` activated. Install a Hugging Face CLI version compatible with Transformers 4.40:

```text
pip install "huggingface_hub>=0.34,<1.0"
hf download thinktraveller/YONOD-datasets --repo-type dataset --include "USPTO/**" --local-dir dataset
hf download thinktraveller/YONOD-weights --local-dir WEIGHTS
```

The first download command retrieves only the USPTO subdirectory, not all data. Complete any login or access authorization required by the service. Binary weights are Git-ignored. Having configuration, tokenizer, or manifest files does not mean that all model weights are present.

### Optional pretrained encodings

| Encoding | Required assets and location |
| --- | --- |
| FISD | Put the [upstream weights](https://github.com/KeantChen/FISD/tree/main/model) in `WEIGHTS/FISD/`: `qm_9_mse_model.pth`, `qm_9_cos_model.pth`, and `qm_9_2in1_model.pth` |
| MolMetaLM | Put the model, configuration, and tokenizer from [MolMetaLM-base](https://huggingface.co/wudejian789/MolMetaLM-base) in `WEIGHTS/MolMetaLM-base/`; see the [upstream project](https://github.com/CSUBioGroup/MolMetaLM) |
| Chemical VAE | Use a verified conversion manifest under `WEIGHTS/chemical_vae/<asset>/v5/` with its adjacent, hash-matched `encoder_state_dict.pt`; ZINC and `zinc_properties` assets are not interchangeable |

To restore MolMetaLM separately (the same command works on either system):

```text
hf download wudejian789/MolMetaLM-base --local-dir WEIGHTS/MolMetaLM-base
```

Chemical VAE uses only raw `z_mean_sample` from a frozen PyTorch encoder. YONOD does not train the VAE or load its decoder or property head. Original HDF5 files, or a manifest alone, cannot substitute for converted weights. TensorFlow/Keras is not required for ordinary execution. See the [loader](yonod/descriptors/chemical_vae.py) for exact constraints.

Download and configure only the assets you intend to use. External data, weights, and reference projects retain their upstream licenses, citation requirements, and scope of use; the repository license does not override them.

## 6. License

Repository code and documentation use [CC BY-NC 4.0](LICENSE), requiring attribution and noncommercial use. Third-party data, pretrained weights, and reference code follow their own licenses.
