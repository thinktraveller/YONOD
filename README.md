# YONOD

[English](README.md) · [简体中文](README.zh-CN.md)

YONOD (You Only Need Outstanding Descriptors) helps chemistry researchers build and compare models for numerical reaction properties such as yield. You provide a table of reaction records, molecular structures and measured values. YONOD converts the inputs into numbers, trains regression models, and saves predictions and reports showing how well the models predict held-out records.

You can use it without writing Python code. Start the program from a terminal and describe each calculation in a text settings file called YAML. The interactive prompts are currently in Chinese.

## What can I do with it?

- Compare molecular descriptors, which are numerical encodings of structures, and regression models on the same data.
- Include recorded numerical conditions, such as temperature, or categorical inputs when the selected workflow supports them.
- Evaluate predictions by random splitting, a saved list of training/test members, or a separate external test table.
- Inspect per-record predictions, error metrics and HTML reports, with saved settings and run records for tracing the calculation.

Model training and evaluation need known reference values. YONOD does not generate product structures, explain reaction mechanisms or automatically choose the best laboratory conditions. If you use product structures as inputs, those structures must be known when you make predictions.

The included teaching data run on a CPU without pretrained model weights. You can try them before preparing your own dataset.

## Install YONOD

### Before running commands

Install [Git](https://git-scm.com/downloads) to download the project and [Miniconda](https://www.anaconda.com/docs/getting-started/miniconda/install) to manage its Python environment. An environment is a separate set of Python packages; this project's environment is named `yonod` and uses Python 3.9.

On Linux, open a Bash terminal. On Windows, open Anaconda PowerShell Prompt, or a PowerShell window where Conda has been initialized. If necessary, run `conda init powershell` from Anaconda Prompt, then close and reopen PowerShell.

Enter the commands one line at a time, pressing Enter after each. They belong in the terminal, not at a Python `>>>` prompt. The project root means the `YONOD` folder containing `yonod.py`, `example.yaml` and `requirements.txt`.

### First-time installation

Choose the block for your system. The Linux block assumes Miniconda is installed in `$HOME/miniconda3`; change that prefix if yours is elsewhere.

Linux Bash:

```bash
source "$HOME/miniconda3/etc/profile.d/conda.sh"
git clone https://github.com/thinktraveller/YONOD.git
cd YONOD
conda create -n yonod python=3.9 -y
conda activate yonod
python -m pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

Windows Conda PowerShell:

```powershell
git clone https://github.com/thinktraveller/YONOD.git
Set-Location YONOD
conda create -n yonod python=3.9 -y
conda activate yonod
python -m pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

If you already downloaded the project, enter its root folder and skip `git clone` and the following directory-change command. If you already installed the `yonod` environment, activate it instead of creating it again.

Install PyTorch before the other requirements. The commands above use CPU PyTorch; the three teaching examples need no GPU. LightGBM and SciPy are included in [requirements.txt](requirements.txt). CUDA installation notes are also in that file; choose the PyTorch package for your hardware using the [official version instructions](https://pytorch.org/get-started/previous-versions/).

### Check the environment

With `yonod` active, these commands work in both systems:

```text
python --version
python -c "import torch, rdkit, sklearn, xgboost, lightgbm; print('YONOD imports OK')"
```

Expect Python 3.9 and `YONOD imports OK`. This checks that the main packages can be imported. Keep the terminal open for the first run. In each new terminal, activate `yonod` again; Linux also needs the `source` line when Conda has not been initialized in that shell.

## Complete your first run

### 1. Look at the teaching files

Open [the data table](example/random_split/dataset/reactions.csv) and [its settings](example/random_split/run.yaml). A CSV is a text table; SMILES is a text representation of a molecular structure.

| Column | Meaning in this example |
| --- | --- |
| `reaction_id` | A unique record ID, used to match outputs to inputs |
| `yield` | Reference yield on a scale from 0 to 1; `0.10` means 10% |
| `reactant_1_smiles`, `reactant_2_smiles` | The two molecular structures used as model inputs |

All 12 records and their yields are artificial teaching values, not experimental measurements. The molecular combinations illustrate the file format and do not establish that these are chemically valid reactions.

The prepared `run.yaml` selects Morgan fingerprints and three models: Random Forest (`rf`), XGBoost (`xgb`) and LightGBM (`lightgbm`). It divides the records into two groups of six, trains on one group and predicts the other, then swaps their roles. The first full run fits each model twice, for six small fits in total.

### 2. Start the program

Start from the project root. If you just followed the installation steps, you are already there. Keep this terminal open while the calculation runs.

Linux Bash:

```bash
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate yonod
export PYTHONUTF8=1
cd example/random_split
python -u ../../yonod.py
```

Windows Conda PowerShell:

```powershell
conda activate yonod
$env:PYTHONUTF8 = "1"
Set-Location .\example\random_split
python -u ..\..\yonod.py
```

The program asks `请输入配置文件路径(schema-2 YAML):` (enter the configuration file path). Type the following and press Enter:

```text
run.yaml
```

This file is already prepared. You do not need to copy it or run the YAML generator first. Run one task at a time, waiting for it to finish before starting another.

### 3. Check completion and open the report

At the end, look for `features=ready` and a `complete` or `reused` status for each of `morgan × rf`, `morgan × xgb` and `morgan × lightgbm`. In the ordinary workflow, `reused` means matching saved results were available. The `[report]` message prints the report locations. An error message or a missing report needs checking even if the process has ended.

Still in `example/random_split/`, open `result/example_random_split_20261113/report/report.html` in your file manager. On desktop Linux you can also use:

```bash
xdg-open result/example_random_split_20261113/report/report.html
```

Windows Conda PowerShell:

```powershell
Invoke-Item .\result\example_random_split_20261113\report\report.html
```

The same folder contains `report.md`, a text report you can read in an editor. Each model's per-record predictions are under `result/example_random_split_20261113/runs/run_morgan-<model>/predictions.csv`, where `<model>` is `rf`, `xgb` or `lightgbm`. Look inside the example folder for these results; the project-root `result/` folder holds research tasks.

### 4. Read the results

Each record receives a prediction from a model that did not train on that record. These are called out-of-fold (OOF) predictions. Random splitting can still put the same substrate in both training and evaluation records, so it does not establish performance on unseen substrates.

| Metric | What it tells you |
| --- | --- |
| MAE, mean absolute error | Average size of the prediction error. On this 0 to 1 yield scale, `MAE=0.10` means 10 percentage points. |
| RMSE, root mean squared error | Error in the same units as the target, with greater weight on large errors. |
| R², coefficient of determination | Error compared with using the evaluation data's mean as a reference. It can be negative and is not a percentage of correct predictions. |

Training scores describe records the model has already learned from; use held-out or external scores to assess prediction. Low scores and negative R² are retained in these examples. Their purpose is to teach the workflow, not to establish performance on real chemistry or recommend a model.

## Use your own reaction table

Return to the project root before continuing: use `cd ../..` in Linux Bash or `Set-Location ..\..` in Windows PowerShell if you are still in the example folder.

### Prepare the CSV

Save a comma-separated CSV with a header row, using UTF-8 encoding. In a spreadsheet editor, use Save As or Export to CSV UTF-8; changing an `.xlsx` filename to `.csv` does not convert it. Structure columns need valid SMILES text, not molecular drawings or reagent names. Each row should be one record, with a stable, nonempty, unique ID. IDs such as `record-001` preserve their meaning through CSV readers; ambiguous values such as `001` or `NA` can be changed by automatic type inference and the generator rejects them when that would happen.

Check the reference value's meaning and unit. Yield, conversion, LC area ratio and selectivity are different targets. Use one consistent scale, such as 0 to 1 or 0 to 100, and write numbers without a `%` suffix. YONOD does not infer the intended scale or convert it automatically.

The current interactive generator requires a numerical target and exactly one product SMILES column. You can also select reactant columns, other structure columns and numerical conditions. If your data have no product column, use the manual YAML route below; the runtime itself supports configurations with `products: []`. Whenever product structures are included in the model inputs, they must be available at prediction time.

### Generate settings, then run them

From the project root, launch the generator.

Linux Bash:

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

Follow the displayed prompts in this order:

1. Enter the CSV path and a new task name, for example `my_reaction`. Use letters, numbers, `_` or `-`; existing configurations and nonempty task folders are protected from overwriting.
2. Select the stable ID column. It is used for record matching and excluded from the target and model inputs. Then assign the target, reactants, product, other structures and numerical conditions. Follow the indices shown at each prompt; their numbering is not the same in every menu.
3. Check the invalid-input report and the number of retained records. The generator can exclude rows with invalid inputs, so confirm that the remaining table represents the data you intend to use.
4. At the descriptor menu, enter `1` for Morgan. Check the displayed input columns, which default to the selected reactants, other structures and product. Press Enter at `选择 [N]:` to retain them, or enter `y` to edit the descriptor configuration. All included structures must be known at prediction time.
5. At the model menu, enter `2` for Random Forest. Pressing Enter at the descriptor or model menu selects several methods, including ones requiring pretrained weights or optional packages.
6. Add source/DOI notes if available. At the report-format menu, press Enter to retain both HTML and Markdown. Review the optional cleaned-CSV choice and the generated file paths.

The generator saves `config/my_reaction.yaml` and exits without calculating descriptors, fitting models or predicting. Prepared files stay under `result/my_reaction/`: the normalized CSV is in `feature/docs/`, and an invalid-input report, when needed, is in `report/`. Supplying an existing YAML to the generator only validates it.

Open the saved YAML in a text editor before launching it. The generated ordinary workflow uses five folds, one repeat and random seed 42, with model-library defaults and no HPO. It needs at least five retained records; this is a software minimum, not evidence that a five-record dataset is sufficient for research. Check the split design and calculation budget for your question. Numerical conditions default to rejecting missing values; configure imputation, scaling and explicit unit conversion when needed.

With the same environment active and from the project root, run `python -u yonod.py` on Linux or `python -u .\yonod.py` on Windows. At its first prompt, enter:

```text
config/my_reaction.yaml
```

Ordinary reports will be under `result/my_reaction/report/`. The generator prepares settings; `yonod.py` validates and executes those settings.

### Prepare YAML manually

Use [example.yaml](example.yaml) as the authoritative complete schema-2 format. To start a separate file from the project root:

Linux Bash:

```bash
mkdir -p config
cp example.yaml config/my_reaction.yaml
```

Windows Conda PowerShell:

```powershell
New-Item -ItemType Directory -Path config -Force | Out-Null
Copy-Item example.yaml config\my_reaction.yaml
```

Choose an unused filename. Edit the copy's task name, data path, ID and column roles, descriptor input columns, model settings and evaluation design. Set `outputs.root` to `./result/my_reaction` and `artifacts.output_dir` to `./result/my_reaction/feature`, retaining `outputs.report_formats: [html, markdown]`. The original file is a two-fold, 12-record Morgan/RF smoke configuration with 10 trees; its settings are not research recommendations. Launch your edited file through the same `yonod.py` prompt.

## Understand the main YAML settings

YAML uses indentation to group settings; use spaces and preserve the structure in [example.yaml](example.yaml). Save it as a plain UTF-8 text file. On Windows, show filename extensions and check that the editor has not saved it as `my_reaction.yaml.txt`. Executable configurations use `schema_version: "2.0"`. JSON audit files and manifests are evidence files, not executable run configurations.

| Setting | Purpose |
| --- | --- |
| `project_name` | A unique task name with its own result folder. |
| `dataset.path`, `sample_id_col`, `column_roles` | The input table, stable record ID and each column's purpose; the target is `label`. |
| `descriptors` | Structure encodings and their input columns. `mode: concat` joins columns within one descriptor; separate descriptor entries remain separate candidates. |
| `models`, `model_params` | Selected models and their settings. Declare parameters only for selected models. |
| `evaluation` | How training and evaluation members are separated. |
| `hpo` | Optional search for model settings within training data; this does not optimize laboratory conditions. |
| `outputs` | Isolated result location, report formats and optional training/held-out diagnostics. |

`stage: all` prepares features and runs ordinary internal evaluation. `features` only prepares encodings; `train` reads existing encodings through `artifacts.input_manifest`. `benchmark` selects a dedicated evaluation protocol. Ordinary `all/train` uses KFold; fixed membership needs `manifest_outer_cv` and a split manifest. The dedicated executor also supports `repeated_kfold`, so "strict" alone does not mean unseen substrates. Changing a protocol name without its required fields does not change a configuration into a valid new workflow.

For root `config/` files, paths beginning `./` resolve from the project root. Explicit `../` paths and configurations elsewhere resolve from the YAML file's folder; absolute paths stay absolute. Teaching `run.yaml` files therefore keep inputs and results inside their own package. Give every research task its own `result/<task_name>/`, with features in a subfolder, and keep operational logs and PID files in project-root `logs/`.

The examples leave CPU thread counts unspecified, using the installed model libraries' defaults. If needed, set RF/XGBoost/LightGBM `model_params.<model>.estimator.n_jobs` or AutoGluon `model_params.autogluon.fit.num_cpus` for your machine. Run tasks, descriptor/model combinations and folds in sequence.

## Try the other evaluation examples

After the first run, open the [offline guide](example/index.html) or [example instructions](example/README.md). Run each package's `run.yaml` from its own folder, using the same launcher pattern as above.

| Example and question | What happens | Report inside that example folder |
| --- | --- | --- |
| [Random splitting](example/random_split/README.md): how well do models predict held-out records from my table? | The first-run example compares three models using the same two-fold split. | `result/example_random_split_20261113/report/report.html` |
| [Fixed membership](example/fixed_split/README.md): how can I hold out members chosen in advance? | A saved manifest chooses the two six-record held-out groups; RF is fitted twice. | `result/example_fixed_split_20261113/report/benchmark_report.html` |
| [External testing](example/external_test/README.md): how does a fixed model recipe perform on a separate labeled table? | Fixed settings, one fit on 12 training records, predictions and scoring on six external records. | `result/example_external_test_20261113/report/external_test_report.html` |

Fixed membership requires matching IDs, source rows, data checksum and group rules; the current interface needs at least two folds with each record held out once per repeat. The external example maps columns between its two tables and requires provenance/overlap audit information and reference values for scoring. The current adapter requires the audit to declare zero exact canonical structure overlap for amines, acids and products, and at least two nonempty external groups. A separate CSV or zero structural overlap does not prove independent experimental evidence. The external example uses known product structures and artificial values.

The three workflows save different files. Frozen external testing currently excludes HPO, OHE, numerical conditions, AutoGluon and external-test early stopping, and does not save a reusable final model or the ordinary feature bundle. Its YAML must omit internal split fields such as `n_splits`, `n_repeats`, `shuffle`, `grouping` and `split_manifest`. Existing external metrics are protected from overwriting; the ordinary workflow's result reuse does not apply to every protocol. Fixed-fold averages and pooled OOF scores also differ, so do not directly rank scores across the examples.

[Download all three examples](example/archives/all_examples.zip) to inspect saved `completed_run/` snapshots. Source clones generate fresh `result/` folders when run; snapshots and the random example's historical `report/` and `evidence/` are reference material. The saved [random](example/random_split/verification.json), [fixed-membership](example/fixed_split/verification.json) and [external](example/external_test/verification.json) acceptance records document Linux execution. Windows commands are provided, but do not establish Windows runtime acceptance. For unattended runs and progress checks, follow the platform-specific commands in [example/README.md](example/README.md); logs and PID files belong in root `logs/`.

## Optional models, data and pretrained weights

Available models are `rf`, `xgb`, `lightgbm`, `svm` and `autogluon`. Install AutoGluon only when selecting it. With `yonod` active, the following command works in Linux Bash and Windows Conda PowerShell:

```text
python -m pip install "autogluon.tabular[lightgbm,catboost]==1.1.1"
```

Encodings without neural pretrained weights include `morgan`, count-based Morgan `mfp`, `maccs`, `maf`, `rdkit2d`, `drfp` and categorical `ohe`. DRFP requires reaction inputs; OHE and numerical preprocessing learn from the corresponding training fold. HPO is integrated for RF, XGBoost and LightGBM.

The [dataset notes](dataset/README.md) describe data sources, target definitions and licenses. Additional distribution is available through [YONOD-datasets](https://huggingface.co/datasets/thinktraveller/YONOD-datasets) and [YONOD-weights](https://huggingface.co/thinktraveller/YONOD-weights). From the root with `yonod` active, these commands work on both systems:

```text
python -m pip install "huggingface_hub>=0.34,<1.0"
hf download thinktraveller/YONOD-datasets --repo-type dataset --include "USPTO/**" --local-dir dataset
hf download thinktraveller/YONOD-weights --local-dir WEIGHTS
```

The dataset command downloads only `USPTO/`. Some source tables use tabs despite their CSV extension; read their source notes and prepare comma-separated data before using the generator. Complete any access authorization required by the service. Download only the assets you intend to use; the teaching examples already include their inputs.

| Pretrained encoding | Required assets |
| --- | --- |
| FISD | [Upstream weights](https://github.com/KeantChen/FISD/tree/main/model) in `WEIGHTS/FISD/`: `qm_9_mse_model.pth`, `qm_9_cos_model.pth`, `qm_9_2in1_model.pth`. |
| MolMetaLM | Model, configuration and tokenizer from [MolMetaLM-base](https://huggingface.co/wudejian789/MolMetaLM-base) in `WEIGHTS/MolMetaLM-base/`; see the [upstream project](https://github.com/CSUBioGroup/MolMetaLM). |
| Chemical VAE | A verified v5 conversion manifest under `WEIGHTS/chemical_vae/<asset>/v5/` and its adjacent, hash-matched `encoder_state_dict.pt`. ZINC and `zinc_properties` assets are distinct. |

A configuration, tokenizer or manifest alone does not supply the binary weights. Chemical VAE uses the frozen PyTorch encoder's raw `z_mean_sample`; it does not train a VAE or load a decoder/property head, and ordinary execution does not need TensorFlow/Keras. See the [weight notes](WEIGHTS/README.md) and [Chemical VAE loader](yonod/descriptors/chemical_vae.py).

## Where files belong

```text
YONOD/
├── yonod.py                 Launch and execute a saved YAML
├── generate_yaml_config.py  Prepare and validate YAML without modeling
├── example.yaml             Complete schema-2 format and engineering smoke
├── requirements.txt         Dependencies; install PyTorch separately first
├── example/                 Teaching files, offline guide and download packages
├── yonod/                   Implementation
├── dataset/                 Data and source notes
├── WEIGHTS/                 Optional pretrained assets
├── config/                  Local research YAMLs
├── result/                  Separate folders for research-task outputs
├── logs/                    Operational logs and PID files
├── _verify/                 Local validation scripts
├── backup/                  Backups and engineering support
├── reference-projects/      External reference material
└── project-docs/            Independent documentation repository
```

The last seven local working folders are excluded from the main repository. `project-docs/` has its own Git history; extra engineering notes live in [YONOD-docs](https://github.com/thinktraveller/YONOD-docs), which requires access permission. You do not need that repository for the teaching examples.

## If something goes wrong

| Symptom | Check |
| --- | --- |
| `conda` is not recognized | Confirm Miniconda is installed. Linux: use the correct `conda.sh` location. Windows: reopen an initialized Conda PowerShell session. |
| A terminal command gives `SyntaxError` at `>>>` | You are inside Python. Enter `exit()` to return to the terminal, then run the command there. |
| Python cannot find `yonod.py`, `run.yaml` or `requirements.txt` | Check the current folder. Installation uses the root; the teaching launch uses `example/random_split/`. |
| A path contains spaces, or a YAML file cannot be found | Quote paths in shell commands. Check the actual filename, including extensions; a file ending in `.yaml.txt` is a text file with the wrong extension. |
| `ModuleNotFoundError` | Activate `yonod`, then install the listed requirements there. PyTorch is a separate first installation; AutoGluon is optional. |
| No product column in the CSV wizard | Prepare a manual YAML from `example.yaml`; the wizard currently requires one product column. |
| ID or invalid-input error | Check for missing/duplicate IDs, ambiguous ID text and the generator's exclusion report. Keep the original-to-prepared ID mapping. |
| YAML validation fails | Check indentation, exact field names, paths and the selected protocol. Duplicate keys, aliases and unknown fields are rejected. |
| You want to rerun with changed data or settings | Use a new task name, YAML and result directory. The generator protects existing tasks; frozen external metrics also reject overwriting. Ordinary matching results may be reused. |
| Garbled Chinese paths or output | Keep CSV/YAML in UTF-8 and set `PYTHONUTF8=1` as shown for your shell. |
| A warning appears or the process ends | Read the final statuses and confirm the expected reports exist. Warnings alone do not establish failure or success. |

For a bug report in [project Issues](https://github.com/thinktraveller/YONOD/issues), include your operating system, Python/package versions, the error text and a small shareable example. Remove confidential data and credentials. See [AGENTS.md](AGENTS.md) for development conventions. The engineering smoke interface is `python yonod.py --config example.yaml`; it performs a small modeling run. Formal tasks use the saved-YAML prompt procedure above.

## License

Project code and documentation use [CC BY-NC 4.0](LICENSE), requiring attribution and limiting use to noncommercial purposes. Third-party data, pretrained weights and reference code retain their own licenses and citation requirements.

Last reviewed against this checkout: 2026-11-14.
