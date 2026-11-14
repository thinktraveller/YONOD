# YONOD

[English](README.md) · [简体中文](README.zh-CN.md)

[Project structure](#project-structure) · [Acknowledgments](#acknowledgments) · [License](#license)

YONOD (You Only Need Outstanding Descriptors) helps chemistry researchers build and compare models for numerical reaction properties such as yield. You provide a table of reaction records, molecular structures and measured values. YONOD converts the inputs into numbers, trains regression models, and saves predictions and reports showing how well the models predict held-out records.

You can use it without writing Python code. Start the program from a terminal and describe each calculation in a text settings file called YAML. The interactive prompts are currently in Chinese.

Read [beginner content](#beginner-content) to install and complete a smoke test, [advanced content](#advanced-content) to configure an evaluation, and [developer content](#developer-content) to inspect parameters or extend the code.

## Beginner content

- [What can I do with it?](#what-can-i-do-with-it)
- [Install YONOD](#install-yonod)
- [Complete the smoke test](#complete-the-smoke-test)
- [If something goes wrong](#if-something-goes-wrong)

### What can I do with it?

- Compare molecular descriptors, which are numerical encodings of structures, and regression models on the same data.
- Include recorded numerical conditions, such as temperature, or categorical inputs when the selected workflow supports them.
- Evaluate predictions by random splitting, a saved list of training/test members, or a separate external test table.
- Inspect per-record predictions, error metrics and HTML reports, with saved settings and run records for tracing the calculation.

- Model training and evaluation need known reference values.
- YONOD does not generate product structures, explain reaction mechanisms or automatically choose the best laboratory conditions.
- If you use product structures as inputs, those structures must be known when you make predictions.

The included first-use examples run on a CPU without pretrained model weights. Use them as smoke tests before preparing your own dataset.

### Install YONOD

#### Before running commands

- Install [Git](https://git-scm.com/downloads) to download the project and [Miniconda](https://www.anaconda.com/docs/getting-started/miniconda/install) to manage its Python environment.
- An environment is a separate set of Python packages; this project's environment is named `yonod` and uses Python 3.9.

- On Linux, open a Bash terminal.
- On Windows, open Anaconda PowerShell Prompt, or a PowerShell window where Conda has been initialized.
- If necessary, run `conda init powershell` from Anaconda Prompt, then close and reopen PowerShell.

- Enter the commands one line at a time, pressing Enter after each.
- They belong in the terminal, not at a Python `>>>` prompt.
- The project root means the `YONOD` folder containing `yonod.py`, `example.yaml` and `requirements.txt`.

#### First-time installation

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

- If you already downloaded the project, enter its root folder and skip `git clone` and the following directory-change command.
- If you already installed the `yonod` environment, activate it instead of creating it again.

- Install PyTorch before the other requirements.
- The commands above use CPU PyTorch; the three first-use examples need no GPU.
- LightGBM and SciPy are included in [requirements.txt](requirements.txt).
- CUDA installation notes are also in that file; choose the PyTorch package for your hardware using the [official version instructions](https://pytorch.org/get-started/previous-versions/).

#### Check the environment

With `yonod` active, these commands work in both systems:

```text
python --version
python -c "import torch, rdkit, sklearn, xgboost, lightgbm; print('YONOD imports OK')"
```

- Expect Python 3.9 and `YONOD imports OK`.
- This checks that the main packages can be imported.
- Keep the terminal open for the smoke test.
- In each new terminal, activate `yonod` again; Linux also needs the `source` line when Conda has not been initialized in that shell.

### Complete the smoke test

#### What is a smoke test?

A smoke test runs the main steps on a small dataset to catch obvious installation, configuration or workflow failures. The included first-use examples mainly serve as smoke tests. They check the selected path through:

- The installation and dependencies needed by that example.
- Loading its input data and saved configuration.
- Calculating the selected molecular features.
- Fitting the selected models and producing predictions.
- Writing the prediction files and both HTML and Markdown reports.

- After the run, open the report as described below and confirm that you can view it.
- A pass means that the example ran and produced its expected outputs in the tested environment; a high prediction score is not required.
- It does not establish chemical prediction accuracy, scientific validity or support for every descriptor, model and workflow.
- The artificial data's low scores and negative R² remain visible so you can inspect those outputs during first use.

#### 1. Look at the smoke-test files

Open [the data table](example/random_split/dataset/reactions.csv) and [its settings](example/random_split/run.yaml). A CSV is a text table; SMILES is a text representation of a molecular structure.

| Column | Meaning in this example |
| --- | --- |
| `reaction_id` | A unique record ID, used to match outputs to inputs |
| `yield` | Reference yield on a scale from 0 to 1; `0.10` means 10% |
| `reactant_1_smiles`, `reactant_2_smiles` | The two molecular structures used as model inputs |

- All 12 records and their yields are artificial values for first-use smoke tests, not experimental measurements.
- The molecular combinations illustrate the file format and do not establish that these are chemically valid reactions.

- The prepared `run.yaml` selects Morgan fingerprints and three models: Random Forest (`rf`), XGBoost (`xgb`) and LightGBM (`lightgbm`).
- It divides the records into two groups of six, trains on one group and predicts the other, then swaps their roles.
- The first full run fits each model twice, for six small fits in total.

#### 2. Start the program

Start from the project root. If you just followed the installation steps, you are already there. Keep this terminal open while the calculation runs.

Linux Bash:

```bash
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate yonod
export PYTHONUTF8=1
cd example/random_split
mkdir -p ../../config/example
cp -n run.yaml ../../config/example/example_random_split_20261113.yaml
python -u ../../yonod.py
```

Windows Conda PowerShell:

```powershell
conda activate yonod
$env:PYTHONUTF8 = "1"
Set-Location .\example\random_split
New-Item -ItemType Directory -Path ..\..\config\example -Force | Out-Null
if (-not (Test-Path ..\..\config\example\example_random_split_20261113.yaml)) { Copy-Item run.yaml ..\..\config\example\example_random_split_20261113.yaml }
python -u ..\..\yonod.py
```

The program asks `请输入配置文件路径(schema-2 YAML):` (enter the configuration file path). Type the following and press Enter:

```text
../../config/example/example_random_split_20261113.yaml
```

The commands copy the prepared template into `config/example/`, preserving an existing copy. No YAML generator is needed. Run one task at a time, waiting for it to finish before starting another.

#### 3. Check completion and open the report

- At the end, look for `features=ready` and a `complete` or `reused` status for each of `morgan × rf`, `morgan × xgb` and `morgan × lightgbm`.
- In the ordinary workflow, `reused` means matching saved results were available.
- The `[report]` message prints the report locations.
- Check any error message or missing report, even after the process ends.

Still in `example/random_split/`, open `../../result/example/example_random_split_20261113/report/report.html` in your file manager. On desktop Linux you can also use:

```bash
xdg-open ../../result/example/example_random_split_20261113/report/report.html
```

Windows Conda PowerShell:

```powershell
Invoke-Item ..\..\result\example\example_random_split_20261113\report\report.html
```

- The same folder contains `report.md`, a text report you can read in an editor.
- Each model's per-record predictions are under `../../result/example/example_random_split_20261113/runs/run_morgan-<model>/predictions.csv`, where `<model>` is `rf`, `xgb` or `lightgbm`.
- All example outputs are under the project-root `result/example/<task_name>/`; each example has its own task folder.

#### 4. Read the results

- Each record receives a prediction from a model that did not train on that record.
- These are called out-of-fold (OOF) predictions.
- Random splitting can still put the same substrate in both training and evaluation records, so it does not establish performance on unseen substrates.

| Metric | What it tells you |
| --- | --- |
| MAE, mean absolute error | Average size of the prediction error. On this 0 to 1 yield scale, `MAE=0.10` means 10 percentage points. |
| RMSE, root mean squared error | Error in the same units as the target, with greater weight on large errors. |
| R², coefficient of determination | Error compared with using the evaluation data's mean as a reference. It can be negative and is not a percentage of correct predictions. |

- Training scores describe the records used to fit the model; use held-out or external scores to assess prediction.
- The examples retain low scores and negative R².
- Their main purpose is to smoke-test workflow execution; these scores do not establish performance on real chemistry or recommend a model.

### If something goes wrong

| Symptom | Check |
| --- | --- |
| `conda` is not recognized | Confirm Miniconda is installed. Linux: use the correct `conda.sh` location. Windows: reopen an initialized Conda PowerShell session. |
| A terminal command gives `SyntaxError` at `>>>` | You are inside Python. Enter `exit()` to return to the terminal, then run the command there. |
| Python cannot find `yonod.py`, `run.yaml` or `requirements.txt` | Check the current folder. Installation uses the root; the first-use launch uses `example/random_split/`. |
| A path contains spaces, or a YAML file cannot be found | Quote paths in shell commands. Check the actual filename, including extensions; a file ending in `.yaml.txt` is a text file with the wrong extension. |
| `ModuleNotFoundError` | Activate `yonod`, then install the listed requirements there. PyTorch is a separate first installation; AutoGluon is optional. |
| No product column in the CSV wizard | Prepare a manual YAML from `example.yaml`; the wizard currently requires one product column. |
| ID or invalid-input error | Check for missing/duplicate IDs, ambiguous ID text and the generator's exclusion report. Keep the original-to-prepared ID mapping. |
| YAML validation fails | Check indentation, exact field names, paths and the selected protocol. Duplicate keys, aliases and unknown fields are rejected. |
| You want to rerun with changed data or settings | Use a new task name, YAML and result directory. The generator protects existing tasks; frozen external metrics also reject overwriting. Ordinary matching results may be reused. |
| Garbled Chinese paths or output | Keep CSV/YAML in UTF-8 and set `PYTHONUTF8=1` as shown for your shell. |
| A warning appears or the process ends | Read the final statuses and confirm the expected reports exist. Warnings alone do not establish failure or success. |

- For a bug report in [project Issues](https://github.com/thinktraveller/YONOD/issues), include your operating system, Python/package versions, the error text and a small shareable example.
- Remove confidential data and credentials.

## Advanced content

- [Use your own reaction table](#use-your-own-reaction-table)
- [Configuration paths and execution stages](#configuration-paths-and-execution-stages)
- [Choose and run an evaluation workflow](#choose-and-run-an-evaluation-workflow)
- [Read example.yaml line by line](#read-exampleyaml-line-by-line)
- [Adjust numerical conditions](#adjust-numerical-conditions)
- [Search model settings with HPO](#search-model-settings-with-hpo)
- [Additional workflow fields and artifact inputs](#additional-workflow-fields-and-artifact-inputs)
- [Choose what to adjust](#choose-what-to-adjust)
- [Saved examples and reports](#saved-examples-and-reports)
- [Optional models, data and pretrained weights](#optional-models-data-and-pretrained-weights)

### Use your own reaction table

Open the [example CSV dataset](example/random_split/dataset/reactions.csv) to see the column names and record layout. Its values are artificial smoke-test data.

Return to the project root before continuing: use `cd ../..` in Linux Bash or `Set-Location ..\..` in Windows PowerShell if you are still in the example folder.

#### Prepare the CSV

- Save a comma-separated CSV with a header row, using UTF-8 encoding.
- In a spreadsheet editor, use Save As or Export to CSV UTF-8; changing an `.xlsx` filename to `.csv` does not convert it.
- Structure columns need valid SMILES text, not molecular drawings or reagent names.
- Each row should be one record, with a stable, nonempty, unique ID.
- IDs such as `record-001` stay stable through CSV readers.
- Automatic type inference can change ambiguous values such as `001` or `NA`; the generator rejects them when that would happen.

- Check the reference value's meaning and unit.
- Yield, conversion, LC area ratio and selectivity are different targets.
- Use one consistent scale, such as 0 to 1 or 0 to 100, and write numbers without a `%` suffix.
- YONOD does not infer the intended scale or convert it automatically.

- The current interactive generator requires a numerical target and exactly one product SMILES column.
- You can also select reactant columns, other structure columns and numerical conditions.
- If your data have no product column, use the manual YAML route below; the runtime itself supports configurations with `products: []`.
- Whenever product structures are included in the model inputs, they must be available at prediction time.

#### Generate settings, then run them

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

1. Enter the CSV path and a new task name, for example `my_reaction`.

    - Use letters, numbers, `_` or `-`; existing configurations and nonempty task folders are protected from overwriting.

2. Select the stable ID column.

    - It is used for record matching and excluded from the target and model inputs.
    - Then assign the target, reactants, product, other structures and numerical conditions.
    - Follow the indices shown at each prompt; their numbering is not the same in every menu.

3. Check the invalid-input report and the number of retained records.

    - The generator can exclude rows with invalid inputs, so confirm that the remaining table represents the data you intend to use.

4. At the descriptor menu, enter `1` for Morgan.

    - Check the displayed input columns, which default to the selected reactants, other structures and product.
    - Press Enter at `选择 [N]:` to retain them, or enter `y` to edit the descriptor configuration.
    - All included structures must be known at prediction time.

5. At the model menu, enter `2` for Random Forest.

    - Pressing Enter at the descriptor or model menu selects several methods, including ones requiring pretrained weights or optional packages.

6. Add source/DOI notes if available.

    - At the report-format menu, press Enter to retain both HTML and Markdown.
    - Review the optional cleaned-CSV choice and the generated file paths.

- The generator saves `config/my_reaction.yaml` and exits without calculating descriptors, fitting models or predicting.
- Prepared files stay under `result/my_reaction/`: the normalized CSV is in `feature/docs/`, and an invalid-input report, when needed, is in `report/`.
- Supplying an existing YAML to the generator only validates it.

- Open the saved YAML in a text editor before launching it.
- The generated ordinary workflow uses five folds, one repeat and random seed 42, with model-library defaults and no HPO.
- It needs at least five retained records; this is a software minimum, not evidence that a five-record dataset is sufficient for research.
- Check the split design and calculation budget for your question.
- Numerical conditions default to rejecting missing values; configure imputation, scaling and explicit unit conversion when needed.

With the same environment active and from the project root, run `python -u yonod.py` on Linux or `python -u .\yonod.py` on Windows. At its first prompt, enter:

```text
config/my_reaction.yaml
```

Ordinary reports will be under `result/my_reaction/report/`. The generator prepares settings; `yonod.py` validates and executes those settings.

#### Prepare YAML manually

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

- Choose an unused filename.
- Edit the copy's task name, data path, ID and column roles, descriptor input columns, model settings and evaluation design.
- Set `outputs.root` to `./result/my_reaction` and `artifacts.output_dir` to `./result/my_reaction/feature`, retaining `outputs.report_formats: [html, markdown]`.
- The original file is a two-fold, 12-record Morgan/RF smoke configuration with 10 trees; its settings are not research recommendations.
- Launch your edited file through the same `yonod.py` prompt.

### Configuration paths and execution stages

- We recommend keeping run configurations under root `config/`: research YAMLs in `config/<task_name>.yaml` and example copies in `config/example/<task_name>.yaml`.
- Paths beginning `./` inside those files resolve from the project root.
- Explicit `../` paths and relative paths in configurations elsewhere resolve from the YAML's folder; absolute paths stay absolute.
- Example templates use `../../example/<package>/` input paths so their copies in `config/example/` resolve the same files; `../../result/example/<task_name>` saves outputs under the project root.
- Use UTF-8 plain text, spaces for indentation, and check that Windows has not added a `.txt` extension.

| `stage` | Execution |
| --- | --- |
| `features` | Prepare or reuse encodings only; reference labels may be absent. |
| `train` | Train from the verifiable package named by `artifacts.input_manifest`; it does not compute missing encodings. |
| `all` | Prepare encodings and run ordinary internal evaluation. |
| `benchmark` | Dispatch a dedicated evaluation adapter with its required fields. |

- We recommend saving all run results under the project-root `result/`, with an independent `result/<task_name>/` for each task.
- Each research task uses its own `outputs.root: ./result/<task_name>` and a feature child such as `artifacts.output_dir: ./result/<task_name>/feature`, retaining `outputs.report_formats: [html, markdown]`.
- Operational logs and PIDs belong in root `logs/`; final modeling outputs belong under the task, not `derived/`.

### Choose and run an evaluation workflow

- All three first-use examples use artificial records and mainly serve as smoke tests of their respective workflows.
- Run them one at a time.
- Start each command block from the project root; if you are still inside an example, return with `cd ../..` in Bash or `Set-Location ..\..` in PowerShell.
- The commands save each example configuration under the project-root `config/example/`. At the first prompt, enter the saved path shown for that example.
- Report paths below are relative to the project root.

| Your question | Workflow and configuration | What is kept separate |
| --- | --- | --- |
| How well do methods predict held-out records from this table? | Ordinary `stage: all`, `evaluation.protocol: outer_kfold`; [random example](example/random_split/run.yaml) | Training and held-out rows within each fold; repeated substrates can occur on both sides. |
| How do I enforce selected members or whole groups? | `stage: benchmark`, `manifest_outer_cv`; [fixed-membership example](example/fixed_split/run.yaml) | The actual `train`/`valid` members in a saved manifest and the groups represented by its keys. |
| How does a recipe fixed in advance perform on another labeled table? | `stage: benchmark`, `frozen_train_external_test`; [external example](example/external_test/run.yaml) | Each combination fits and predicts without external labels, then reads those labels for scoring. |

#### Random-split internal evaluation

1. Inspect [the 12-row CSV](example/random_split/dataset/reactions.csv) and its [run.yaml](example/random_split/run.yaml).

    - The example uses two reactant columns, Morgan, RF/XGBoost/LightGBM, two folds and one repeat.

2. Run the block for your system and enter `../../config/example/example_random_split_20261113.yaml` when asked.

Linux Bash:

```bash
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate yonod
export PYTHONUTF8=1
cd example/random_split
mkdir -p ../../config/example
cp -n run.yaml ../../config/example/example_random_split_20261113.yaml
python -u ../../yonod.py
```

Windows Conda PowerShell:

```powershell
conda activate yonod
$env:PYTHONUTF8 = "1"
Set-Location .\example\random_split
New-Item -ItemType Directory -Path ..\..\config\example -Force | Out-Null
if (-not (Test-Path ..\..\config\example\example_random_split_20261113.yaml)) { Copy-Item run.yaml ..\..\config\example\example_random_split_20261113.yaml }
python -u ..\..\yonod.py
```

3. Check `features=ready`, all three combinations' `complete` or `reused` statuses, and both reports.

    - The new HTML report is `result/example/example_random_split_20261113/report/report.html`; per-record predictions are under its task's `runs/run_morgan-<model>/predictions.csv`.

4. For your own CSV, generate a new root `config/<task_name>.yaml` or edit an independent copy of [example.yaml](example.yaml).

    - Set the ID, label, input columns, methods and isolated outputs.
    - `evaluation.n_splits` is an integer of at least 2; `n_repeats` is at least 1; `shuffle` is a Boolean; `seed` controls shuffling.
    - Each shuffled repeat uses `seed + repeat - 1`.
    - With `shuffle: false`, repeats use the same row partition.
    - The valid encoded row count must be at least `n_splits`.

- The ordinary runtime always constructs KFold splits.
- Merely writing `manifest_outer_cv` in an ordinary `all/train` configuration does not import your manifest; use the dedicated benchmark route below for actual fixed members.

#### Fixed membership and grouped evaluation

1. Read the [role guide](example/fixed_split/splits/membership_guide.csv), [actual 24-row manifest](example/fixed_split/splits/fixed_split_manifest.csv) and [configuration](example/fixed_split/run.yaml).

    - Every fold lists all 12 sample IDs.
    - Fold 1 holds out `block_A`; fold 2 holds out `block_B`.

2. Keep `stage: benchmark`, `evaluation.protocol: manifest_outer_cv`, the declared grouping, `evaluation.split_manifest`, and the `benchmark.task_state` block together.
3. Launch from the project root, then enter `../../config/example/example_fixed_split_20261113.yaml`.

Linux Bash:

```bash
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate yonod
export PYTHONUTF8=1
cd example/fixed_split
mkdir -p ../../config/example
cp -n run.yaml ../../config/example/example_fixed_split_20261113.yaml
python -u ../../yonod.py
```

Windows Conda PowerShell:

```powershell
conda activate yonod
$env:PYTHONUTF8 = "1"
Set-Location .\example\fixed_split
New-Item -ItemType Directory -Path ..\..\config\example -Force | Out-Null
if (-not (Test-Path ..\..\config\example\example_fixed_split_20261113.yaml)) { Copy-Item run.yaml ..\..\config\example\example_fixed_split_20261113.yaml }
python -u ..\..\yonod.py
```

4. Check two successful RF fold records.

    - The HTML report is `result/example/example_fixed_split_20261113/report/benchmark_report.html`; the copied manifest and predictions are in that task's `docs/manifests/` and `docs/predictions/`.

5. For your data, prepare stable IDs and group keys before assigning folds.

    - Save a new CSV or Parquet manifest, update its dataset checksum and members, and reference it from a new root `config/` YAML.
    - Keep group columns out of the descriptor inputs unless they intentionally represent available predictors.
    - Set the new task's outputs to `./result/<task_name>` and `./result/<task_name>/feature`.

- The current interface requires at least two folds.
- Every fold must contain nonempty training and validation sets and the same sample IDs as the CSV.
- A group must not cross the training/validation boundary, and every sample must be `valid` exactly once per repeat.
- A one-off eight-training/four-test list cannot replace this CV manifest.
- Fixed seeds or disabling shuffle do not specify arbitrary selected members.

When `evaluation.split_manifest` is absent, the dedicated executor creates a manifest from `evaluation.grouping`. This supports five strategies:

| `grouping.strategy` | Settings and behavior |
| --- | --- |
| `repeated_kfold` | No molecular grouping; shuffled KFold in original CSV row order, with `seed + repeat - 1`. The dedicated branch always shuffles; `evaluation.shuffle` does not control it. |
| `precomputed_column` | Required `group_column`: use a nonempty existing group key per row, adding the `precomputed:` prefix internally. |
| `component_holdout` | Required nonempty `component_cols`: canonicalize the listed structure columns and use their ordered joint identity as the group. With several columns this groups the combination, not each component independently. |
| `substrate_scaffold` | Required `substrate_col`: use its chiral Murcko scaffold. Acyclic structures use a canonical-structure fallback. |
| `reaction_fingerprint_cluster` | `reaction_cols` defaults to the molecular role columns; `similarity_threshold` defaults to `0.7`, range `(0, 1]`; Morgan grouping `radius` defaults to `2`, at least `0`; `n_bits` defaults to `2048`, at least `8`. The implementation clusters concatenated molecular components using Tanimoto/Butina. |
| All four genuinely grouped strategies | Optional `max_group_fraction`, default `0.8`, range `(0, 1]`: reject a generated split if its largest group exceeds that fraction. At least `n_splits` groups are needed. Complete groups are greedily balanced across folds. |

- Invalid grouping structures receive explicit fallback keys; they are not silently removed or switched to ordinary KFold.
- A grouping name alone does not prove that a chemical substrate or source is absent from the other side.
- With an imported manifest, the imported group IDs and memberships determine the split; changing the grouping settings does not regenerate those imported members.

The manifest's required columns are:

| Column | Meaning and preparation rule |
| --- | --- |
| `run_id` | One nonempty source-run identifier; import keeps it as `source_run_id` and uses the new task's run ID in the task-local copy. |
| `split_id` | One split-plan identifier across the file. |
| `sample_id` | The stable CSV record ID; every fold contains the same full population. |
| `group_id` | The audited group key; no group may appear on both sides within a fold. |
| `group_strategy` | The declared grouping strategy. |
| `repeat`, `fold` | Repeat and fold numbers; prepare them from 1, with the declared number of repeat/fold pairs. |
| `role` | `train` or `valid`, not `test` or the role guide's explanatory `held_out`. |
| `seed` | The split seed recorded for this repeat. |
| `source_row_index` | Zero-based data-row position in the original CSV, excluding its header; keep it accurate when preparing the manifest. |
| `dataset_sha256` | SHA-256 of the exact CSV file bytes, repeated consistently. |
| `grouping_params_json` | A JSON object recording the grouping settings; correctly quote it inside CSV. |
| Optional `population_id`, `split_hash` | Extra provenance fields used by generated manifests and newer protocols. |

For a file named `dataset/my_reactions.csv`, calculate its hash before completing the manifest. The following only reads the file.

Linux Bash:

```bash
sha256sum dataset/my_reactions.csv
```

Windows Conda PowerShell:

```powershell
Get-FileHash -Algorithm SHA256 -LiteralPath .\dataset\my_reactions.csv
```

- The importer checks the dataset hash, source run/split identity, population membership, fold coverage and group separation.
- You must also verify that source-row positions, strategy notes and group definitions accurately describe your CSV; the importer does not independently reconstruct their chemical meaning.

#### Frozen external testing

1. Inspect the [12-row training table](example/external_test/dataset/train.csv), [six-row external table](example/external_test/dataset/external.csv), [audit record](example/external_test/audit/external_audit.json) and [run.yaml](example/external_test/run.yaml).

    - Freeze descriptor inputs, model parameters and the scoring plan before examining external outcomes.

2. Keep `stage: benchmark`, `frozen_train_external_test`, `evaluation.seed` and the external-test block.

    - Do not include `n_splits`, `n_repeats`, `shuffle`, `grouping`, `split_manifest` or `population_manifest`.

3. Launch from the project root and enter `../../config/example/example_external_test_20261113.yaml`.

Linux Bash:

```bash
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate yonod
export PYTHONUTF8=1
cd example/external_test
mkdir -p ../../config/example
cp -n run.yaml ../../config/example/example_external_test_20261113.yaml
python -u ../../yonod.py
```

Windows Conda PowerShell:

```powershell
conda activate yonod
$env:PYTHONUTF8 = "1"
Set-Location .\example\external_test
New-Item -ItemType Directory -Path ..\..\config\example -Force | Out-Null
if (-not (Test-Path ..\..\config\example\example_external_test_20261113.yaml)) { Copy-Item run.yaml ..\..\config\example\example_external_test_20261113.yaml }
python -u ..\..\yonod.py
```

4. RF fits once on all 12 training rows and predicts six external rows.

    - Read `result/example/example_external_test_20261113/report/external_test_report.html`, `docs/predictions/morgan__rf.parquet` and `docs/metrics/external_test_metrics.csv` under that task root.
    - There is no external CV; a saved score is not an independent confidence interval.

5. For a real task, make independent training, external and audit files, then a new root `config/` YAML with new task/output identities.

    - Use `external_test.column_map` from each training structure column to its external column.
    - The target is specified separately, through training `column_roles.label` and external `label_col`.

| Field under `evaluation.external_test` | Meaning |
| --- | --- |
| `path` | External CSV path. |
| `sample_id_col` | Unique, nonempty external ID; training and external IDs must be disjoint. |
| `label_col` | External numerical reference target for final scoring. |
| `source_id_col` | Unique, nonempty source-record ID. If the audit gives `physical_dataset_id`, each value must start with that ID plus `:`. |
| `group_col` | External group column; at least two nonempty groups are required. The current report records group count and retains keys in predictions for later grouped analysis. |
| `column_map` | One-to-one mapping from training feature-column names to external column names; covers all selected structure inputs, not target labels. |
| `expected_sha256` | Exact external CSV SHA-256. |
| `audit_manifest` | JSON provenance/overlap audit path. |
| `audit_manifest_sha256` | Exact audit JSON SHA-256. |

- The audit needs `candidate_source_sha256` and `exact_canonical_overlap` with `amine`, `acid` and `product` all zero.
- If present, `row_count` must match; `physical_dataset_id` controls the source-ID prefix.
- Prepare and review the audit from the actual data: the adapter verifies declarations and file identity, but does not recalculate the molecular structure intersections for you.
- Calculate hashes with the same Bash/PowerShell commands above, substituting the external and audit paths, then enter them in the new YAML.
- Do not copy the first-use example audit's zero counts.

- This adapter currently accepts only static descriptors, excludes numeric conditions, OHE, AutoGluon, HPO and external-test early stopping, and does not save the ordinary complete feature/final-model package.
- It rejects existing external run output before fitting and does not resume or overwrite it; use a new task name and output directory.
- The example includes product structure, which must be known at prediction time.
- Separate files and zero exact structure overlap do not establish independent experimental provenance or distributional independence.

### Read example.yaml line by line

- [example.yaml](example.yaml) is the sole complete YAML specification.
- The table explains every active, nonblank line in the current 104-line file by its full field path.
- YAML indentation assigns children to their parent; `-` starts a list item, `[]` is an empty list, `{}` an empty mapping, `null` a null value, and `true`/`false` Boolean values.
- Use spaces, not tabs.
- Line numbers refer to this checkout and change if the specification is edited.

| Line | Full field path | Meaning and what you can change |
| --- | --- | --- |
| 8 | `schema_version` | `"2.0"`; keep the schema version supported by the loader. |
| 9 | `project_name` | Task name; replace with a unique name and matching isolated output paths. |
| 10 | `stage` | `all`; choose `features`, `train`, `all` or `benchmark` for the required route. |
| 12 | `dataset` | Container for the input table and column contracts. |
| 13 | `dataset.path` | Fixture CSV path; replace with your input CSV. |
| 14 | `dataset.sample_id_col` | Stable, nonempty, unique record-ID column. |
| 15 | `dataset.column_roles` | Container assigning columns their purposes. |
| 16 | `dataset.column_roles.label` | Numerical reference target; required for training/scoring. |
| 17 | `dataset.column_roles.reactants` | Reactant structure column list. |
| 18 | `dataset.column_roles.products` | Product structure column list; empty in this fixture. |
| 19 | `dataset.column_roles.others` | Other molecular structure columns; empty here. |
| 20 | `dataset.column_roles.conditions` | Ordered numerical condition columns; empty here. |
| 21 | `dataset.column_roles.categoricals` | Categorical input columns, for an explicitly selected encoding. |
| 27 | `dataset.numeric_conditions` | Container for numerical preprocessing declarations. |
| 28 | `dataset.numeric_conditions.defaults` | Rules shared by numerical condition columns. |
| 29 | `dataset.numeric_conditions.defaults.missing` | Here `strategy: error`; other strategies are `mean`, `median`, `constant`. |
| 30 | `dataset.numeric_conditions.defaults.scaling` | `none`; alternatives `standard` and `minmax`, learned from training rows. |
| 31 | `dataset.numeric_conditions.columns` | Per-source-column overrides; empty here because no numerical conditions are selected. |
| 35 | `descriptors` | List of separate encoding candidates, not an instruction to concatenate candidates together. |
| 36 | `descriptors[0].id` | Unique candidate ID, `morgan`; output paths use this identity. |
| 37 | `descriptors[0].descriptor` | Encoding algorithm, `morgan`; select an implemented registry name. |
| 38 | `descriptors[0].lifecycle` | `static_descriptor`; use the algorithm's supported lifecycle, not an arbitrary switch. |
| 39 | `descriptors[0].mode` | `concat` joins the listed component encodings in order; accepted modes depend on the descriptor. |
| 40 | `descriptors[0].columns` | Ordered source inputs used by this descriptor. |
| 41 | `descriptors[0].params` | Descriptor-specific parameters; empty for Morgan, which exposes none in this YAML interface. |
| 43 | `artifacts` | Feature-artifact locations. |
| 44 | `artifacts.output_dir` | This task's `feature/` child; change with the task name. `train` additionally requires `input_manifest`. |
| 46 | `models` | Selected model IDs; here only `rf`. |
| 47 | `model_params` | Parameter sections for selected models only. |
| 52 | `model_params.rf` | Settings for Random Forest. |
| 53 | `model_params.rf.estimator` | Constructor parameters for the model factory. |
| 54 | `model_params.rf.estimator.n_estimators` | `10` trees; a small engineering budget. |
| 55 | `model_params.rf.estimator.max_depth` | `null`, so no explicit depth limit. |
| 56 | `model_params.rf.estimator.min_samples_leaf` | `1`, minimum samples per leaf. |
| 57 | `model_params.rf.estimator.max_features` | `1.0`, all available feature columns considered at a split. |
| 58 | `model_params.rf.estimator.random_state` | Model seed `42`; distinct from the data-split seed. |
| 59 | `model_params.rf.estimator.verbose` | Library verbosity, `0`. |
| 66 | `hpo` | Container for optional model-parameter search. |
| 67 | `hpo.enabled` | `false`; no search. A disabled HPO block accepts only `enabled`. |
| 86 | `evaluation` | Container for training/held-out membership rules. |
| 91 | `evaluation.protocol` | `outer_kfold`, the ordinary random-CV route. |
| 92 | `evaluation.n_splits` | `2` folds; integer at least 2 and no more than valid rows. |
| 93 | `evaluation.n_repeats` | `1` complete CV repeat; integer at least 1. |
| 94 | `evaluation.shuffle` | `true`; shuffle rows before ordinary KFold. |
| 95 | `evaluation.seed` | Split seed `42`; does not specify arbitrary member IDs. |
| 97 | `outputs` | Result locations and report settings. |
| 98 | `outputs.root` | Task root: this smoke test uses `result/example/yonod_schema2_smoke/`; use `result/<task_name>/` for your own tasks. |
| 99 | `outputs.report_formats` | Both `html` and `markdown`; retain both for normal use. |
| 100 | `outputs.diagnostics` | Optional training-versus-held-out diagnostic settings. |
| 101 | `outputs.diagnostics.enabled` | `true`; enable diagnostics without changing the evaluation membership. |
| 102 | `outputs.diagnostics.save_train_predictions` | `false`; whether to save per-record training predictions. It cannot be true while diagnostics are disabled. |
| 103 | `metadata` | Human notes, not model input. |
| 104 | `metadata.notes` | Explains the artificial smoke fixture and task isolation. Replace with your provenance and purpose. |

- Blank lines only improve readability.
- Full-line `#` comments and text after `#` are not executed.
- Lines 1 to 7 explain launch locations; 22 to 26 explain numerical conditions; 33 to 34 explain candidates and lifecycles; 48 to 51 explain model sections and resources; 61 to 65 explain HPO scope; 68 to 84 contain an inactive HPO example; 87 to 90 explain protocol selection.
- The commented HPO fields are covered below.
- Do not uncomment a second `hpo` key beside the existing one: duplicate YAML keys are rejected.

### Adjust numerical conditions

- Declare source columns in `dataset.column_roles.conditions` in their intended input order.
- They cannot overlap the ID, target, molecular or categorical roles.
- Rules live under `dataset.numeric_conditions`; every key supported by this interface is listed here.

| Location | Accepted keys, values and behavior |
| --- | --- |
| `defaults` | Only `missing` and `scaling`; default to `missing.strategy: error` and `scaling: none`. |
| `columns.<source_column>` | Only `name`, `missing_values`, `missing`, `scaling`, `unit`. The source column must already be in `conditions`. |
| `name` | Nonempty output name, unique among numerical outputs; defaults to the source name. |
| `missing_values` | A list of distinct strings and/or `null`, default `[]`. Declare `null` to recognize nulls, and a string such as `""` for that exact raw token. Undeclared nonnumeric/null values are errors. |
| `missing.strategy` | `error`, `mean`, `median`, `constant`; it may also be supplied as the strategy string. Mean/median require at least one observed training value. |
| `missing.constant` | Required finite number only with `constant`. It fills missing values on the converted numerical scale. |
| `scaling` | `none`; `standard` subtracts training mean and divides by population SD (`ddof=0`); `minmax` uses training minimum/range. A zero training scale is replaced by 1, retaining the column. |
| `unit` | Only `source`, `target`, `scale`, `offset`. Source/target strings are required when specified; finite `scale` defaults to 1 and cannot be 0; finite `offset` defaults to 0. Converted value = raw value × scale + offset. Equal source/target units require identity conversion. Omission means dimensionless identity conversion. |

- Unit conversion is explicit arithmetic, not automatic unit recognition.
- Imputation and scaling fit only on the current training subset, including inner training folds during HPO.
- Validation and future data use that saved state.
- Frozen external testing does not currently accept numerical-condition inputs.

### Search model settings with HPO

- HPO searches model parameters within training data.
- It does not choose laboratory conditions.
- The project integrates RF, XGBoost and LightGBM, using nested folds so outer held-out labels do not select the winning settings.
- Every encoding × selected model × outer fold gets its own study; the objective is the unweighted arithmetic mean of inner-fold scores.

| HPO field | Values and restrictions |
| --- | --- |
| `enabled` | Boolean. Omission or `false` disables HPO; when false no other HPO keys are permitted. |
| `models` | Nonempty unique subset of top-level `models`, containing only `rf`, `xgb`, `lightgbm`. |
| `sampler` | Only `tpe`. |
| `seed` | Required integer search seed. |
| `objective.metric`, `objective.direction` | `rmse`/`mae` with `minimize`; `r2`/`kendall_tau` with `maximize`. Both must be explicit. |
| `inner_cv.n_splits` | Integer at least 2, not greater than the current outer-training row count; grouped searches also need this many groups. |
| `inner_cv.seed`, `inner_cv.shuffle` | Explicit integer and Boolean. Grouped strict HPO uses GroupKFold and requires `shuffle: false`; `repeated_kfold` is the ungrouped alternative. |
| `budget.max_trials` | Required positive integer trial budget per study. |
| `budget.study_timeout_s` | Optional finite positive soft study-time limit; an already running trial can finish. |
| `budget.task_timeout_s` | Optional finite positive cumulative task-time budget, retained across resumed sessions; it is a soft scheduling limit, not a process-kill timer. |
| `search_spaces` | Exactly one entry for every `hpo.models` model. |
| `search_spaces.<model>.preset` | Optional `compact-v1`; no other preset name is accepted. |
| `search_spaces.<model>.parameters` | Parameter-name-to-distribution mapping; optional with a preset, otherwise nonempty. Same-name entries replace the whole preset distribution. Searched values override same-name fixed estimator values, with that override recorded. |
| `final_model.enabled` | Optional Boolean, default false. If false, no final-population/budget keys may accompany it. |
| `final_model.development_population_id` | Required nonempty identifier when final-model fitting is enabled. |
| `final_model.budget` | Independent final-purpose search budget with `max_trials` and optional study/task timeout keys. A separate study selects settings on the development population, then the final model fits that full population; it does not create an independent test score. |

- A distribution has `type: int`, `float` or `categorical`.
- Integer/float distributions require finite `low` and `high` with `low <= high`; optional `step` must be positive and exactly divide the interval.
- Integer bounds and steps must be integers; the integer default step is 1.
- Optional `log` is Boolean and requires positive `low`; integer log requires step 1, and float log cannot also declare step.
- Categorical distributions accept only a nonempty `choices` list of distinct safe scalars (`null`, strings, Booleans and finite numbers).
- Arbitrary expressions and callable distributions are not accepted.

| Model | All currently searchable estimator keys |
| --- | --- |
| RF | `n_estimators`, `max_depth`, `min_samples_leaf`, `min_samples_split`, `max_features`, `max_samples`, `min_impurity_decrease`, `ccp_alpha` |
| XGBoost | `n_estimators`, `learning_rate`, `max_depth`, `min_child_weight`, `subsample`, `colsample_bytree`, `gamma`, `reg_alpha`, `reg_lambda`, `max_delta_step` |
| LightGBM | `n_estimators`, `learning_rate`, `num_leaves`, `min_child_samples`, `min_child_weight`, `subsample`, `colsample_bytree`, `reg_alpha`, `reg_lambda`, `min_split_gain`, `max_depth` |

- `compact-v1` searches RF trees 100 to 800 by 100, depths from `null/4/8/12/20/32`, leaf minimum 1 to 20 and feature fraction 0.3 to 1.0.
- For XGBoost it searches trees 100 to 800 by 100, learning rate 0.01 to 0.2 logarithmically, depth 2 to 10, child weight 1 to 20 logarithmically, row fraction 0.6 to 1.0 and feature fraction 0.5 to 1.0.
- For LightGBM it searches trees 100 to 800 by 100, learning rate 0.01 to 0.2 logarithmically, leaves 8 to 64, child samples 5 to 50 and feature fraction 0.5 to 1.0.
- These are software presets, not scientifically validated budgets for every dataset.

- Ordinary HPO requires `outer_kfold`; strict HPO requires `stage: benchmark` with `manifest_outer_cv`.
- It is unavailable for frozen external testing and fixed paper-reproduction protocols.
- HPO currently rejects runtime early stopping; strict HPO also rejects nonempty model `fit` sections.
- The feature-only stage validates the declaration without starting a study.
- LightGBM search rejects fixed aliases of searched parameters, requires positive fixed `subsample_freq` when searching `subsample`, and checks searched `num_leaves` against fixed positive `max_depth`.
- The XGBoost preset requires `booster: gbtree`.

### Additional workflow fields and artifact inputs

| Field | Current use |
| --- | --- |
| `artifacts.input_manifest` | Required for `stage: train`; path to an existing, integrity-checked feature artifact manifest. It is not a CSV path. The stage does not calculate missing encodings. Do not substitute an artifact from another population or input contract. |
| `benchmark.task_state.backend` | Required value `sqlite` for `stage: benchmark`. |
| `benchmark.task_state.resumable` | Required value `true` for `stage: benchmark`. The manifest workflow uses task state for audited recovery; the frozen external adapter still refuses existing run output and does not resume it. |
| `benchmark.population_id`, `benchmark.dataset_id` | Optional nonempty provenance identifiers. They describe the population/data version; their strings do not prove independent origin. |
| `benchmark.reproduction_protocol` | Optional mapping for dedicated, already-established reproduction behavior. It is not required by these first-use workflows; do not copy archival reproduction settings into a new ordinary task. |
| `outputs.diagnostics.enabled` | Boolean, default enabled for supported models in ordinary/manifest workflows. Current training/held-out diagnostics support `rf`, `xgb`, `lightgbm`; SVM and AutoGluon are recorded as unsupported. Frozen external scoring does not create these paired diagnostics. |
| `outputs.diagnostics.save_train_predictions` | Boolean, default false; save per-record training predictions for supported diagnostics. It cannot be true with diagnostics disabled. Turning diagnostics off does not turn off normal held-out scoring. |

- The schema recognizes the following top-level keys under `evaluation`: `protocol`, `n_splits`, `n_repeats`, `shuffle`, `seed`, `grouping`, `split_manifest`, `population_manifest`, `metrics`, `external_test`.
- The three current launch routes use only the fields described above.
- The generic schema accepts `population_manifest` and `metrics`, but these routes do not use them to select populations or customize scoring.
- Omit them here.
- Frozen external testing explicitly rejects `population_manifest` along with the CV fields.
- A field must be implemented by the selected adapter as well as accepted by the schema.

### Choose what to adjust

- Start with a fixed, reviewable configuration.
- Change the setting that answers your question, and keep the evaluation members and metric definition consistent when comparing methods.
- The full descriptor and model references below identify the supported names and protocol restrictions.

| What you want to change | YAML location | Effect to consider |
| --- | --- | --- |
| Input information | `dataset.column_roles`, `descriptors[].columns` | Select which structures, conditions or categories the model can use. An input must be available when predicting. |
| Ordinary random evaluation | `evaluation.n_splits`, `n_repeats`, `shuffle`, `seed` | Changes membership or repetition; more folds/repeats increase the number of fits. Fixed membership and external testing follow their own contracts. |
| Count Morgan representation | `descriptors[].params.radius`, `fp_size`, `profile` for `mfp` | Changes count fingerprints; binary `morgan` does not expose these YAML parameters. |
| Tree budget and complexity | `model_params.<model>.estimator.n_estimators`, `max_depth` | Controls model size for the supported tree models. Types and valid limits differ by library. |
| Random Forest sampling | `model_params.rf.estimator.max_features`, `min_samples_leaf`, `max_samples` | Controls feature selection, leaf size and bootstrap sampling; check the RF reference for dependencies. |
| Boosting | XGBoost/LightGBM `estimator.learning_rate`, row/column sampling and regularization | Changes how boosting proceeds. Use the names accepted by the selected model. |
| Support vector regression | `model_params.svm.estimator.C`, `epsilon`, `kernel`, `gamma` | Controls the penalty, insensitive error band and kernel. Scaling and PCA are separate preprocessing settings. |
| AutoGluon budget | `model_params.autogluon.fit.time_limit`, `presets`, `num_cpus` | Controls training time, recipe and CPU resources; it requires the optional dependency. |
| Automatic parameter search | `hpo.enabled`, `hpo.budget.max_trials`, `hpo.search_spaces` | Runs an inner search within each outer training fold for supported models. Set a budget before launching. |
| Reports | `outputs.report_formats`, `outputs.diagnostics` | Selects reports and optional training diagnostics; diagnostics do not replace held-out evaluation. |

- Only declare parameters for selected models.
- Use a new task name and separate output directory for a changed research configuration.
- Thread settings depend on your machine; tasks and folds still run in sequence.

### Saved examples and reports

[Offline guide](example/index.html) · [Operating instructions](example/README.md) · [Download all examples](example/archives/all_examples.zip)

- The ZIPs include `completed_run/` reference snapshots.
- New example runs write to the project-root `result/example/<task_name>/`; the random example's historical `report/` and `evidence/` remain reference material.
- The saved [random](example/random_split/verification.json), [fixed-membership](example/fixed_split/verification.json) and [external](example/external_test/verification.json) verification records document Linux execution, not Windows runtime acceptance.
- For unattended execution and status checks, use the Bash/PowerShell commands in the example instructions, keeping logs and PIDs in root `logs/`.

- The ordinary random report scores pooled OOF predictions within each repeat; fixed-membership reports summarize fold metrics.
- These R² values use different aggregation rules, so do not rank scores across the routes as though the evaluation were identical.
- Artificial values, low scores and negative R² remain visible; they do not establish performance on real chemistry.

### Optional models, data and pretrained weights

Available models are `rf`, `xgb`, `lightgbm`, `svm` and `autogluon`. Install AutoGluon only when selecting it. With `yonod` active, the following command works in Linux Bash and Windows Conda PowerShell:

```text
python -m pip install "autogluon.tabular[lightgbm,catboost]==1.1.1"
```

- Encodings without neural pretrained weights include `morgan`, count-based Morgan `mfp`, `maccs`, `maf`, `rdkit2d`, `drfp` and categorical `ohe`.
- DRFP requires reaction inputs; OHE and numerical preprocessing learn from the corresponding training fold.
- HPO is integrated for RF, XGBoost and LightGBM.

- The [dataset notes](dataset/README.md) describe data sources, target definitions and licenses.
- Additional distribution is available through [YONOD-datasets](https://huggingface.co/datasets/thinktraveller/YONOD-datasets) and [YONOD-weights](https://huggingface.co/thinktraveller/YONOD-weights).
- From the root with `yonod` active, these commands work on both systems:

```text
python -m pip install "huggingface_hub>=0.34,<1.0"
hf download thinktraveller/YONOD-datasets --repo-type dataset --include "USPTO/**" --local-dir dataset
```

- The dataset command downloads only `USPTO/`.
- Some source tables use tabs despite their CSV extension; read their source notes and prepare comma-separated data before using the generator.
- Complete any access authorization required by the service.
- Download only the assets you intend to use; the first-use examples already include their inputs.

#### Download pretrained descriptor weights

**Before selecting `fisd`, `chemical_vae` or `molmetalm`, download their pretrained weights from [Hugging Face](https://huggingface.co/thinktraveller/YONOD-weights).** A source clone supplies configuration and asset notes, but does not include the binary weights. The Morgan smoke tests do not require them.

With `yonod` active, run only the command for the descriptor you need from the project root. These commands work in Linux Bash and Windows Conda PowerShell, using the CLI installed above:

```text
hf download thinktraveller/YONOD-weights --include "FISD/**" --local-dir WEIGHTS
hf download thinktraveller/YONOD-weights --include "chemical_vae/**" --local-dir WEIGHTS
hf download thinktraveller/YONOD-weights --include "MolMetaLM-base/**" --local-dir WEIGHTS
```

The three commands restore `WEIGHTS/FISD/`, `WEIGHTS/chemical_vae/` and `WEIGHTS/MolMetaLM-base/`, respectively. To download all weights together, use `hf download thinktraveller/YONOD-weights --local-dir WEIGHTS`. See the [Hugging Face CLI guide](https://huggingface.co/docs/huggingface_hub/v0.36.0/guides/cli) for download options.

| Pretrained encoding | Required assets |
| --- | --- |
| FISD | Download `FISD/` from [YONOD-weights](https://huggingface.co/thinktraveller/YONOD-weights) into `WEIGHTS/FISD/` ([upstream source](https://github.com/KeantChen/FISD/tree/main/model)): `qm_9_mse_model.pth`, `qm_9_cos_model.pth`, `qm_9_2in1_model.pth`. |
| MolMetaLM | Model, configuration and tokenizer from [MolMetaLM-base](https://huggingface.co/wudejian789/MolMetaLM-base) in `WEIGHTS/MolMetaLM-base/`; see the [upstream project](https://github.com/CSUBioGroup/MolMetaLM). |
| Chemical VAE | A verified v5 conversion manifest under `WEIGHTS/chemical_vae/<asset>/v5/` and its adjacent, hash-matched `encoder_state_dict.pt`. ZINC and `zinc_properties` assets are distinct. |

- A configuration, tokenizer or manifest alone does not supply the binary weights.
- Chemical VAE uses the frozen PyTorch encoder's raw `z_mean_sample`; it does not train a VAE or load a decoder/property head, and ordinary execution does not need TensorFlow/Keras.
- See the [weight notes](WEIGHTS/README.md) and [Chemical VAE loader](yonod/descriptors/chemical_vae.py).

## Developer content

- [Develop and verify a change](#develop-and-verify-a-change)
- [Descriptor parameters and protocol support](#descriptor-parameters-and-protocol-support)
- [Add a new descriptor](#add-a-new-descriptor)
- [Model parameters and their execution boundary](#model-parameters-and-their-execution-boundary)
- [Add a new model](#add-a-new-model)

### Develop and verify a change

- Start from [AGENTS.md](AGENTS.md) and the implementation for the workflow you intend to support.
- The schema-2 runtime goes from `yonod.py` to the configuration loader, then to ordinary `pipeline/` stages or a dedicated `benchmark/` adapter.
- Editing an older adapter under `yonod/models/` alone does not extend the schema-2 model factory.

1. Define the new provider's input type, output shape, supported parameters, missing-input behavior and protocol support.

    - Keep pure configuration validation separate from descriptor/model execution.

2. Add its implementation and all relevant registration/dispatch points listed below.

    - Import optional dependencies only when selected; record package versions and asset identity where applicable.

3. Preserve sample IDs, masks, column ordering, split identity, hashes and atomic publication.

    - Fit learned transforms and model selection only on the corresponding training data.
    - A cached result must change identity when its inputs or effective settings change.

4. Put local validation scripts under `_verify/`.

    - Check invalid/empty inputs, parameter rejection, output alignment, deterministic fixed seeds, fold isolation, resume behavior and report paths.
    - Begin with configuration and mocked execution checks; a training smoke is a separate execution check.

5. Check Linux and Windows path handling with `pathlib.Path`, UTF-8 files and shell-specific launch instructions. Record which operating system was actually run.

For the existing engineering smoke, activate `yonod` and run this from the root (the command is the same in Linux Bash and Windows Conda PowerShell):

```text
python yonod.py --config example.yaml
```

- This command computes Morgan features and fits RF on the synthetic fixture.
- A new descriptor or model needs its own saved schema-2 YAML under `config/`, a unique task name and isolated `result/<task_name>/` outputs before testing it.
- Formal tasks use the saved-YAML prompt procedure.
- A smoke pass verifies execution, not scientific validity or acceptance of an untested platform.

- Keep engineering support and backups in `backup/`, operational logs/PIDs in root `logs/`, and maintained documentation in the independent `project-docs/` repository.
- Explicitly stage task-owned files and commit each changed repository separately; do not commit `_verify/`, `config/`, `backup/` or external reference material.

### Descriptor parameters and protocol support

- The ordinary schema-2 feature registry is defined in [registry.py](yonod/descriptors/registry.py).
- A descriptor entry uses the fields below.
- Some arguments accepted by a descriptor's Python constructor are not exposed through YAML.

| Field | What you can change |
| --- | --- |
| `id` | A distinct name for this feature candidate. Use different IDs for different settings; IDs must be unique within the task. |
| `descriptor` | One of the ten registered names listed below. |
| `lifecycle` | Fixed by the provider: `static_descriptor` for nine structure encodings; `fold_transform` for OHE. Ordinary configuration may omit it; benchmark configurations require it explicitly. |
| `columns` | Ordered input columns. For ordinary molecular encodings, an empty/omitted list selects all declared `reactants`, `products` and `others`; explicit lists may select a subset. OHE requires a nonempty, unique list. The target must never be an input. Benchmark configurations require explicit molecular-role columns. |
| `mode` | The provider's supported combination rule, as listed below. Multiple descriptor entries remain separate candidates. |
| `params` | Only the descriptor-specific keys listed below are exposed by the ordinary registry. Omit it or use `{}` for descriptors without exposed parameters. |
| `extra_reactants` | In the ordinary DRFP route, add declared structure columns to the reaction's reactant side; default empty. DRFP otherwise uses `dataset.column_roles.reactants/products`, rather than taking its reaction membership from `columns`. Both benchmark adapters reject nonempty `extra_reactants`. Other ordinary descriptors do not consume it. |

- The dimensions below describe the structure/categorical block, before numerical conditions are appended.
- For `concat`, a fixed per-column dimension is multiplied by the number of selected columns.
- OHE learns its width separately in each training fold.

| Descriptor | Ordinary lifecycle / mode | YAML `params` and output |
| --- | --- | --- |
| `morgan` | `static_descriptor` / `concat` | No exposed parameters. Binary Morgan, fixed radius 2 and 1,024 values per column. |
| `mfp` | `static_descriptor` / `concat` | `radius`, `fp_size`, `profile`; count Morgan, `fp_size` values per column. |
| `maccs` | `static_descriptor` / `concat` | No exposed parameters. 166 MACCS keys per column; RDKit bit 0 is dropped. |
| `maf` | `static_descriptor` / `sum` | No exposed parameters. Sum the selected molecular components' binary fingerprints into one 128-value block, using fixed radius 2. |
| `rdkit2d` | `static_descriptor` / `concat` | No exposed parameters. The current fixed list has 210 values per column. The implementation uses the first component of a multi-component cell and replaces nonfinite calculated values with zero. |
| `drfp` | `static_descriptor` / `reaction` | No exposed parameters. A 2,048-value reaction block; membership comes from reaction roles and optional `extra_reactants`. |
| `fisd` | `static_descriptor` / `concat` | No exposed parameters. 50 values per column. Weights are read from `WEIGHTS/FISD/`, with the implementation's legacy-path fallback. |
| `molmetalm` | `static_descriptor` / `concat` | No exposed parameters. Nominally 768 values per column; the loaded checkpoint's hidden size determines the actual width. Weights are read from `WEIGHTS/MolMetaLM-base/`. |
| `chemical_vae` | `static_descriptor` / `concat` | Six exposed keys listed below. Verified frozen encoder output has 196 values per column. |
| `ohe` | `fold_transform` / `concat` | `missing_policy`, `dtype`, `handle_unknown`; width is the sum of category counts learned from the training fold. |

#### Every exposed descriptor-specific parameter

| Descriptor / parameter | Default and accepted values | Meaning |
| --- | --- | --- |
| `mfp.radius` | `3`; integer ≥ 0 | Morgan neighborhood radius. |
| `mfp.fp_size` | `1024`; integer ≥ 8 | Number of count-fingerprint bins per input column. |
| `mfp.profile` | `standard`; `standard` or `vjethbkm` | `standard` uses ordinary input normalization and valid-row masking. `vjethbkm` encodes raw CSV component values and retains every row, representing blank/invalid components by zero blocks. This changes input/missingness semantics; use a separate task. |
| `chemical_vae.model_manifest` | Required nonempty path | A verified v5 conversion manifest and its adjacent hash-matched `encoder_state_dict.pt`. In the ordinary feature service, the path follows YAML path resolution. |
| `chemical_vae.backend` | `pytorch`; only `pytorch` accepted | Selects the supported frozen-encoder backend; this is not an alternative-backend switch. |
| `chemical_vae.device` | `cpu`; a valid PyTorch device string | Execution device, such as `cpu` or `cuda`. CUDA must actually be available. |
| `chemical_vae.batch_size` | `64`; positive integer | Encoder inference batch size. |
| `chemical_vae.input_preprocessing` | `identity`; only `identity` accepted | Preserves the input text; no salt splitting, canonicalization or truncation. |
| `chemical_vae.output` | `z_mean_sample`; only `z_mean_sample` accepted | Uses raw deterministic encoder means, without sampling or normalization. Current verified assets require maximum input length 120 and output width 196; these are asset-contract limits, not YAML knobs. |
| `ohe.missing_policy` | `as_category`; `as_category`, `zero_block` or `error` | Encode a missing-value token as a category, force the corresponding block to zero, or reject missing values. A missing category absent from training is still unknown at evaluation. |
| `ohe.dtype` | `float32`; `float32` or `float64` | Output matrix type. |
| `ohe.handle_unknown` | `ignore`; only `ignore` accepted | Evaluation categories absent from the training fold produce an all-zero block. |

- The following settings exist in Python implementations but are **not exposed as ordinary YAML `params`**: Morgan/MAF `radius` and `n_bits`; RDKit2D `descriptor_names`; FISD `model_dir`, `device`, `batch_size`; MolMetaLM `weight_path`, `device`, `batch_size`, `max_length`; OHE `missing_token`.
- DRFP's Python `n_bits` is itself restricted to 2,048.
- Adding these keys to the ordinary YAML does not activate them.
- Extending support requires validation, constructor routing and identity handling in code.

#### Descriptor support differs between the three workflows

| Descriptor/settings | Ordinary random evaluation | Fixed-membership benchmark | Frozen external testing |
| --- | --- | --- | --- |
| `morgan`, `maccs`, `fisd`, `molmetalm`, `rdkit2d` | `concat`, ordinary registry rules | Explicit columns and lifecycle; `concat` is the usual column-wise route | Explicit mapped columns, `static_descriptor`, and no invalid row may be dropped |
| `maf` | `sum` | Declare `mode: sum` to retain additive semantics | Declare `mode: sum`; all rows must remain valid |
| `mfp` | Three parameters above | Static parameters reach the feature builder; explicit columns required | Static parameters reach the feature builder; no row may be dropped |
| `chemical_vae` | Six parameters above; manifest identity checked by the feature service | Uses the feature-builder/asset route; preserve a valid manifest location and verified adjacent state | Same frozen encoder contract; no row may be dropped |
| `drfp` | Role-aware `reaction` route and optional extra reactants | Not directly supported by the current schema-2 adapter's reaction-role/mode routing | Not directly supported by the current schema-2 adapter's reaction-role/mode routing |
| `ohe` | Fold-local settings above | Current `ReactionComponentOHE` fixes `missing_policy=zero_block`, `dtype=float64`, `handle_unknown=ignore`; ordinary OHE parameter settings are not forwarded | Rejected: the adapter accepts only `static_descriptor` |

- Benchmark adapters have their own validation and execution paths.
- Their mode checks accept `concat/sum`, and they do not forward DRFP reaction roles as the ordinary feature service does.
- A descriptor appearing in a menu therefore does not guarantee compatibility with every protocol.
- Use the runnable examples as protocol-specific starting points.

### Add a new descriptor

To add a descriptor, edit its source registration points; the project does not discover plugins automatically. Start with the ordinary workflow, then add dedicated-protocol support where needed.

1. Implement a class under `yonod/descriptors/` following [BaseDescriptor](yonod/descriptors/base.py).

    - Define `name`, `output_dim`, and `featurize(smiles_list) -> (features, mask)`.
    - The class returns a matrix shaped `(n_input, output_dim)` and a Boolean mask shaped `(n_input,)`, retaining input order.
    - Failed inputs receive zero rows and `False`; a zero vector alone must not be treated as success.

2. Add provider metadata to `_PROVIDERS` in [registry.py](yonod/descriptors/registry.py): name, lifecycle, input type, supported modes and parameters.

    - Add normalization, defaults and rejection of unknown keys in `_normalise_params`.
    - For a new parameterized provider, do not rely on the constructor accepting values that the YAML never routes.

3. Add a lazy import to `_DESCRIPTOR_IMPORT_MAP` and route validated constructor parameters through `_get_descriptor` in [feature_builder.py](yonod/universal/feature_builder.py).

    - The current generic branch calls the class without arguments; MFP and Chemical VAE illustrate explicit parameter routing.
    - Add a lazy public export in [descriptors/__init__.py](yonod/descriptors/__init__.py) if that API should expose the class.

4. Check combination and sample alignment in [pipeline/features.py](yonod/pipeline/features.py).

    - At this service boundary, the feature computer returns a matrix with `sum(valid_mask)` rows, plus the original-length Boolean mask; the builder performs the filtering.
    - Publish ordered sample IDs, feature names, validity and hashes through the existing artifact APIs.
    - `FeatureSpec.schema_hash` and the feature identity include normalized configuration and resolved columns/roles; include external asset hashes where representation content depends on weights.
    - Changing an asset in place must not silently reuse old vectors.

5. Update the explicit descriptor menus/default builders in [generate_yaml_config.py](generate_yaml_config.py) and check the launcher choices in [yonod.py](yonod.py).

    - Keep heavy libraries lazily imported.
    - Add dependencies to the documented installation route only if this provider needs them.

6. For fixed membership or external testing, inspect [benchmark/config.py](yonod/benchmark/config.py), [benchmark/runner.py](yonod/benchmark/runner.py) and [frozen_external.py](yonod/benchmark/frozen_external.py).

    - Connect the required roles, mode, parameters and asset paths explicitly.
    - Ordinary routing is not reused automatically.
    - Preserve frozen external testing's population and overlap checks, and its requirement to retain every row.

7. Put validation scripts under `_verify/`.

    - Check malformed declarations, unknown parameters, invalid inputs, output dimensions, row/ID alignment, deterministic identity, and changes in parameters or asset content.
    - Use independent configurations and result directories for any later modeling acceptance, through `yonod.py`; follow Linux and Windows path/UTF-8 conventions.

- A learned representation belongs to `fold_transform` if it learns categories, scaling or any other state from the task data.
- OHE's current lifecycle saves raw inputs during `features`, then fits on each outer training fold in [pipeline/training.py](yonod/pipeline/training.py); evaluation records are transformed only.
- A new fold transformer also needs dispatch and persisted-state support in the feature publisher, training/prediction code, [ordinary HPO](yonod/hpo/ordinary.py), strict benchmark preprocessing and any supported inner-CV search.
- Those paths currently contain OHE-specific logic.
- Setting `lifecycle: fold_transform` in the registry alone does not implement them.
- Never fit a learned transform on the full dataset before splitting.

- When an implementation change alters the representation, explicitly version its persisted identity, such as `FEATURE_SERVICE_SCHEMA_VERSION`, or use a new candidate ID and configuration.
- The feature identity does not automatically hash Python source files; unchanged parameter names do not make an old cache valid for a changed algorithm.

### Model parameters and their execution boundary

- Schema-2 YAML model parameters are defined in [model_factory.py](yonod/model_factory.py) and [the section routes](yonod/config/contracts.py).
- Files under `yonod/models/` include older adapters; their defaults, such as 300 RF trees or automatic SVM scaling/PCA, do not become ordinary schema-2 defaults.

- The catalog below was checked against scikit-learn 1.4.0, XGBoost 2.0.3, LightGBM 4.3.0 and AutoGluon 1.1.1.
- An omitted estimator parameter uses the installed library's default.
- The project validates parameter names before fitting; the library also validates values and combinations.
- Safe YAML can describe numbers, strings, lists and mappings, but cannot construct Python functions, callback objects or estimator classes.

| Model | Parameter sections | Project behavior |
| --- | --- | --- |
| `rf` | `estimator`, `fit`, `runtime` | RF constructor and native fit parameters; `runtime` must be empty. |
| `xgb` | `estimator`, `fit`, `runtime` | XGBoost constructor and selected native fit parameters; optional device policy and inner early stopping. |
| `lightgbm` | `estimator`, `fit`, `runtime` | Installed constructor names plus the project's explicit booster-parameter allowlist; optional inner early stopping. |
| `svm` | `estimator`, `fit`, `preprocessing`, `runtime` | SVR, optional training-fold scaling/PCA/subsampling; `runtime` must be empty. |
| `autogluon` | `predictor`, `fit`, `runtime` | Explicit project allowlists; backend-version restrictions still apply. |

Use a full path such as `model_params.rf.estimator.n_estimators` when editing a setting. Only declare parameters for models selected in `models`.

#### Random Forest (`rf`)

These are the 18 names accepted under `model_params.rf.estimator` in the supported scikit-learn installation.

| Parameter | Meaning and value |
| --- | --- |
| `n_estimators` | Positive tree count; library default 100. |
| `criterion` | Split criterion: `squared_error`, `absolute_error`, `friedman_mse` or `poisson`; default `squared_error`. Target restrictions apply to Poisson. |
| `max_depth` | Positive depth limit or `null` for no explicit limit; default `null`. |
| `min_samples_split` | Minimum samples to split: integer at least 2 or fraction in `(0, 1]`; default 2. |
| `min_samples_leaf` | Minimum leaf samples: positive integer or fraction in `(0, 1)`; default 1. |
| `min_weight_fraction_leaf` | Minimum weighted leaf fraction, from 0 to 0.5; default 0. |
| `max_features` | Features considered at a split: positive count, fraction, `sqrt`, `log2` or `null`; default 1.0. |
| `max_leaf_nodes` | Leaf-count limit, integer at least 2 or `null`; default `null`. |
| `min_impurity_decrease` | Nonnegative minimum split improvement; default 0. |
| `bootstrap` | Sample training rows with replacement for each tree; boolean, default `true`. |
| `max_samples` | Bootstrap row count/fraction or `null`; meaningful with `bootstrap: true`, default `null`. |
| `oob_score` | Calculate the forest's bootstrap out-of-bag score; default `false`. This does not replace outer-fold evaluation. |
| `n_jobs` | CPU workers, with library semantics for `null`, positive values and `-1`; omitted by the first-use configurations. |
| `random_state` | Integer seed or `null`; library default `null`. This is separate from `evaluation.seed`. |
| `verbose` | Nonnegative verbosity level; default 0. |
| `warm_start` | Reuse trees when fitting the same estimator again; default `false`. YONOD constructs a fresh estimator for every fold. |
| `ccp_alpha` | Nonnegative tree-pruning penalty; default 0. |
| `monotonic_cst` | Per-feature constraints `-1`, `0`, `1`, or `null`; default `null`. Length must match the final feature matrix. |

- `rf.fit` accepts only `sample_weight`.
- Declare it as `sample_weight: {column: weight_column}`; YONOD reads that column on the current training rows.
- Do not embed a list of weights.
- `rf.runtime` has no nonempty settings.
- The [scikit-learn RF reference](https://scikit-learn.org/1.4/modules/generated/sklearn.ensemble.RandomForestRegressor.html) explains estimator-specific restrictions.

#### XGBoost (`xgb`)

- The following 39 names are accepted under `model_params.xgb.estimator`.
- Defaults omitted by YAML are resolved by XGBoost 2.0.3; many Python constructor defaults are `None`, which lets the booster choose its effective setting.

| Parameters | Meaning and constraints |
| --- | --- |
| `n_estimators`, `num_parallel_tree` | Boosting-round count and trees built in each round. |
| `max_depth`, `max_leaves`, `max_bin`, `grow_policy` | Tree depth, leaf cap, histogram resolution and growth policy (`depthwise` or `lossguide`). |
| `learning_rate`, `gamma`, `min_child_weight`, `max_delta_step` | Learning rate, split-loss threshold, child Hessian threshold and leaf-update cap. |
| `subsample`, `sampling_method` | Training-row fraction and sampling policy (`uniform` or GPU-supported `gradient_based`). |
| `colsample_bytree`, `colsample_bylevel`, `colsample_bynode` | Feature fractions per tree, depth level and node. |
| `reg_alpha`, `reg_lambda` | Nonnegative L1 and L2 regularization. |
| `objective`, `base_score` | Loss name and initial prediction bias; the regressor's default objective is `reg:squarederror`. Keep a scalar regression objective for these workflows. |
| `booster`, `tree_method` | Booster (`gbtree`, `gblinear`, `dart`) and tree algorithm (`auto`, `exact`, `approx`, `hist`); supported combinations depend on the backend. |
| `device`, `n_jobs` | CPU/GPU selection and CPU workers; explicit device strings may include `cpu`, `cuda` or a supported CUDA ordinal. |
| `random_state`, `verbosity`, `validate_parameters` | Seed, log level and native parameter checking. |
| `missing` | Input value treated as missing; constructor default is NaN. This does not bypass YONOD feature-validity checks. |
| `monotone_constraints`, `interaction_constraints` | Allowed feature trends and interaction groups, indexed against the final feature matrix. |
| `importance_type` | Importance calculation method; affects importance reporting, not the split design. |
| `enable_categorical`, `feature_types`, `max_cat_to_onehot`, `max_cat_threshold` | Native categorical handling and split thresholds. Naming a source categorical column here does not replace the project's OHE pipeline. |
| `multi_strategy`, `scale_pos_weight` | Multi-output strategy and positive-class weighting; recognized native names, not additional scalar-regression tasks provided by this guide. |
| `eval_metric`, `early_stopping_rounds`, `callbacks` | Evaluation metric and stopping/callback controls. Custom Python callbacks are not portable YAML values; use the audited inner-stopping declaration below. |

- `xgb.fit` admits `sample_weight`, `eval_metric`, `early_stopping_rounds` and `verbose`.
- Weights use the column mapping above.
- Constructor versions of the metric/stopping settings are preferred by XGBoost 2.x; do not declare the same option in both places.
- `eval_set`, `sample_weight_eval_set`, `base_margin`, `base_margin_eval_set`, `feature_weights`, `xgb_model` and `callbacks` are rejected in `fit` to prevent user-supplied external arrays or objects crossing the fold boundary.

- `xgb.runtime.device_policy` accepts `auto`, `cpu` or `cuda`.
- If using `estimator.device`, omit the entire `runtime` section: the current validator rejects `estimator.device` together with any declared `runtime` section.
- `runtime.early_stopping` has three keys: positive integer `rounds`, `validation_fraction` strictly between 0 and 1, and integer `seed` (default 42).
- YONOD creates that validation split solely from the current outer training fold.
- Set an early-stopping budget through this route rather than a bare stopping parameter without evaluation data.
- See the [XGBoost 2.0 parameter reference](https://xgboost.readthedocs.io/en/release_2.0.0/parameter.html).

#### LightGBM (`lightgbm`)

- `model_params.lightgbm.estimator` accepts the 71 names below.
- Some are aliases for the same native setting.
- Choose one spelling per setting; duplicate alias declarations can have backend-dependent precedence.
- The project's allowlist is narrower than the entire upstream parameter catalog.

| Parameters | Meaning and constraints |
| --- | --- |
| `boosting_type`, `objective`, `n_estimators`, `learning_rate` | Boosting family, regression loss, round budget and learning rate. Constructor defaults are `gbdt`, inferred regression, 100 and 0.1. |
| `num_leaves`, `max_depth` | Leaf and depth limits; defaults 31 and -1 (no positive depth cap). |
| `min_child_samples`, `min_child_weight`, `min_sum_hessian_in_leaf` | Leaf row/Hessian minima; the latter two name the same native threshold. |
| `min_split_gain`, `min_gain_to_split` | Minimum split improvement; aliases. |
| `subsample`, `bagging_fraction`, `subsample_freq`, `bagging_freq`, `bagging_seed` | Row sampling fraction, frequency and seed; the first two and the middle two are alias pairs. Sampling needs an enabled frequency. |
| `colsample_bytree`, `feature_fraction`, `feature_fraction_bynode`, `feature_fraction_seed` | Feature sampling per tree/per node and its seed; the first two are aliases. |
| `reg_alpha`, `lambda_l1`, `reg_lambda`, `lambda_l2` | L1/L2 penalties and aliases. |
| `subsample_for_bin`, `bin_construct_sample_cnt`, `max_bin`, `min_data_in_bin` | Sampling budget for bins, its alias, bin cap and minimum data per bin. |
| `max_delta_step`, `path_smooth`, `linear_tree`, `extra_trees` | Leaf-update cap, leaf smoothing, linear leaves and randomized thresholds; feature/backend restrictions apply. |
| `cat_l2`, `cat_smooth`, `max_cat_to_onehot`, `max_cat_threshold`, `min_data_per_group` | Native categorical regularization, smoothing and split controls. These refer to encoded model features. |
| `monotone_constraints`, `monotone_constraints_method`, `monotone_penalty` | Per-feature trend constraints, constraint algorithm and depth penalty. |
| `feature_contri`, `interaction_constraints` | Per-feature gain multipliers and allowed feature interactions. |
| `drop_rate`, `max_drop`, `skip_drop`, `xgboost_dart_mode`, `uniform_drop`, `dart_seed` | DART dropout controls; meaningful for the corresponding boosting mode. |
| `data_sample_strategy`, `top_rate`, `other_rate` | `bagging`/`goss` data sampling and GOSS retention fractions. |
| `deterministic`, `force_col_wise`, `force_row_wise` | CPU determinism and histogram layout; do not force both layouts. |
| `device_type`, `n_jobs`, `num_threads` | Device and CPU resources; the last two are related worker/thread controls. GPU execution needs a compatible LightGBM build. |
| `random_state`, `seed`, `verbosity`, `importance_type` | Seeds, logging and feature-importance method (`split`/`gain`). |
| `boost_from_average`, `first_metric_only`, `metric`, `early_stopping_round`, `early_stopping_rounds` | Initial-score behavior, metric selection and stopping aliases. Use the audited inner-stopping route for ordinary evaluation. |
| `class_weight`, `is_unbalance`, `scale_pos_weight`, `eval_at` | Recognized classification/ranking controls; they do not convert YONOD's scalar regression workflow to classification/ranking. |
| `callbacks` | A name included in the project allowlist, but a Python callback object is not a safe YAML value and this is not the supported stopping route. |

- `lightgbm.fit` admits `sample_weight`, `eval_names`, `eval_metric`, `feature_name` and `categorical_feature`.
- Weights use the column mapping.
- Names and categorical indices must match the model matrix, not an unrelated source-table layout.
- `init_score`, `eval_set`, `eval_sample_weight`, `eval_init_score`, `init_model` and `callbacks` are rejected in `fit`.
- `lightgbm.runtime` accepts only `early_stopping`, with the same `rounds`, `validation_fraction`, `seed` structure as XGBoost.
- The trainer supplies the inner validation rows and LightGBM stopping callback internally.
- Native details are in the [LightGBM 4.3 regressor API](https://lightgbm.readthedocs.io/en/v4.3.0/pythonapi/lightgbm.LGBMRegressor.html) and [parameter reference](https://lightgbm.readthedocs.io/en/v4.3.0/Parameters.html).

#### Support vector regression (`svm`)

`model_params.svm.estimator` accepts all 11 parameters below. It constructs `sklearn.svm.SVR`, not a classifier.

| Parameter | Meaning and library default |
| --- | --- |
| `kernel` | `linear`, `poly`, `rbf`, `sigmoid` or `precomputed`; default `rbf`. Precomputed kernels require a different matrix contract and are not produced by the ordinary descriptor route. |
| `degree` | Nonnegative polynomial degree, default 3; relevant to `poly`. |
| `gamma` | Kernel coefficient: `scale`, `auto` or a nonnegative number; default `scale`. |
| `coef0` | Independent term for polynomial/sigmoid kernels; default 0. |
| `C` | Positive error-penalty parameter; larger values generally mean weaker regularization; default 1. |
| `epsilon` | Nonnegative error-tube width; default 0.1, in the target's units. |
| `tol` | Positive solver stopping tolerance; default 0.001. |
| `shrinking` | Shrinking heuristic; default `true`. |
| `cache_size` | Positive kernel-cache size in MB; default 200. |
| `verbose` | Solver output; default `false`. |
| `max_iter` | Iteration cap, or -1 for no cap; default -1. |

- `svm.fit` accepts only `sample_weight` using a source-column mapping.
- The factory routes it internally to the final pipeline step.
- `svm.runtime` must be empty.
- Unlike the older SVM adapter, the ordinary schema-2 factory adds no scaler, PCA or row subsampling unless declared.

| `svm.preprocessing` setting | All supported fields and behavior |
| --- | --- |
| `scaler` | Requires boolean `enabled`; `params` accepts `copy` (copy input, default `true`), `with_mean` (center columns, default `true`), `with_std` (scale to unit variance, default `true`) from StandardScaler. Scaling fits on the outer training rows. |
| `pca` | Requires boolean `enabled`; optional nonnegative integer `feature_threshold` applies PCA only when the input feature count is greater than the threshold. `null` means no threshold. |
| `pca.params` | The nine PCA constructor parameters are listed individually below. Integer component counts are capped by training rows and features. |
| `subsample` | `n_samples`: positive integer or `null`; `random_state`: nonnegative integer, default 42 when sampling. Only the SVR fit is reduced; scaler/PCA still fit on all outer training rows. The sampler adds the fold offset to its seed. |

| Parameter under `pca.params` | Meaning and library default |
| --- | --- |
| `n_components` | Output dimension, explained-variance fraction, `mle` or `null`; default `null`, subject to solver restrictions. |
| `copy` | Copy the training input before decomposition; default `true`. |
| `whiten` | Scale the principal components to unit variance; default `false`. |
| `svd_solver` | Decomposition method: `auto`, `full`, `arpack`, `randomized`; default `auto`. |
| `tol` | ARPACK convergence tolerance; nonnegative, default 0. |
| `iterated_power` | Power iterations for randomized SVD, integer or `auto`; default `auto`. |
| `n_oversamples` | Extra random vectors for randomized SVD; positive integer, default 10. |
| `power_iteration_normalizer` | Normalization during power iteration: `auto`, `QR`, `LU`, `none`; default `auto`. |
| `random_state` | Seed for randomized/ARPACK solvers, integer or `null`; default `null`. |

- Preprocessing constructor parameters retain scikit-learn defaults when omitted; scaler defaults to centering and standardizing, PCA defaults to no whitening.
- Native details are in the [SVR](https://scikit-learn.org/1.4/modules/generated/sklearn.svm.SVR.html), [StandardScaler](https://scikit-learn.org/1.4/modules/generated/sklearn.preprocessing.StandardScaler.html) and [PCA](https://scikit-learn.org/1.4/modules/generated/sklearn.decomposition.PCA.html) references.

#### AutoGluon (`autogluon`)

- These names pass the project's ordinary schema-2 name checks.
- Some need further code integration or fail in the pinned backend, as marked below.
- The factory binds `predictor.label` to `dataset.column_roles.label`; declare `problem_type: regression` to state the scalar-regression task explicitly.
- It builds DataFrames containing encoded `f0`, `f1`, ... features and the target.
- Arbitrary original CSV columns are not automatically copied into that frame.

| `autogluon.predictor` names | Meaning and current restrictions |
| --- | --- |
| `label`, `problem_type`, `eval_metric` | Target name (must match the dataset), problem identity and model-selection metric. Classification/quantile settings are not additional validated task types in this scalar-regression guide. |
| `path`, `verbosity`, `log_to_file` | Predictor artifact path, logging level and file logging. An explicit `path` overrides the runtime path. |
| `sample_weight`, `weight_evaluation`, `groups` | Backend weight-column name, weighted evaluation and bagging-group column. Original source weight/group columns have no automatic propagation, so these are not a general CSV-column adapter. |
| `quantile_levels` | Quantile list for a quantile-regression task; ordinary scalar scoring is not a validated quantile-output contract. |
| `learner_kwargs`, `learner_type` | Native learner customization; a Python learner class cannot be constructed from safe YAML. |
| `positive_class` | Included in the project's name allowlist, but rejected by the installed AutoGluon 1.1.1 constructor; do not use it. |

| `autogluon.fit` names | Meaning and current restrictions |
| --- | --- |
| `time_limit`, `presets`, `hyperparameters` | Fit-time budget, preset name/list and model-family settings; these do not change the outer membership. |
| `feature_metadata`, `num_cpus`, `num_gpus` | Feature typing and backend resource allocation; choose settings for your machine. |
| `fit_weighted_ensemble`, `fit_full_last_level_weighted_ensemble` | Whether and how the backend fits its weighted ensembles. |
| `holdout_frac`, `num_bag_folds`, `num_bag_sets`, `num_stack_levels` | Internal holdout fraction, bagging folds/repetitions and stacking depth. These consume only the provided outer training population. |
| `auto_stack`, `dynamic_stacking`, `use_bag_holdout` | Backend choices for automatic stacking, stacked-overfitting checks and a bagging holdout. |
| `included_model_types`, `excluded_model_types` | Allowed/excluded backend model families. |
| `refit_full`, `set_best_to_refit_full`, `keep_only_best`, `save_space` | Backend refitting and artifact-retention decisions; preserve the artifacts required for later work. |
| `ag_args`, `ag_args_fit`, `ag_args_ensemble` | Nested settings for backend model identity, individual fitting and ensemble behavior. Valid nested keys belong to AutoGluon 1.1.1, not YONOD's own schema. |
| `verbosity`, `calibrate` | Fit logging and calibration (`true`, `false` or `auto`, where applicable). |
| `fit_extra`, `fit_strategy`, `delay_bag_sets`, `raise_on_no_models_fitted` | Included in the project's name allowlist but not accepted by the installed AutoGluon 1.1.1 `fit` argument validator. Do not use them as YAML fit options. `fit_extra()` is a separate backend method. |

- `train_data` is bound by the pipeline and cannot be supplied in YAML; neither can `tuning_data` or `unlabeled_data`.
- Upstream options absent from the allowlist, such as `infer_limit` and `log_file_path`, are not exposed merely because AutoGluon documents them.
- `autogluon.runtime.save_path` accepts a string or `null`; its default artifact location is the current fold's `autogluon/` folder.
- `runtime.cleanup` is boolean, default `false` in the ordinary pipeline.
- Leave both `predictor.path` and `runtime.save_path` unspecified to retain task/fold isolation: ordinary execution does not append a fold suffix to an explicit path, so a shared path can mix or overwrite artifacts or make fitting fail.
- Any custom path must remain inside that task's designated output directory and isolate every fold and combination.
- There is no single top-level `random_state` option here that seeds every AutoGluon submodel.

- The version-pinned upstream [constructor documentation](https://github.com/autogluon/autogluon/blob/v1.1.1/tabular/src/autogluon/tabular/predictor/predictor.py#L222) and [fit documentation](https://github.com/autogluon/autogluon/blob/v1.1.1/tabular/src/autogluon/tabular/predictor/predictor.py#L392) also define the nested `hyperparameters`, `ag_args`, `ag_args_fit` and `ag_args_ensemble` dictionaries.
- YONOD exposes those dictionaries but does not define or validate every backend-family subparameter.
- Check the 1.1.1 source/API for that nested range before declaring it.

#### Parameter support differs by evaluation protocol

| Capability | Ordinary `all/train` | Manifest benchmark | Frozen external test |
| --- | --- | --- | --- |
| Constructor parameters | Shared factory catalog above | Shared catalog, with protocol-specific seed restrictions | Shared catalog for supported non-AutoGluon models |
| Nonempty `fit` for RF/XGB/LightGBM/SVM | Supported native routes above | Rejected: this executor does not convert declared data options into training-fold values | Rejected |
| XGB/LightGBM `runtime.early_stopping` | Training-only inner split | Rejected | Rejected |
| AutoGluon | Ordinary allowlists above | Only `fit.time_limit`, `fit.presets`, `fit.num_cpus`, supported runtime path/cleanup; omit `predictor.label` (internal label is `yield`) | Unsupported |
| SVM subsampling | Samples only final SVR training rows | Does not execute `preprocessing.subsample.n_samples`; declared scaler/PCA still execute through the factory Pipeline | Does not execute `preprocessing.subsample.n_samples`; declared scaler/PCA still execute through the factory Pipeline |

- The standard `manifest_outer_cv` benchmark does not replace `estimator.random_state` with `evaluation.seed`.
- The explicit reproduction setting `benchmark.reproduction_protocol.name: vjethbkm_rf_5x5` and the separate `paper_exact_5x5` evaluation route bind RF's model seed to the fold manifest and reject conflicts.
- Factory acceptance alone does not mean that every benchmark adapter executes a parameter.

- The manifest benchmark's AutoGluon route uses the dedicated adapter, whose omitted settings are a 300-second fit budget, `medium_quality`, one CPU and cleanup enabled.
- These are that adapter's compatibility defaults; ordinary schema-2 AutoGluon uses the shared predictor factory and its installed-library defaults instead.
- Select resources explicitly when using the dedicated route.

### Add a new model

1. Implement or wrap an estimator with `fit(X_train, y_train, **fit_options)`, `predict(X_valid)` and `get_params()`.

    - Return one finite numeric prediction per validation record; ensure the fitted estimator can be saved and restored with the pipeline's artifact format.
    - Keep data-dependent preprocessing inside each training fold.

2. Add the model ID and allowed parameter sections to `MODEL_PARAMETER_ROUTES` in [config/contracts.py](yonod/config/contracts.py).

    - If aliases are needed, add them to `_MODEL_ALIASES` in [config/loader.py](yonod/config/loader.py).

3. Extend `_library_catalog`, `resolve_model_config`, `_validate_runtime`, `construct_estimator` (or a deliberate predictor-specific route) and `software_versions` in [model_factory.py](yonod/model_factory.py).

    - Keep optional dependencies lazily imported.
    - Reject unknown parameters and external data/callback objects before expensive work.

4. Check `_default_fold_runner` in [pipeline/training.py](yonod/pipeline/training.py), serialization and [saved prediction](yonod/pipeline/prediction.py).

    - For non-sklearn models, implement fit-option routing and artifact loading explicitly; the generic runner may not be sufficient.
    - Add diagnostic support only after verifying the model's train/holdout prediction contract.

5. Add discovery in `_MODEL_NAMES` in [yonod.py](yonod.py), the model menu and `_canonical_wizard_models` in [generate_yaml_config.py](generate_yaml_config.py).

    - The legacy `_make_model` and files under `yonod/models/` matter only if you also support that older direct-CSV interface; editing them alone does not register a schema-2 model.

6. Add HPO support separately in [hpo/contracts.py](yonod/hpo/contracts.py), [hpo/spaces.py](yonod/hpo/spaces.py), [hpo/search.py](yonod/hpo/search.py) and finalization where needed.

    - Define bounded, versioned search spaces and inner-fold scoring; constructor support alone does not enable HPO.

7. Enable dedicated protocols deliberately: review [benchmark/config.py](yonod/benchmark/config.py), [benchmark/executor.py](yonod/benchmark/executor.py) and [frozen_external.py](yonod/benchmark/frozen_external.py).

    - Preserve each protocol's data boundary, allowed models and recovery artifacts.
    - Keep an unsupported route rejected until its adapter is implemented.

8. Put validation scripts in `_verify/`.

    - Check unknown-key rejection, shape/finite-value errors, training-only preprocessing, no outer-label leakage, saved model/prediction round trips and Linux/Windows paths.
    - Keep the engineering smoke separate from real research tasks.
    - Update dependencies, bilingual documentation and the independent documentation log, then commit actual changes separately in the two repositories.

Last reviewed against this checkout: 2026-11-14.

## Project structure

The `config/example/` and `result/example/` folders initially contain only Git placeholders, with no task configurations or run results. Other local task folders appear when you prepare or run a task.

```text
YONOD/
├── README.md                English user guide
├── README.zh-CN.md          Chinese user guide
├── AGENTS.md                Development and directory conventions
├── LICENSE                  Project license
├── yonod.py                 Launch and execute a saved YAML
├── generate_yaml_config.py  Prepare and validate YAML without modeling
├── example.yaml             Complete schema-2 format and engineering smoke
├── requirements.txt         Dependencies; install PyTorch separately first
├── example/                 First-use examples and their own run.yaml configurations
│   ├── README.md            Example operating instructions
│   ├── index.html           Offline browser guide
│   ├── random_split/        Random-split internal evaluation
│   ├── fixed_split/         Evaluation using saved membership lists
│   ├── external_test/       Separate training and external-test tables
│   └── archives/            Downloadable example packages
├── yonod/                   Python implementation; main components below
│   ├── model_factory.py     Schema-2 model validation and construction
│   ├── config/              YAML loading, validation and path resolution
│   ├── descriptors/         Molecular and categorical encodings
│   ├── features/            Data preparation and numerical-condition processing
│   ├── models/              Legacy model adapters
│   ├── pipeline/            Ordinary feature, training, prediction and reporting stages
│   ├── benchmark/           Dedicated fixed-membership and external-test workflows
│   ├── splits/              Grouping and membership manifests
│   ├── artifacts/           Feature-package storage and integrity checks
│   ├── hpo/                 Model-parameter search within training data
│   └── universal/           Shared CSV-reading and feature-building components
├── dataset/                 Data and source notes
├── WEIGHTS/                 Optional pretrained assets
├── config/                  Recommended location for run configurations
│   ├── example/             Saved example YAMLs; initially empty
│   └── <task_name>.yaml     Local research-task configuration
├── result/                  Recommended location for all run outputs
│   ├── example/             Example outputs; initially no run results
│   │   └── <task_name>/     Created when an example runs
│   └── <task_name>/         All outputs belonging to one research task
├── logs/                    Operational logs and PID files
├── _verify/                 Local validation scripts
├── backup/                  Backups and engineering support
├── reference-projects/      External reference material
└── project-docs/            Independent documentation repository
    ├── goal.md              Project goals
    ├── project-plan.md      Development plan
    ├── buildlog.md          Change and verification records
    ├── teach.md             Learning notes
    └── docs/                Other maintained project documentation
```

- For everyday use, start with the two root scripts, [example.yaml](example.yaml) and [example/](example/).
- The [dataset notes](dataset/README.md) and [weight notes](WEIGHTS/README.md) explain the inputs and optional assets.

- Under `config/` and `result/`, Git keeps only each `example/` subtree; all other files and folders are ignored. The root folders `logs/`, `_verify/`, `backup/`, `reference-projects/` and `project-docs/` are also ignored.
- `project-docs/` has its own Git history in [YONOD-docs](https://github.com/thinktraveller/YONOD-docs), which requires access permission.
- Cloning YONOD does not fetch that repository; the first-use examples work without it.

An ordinary modeling task saves its main outputs in this layout:

```text
result/<task_name>/
├── feature/                         Encodings and feature records
├── runs/
│   └── run_<feature_id>-<model>/     One encoding/model combination
│       ├── predictions.csv          Per-record held-out predictions
│       ├── fold_metrics.csv         Metrics for each fold
│       └── run_manifest.yaml        Settings and run record
├── pictures/                        Report figures
└── report/                          HTML and Markdown reports
```

- Fixed-membership benchmarks instead organize membership lists, predictions, fold records, metrics and recovery state under `docs/manifests/`, `docs/predictions/`, `docs/folds/`, `docs/metrics/` and `docs/state/` inside the task folder.
- Frozen external testing also uses `docs/predictions/`, `docs/metrics/` and `report/`; it does not save the ordinary workflow's complete feature and final-model packages.
- The files available depend on the selected protocol.

- Example tasks save new outputs in the project-root `result/example/<task_name>/`, with one folder per task.
- We recommend keeping your own configurations in `config/<task_name>.yaml` and results under `result/<task_name>/`; both are Git-ignored outside the example subfolders.
- The `completed_run/` folders in downloadable ZIPs contain reference snapshots; they are separate from the outputs you generate.

## Acknowledgments

We thank the authors and contributors of the projects that provide molecular representations and pretrained resources used in YONOD:

- [DRFP](https://github.com/reymond-group/drfp): Daniel Probst, Philippe Schwaller, Jean-Louis Reymond and contributors, for the Differential Reaction Fingerprint method and implementation.
- [RDKit](https://www.rdkit.org/): the developers and contributors, for the cheminformatics toolkit used for molecular parsing, fingerprints and descriptors, including YONOD's MACCS implementation.
- [MolMetaLM](https://github.com/CSUBioGroup/MolMetaLM): the authors, for the molecular language model and the [MolMetaLM-base pretrained checkpoint and tokenizer](https://huggingface.co/wudejian789/MolMetaLM-base).

## License

- Project code and documentation use [CC BY-NC 4.0](LICENSE), requiring attribution and limiting use to noncommercial purposes.
- Third-party data, pretrained weights and reference code retain their own licenses and citation requirements.
