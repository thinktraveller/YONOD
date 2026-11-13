<a id="en"></a>

# YONOD — Organic Synthesis Reaction Prediction Platform

[English](README.md) · [简体中文](README.zh-CN.md)

> **Y**ou **O**nly **N**eed **O**utstanding **D**escriptors
>
> Compare molecular descriptors and machine learning models for organic synthesis reaction prediction using SMILES, explicit column roles, and auditable feature versions.

## Current interface

Reproducible runs use schema-2 YAML. Start modeling tasks through `yonod.py`, which validates the saved configuration and passes it to the appropriate runtime:

```bash
python -u yonod.py
# At the first prompt, enter a saved YAML path, for example example.yaml.
```

The repository-root [example.yaml](example.yaml) is a runnable 12-record Morgan × Random Forest smoke configuration with two folds. A successful run reports `features=ready` and `morgan × rf: complete` or `reused`. Run IDs are derived from content; do not hard-code them. Historical Chemical VAE acceptance configurations are stored in [configs_20260916.tar.gz](backup/configs_20260916.tar.gz); they are not implicit parts of `example.yaml`.

The [three evaluation examples for chemistry researchers](example/index.html) are collected under [example/](example/README.md): random-split internal evaluation, predefined-membership two-fold internal evaluation, and an independent external-test workflow. All three `run.yaml` files have been executed from their own folders, with new outputs written to each package's `result/`. No configuration copy is needed. Downloadable packages include snapshots of the completed outputs. All example data are synthetic teaching records.

| Purpose | Current entry point |
| --- | --- |
| Run a saved, reproducible configuration | `python -u yonod.py`; enter the YAML path at the first prompt |
| Create and execute a configuration interactively | `python yonod.py` |
| Inspect or derive feature versions without modifying the source | `inspect_feature_artifact` / `derive_features` APIs in `yonod.artifacts.operations` |

`yonod.py` is an interactive wizard. It **does not parse** command-line options such as `--csv` or `--models`. It can accept a CSV, collect column roles, and write `*_run.yaml`, or validate and run an existing schema-2 YAML supplied at the first prompt. All modeling tasks must use a saved YAML configuration and start through `yonod.py`.

JSON is no longer an executable run configuration for `main.py` or the wizard. JSON configuration migration and JSON-dependent launchers are no longer maintained for compatibility. `main.py --json ...` explicitly rejects that input. Historical JSON metrics, fold audits, third-party material, and existing results retain their original formats; they are neither training entry points nor dependencies that need migration.

## Three teaching examples: complete runs from their own folders

[Example overview](example/index.html) · [Detailed instructions](example/README.md) · [Download all examples](example/archives/all_examples.zip)

All three examples completed on 2026-11-13 in the Linux `yonod` environment, starting through `yonod.py` and reading `run.yaml` from each example's own folder. Every process returned 0. Per-record predictions, split membership, scores, and HTML/Markdown reports passed acceptance checks. These teaching examples do not require copying their YAML into the root `config/` folder. Research tasks still require separate task configurations under the root `config/`.

| Method | Data and configuration | Completed computation | Report snapshot and acceptance record |
| --- | --- | --- | --- |
| Random-split internal evaluation | [12 records](example/random_split/dataset/reactions.csv), [run.yaml](example/random_split/run.yaml) | Randomly divide the same data into two parts and exchange their roles; two folds each for RF, XGBoost, and LightGBM | [Report](example/random_split/completed_run/report/report.html) · [Acceptance](example/random_split/verification.json) |
| Predefined-membership two-fold internal evaluation | [12 records](example/fixed_split/dataset/reactions.csv), [membership manifest](example/fixed_split/splits/fixed_split_manifest.csv), [run.yaml](example/fixed_split/run.yaml) | Follow predefined membership; six training and six held-out records per fold, then exchange roles; two RF folds | [Report](example/fixed_split/completed_run/report/benchmark_report.html) · [Acceptance](example/fixed_split/verification.json) |
| Independent external-test workflow | [12 training records](example/external_test/dataset/train.csv), [6 external-test records](example/external_test/dataset/external.csv), [run.yaml](example/external_test/run.yaml), [source audit](example/external_test/audit/external_audit.json) | Fix model settings in advance; fit RF once on the complete training table, then evaluate the external table, which contains three amine groups | [Report](example/external_test/completed_run/report/external_test_report.html) · [Acceptance](example/external_test/verification.json) |

From the selected example folder, launch `../../yonod.py` and enter `run.yaml` at the first prompt. Input paths are relative to that YAML's folder. New outputs go to the package-local `result/<task_name>/`, with numerical structure encodings under its `feature/` subfolder. On Linux, use a headless background process, exactly 19 available logical CPUs, and dedicated log/PID files in the repository-root `logs/`. Run tasks serially. Full commands are in the [instructions](example/README.md). When launching from an example folder, the entry point fixes the relative YAML path before changing the subprocess working directory, so the runtime reads the intended configuration.

Windows Conda Prompt example, starting at the repository root:

```text
conda activate yonod
cd example\random_split
python -u ..\..\yonod.py
```

Enter `run.yaml` at the first prompt. For either other example, enter its corresponding folder instead. These commands retain Windows usage, but this round of actual execution acceptance covered Linux only.

The download's `completed_run/` contains the complete output snapshot from this acceptance run. The active `result/` directory is excluded from the ZIP, leaving a fresh output destination after extraction. Neither runtime directory is tracked by source Git. A source checkout can generate new results by running the examples; extracting the download also provides the completed reports. The random package's `report/` separately retains the historical report from 2026-09-29. YAML files under `evidence/` document historical provenance and are not among the three active examples.

All data and yields are synthetic teaching records, not experimental measurements. Predefined membership does not automatically establish unseen-substrate evaluation. The current interface requires at least two folds and exactly one held-out appearance per record per repeat. A separately saved external CSV does not establish independent provenance either; this example demonstrates the external-test workflow. R², the coefficient of determination, is not prediction accuracy. For yields on a 0–1 scale, multiply MAE, the mean absolute error, by 100 to obtain percentage points. The random report scores pooled held-out predictions within each repeat, whereas the predefined-membership report summarizes fold-level metrics; do not directly rank results across these different aggregation rules.

## Quick start

### 1. Environment

The project baseline is Python 3.9. Before Linux development or testing, activate `yonod` in the same shell invocation:

```bash
source /home/wangzh685/miniconda3/etc/profile.d/conda.sh
conda activate yonod
```

For a new environment, install in the following order. Match the PyTorch CUDA wheel to the local CUDA/driver setup. CPU users should replace the wheel index with the official PyTorch CPU index.

```bash
conda create -n yonod python=3.9 -y
conda activate yonod

# Example: CUDA 12.1
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 \
  --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install "autogluon.tabular[lightgbm,catboost]==1.1.1"
```

`requirements.txt` lists the base dependencies. AutoGluon, LightGBM, FISD, and MolMetaLM may need additional assets or substantial disk space. Running only `example.yaml` does not require FISD or MolMetaLM weights.

### 2. Run the smoke configuration

Linux modeling tasks use saved YAML, `yonod.py`, and task-specific log/PID files. Do not launch a modeling task directly with `main.py --config`:

```bash
mkdir -p logs
nohup setsid bash -lc '
  source /home/wangzh685/miniconda3/etc/profile.d/conda.sh
  conda activate yonod
  printf "%s\n" "example.yaml" | python -u yonod.py
' </dev/null > logs/example_morgan_smoke.log 2>&1 &
echo $! > logs/example_morgan_smoke.pid

task_pid="$(cat logs/example_morgan_smoke.pid)"
ps -p "$task_pid" -o pid,ppid,stat,etime,cmd
tail -n 100 logs/example_morgan_smoke.log
```

This configuration uses `dataset/benchmark_smoke_fixture.csv` from the repository and writes features, training outputs, and reports under `result/smoke-test/`:

```text
result/smoke-test/
├── feature/
│   ├── feature_runs/<feature_run_id>.yaml
│   └── features/<artifact_id>/
├── pictures/
│   └── scatter_morgan_rf_repeat-1.png
├── report/
│   ├── report.html
│   └── report.md
└── runs/
    └── run_morgan-rf/
        ├── run_manifest.yaml
        ├── source_config.yaml
        ├── effective_config.yaml
        ├── predictions.csv
        └── fold_metrics.csv
```

Paths resolve according to the configuration's location. For YAML under the root `config/`, references beginning with `./` resolve from the repository root. Explicit `../` references and configurations elsewhere, including `example/`, resolve from the YAML's own folder. Absolute paths remain absolute. `outputs.root` is the task root; the ordinary-run loader normalizes the effective `artifacts.output_dir` to its `feature/` subfolder. Startup logs and PID files belong in the repository's `logs/`. Do not reuse this smoke output for research. Research tasks need new YAML under `config/` and their own `result/<task_name>/`; the teaching examples above use `run.yaml` and `result/<task_name>/` inside their respective folders.

### 3. Create a configuration with the wizard

```bash
python yonod.py
```

The wizard validates the CSV, declares target/SMILES/numerical column roles, selects features and models, and requires a stable `sample_id` column with nonempty, unique values. Save the resulting `*_run.yaml` under the project's `config/` folder, then launch it through:

```bash
python -u yonod.py
# At the first prompt, enter path/to/project_run.yaml.
```

On Windows, activate the same environment through Conda Prompt. Application code uses configurable paths and `pathlib`. YAML references follow the location rules above, including the root `config/` exception for `./` references.

## Schema-2 YAML

The repository-root [example.yaml](example.yaml) is the sole authoritative, copyable, and validatable YAML specification. Create a separate schema-2 file under `config/` for each new research task, replacing only task-specific data, descriptors, models, evaluation settings, and isolated output paths. The teaching examples at `example/<example>/run.yaml` are an explicit exception that supports running in place. Do not maintain a second complete YAML specification in Markdown. Selected RF/XGBoost/LightGBM models use `model_params.<model>.estimator.n_jobs: 19`; AutoGluon uses `model_params.autogluon.fit.num_cpus: 19`. Declare only selected models, and run tasks serially.

Step 2-14 HPO passed engineering acceptance in the Linux `yonod` environment for ordinary RF/XGBoost/LightGBM, strict numerical and OHE entry points, soft time limits, resume, and independent final models. Windows hardware validation was deferred at the user's direction and remains `not_verified`. Omitting `hpo` or setting `enabled: false` preserves the fixed-parameter path and does not import Optuna; its dependency version is pinned to 4.5.0. Declare the budget explicitly with `max_trials`. Fields, search spaces, and the disabled-by-default example are defined only in [example.yaml](example.yaml). Each outer-training-fold HPO study and its inner folds are persisted separately under the task's `result/<task_name>/hpo/`. Inner search scores are not outer out-of-fold evaluation scores, and a higher search score is not an acceptance criterion. The full Linux acceptance entry point is `_verify/accept_hpo_all.py`; its script and local evidence are excluded from the main repository under project conventions.

Every run configuration declares a schema version, project name, stage, dataset, and stable sample ID. The parser accepts a single UTF-8 `.yaml`/`.yml` document and rejects duplicate keys, aliases, unknown fields, and unsupported model/parameter sections. `dataset.column_roles.label` is required for `train` and `all`; feature-only runs may omit it. `sample_id_col` must exist in the dataset and contain unique, nonempty values. DataFrame row numbers cannot replace it.

### Stages

| `stage` | Required inputs | Behavior |
| --- | --- | --- |
| `features` | `dataset`, `descriptors`, `artifacts.output_dir` | Initialize feature outputs and compute or reuse feature packages. No models are constructed and labels are not required. Explicitly declaring `outputs.root` is also recommended. |
| `train` | Labeled `dataset`, `artifacts.input_manifest`, `models` | Read only the specified feature packages and train. Never fall back to descriptor computation. |
| `all` | Declarations required by both features and training | First bring every candidate feature to a ready/failed terminal state, then train all ready packages. Publish each ready feature × model combination independently. |
| `benchmark` | Labeled `dataset`, structure inputs, models, and protocol-specific `evaluation`/`benchmark` declarations | Use `manifest_outer_cv` for grouped/predefined-membership internal evaluation, or `frozen_train_external_test` for one external test after training. Each protocol has its own constraints. |

External feature manifests, labels, and auxiliary columns are aligned by stable `sample_id`. Static feature identities exclude labels, so changing only labels changes the training identity rather than the feature identity.

If an identified feature × model combination fails during actual training, YONOD atomically publishes a non-overwritable manifest under `outputs.root/failed_runs/` containing the reason, original YAML, and effective YAML, then continues with sibling combinations. Failed combinations do not publish incomplete predictions/metrics. Any failed combination makes `main.py --config` return 2, preventing schedulers from treating partial success as complete success. Actual fixture evidence is in `derived/interface_migration/step30_acceptance/training_failure_isolation_evidence.md`.

The six historical local acceptance records listed below as `derived/...` paths are not included in the current checkout. Their paths are retained for traceability rather than as download links; those historical acceptance runs were not repeated for this documentation update.

### Reports and offline rebuilding

For schema-2 `train` / `all`, omitting `outputs.report_formats` generates both HTML and Markdown. Supported values are `html` and `markdown` (or `md`); an empty list disables automatic reports. Each completed run generates `pictures/` and `report/` from its own `run_manifest.yaml`, predictions, fold metrics, and YAML snapshots. Scatter plots use each repeat's out-of-fold predictions. HTML embeds PNG images; Markdown uses `../pictures/` relative links. The `features` stage does not generate reports.

Rebuild ordinary result reports with `yonod.pipeline.reporting.rebuild_schema2_report` without the original dataset, descriptor weights, or models. It validates manifests, configuration snapshot hashes, out-of-fold identities, and finite values; it neither trains nor rewrites source results. Old dedicated scripts are archived locally under `backup/scripts/` and are no longer current entry points.

Completed HPO combinations also include a “Hyperparameter search and nested evaluation” section: study/trial histories, actual parameters and their sources for every fold, independent outer-fold metrics, and separate search, outer-training, and prediction times. Ordinary summary tables still use pooled out-of-fold predictions within each repeat; do not equate these with mean fold scores or inner search scores. Explicit final models show development-set search and refitting separately, with no test score when there is no independent test. The report rebuilder verifies hashes of persisted study exports without opening databases or refitting. HPO failure/soft-time-limit reports at `report/hpo_status.html` and `.md` display status and verified search evidence only; incomplete folds are not presented as complete out-of-fold evaluation.

### Training and outer-evaluation diagnostics — Step 2-15, accepted Linux scope for three tree models

Nine isolated small-data tasks covered ordinary and strict RF, XGBoost, and LightGBM, fixed settings/HPO, diagnostics on/off, and ordinary early stopping. Outer-fold predictions, diagnostic values, model reloads, and both report formats were checked independently. `logs/diag215_20260929_acceptance_resume5.log` retains the unified acceptance record. Historical SVM/AutoGluon results remain readable and explicitly display diagnostics as “not recorded.” AutoGluon was not refitted in that acceptance round, and Windows hardware remained unverified. These engineering checks do not establish model performance on real chemistry data.

`outputs.diagnostics.enabled` controls fold-level diagnostics for RF, XGBoost, and LightGBM (default `true`). `outputs.diagnostics.save_train_predictions` controls whether per-record training predictions are saved (default `false`; acceptance tasks should set it to `true`). The sole complete YAML structure is in `example.yaml`. These options control diagnostics only; they do not change the outer out-of-fold file format or scoring rules. SVM and AutoGluon do not yet have diagnostics on the same basis; their existing modeling capabilities remain available.

Diagnostics reuse the final outer-fold model and fitted feature state. They compute outer-training scores, outer-held-out scores, and a held-out baseline that always predicts the arithmetic mean of that fold's outer-training labels. The R² gap is training minus outer-held-out; RMSE/MAE gaps are outer-held-out minus training. With early stopping, diagnostics also record the actual fit records and the internal early-stopping holdout. Training scores describe fit to previously seen records, not generalization. Units are marked unknown if undeclared. Undefined R², such as for constant labels, retains its reason rather than being replaced with 0.

Ordinary per-fold compact diagnostics are at `runs/<combination>/diagnostics/repeat-NN-fold-NN.yaml`, with optional per-record predictions in the corresponding `-train.csv`. The `diagnostics.folds` entries of `run_manifest.yaml` reference them by relative path and SHA-256. Strict compact diagnostics are at `diagnostics/<descriptor>__<model>__rNN__fNN.json` under the task root, with optional per-record `-train.parquet`; fold metadata references them through `diagnostic`. Reports read this evidence only. Missing historical diagnostics display “not recorded,” while corruption or failures retain their reasons. Training performance is not inferred from out-of-fold or HPO inner-fold scores. Offline report rebuilding neither fits models nor supplies missing diagnostics.

The strict report's “Complete outer-held-out sample and group trace” section is rebuilt from saved split manifests and prediction shards. Full rows are in `report/diagnostic_sample_trace.csv` and `report/diagnostic_group_trace.csv`; the report shows previews and links to the CSVs. Per-record tables retain residual direction, group keys, and source row indices. Per-group tables retain training/held-out counts, target ranges, and errors. Missing, duplicate, or mismatched group identities display `unavailable`; single-record groups are marked unstable. Opaque group keys cannot establish reaction types.

`yonod.diagnostic_repair` is a separate **prediction-only postprocessing operation**. It does not fit models, run HPO, fit features, or launch new modeling tasks. It recomputes diagnostics from the original data, model, and transformation state of a completed fold and writes evidence to a new `result/<repair_task>/`. Original out-of-fold predictions, models, studies, manifests, and reports remain unchanged, and the old report does not automatically reference the repair. It supports only ordinary/strict × fixed/HPO folds for the three tree models when complete, verifiable model bundles exist. Missing bundles, mismatched data identities, or disagreement with saved outer-fold predictions prevent publication. All formal modeling still requires a separate schema-2 YAML under `config/` and launch through `yonod.py`.

From the repository root in Bash or PowerShell, with `yonod` activated, postprocess one saved fold as follows. Replace the task name and source paths as appropriate; logs belong in `logs/`:

```text
python -m yonod.diagnostic_repair --mode strict --source-root result/diag215_20260929_strict_fixed_v1 --dataset-path dataset/numeric_strict_smoke_fixture.csv --destination result/my_repair_task --feature morgan --model rf --repeat 1 --fold 1 > logs/my_repair_task.log 2>&1
```

On success, the log provides the `repair_manifest.json` path. An existing destination directory is not overwritten. The repair outputs' `fit_calls=0` and `search_calls=0` are operational declarations; acceptance verifies them by intercepting fit/search calls, independently reloading outputs, and comparing source file hashes.

The feature service publishes static matrices, sample IDs, validity masks, identities, and hashes in separate packages. The training service publishes each feature × model result under `outputs.root/runs/run_<feature-id>-<model>/`. The content-derived `run_id` remains in the manifest for reuse validation. Outputs include:

- `predictions.csv` and `fold_metrics.csv`;
- `run_manifest.yaml`;
- the original `source_config.yaml` and normalized `effective_config.yaml`;
- fold-local transformation states and audits for features transformed within each fold.

A complete package with the same identity is validated before reuse. Incomplete existing results are never silently treated as complete. Failure records are stored under the combination directory's `failed_attempts/`. If the same combination directory already contains a different result identity, the run refuses to overwrite it; use a new task root.

### Features

The public feature registry currently includes:

- Static molecular/reaction descriptors: `morgan`, `mfp`, `maccs`, `fisd`, `molmetalm`, `maf`, `rdkit2d`, and `drfp`.
- Validated static molecular descriptor: `chemical_vae`, using the independently accepted ZINC/v5 and `zinc_properties` encoder/v5 assets and raw `z_mean_sample`; property heads are outside its scope.
- Fold-transformed features: `ohe`. It stores raw categorical inputs; OneHotEncoder is fitted only on each outer-training fold, and the validation fold is transformed only.

`mfp` is a Morgan **count** fingerprint, not an alias for binary `morgan`. It accepts explicit `radius`, `fp_size`, and `profile`. `drfp` requires compatible reaction inputs; `fisd` and `molmetalm` require the weights described below.

Numerical scaling, OHE, SVM scaling/PCA/subsampling, and internal early-stopping validation for XGBoost/LightGBM are restricted to the outer-training fold. The outer-validation fold is used only for final prediction.

### Chemical VAE — frozen encoder with independent acceptance

Chemical VAE is an integrated but deliberately restricted optional descriptor, not the default descriptor in `example.yaml`. It accepts only separately validated ZINC/v5 or `zinc_properties` encoder/v5 conversion manifests and obtains encoder outputs from adjacent, hashed PyTorch state. Declare it as a separate `chemical_vae` descriptor, specifying the model manifest, `pytorch` backend, device, batch size, `identity` input preprocessing, and `z_mean_sample` output. `model_manifest` resolves relative to the YAML's folder. `zinc_properties` exposes only the encoder's raw `z_mean_sample`, without loading its property head; the two assets are not interchangeable. It does not load decoders, property heads, training CSVs, original HDF5, or TensorFlow/Keras; ordinary runtime requires only PyTorch.

The current assets produce 196 raw `z_mean_sample` dimensions per molecular column, or 392 for two columns in `concat` mode. Unknown parameters or unverified assets fail explicitly rather than silently falling back. Historical complete Chemical VAE YAML is in the [configuration archive](backup/configs_20260916.tar.gz), not duplicated in this README.

Inputs use byte-for-byte identity semantics: no trimming, canonicalization, salt splitting, delimiter replacement, truncation, or vocabulary extension. The adapter first checks the fixed 35-character vocabulary, maximum length 120, and RDKit parseability. Missing, overlong, unsupported-character, or RDKit-invalid inputs are recorded in a versioned diagnostics sidecar. `concat` preserves existing semantics: retain a reaction row if at least one molecular column succeeds, replacing failed columns with zero blocks. Only failure of all selected columns makes the row's mask false. Strict common-subset comparisons must separately require success for every selected role; a retained set containing zero blocks is not strict coverage.

The Step 31 Chemical VAE features/train/all, paired comparisons, time/cost, GPU smoke, and Windows handoff YAML files are historical engineering acceptance inputs in the [configuration archive](backup/configs_20260916.tar.gz). For audits, extract it from the repository root to restore the original `configs/` paths. New tasks need their own separate YAML, rather than editing or reusing those small acceptance inputs.

Examples and execution evidence establish engineering acceptance, not predictive performance. Actual runs covered Linux CPU and `cuda:0`, one of the two available RTX 5090 GPUs. The `zinc_properties` encoder also ran a GPU smoke after comparison against original HDF5 TensorFlow/Keras and independent NumPy outputs under the frozen CPU float32 tolerance `atol=rtol=2e-5`. GPU-to-CPU-artifact differences are recorded separately and are never relabeled as C03 CPU parity. The user authorized marking Step 31 complete while deferring Windows CPU validation. Windows CPU has not been tested and must not be described as verified support. Full status is in `derived/chemical_vae/step31_9_gpu_smoke/verification.json`, the `derived/chemical_vae/step31_acceptance/index.json`, and the `derived/chemical_vae/step31_7_comparison/comparison_report.json`.

#### Chemical VAE dependencies and numerical comparisons

`h5py` in `requirements.txt` supports asset audits/conversion; it is not a required import for PyTorch runtime. TensorFlow/Keras must not become prerequisites for ordinary Chemical VAE feature generation, Morgan/MFP, or train-only manifest consumption.

### Models and parameters

Supported model identifiers are `rf`, `xgb`, `svm`, `lightgbm`, and `autogluon`. YAML loading normalizes model aliases. To avoid ambiguous experiment identities, do not specify an alias and its canonical name together.

Model parameters belong in the explicit API sections below. Omitted fields use the installed supported library's defaults; `null`, `false`, and `0` remain explicit values.

| Model | Available sections |
| --- | --- |
| `rf`, `xgb`, `lightgbm` | `estimator`, `fit`, `runtime` |
| `svm` | `estimator`, `fit`, `preprocessing`, `runtime` |
| `autogluon` | `predictor`, `fit`, `runtime` |

For XGBoost or LightGBM early stopping, declare only the internal validation strategy derived from the outer-training fold—rounds, validation fraction, and seed—under the model's `model_params.<model>.runtime.early_stopping`. Do not pass an external `eval_set`, callbacks, or arrays through `fit`.

Complete runnable fields are defined in [example.yaml](example.yaml); parameter routing and contracts are documented in the [Schema-2 contract](project-docs/docs/step30-contracts.md).

## Feature version operations

Inspect a published feature package with `yonod.artifacts.operations.inspect_feature_artifact(manifest_path)`. Derive a new version from an operations YAML with `yonod.artifacts.operations.derive_features(config_path)`. Old command-line scripts are archived locally under `backup/scripts/`.

`derive_features` supports only explicit sample selection, feature selection, and numerical CSV joins using unique sample IDs. It always publishes a new package and records the parent version, operations, input/output mappings, and hashes. Labels, duplicate/missing IDs, column conflicts, and unsafe fold-transformation operations are rejected.

Old JSON run configurations are not current executable entry points. Historical migration tools remain under local `backup/scripts/` for audits only. Create a separate schema-2 YAML following [example.yaml](example.yaml) for a new or resumed modeling task.

## Historical CSV CLI

`main.py --csv ...` remains a code-level debugging/historical compatibility interface, but must not launch modeling tasks. Its old `docs/`, `pictures/`, and `report/` layout is outside the schema-2 artifact and training contracts. New tasks must first save a separate YAML and then launch through `yonod.py`.

## Dedicated benchmarks and historical runs

Dedicated executors use schema-2 `stage: benchmark` and still start publicly through `yonod.py`. For research tasks, `outputs.root` is a separate `./result/<task_name>`, while `artifacts.output_dir` must be its `feature/` subfolder; they cannot be the same directory. Declare a 19-CPU budget for selected models. On Linux, bind exactly 19 available logical CPUs as required by [AGENTS.md](AGENTS.md); complete background-launch instructions are in the [example guide](example/README.md). The launch pattern is:

```bash
source /home/wangzh685/miniconda3/etc/profile.d/conda.sh
conda activate yonod
benchmark_cpu_ids="$(python -c 'import os; ids=sorted(os.sched_getaffinity(0)); assert len(ids)>=19, "19 available logical CPUs are required"; print(",".join(map(str, ids[:19])))')"
if [ -n "$benchmark_cpu_ids" ]; then
mkdir -p logs
nohup setsid bash -lc '
  source /home/wangzh685/miniconda3/etc/profile.d/conda.sh
  conda activate yonod
  printf "%s\n" "config/my_benchmark.yaml" | taskset --cpu-list "$1" python -u yonod.py
' _ "$benchmark_cpu_ids" </dev/null > logs/my_benchmark.log 2>&1 &
echo $! > logs/my_benchmark.pid
fi
```

`manifest_outer_cv` requires explicit `evaluation.grouping` and `benchmark.task_state: {backend: sqlite, resumable: true}`. When `evaluation.split_manifest` is provided, the program checks the CSV file SHA, complete membership, fold/repeat counts, roles, and group separation before importing and executing it. Otherwise, it generates a manifest from the grouping rules. It never falls back to ordinary `outer_kfold`. However, the dedicated executor also supports the random-split rule `repeated_kfold`, so “strict” does not automatically mean unseen substrates. This path needs `pyarrow` or `fastparquet` to publish splits, per-fold predictions, and metric evidence; `pyarrow` is listed in `requirements.txt`. A pinned installation command for an existing environment is `conda install -y --override-channels -c conda-forge pyarrow=16.1.0 numpy=1.26.4`. Multiple-candidate Tukey HSD additionally requires `statsmodels>=0.14`; a single-model/single-descriptor smoke does not invoke it.

`frozen_train_external_test` uses separate training and external-test CSVs, column mappings, and a source-audit file. Fix model settings in advance. For each encoding × model combination, fit once on the training table and then evaluate an external table with reference values. This protocol cannot contain CV fold-count, shuffling, or split-manifest fields. It currently does not support OHE, numerical conditions, HPO, AutoGluon, or external-test early stopping, and does not save a reusable final model bundle. Researchers must still establish independent provenance. Its reports are `report/external_test_report.html` and `.md`; predefined-membership/grouped internal evaluation reports are `report/benchmark_report.html` and `.md`.

A completed low-cost strict smoke used a separately saved 12-record synthetic fixture. Through `yonod.py`, it produced two component-held-out outer folds in internal cross-validation, SQLite state, parquet predictions/metrics, and HTML/Markdown reports. Those outer folds are not an independent external-source test. Configuration and output checks are in `derived/interface_migration/step30_acceptance/benchmark_schema2_preflight_evidence.md`.

The following are explicitly **outside migration and completion criteria**: all JSON run configurations and JSON-dependent launchers, including paper-exact and historical YieldMaster shell entry points, and `different_order/amide-coupling/run_different_order_tasks.py`. They remain historical/archive material, not supported modeling entry points. Do not add JSON compatibility branches or recreate the `different_order` runner to restore them.

Migration boundaries and verification evidence are in the [Schema-2 contract](project-docs/docs/step30-contracts.md) and `derived/interface_migration/step30_acceptance/acceptance_report.md`. Existing JSON metrics, predictions, run/fold manifests, official material, and `reference-proejct/` contents are historical/audit material and retain their original formats.

## Data and external assets

### Download datasets and model weights

With `yonod` activated and the Hugging Face CLI (`hf`) installed, run these commands from the YONOD repository root in Bash, PowerShell, or Conda Prompt.

[YONOD-datasets](https://huggingface.co/datasets/thinktraveller/YONOD-datasets) hosts redistributable project data assets, including cleaned and standardized USPTO data under `USPTO/`. Download into local `dataset/USPTO/`:

```bash
hf download thinktraveller/YONOD-datasets --repo-type dataset --include "USPTO/**" --local-dir dataset
```

[YONOD-weights](https://huggingface.co/thinktraveller/YONOD-weights) hosts redistributable model assets with a layout matching the project's `WEIGHTS/`. Download into local `WEIGHTS/`:

```bash
hf download thinktraveller/YONOD-weights --local-dir WEIGHTS
```

Check provenance and applicable terms before using data. Check upstream licenses and terms before using model weights.

### Amide-coupling dataset

The repository's `dataset/amide-coupling.csv` comes from [aichemeco/amide_coupling](https://github.com/aichemeco/amide_coupling/tree/main), under the MIT license. It contains 47,015 amide-coupling reactions, with `yield` normalized to `[0, 1]`. `dataset/benchmark_smoke_fixture.csv` is the small schema-2 smoke fixture. This README does not assume that `dataset/test-amide-coupling.csv` exists.

For publications using this dataset, cite Dai *et al.*, *Chem. Sci.*, 2025, DOI [10.1039/D5SC03364K](https://doi.org/10.1039/D5SC03364K).

| Column | Type | Meaning |
| --- | --- | --- |
| `row_id` | int | Candidate sample ID, subject to uniqueness validation |
| `sub_1_smiles`, `sub_2_smiles` | str | Substrate SMILES |
| `product_smiles` | str | Product SMILES; use is determined by YAML feature declarations |
| `activation`, `additive`, `base`, `solvent` | str | Reaction-component SMILES; feature declarations must handle missing and multicomponent semantics |
| `yield` | float | Regression target |

### MolMetaLM weights

- Hugging Face: [wudejian789/MolMetaLM-base](https://huggingface.co/wudejian789/MolMetaLM-base)
- GitHub: [CSUBioGroup/MolMetaLM](https://github.com/CSUBioGroup/MolMetaLM)
- Paper: DOI [10.48550/arXiv.2411.15500](https://doi.org/10.48550/arXiv.2411.15500)

```bash
pip install -U "huggingface_hub>=0.34"
hf download wudejian789/MolMetaLM-base --local-dir WEIGHTS/MolMetaLM-base
```

The download is approximately 500 MB. The directory should contain `config.json`, `model.safetensors`, tokenizer files, and related model assets.

### FISD weights

FISD requires three GCN weight files pretrained on QM9, approximately 19 MB in total. Obtain them from [KeantChen/FISD model](https://github.com/KeantChen/FISD/tree/main/model) and place them under `WEIGHTS/FISD/`:

```text
WEIGHTS/FISD/
├── qm_9_mse_model.pth
├── qm_9_cos_model.pth
└── qm_9_2in1_model.pth
```

Related paper: DOI [10.1039/D5SC00451A](https://doi.org/10.1039/D5SC00451A). Selecting `fisd` without these files causes an explicit feature-computation failure; other independent features can still complete.

### ATMOMACCS / MACCS

The project's `maccs` feature uses standard RDKit MACCS keys: 166 dimensions after removing placeholder bit 0 from the 167-bit representation. No ATMOMACCS asset download is required. If citing the corresponding experimental results, also follow the license and citation requirements of upstream [Zenodo 18669279](https://zenodo.org/records/18669279) and paper [10.1063/5.0308548](https://doi.org/10.1063/5.0308548).

## Project layout

Directory purposes, retention boundaries, and cleanup history are documented in the [directory guide](project-docs/docs/folder-organization.md). Browse [project documents](project-docs/docs/README.md), the [configuration archive](backup/configs_20260916.tar.gz), or [datasets](dataset/README.md).

```text
YONOD/
├── main.py                     Schema-2 runtime and smoke entry; historical CSV debugging interface
├── yonod.py                    Modeling-task entry point; generates/validates schema-2 YAML
├── example.yaml                Sole copyable schema-2 YAML specification and minimal smoke
├── example/                    Three executed synthetic teaching examples and downloads
│   ├── random_split/run.yaml   Random two-fold evaluation, RF/XGBoost/LightGBM
│   ├── fixed_split/run.yaml    Predefined two-fold membership, RF
│   └── external_test/run.yaml  Train on 12 records, externally test 6, RF
├── backup/                     Inactive results, scripts, and recovery material (Git-ignored)
│   └── configs_20260916.tar.gz Historical schema-2 and Chemical VAE acceptance YAML
├── WEIGHTS/chemical_vae/        Content-verified converted encoders (local; not trained from YAML)
├── yonod/
│   ├── config/                 YAML loading and schema contracts
│   ├── artifacts/              Immutable feature-package reading, writing, and derivation
│   ├── pipeline/               features, train, and all orchestration
│   ├── model_factory.py        Five-model parameter routing and effective-parameter audits
│   ├── descriptors/            Feature registry and descriptor implementations
│   └── benchmark/              Predefined/grouped internal and frozen-training external executors
├── project-docs/               Separate Git repository: goals, plan, build log, learning notes, docs/
├── reference-projects/         Local reference projects (Git-ignored)
├── dataset/                    Datasets and smoke fixtures
└── derived/                    Local acceptance and generated artifacts
```

## Verification and contribution notes

Run from the repository root with the `yonod` environment active:

```bash
python -m unittest tests.test_chemical_vae_audit tests.test_chemical_vae_encoder_parity \
  tests.test_chemical_vae_descriptor tests.test_chemical_vae_artifact_lineage \
  tests.test_chemical_vae_training_isolation tests.test_mfp_descriptor -v
python -m unittest tests.test_step30_contracts tests.test_step30_yaml_loader \
  tests.test_step30_feature_service tests.test_step30_training_service -v
```

Do not overwrite committed datasets, historical run outputs, feature packages, or `reference-proejct/`. Development takes place on Linux, but application code must remain Windows-compatible: use `pathlib.Path`/`os.path`, avoid hard-coded path separators, and preserve UTF-8 Chinese-path support.

## License

Project code and documentation use [CC BY-NC 4.0](LICENSE). Third-party data, model weights, and reference projects retain their own licenses. Check the relevant source's terms and citation requirements before using assets or publishing results.
