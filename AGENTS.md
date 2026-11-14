# YONOD Agent Guide

## Required development environment

Use the Conda environment `yonod` on both Linux and Windows. The expected
baseline is Python 3.9. Dependencies are listed in `requirements.txt`;
PyTorch and optional AutoGluon have separate installation notes in
`README.md` and `requirements.txt`.

On Linux Bash, initialize and activate Conda in the same shell invocation
before running Python, tests, or dependency-management commands:

```bash
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate yonod
```

Replace `$HOME/miniconda3` with the local Conda installation directory if it
is different. Do not assume a new shell has initialized Conda.

On Windows, open Anaconda PowerShell Prompt, or initialize PowerShell with
`conda init powershell` and reopen it. Then activate the environment:

```powershell
conda activate yonod
```

Plain Conda Prompt supports the foreground `conda`, `pip`, and `python`
commands, but the Windows background/status examples below require
PowerShell. Preserve UTF-8 when reading or writing configuration and data.

## Mandatory modeling-task launch procedure

Every modeling task must follow this order from the repository root:

1. Create and save a standalone, valid schema-2 YAML run configuration under
   `config/` before starting the task (create the directory if needed). Do not
   start a modeling task from ad-hoc CSV arguments, JSON, or by modifying a
   committed fixture such as `example.yaml`. Give each task a unique task name
   and set its output paths beneath the single isolated directory
   `result/<task_name>/`: `artifacts.output_dir` must be a subdirectory of it
   (for example, `./result/<task_name>/feature`) and `outputs.root` must be
   `./result/<task_name>`. Never share an output directory between tasks or set
   final modeling outputs to a path under `derived/`. By default, every
   modeling YAML must generate both HTML and Markdown reports by setting
   `outputs.report_formats: [html, markdown]`.
   Every modeling YAML must follow the schema-2 structure, field semantics,
   and path conventions in `example.yaml`. This file is the sole authoritative
   YAML specification; README may explain it but must not duplicate a complete
   YAML configuration block. Declare only parameters for selected models.
   Thread counts are optional and machine-dependent: RF/XGBoost/LightGBM use
   `model_params.<model>.estimator.n_jobs`, and AutoGluon uses
   `model_params.autogluon.fit.num_cpus`. Leave them unspecified to use the
   supported installed library's defaults, or explicitly choose resources
   suitable for the local machine. Do not require a fixed workstation CPU
   count or CPU affinity in templates, tutorials, or launch validation.
2. In the shell that will run the task, initialize Conda as appropriate for
   the platform and activate `yonod`.
3. Start through `yonod.py`. At the first prompt, provide the already-created
   YAML path; the script validates it and launches the configured run.

Teaching examples under `example/` are an exception to the root `config/`
location: each has one executable `run.yaml`, inputs relative to its own
folder, and outputs under its own `result/<task_name>/`. From that folder,
run `python ../../yonod.py` on Linux or `python ..\..\yonod.py` on Windows
and enter `run.yaml`; no configuration copy is required. Keep schema-2,
serial execution, dual reports, and repository-root `logs/` for operational
logs and PID files. Research task YAMLs still belong in root `config/`.

Schedule tasks linearly: run one task to completion before starting the
next. Do not launch multiple YONOD runs, descriptors, model combinations,
or CV folds as separate background jobs merely to consume more cores.

For a foreground run from the repository root, Linux Bash:

```bash
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate yonod
export PYTHONUTF8=1
python -u yonod.py
```

Windows Conda PowerShell:

```powershell
conda activate yonod
$env:PYTHONUTF8 = "1"
python -u .\yonod.py
```

For unattended runs, use unbuffered output and task-specific logs and PID
files in root `logs/`. Replace `my_task` and the YAML path below.

Linux Bash, from the repository root:

```bash
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate yonod
export PYTHONUTF8=1
mkdir -p logs
printf '%s\n' 'config/my_task.yaml' > logs/my_task.stdin.txt
nohup python -u yonod.py < logs/my_task.stdin.txt > logs/my_task.log 2>&1 &
echo $! > logs/my_task.pid
```

Windows Conda PowerShell, from the repository root:

```powershell
conda activate yonod
$env:PYTHONUTF8 = "1"
$yonodLogs = Join-Path $PWD.Path "logs"
New-Item -ItemType Directory -Path $yonodLogs -Force | Out-Null
$yonodInput = Join-Path $yonodLogs "my_task.stdin.txt"
"config/my_task.yaml" | Set-Content -LiteralPath $yonodInput -Encoding ascii
$yonodProcess = Start-Process -FilePath (Get-Command python).Source -ArgumentList @("-u", "yonod.py") -WorkingDirectory $PWD.Path -RedirectStandardInput $yonodInput -RedirectStandardOutput (Join-Path $yonodLogs "my_task.log") -RedirectStandardError (Join-Path $yonodLogs "my_task.stderr.log") -PassThru
$yonodProcess.Id | Set-Content -LiteralPath (Join-Path $yonodLogs "my_task.pid") -Encoding ascii
```

The ASCII stdin files above contain only the shown ASCII YAML paths. For a
YAML path containing non-ASCII characters in PowerShell, write the input
without a BOM using `[IO.File]::WriteAllText($yonodInput, "配置路径.yaml`n", [Text.UTF8Encoding]::new($false))`;
`PYTHONUTF8=1` makes the launcher and its child processes read it consistently.
Windows captures standard output and standard errors in separate logs;
inspect both. Keep all operational logs, stdin support files, and PID files
in root `logs/`, not under modeling results.

Do not use `run_modeling.py --config` as the formal task-launch command:
`yonod.py` is the formal entry point and invokes the ordinary runtime after
validation, or dispatches a dedicated benchmark protocol.

## Standard YAML generation

The entry scripts have distinct roles: `generate_yaml_config.py` prepares
standard schema-2 configurations, `yonod.py` validates and launches tasks,
and `run_modeling.py` executes the ordinary `features`/`train`/`all` runtime.
The direct runtime command in Verification is an engineering smoke exception,
not a replacement for the formal modeling-task launch procedure.

For a new CSV, start the dedicated generator from the repository root.
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

Complete its column-role, encoding, model, and task questions. The generator
saves a validated `config/<task_name>.yaml` directly; no YAML copy is needed.
It may write normalized/prepared CSVs and invalid-input reports under the
isolated `result/<task_name>/feature/docs/`, but must not calculate descriptors,
fit models, predict, or automatically launch a modeling task. Existing YAML
input is validated only. Review the saved YAML, then launch `yonod.py` and
provide that path at the first prompt. Teaching `run.yaml` files are already
prepared and do not need to pass through the generator.

Executable run configurations use schema-2 YAML. JSON audit records and
manifests remain supported evidence formats; they are not run configurations.

## Platform compatibility

All project changes and current tutorials must support Linux and Windows.
Label shell-specific commands and give separate Bash and Windows Conda
PowerShell alternatives for setup, launch, file operations, and status checks.
Commands that are identical on both systems may be shown once and labeled
accordingly. Do not infer Windows runtime acceptance from Linux execution.

- Use `pathlib.Path` or `os.path` for paths in Python; never hard-code `/` or
  `\\` as a path separator.
- Resolve project files from configurable or repository-relative paths rather
  than a developer's absolute Linux path.
- Avoid Linux-only commands, shell syntax, process signals, and assumptions
  about case-sensitive filesystems in application code.
- Keep command-line examples platform-aware: label Bash-only commands as such
  and provide a PowerShell/Conda Prompt alternative when it is useful.
- Preserve UTF-8 handling for the repository's Chinese filenames and dataset
  paths.

## Project documentation locations

The main repository remains on `main`. `project-docs/` is an independent Git
repository and is ignored by the main repository. Keep its root for
`goal.md`, `project-plan.md`, `buildlog.md`, and `teach.md`; put
other maintained project documentation under `project-docs/docs/`. Historical
documents and evidence from the former root `docs/` directory were migrated
there; do not recreate a root `docs/` directory for project documentation.

Commit changes to `project-docs/` in its own Git repository in the same task.
When a task changes both repositories, make a separate commit in each. Do not
stage `project-docs/` in the main repository or add it as a submodule.

## Output locations

Write new backups and engineering-support files under `backup/`. Once results
are no longer needed for active work, move them into `backup/`, keeping each
task's files together so the original run remains identifiable. `backup/` is
excluded from the main Git repository.
For each
project run, all result files must be contained in that task's designated
`result/<task_name>/` directory; do not write loose run results directly under
`result/`, share result directories between tasks, or place final modeling
results under `derived/`. All modeling-task operational logs and PID files
must be placed in `logs/` (for example, `logs/<task_name>.log` and
`logs/<task_name>.pid`); do not write task logs under `result/`.

## Verification

Place all test and validation scripts in the `_verify/` directory.

Run changed code from the repository root with the `yonod` environment active.
For a lightweight smoke check of the ordinary configuration runtime, the same
command works in Linux Bash and Windows Conda PowerShell after activation:

```text
python run_modeling.py --config example.yaml
```

`example.yaml` is intentionally a small, deterministic smoke configuration:
it runs the `morgan` descriptor with the `rf` model on a 12-sample fixture,
using two outer CV folds. A successful run reports that all features are ready
and that `morgan × rf` is complete, for example:

```text
[all] features=ready status_manifest=...
  - morgan × rf: complete .../run_manifest.yaml
```

The identifiers in the manifest paths are content-derived and may differ when
the fixture, configuration, or implementation changes; verify the `ready` and
`complete` statuses rather than matching a fixed path or run ID.

Avoid overwriting committed datasets, feature artifacts, historical run
outputs, or reference projects unless the task explicitly requires it.

## Query modeling-task status

Use the PID file from launch to check whether the headless task is still
running and inspect its detailed output. From the repository root, Linux Bash:

```bash
task_pid="$(cat logs/my_task.pid)"
ps -p "$task_pid" -o pid,ppid,stat,etime,cmd
tail -n 100 logs/my_task.log
```

Windows Conda PowerShell:

```powershell
$taskPid = [int](Get-Content logs\my_task.pid)
Get-Process -Id $taskPid -ErrorAction SilentlyContinue
Get-Content logs\my_task.log -Encoding UTF8 -Tail 100
Get-Content logs\my_task.stderr.log -Encoding UTF8 -Tail 100
```

Once task startup has been confirmed, provide the user with the appropriate
platform's command block and end the conversation. Do not keep polling,
monitoring, or reporting unchanged status. When complete end-to-end acceptance
of teaching examples is explicitly requested, run them serially to completion
and verify saved predictions, split membership, metrics, and reports before
finishing. Record the actual tested operating system; do not label another
platform as tested based solely on its documented commands.

<!-- --- project-initializer (managed) START --- -->
## Repository layout and collaboration rules

The main project repository stays on `main`. `project-docs/` is a separate Git
repository that includes its `docs/` subdirectory. The main repository ignores
`project-docs/`, `_verify/`, `config/`, `reference-projects/`, and `backup/`.

### Required directory locations

- Put all test, validation, and acceptance scripts under `_verify/`. Do not
  commit that directory to either Git repository.
- Put task YAML files under `config/`. Keep the directory local and excluded
  from the main Git repository.
- Put the agent documents `goal.md`, `project-plan.md`, `buildlog.md`, and
  `teach.md` at the root of `project-docs/`. Their responsible agents maintain
  their contents; do not create empty placeholders for them.
- Put all other maintained project documents, including analyses and acceptance
  reports, under `project-docs/docs/`. The former root `docs/` contents are
  stored there; do not recreate a root `docs/` directory for project documents.
- Put external project source, examples, assets, and other reference material
  under `reference-projects/`. Do not commit it to either Git repository.
- Keep application source and configuration in the main project tree, outside
  `project-docs/`, `_verify/`, and `reference-projects/`.
- Put new backups and engineering support files under `backup/`. Move results
  there once they are no longer needed for active work, keeping each task's
  files together. Keep `backup/` completely excluded from Git.
- Do not recreate the former root `scripts/` directory; its contents were
  archived under `backup/`. Put validation scripts in `_verify/`.

### Required commits

At the end of each task, commit the task's changes separately in both Git
repositories: one main project commit and one `project-docs/` commit. The main
commit must exclude `project-docs/`, `_verify/`, `config/`,
`reference-projects/`, and `backup/`; the documentation commit contains the
task's maintained documents.
If a repository has no changes to commit, report that fact instead of making
an empty commit. Do not declare a task with pending changes complete.

Inspect each staging area before committing, and explicitly list only the
files changed by the task. Never use `git add .` or `git add -A` in either
repository. Never commit `.env`, credentials, or files from `_verify/`,
`config/`, `reference-projects/`, or `backup/`.

```text
# Git commands below work in Bash and PowerShell; replace paths and message.
# Main repository
git status --short
git add AGENTS.md
git diff --cached --name-only
git commit -m "docs: update agent guidance"

# Independent documentation repository (only when it has task changes)
git -C project-docs status --short
git -C project-docs add docs/path-to-changed-document.md
git -C project-docs diff --cached --name-only
git -C project-docs commit -m "docs: update project documentation"
```
<!-- --- project-initializer (managed) END --- -->
