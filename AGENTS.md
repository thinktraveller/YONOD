# YONOD Agent Guide

## Required development environment

This project is developed on Linux and its required Conda environment is
`yonod`. Before running Python, tests, or dependency-management commands in a
Linux shell, activate it with:

```bash
source /home/wangzh685/miniconda3/etc/profile.d/conda.sh
conda activate yonod
```

Do not assume that the shell used by an agent has already initialized Conda.
When executing a one-off command, initialize and activate this environment in
the same shell invocation before calling `python`, `pip`, or test tools.

The expected baseline is Python 3.9. Project dependencies are listed in
`requirements.txt`; PyTorch and AutoGluon have additional installation notes
in `README.md` and `requirements.txt`.

## Mandatory modeling-task launch procedure

Every modeling task must follow this order from the repository root:

1. Create and save a standalone, valid schema-2 YAML run configuration before
   starting the task. Do not start a modeling task from ad-hoc CSV arguments,
   JSON, or by modifying a committed fixture such as `example.yaml`. Give each
   task its own YAML and its own `artifacts.output_dir` and `outputs.root`.
   Allocate a 19-CPU budget in that YAML for every model that supports it:
   use `model_params.<model>.estimator.n_jobs: 19` for RF, XGBoost, and
   LightGBM, and `model_params.autogluon.fit.num_cpus: 19` for AutoGluon.
   Declare only the sections for models actually selected by `models`.
2. In the shell that will run the task, initialize Conda and activate the
   `yonod` environment.
3. Start the task through the `yonod.py` main script. When the first prompt is
   shown, provide the already-created YAML path; the script validates it and
   launches the configured modeling run.

Use the 19 CPU cores within one modeling task, but schedule tasks linearly:
run one task to completion before starting the next. Do not launch multiple
YONOD runs, descriptors, model combinations, or CV folds as separate
background jobs merely to consume more cores. This avoids CPU oversubscription
and makes logs, resource usage, and task status unambiguous.

For example, add the applicable resource declaration to each task YAML:

```yaml
model_params:
  rf:
    estimator:
      n_jobs: 19
  xgb:
    estimator:
      n_jobs: 19
  lightgbm:
    estimator:
      n_jobs: 19
  autogluon:
    fit:
      num_cpus: 19
```

For an interactive run, use:

```bash
source /home/wangzh685/miniconda3/etc/profile.d/conda.sh
conda activate yonod
python -u yonod.py
```

On Linux, modeling tasks must be launched headlessly, with unbuffered verbose
output captured in a task-specific log. Replace the YAML path and task name
below; `python -u` preserves timely progress output, while YONOD also writes
its detailed runtime log under the configured output directory's `docs/`.
The `taskset` CPU list must contain exactly 19 available logical CPU IDs; the
example uses IDs `0` through `18`.

```bash
mkdir -p logs
nohup setsid bash -lc '
  source /home/wangzh685/miniconda3/etc/profile.d/conda.sh
  conda activate yonod
  printf "%s\\n" "configs/my_task.yaml" | taskset --cpu-list 0-18 python -u yonod.py
' </dev/null > logs/my_task.log 2>&1 &
echo $! > logs/my_task.pid
```

Do not use `main.py --config` as the direct task-launch command in this
procedure: `yonod.py` is the required entry point and will invoke the
configuration runtime after validating the YAML.

## Platform compatibility

Development happens on Linux, but all project changes must remain compatible
with both Linux and Windows.

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

Only these project-related files may be written to `project-docs/`:
`buildlog`, `project-plan`, `teaching`, and `goal.md`. Write every other project-related
file to the Git-ignored `docs/` directory instead. Do not place other files in
`project-docs/`.

## Verification

Place all test and validation scripts in the `_verify/` directory.

Run changed code from the repository root with the `yonod` environment active.
For a lightweight smoke check of the main configuration interface:

```bash
python main.py --config example.yaml
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

Use the PID file created at launch to check whether the headless task is still
running and to view its latest detailed output:

```bash
task_pid="$(cat logs/my_task.pid)"
ps -p "$task_pid" -o pid,ppid,stat,etime,cmd
taskset --pid --cpu-list "$task_pid"
tail -n 100 logs/my_task.log
```

Once task startup has been confirmed, provide the user with the command block
above and end the conversation. Do not keep polling, monitoring, or reporting
unchanged task status; the user can run the commands when they want an update.
