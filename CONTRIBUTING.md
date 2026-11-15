# Contributing to YONOD

Read [AGENTS.md](AGENTS.md) and the [English](README.md) or [Chinese](README.zh-CN.md) guide before changing a runtime contract. Start with a small, reproducible issue or pull request that names the affected workflow: ordinary schema-2, fixed-membership benchmark, or frozen external testing.

## Environment and verification

Use Conda environment `yonod` with Python 3.9. Install CPU PyTorch first, then `requirements.txt`, following the platform-specific README instructions. AutoGluon and pretrained weights are optional and are not needed for the current Morgan smoke tests.

Linux Bash, from the repository root:

```bash
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate yonod
export PYTHONUTF8=1
python yonod.py --config example.yaml
```

Windows Conda PowerShell, from the repository root:

```powershell
conda activate yonod
$env:PYTHONUTF8 = "1"
python .\yonod.py --config example.yaml
```

This engineering smoke computes features and fits RF. Run it in an isolated clean checkout when preserving existing local outputs matters. For formal tasks, save a standalone schema-2 YAML under `config/`, run `python yonod.py`, and supply that existing path at the first prompt. Direct CSV modeling flags, including overrides combined with `--config`, are rejected; use `generate_yaml_config.py` first. Do not modify committed fixtures to start a research task.

Place local test and acceptance scripts under `_verify/`; do not commit that directory. CI creates its own verification scripts from the versioned workflow and must work from a fresh clone. Static CI covers configuration and entrypoint contracts. Maintainers start full runtime acceptance with the YONOD CI workflow's manual `workflow_dispatch` action; the initial `codex/github-remediation-20261115` validation branch also triggers it. Ordinary pushes and PRs run static checks. Full acceptance runs the engineering smoke and the three prepared tests serially on Linux, then on Windows; it checks saved predictions, split membership, metrics and HTML/Markdown reports. Record the operating system actually tested. Static or Linux-only success does not establish Windows runtime acceptance.

Keep source IDs, roles, input hashes, fold membership and train-only transformations explicit. Tests for invalid inputs should fail before creating task outputs. Low or negative scores on artificial smoke data do not constitute a runtime failure or a claim about real chemical prediction.

## Repository boundaries and commits

- Keep the main checkout on `main`. `project-docs/` is a separate private Git repository, not a submodule. Public contributors do not need access to that private repository.
- Explicitly stage task-owned paths and inspect the index before each commit. Never use `git add .` or `git add -A`. Commit actual changes separately in each repository; do not create empty commits.
- Under `config/`, only the three configurations in `config/example/` are tracked; root `example.yaml` is also tracked as the authoritative engineering smoke. Research YAMLs under `config/`, all generated results, root `logs/`, `_verify/`, `backup/` and `reference-projects/` stay local. Only the three empty result `.gitkeep` files are tracked beneath `result/`.
- Put operational logs and PID files in root `logs/`; use isolated per-task output directories. Do not include credentials, private data, model binaries or run outputs in a PR.
- Use `pathlib.Path`, UTF-8 and platform-aware instructions. Preserve historical datasets and result evidence.

The following Git commands work in Bash and PowerShell; replace the explicit paths and message with those for your change:

```text
git status --short
git add path/to/changed-file
git diff --cached --name-only
git diff --cached --check
git commit -m "fix: describe the corrected behavior"
```

Maintainers record private project-document changes in the independent documentation repository. For required CI checks, push the same commit to a `codex/` validation branch first, then normally update the default branch once checks pass; a PR is also supported. Do not force-push or delete protected default branches. Code ownership routes review and does not require an unavailable second maintainer's approval.

## Issues, security and releases

An issue should include platform, dependency versions, a minimal shareable example, the YAML path or a redacted excerpt, the exact error, and expected behavior. Remove credentials and private reaction data from logs. Report vulnerabilities through [SECURITY.md](SECURITY.md), rather than a public issue.

A release requires a reviewed source commit, recorded environment and dependency versions, passing static checks and separate Linux/Windows runtime evidence for the advertised workflows. Select an initial version only after that acceptance; do not move an already published tag. Release notes must name supported workflows, platforms, migration changes and remaining limitations. Public assets must exclude private project documents, credentials, research configurations and local run outputs. Maintain the source/documentation SHA mapping privately. A successful local fix or a release checklist is not itself a published release.
