# VJETHBKM reproduction workspace

This directory contains an isolated reproduction workspace for Zotero item
`VJETHBKM`: "Yield Smarter, Not Harder: Good Practices for Machine Learning of
Reaction Outcomes".

All reproduction code, configs, manifests, cache, outputs, and reports should
stay under this directory. The YONOD main codebase is treated as a dependency
and is not modified by this reproduction stage.

## Current scope

- Stage A: OHE, Morgan fingerprint, and PhysChem baselines with RF and repeated
  5-fold CV smoke runs.
- Stage B: paper-target alignment, external validation, and component holdout
  splits after official data/code inspection.
- Stage C: DFT/SOAP extensions, model matrix expansion, reweighting, imbalance
  analysis, and polished reproduction reports.

## Version-control policy

Commit lightweight, reproducible files:

- `configs/*.yaml`
- `docs/*.md`
- `src/vjethbkm_repro/*.py`
- `scripts/*.py`
- `tests/*.py`
- `data/manifest/*.yaml`
- small summary tables under `outputs/tables/`

Do not commit raw article PDFs, downloaded data packages, large feature caches,
model binaries, or generated run directories by default. Register them in
`data/manifest/` with source, access date, checksum, license note, and local
status.

## Minimal commands

Run commands from `reference-proejct/vjethbkm/`:

```powershell
python scripts/check_environment.py
python scripts/run_benchmark.py --stage smoke
```

These commands are designed to keep outputs inside this workspace.

