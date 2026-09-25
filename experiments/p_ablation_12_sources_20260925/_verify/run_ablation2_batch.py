"""Generate/check/run the remaining 59 pre-registered tasks, strictly serially."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pandas as pd
import yaml
from yonod.benchmark.config import BenchmarkConfig, create_benchmark_contract
from yonod.benchmark.layout import resolve_benchmark_output_layout
from yonod.benchmark.metrics import rebuild_fold_metrics

QUEUE = ROOT / 'config' / 'ablation2_remaining59_queue.json'
BASE = ROOT / 'config' / 'ablation2_formal_morgan_rf_random_repeat_group_full_v1.yaml'
COMBOS = dict(full=['amine_smiles', 'acid_smiles', 'product_smiles'],
              minus_a=['acid_smiles', 'product_smiles'],
              minus_b=['amine_smiles', 'product_smiles'],
              minus_p=['amine_smiles', 'acid_smiles'], minus_a_p=['acid_smiles'],
              minus_b_p=['amine_smiles'], product_conditions=['product_smiles'], conditions_only=[])
GROUPS = dict(random_repeat_group='repeat_group_id', unseen_amine='amine_group_id',
              unseen_acid='acid_group_id', unseen_substrate_pair='substrate_pair_group_id')
CONDITIONS = ['activation_smiles', 'additive_smiles', 'base_smiles', 'solvent_smiles']

def sha(path):
    with Path(path).open('rb') as stream:
        digest = hashlib.sha256()
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()

def save_new(path, text):
    if path.exists():
        if path.read_text(encoding='utf-8') != text:
            raise RuntimeError(f'Refusing to overwrite different file: {path}')
        return
    with path.open('x', encoding='utf-8') as stream:
        stream.write(text)

def generate():
    base = yaml.safe_load(BASE.read_text(encoding='utf-8'))
    entries = []
    for descriptor, model in [('morgan', 'rf'), ('mfp', 'lightgbm')]:
        for protocol, group in GROUPS.items():
            for index, (combo, columns) in enumerate(COMBOS.items()):
                if descriptor == 'morgan' and protocol == 'random_repeat_group' and index < 5:
                    continue
                name = f'ablation2_formal_{descriptor}_{model}_{protocol}_{combo}_v1'
                raw = copy.deepcopy(base)
                raw['project_name'] = name
                spec = dict(id=f'{descriptor}_{combo}', descriptor=descriptor,
                            lifecycle='static_descriptor', mode='concat', columns=columns + CONDITIONS)
                if descriptor == 'mfp':
                    spec['params'] = dict(radius=3, fp_size=1024, profile='standard')
                raw['descriptors'] = [spec]
                raw['models'] = [model]
                params = dict(n_estimators=300, max_features=1.0, min_samples_leaf=1,
                              random_state=20260918, n_jobs=19) if model == 'rf' else dict(
                                  n_estimators=500, learning_rate=0.05, num_leaves=31,
                                  min_child_samples=1, random_state=20260918, n_jobs=19)
                raw['model_params'] = {model: {'estimator': params}}
                raw['artifacts']['output_dir'] = f'./result/{name}/feature'
                raw['outputs'] = dict(root=f'./result/{name}', report_formats=['html', 'markdown'])
                raw['evaluation']['grouping'] = dict(strategy='precomputed_column', group_column=group)
                raw['evaluation']['split_manifest'] = f'./result/ablation2_amide_splits_v1/manifests/{protocol}_split_manifest.parquet'
                raw['metadata'] = dict(notes=f'Pre-registered formal matrix: {descriptor} x {model}; {protocol}; {combo}.')
                path = ROOT / 'config' / f'{name}.yaml'
                save_new(path, yaml.safe_dump(raw, allow_unicode=True, sort_keys=False))
                BenchmarkConfig.from_file(path)
                entries.append(dict(task=name, config=str(path.relative_to(ROOT)), sha256=sha(path)))
    assert len(entries) == 59
    inputs = [ROOT / base['dataset']['path']] + [ROOT / f'result/ablation2_amide_splits_v1/manifests/{p}_split_manifest.parquet' for p in GROUPS]
    payload = dict(tasks=entries, inputs={str(p.relative_to(ROOT)): sha(p) for p in inputs})
    save_new(QUEUE, json.dumps(payload, ensure_ascii=False, indent=2) + '\n')
    print('Generated 59 independent YAML configurations and immutable queue.')

def audit(config):
    contract = create_benchmark_contract(config)
    layout = resolve_benchmark_output_layout(contract.run_dir)
    recorded = json.loads((layout.manifests / 'run_manifest.json').read_text(encoding='utf-8'))
    assert recorded['config_hash'] == contract.config_hash, 'config identity mismatch'
    assert recorded['dataset_sha256'] == contract.dataset_sha256, 'dataset identity mismatch'
    assert recorded['external_split_manifest']['sha256'] == sha(config.split_manifest_path)
    local = pd.read_parquet(layout.manifests / 'split_manifest.parquet')
    source = pd.read_parquet(config.split_manifest_path)
    assert set(local.run_id) == {contract.run_id}
    assert set(local.source_run_id) == set(source.run_id)
    cols = [c for c in source.columns if c != 'run_id']
    sort = ['repeat', 'fold', 'role', 'sample_id']
    pd.testing.assert_frame_equal(local[cols].sort_values(sort).reset_index(drop=True),
                                  source[cols].sort_values(sort).reset_index(drop=True))
    for _, part in local.groupby(['repeat', 'fold']):
        assert not set(part.loc[part.role == 'train', 'group_id']) & set(part.loc[part.role == 'valid', 'group_id'])
    db = layout.state / 'tasks.sqlite'
    with sqlite3.connect(db.as_uri() + '?mode=ro', uri=True) as connection:
        counts = dict(connection.execute('SELECT status, COUNT(*) FROM tasks GROUP BY status'))
    assert counts == {'succeeded': 15}, counts
    metrics = rebuild_fold_metrics(contract.run_dir)
    assert len(metrics.fold_metrics) == 15 and metrics.exclusions.empty
    assert metrics.completeness.is_complete.all()
    for suffix in ['html', 'md']:
        report = layout.report / f'benchmark_report.{suffix}'
        assert report.is_file() and report.stat().st_size > 0

def preflight():
    queue = json.loads(QUEUE.read_text(encoding='utf-8'))
    assert len(queue['tasks']) == 59
    assert len({entry['task'] for entry in queue['tasks']}) == 59
    for path, expected in queue['inputs'].items():
        assert sha(ROOT / path) == expected, f'Input changed: {path}'
    configs = []
    for entry in queue['tasks']:
        path = ROOT / entry['config']
        assert sha(path) == entry['sha256'], f'Configuration changed: {path}'
        config = BenchmarkConfig.from_file(path)
        assert config.outputs_root == ROOT / 'result' / entry['task']
        configs.append(config)
    for combo in list(COMBOS)[:5]:
        path = ROOT / 'config' / f'ablation2_formal_morgan_rf_random_repeat_group_{combo}_v1.yaml'
        audit(BenchmarkConfig.from_file(path))
    print('PASS: 59 configurations, input hashes, and five completed predecessors.', flush=True)
    return queue, configs

def run():
    # A portable exclusive lock prevents two batch invocations from overlapping.
    logs = ROOT / 'logs'
    logs.mkdir(exist_ok=True)
    lock = logs / 'ablation2_remaining59.lock'
    lock.mkdir()
    child = None
    try:
        (lock / 'owner.pid').write_text(str(os.getpid()), encoding='utf-8')
        queue, configs = preflight()
        cpus = sorted(os.sched_getaffinity(0))[:19] if sys.platform.startswith('linux') else []
        if sys.platform.startswith('linux') and len(cpus) != 19:
            raise RuntimeError('Exactly 19 available CPUs required.')
        for index, (entry, config) in enumerate(zip(queue['tasks'], configs), 1):
            name = entry['task']
            if config.outputs_root.exists() and any(config.outputs_root.iterdir()):
                audit(config)  # Incomplete or damaged outputs stop the queue.
                print(f'[{index}/59] verified complete, skipping {name}', flush=True)
                continue
            # Recheck pinned inputs immediately before every launch.
            for path, expected in queue['inputs'].items():
                assert sha(ROOT / path) == expected, f'Input changed: {path}'
            assert sha(ROOT / entry['config']) == entry['sha256']
            command = [sys.executable, '-u', str(ROOT / 'yonod.py')]
            if cpus:
                command = ['taskset', '--cpu-list', ','.join(map(str, cpus))] + command
            env = dict(os.environ, PYTHONUNBUFFERED='1', OMP_NUM_THREADS='19',
                       MKL_NUM_THREADS='19', OPENBLAS_NUM_THREADS='19')
            print(f'[{index}/59] launching {name}', flush=True)
            with (logs / f'{name}.log').open('x', encoding='utf-8') as output:
                child = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.PIPE,
                                         stdout=output, stderr=subprocess.STDOUT, text=True)
                (logs / f'{name}.pid').write_text(str(child.pid), encoding='utf-8')
                child.communicate(entry['config'] + '\n')
            if child.returncode:
                raise RuntimeError(f'{name} exited with code {child.returncode}; queue stopped.')
            child = None
            audit(config)  # yonod.py can return zero even when its adapter failed.
            print(f'[{index}/59] verified complete {name}', flush=True)
        print('All 59 tasks verified complete.', flush=True)
    finally:
        if child is not None and child.poll() is None:
            child.terminate()
            child.wait()
        (lock / 'owner.pid').unlink(missing_ok=True)
        lock.rmdir()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['generate', 'check', 'run'])
    args = parser.parse_args()
    if sys.version_info[:2] != (3, 9) or Path(sys.prefix).name != 'yonod':
        parser.error('Activate the yonod Conda environment (Python 3.9).')
    os.chdir(ROOT)
    {'generate': generate, 'check': preflight, 'run': run}[args.action]()
