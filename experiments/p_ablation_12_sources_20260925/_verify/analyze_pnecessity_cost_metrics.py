"""Audit saved Full/no-P predictions and fit timers; never fit models.

Writes derived numerical/figure artifacts to its own result directory.  All
intervals are conditional, pointwise cluster-bootstrap intervals, not claims
of independent training replications or randomized runtime benchmarking.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr

ROOT = Path(__file__).resolve().parents[1]
FILES = {}


def record(path):
    path = Path(path)
    FILES[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return path


def read_json(path):
    return json.loads(record(path).read_text(encoding="utf-8"))


def read_table(path):
    path = record(path)
    return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)


def discover():
    pairs = []
    partitions = read_table(ROOT / "result/ablation2_statistics_v1/tables/repeat_partition_signatures.csv")
    for line in ("morgan_rf", "mfp_lightgbm"):
        for protocol in partitions.protocol.unique():
            reps = partitions.loc[(partitions.protocol == protocol) & partitions.is_representative, "repeat"].tolist()
            prefix = f"ablation2_formal_{line}_{protocol}"
            pairs.append(dict(batch="amide_development", source=protocol, line=line,
                              full=f"{prefix}_full_v1", no_p=f"{prefix}_minus_p_v1", repeats=reps))
        for tag, batch in (("frozen", "amide_frozen_external"), ("unseen_amine", "amide_same_source")):
            prefix = f"ablation2_chemrxiv_{tag}_{line}"
            pairs.append(dict(batch=batch, source="chemrxiv_amide", line=line,
                              full=f"{prefix}_full_v1", no_p=f"{prefix}_minus_p_v1", repeats=[1] if tag == "frozen" else [1, 2, 3]))
    matrix = read_table(ROOT / "result/pnecessity_utility_prepare_v1/static_config_matrix.csv")
    for (source, line), sub in matrix.groupby(["source_id", "line"], sort=True):
        tasks = sub.set_index("arm").task_name.to_dict()
        pairs.append(dict(batch="legacy_nine", source=source, line=line, full=tasks["full"], no_p=tasks["minus_p"], repeats=[1, 2, 3]))
    matrix = read_table(ROOT / "project-docs" / "docs" / "product_necessity_multidataset" / "numeric_conditions_analysis_v1" / "tables" / "line_statistics.csv")
    for row in matrix.itertuples():
        pairs.append(dict(batch="numeric_four", source=row.source_id, line=row.line,
                          full=row.full_task, no_p=row.minus_p_task, repeats=[1, 2, 3]))
    for line in ("morgan_rf", "mfp_lightgbm"):
        prefix = f"pnecessity_nonstandard_sufex_{line}"
        pairs.append(dict(batch="sufex", source="sufex", line=line, full=f"{prefix}_full_v1", no_p=f"{prefix}_minus_p_v1", repeats=[1, 2, 3]))
    assert len(pairs) == 40
    return pairs


def scores(y, p):
    sst = np.sum((y-y.mean())**2)
    assert sst > 0
    e = y-p
    return {"mae": float(np.abs(e).mean()), "rmse": float(np.sqrt((e*e).mean())),
            "r2": float(1-(e*e).sum()/sst), "kendall_tau": float(kendalltau(y, p).statistic),
            "spearman_rho": float(spearmanr(y, p).statistic)}


def load_task(task, frozen):
    base = ROOT / "result" / task
    dirs = [p for p in base.glob("**/docs/predictions") if list(p.glob("*.parquet"))]
    assert len(dirs) == 1, (task, dirs)
    docs = dirs[0].parent
    manifest = read_json(docs / "manifests/run_manifest.json")
    preds = [read_table(p) for p in sorted(dirs[0].glob("*.parquet"))]
    folds = [read_json(p) for p in sorted((docs / "folds").glob("*.json"))]
    expected = 1 if frozen else 15
    assert len(preds) == len(folds) == expected, task
    df = pd.concat(preds, ignore_index=True)
    assert set(df.run_id) == {manifest["run_id"]}
    if frozen:
        df["repeat"] = 1
        df["fold"] = 1
        df["group_id"] = df["external_group_id"]
    else:
        db = docs / "state/tasks.sqlite"
        with sqlite3.connect(db.resolve().as_uri()+"?mode=ro", uri=True) as c:
            statuses = dict(c.execute("select status,count(*) from tasks group by status"))
        assert statuses == {"succeeded": 15}, (task, statuses)
    assert not df.duplicated(["sample_id", "repeat"]).any(), task
    assert np.isfinite(df[["y_true", "y_pred"]].to_numpy()).all()
    timing = []
    meta = {}
    for f in folds:
        key = (f.get("repeat", 1), f.get("fold", 1))
        assert key not in meta
        meta[key] = f
        tr, pr = float(f["train_time_s"]), float(f["predict_time_s"])
        assert tr > 0 and pr >= 0 and np.isfinite([tr, pr]).all()
        timing.append(dict(task=task, repeat=key[0], fold=key[1], train_s=tr, predict_s=pr, model_s=tr+pr))
    timing = pd.DataFrame(timing)
    metrics_path = docs / "metrics/fold_metrics.parquet"
    reported_r2 = np.nan
    if metrics_path.exists():
        metrics = read_table(metrics_path)
        assert len(metrics) == expected
        for r in metrics.itertuples():
            part = df[(df.repeat == r.repeat) & (df.fold == r.fold)]
            calc = scores(part.y_true.to_numpy(), part.y_pred.to_numpy())
            for k in ("mae", "rmse", "r2"):
                assert np.isclose(calc[k], getattr(r, k), rtol=1e-7, atol=1e-9), (task, k)
            t = meta[(r.repeat, r.fold)]
            assert np.isclose(t["train_time_s"], r.train_time_s)
        reported_r2 = float(metrics.r2.mean())
    time_path = docs / "metrics/combination_time_summary.parquet"
    if time_path.exists():
        table = read_table(time_path)
        assert len(table) == 1
        for field, col in (("total_train_time_s", "train_s"), ("total_predict_time_s", "predict_s")):
            assert np.isclose(float(table[field].iloc[0]), timing[col].sum()), (task, field)
    return df, timing, meta, reported_r2, docs


def cluster_intervals(frame, draws, seed):
    # Repeat-averaged sample losses; averaging predictions would estimate an ensemble instead.
    d = pd.DataFrame({"sample": frame.sample_id, "group": frame.group_id_f,
                      "y": frame.y_true_f, "y2": frame.y_true_f**2,
                      "af": abs(frame.y_true_f-frame.y_pred_f), "an": abs(frame.y_true_n-frame.y_pred_n),
                      "sf": (frame.y_true_f-frame.y_pred_f)**2, "sn": (frame.y_true_n-frame.y_pred_n)**2})
    d = d.groupby(["sample", "group"], sort=True).mean().reset_index()
    d["n"] = 1
    columns = ["n", "y", "y2", "af", "an", "sf", "sn"]
    groups = d.groupby("group", sort=True)[columns].sum().to_numpy()

    def calc(total):
        n, y, y2, af, an, sf, sn = total.T
        sst = y2-y*y/n
        return np.column_stack(((an-af)/n, np.sqrt(sn/n)-np.sqrt(sf/n), (sn-sf)/np.where(sst > 1e-14, sst, np.nan)))

    rng = np.random.default_rng(seed)
    out = []
    for start in range(0, draws, 16):
        ix = rng.integers(0, len(groups), size=(min(16, draws-start), len(groups)))
        out.append(calc(groups[ix].sum(axis=1)))
    out = np.concatenate(out)
    result = {"n_samples": len(d), "n_groups": len(groups)}
    total = groups.sum(axis=0)
    result.update(full_mae=total[3]/total[0], no_p_mae=total[4]/total[0],
                  full_rmse=np.sqrt(total[5]/total[0]), no_p_rmse=np.sqrt(total[6]/total[0]),
                  full_r2=1-total[5]/(total[2]-total[1]**2/total[0]), no_p_r2=1-total[6]/(total[2]-total[1]**2/total[0]))
    point = calc(total[None, :])[0]
    for i, k in enumerate(("mae", "rmse", "r2")):
        result[f"delta_{k}"] = point[i]
        result[f"delta_{k}_ci_low"], result[f"delta_{k}_ci_high"] = np.nanquantile(out[:, i], [.025, .975])
        result[f"{k}_bootstrap_finite_draws"] = int(np.isfinite(out[:, i]).sum())
    return result


def audit_pair(pair, draws):
    frozen = pair["batch"] == "amide_frozen_external"
    f, tf, mf, rf, df = load_task(pair["full"], frozen)
    n, tn, mn, rn, dn = load_task(pair["no_p"], frozen)
    assert mf.keys() == mn.keys()
    for key in mf:
        a, b = mf[key], mn[key]
        # Recent split_hash includes arm-specific run_id: compare actual
        # train/valid identity instead, then pair every saved prediction below.
        for field in ("train_sample_ids_hash", "valid_sample_ids_hash", "train_sample_id_sha256", "external_sample_id_sha256", "n_train", "n_valid"):
            if field in a or field in b:
                assert a.get(field) == b.get(field), (pair, field)
        pa = a.get("factory_audit", a.get("estimator_params_snapshot", {})).get("effective_estimator_params")
        pb = b.get("factory_audit", b.get("estimator_params_snapshot", {})).get("effective_estimator_params")
        assert pa and pb and pa == pb, (pair, "estimator params")
        if "software_versions" in a and "software_versions" in b:
            assert a["software_versions"] == b["software_versions"], (pair, "versions")
        if "numeric_conditions" in (a.get("feature_transformer") or {}):
            assert a["feature_transformer"]["numeric_conditions"] == (b.get("feature_transformer") or {}).get("numeric_conditions")
    keys = ["sample_id", "repeat", "fold"]
    cols = keys + ["group_id", "y_true", "y_pred"]
    m = f[cols].merge(n[cols], on=keys, how="outer", suffixes=("_f", "_n"), validate="one_to_one", indicator=True)
    assert m._merge.eq("both").all() and m.y_true_f.eq(m.y_true_n).all() and m.group_id_f.eq(m.group_id_n).all()
    counts = m.groupby("repeat").sample_id.nunique()
    assert counts.nunique() == 1
    expected_ids = set(m.loc[m.repeat == counts.index[0], "sample_id"])
    assert all(set(p.sample_id) == expected_ids for _, p in m.groupby("repeat"))
    m = m[m.repeat.isin(pair["repeats"])].copy()
    identity = "|".join((pair["batch"], pair["source"], pair["line"]))
    seed = int.from_bytes(hashlib.sha256(identity.encode()).digest()[:4], "little")
    row = {**pair, "pair_id": identity, "repeats": ",".join(map(str, pair["repeats"])),
           "bootstrap_draws": draws, "bootstrap_seed": seed,
           "full_prediction_dir": str(df / "predictions"), "no_p_prediction_dir": str(dn / "predictions"),
           "full_report_fold_mean_r2": rf, "no_p_report_fold_mean_r2": rn,
           "input_audit_scope": "legacy_input_blocks_unverified" if pair["batch"] == "legacy_nine" else "saved_artifact_pairing_not_retraining"}
    row.update(cluster_intervals(m, draws, seed))
    repeat_rows = []
    for rep, part in m.groupby("repeat"):
        sf = scores(part.y_true_f.to_numpy(), part.y_pred_f.to_numpy())
        sn = scores(part.y_true_n.to_numpy(), part.y_pred_n.to_numpy())
        r = {"pair_id": identity, "repeat": rep}
        for metric in sf:
            r[f"full_{metric}"] = sf[metric]
            r[f"no_p_{metric}"] = sn[metric]
            r[f"delta_{metric}"] = sf[metric]-sn[metric] if metric in ("r2", "kendall_tau", "spearman_rho") else sn[metric]-sf[metric]
        repeat_rows.append(r)
    repdf = pd.DataFrame(repeat_rows)
    row["n_metric_repeats"] = len(repdf)
    for metric in ("mae", "rmse", "r2", "kendall_tau", "spearman_rho"):
        for arm in ("full", "no_p", "delta"):
            c = f"{arm}_{metric}"
            row[f"{c}_repeat_mean"] = repdf[c].mean()
            row[f"{c}_repeat_sd"] = repdf[c].std(ddof=1)
    for metric in ("mae", "rmse"):
        row[f"{metric}_reduction_pct"] = 100*row[f"delta_{metric}"]/row[f"no_p_{metric}"]
    t = tf.merge(tn, on=["repeat", "fold"], suffixes=("_full", "_no_p"), validate="one_to_one")
    t["pair_id"] = identity
    for metric in ("train", "predict", "model"):
        a, b = t[f"{metric}_s_full"].sum(), t[f"{metric}_s_no_p"].sum()
        row[f"full_{metric}_s"] = a
        row[f"no_p_{metric}_s"] = b
        row[f"{metric}_increase_s"] = a-b
        row[f"{metric}_increase_pct"] = 100*(a/b-1) if b else np.nan
        row[f"no_p_{metric}_saving_pct"] = 100*(1-b/a)
        t[f"{metric}_increase_pct"] = 100*(t[f"{metric}_s_full"]/t[f"{metric}_s_no_p"]-1)
        for arm in ("full", "no_p"):
            row[f"{arm}_{metric}_fold_mean_s"] = t[f"{metric}_s_{arm}"].mean()
            row[f"{arm}_{metric}_fold_sd_s"] = t[f"{metric}_s_{arm}"].std(ddof=1)
    row["n_timed_fits_per_arm"] = len(t)
    row["time_inference"] = "descriptive_only_single_historical_schedule_no_randomized_replicates"
    row["feature_time_increase_pct"] = np.nan
    row["end_to_end_time_increase_pct"] = np.nan
    return row, repeat_rows, t


def figures(summary, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "pdf.fonttype": 42})
    recent = summary[summary.batch.isin(["legacy_nine", "numeric_four", "sufex"])].sort_values(["source", "line"])
    fig, axes = plt.subplots(1, 2, figsize=(12, 10), sharey=True, constrained_layout=True)
    y = np.arange(len(recent))
    labels = [f'{r.source} / {"RF" if r.line == "morgan_rf" else "LGBM"}' for r in recent.itertuples()]
    axes[0].barh(y, recent.train_increase_pct, color="#0072B2")
    axes[0].set_xlabel("Fit time increase with P (%)\nHistorical measurements; no timing CIs")
    axes[0].set_yticks(y, labels)
    axes[0].invert_yaxis()
    for i, r in enumerate(recent.itertuples()):
        axes[1].plot([r.delta_r2_ci_low, r.delta_r2_ci_high], [i, i], color="#555555")
        axes[1].plot(r.delta_r2, i, "o", color="#D55E00", markersize=4)
    axes[1].set_xlabel("R-squared gain with P (absolute)\n95% pointwise conditional cluster intervals")
    for ax in axes:
        ax.axvline(0, color="black", linewidth=.7)
        ax.grid(axis="x", alpha=.2)
    for ext in ("png", "pdf"):
        fig.savefig(out/f"figure-01-time-r2.{ext}", dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    for batch, sub in summary.groupby("batch", sort=False):
        ax.scatter(sub.train_increase_pct, sub.delta_r2, label=batch, s=32, alpha=.8)
    ax.axhline(0, color="black", linewidth=.8)
    ax.axvline(0, color="black", linewidth=.8)
    ax.set_xlabel("Fit time increase with P (%)")
    ax.set_ylabel("R-squared gain with P (absolute)")
    ax.legend(fontsize=8)
    ax.grid(alpha=.2)
    for ext in ("png", "pdf"):
        fig.savefig(out/f"figure-02-cost-benefit.{ext}", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draws", type=int, default=4000)
    args = parser.parse_args()
    assert args.draws >= 1000
    out = ROOT / "result/pnecessity_cost_metrics_audit_20260924"
    out.mkdir(exist_ok=True)
    summaries, repeats, timings = [], [], []
    for pair in discover():
        row, rep, time = audit_pair(pair, args.draws)
        summaries.append(row); repeats.extend(rep); timings.append(time)
        print(f"verified {len(summaries)}/40 {row['pair_id']} time={row['train_increase_pct']:+.2f}% R2={row['delta_r2']:+.6f}", flush=True)
    df = pd.DataFrame(summaries)
    df.to_csv(out/"paired_summary.csv", index=False)
    pd.DataFrame(repeats).to_csv(out/"repeat_metrics.csv", index=False)
    pd.concat(timings, ignore_index=True).to_csv(out/"fold_timings.csv", index=False)
    figures(df, out)
    selected = {r[k] for r in summaries for k in ("full", "no_p")}
    inventory = []
    for base in sorted((ROOT/"result").iterdir()):
        if not base.is_dir():
            continue
        reports = sorted({str(p.relative_to(ROOT)) for p in base.glob("**/*report.md")})
        if reports or base.name.startswith("pnecessity_nonstandard_uspto"):
            inventory.append({"task": base.name, "reports": reports, "included": base.name in selected,
                              "reason": "primary Full/no-P pair" if base.name in selected else "not a complete primary Full/no-P task in the proposal; historical alternate-input/preparation/partial or aggregate report"})
    manifest = dict(created_at=datetime.now(timezone.utc).isoformat(), comparisons=len(df), model_fits_started=0,
                    metric_estimand="row weighted repeat-averaged losses; R2=1-mean_repeat_SSE/SST; RMSE=sqrt(mean_repeat_MSE)",
                    ci="pointwise conditional group percentile bootstrap; fixed saved fits, no split resampling; not multiplicity-adjusted",
                    timing="sum fit/predict perf_counter seconds per saved fold, nominal 15 or frozen 1; no feature/end-to-end durations",
                    files=FILES, script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), report_inventory=inventory)
    (out/"run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(df.groupby("batch").agg(n=("pair_id","size"),time_median=("train_increase_pct","median"),r2_min=("delta_r2","min"),r2_max=("delta_r2","max")).to_string())


if __name__ == "__main__":
    main()
