"""Fit the tier models and estimate the information cost of missing laboratory inputs.

Analysis set: all-labs participants from cycles whose censoring survival at the horizon is at least
``min_g``. Each tier is a ridge-penalized cause-specific Cox model for CVD death (non-CVD death and the
horizon censor), fitted with normalized MEC weights. Penalty chosen by cross-validated partial likelihood;
performance uses out-of-fold predictions with the penalty selected inside each outer fold.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from lablite_cvd import cox
from lablite_cvd.model import FEATURES, LabLiteModel, fit_tier, model_frame, prepare
from lablite_cvd.pipeline.cost import bootstrap, cost_rows, tier_metrics
from lablite_cvd.pipeline.metrics import ipcw

LAMBDAS = (0.0, 1.0, 10.0, 100.0, 1000.0)


def analysis_set(cohort: pd.DataFrame, horizon: float, min_g: float):
    d = cohort[cohort["all_labs"] == 1].reset_index(drop=True)
    _, _, g_h = ipcw(d, horizon)
    keep = sorted(c for c, g in g_h.items() if g >= min_g)
    return d[d["cycle"].isin(keep)].reset_index(drop=True), keep, g_h


def _folds(y, k, rng):
    """Stratified fold labels."""
    f = np.empty(len(y), int)
    for cls in (0, 1):
        idx = rng.permutation(np.flatnonzero(y == cls))
        f[idx] = np.arange(len(idx)) % k
    return f


def _choose_lambda(mf, feats, time, event, sw, folds):
    Z = prepare(mf, feats)[0]
    scores = {lam: cox.cv_log_likelihood(Z, time, event, sw, lam, folds) for lam in LAMBDAS}
    return max(scores, key=scores.get), scores


def run_fit(data_dir: Path, out: Path, horizon=10.0, min_g=0.10, k=5, seed=20261007):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    cohort = pd.read_csv(Path(data_dir) / "processed" / "cohort.csv.gz")
    d, kept, g_h = analysis_set(cohort, horizon, min_g)
    y, w_ipcw, _ = ipcw(d, horizon)
    mf = model_frame(d)
    time = np.minimum(d["time"].to_numpy(float), horizon)
    event = ((d["event"] == 1) & (d["time"] <= horizon)).to_numpy(float)
    sw = (d["wt"] / d["wt"].mean()).to_numpy(float)
    rng = np.random.default_rng(seed)
    outer = _folds(event, k, rng)
    print(f"Analysis set n={len(d):,}, CVD deaths within {horizon:g}y={int(event.sum())}, cycles {kept}")

    # Out-of-fold predictions, penalty chosen within each outer training set
    oof = pd.DataFrame(index=d.index, columns=list(FEATURES), dtype=float)
    lam_by_fold = {}
    for f in range(k):
        tr, te = outer != f, outer == f
        inner = _folds(event[tr], k, rng)
        for tier, feats in FEATURES.items():
            lam, _ = _choose_lambda(mf[tr], feats, time[tr], event[tr], sw[tr], inner)
            lam_by_fold.setdefault(tier, []).append(lam)
            m = fit_tier(tier, mf[tr], time[tr], event[tr], sw[tr], lam, horizon)
            oof.loc[te, tier] = m.risk(mf[te])
        print(f"  outer fold {f + 1}/{k} done", flush=True)

    # Final models on the full analysis set
    tiers, cv_scores = {}, {}
    full_folds = _folds(event, k, rng)
    for tier, feats in FEATURES.items():
        lam, scores = _choose_lambda(mf, feats, time, event, sw, full_folds)
        cv_scores[tier] = {str(l): s for l, s in scores.items()}
        tiers[tier] = fit_tier(tier, mf, time, event, sw, lam, horizon)
    meta = {"horizon": horizon, "outcome": "CVD death (cause-specific; non-CVD death censored)",
            "cycles": kept, "n": int(len(d)), "events": int(event.sum()), "seed": seed,
            "status": "DRAFT - not verified by the maintainer"}
    model = LabLiteModel(tiers, meta)
    model.to_json(out / "lablite_params.json")
    pd.concat([m.coefficients().assign(tier=t) for t, m in tiers.items()]).to_csv(out / "coefficients.csv",
                                                                                 index=False)

    # Performance per tier and information cost (point estimates; `lablite-cvd bootstrap` adds intervals)
    oof = oof.astype(float)
    perf_d = tier_metrics(oof, y, w_ipcw, sw)
    perf = pd.DataFrame(perf_d).T
    perf.index.name = "tier"
    perf.to_csv(out / "performance.csv")
    cost = pd.Series(cost_rows(oof, y, w_ipcw, sw, perf_d)).unstack()
    cost.index.name = "comparison"
    cost.to_csv(out / "information_cost.csv")

    oof.assign(SEQN=d["SEQN"], fold=outer).to_csv(out / "oof_predictions.csv.gz", index=False)
    (out / "fit_meta.json").write_text(json.dumps(
        {**meta, "lambdas_grid": LAMBDAS, "lambda_chosen": {t: m.lam for t, m in tiers.items()},
         "lambda_by_outer_fold": lam_by_fold, "cv_loglik": cv_scores,
         "G_horizon_by_cycle": {c: float(g) for c, g in g_h.items()}}, indent=1))

    pd.set_option("display.width", 160)
    print("\nPerformance (out-of-fold, survey-weighted, IPCW):")
    print(perf[["OE", "cal_slope", "AUC", "NB_1", "NB_5", "NB_7.5", "NB_10", "NB_20"]].round(4).to_string())
    print("\nInformation cost (richer minus poorer tier; positive = lost without the labs):")
    print(cost[["dAUC", "dNB_1", "dNB_5", "dNB_10", "pct_reclassified_1/5/10"]].round(4).to_string())
    print(f"\nWritten: {out}")
    return model, perf, cost


def run_bootstrap(data_dir: Path, out: Path, B=200, seed=20261008, horizon=10.0, min_g=0.10):
    """Survey-bootstrap intervals from the saved out-of-fold predictions of `run_fit`."""
    out = Path(out)
    cohort = pd.read_csv(Path(data_dir) / "processed" / "cohort.csv.gz")
    d, _, _ = analysis_set(cohort, horizon, min_g)
    oof = pd.read_csv(out / "oof_predictions.csv.gz")
    if not oof["SEQN"].equals(d["SEQN"]):
        raise ValueError("oof_predictions.csv.gz does not match the analysis set; rerun `lablite-cvd fit`")
    preds = oof[list(FEATURES)].astype(float)
    y, w, _ = ipcw(d, horizon)
    print(f"Bootstrap B={B} over {d['strata'].nunique()} strata", flush=True)
    tiers, cost = bootstrap(preds, y, w, d["wt"].to_numpy(float), d["strata"].to_numpy(), d["psu"].to_numpy(),
                            B=B, seed=seed)
    tiers.to_csv(out / "performance_ci.csv", index=False)
    cost.to_csv(out / "information_cost_ci.csv", index=False)
    meta_path = out / "fit_meta.json"
    meta = json.loads(meta_path.read_text())
    meta["bootstrap"] = {"B": B, "seed": seed, "method": "Rao-Wu rescaling over PSUs within strata; "
                         "fixed out-of-fold predictions; 95% percentile intervals"}
    meta_path.write_text(json.dumps(meta, indent=1))

    show = cost[cost["metric"].isin(["dAUC", "dNB_1", "dNB_5", "dNB_7.5", "dNB_10", "dNB_20",
                                     "pct_reclassified_1/5/10"])]
    fmt = show.assign(ci=[f"{e:.4f} ({l:.4f}, {h:.4f})" for e, l, h in zip(show.estimate, show.lo, show.hi)])
    pd.set_option("display.width", 200)
    print("\nInformation cost, estimate (95% CI):")
    print(fmt.pivot(index="comparison", columns="metric", values="ci").to_string())
    print(f"\nWritten: {out / 'information_cost_ci.csv'}, {out / 'performance_ci.csv'}")
    return tiers, cost
