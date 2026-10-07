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
from lablite_cvd.pipeline.metrics import THRESHOLDS, ipcw, performance

LAMBDAS = (0.0, 1.0, 10.0, 100.0, 1000.0)
CATEGORIES = (0.0, 0.075, 0.20, 1.0)


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

    # Performance per tier and information cost relative to the full-lab tier (T3)
    perf = pd.DataFrame({t: performance(oof[t].to_numpy(float), y, w_ipcw, sw) for t in FEATURES}).T
    perf.index.name = "tier"
    perf.to_csv(out / "performance.csv")
    cats = {t: np.digitize(oof[t], CATEGORIES[1:-1]) for t in FEATURES}
    cost = []
    for t in list(FEATURES)[:-1]:
        row = {"tier": t, "vs": "T3", "dAUC": perf.loc["T3", "AUC"] - perf.loc[t, "AUC"]}
        for th in THRESHOLDS:
            tag = f"{th * 100:g}"
            row[f"dNB_{tag}"] = perf.loc["T3", f"NB_{tag}"] - perf.loc[t, f"NB_{tag}"]
        row["pct_reclassified"] = float(np.sum(sw * (cats[t] != cats["T3"])) / sw.sum())
        cost.append(row)
    cost = pd.DataFrame(cost)
    cost.to_csv(out / "information_cost.csv", index=False)

    oof.assign(SEQN=d["SEQN"], fold=outer).to_csv(out / "oof_predictions.csv.gz", index=False)
    (out / "fit_meta.json").write_text(json.dumps(
        {**meta, "lambdas_grid": LAMBDAS, "lambda_chosen": {t: m.lam for t, m in tiers.items()},
         "lambda_by_outer_fold": lam_by_fold, "cv_loglik": cv_scores,
         "G_horizon_by_cycle": {c: float(g) for c, g in g_h.items()}}, indent=1))

    pd.set_option("display.width", 160)
    print("\nPerformance (out-of-fold, survey-weighted, IPCW):")
    print(perf[["OE", "cal_slope", "AUC", "NB_7.5", "NB_20", "pct_flagged_7.5"]].round(4).to_string())
    print("\nInformation cost vs T3 (positive = performance lost without the labs):")
    print(cost.round(4).to_string(index=False))
    print(f"\nWritten: {out}")
    return model, perf, cost
