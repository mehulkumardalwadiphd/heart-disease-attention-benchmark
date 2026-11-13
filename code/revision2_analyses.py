"""
revision2_analyses.py - two analyses added for the second NJES review round
(Ms. NJES-26-0105R1, Reviewer 1, comments 2 and 3).

A. Training-set size (Reviewer 1, comment 3).
   The networks are fitted on the 85% "fit" subset of each training fold (the
   other 15% is the inner validation subset used for early stopping), while the
   classical models in the main analysis are fitted on the whole training fold.
   Here the four classical models are refitted on the same 50 folds with
       100%  the whole training fold (must reproduce cv_folds_onehot.csv),
        85%  exactly the fit subset each network saw in that fold,
        70%  and 50%  stratified random subsets of the training fold.
   The encoder is fitted on the whole training fold in every case, as it is for
   the networks, so the only thing that changes is the rows the model is fitted
   on. Test folds are unchanged.

B. External-validation variability (Reviewer 1, comment 2).
   Re-uses the stored predictions of the external run (external_predictions.csv,
   10 training seeds, 8 shared features) and trains nothing:
     - a class-stratified patient bootstrap of the seed-mean ROC-AUC at every
       site (positives and negatives resampled separately, so every resample of
       the Swiss cohort keeps exactly its 8 patients without CAD);
     - for the Swiss cohort, leave-one-out over its 8 patients without CAD:
       the seed-mean ROC-AUC recomputed 8 times, each time without one of them.

Imports pipeline.py for data loading, encoding, model definitions and
statistics so that nothing is re-implemented; pipeline.py itself is not edited
(its hash keys the stored checkpoints and is recorded in run_info.json).

Writes (results/):
  rev2_training_size_folds.csv      one row per repeat x fold x model x fraction
  rev2_training_size_summary.csv    mean, SD, corrected 95% CI per model x fraction
  rev2_training_size_pairs.csv      paired differences (vs 100%, and the 85%
                                    classical models vs the networks)
  rev2_external_stratified_ci.csv   stratified-bootstrap CI per site x model
  rev2_swiss_leave_one_out.csv      Swiss leave-one-negative-out per model
  rev2_info.json                    run settings, timing, reproduction check

Usage:  python revision2_analyses.py [--smoke] [--jobs N]
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pipeline as P  # noqa: E402

RES = P.RESULTS
CLASSICAL = ["LR", "DT", "RF", "XGB"]
FRACTIONS = [1.00, 0.85, 0.70, 0.50]
NN_COMPARE = ["MLP", "FT-Attn", "FT-Affine", "ST-Attn", "ST-Affine"]
N_BOOT = P.N_BOOT
SEED = P.SEED


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


# ---------------------------------------------------------------- A. size ---

def subset(y_tr, frac, seed):
    """Stratified random subset of positions 0..len(y_tr)-1 of size frac."""
    rng = np.random.RandomState(seed)
    keep = []
    for c in (0, 1):
        idx = np.where(y_tr == c)[0]
        k = int(round(frac * len(idx)))
        keep.append(rng.choice(idx, k, replace=False))
    return np.sort(np.concatenate(keep))


def run_fold(repeat, fold, tr, fit_ids, te, X, y):
    seed = SEED + 1000 * repeat + fold          # same per-fold seed as pipeline
    X_tr, y_tr = X[tr], y[tr]
    enc = P.Encoder(P.FEATURES, "onehot").fit(X_tr)
    pos = {pid: i for i, pid in enumerate(tr)}
    rows = []
    for frac in FRACTIONS:
        if frac == 1.00:
            sel = np.arange(len(tr))
        elif frac == 0.85:
            sel = np.sort(np.array([pos[p] for p in fit_ids]))
        else:
            sel = subset(y_tr, frac, seed + int(frac * 100))
        for name in CLASSICAL:
            m = P.build_sklearn(name, seed)
            m.fit(enc.flat(X_tr[sel]), y_tr[sel])
            p = m.predict_proba(enc.flat(X[te]))[:, 1]
            rows.append({"repeat": repeat, "fold": fold, "model": name,
                         "fraction": frac, "n_fit": len(sel),
                         **P.metrics(y[te], p)})
    return rows


def training_size(smoke, jobs):
    df = P.load_cleveland()
    X = df[P.FEATURES].values.astype(float)
    y = df["target"].values
    roles = pd.read_csv(RES / "split_indices_onehot.csv")
    tasks = []
    for (r, f), g in roles.groupby(["repeat", "fold"]):
        if smoke and r > 0:
            continue
        fit_ids = g.loc[g.role == "fit", "patient"].values
        val_ids = g.loc[g.role == "val", "patient"].values
        te = np.sort(g.loc[g.role == "test", "patient"].values)
        tr = np.sort(np.concatenate([fit_ids, val_ids]))
        tasks.append((int(r), int(f), tr, fit_ids, te))
    log(f"[size] {len(tasks)} folds x {len(CLASSICAL)} models x {len(FRACTIONS)} fractions")
    out = Parallel(n_jobs=jobs)(delayed(run_fold)(r, f, tr, fi, te, X, y)
                                for r, f, tr, fi, te in tasks)
    folds = pd.DataFrame([row for rows in out for row in rows])

    # Reproduction check: 100% must equal the main analysis fold by fold
    main = pd.read_csv(RES / "cv_folds_onehot.csv")
    full = folds[folds.fraction == 1.00].merge(
        main[["repeat", "fold", "model", "roc_auc"]],
        on=["repeat", "fold", "model"], suffixes=("", "_main"))
    max_dev = float((full.roc_auc - full.roc_auc_main).abs().max())
    log(f"[size] reproduction of main analysis at 100%: max |dAUC| = {max_dev:.2e}")

    sizes = pd.read_csv(RES / "fold_sizes_onehot.csv")
    ratio = float((sizes["n_test"] / sizes["n_train"]).mean())
    summ = []
    for (m, frac), g in folds.groupby(["model", "fraction"]):
        mean, lo, hi = P.corrected_ci(g.roc_auc, ratio)
        summ.append({"model": P.MODEL_NAMES[m], "key": m, "fraction": frac,
                     "n_fit_mean": round(g.n_fit.mean(), 1), "folds": len(g),
                     "roc_auc_mean": round(mean, 4),
                     "roc_auc_sd": round(g.roc_auc.std(ddof=1), 4),
                     "roc_auc_ci_low": round(lo, 4), "roc_auc_ci_high": round(hi, 4),
                     "brier_mean": round(g.brier.mean(), 4)})
    summ = pd.DataFrame(summ).sort_values(["key", "fraction"], ascending=[True, False])

    # Paired differences
    wide = folds.pivot_table(index=["repeat", "fold"], columns=["model", "fraction"],
                             values="roc_auc")
    nn = main[main.model.isin(NN_COMPARE)].pivot_table(
        index=["repeat", "fold"], columns="model", values="roc_auc")
    pairs = []

    def add(label_a, label_b, d, kind):
        mean, lo, hi = P.corrected_ci(d, ratio)
        t, p = P.corrected_ttest(d, ratio)
        pairs.append({"comparison": kind, "a": label_a, "b": label_b,
                      "mean_diff": round(mean, 4), "ci_low": round(lo, 4),
                      "ci_high": round(hi, 4), "t": round(t, 3), "p": p,
                      "folds": len(d)})
    for m in CLASSICAL:
        for frac in FRACTIONS[1:]:
            d = (wide[(m, frac)] - wide[(m, 1.00)]).values
            add(f"{m}@{int(frac * 100)}%", f"{m}@100%", d, "size_vs_full")
    if not smoke:
        idx = wide.index.intersection(nn.index)
        for m in CLASSICAL:
            for n in NN_COMPARE:
                d = (nn.loc[idx, n] - wide.loc[idx, (m, 0.85)]).values
                add(n, f"{m}@85%", d, "nn_vs_classical_same_rows")
    pairs = pd.DataFrame(pairs)
    pairs["p_holm"] = np.nan
    for kind, g in pairs.groupby("comparison"):
        pairs.loc[g.index, "p_holm"] = P.holm(g.p.values)
    pairs["p"] = pairs.p.round(4)
    pairs["p_holm"] = pairs.p_holm.round(4)
    return folds, summ, pairs, max_dev, ratio


# ------------------------------------------------------------- B. external --

def seed_mean_auc(y, Pm):
    return float(np.mean([roc_auc_score(y, Pm[:, k]) for k in range(Pm.shape[1])]))


def external_variability(smoke):
    preds = pd.read_csv(RES / "external_predictions.csv")
    preds = preds[preds.feature_set == "ext"]
    rng = np.random.RandomState(SEED)
    n_boot = 100 if smoke else N_BOOT
    ci_rows, loo_rows = [], []
    for (site, m), g in preds.groupby(["site", "model"]):
        wide = g.pivot_table(index="row", columns="seed", values="prob")
        y = g.groupby("row")["y"].first().loc[wide.index].values
        Pm = wide.values
        point = seed_mean_auc(y, Pm)
        pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
        boots = []
        for _ in range(n_boot):
            s = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
            boots.append(seed_mean_auc(y[s], Pm[s]))
        ci_rows.append({"site": site, "model": P.MODEL_NAMES[m], "key": m,
                        "n": len(y), "positive": len(pos), "negative": len(neg),
                        "roc_auc_mean": round(point, 4),
                        "strat_ci_low": round(float(np.percentile(boots, 2.5)), 4),
                        "strat_ci_high": round(float(np.percentile(boots, 97.5)), 4),
                        "boot_sd": round(float(np.std(boots, ddof=1)), 4),
                        "n_boot": n_boot})
        if site == "Switzerland":
            loo = []
            for j in neg:
                keep = np.setdiff1d(np.arange(len(y)), [j])
                loo.append(seed_mean_auc(y[keep], Pm[keep]))
            loo = np.array(loo)
            # per-negative contribution: share of positives scored above it
            ranks = [float(np.mean(Pm[pos].mean(1) > Pm[j].mean())) for j in neg]
            loo_rows.append({"model": P.MODEL_NAMES[m], "key": m,
                             "roc_auc_all": round(point, 4),
                             "loo_min": round(float(loo.min()), 4),
                             "loo_max": round(float(loo.max()), 4),
                             "loo_range": round(float(loo.max() - loo.min()), 4),
                             "loo_sd": round(float(loo.std(ddof=1)), 4),
                             "negatives": len(neg),
                             "per_negative_pos_above_min": round(min(ranks), 4),
                             "per_negative_pos_above_max": round(max(ranks), 4)})
    return pd.DataFrame(ci_rows), pd.DataFrame(loo_rows)


# ------------------------------------------------------------------ main ---

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="first repeat only, 100 bootstrap draws, writes *_smoke.csv")
    ap.add_argument("--jobs", type=int, default=-1)
    a = ap.parse_args()
    sfx = "_smoke" if a.smoke else ""
    t0 = time.time()
    folds, summ, pairs, max_dev, ratio = training_size(a.smoke, a.jobs)
    folds.round(6).to_csv(RES / f"rev2_training_size_folds{sfx}.csv", index=False)
    summ.to_csv(RES / f"rev2_training_size_summary{sfx}.csv", index=False)
    pairs.to_csv(RES / f"rev2_training_size_pairs{sfx}.csv", index=False)
    log(f"[size] done in {time.time() - t0:.0f}s")
    print(summ.to_string(index=False), flush=True)
    t1 = time.time()
    ci, loo = external_variability(a.smoke)
    ci.to_csv(RES / f"rev2_external_stratified_ci{sfx}.csv", index=False)
    loo.to_csv(RES / f"rev2_swiss_leave_one_out{sfx}.csv", index=False)
    log(f"[external] done in {time.time() - t1:.0f}s")
    print(ci.to_string(index=False), flush=True)
    print(loo.to_string(index=False), flush=True)
    info = {"smoke": a.smoke, "seconds": round(time.time() - t0, 1),
            "pipeline_code_hash": P.CODE_HASH, "fractions": FRACTIONS,
            "classical_models": CLASSICAL, "n_boot": N_BOOT if not a.smoke else 100,
            "nadeau_bengio_ratio": round(ratio, 4),
            "reproduction_max_abs_auc_diff_at_100pct": max_dev}
    json.dump(info, open(RES / f"rev2_info{sfx}.json", "w"), indent=2)


if __name__ == "__main__":
    main()
