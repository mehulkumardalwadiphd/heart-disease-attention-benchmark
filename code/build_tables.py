"""
build_tables.py - derives the manuscript's tables from the pipeline's raw
results, so every number printed in a table traces to one CSV row.

Reads only files written by code/pipeline.py; trains nothing and never edits
those files. Run after pipeline.py and before writing or rebuilding the draft.

Outputs (all in results/):
  model_comparison.csv     main 10 x 5 CV table, one-hot encoding (Table 3)
  encoding_sensitivity.csv one-hot vs integer-code ROC-AUC per model
  external_table.csv       Cleveland-trained models tested on three other sites
  ablation_table.csv       heads x attention dropout grid (mean ROC-AUC)
  fold_sizes_table.csv     the two fold layouts of the 10 x 5 CV
  aliases.json             prose names -> row labels, for the numbers gate

pipeline.py is deliberately not imported: its source hash keys the training
checkpoints, so this script stays a separate file.
"""

import json
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parent.parent
RES = BASE / "results"

METRICS = ["roc_auc", "accuracy", "sensitivity", "specificity", "precision",
           "f1", "brier"]
SITES = {"Hungarian": "hungarian", "Switzerland": "switzerland",
         "VA Long Beach": "va"}


def model_comparison():
    cv = pd.read_csv(RES / "cv_summary_onehot.csv")
    out = pd.DataFrame({"model": cv["model"], "key": cv["key"],
                        "folds": cv["folds"]})
    for m in METRICS:
        out[m] = cv[f"{m}_mean"]
        out[f"{m}_sd"] = cv[f"{m}_sd"]
        out[f"{m}_ci_low"] = cv[f"{m}_ci_low"]
        out[f"{m}_ci_high"] = cv[f"{m}_ci_high"]
    out.to_csv(RES / "model_comparison.csv", index=False)
    return out


def encoding_sensitivity():
    one = pd.read_csv(RES / "cv_summary_onehot.csv").set_index("key")
    integer = pd.read_csv(RES / "cv_summary_integer.csv").set_index("key")
    out = pd.DataFrame({
        "model": one["model"],
        "onehot_auc": one["roc_auc_mean"],
        "onehot_auc_sd": one["roc_auc_sd"],
        "integer_auc": integer.loc[one.index, "roc_auc_mean"],
        "integer_auc_sd": integer.loc[one.index, "roc_auc_sd"],
    })
    out["difference"] = (out["integer_auc"] - out["onehot_auc"]).round(4)
    out.reset_index().rename(columns={"key": "key"})[
        ["model", "key", "onehot_auc", "onehot_auc_sd", "integer_auc",
         "integer_auc_sd", "difference"]
    ].to_csv(RES / "encoding_sensitivity.csv", index=False)
    return out


def external_table():
    ext = pd.read_csv(RES / "external_summary.csv")
    ext = ext[ext["feature_set"] == "ext"]
    cv = pd.read_csv(RES / "cv_summary_ext.csv").set_index("key")
    rows = []
    for key in cv.index:
        row = {"model": cv.loc[key, "model"], "key": key,
               "cleveland_cv_auc": cv.loc[key, "roc_auc_mean"],
               "cleveland_cv_auc_sd": cv.loc[key, "roc_auc_sd"]}
        for site, short in SITES.items():
            r = ext[(ext["site"] == site) & (ext["key"] == key)]
            assert len(r) == 1, (site, key)
            r = r.iloc[0]
            row[f"{short}_auc"] = r["roc_auc_mean"]
            row[f"{short}_auc_ci_low"] = r["roc_auc_ci_low"]
            row[f"{short}_auc_ci_high"] = r["roc_auc_ci_high"]
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(RES / "external_table.csv", index=False)
    return out


def ablation_table():
    ab = pd.read_csv(RES / "ablation_summary.csv")
    grid = ab.pivot(index="heads", columns="attn_dropout", values="roc_auc_mean")
    sd = ab.pivot(index="heads", columns="attn_dropout", values="roc_auc_sd")
    out = pd.DataFrame({"heads": grid.index})
    for d in grid.columns:
        out[f"dropout_{d}"] = grid[d].values
        out[f"dropout_{d}_sd"] = sd[d].values
    out.to_csv(RES / "ablation_table.csv", index=False)
    return out


def fold_sizes_table():
    f = pd.read_csv(RES / "fold_sizes_onehot.csv")
    cols = ["n_train", "n_fit", "n_val", "n_test",
            "pos_train", "pos_fit", "pos_val", "pos_test"]
    # Every repeat must share one layout per fold position, or this summary lies.
    assert (f.groupby("fold")[cols].nunique() == 1).all().all()
    layout = f.drop_duplicates(subset=["fold"] + cols)[["fold"] + cols]
    groups = layout.groupby(cols)["fold"].apply(list).reset_index()
    rows = []
    for _, g in groups.iterrows():
        folds = [i + 1 for i in g["fold"]]
        label = (f"Folds {folds[0]}-{folds[-1]}" if len(folds) > 1
                 else f"Fold {folds[0]}")
        row = {"layout": label, "folds_per_repeat": len(folds)}
        row.update({c: int(g[c]) for c in cols})
        rows.append(row)
    out = pd.DataFrame(rows).sort_values("n_test", ascending=False)
    out.to_csv(RES / "fold_sizes_table.csv", index=False)
    return out


def pairwise_table():
    """The pre-specified ROC-AUC comparisons, one row per pair, labelled
    'A vs B' with the model keys used in the manuscript."""
    pw = pd.read_csv(RES / "pairwise_tests_onehot.csv")
    pw = pw[pw["metric"] == "roc_auc"]
    out = pd.DataFrame({
        "comparison": pw["key_a"] + " vs " + pw["key_b"],
        "difference": pw["mean_diff"], "ci_low": pw["ci_low"],
        "ci_high": pw["ci_high"], "p": pw["p"], "p_holm": pw["p_holm"],
        "folds_a_higher": pw["a_wins_folds"], "ties": pw["ties"],
    })
    out.to_csv(RES / "pairwise_auc_table.csv", index=False)
    return out


def training_summary():
    """Which checkpoint each network was evaluated at, over the 50 folds, and
    how inner-validation accuracy at that checkpoint compares with the
    held-out fold. Answers 'which run and which epoch' for every number."""
    folds = pd.read_csv(RES / "cv_folds_onehot.csv")
    hist = pd.read_csv(RES / "training_histories_onehot.csv")
    names = pd.read_csv(RES / "cv_summary_onehot.csv").set_index("key")["model"]
    rows = []
    for key, g in folds.dropna(subset=["best_epoch"]).groupby("model", sort=False):
        best = g.set_index(["repeat", "fold"])["best_epoch"]
        h = hist[hist.model == key].set_index(["repeat", "fold", "epoch"])
        at_best = h.loc[[(r, f, int(e)) for (r, f), e in best.items()]]
        trained = hist[hist.model == key].groupby(["repeat", "fold"])["epoch"].max()
        rows.append({
            "model": names[key], "key": key, "folds": len(g),
            "best_epoch_median": float(best.median()),
            "best_epoch_q1": float(best.quantile(0.25)),
            "best_epoch_q3": float(best.quantile(0.75)),
            "best_epoch_min": int(best.min()), "best_epoch_max": int(best.max()),
            "epochs_trained_max": int(trained.max()),
            "val_accuracy_at_best": round(float(at_best["val_acc"].mean()), 4),
            "test_accuracy": round(float(g["accuracy"].mean()), 4),
        })
    out = pd.DataFrame(rows)
    out.to_csv(RES / "training_summary.csv", index=False)
    return out


def aliases(models):
    keys = dict(zip(models["key"], models["model"]))
    names = {k: v for k, v in keys.items()}
    names.update({
        "logistic regression": keys["LR"],
        "decision tree": keys["DT"],
        "random forest": keys["RF"],
        "xgboost": keys["XGB"],
        "feature-token attention model": keys["FT-Attn"],
        "feature-token affine control": keys["FT-Affine"],
        "single-token attention model": keys["ST-Attn"],
        "single-token affine control": keys["ST-Affine"],
    })
    (RES / "aliases.json").write_text(json.dumps(names, indent=2) + "\n",
                                      encoding="utf-8")


def external_table_rev2():
    """Table 8 as revised for the second NJES round: seed-mean external ROC-AUC
    with the class-stratified patient-bootstrap 95% CI from
    revision2_analyses.py. `*_cell` columns hold the printed cell text."""
    base = pd.read_csv(RES / "external_table.csv").set_index("key")
    ci = pd.read_csv(RES / "rev2_external_stratified_ci.csv")
    out = base[["model", "cleveland_cv_auc", "cleveland_cv_auc_sd"]].copy()
    for site, short in SITES.items():
        s = ci[ci["site"] == site].set_index("key").loc[base.index]
        out[f"{short}_auc"] = s["roc_auc_mean"]
        out[f"{short}_strat_ci_low"] = s["strat_ci_low"]
        out[f"{short}_strat_ci_high"] = s["strat_ci_high"]
        out[f"{short}_cell"] = [f"{a:.4f} ({lo:.4f}–{hi:.4f})" for a, lo, hi in
                                zip(s["roc_auc_mean"], s["strat_ci_low"],
                                    s["strat_ci_high"])]
    out.reset_index()[["model", "key"] + [c for c in out.columns if c != "model"]] \
        .to_csv(RES / "external_table_rev2.csv", index=False)
    return out


def training_size_table():
    """Training-set size check (revision2_analyses.py): mean ± SD ROC-AUC of
    the classical models fitted on 100/85/70/50% of each training fold."""
    s = pd.read_csv(RES / "rev2_training_size_summary.csv")
    rows = []
    for key, g in s.groupby("key", sort=False):
        row = {"model": g["model"].iloc[0], "key": key}
        for frac in (1.0, 0.85, 0.7, 0.5):
            r = g[g["fraction"] == frac].iloc[0]
            pct = str(int(round(frac * 100)))
            row[pct] = r["roc_auc_mean"]
            row[f"{pct}_sd"] = r["roc_auc_sd"]
        rows.append(row)
    order = ["LR", "DT", "RF", "XGB"]
    out = pd.DataFrame(rows).set_index("key").loc[order].reset_index()
    out[["model", "key"] + [c for c in out.columns if c not in ("model", "key")]] \
        .to_csv(RES / "training_size_table.csv", index=False)
    return out


if __name__ == "__main__":
    mc = model_comparison()
    encoding_sensitivity()
    external_table()
    external_table_rev2()
    training_size_table()
    ablation_table()
    fold_sizes_table()
    training_summary()
    pairwise_table()
    aliases(mc)
    print("wrote model_comparison.csv, encoding_sensitivity.csv, "
          "external_table.csv, ablation_table.csv, fold_sizes_table.csv, "
          "training_summary.csv, pairwise_auc_table.csv, aliases.json")
